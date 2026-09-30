import streamlit as st
import ezdxf
from ezdxf import path
import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import polygonize, unary_union
from shapely.affinity import translate, rotate
import matplotlib.pyplot as plt
import tempfile
import os
import io
import time

st.set_page_config(page_title="High-Speed Server Nester", layout="wide")

# --- CORE LOGIC ---
def extract_smart_parts(file_bytes):
    # Safely handle DXF encodings by letting ezdxf read from a physical temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
        
    try:
        doc = ezdxf.readfile(tmp_path)
    finally:
        os.remove(tmp_path)

    msp = doc.modelspace()
    lines_and_arcs = []
    
    # 1. Standard LWPOLYLINEs
    for entity in msp.query('LWPOLYLINE'):
        pts = [(round(p[0], 3), round(p[1], 3)) for p in entity.get_points('xy')]
        if len(pts) > 1:
            lines_and_arcs.append(LineString(pts))
            if entity.closed:
                lines_and_arcs.append(LineString([pts[-1], pts[0]]))
                
    # 2. Universal Extractor: Lines, Arcs, Splines, Ellipses
    for entity in msp.query('LINE ARC SPLINE ELLIPSE'):
        try:
            p = path.make_path(entity)
            for sub_path in p.flattening(0.1): 
                pts = [(round(v.x, 3), round(v.y, 3)) for v in sub_path]
                if len(pts) > 1:
                    lines_and_arcs.append(LineString(pts))
        except:
            pass
                
    # 3. Gather Circles
    circles = []
    for circle in msp.query('CIRCLE'):
        center = circle.dxf.center
        c_poly = Point(round(center.x, 3), round(center.y, 3)).buffer(round(circle.dxf.radius, 3), resolution=16)
        circles.append(c_poly)
        
    if not lines_and_arcs and not circles:
        return []
        
    merged_lines = unary_union(lines_and_arcs) if lines_and_arcs else LineString()
    formed_polys = list(polygonize(merged_lines))
    all_polys = formed_polys + circles

    # Destructive Frame Filter removed. The UI will safely set oversized frames to Qty 0.
    clean_polys = [p.simplify(0.2, preserve_topology=True) for p in all_polys]
    clean_polys.sort(key=lambda x: x.area, reverse=True)
    
    geom_lines = [merged_lines] if merged_lines.geom_type == 'LineString' else list(getattr(merged_lines, 'geoms', []))
    
    loose_lines = []
    for line in geom_lines:
        is_boundary = False
        for p in clean_polys:
            if p.exterior.distance(line) < 0.1: 
                is_boundary = True
                break
        if not is_boundary: loose_lines.append(line)

    # 4. Spatial Grouping
    parts = []
    assigned = set()
    
    for i, poly in enumerate(clean_polys):
        if i in assigned: continue
        
        part = {'outer': poly, 'inners': [], 'area': poly.area}
        assigned.add(i)
        solid_outer = Polygon(poly.exterior).buffer(0.1)
        
        for j in range(i + 1, len(clean_polys)):
            if j not in assigned:
                inner_poly = clean_polys[j]
                if solid_outer.covers(inner_poly) or solid_outer.contains(inner_poly.representative_point()):
                    part['inners'].append(inner_poly)
                    assigned.add(j)
                    
        lines_to_keep = []
        for line in loose_lines:
            if solid_outer.covers(line) or solid_outer.contains(line.representative_point()):
                part['inners'].append(line)
            else:
                lines_to_keep.append(line)
        loose_lines = lines_to_keep
        
        minx, miny, _, _ = part['outer'].bounds
        part['outer'] = translate(part['outer'], xoff=-minx, yoff=-miny)
        part['inners'] = [translate(inner, xoff=-minx, yoff=-miny) for inner in part['inners']]
        parts.append(part)
        
    return parts

def bounds_overlap(b1, b2):
    return not (b1[2] <= b2[0] or b1[0] >= b2[2] or b1[3] <= b2[1] or b1[1] >= b2[3])

# --- WEB USER INTERFACE ---
st.title("⚡ True Server-Side Nester")

st.sidebar.header("Machine Settings")
sheet_w = st.sidebar.number_input("Sheet Width (mm)", value=2500.0)
sheet_h = st.sidebar.number_input("Sheet Height (mm)", value=1250.0)
spacing = st.sidebar.number_input("Part Spacing (mm)", value=3.0)
margin = st.sidebar.number_input("Edge Margin (mm)", value=5.0)
rotations = st.sidebar.number_input("Rotations (4=90°, 8=45°)", value=4)
coarse_res = st.sidebar.number_input("Grid Resolution", value=25.0)

uploaded_file = st.sidebar.file_uploader("1. Upload DXF", type=['dxf'])

if uploaded_file is not None:
    if 'parts' not in st.session_state or st.session_state.file_name != uploaded_file.name:
        with st.spinner("Analyzing DXF..."):
            parts = extract_smart_parts(uploaded_file.getvalue())
            st.session_state.parts = parts
            st.session_state.file_name = uploaded_file.name
    
    parts = st.session_state.parts
    
    if not parts:
        st.error("No valid closed boundaries found.")
    else:
        st.sidebar.success(f"Grouped {len(parts)} Master Parts.")
        
        usable_w = sheet_w - (margin * 2)
        usable_h = sheet_h - (margin * 2)
        
        quantities = {}
        st.sidebar.subheader("Quantities")
        for i, part in enumerate(parts):
            minx, miny, maxx, maxy = part['outer'].bounds
            w, h = round(maxx - minx, 1), round(maxy - miny, 1)
            
            fits = (w <= usable_w and h <= usable_h) or (h <= usable_w and w <= usable_h)
            default_qty = 1 if fits else 0
            
            label = f"Part {i+1} ({w} x {h} mm)"
            if not fits: label += " ⚠️ Exceeds Sheet"
            
            quantities[i] = st.sidebar.number_input(label, value=default_qty, min_value=0, key=f"qty_{i}")
            
        if st.sidebar.button("2. Run Server Nest", use_container_width=True):
            expanded_parts = []
            for i, part in enumerate(parts):
                for _ in range(quantities[i]): expanded_parts.append(part)
                
            expanded_parts.sort(key=lambda p: p['area'], reverse=True)
            
            if not expanded_parts:
                st.warning("All quantities are 0.")
            else:
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                start_time = time.time()
                all_sheets_data = []
                unplaced_parts = expanded_parts.copy()
                total_parts = len(unplaced_parts)
                parts_placed = 0

                while unplaced_parts:
                    placed_on_this_sheet = []
                    failed_parts = []
                    max_search_y = 0.0 
                    sheet_is_empty = True 
                    
                    angles = [i * (360.0 / rotations) for i in range(rotations)]
                    
                    for idx, part in enumerate(unplaced_parts):
                        parts_placed += 1
                        progress_text.text(f"Calculating Part {parts_placed} of {total_parts} on Sheet {len(all_sheets_data)+1}...")
                        progress_bar.progress(int((parts_placed / total_parts) * 100))
                        
                        best_score = float('inf')
                        best_outer = None
                        best_inners = None
                        
                        for angle in angles:
                            origin = part['outer'].centroid
                            r_outer = rotate(part['outer'], angle, origin=origin)
                            r_inners = [rotate(inner, angle, origin=origin) for inner in part['inners']]
                            
                            minx, miny, maxx, maxy = r_outer.bounds
                            r_outer = translate(r_outer, xoff=-minx, yoff=-miny)
                            r_inners = [translate(inner, xoff=-minx, yoff=-miny) for inner in r_inners]
                            
                            part_w, part_h = maxx - minx, maxy - miny
                            if part_w > usable_w or part_h > usable_h: continue
                                
                            limit_y = min(usable_h - part_h, max_search_y + part_h + 100)
                            coarse_best_x, coarse_best_y = None, None
                            coarse_score = float('inf')

                            for x in np.arange(0, usable_w - part_w + 1, coarse_res):
                                for y in np.arange(0, limit_y + 1, coarse_res):
                                    cand_bounds = (x, y, x + part_w, y + part_h)
                                    collision = False
                                    for placed in placed_on_this_sheet:
                                        if bounds_overlap(cand_bounds, placed['bounds']):
                                            if translate(r_outer, xoff=x, yoff=y).intersects(placed['buffered']):
                                                collision = True; break
                                    if not collision:
                                        score = (y * sheet_w) + x
                                        if score < coarse_score:
                                            coarse_score = score
                                            coarse_best_x, coarse_best_y = x, y
                                        break 

                            if coarse_best_x is not None:
                                start_x = max(0, coarse_best_x - coarse_res)
                                end_x = min(usable_w - part_w, coarse_best_x + coarse_res)
                                start_y = max(0, coarse_best_y - coarse_res)
                                end_y = min(usable_h - part_h, coarse_best_y + coarse_res)

                                for fx in np.arange(start_x, end_x + 1, 2.0):
                                    for fy in np.arange(start_y, end_y + 1, 2.0):
                                        cand_bounds = (fx, fy, fx + part_w, fy + part_h)
                                        collision = False
                                        for placed in placed_on_this_sheet:
                                            if bounds_overlap(cand_bounds, placed['bounds']):
                                                if translate(r_outer, xoff=fx, yoff=fy).intersects(placed['buffered']):
                                                    collision = True; break
                                        if not collision:
                                            score = (fy * sheet_w) + fx
                                            if score < best_score:
                                                best_score = score
                                                best_outer = translate(r_outer, xoff=fx, yoff=fy)
                                                best_inners = [translate(inner, xoff=fx, yoff=fy) for inner in r_inners]
                                            break
                                        
                        if best_outer:
                            buffered = best_outer.buffer(spacing)
                            placed_on_this_sheet.append({
                                'outer': best_outer, 'inners': best_inners,
                                'buffered': buffered, 'bounds': buffered.bounds
                            })
                            max_search_y = max(max_search_y, buffered.bounds[3])
                            sheet_is_empty = False
                        else:
                            failed_parts.append(part)

                    if sheet_is_empty: break 
                    all_sheets_data.append(placed_on_this_sheet)
                    unplaced_parts = failed_parts

                elapsed = time.time() - start_time
                progress_text.text(f"Done in {elapsed:.2f}s! Generating visual layouts and DXF...")
                
                # Plot Results
                for sheet_idx, sheet_parts in enumerate(all_sheets_data):
                    fig, ax = plt.subplots(figsize=(10, (sheet_h/sheet_w)*10))
                    ax.set_xlim(0, sheet_w)
                    ax.set_ylim(0, sheet_h)
                    ax.set_facecolor('#1e1e1e')
                    fig.patch.set_facecolor('#1e1e1e')
                    
                    # Sheet bounds
                    ax.plot([margin, sheet_w-margin, sheet_w-margin, margin, margin], 
                            [margin, margin, sheet_h-margin, sheet_h-margin, margin], 
                            color='#555', linestyle='dashed')
                    
                    for item in sheet_parts:
                        x, y = translate(item['outer'], xoff=margin, yoff=margin).exterior.xy
                        ax.plot(x, y, color='#4daafc')
                        ax.fill(x, y, alpha=0.5, color='#4daafc')
                        for inner in item['inners']:
                            if inner.geom_type == 'Polygon':
                                ix, iy = translate(inner, xoff=margin, yoff=margin).exterior.xy
                                ax.plot(ix, iy, color='#111')
                                ax.fill(ix, iy, color='#111')
                            
                    plt.title(f"Sheet {sheet_idx + 1}", color='white')
                    st.pyplot(fig)
                
                # Generate DXF memory buffer
                out_doc = ezdxf.new(dxfversion='R2010')
                out_doc.header['$INSUNITS'] = 4 
                out_doc.header['$MEASUREMENT'] = 1 
                msp = out_doc.modelspace()
                
                for sheet_idx, sheet_parts in enumerate(all_sheets_data):
                    offset_x = sheet_idx * (sheet_w + 500)
                    msp.add_lwpolyline([(offset_x, 0), (offset_x + sheet_w, 0), (offset_x + sheet_w, sheet_h), (offset_x, sheet_h), (offset_x, 0)], dxfattribs={'color': 1})
                    for item in sheet_parts:
                        final_outer = translate(item['outer'], xoff=margin + offset_x, yoff=margin)
                        msp.add_lwpolyline(list(final_outer.exterior.coords), dxfattribs={'color': 7})
                        for inner in item['inners']:
                            final_inner = translate(inner, xoff=margin + offset_x, yoff=margin)
                            if final_inner.geom_type == 'Polygon':
                                msp.add_lwpolyline(list(final_inner.exterior.coords), dxfattribs={'color': 7})
                            elif final_inner.geom_type in ['LineString', 'LinearRing']:
                                msp.add_lwpolyline(list(final_inner.coords), dxfattribs={'color': 7})
                
                buffer = io.BytesIO()
                out_doc.write(buffer)
                
                st.download_button(
                    label="⬇️ Download Nested DXF",
                    data=buffer.getvalue(),
                    file_name="nested_result.dxf",
                    mime="application/dxf",
                    type="primary"
                )
