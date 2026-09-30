"""
Thinking Nester  -  Streamlit DXF nesting app

requirements.txt (one per line):
    streamlit
    ezdxf
    shapely
    numpy
    matplotlib

(Pillow, which the app also uses, is installed automatically together with matplotlib.)

Run with:  streamlit run thinking_nester.py
"""
import importlib.util

# Friendly check: if a package is missing on the server, say exactly which one
# instead of crashing with a long traceback.
_REQUIRED = {"streamlit": "streamlit", "ezdxf": "ezdxf", "shapely": "shapely",
             "numpy": "numpy", "matplotlib": "matplotlib", "PIL": "pillow"}
_missing = [pip_name for mod, pip_name in _REQUIRED.items() if importlib.util.find_spec(mod) is None]
if _missing:
    _msg = ("Missing packages: " + ", ".join(_missing) +
            ". Add them (one per line) to requirements.txt and redeploy.")
    try:
        import streamlit as _st
    except ImportError:
        raise ImportError(_msg)
    _st.error(_msg)
    _st.stop()

import io
import math
import os
import random
import tempfile
import time

import ezdxf
import matplotlib.pyplot as plt
import numpy as np
from ezdxf import path
from PIL import Image, ImageDraw
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union


# =====================================================================
#  1. DXF READING  (unchanged from your version)
# =====================================================================
def extract_smart_parts(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        doc = ezdxf.readfile(tmp_path)
    finally:
        os.remove(tmp_path)

    msp = doc.modelspace()
    lines_and_arcs = []

    for entity in msp.query('LWPOLYLINE'):
        pts = [(round(p[0], 3), round(p[1], 3)) for p in entity.get_points('xy')]
        if len(pts) > 1:
            lines_and_arcs.append(LineString(pts))
            if entity.closed:
                lines_and_arcs.append(LineString([pts[-1], pts[0]]))

    for entity in msp.query('LINE ARC SPLINE ELLIPSE'):
        try:
            p = path.make_path(entity)
            for sub_path in p.flattening(0.1):
                pts = [(round(v.x, 3), round(v.y, 3)) for v in sub_path]
                if len(pts) > 1:
                    lines_and_arcs.append(LineString(pts))
        except Exception:
            pass

    circles = []
    for circle in msp.query('CIRCLE'):
        center = circle.dxf.center
        c_poly = Point(round(center.x, 3), round(center.y, 3)).buffer(round(circle.dxf.radius, 3), 16)
        circles.append(c_poly)

    if not lines_and_arcs and not circles:
        return []

    merged_lines = unary_union(lines_and_arcs) if lines_and_arcs else LineString()
    formed_polys = list(polygonize(merged_lines))
    all_polys = formed_polys + circles

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
        if not is_boundary:
            loose_lines.append(line)

    parts = []
    assigned = set()

    for i, poly in enumerate(clean_polys):
        if i in assigned:
            continue

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


def generate_part_thumbnail(part):
    """Small image thumbnail of the master part for the sidebar (unchanged)."""
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.set_facecolor('#2d2d2d')
    fig.patch.set_facecolor('#2d2d2d')

    x, y = part['outer'].exterior.xy
    ax.plot(x, y, color='#4daafc', linewidth=1.5)
    ax.fill(x, y, alpha=0.3, color='#4daafc')

    for inner in part['inners']:
        if inner.geom_type == 'Polygon':
            ix, iy = inner.exterior.xy
            ax.plot(ix, iy, color='#111', linewidth=1)
            ax.fill(ix, iy, color='#111')
        elif inner.geom_type in ['LineString', 'LinearRing']:
            ix, iy = inner.xy
            ax.plot(ix, iy, color='#ff4444', linewidth=1)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#444')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# =====================================================================
#  2. NESTING ENGINE
#
#  How it "thinks" (the chess analogy):
#    * MOVE GENERATION : for one part, ALL rotations x ALL positions on the sheet are
#                        tested at once with an FFT collision map (no grid stepping).
#    * EVALUATION      : a finished layout is scored by how much sheet length it consumes
#                        (fewer sheets first, then the shortest used length on the last one).
#    * SEARCH          : many complete "games" are played with different part orders and
#                        placement styles (row-first, column-first, snug-fit...).
#                        Losing lines are abandoned early (alpha-beta style cut-off) and
#                        the best lines are mutated and replayed until time runs out.
# =====================================================================
SUPERSAMPLE = 4
WY_CHOICES = (0.05, 0.15, 0.4, 1.0, 2.5)   # low = fill in columns, high = fill in rows
WC_CHOICES = (0.0, 0.03, 0.08)             # reward for snug contact with neighbours
CROP_QUANT = 24


def _next_fast(n):
    """Next FFT-friendly size (only factors 2, 3, 5)."""
    n = max(1, int(n))
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def _xf(member, fn):
    return {'part': member['part'], 'net': member['net'],
            'outer': fn(member['outer']),
            'inners': [fn(g) for g in member['inners']],
            'material': fn(member['material'])}


def _members_bounds(members):
    bs = [m['outer'].bounds for m in members]
    return (min(b[0] for b in bs), min(b[1] for b in bs),
            max(b[2] for b in bs), max(b[3] for b in bs))


def make_member(idx, part, allow_holes):
    """One copy of a detected part. 'material' is the shape other parts must avoid."""
    outer = part['outer']
    solid = Polygon(outer.exterior)
    material = solid
    net = solid.area
    holes = [g for g in part['inners'] if g.geom_type == 'Polygon' and g.area > 1.0]
    if holes:
        net = max(0.0, solid.area - sum(g.area for g in holes))
        if allow_holes:
            try:
                material = solid.difference(unary_union(holes))
            except Exception:
                material = solid
    return {'part': idx, 'net': net, 'outer': outer,
            'inners': list(part['inners']), 'material': material}


def _hull_edge_angles(points, top=2):
    """Rotations that lay the longest hull edges flat against the sheet edges."""
    pts = sorted(set((round(x, 4), round(y, 4)) for x, y in points))
    if len(pts) < 3:
        return []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    edges = []
    for i in range(len(hull)):
        (x1, y1), (x2, y2) = hull[i], hull[(i + 1) % len(hull)]
        edges.append((math.hypot(x2 - x1, y2 - y1), math.degrees(math.atan2(y2 - y1, x2 - x1))))
    edges.sort(reverse=True)
    out = []
    for _, a in edges[:top]:
        out += [(-a + k * 90.0) % 360.0 for k in range(4)]
    return out


def _angle_set(n_rot, smart, members):
    n = max(1, int(n_rot))
    angles = [i * 360.0 / n for i in range(n)]
    if smart:
        pts = []
        for m in members:
            pts.extend((c[0], c[1]) for c in m['outer'].exterior.coords)
        angles += _hull_edge_angles(pts)
    out = []
    for a in sorted(x % 360.0 for x in angles):
        if all(min(abs(a - b), 360.0 - abs(a - b)) > 0.75 for b in out):
            out.append(a)
    return out


def _rasterize(geoms, res, pad, w, h):
    """Footprint mask: shape grown by 'pad' (= spacing/2) drawn onto a res-mm pixel grid.
    Pixel (0,0) sits at (-pad,-pad) relative to the shape's bounding-box corner.
    Any pixel touched by the footprint is marked, so masks never under-estimate."""
    W = max(1, int(math.ceil((w + 2 * pad) / res - 1e-9)))
    H = max(1, int(math.ceil((h + 2 * pad) / res - 1e-9)))
    sc = SUPERSAMPLE / res
    total = np.zeros((H, W), dtype=bool)
    polys = []
    for g in geoms:
        fp = g.buffer(pad, 4) if pad > 0 else g
        polys.extend([fp] if fp.geom_type == 'Polygon' else list(fp.geoms))
    for p in polys:
        img = Image.new('L', (W * SUPERSAMPLE, H * SUPERSAMPLE), 0)
        dr = ImageDraw.Draw(img)
        dr.polygon([((x + pad) * sc, (y + pad) * sc) for x, y in p.exterior.coords], fill=1, outline=1)
        for ring in p.interiors:
            dr.polygon([((x + pad) * sc, (y + pad) * sc) for x, y in ring.coords], fill=0)
        a = np.asarray(img, dtype=np.uint8).reshape(H, SUPERSAMPLE, W, SUPERSAMPLE).max(axis=(1, 3))
        total |= a.astype(bool)
    return total


class Variant:
    """One rotation of a piece, ready for collision search."""
    __slots__ = ('angle', 'members', 'w', 'h', 'mask', 'maskf', 'npx')

    def __init__(self, angle, members, w, h, mask):
        self.angle, self.members, self.w, self.h = angle, members, w, h
        self.mask = mask
        self.maskf = mask.astype(np.float32)
        self.npx = int(mask.sum())


class Piece:
    """A thing to place: one part, or a rigid group (e.g. an interlocked pair)."""

    def __init__(self, key, members, n_rot, smart):
        minx, miny, maxx, maxy = _members_bounds(members)
        self.members = [_xf(m, lambda g: translate(g, -minx, -miny)) for m in members]
        self.w, self.h = maxx - minx, maxy - miny
        self.key = key
        self.n_rot, self.smart = n_rot, smart
        self.area = sum(m['net'] for m in self.members)
        self.angles = _angle_set(n_rot, smart, self.members)
        self._vc = {}

    def variants(self, res, pad):
        k = (res, pad)
        if k in self._vc:
            return self._vc[k]
        out, seen = [], set()
        for a in self.angles:
            ms = self.members if a == 0 else [
                _xf(m, lambda g, a=a: rotate(g, a, origin=(0, 0))) for m in self.members]
            minx, miny, maxx, maxy = _members_bounds(ms)
            if minx != 0 or miny != 0:
                ms = [_xf(m, lambda g, x=minx, y=miny: translate(g, -x, -y)) for m in ms]
            w, h = maxx - minx, maxy - miny
            mask = _rasterize([m['material'] for m in ms], res, pad, w, h)
            sig = (mask.shape, hash(mask.tobytes()))
            if sig in seen:
                continue
            seen.add(sig)
            out.append(Variant(a, ms, w, h, mask))
        self._vc[k] = out
        return out


class Ctx:
    """Sheet grid settings for one resolution."""

    def __init__(self, sheet_w, sheet_h, margin, spacing, res):
        self.res = float(res)
        self.pad = spacing / 2.0 + 0.05      # +0.05 mm covers polygon/raster rounding
        self.spacing = spacing
        self.uw = sheet_w - 2 * margin
        self.uh = sheet_h - 2 * margin
        self.W = int((self.uw + 2 * self.pad) / self.res + 1e-6)
        self.H = int((self.uh + 2 * self.pad) / self.res + 1e-6)
        self.cant = set()          # piece keys that cannot fit an empty sheet


class Sheet:
    def __init__(self, ctx):
        self.ctx = ctx
        self.H, self.W = ctx.H, ctx.W
        self.occ = np.zeros((self.H, self.W), dtype=bool)
        self.free = self.H * self.W
        self.right = 0              # frontier in pixels
        self.right_mm = 0.0         # frontier in mm (real part edge)
        self.placed = []            # (piece, variant, ox, oy)
        self._c = {}
        self.failed = set()         # piece keys already known not to fit right now

    def _fo(self, crop):
        k = ('o', crop)
        if k not in self._c:
            shape = (_next_fast(self.H), _next_fast(crop))
            self._c[k] = (shape, np.fft.rfft2(self.occ[:, :crop].astype(np.float32), s=shape))
        return self._c[k]

    def _fd(self, crop):
        """FFT of 'occupied or next to occupied or wall' - used to measure snug contact."""
        k = ('d', crop)
        if k not in self._c:
            occ = self.occ[:, :crop]
            d = occ.copy()
            d[1:] |= occ[:-1]
            d[:-1] |= occ[1:]
            d[:, 1:] |= occ[:, :-1]
            d[:, :-1] |= occ[:, 1:]
            d[0] = True
            d[-1] = True
            d[:, 0] = True
            if crop == self.W:
                d[:, -1] = True
            shape = (_next_fast(self.H), _next_fast(crop))
            self._c[k] = np.fft.rfft2(d.astype(np.float32), s=shape)
        return self._c[k]

    def find_move(self, variants, wy, wc):
        """Best (score, variant, ox, oy) over every rotation and every free position."""
        best = None
        for v in variants:
            h, w = v.mask.shape
            if h > self.H or w > self.W:
                continue
            q = CROP_QUANT
            crop = min(self.W, -(-(self.right + w) // q) * q)
            shape, fo = self._fo(crop)
            fm = np.conj(np.fft.rfft2(v.maskf, s=shape))
            ny, nx = self.H - h + 1, crop - w + 1
            overlap = np.fft.irfft2(fo * fm, s=shape)[:ny, :nx]
            valid = overlap < 0.5
            if not valid.any():
                continue
            score = (np.arange(nx)[None, :] + w) + wy * np.arange(ny)[:, None]
            if wc > 0:
                contact = np.fft.irfft2(self._fd(crop) * fm, s=shape)[:ny, :nx]
                score = score - wc * contact
            score = np.where(valid, score, np.inf)
            k = int(np.argmin(score))
            oy, ox = divmod(k, nx)
            s = float(score[oy, ox])
            if best is None or s < best[0]:
                best = (s, v, ox, oy)
        return best

    def place(self, piece, v, ox, oy):
        h, w = v.mask.shape
        self.occ[oy:oy + h, ox:ox + w] |= v.mask
        self.free -= v.npx
        self.right = max(self.right, ox + w)
        self.right_mm = max(self.right_mm, ox * self.ctx.res + v.w)
        self.placed.append((piece, v, ox, oy))
        self._c.clear()
        self.failed.clear()


def _primary(sheets, unplaced, ctx):
    if not sheets:
        return unplaced * 1e9
    return unplaced * 1e9 + (len(sheets) - 1) * ctx.uw + sheets[-1].right_mm


class Layout:
    def __init__(self, sheets, unplaced, ctx):
        self.sheets, self.unplaced, self.ctx = sheets, unplaced, ctx
        self.primary = _primary(sheets, unplaced, ctx)
        last = sheets[-1].placed if sheets else []
        res = ctx.res
        self.secondary = (sum(ox * res + v.w + oy * res + v.h for _, v, ox, oy in last) / len(last)) if last else 0.0
        self.area = sum(p.area for sh in sheets for p, _, _, _ in sh.placed)
        self.count = sum(len(v.members) for sh in sheets for _, v, _, _ in sh.placed)   # parts, not groups

    @property
    def n_sheets(self):
        return len(self.sheets)

    @property
    def last_len(self):
        return self.sheets[-1].right_mm if self.sheets else 0.0

    def utilisation(self, sheet_w, sheet_h):
        return 100.0 * self.area / (max(1, self.n_sheets) * sheet_w * sheet_h)

    def describe(self, sheet_w, sheet_h):
        return (f"{self.n_sheets} sheet(s), last sheet uses {self.last_len:.0f} mm of length, "
                f"utilisation {self.utilisation(sheet_w, sheet_h):.1f}%")


def run_layout(pieces, order, wts, ctx, bound=None):
    """Play one complete game: place every piece (in the given order) using the best move.
    Returns None if the game is abandoned because it can no longer beat 'bound'."""
    wy, wc = wts
    sheets, unplaced = [], 0
    for idx in order:
        p = pieces[idx]
        if p.key in ctx.cant:
            unplaced += 1
            continue
        vs = p.variants(ctx.res, ctx.pad)
        min_px = min(v.npx for v in vs)
        done = False
        for sh in sheets:
            if p.key in sh.failed or sh.free < min_px:
                continue
            mv = sh.find_move(vs, wy, wc)
            if mv is None:
                sh.failed.add(p.key)
                continue
            sh.place(p, mv[1], mv[2], mv[3])
            done = True
            break
        if not done:
            sh = Sheet(ctx)
            mv = sh.find_move(vs, wy, wc)
            if mv is None:
                ctx.cant.add(p.key)
                unplaced += 1
                continue
            sh.place(p, mv[1], mv[2], mv[3])
            sheets.append(sh)
        if bound is not None and _primary(sheets, unplaced, ctx) > bound:
            return None
    return Layout(sheets, unplaced, ctx)


def build_pair(piece, res, pad):
    """Find the tightest way to interlock two copies of a piece (any rotation of the 2nd copy).
    Returns a rigid 2-copy Piece, or None if pairing does not beat a single copy."""
    vs = piece.variants(res, pad)
    v1 = next((v for v in vs if v.angle == 0), vs[0])
    h1, w1 = v1.mask.shape
    hm = max(v.mask.shape[0] for v in vs)
    wm = max(v.mask.shape[1] for v in vs)
    H, W = h1 + 2 * hm, w1 + 2 * wm
    oy1, ox1 = hm, wm
    occ = np.zeros((H, W), np.float32)
    occ[oy1:oy1 + h1, ox1:ox1 + w1] = v1.maskf
    shape = (_next_fast(H), _next_fast(W))
    fo = np.fft.rfft2(occ, s=shape)
    best = None
    for v in vs:
        h, w = v.mask.shape
        ov = np.fft.irfft2(fo * np.conj(np.fft.rfft2(v.maskf, s=shape)), s=shape)[:H - h + 1, :W - w + 1]
        ox, oy = np.arange(W - w + 1), np.arange(H - h + 1)
        uw = np.maximum(ox1 + w1, ox + w) - np.minimum(ox1, ox)
        uh = np.maximum(oy1 + h1, oy + h) - np.minimum(oy1, oy)
        area = np.where(ov < 0.5, uh[:, None].astype(float) * uw[None, :], np.inf)
        k = int(np.argmin(area))
        a = float(area.flat[k])
        if np.isfinite(a) and (best is None or a < best[0]):
            iy, ix = divmod(k, area.shape[1])
            best = (a, v, ix, iy)
    if best is None:
        return None
    _, v2, ix, iy = best
    dx, dy = (ix - ox1) * res, (iy - oy1) * res
    m2 = [_xf(m, lambda g: translate(g, dx, dy)) for m in v2.members]
    pair = Piece((piece.key[0], piece.key[0]), v1.members + m2, piece.n_rot, piece.smart)
    single_util = max(piece.area / (v.w * v.h) for v in vs if v.w * v.h > 0)
    pair_util = pair.area / (pair.w * pair.h)
    return pair if pair_util > single_util * 1.03 else None


def _noisy_order(pieces, rng):
    sigma = rng.choice((0.05, 0.15, 0.3, 0.6))
    mode = rng.choice(('bbox', 'area', 'side'))

    def base(p):
        return p.w * p.h if mode == 'bbox' else (p.area if mode == 'area' else max(p.w, p.h))

    keys = [(math.log(max(base(p), 1e-6)) + rng.gauss(0, sigma), i) for i, p in enumerate(pieces)]
    return [i for _, i in sorted(keys, reverse=True)]


def _mutate(genome, modes, rng):
    mode, order, wi = genome
    order = list(order)
    r = rng.random()
    if r < 0.08 and len(modes) > 1:
        mode = rng.choice([m for m in modes if m != mode])
        return (mode, tuple(_noisy_order(modes[mode], rng)), wi)
    n = len(order)
    if n > 1:
        op = rng.random()
        if op < 0.45:
            i, j = rng.sample(range(n), 2)
            order[i], order[j] = order[j], order[i]
        elif op < 0.8:
            i, j = rng.randrange(n), rng.randrange(n)
            order.insert(j, order.pop(i))
        else:
            i = rng.randrange(n)
            j = min(n, i + rng.randint(2, 6))
            order[i:j] = reversed(order[i:j])
    if rng.random() < 0.3:
        wi = (rng.randrange(len(WY_CHOICES)), rng.randrange(len(WC_CHOICES)))
    return (mode, tuple(order), wi)


def optimise(modes, ctx_s, ctx_f, budget, seed=1, keep=3, cb=None, spacing=None):
    """Search for the best layout for 'budget' seconds.
    modes : {'single': [Piece,...], 'paired': [Piece,...]}  (same parts, different groupings)
    Searches on the coarse grid (ctx_s), then replays the best lines on the fine grid (ctx_f)."""
    rng = random.Random(seed)
    t0 = time.time()
    state = {'evals': 0, 'pruned': 0, 'baseline': None}
    top = []

    def weights(wi):
        return WY_CHOICES[wi[0]], WC_CHOICES[wi[1]]

    def evaluate(genome):
        mode, order, wi = genome
        bound = top[-1][0][0] if len(top) >= keep else None
        lay = run_layout(modes[mode], order, weights(wi), ctx_s, bound)
        state['evals'] += 1
        if lay is None:
            state['pruned'] += 1
            return
        if state['baseline'] is None:
            state['baseline'] = lay
        key = (lay.primary, lay.secondary)
        if any(abs(t[0][0] - key[0]) < 1e-9 and abs(t[0][1] - key[1]) < 1e-9 for t in top):
            return
        top.append((key, state['evals'], genome, lay))
        top.sort(key=lambda t: (t[0], t[1]))
        del top[keep:]

    def report(stage='thinking'):
        if cb:
            cb({'stage': stage, 'evals': state['evals'], 'pruned': state['pruned'],
                'elapsed': time.time() - t0, 'budget': budget,
                'best': top[0][3] if top else None, 'baseline': state['baseline']})

    default_wi = (1, 1)
    first_mode = 'single' if 'single' in modes else next(iter(modes))
    # Seed lines: classic "biggest first" orderings, then a spread of placement styles
    for mode, pieces in modes.items():
        idx = list(range(len(pieces)))
        keyfs = [lambda i, p=pieces: p[i].w * p[i].h,
                 lambda i, p=pieces: p[i].area,
                 lambda i, p=pieces: max(p[i].w, p[i].h)]
        seeds = [tuple(sorted(idx, key=kf, reverse=True)) for kf in keyfs]
        if mode == first_mode:
            evaluate((mode, seeds[0], default_wi))        # this becomes the "plain greedy" baseline
        for wi in [(0, 0), (2, 1), (3, 1), (4, 2), (1, 2)]:
            if top and time.time() - t0 > budget:
                break
            evaluate((mode, seeds[0], wi))
        report()
    # Think until the clock runs out
    mode_names = list(modes)
    while time.time() - t0 < budget:
        if not top or rng.random() < 0.3:
            mode = rng.choice(mode_names)
            g = (mode, tuple(_noisy_order(modes[mode], rng)),
                 (rng.randrange(len(WY_CHOICES)), rng.randrange(len(WC_CHOICES))))
        else:
            g = _mutate(top[rng.randrange(min(len(top), 2))][2], modes, rng)
        evaluate(g)
        report()

    # Replay the best lines on the fine grid and keep the winner
    report('refining')
    finals = [t[3] for t in top]
    if ctx_f is not None and ctx_f.res != ctx_s.res:
        for _, _, g, _ in top:
            mode, order, wi = g
            finals.append(run_layout(modes[mode], order, weights(wi), ctx_f, None))
    spacing = ctx_s.spacing if spacing is None else spacing

    def rank(L):
        # exact-geometry safety first (no overlap / out-of-bounds / gap clearly below spacing), then the score
        ov, oob, gap = verify_layout(L, spacing)
        bad = 1 if (ov or oob or gap < spacing - 0.25) else 0
        return (bad, L.primary, L.secondary)

    best = min(finals, key=rank)
    report('done')
    return {'best': best, 'baseline': state['baseline'], 'evals': state['evals'],
            'pruned': state['pruned'], 'elapsed': time.time() - t0}


# =====================================================================
#  3. VERIFY / EXPORT HELPERS
# =====================================================================
def verify_layout(layout, spacing):
    """Exact-geometry safety check. Returns (overlaps, out_of_bounds, min_gap_mm)."""
    ctx, res = layout.ctx, layout.ctx.res
    overlaps, oob, min_gap = 0, 0, float('inf')
    for sh in layout.sheets:
        geoms = []
        for _, v, ox, oy in sh.placed:
            for m in v.members:
                geoms.append(translate(m['material'], ox * res, oy * res))
        bs = [g.bounds for g in geoms]
        for b in bs:
            if b[0] < -1e-6 or b[1] < -1e-6 or b[2] > ctx.uw + 1e-6 or b[3] > ctx.uh + 1e-6:
                oob += 1
        for i in range(len(geoms)):
            for j in range(i + 1, len(geoms)):
                a, b = bs[i], bs[j]
                near = spacing + 10.0
                if (a[2] + near < b[0] or b[2] + near < a[0] or
                        a[3] + near < b[1] or b[3] + near < a[1]):
                    continue
                if geoms[i].intersects(geoms[j]):
                    overlaps += 1
                else:
                    min_gap = min(min_gap, geoms[i].distance(geoms[j]))
    return overlaps, oob, min_gap


def layout_items(layout, margin):
    """Placed geometry per sheet, in real sheet coordinates (margin included)."""
    res = layout.ctx.res
    sheets = []
    for sh in layout.sheets:
        items = []
        for _, v, ox, oy in sh.placed:
            dx, dy = ox * res + margin, oy * res + margin
            for m in v.members:
                items.append({'outer': translate(m['outer'], dx, dy),
                              'inners': [translate(g, dx, dy) for g in m['inners']]})
        sheets.append(items)
    return sheets


def plot_sheets(sheets, sheet_w, sheet_h, margin):
    figs = []
    for idx, items in enumerate(sheets):
        fig, ax = plt.subplots(figsize=(10, (sheet_h / sheet_w) * 10))
        ax.set_xlim(0, sheet_w)
        ax.set_ylim(0, sheet_h)
        ax.set_facecolor('#1e1e1e')
        fig.patch.set_facecolor('#1e1e1e')
        ax.plot([margin, sheet_w - margin, sheet_w - margin, margin, margin],
                [margin, margin, sheet_h - margin, sheet_h - margin, margin],
                color='#555', linestyle='dashed')
        for item in items:
            x, y = item['outer'].exterior.xy
            ax.plot(x, y, color='#4daafc')
            ax.fill(x, y, alpha=0.5, color='#4daafc')
            for inner in item['inners']:
                if inner.geom_type == 'Polygon':
                    ix, iy = inner.exterior.xy
                    ax.plot(ix, iy, color='#111')
                    ax.fill(ix, iy, color='#111')
        plt.title(f"Sheet {idx + 1} ({len(items)} parts)", color='white')
        figs.append(fig)
    return figs


def export_dxf(sheets, sheet_w, sheet_h):
    out_doc = ezdxf.new(dxfversion='R2010')
    out_doc.header['$INSUNITS'] = 4
    out_doc.header['$MEASUREMENT'] = 1
    msp = out_doc.modelspace()
    for idx, items in enumerate(sheets):
        off = idx * (sheet_w + 500)
        msp.add_lwpolyline([(off, 0), (off + sheet_w, 0), (off + sheet_w, sheet_h),
                            (off, sheet_h), (off, 0)], dxfattribs={'color': 1})
        for item in items:
            msp.add_lwpolyline(list(translate(item['outer'], xoff=off).exterior.coords), dxfattribs={'color': 7})
            for inner in item['inners']:
                t = translate(inner, xoff=off)
                if t.geom_type == 'Polygon':
                    msp.add_lwpolyline(list(t.exterior.coords), dxfattribs={'color': 7})
                elif t.geom_type in ['LineString', 'LinearRing']:
                    msp.add_lwpolyline(list(t.coords), dxfattribs={'color': 7})
    buffer = io.StringIO()
    out_doc.write(buffer)
    return buffer.getvalue()


# =====================================================================
#  4. WEB USER INTERFACE
# =====================================================================
def main():
    import streamlit as st

    st.set_page_config(page_title="Thinking Nester", layout="wide")
    
    # --- HIDE STREAMLIT UI (GitHub Logo, Menu, Header, Footer) ---
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                .stDeployButton {display:none;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    st.title("♟️ Thinking Nester")

    st.sidebar.header("Machine Settings")
    sheet_w = st.sidebar.number_input("Sheet Width (mm)", value=2500.0)
    sheet_h = st.sidebar.number_input("Sheet Height (mm)", value=1250.0)
    spacing = st.sidebar.number_input("Part Spacing (mm)", value=3.0)
    margin = st.sidebar.number_input("Edge Margin (mm)", value=5.0)
    rotations = int(st.sidebar.number_input("Rotations (4=90°, 8=45°)", value=4, min_value=1, max_value=36))

    st.sidebar.header("Thinking Settings")
    think_time = st.sidebar.number_input("Thinking time (seconds)", value=30, min_value=1, max_value=900, step=5)
    search_res = st.sidebar.number_input("Search resolution (mm, coarse)", value=5.0, min_value=1.0)
    final_res = st.sidebar.number_input("Final resolution (mm, fine)", value=2.0, min_value=0.5)
    try_pairs = st.sidebar.checkbox("Try interlocked pairs", value=True)
    smart_angles = st.sidebar.checkbox("Also try laying part edges flat", value=True)
    allow_holes = st.sidebar.checkbox("Allow parts inside holes of other parts", value=True)

    uploaded_file = st.sidebar.file_uploader("1. Upload DXF", type=['dxf'])

    if uploaded_file is None:
        return

    if 'parts' not in st.session_state or st.session_state.file_name != uploaded_file.name:
        with st.spinner("Analyzing DXF..."):
            st.session_state.parts = extract_smart_parts(uploaded_file.getvalue())
            st.session_state.file_name = uploaded_file.name
            st.session_state.pop('result', None)

    parts = st.session_state.parts
    if not parts:
        st.error("No valid closed boundaries found.")
        return

    st.sidebar.success(f"Grouped {len(parts)} Master Parts.")
    usable_w, usable_h = sheet_w - margin * 2, sheet_h - margin * 2

    quantities = {}
    st.sidebar.subheader("Detected Parts & Quantities")
    for i, part in enumerate(parts):
        minx, miny, maxx, maxy = part['outer'].bounds
        w, h = round(maxx - minx, 1), round(maxy - miny, 1)
        fits = (w <= usable_w and h <= usable_h) or (h <= usable_w and w <= usable_h)
        st.sidebar.image(generate_part_thumbnail(part), use_container_width=True)
        label = f"Part {i + 1} ({w} x {h} mm)" + ("" if fits else " ⚠️ Exceeds Sheet")
        quantities[i] = st.sidebar.number_input(label, value=1 if fits else 0, min_value=0, key=f"qty_{i}")
        st.sidebar.markdown("---")

    if st.sidebar.button("2. Let it think", use_container_width=True):
        if sum(quantities.values()) == 0:
            st.warning("All quantities are 0.")
        else:
            ctx_s = Ctx(sheet_w, sheet_h, margin, spacing, search_res)
            ctx_f = Ctx(sheet_w, sheet_h, margin, spacing, final_res)
            status, bar = st.empty(), st.progress(0.0)

            status.text("Preparing part shapes...")
            singles, pairs = {}, {}
            for i, part in enumerate(parts):
                if quantities[i] > 0:
                    singles[i] = Piece((i,), [make_member(i, part, allow_holes)], rotations, smart_angles)
                    if try_pairs and quantities[i] >= 2:
                        pairs[i] = build_pair(singles[i], min(search_res, 3.0), ctx_s.pad)

            modes = {'single': [singles[i] for i in singles for _ in range(quantities[i])]}
            if any(p is not None for p in pairs.values()):
                paired = []
                for i in singles:
                    q = quantities[i]
                    if pairs.get(i) is not None:
                        paired += [pairs[i]] * (q // 2) + [singles[i]] * (q % 2)
                    else:
                        paired += [singles[i]] * q
                modes['paired'] = paired

            def on_progress(info):
                bar.progress(min(1.0, info['elapsed'] / info['budget']))
                b = info['best']
                head = "Refining on the fine grid..." if info['stage'] in ('refining', 'done') else "Thinking..."
                if b is not None:
                    status.markdown(f"**{head}** {info['evals']} lines analysed "
                                    f"({info['pruned']} abandoned early) · best so far: "
                                    f"{b.describe(sheet_w, sheet_h)}")

            res = optimise(modes, ctx_s, ctx_f, float(think_time), seed=1, cb=on_progress)
            best, base = res['best'], res['baseline']
            bar.progress(1.0)
            status.empty()

            overlaps, oob, min_gap = verify_layout(best, spacing)
            sheets = layout_items(best, margin)
            st.session_state.result = {
                'summary': best.describe(sheet_w, sheet_h),
                'placed': best.count, 'requested': sum(quantities.values()), 'unplaced': best.unplaced,
                'evals': res['evals'], 'pruned': res['pruned'], 'elapsed': res['elapsed'],
                'gain': (100.0 * (base.primary - best.primary) / base.primary) if base and base.primary > 0 else 0.0,
                'base_desc': base.describe(sheet_w, sheet_h) if base else '',
                'util': best.utilisation(sheet_w, sheet_h), 'last_len': best.last_len,
                'n_sheets': best.n_sheets, 'overlaps': overlaps, 'oob': oob, 'min_gap': min_gap,
                'figs': plot_sheets(sheets, sheet_w, sheet_h, margin),
                'dxf': export_dxf(sheets, sheet_w, sheet_h),
            }

    r = st.session_state.get('result')
    if r:
        st.success(f"Best nest found after analysing {r['evals']} lines in {r['elapsed']:.0f}s: {r['summary']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sheets", r['n_sheets'])
        c2.metric("Parts placed", f"{r['placed']} / {r['requested']}")
        c3.metric("Utilisation", f"{r['util']:.1f}%")
        c4.metric("vs. plain greedy", f"{r['gain']:.1f}% shorter")
        st.caption(f"Plain greedy result was: {r['base_desc']}")
        if r['unplaced']:
            st.warning(f"{r['unplaced']} part(s) could not be placed (too big for the sheet).")
        if r['overlaps'] or r['oob']:
            st.error(f"Geometry check FAILED: {r['overlaps']} overlaps, {r['oob']} out-of-bounds parts.")
        else:
            if r['min_gap'] == float('inf'):
                st.caption("✅ Exact geometry check passed: no overlaps, all parts inside the sheet margins.")
            elif r['min_gap'] < spacing - 0.25:
                st.warning(f"No overlaps, but the tightest gap is {r['min_gap']:.2f} mm (requested {spacing:g} mm). "
                           f"Lower the Final resolution for a tighter guarantee.")
            else:
                st.caption(f"✅ Exact geometry check passed: no overlaps, tightest gap {r['min_gap']:.2f} mm "
                           f"(requested {spacing:g} mm).")
        for fig in r['figs']:
            st.pyplot(fig)
        st.download_button(label="⬇️ Download Nested DXF", data=r['dxf'], file_name="nested_result.dxf",
                           mime="application/dxf", type="primary")


if __name__ == "__main__":
    main()
