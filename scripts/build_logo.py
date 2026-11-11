#!/usr/bin/env python3
"""Build the signal_spatial mark from the Tennessee River Gorge.

The letter S is the river meander directly below Signal Mountain. The gorge
is drawn as stacked relief layers, the way a cut terrain model is built:
broad filled terraces, a soft shadow under each step, and a faint grid on
the lowest board. The meander is rotated so it stands upright, on a warm
black ground.

Needs numpy, scipy, pillow, matplotlib, and fontTools. The script requests public
AWS terrain tiles. Behind a Zscaler proxy, point Python at the local
certificate bundle rather than disabling TLS verification.

Rebuild from the repo root:

    python scripts/build_logo.py
"""

from __future__ import annotations

import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "brand"

# Tennessee River centerline through the gorge meander under Signal Mountain,
# from near Signal Point (east) upstream through the bend (west).
# Source: OpenStreetMap way 77441793, trimmed to this meander.
RIVER = np.array([
    (35.10875, -85.36586), (35.10900, -85.36624), (35.11014, -85.36782),
    (35.11281, -85.37087), (35.11333, -85.37171), (35.11450, -85.37313),
    (35.11471, -85.37421), (35.11524, -85.37570), (35.11547, -85.37691),
    (35.11660, -85.37993), (35.11727, -85.38193), (35.11754, -85.38384),
    (35.11770, -85.38443), (35.11890, -85.38774), (35.11982, -85.38944),
    (35.12000, -85.39039), (35.12049, -85.39188), (35.12112, -85.39356),
    (35.12132, -85.39562), (35.12146, -85.39844), (35.12126, -85.40206),
    (35.12084, -85.40400), (35.12064, -85.40602), (35.12022, -85.40770),
    (35.11991, -85.40847), (35.11919, -85.40990), (35.11836, -85.41086),
    (35.11734, -85.41257), (35.11681, -85.41328), (35.11638, -85.41355),
    (35.11458, -85.41501), (35.11305, -85.41516), (35.11006, -85.41499),
    (35.10906, -85.41512), (35.10780, -85.41548), (35.10641, -85.41514),
    (35.10573, -85.41508), (35.10362, -85.41441), (35.10262, -85.41391),
    (35.10099, -85.41317), (35.09984, -85.41286), (35.09803, -85.41185),
    (35.09621, -85.41118), (35.09535, -85.41021), (35.09468, -85.40952),
    (35.09369, -85.40808), (35.09281, -85.40698), (35.09151, -85.40493),
    (35.09112, -85.40375), (35.09099, -85.40280), (35.09102, -85.40110),
    (35.09128, -85.39991), (35.09148, -85.39736), (35.09105, -85.39558),
    (35.09065, -85.39406), (35.09069, -85.39313), (35.08987, -85.39108),
    (35.08921, -85.38979), (35.08867, -85.38904), (35.08599, -85.38586),
    (35.08345, -85.38385), (35.08236, -85.38331), (35.08103, -85.38306),
    (35.07939, -85.38314), (35.07726, -85.38366), (35.07596, -85.38448),
    (35.07482, -85.38555), (35.07417, -85.38651), (35.07364, -85.38766),
    (35.07326, -85.38904), (35.07269, -85.38988), (35.07232, -85.39014),
    (35.07173, -85.39035), (35.07103, -85.39087), (35.07043, -85.39115),
    (35.06869, -85.39226), (35.06631, -85.39910), (35.06450, -85.40700),
    (35.06273, -85.41034), (35.06081, -85.41750), (35.05890, -85.42240),
    (35.05815, -85.42362), (35.05688, -85.42568), (35.05605, -85.42689),
    (35.05473, -85.42792), (35.05440, -85.42940), (35.05368, -85.43044),
    (35.05237, -85.43153), (35.05110, -85.43220), (35.05043, -85.43234),
    (35.04986, -85.43232), (35.04886, -85.43279), (35.04738, -85.43306),
    (35.04609, -85.43306), (35.04294, -85.43265), (35.04129, -85.43223),
    (35.03722, -85.43200), (35.03438, -85.43236), (35.03234, -85.43313),
    (35.03101, -85.43400), (35.02871, -85.43604), (35.02773, -85.43740),
    (35.02632, -85.43957), (35.02581, -85.44069), (35.02554, -85.44178),
    (35.02435, -85.44447), (35.02406, -85.44561), (35.02369, -85.44643),
    (35.02365, -85.44744), (35.02323, -85.44995), (35.02296, -85.45077),
    (35.02244, -85.45174), (35.02235, -85.45197), (35.02235, -85.45343),
    (35.02207, -85.45511), (35.02188, -85.45722), (35.02193, -85.45883),
    (35.02232, -85.46066), (35.02243, -85.46247), (35.02288, -85.46396),
    (35.02390, -85.46643), (35.02414, -85.46736), (35.02459, -85.46817),
    (35.02633, -85.46998), (35.02873, -85.47163), (35.02936, -85.47196),
    (35.02978, -85.47202), (35.03063, -85.47191), (35.03110, -85.47169),
    (35.03169, -85.47220), (35.03252, -85.47259), (35.03332, -85.47281),
    (35.03445, -85.47335), (35.03520, -85.47340), (35.03581, -85.47363),
    (35.03632, -85.47403), (35.03682, -85.47470), (35.03772, -85.47514),
    (35.03843, -85.47565), (35.03971, -85.47619), (35.04043, -85.47661),
    (35.04116, -85.47709), (35.04200, -85.47725), (35.04247, -85.47746),
    (35.04312, -85.47793), (35.04361, -85.47849), (35.04441, -85.47898),
    (35.04530, -85.47936), (35.04580, -85.47978), (35.04712, -85.48048),
    (35.04782, -85.48050), (35.04845, -85.48082), (35.04888, -85.48122),
    (35.04961, -85.48139), (35.05040, -85.48131), (35.05278, -85.48189),
    (35.05422, -85.48195), (35.05582, -85.48213), (35.05641, -85.48201),
    (35.05705, -85.48171), (35.05734, -85.48152), (35.06087, -85.48040),
    (35.06258, -85.48024), (35.06375, -85.47986), (35.06435, -85.47975),
    (35.06466, -85.47985), (35.06569, -85.47950), (35.06655, -85.47953),
    (35.06662, -85.47949), (35.06952, -85.47877), (35.07030, -85.47816),
    (35.07141, -85.47866), (35.07367, -85.47904), (35.07483, -85.47969),
    (35.07599, -85.48070), (35.07665, -85.48144), (35.07763, -85.48209),
    (35.07803, -85.48272), (35.07935, -85.48421),
], dtype=float)

ZOOM = 13
CROP_PAD_PX = 180
BAND_PX = 96
SMOOTH_SIGMA = 9.0
MASK_SIGMA = 3.4
N_LAYERS = 7
# scipy rotates the array counterclockwise. This stands the meander up as an S.
ROTATE_DEG = 70
MARK_SIZE = 512
BG = "#12100e"
BG_RGB = (18, 16, 14)
WORDMARK = "signal_spatial"
WORD_COLOR = "#f3ecdf"
# Lowest board to the crest, in the sand / sage / cream range of a cut terrain model.
PALETTE = np.array([
    [198, 172, 138],
    [188, 180, 160],
    [214, 196, 168],
    [176, 162, 140],
    [226, 214, 192],
    [236, 226, 210],
    [246, 240, 228],
], dtype=np.float32)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/macos/Inter-Medium.ttf",
    "/usr/share/fonts/truetype/inter/Inter-Medium.ttf",
    "/Library/Fonts/Inter-Medium.ttf",
    str(Path.home() / "Library/Fonts/Inter-Medium.ttf"),
]


def deg2num(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def num2deg(x: float, y: float, zoom: int) -> tuple[float, float]:
    n = 2.0 ** zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def fetch_mosaic() -> tuple[np.ndarray, float, float, float, float]:
    south, north = float(RIVER[:, 0].min()) - 0.02, float(RIVER[:, 0].max()) + 0.02
    west, east = float(RIVER[:, 1].min()) - 0.02, float(RIVER[:, 1].max()) + 0.02
    x0, y_n = deg2num(north, west, ZOOM)
    x1, y_s = deg2num(south, east, ZOOM)
    xs, ys = range(x0, x1 + 1), range(y_n, y_s + 1)
    tiles = [(x, y) for y in ys for x in xs]
    cache = Path("/tmp/ss-logo-tiles")
    cache.mkdir(parents=True, exist_ok=True)

    def fetch(xy: tuple[int, int]) -> None:
        x, y = xy
        path = cache / f"{ZOOM}_{x}_{y}.png"
        if path.exists() and path.stat().st_size > 100:
            return
        url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{ZOOM}/{x}/{y}.png"
        req = urllib.request.Request(url, headers={"User-Agent": "signal-spatial-logo/1.0"})
        with urllib.request.urlopen(req, timeout=40) as resp:
            path.write_bytes(resp.read())

    with ThreadPoolExecutor(max_workers=12) as pool:
        for fut in as_completed([pool.submit(fetch, t) for t in tiles]):
            fut.result()

    sample = np.asarray(Image.open(cache / f"{ZOOM}_{x0}_{y_n}.png").convert("RGB"))
    th, tw = sample.shape[:2]
    mosaic = np.zeros((len(ys) * th, len(xs) * tw), dtype=np.float32)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            im = np.asarray(Image.open(cache / f"{ZOOM}_{x}_{y}.png").convert("RGB"), dtype=np.float32)
            elev = (im[:, :, 0] * 256.0 + im[:, :, 1] + im[:, :, 2] / 256.0) - 32768.0
            mosaic[iy * th:(iy + 1) * th, ix * tw:(ix + 1) * tw] = elev
    lat_n, lon_w = num2deg(x0, y_n, ZOOM)
    lat_s, lon_e = num2deg(x1 + 1, y_s + 1, ZOOM)
    return mosaic, lat_n, lat_s, lon_w, lon_e


def chamfer(mask: np.ndarray) -> np.ndarray:
    """Approximate pixel distance from the river. Enough to clip a gorge wall."""
    h, w = mask.shape
    inf = np.full((h, w), 1e6, dtype=np.float32)
    inf[mask] = 0
    for y in range(h):
        row = inf[y]
        for x in range(w):
            v = row[x]
            if x:
                v = min(v, row[x - 1] + 1)
            if y:
                v = min(v, inf[y - 1, x] + 1)
                if x:
                    v = min(v, inf[y - 1, x - 1] + 1.4142)
            row[x] = v
    for y in range(h - 1, -1, -1):
        row = inf[y]
        for x in range(w - 1, -1, -1):
            v = row[x]
            if x + 1 < w:
                v = min(v, row[x + 1] + 1)
            if y + 1 < h:
                v = min(v, inf[y + 1, x] + 1)
                if x + 1 < w:
                    v = min(v, inf[y + 1, x + 1] + 1.4142)
            row[x] = v
    return inf


def rdp(pts: np.ndarray, eps: float) -> np.ndarray:
    if len(pts) < 3:
        return pts
    start, end = pts[0], pts[-1]
    seg = end - start
    length = float(math.hypot(seg[0], seg[1]))
    if length < 1e-9:
        dist = np.hypot(pts[:, 0] - start[0], pts[:, 1] - start[1])
    else:
        dist = np.abs(seg[0] * (start[1] - pts[:, 1]) - seg[1] * (start[0] - pts[:, 0])) / length
    i = int(np.argmax(dist))
    if dist[i] > eps:
        return np.vstack([rdp(pts[: i + 1], eps)[:-1], rdp(pts[i:], eps)])
    return np.vstack([start, end])


def smooth_open(pts: np.ndarray, win: int) -> np.ndarray:
    if len(pts) < win:
        return pts
    kernel = np.ones(win) / win
    pad = win // 2
    x = np.pad(pts[:, 0], pad, mode="edge")
    y = np.pad(pts[:, 1], pad, mode="edge")
    return np.column_stack([np.convolve(x, kernel, mode="valid"), np.convolve(y, kernel, mode="valid")])


def hex_color(rgb: np.ndarray) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(round(c)) for c in rgb))


def darken(rgb: np.ndarray, factor: float) -> str:
    return hex_color(np.clip(rgb * factor, 0, 255))


def gorge_index(elev: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Smoothed terrace index for the meander. -1 is outside the mark."""
    smooth = ndimage.gaussian_filter(elev, sigma=SMOOTH_SIGMA)
    raw = dist < BAND_PX
    labels, count = ndimage.label(raw)
    if count:
        sizes = ndimage.sum(raw, labels, range(1, count + 1))
        raw = labels == (int(np.argmax(sizes)) + 1)
    corridor = ndimage.gaussian_filter(raw.astype(np.float32), sigma=MASK_SIGMA) > 0.5
    # Rotate the array counterclockwise so the river meander stands up as an S.
    elev_r = ndimage.rotate(smooth, ROTATE_DEG, reshape=True, order=1, cval=float(smooth.min()))
    mask_r = ndimage.rotate(corridor.astype(np.float32), ROTATE_DEG, reshape=True, order=1, cval=0) > 0.5
    inside = elev_r[mask_r]
    lo, hi = np.percentile(inside, [12, 90])
    levels = np.linspace(lo, hi, N_LAYERS)
    index = np.clip(np.digitize(elev_r, levels[1:]), 0, N_LAYERS - 1)
    return np.where(mask_r, index, -1)


def shade_mark(index: np.ndarray) -> np.ndarray:
    """Paint terraces with a step shadow, a cut edge, and a grid on the base board."""
    height, width = index.shape
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    shadow = np.zeros((height, width), dtype=np.float32)
    for level in range(N_LAYERS):
        if level:
            mask = (index >= level).astype(np.float32)
            shifted = np.zeros_like(mask)
            shifted[6:, 5:] = mask[:-6, :-5]
            blurred = ndimage.gaussian_filter(shifted, sigma=2.8)
            lower = (index >= 0) & (index < level)
            shadow[lower] = np.maximum(shadow[lower], blurred[lower] * 0.9)
        rgb[index == level] = PALETTE[level]
    land = index >= 0
    rgb[land] *= (1.0 - 0.5 * np.clip(shadow[land], 0, 1))[:, None]
    riser = np.zeros(index.shape, dtype=bool)
    riser[:-4, :-3] = (index[:-4, :-3] > index[4:, 3:]) & (index[:-4, :-3] >= 0)
    rgb[riser] *= 0.74
    lip = np.zeros(index.shape, dtype=bool)
    lip[4:, 3:] = (index[4:, 3:] > index[:-4, :-3]) & (index[4:, 3:] >= 0)
    rgb[lip] = np.minimum(255, rgb[lip] * 1.06 + 10)
    grid = np.zeros(index.shape, dtype=bool)
    grid[::11, :] = True
    grid[:, ::11] = True
    rgb[(index == 0) & grid] *= 0.92
    base = land.astype(np.float32)
    shifted = np.zeros_like(base)
    shifted[16:, 12:] = base[:-16, :-12]
    soft = ndimage.gaussian_filter(shifted, sigma=14)
    bg = np.array(BG_RGB, dtype=np.float32)
    out = bg * (1 - 0.5 * np.clip(soft, 0, 1)[..., None])
    out[land] = rgb[land]
    return np.clip(out, 0, 255).astype(np.uint8)


def frame_of(index: np.ndarray, pad: int = 56) -> tuple[float, float, float]:
    rows, cols = np.where(index >= 0)
    x0, x1 = float(cols.min() - pad), float(cols.max() + pad)
    y0, y1 = float(rows.min() - pad), float(rows.max() + pad)
    side = max(x1 - x0, y1 - y0)
    origin_x = x0 - (side - (x1 - x0)) / 2
    origin_y = y0 - (side - (y1 - y0)) / 2
    return origin_x, origin_y, side


def to_mark(points: np.ndarray, origin_x: float, origin_y: float, side: float) -> np.ndarray:
    xy = np.empty_like(points)
    xy[:, 0] = (points[:, 0] - origin_x) / side * MARK_SIZE
    xy[:, 1] = (points[:, 1] - origin_y) / side * MARK_SIZE
    return xy


def terrace_rings(index: np.ndarray, origin_x: float, origin_y: float, side: float) -> list[tuple[str, list[np.ndarray]]]:
    layers: list[tuple[str, list[np.ndarray]]] = []
    for level in range(N_LAYERS):
        fig, ax = plt.subplots()
        contours = ax.contour((index >= level).astype(float), levels=[0.5])
        plt.close(fig)
        rings: list[np.ndarray] = []
        for seg in contours.allsegs[0]:
            if len(seg) < 12 or float(np.hypot(*(seg[0] - seg[-1]))) > 2.5:
                continue
            simplified = rdp(smooth_open(seg, 11), 2.0)
            if len(simplified) < 8:
                continue
            step = np.diff(simplified, axis=0)
            if float(np.hypot(step[:, 0], step[:, 1]).sum()) < 48:
                continue
            rings.append(to_mark(simplified, origin_x, origin_y, side))
        if rings:
            layers.append((hex_color(PALETTE[level]), rings))
    return layers


def path_d(pts: np.ndarray) -> str:
    cmds = [f"M {pts[0, 0]:.2f} {pts[0, 1]:.2f}"]
    cmds.extend(f"L {x:.2f} {y:.2f}" for x, y in pts[1:])
    cmds.append("Z")
    return " ".join(cmds)


def layer_markup(layers: list[tuple[str, list[np.ndarray]]]) -> str:
    filters = """  <defs>
    <filter id="cast" x="-20%" y="-20%" width="150%" height="150%">
      <feDropShadow dx="0" dy="7" stdDeviation="8" flood-color="#000000" flood-opacity="0.45"/>
    </filter>
    <filter id="step" x="-15%" y="-15%" width="140%" height="140%">
      <feDropShadow dx="1.1" dy="2.2" stdDeviation="1.3" flood-color="#3a2b1c" flood-opacity="0.38"/>
    </filter>
    <pattern id="board" width="14" height="14" patternUnits="userSpaceOnUse">
      <path d="M 14 0 H 0 V 14" fill="none" stroke="#000000" stroke-opacity="0.14" stroke-width="0.7"/>
    </pattern>
  </defs>"""
    parts = [filters]
    for i, (color, rings) in enumerate(layers):
        paths = " ".join(path_d(ring) for ring in rings)
        filt = "cast" if i == 0 else "step"
        edge = darken(PALETTE[i], 0.72)
        parts.append(f'  <g filter="url(#{filt})">')
        parts.append(f'    <path fill="{color}" stroke="{edge}" stroke-width="1.4" stroke-linejoin="round" d="{paths}"/>')
        if i == 0:
            parts.append(f'    <path fill="url(#board)" d="{paths}"/>')
        parts.append("  </g>")
    return "\n".join(parts)


def mark_svg(layers: list[tuple[str, list[np.ndarray]]], background: bool) -> str:
    desc = (
        "Relief mark of the Tennessee River Gorge below Signal Mountain, Tennessee. "
        "Stacked terraces follow the river meander, turned upright so it reads as an S."
    )
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {MARK_SIZE} {MARK_SIZE}" role="img">',
        "  <title>signal_spatial</title>",
        f"  <desc>{desc}</desc>",
    ]
    if background:
        lines.append(f'  <rect width="{MARK_SIZE}" height="{MARK_SIZE}" fill="{BG}"/>')
    lines.append(layer_markup(layers))
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_mark_png(index: np.ndarray, origin_x: float, origin_y: float, side: float, path: Path, px: int) -> None:
    shaded = shade_mark(index)
    x0 = int(math.floor(origin_x))
    y0 = int(math.floor(origin_y))
    x1 = int(math.ceil(origin_x + side))
    y1 = int(math.ceil(origin_y + side))
    canvas = Image.new("RGB", (x1 - x0, y1 - y0), BG_RGB)
    src_x0, src_y0 = max(0, x0), max(0, y0)
    src_x1, src_y1 = min(shaded.shape[1], x1), min(shaded.shape[0], y1)
    tile = Image.fromarray(shaded[src_y0:src_y1, src_x0:src_x1], "RGB")
    canvas.paste(tile, (src_x0 - x0, src_y0 - y0))
    canvas.resize((px, px), Image.Resampling.LANCZOS).save(path, optimize=True)


def find_font() -> Path:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError("Inter Medium was not found. Install it or edit FONT_CANDIDATES.")


def wordmark_group(font_path: Path, em_px: float, origin_x: float, baseline_y: float) -> tuple[str, float]:
    font = TTFont(str(font_path))
    glyphs = font.getGlyphSet()
    cmap = font.getBestCmap()
    upem = font["head"].unitsPerEm
    sx_height = font["OS/2"].sxHeight
    scale = em_px / upem
    tracking = 0.045 * upem
    cursor = 0.0
    parts = []
    for char in WORDMARK:
        name = cmap[ord(char)]
        pen = SVGPathPen(glyphs)
        glyphs[name].draw(pen)
        parts.append(f'    <path transform="translate({cursor:.2f} 0)" d="{pen.getCommands()}"/>')
        cursor += glyphs[name].width + tracking
    center_shift = scale * (sx_height / 2)
    group = (
        f'  <g fill="{WORD_COLOR}" transform="translate({origin_x:.2f} {baseline_y + center_shift:.2f}) '
        f'scale({scale:.6f} {-scale:.6f})">\n'
        + "\n".join(parts)
        + "\n  </g>"
    )
    return group, cursor * scale


def lockup_svg(layers: list[tuple[str, list[np.ndarray]]], font_path: Path) -> str:
    em = 78
    gap = 36
    ink_right = max(float(ring[:, 0].max()) for _, rings in layers for ring in rings)
    text, text_w = wordmark_group(font_path, em, ink_right + gap, MARK_SIZE / 2)
    width = ink_right + gap + text_w + 72
    desc = (
        "signal_spatial lockup. The mark is the Tennessee River Gorge below Signal Mountain, "
        "drawn as stacked relief terraces."
    )
    return "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {MARK_SIZE}" role="img">',
        "  <title>signal_spatial</title>",
        f"  <desc>{desc}</desc>",
        f'  <rect width="{width:.0f}" height="{MARK_SIZE}" fill="{BG}"/>',
        layer_markup(layers),
        text,
        "</svg>",
        "",
    ])


def render_lockup_png(mark_png: Path, font_path: Path, dest: Path) -> None:
    mark = Image.open(mark_png).convert("RGB")
    em = int(round(mark.height * 78 / MARK_SIZE))
    font = ImageFont.truetype(str(font_path), em)
    tracking = int(round(em * 0.045))
    widths = [int(round(font.getlength(char))) for char in WORDMARK]
    text_w = sum(widths) + tracking * (len(WORDMARK) - 1)
    ink = np.abs(np.asarray(mark).astype(int) - np.array(BG_RGB)).sum(axis=2) > 24
    ink_right = int(np.where(ink.any(axis=0))[0].max())
    gap = int(round(mark.width * 36 / MARK_SIZE))
    pad = int(round(mark.width * 72 / MARK_SIZE))
    text_x = ink_right + gap
    canvas = Image.new("RGB", (text_x + text_w + pad, mark.height), BG_RGB)
    canvas.paste(mark, (0, 0))
    draw = ImageDraw.Draw(canvas)
    ascent, _descent = font.getmetrics()
    x_height = font.getbbox("x")[3] - font.getbbox("x")[1]
    baseline = mark.height / 2 + x_height / 2
    x = text_x
    for char, width in zip(WORDMARK, widths):
        draw.text((x, baseline - ascent), char, font=font, fill=WORD_COLOR)
        x += width + tracking
    canvas.save(dest, optimize=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading elevation")
    elev, lat_n, lat_s, lon_w, lon_e = fetch_mosaic()
    height, width = elev.shape

    def to_px(lat: float, lon: float) -> tuple[float, float]:
        py = (lat_n - lat) / (lat_n - lat_s) * (height - 1)
        px = (lon - lon_w) / (lon_e - lon_w) * (width - 1)
        return px, py

    river_px = np.array([to_px(lat, lon) for lat, lon in RIVER])
    x0 = max(0, int(math.floor(river_px[:, 0].min() - CROP_PAD_PX)))
    y0 = max(0, int(math.floor(river_px[:, 1].min() - CROP_PAD_PX)))
    x1 = min(width - 1, int(math.ceil(river_px[:, 0].max() + CROP_PAD_PX)))
    y1 = min(height - 1, int(math.ceil(river_px[:, 1].max() + CROP_PAD_PX)))
    crop = elev[y0:y1, x0:x1]
    mask_img = Image.new("L", (crop.shape[1], crop.shape[0]), 0)
    draw = ImageDraw.Draw(mask_img)
    local = [(float(px - x0), float(py - y0)) for px, py in river_px]
    draw.line(local, fill=1, width=3)
    print("stacking terraces")
    index = gorge_index(crop, chamfer(np.asarray(mask_img) > 0))
    origin_x, origin_y, side = frame_of(index)
    layers = terrace_rings(index, origin_x, origin_y, side)
    print(f"{len(layers)} terraces")

    font_path = find_font()
    (OUT / "logo.svg").write_text(mark_svg(layers, background=True), encoding="utf-8")
    (OUT / "logo-mark.svg").write_text(mark_svg(layers, background=False), encoding="utf-8")
    (OUT / "logo-lockup.svg").write_text(lockup_svg(layers, font_path), encoding="utf-8")
    render_mark_png(index, origin_x, origin_y, side, OUT / "logo.png", 1400)
    render_lockup_png(OUT / "logo.png", font_path, OUT / "logo-lockup.png")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()


