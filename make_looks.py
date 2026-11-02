r"""looks_art.py (v3, 2026-11-03): draws the background pictures of the Ready-made looks (Quick Access > Theme, Settings > Theme & Sharing).

Owner 2026-11-03: "build more UI theme presets ... make sure all u build got good quality images and gifs, redo current preset images as well, i want the best
quality" and "for the themes make a christmas, new year, and other major events". Every picture is generated here from code (gradients, noise, particles, light,
shapes): nothing is copied from anywhere, so the public repo HyperionElite/HyperionLooks can host them. The hub downloads one when its look is applied.

  python tools/looks_art.py [out_dir] [id ...]   default out_dir = %LOCALAPPDATA%\HyperionDeploy\hyperion-looks; ids = only those pictures (the atlas needs all)

Quality (v3): stills are drawn at 2048 x 1536 and reduced to 1024 x 768 (anti-aliased; 1024 is the most a Roblox texture keeps), JPEG quality 95 without chroma
subsampling; animated looks are drawn at 512 x 384 and reduced to 256 x 192, 20 frames that loop without a seam, one shared 255-colour palette with dithering (the
hub decodes them into one 1024 px sprite sheet: 4 x 5 cells of 256 x 192); the thumbnail atlas has 256 x 160 cells, 4 columns.

Writes   looks/<id>_v3.jpg | .png | .gif   one background per look
         thumbs_v3.png                    the atlas of every look's preview, in the order of LOOKS below
         contact.png                      a contact sheet to look at (not used by the hub)
The colours of each look live in the hub (Features.Looks.LIST); the ones here only tint the thumbnails and must match it.
"""
import math, os, random, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

FW, FH = 1024, 768                  # a still as the hub gets it
SS = 2                              # drawn this many times larger, then reduced
W, H = FW * SS, FH * SS
K = W / 800.0                       # the v2 pictures were drawn at 800 px: their pixel sizes are scaled by this
GW, GH, GN, GD = 256, 192, 20, 100  # GIF size, frames, ms per frame (a 2 s loop)
GS = 2                              # GIFs drawn at 2x
TW, TH, TCOLS = 256, 160, 4         # thumbnail cell, atlas columns
VER = "v3"

# id, file, panel hex, accent hex, card hex, image transparency on windows (the thumbnail previews the window), kind
LOOKS = [
    ("carbon",    "carbon_v3.jpg",    "#111113", "#E8B84A", "#19191C", 0.30, "daily"),
    ("ember",     "ember_v3.gif",     "#140C0A", "#FF7A3D", "#1E1210", 0.30, "daily"),
    ("velvet",    "velvet_v3.jpg",    "#14080B", "#E5484D", "#1E0D11", 0.28, "daily"),
    ("synthwave", "synthwave_v3.gif", "#120A1A", "#FF4FA3", "#1C1026", 0.40, "daily"),
    ("nebula",    "nebula_v3.jpg",    "#0F0B1A", "#9B6BFF", "#171127", 0.28, "daily"),
    ("midnight",  "midnight_v3.jpg",  "#0A0F1E", "#4C8DFF", "#111A2E", 0.25, "daily"),
    ("aurora",    "aurora_v3.gif",    "#08121A", "#5AD1E6", "#0F1D26", 0.30, "daily"),
    ("jade",      "jade_v3.png",      "#08130F", "#30C48D", "#0E1D17", 0.30, "daily"),
    ("mono",      "mono_v3.png",      "#0F0F11", "#D8D8E0", "#18181B", 0.30, "daily"),
    ("sakura",    "sakura_v3.gif",    "#160C12", "#FF8FB8", "#21121B", 0.32, "daily"),
    ("abyss",     "abyss_v3.jpg",     "#04101A", "#2FB8FF", "#0A1A27", 0.26, "daily"),
    ("dusk",      "dusk_v3.jpg",      "#150D14", "#FF9F43", "#20131C", 0.30, "daily"),
    ("terminal",  "terminal_v3.gif",  "#050B07", "#39FF88", "#0A140D", 0.34, "daily"),
    ("royal",     "royal_v3.jpg",     "#0C0A07", "#D4AF37", "#16130D", 0.28, "daily"),
    ("glacier",   "glacier_v3.jpg",   "#081018", "#9ED8FF", "#0F1A24", 0.28, "daily"),
    ("newyear",   "newyear_v3.gif",   "#070B1A", "#FFD45A", "#0E1528", 0.30, "event"),
    ("lunar",     "lunar_v3.gif",     "#170708", "#FF5A3C", "#230C0D", 0.30, "event"),
    ("valentine", "valentine_v3.gif", "#17070E", "#FF5C8A", "#230D16", 0.32, "event"),
    ("clover",    "clover_v3.jpg",    "#06130B", "#3DDC84", "#0C1D13", 0.28, "event"),
    ("easter",    "easter_v3.jpg",    "#13111B", "#C59BFF", "#1C1927", 0.30, "event"),
    ("halloween", "halloween_v3.gif", "#0F0814", "#FF8A1F", "#1A0F20", 0.30, "event"),
    ("christmas", "christmas_v3.gif", "#06110C", "#E8394A", "#0D1C15", 0.30, "event"),
]


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def hexrgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def rng(seed):
    return np.random.default_rng(seed)


def value_noise(w, h, cells, r, octaves=5, gain=0.5):
    """fBm made of upscaled random grids (bicubic): smooth, 0..1."""
    out = np.zeros((h, w), np.float32)
    amp, tot, c = 1.0, 0.0, cells
    for _ in range(octaves):
        gw, gh = max(2, int(c * w / max(w, h)) + 2), max(2, int(c * h / max(w, h)) + 2)
        g = r.random((gh, gw)).astype(np.float32)
        im = Image.fromarray((g * 255).astype(np.uint8), "L").resize((w, h), Image.BICUBIC)
        out += amp * (np.asarray(im, np.float32) / 255.0)
        tot += amp
        amp *= gain
        c *= 2
    out /= tot
    lo, hi = np.percentile(out, 1), np.percentile(out, 99)
    return np.clip((out - lo) / (hi - lo + 1e-6), 0, 1)


def tile_noise(w, h, cells, r, octaves=4, gain=0.5):
    """noise that wraps left-right (a loop can scroll it sideways without a seam)."""
    a = value_noise(w * 2, h, cells * 2, r, octaves, gain)
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    return a[:, :w] * (1 - x) + a[:, w:] * x


def grad_v(w, h, top, bottom, power=1.0):
    t = (np.linspace(0, 1, h, dtype=np.float32) ** power)[:, None, None]
    return np.broadcast_to(top * (1 - t) + bottom * t, (h, w, 3)).copy()


def grad_stops(w, h, stops):
    """vertical gradient through [(pos, hex), ...]."""
    t = np.linspace(0, 1, h, dtype=np.float32)
    out = np.zeros((h, 3), np.float32)
    for i in range(len(stops) - 1):
        p0, c0 = stops[i]
        p1, c1 = stops[i + 1]
        m = (t >= p0) & (t <= p1)
        k = ((t[m] - p0) / max(1e-6, p1 - p0))[:, None]
        k = k * k * (3 - 2 * k)
        out[m] = hexrgb(c0) * (1 - k) + hexrgb(c1) * k
    return np.broadcast_to(out[:, None, :], (h, w, 3)).copy()


def radial(w, h, cx, cy, rad):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((x - cx) / rad) ** 2 + ((y - cy) / rad) ** 2)
    return np.clip(1 - d, 0, 1)


def vignette(img, k=0.55):
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2) / math.sqrt(2)
    return img * (1 - k * d[..., None] ** 2)


def grain(img, r, amt=3.0):
    return img + r.normal(0, amt, img.shape[:2])[..., None]


def splat(canvas, x, y, sigma, col, a):
    """additive gaussian dot (float canvas)."""
    h, w = canvas.shape[:2]
    rad = int(sigma * 3) + 1
    x0, x1, y0, y1 = max(0, int(x) - rad), min(w, int(x) + rad + 2), max(0, int(y) - rad), min(h, int(y) + rad + 2)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma)) * a
    canvas[y0:y1, x0:x1] += g[..., None] * col


def disc(canvas, x, y, rad, col, a, soft=1.5):
    """a soft-edged filled disc blended over the canvas (bokeh)."""
    h, w = canvas.shape[:2]
    r = int(rad + soft * 2) + 2
    x0, x1, y0, y1 = max(0, int(x) - r), min(w, int(x) + r + 1), max(0, int(y) - r), min(h, int(y) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    m = np.clip((rad - d) / soft + 0.5, 0, 1) * a
    rim = np.clip(1 - np.abs(d - rad * 0.86) / (rad * 0.16 + 1e-3), 0, 1) * 0.35 * a  # a brighter rim, like a lens
    m = np.clip(m + rim, 0, 1)
    canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1 - m[..., None]) + col * m[..., None]


def to_img(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def to_arr(im):
    return np.asarray(im.convert("RGB"), np.float32)


def bloom(im, radius, amount=0.6, threshold=150):
    """the bright parts glow: a blurred copy of what is over the threshold, added on top."""
    a = to_arr(im)
    lum = a.mean(-1, keepdims=True)
    hi = np.clip(a * np.clip((lum - threshold) / (255 - threshold), 0, 1), 0, 255)
    bl = to_arr(to_img(hi).filter(ImageFilter.GaussianBlur(radius)))
    return to_img(a + bl * amount)


def stars(canvas, r, n, ymax=1.0, big=0.02, bright=(120, 255), tint=(1, 1, 1), pw=1.4, k=1.0):
    h, w = canvas.shape[:2]
    for _ in range(n):
        x, y = r.random() * w, (r.random() ** pw) * h * ymax
        b = r.uniform(*bright) * (r.random() ** 2.2)
        s = (0.55 + (1.4 if r.random() < big else 0.35) * r.random()) * k
        c = np.array(tint, np.float32) * np.array([0.85 + 0.15 * r.random(), 0.9, 1.0], np.float32)
        splat(canvas, x, y, s, c, b)


def down(im, w=FW, h=FH):
    return im.resize((w, h), Image.LANCZOS)


def heart_poly(cx, cy, s, rot=0.0, n=60):
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        x, y = x * s / 17, y * s / 17
        c, sn = math.cos(rot), math.sin(rot)
        pts.append((cx + x * c - y * sn, cy + x * sn + y * c))
    return pts


def ellipse_poly(cx, cy, rx, ry, rot=0.0, n=40):
    c, s = math.cos(rot), math.sin(rot)
    return [(cx + rx * math.cos(2 * math.pi * i / n) * c - ry * math.sin(2 * math.pi * i / n) * s,
             cy + rx * math.cos(2 * math.pi * i / n) * s + ry * math.sin(2 * math.pi * i / n) * c) for i in range(n)]


# ── the stills (drawn at W x H = 2048 x 1536) ────────────────────────────────────────────────────────────────────────────────
def carbon():
    r = rng(11)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    s = 14.0 * K
    cx, cy = np.floor(x / s), np.floor(y / s)
    fx, fy = (x / s) - cx, (y / s) - cy
    horiz = ((cx + cy) % 4) < 2                                    # 2 x 2 twill: diagonal steps of horizontal / vertical tows
    across = np.where(horiz, fy, fx)
    along = np.where(horiz, fx, fy)
    tow = np.sin(np.pi * across) ** 0.8
    fibre = 0.9 + 0.1 * np.sin(across * np.pi * 9 + along * 0.6)
    light = (x / W) * 0.6 + (1 - y / H) * 0.4
    lit = np.where(horiz, 0.45 + 0.55 * light, 0.45 + 0.55 * (1 - light))
    v = 11 + 34 * tow * fibre * lit
    img = np.stack([v, v, v * 1.06], -1)
    gold = hexrgb("#E8B84A")
    img += (radial(W, H, W * 0.92, H * 0.02, W * 0.75) ** 2.2)[..., None] * gold * 0.30
    img += (radial(W, H, W * 0.05, H * 1.0, W * 0.55) ** 2.5)[..., None] * gold * 0.10
    img = vignette(img, 0.5)
    img = grain(img, r, 1.5)
    return down(to_img(img))


def velvet():
    r = rng(23)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    n = value_noise(W, H, 3, r, 4) * 2 - 1
    n2 = value_noise(W, H, 2, r, 3) * 2 - 1
    th = math.radians(28)
    u = (x * math.cos(th) + y * math.sin(th)) / W
    u = u + 0.10 * n + 0.05 * n2
    hgt = 0.62 * np.sin(u * 2 * np.pi * 3.2) + 0.30 * np.sin(u * 2 * np.pi * 6.1 + 1.3 + 2 * n2) + 0.08 * np.sin(u * 2 * np.pi * 9.0 + 2 * n)
    hgt = np.asarray(Image.fromarray(((hgt + 1) * 127.5).astype(np.uint8), 'L').filter(ImageFilter.GaussianBlur(3 * K)), np.float32) / 127.5 - 1
    gy, gx = np.gradient(hgt * 40 * K)
    nz = np.ones_like(gx)
    norm = np.sqrt(gx * gx + gy * gy + nz)
    nx, ny, nz = -gx / norm, -gy / norm, nz / norm
    L = np.array([-0.45, -0.55, 0.70]); L /= np.linalg.norm(L)
    diff = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    hv = L + np.array([0, 0, 1.0]); hv /= np.linalg.norm(hv)
    spec = np.clip(nx * hv[0] + ny * hv[1] + nz * hv[2], 0, 1) ** 28
    img = hexrgb("#2E0812") * (0.35 + 0.9 * diff[..., None] ** 1.6) + hexrgb("#B0283A") * (spec[..., None] * 0.55)
    img = img * (0.75 + 0.45 * (radial(W, H, W * 0.15, H * 0.1, W * 0.8) ** 2)[..., None])
    img = vignette(img, 0.6)
    img = grain(img, r, 1.8)
    return down(to_img(img))


def nebula():
    r = rng(37)
    a = value_noise(W, H, 3, r, 7, 0.55)
    b = value_noise(W, H, 4, r, 7, 0.5)
    c = value_noise(W, H, 2, r, 5, 0.5)
    dens = np.clip((a - 0.35) / 0.65, 0, 1) ** 1.6
    img = hexrgb("#0A0716") + dens[..., None] * hexrgb("#5A2C9E") * 0.85
    img += (np.clip(b - 0.45, 0, 1) * dens)[..., None] * hexrgb("#B0347E") * 1.2
    img += (np.clip(c - 0.55, 0, 1) * (1 - dens * 0.5))[..., None] * hexrgb("#2848A8") * 0.9
    lane = np.clip(0.55 - value_noise(W, H, 6, r, 5), 0, 1) * 1.4
    img *= (1 - 0.6 * lane[..., None] * dens[..., None])
    st = np.zeros_like(img)
    stars(st, r, int(1400 * 2.2), 1.0, 0.015, (90, 255), k=K)
    img += st
    for _ in range(6):
        x, y = r.random() * W, r.random() * H
        splat(img, x, y, 2.2 * K, np.array([1.0, 0.9, 1.0]), 210)
        splat(img, x, y, 14 * K, np.array([0.7, 0.5, 1.0]), 22)
    img = vignette(img, 0.45)
    img = grain(img, r, 1.6)
    return down(bloom(to_img(img), 6 * K, 0.35, 170))


def midnight():
    r = rng(53)
    img = grad_v(W, H, hexrgb("#03060F"), hexrgb("#13284F"), 1.6)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    band = np.exp(-((y - (H * 0.95 - x * 0.55)) / (H * 0.16)) ** 2)
    img += (band * value_noise(W, H, 5, r, 6))[..., None] * hexrgb("#4A5F96") * 0.35
    st = np.zeros_like(img)
    stars(st, r, int(1700 * 2.2), 0.95, 0.02, (80, 255), (0.92, 0.96, 1.0), k=K)
    img += st * (1 - 0.6 * (y / H))[..., None]
    img += (radial(W, H, W * 0.62, H * 1.05, W * 0.75) ** 2)[..., None] * hexrgb("#3A6BD0") * 0.35

    def ridge(base_y, amp, seed, cells):
        return base_y + amp * (value_noise(W, 4, cells, rng(seed), 6)[0] * 2 - 1)
    far = ridge(H * 0.80, H * 0.07, 61, 6)
    near = ridge(H * 0.90, H * 0.06, 67, 4)
    far_mask = np.clip((y - far[None, :]) / 2.0, 0, 1)
    near_mask = np.clip((y - near[None, :]) / 2.0, 0, 1)
    img = img * (1 - far_mask[..., None]) + far_mask[..., None] * hexrgb("#0C1630")
    img = img * (1 - near_mask[..., None]) + near_mask[..., None] * hexrgb("#04070E")
    return down(bloom(to_img(grain(img, r, 1.2)), 4 * K, 0.3, 175))


def jade():
    r = rng(71)
    hgt = value_noise(W, H, 3, r, 5, 0.45)
    N = 26
    v = hgt * N * 0.97 + 0.37
    d = np.abs(v - np.round(v))
    gy, gx = np.gradient(v)
    gm = np.sqrt(gx * gx + gy * gy) + 1e-4
    px = d / gm
    line = np.clip(1.15 * K - px, 0, 1)
    idx = (np.round(v) % 5 == 0)
    a = np.where(idx, line * 0.75, line * 0.32)
    bg = grad_v(W, H, hexrgb("#06120E"), hexrgb("#040C09"))
    bg += (radial(W, H, W * 0.3, H * 0.35, W * 0.8) ** 2)[..., None] * hexrgb("#12382A") * 0.9
    img = bg * (1 - a[..., None]) + hexrgb("#30C48D") * a[..., None] * 0.9 + bg * a[..., None] * 0.1
    img = vignette(img, 0.45)
    return down(to_img(img))


def mono():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    img = 12 + 10 * ((x / W) * 0.5 + (1 - y / H) * 0.5)
    img = np.stack([img, img, img * 1.05], -1)
    s = 18.0 * K
    cx, cy = (np.floor(x / s) + 0.5) * s, (np.floor(y / s) + 0.5) * s
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    field = 0.25 + 0.75 * np.clip(1 - np.sqrt(((cx - W * 0.1) / W) ** 2 + ((cy - H * 0.05) / H) ** 2) / 1.1, 0, 1) ** 1.5
    rad = (0.5 + 3.4 * field) * K
    dot = np.clip(rad - d + 0.5, 0, 1)
    img += dot[..., None] * np.array([34, 34, 38], np.float32)
    img += (radial(W, H, W * 0.1, H * 0.0, W * 0.9) ** 2.5)[..., None] * np.array([40, 40, 46], np.float32)
    img = vignette(img, 0.4)
    return down(to_img(img))


def abyss():
    """the deep sea: light from the surface in rays and caustics, marine snow drifting, darkness below."""
    r = rng(211)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    img = grad_stops(W, H, [(0, "#0B4A66"), (0.28, "#06283E"), (0.65, "#03111F"), (1, "#010408")])
    n = value_noise(W, H, 5, r, 5)                                                  # caustics: bright thin lines where the noise crosses its middle
    n2 = value_noise(W, H, 9, r, 4)
    caus = np.clip(1 - np.abs(n - 0.5) * 9, 0, 1) ** 2 + 0.6 * np.clip(1 - np.abs(n2 - 0.5) * 11, 0, 1) ** 2
    caus *= np.clip(1 - y / (H * 0.42), 0, 1) ** 1.6
    img += caus[..., None] * hexrgb("#5FD6FF") * 0.32
    rays = np.zeros((H, W), np.float32)                                             # god rays: slanted beams from the surface, fading with depth
    for i in range(14):
        x0 = r.uniform(-0.1, 1.1) * W
        ang = r.uniform(-0.25, 0.05)
        wid = r.uniform(0.012, 0.05) * W
        xc = x0 + ang * y
        beam = np.exp(-((x - xc) / wid) ** 2) * np.clip(1 - y / (H * r.uniform(0.6, 1.0)), 0, 1) ** 1.4
        rays += beam * r.uniform(0.25, 0.7)
    img += rays[..., None] * hexrgb("#4CC4EE") * 0.30
    snow = np.zeros_like(img)
    stars(snow, r, 2600, 1.0, 0.05, (40, 160), (0.75, 0.92, 1.0), 1.0, k=K * 1.2)  # marine snow
    img += snow * (0.6 + 0.4 * (1 - y / H))[..., None]
    for _ in range(18):                                                             # a few soft out-of-focus motes in front
        disc(img, r.random() * W, r.random() * H, r.uniform(8, 26) * K, hexrgb("#7FD8F5"), r.uniform(0.04, 0.10), 3 * K)
    img = vignette(img, 0.5)
    img = grain(img, r, 1.4)
    return down(bloom(to_img(img), 8 * K, 0.35, 150))


def dusk():
    """a sunset over layered hills: the sky from violet to amber, a low sun, every ridge a little bluer with distance."""
    r = rng(223)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    img = grad_stops(W, H, [(0, "#1B0E2E"), (0.30, "#4A1D4F"), (0.52, "#B2405A"), (0.66, "#F07D4A"), (0.74, "#FFC27A"), (1, "#FFC27A")])
    sx, sy = W * 0.64, H * 0.66
    img += (radial(W, H, sx, sy, W * 0.55) ** 2.4)[..., None] * hexrgb("#FFB46A") * 0.55
    sun = np.clip((W * 0.055 - np.sqrt((x - sx) ** 2 + (y - sy) ** 2)) / (2 * K), 0, 1)
    img = img * (1 - sun[..., None]) + hexrgb("#FFF1D0") * sun[..., None]
    clouds = value_noise(W, H, 4, r, 6)
    band = np.exp(-((y - H * 0.45) / (H * 0.09)) ** 2) + 0.6 * np.exp(-((y - H * 0.25) / (H * 0.06)) ** 2)
    cl = np.clip((clouds - 0.55) * 3, 0, 1) * band
    img = img * (1 - 0.35 * cl[..., None]) + hexrgb("#FF8A6A") * (0.25 * cl[..., None])
    st = np.zeros_like(img)
    stars(st, r, 900, 0.30, 0.02, (60, 200), k=K)
    img += st * np.clip(1 - y / (H * 0.3), 0, 1)[..., None]
    layers = [(0.70, 0.050, "#7A3A5C", 5), (0.76, 0.060, "#55284A", 4), (0.83, 0.070, "#341A35", 3), (0.91, 0.060, "#1A0E1C", 3)]
    for i, (by, amp, col, cells) in enumerate(layers):
        ridge = by * H + amp * H * (value_noise(W, 4, cells, rng(300 + i), 6)[0] * 2 - 1)
        m = np.clip((y - ridge[None, :]) / 2.0, 0, 1)
        haze = np.clip((y - ridge[None, :]) / (H * 0.25), 0, 1)
        layer = hexrgb(col) * (1 - 0.35 * haze[..., None])
        img = img * (1 - m[..., None]) + layer * m[..., None]
    img = vignette(img, 0.35)
    img = grain(img, r, 1.3)
    return down(bloom(to_img(img), 10 * K, 0.30, 200))


def royal():
    """black and gold art deco: rows of fanned scallops in fine gold lines, a brushed metal sheen."""
    r = rng(229)
    big = Image.new("RGB", (W, H), (11, 9, 7))
    dr = ImageDraw.Draw(big)
    R = 64 * K
    rows = int(H / (R * 0.5)) + 3
    cols = int(W / R) + 3
    for j in range(rows):
        for i in range(cols):
            cx = i * R + (R / 2 if j % 2 else 0)
            cy = j * R * 0.5
            for k in range(6, 0, -1):                                               # nested arcs: the fan
                rr = R * 0.5 * k / 6
                dr.pieslice([cx - rr, cy - rr, cx + rr, cy + rr], 180, 360, fill=(11, 9, 7))
                dr.arc([cx - rr, cy - rr, cx + rr, cy + rr], 180, 360, fill=(212, 175, 55), width=max(1, int(1.1 * K)))
    a = to_arr(big)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    sheen = 0.35 + 0.65 * (radial(W, H, W * 0.75, H * 0.15, W * 0.9) ** 1.6)
    brush = 0.88 + 0.12 * value_noise(W, H, 40, r, 2)
    gold = (a.mean(-1, keepdims=True) > 30).astype(np.float32)
    base = np.array([11, 9, 7], np.float32) + (radial(W, H, W * 0.5, H * 0.4, W * 0.8) ** 2)[..., None] * np.array([20, 15, 6], np.float32)
    img = base * (1 - gold) + a * gold * (sheen * brush)[..., None]
    img = vignette(img, 0.55)
    img = grain(img, r, 1.2)
    return down(bloom(to_img(img), 3 * K, 0.35, 120))


def glacier():
    """ice: crystal facets (a Voronoi field) lit from the top left, frost sparkle, cold blue light."""
    r = rng(233)
    sw, sh = W // 4, H // 4                                                         # the facets are computed small and upscaled (smooth edges)
    pts = np.stack([r.random(90) * sw, r.random(90) * sh], 1).astype(np.float32)
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    d = np.sqrt((xx[..., None] - pts[:, 0]) ** 2 + (yy[..., None] - pts[:, 1]) ** 2)
    order = np.argsort(d, -1)
    d1 = np.take_along_axis(d, order[..., :1], -1)[..., 0]
    d2 = np.take_along_axis(d, order[..., 1:2], -1)[..., 0]
    cell = order[..., 0]
    edge = np.clip(1 - (d2 - d1) / 1.2, 0, 1)
    tone = r.random(90).astype(np.float32)[cell]
    ang = (r.random(90) * 2 * np.pi).astype(np.float32)[cell]
    shade = 0.45 + 0.35 * tone + 0.2 * np.cos(ang - 2.4)
    small = np.stack([shade, edge], -1)
    up = lambda a: np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "L").resize((W, H), Image.BICUBIC), np.float32) / 255
    shade_b, edge_b = up(small[..., 0]), up(small[..., 1])
    img = grad_stops(W, H, [(0, "#1E4A6A"), (0.5, "#0E2539"), (1, "#060E17")])
    img = img * (0.55 + 0.6 * shade_b[..., None])
    img += (edge_b ** 2)[..., None] * hexrgb("#BFE8FF") * 0.55
    img += (radial(W, H, W * 0.15, 0, W * 0.9) ** 2)[..., None] * hexrgb("#9ED8FF") * 0.35
    sp = np.zeros_like(img)
    stars(sp, r, 700, 1.0, 0.08, (80, 255), (0.9, 0.97, 1.0), 1.0, k=K)
    img += sp
    img = vignette(img, 0.45)
    img = grain(img, r, 1.0)
    return down(bloom(to_img(img), 5 * K, 0.40, 180))


def clover():
    """St Patrick's: shamrocks of every size on deep green, gold rims, warm gold sparks."""
    r = rng(241)
    img = grad_stops(W, H, [(0, "#0F3A22"), (0.6, "#08200F"), (1, "#03100A")])
    img += (radial(W, H, W * 0.5, H * 0.45, W * 0.75) ** 2)[..., None] * hexrgb("#1F6B3A") * 0.45
    base = to_img(img)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(lay)
    rr = random.Random(5)
    placed = 0
    while placed < 34:
        cx, cy, s, rot = rr.random() * W, rr.random() * H, rr.uniform(22, 96) * K, rr.uniform(0, 2 * math.pi)
        if abs(cx - W / 2) < W * 0.28 and abs(cy - H / 2) < H * 0.26 and rr.random() < 0.8:    # the middle stays calm (the window's text sits there)
            continue
        placed += 1
        depth = rr.random()
        g = (int(40 + 60 * depth), int(150 + 80 * depth), int(70 + 40 * depth), int(60 + 150 * depth))
        rim = (232, 200, 90, int(80 + 140 * depth))
        for k in range(3):                                                          # three heart leaves around the middle
            a = rot + k * 2 * math.pi / 3
            hx, hy = cx + math.cos(a) * s * 0.48, cy + math.sin(a) * s * 0.48
            dr.polygon(heart_poly(hx, hy, s * 0.55, a + math.pi / 2), fill=g, outline=rim)
        stem = [(cx, cy), (cx + math.cos(rot + 2.6) * s * 0.9, cy + math.sin(rot + 2.6) * s * 0.9)]
        dr.line(stem, fill=g, width=max(2, int(s * 0.08)))
    lay = lay.filter(ImageFilter.GaussianBlur(0.6 * K))
    base.paste(lay, (0, 0), lay)
    a = to_arr(base)
    sp = np.zeros_like(a)
    stars(sp, rng(9), 500, 1.0, 0.1, (60, 220), (1.0, 0.85, 0.45), 1.0, k=K * 1.3)
    a = vignette(a + sp, 0.5)
    return down(bloom(to_img(grain(a, r, 1.2)), 6 * K, 0.35, 150))


def easter():
    """spring: a soft lilac-to-mint sky, rolling meadow, decorated eggs in the grass, petals of light."""
    r = rng(251)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    img = grad_stops(W, H, [(0, "#2A2346"), (0.45, "#5A4C86"), (0.70, "#7FA6A0"), (1, "#2D4A3A")])
    img += (radial(W, H, W * 0.25, H * 0.2, W * 0.6) ** 2)[..., None] * hexrgb("#FFE6F2") * 0.25
    for i, (by, amp, col) in enumerate([(0.70, 0.04, "#4F7F5E"), (0.80, 0.05, "#3B6A4B"), (0.90, 0.04, "#2A5139")]):
        ridge = by * H + amp * H * (value_noise(W, 4, 3, rng(400 + i), 5)[0] * 2 - 1)
        m = np.clip((y - ridge[None, :]) / 2.0, 0, 1)
        img = img * (1 - m[..., None]) + hexrgb(col) * m[..., None]
    base = to_img(img)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(lay)
    rr = random.Random(17)
    pastel = [(255, 182, 213), (190, 160, 255), (160, 220, 255), (255, 226, 150), (170, 235, 190), (255, 196, 160)]
    for _ in range(16):                                                             # eggs, front ones bigger
        depth = rr.random()
        ex = rr.random() * W
        ry = (34 + 50 * depth) * K
        rx = ry * 0.74
        ey = H * (0.74 + 0.12 * depth) - ry * 0.2
        tilt = rr.uniform(-0.35, 0.35)
        col = rr.choice(pastel)
        dr.polygon(ellipse_poly(ex, ey, rx, ry, tilt), fill=col + (255,))
        c2 = rr.choice([p for p in pastel if p != col])
        for band in (-0.35, 0.0, 0.35):                                             # stripes and dots
            yb = ey + band * ry
            half = rx * math.sqrt(max(0.05, 1 - band * band))
            dr.line([(ex - half * 0.95, yb), (ex + half * 0.95, yb)], fill=c2 + (255,), width=max(2, int(ry * 0.10)))
        for k in range(5):
            dx = (k - 2) * rx * 0.35
            dr.ellipse([ex + dx - rx * 0.07, ey + ry * 0.18 - rx * 0.07, ex + dx + rx * 0.07, ey + ry * 0.18 + rx * 0.07], fill=(255, 255, 255, 220))
        dr.polygon(ellipse_poly(ex - rx * 0.32, ey - ry * 0.42, rx * 0.18, ry * 0.12, tilt), fill=(255, 255, 255, 90))   # a highlight
    lay = lay.filter(ImageFilter.GaussianBlur(0.7 * K))
    base.paste(lay, (0, 0), lay)
    a = to_arr(base)
    for _ in range(40):
        disc(a, r.random() * W, r.random() * H * 0.62, r.uniform(5, 16) * K, hexrgb(rr.choice(["#FFD6EA", "#E2D4FF", "#D8F6FF"])), r.uniform(0.03, 0.07), 4 * K)
    a = vignette(a, 0.35)
    return down(bloom(to_img(grain(a, r, 1.1)), 8 * K, 0.25, 190))


# ── the animated looks (frames drawn at 512 x 384, reduced to 256 x 192; t runs 0..1 over one loop) ───────────────────────────────
AW, AH = GW * GS, GH * GS
AK = AW / 160.0                     # the v2 GIFs were drawn at 160 px


def gif_frames_ember():
    r = random.Random(97)
    parts = [dict(x=r.random() * AW, ph=r.random(), k=r.choice([1, 1, 2]), trav=r.uniform(0.35, 1.0), sw=r.uniform(2, 7) * AK / 2,
                  wph=r.random(), sig=r.uniform(0.45, 1.25) * AK / 2, br=r.uniform(120, 260)) for _ in range(150)]
    frames = []
    for f in range(GN):
        t = f / GN
        img = grad_v(AW, AH, hexrgb("#0A0605"), hexrgb("#2C0B05"), 1.4)
        flick = 1 + 0.10 * math.sin(2 * math.pi * t) + 0.05 * math.sin(2 * math.pi * 3 * t + 1)
        img += (radial(AW, AH, AW * 0.5, AH * 1.15, AW * 0.8) ** 1.8)[..., None] * hexrgb("#8A2208") * 0.65 * flick
        img += (radial(AW, AH, AW * 0.2, AH * 1.1, AW * 0.5) ** 2)[..., None] * hexrgb("#A0400A") * 0.4 * (2 - flick)
        for p in parts:
            life = (t * p["k"] + p["ph"]) % 1
            y = AH * (1.05 - life * p["trav"] * 1.1)
            x = p["x"] + p["sw"] * math.sin(2 * math.pi * (life + p["wph"]))
            a = math.sin(math.pi * life) ** 1.5
            col = np.array([1.0, 0.75 - 0.4 * life, 0.35 - 0.25 * life], np.float32)
            splat(img, x, y, p["sig"], col, p["br"] * a)
            splat(img, x, y, p["sig"] * 3.2, col, p["br"] * a * 0.12)
        frames.append(down(bloom(to_img(img), 3 * AK / 2, 0.4, 150), GW, GH))
    return frames


def gif_frames_synthwave():
    S = 2
    w, h = AW * S, AH * S
    hz = int(h * 0.60)
    r = random.Random(5)
    starl = [(r.random() * w, r.random() * hz * 0.8, r.random(), r.uniform(0.4, 1)) for _ in range(70)]
    frames = []
    for f in range(GN):
        t = f / GN
        img = grad_v(w, h, hexrgb("#0E0420"), hexrgb("#4A0E4E"), 1.3)
        img[hz:] = grad_v(w, h - hz, hexrgb("#16061F"), hexrgb("#07020C"))
        for (sx, sy, ph, b) in starl:
            a = 0.55 + 0.45 * math.sin(2 * math.pi * (t * 2 + ph))
            splat(img, sx, sy, 1.6 * S, np.array([1, 0.85, 1.0]), 220 * a * b)
        cx, cy, rad = w * 0.5, hz - h * 0.11, h * 0.26
        yy, xx = np.mgrid[0:hz, 0:w].astype(np.float32)
        inside = ((xx - cx) ** 2 + (yy - cy) ** 2) <= rad * rad
        tt = np.clip((yy - (cy - rad)) / (2 * rad), 0, 1)
        sun = hexrgb("#D9A040")[None, None, :] * (1 - tt[..., None]) + hexrgb("#C8206E")[None, None, :] * tt[..., None]
        period = h * 0.045
        rel = (yy - cy + t * period) % period
        gap = (yy > cy - rad * 0.25) & (rel < period * (0.10 + 0.60 * np.clip((yy - cy + rad * 0.25) / rad, 0, 1)))
        m = inside & ~gap
        img[:hz][m] = sun[m]
        img += (radial(w, h, cx, cy, rad * 2.2) ** 2.4)[..., None] * hexrgb("#FF3A8C") * 0.22
        im = to_img(img)
        dr = ImageDraw.Draw(im)
        line = (255, 64, 190)
        for k in range(0, 40):
            z = 1.0 + (k - t) * 0.6
            if z <= 0.15:
                continue
            yv = hz + (h - hz) * (0.9 / z) * 0.55
            if yv >= h:
                continue
            fade = min(1, (yv - hz) / (h * 0.05))
            c = tuple(int(v * (0.35 + 0.65 * fade)) for v in line)
            dr.line([(0, yv), (w, yv)], fill=c, width=max(1, int(S * 2 * (0.5 + fade))))
        for k in range(-16, 17):
            dr.line([(w * 0.5 + k * w * 0.012, hz), (w * 0.5 + k * w * 0.11, h)], fill=line, width=max(1, int(S * 2)))
        dr.line([(0, hz), (w, hz)], fill=(255, 140, 220), width=S * 3)
        im = Image.blend(im, im.filter(ImageFilter.GaussianBlur(S * 5)), 0.35)
        frames.append(down(im, GW, GH))
    return frames


def gif_frames_aurora():
    w, h = AW, AH
    r = rng(131)
    st = np.zeros((h, w, 3), np.float32)
    stars(st, r, 700, 0.92, 0.02, (60, 230), (0.9, 0.95, 1.0), 1.0, k=AK / 2)
    xs = np.arange(w, dtype=np.float32) / w
    yy = np.arange(h, dtype=np.float32)[:, None]
    ray_seed = r.random(64).astype(np.float32)
    trees, rr, x = [], random.Random(9), 0
    while x < w:
        th = rr.uniform(0.08, 0.20) * h
        tw = th * rr.uniform(0.35, 0.5)
        trees.append((x, th, tw))
        x += tw * rr.uniform(0.45, 0.9)
    frames = []
    for f in range(GN):
        t = f / GN
        img = grad_v(w, h, hexrgb("#02060C"), hexrgb("#0A1B26"), 1.2) + st
        for (base, amp, cyc, spd, colb, colt, strength) in ((0.52, 0.08, 1.2, 1, "#3CF0B4", "#7A5CFF", 1.0), (0.36, 0.06, 2.1, -1, "#46D8F0", "#C060FF", 0.55)):
            yc = h * (base + amp * np.sin(2 * np.pi * (xs * cyc + spd * t)) + 0.04 * np.sin(2 * np.pi * (xs * 3.3 - 2 * spd * t) + 1.1))
            dy = yc[None, :] - yy
            above = np.exp(-np.clip(dy, 0, None) / (h * 0.11)) * (dy >= 0)
            below = np.exp(-(np.clip(-dy, 0, None) / (h * 0.025)) ** 2) * (dy < 0)
            idx = (xs * 64).astype(int) % 64
            rays = 0.55 + 0.45 * np.sin(2 * np.pi * (xs * 23 + ray_seed[idx] + t * 2 * spd))
            inten = (above + below) * rays[None, :] * strength
            mix = np.clip(dy / (h * 0.16), 0, 1)[..., None]
            col = hexrgb(colb)[None, None, :] * (1 - mix) + hexrgb(colt)[None, None, :] * mix
            img += inten[..., None] * col * 0.75
        im = to_img(img).filter(ImageFilter.GaussianBlur(1.6))
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, h * 0.94, w, h], fill=(2, 5, 8))
        for (tx, th, tw) in trees:
            dr.polygon([(tx - tw / 2, h * 0.95), (tx + tw / 2, h * 0.95), (tx, h * 0.95 - th)], fill=(2, 5, 8))
        frames.append(down(bloom(im, 4, 0.3, 160), GW, GH))
    return frames


def gif_frames_sakura():
    """dusk over far mountains, a blossoming branch in the corner, petals falling and turning."""
    w, h = AW, AH
    r = rng(271)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    sky = grad_stops(w, h, [(0, "#2A1638"), (0.45, "#7A3A66"), (0.75, "#E58BA8"), (1, "#F6C4C8")])
    sky += (radial(w, h, w * 0.7, h * 0.72, w * 0.5) ** 2)[..., None] * hexrgb("#FFE0C8") * 0.35
    ridge = h * 0.78 + h * 0.06 * (value_noise(w, 4, 4, rng(5), 5)[0] * 2 - 1)
    m = np.clip((y - ridge[None, :]) / 1.5, 0, 1)
    sky = sky * (1 - m[..., None]) + hexrgb("#4A2440") * m[..., None]
    ridge2 = h * 0.88 + h * 0.04 * (value_noise(w, 4, 3, rng(6), 5)[0] * 2 - 1)
    m2 = np.clip((y - ridge2[None, :]) / 1.5, 0, 1)
    sky = sky * (1 - m2[..., None]) + hexrgb("#2A1428") * m2[..., None]
    base = to_img(sky)
    br = Image.new("RGBA", (w, h), (0, 0, 0, 0))                                   # the branch (top left) with blossoms
    d = ImageDraw.Draw(br)
    rr = random.Random(3)

    def branch(x0, y0, ang, ln, wd, depth):
        x1, y1 = x0 + math.cos(ang) * ln, y0 + math.sin(ang) * ln
        d.line([(x0, y0), (x1, y1)], fill=(36, 16, 24, 255), width=max(1, int(wd)))
        if depth > 0:
            for da in (-0.5, 0.35):
                branch(x1, y1, ang + da + rr.uniform(-0.2, 0.2), ln * rr.uniform(0.6, 0.8), wd * 0.65, depth - 1)
        for _ in range(3 if depth < 2 else 1):
            bx, by = x0 + (x1 - x0) * rr.random(), y0 + (y1 - y0) * rr.random()
            rad = rr.uniform(4, 8)
            d.ellipse([bx - rad, by - rad, bx + rad, by + rad], fill=(255, rr.randint(170, 205), rr.randint(200, 225), 235))
            d.ellipse([bx - rad * 0.35, by - rad * 0.35, bx + rad * 0.35, by + rad * 0.35], fill=(255, 235, 240, 255))
    branch(-20, 40, 0.25, 150, 16, 4)
    petals = [dict(x=r.random() * w, ph=r.random(), k=int(r.integers(1, 3)), sw=r.uniform(8, 26), size=r.uniform(3.5, 7.5), spin=float(r.choice([-1, 1])) * int(r.integers(1, 3)),
                   wph=r.random(), drift=r.uniform(-0.4, 0.4)) for _ in range(55)]
    frames = []
    for f in range(GN):
        t = f / GN
        im = base.copy()
        im.paste(br, (0, 0), br)
        lay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        dp = ImageDraw.Draw(lay)
        for p in petals:
            life = (t * p["k"] + p["ph"]) % 1
            py = -20 + (h + 40) * life
            px = (p["x"] + p["drift"] * w * life + p["sw"] * math.sin(2 * math.pi * (life * 2 + p["wph"]))) % w
            rot = 2 * math.pi * (life * p["spin"] + p["ph"])
            flat = abs(math.cos(rot * 1.0)) * 0.7 + 0.3
            dp.polygon(ellipse_poly(px, py, p["size"], p["size"] * 0.55 * flat, rot, 16), fill=(255, 190, 210, 220))
        lay = lay.filter(ImageFilter.GaussianBlur(0.5))
        im.paste(lay, (0, 0), lay)
        frames.append(down(bloom(im, 4, 0.25, 200), GW, GH))
    return frames


def gif_frames_terminal():
    """green code rain: columns of glyphs fall, each head bright and its trail fading."""
    w, h = AW, AH
    cell = 12
    cols, rows = w // cell, h // cell + 2
    r = random.Random(11)

    def glyph(seed):
        g = random.Random(seed)
        return [((g.randint(0, 2), g.randint(0, 4)), (g.randint(0, 2), g.randint(0, 4))) for _ in range(g.randint(2, 4))]
    glyphs = [glyph(i) for i in range(64)]
    streams = [dict(col=c, ph=r.random(), k=r.choice([1, 1, 2]), ln=r.randint(6, 18), seed=r.randint(0, 999)) for c in range(cols) for _ in range(r.choice([1, 1, 2]))]
    frames = []
    for f in range(GN):
        t = f / GN
        im = Image.new("RGB", (w, h), (2, 8, 4))
        d = ImageDraw.Draw(im)
        for s in streams:
            head = ((t * s["k"] + s["ph"]) % 1) * (rows + s["ln"]) - 1
            for i in range(s["ln"]):
                ry = int(head) - i
                if ry < 0 or ry >= rows:
                    continue
                fade = 1 - i / s["ln"]
                gx, gy = s["col"] * cell + 2, ry * cell
                g = glyphs[(s["seed"] + ry * 7 + (f // 4 if i == 0 else 0)) % 64]
                col = (int(180 + 75 * fade), 255, int(180 + 75 * fade)) if i == 0 else (int(20 * fade), int(60 + 170 * fade), int(40 + 60 * fade))
                for (a, b) in g:
                    d.line([(gx + a[0] * 4, gy + a[1] * 2.5), (gx + b[0] * 4, gy + b[1] * 2.5)], fill=col, width=2)
        im = bloom(im, 4, 0.55, 90)
        a = vignette(to_arr(im), 0.5)
        frames.append(down(to_img(a), GW, GH))
    return frames


def gif_frames_newyear():
    """fireworks over a city skyline: bursts in gold, white and blue at staggered times, sparks falling with gravity."""
    w, h = AW, AH
    r = random.Random(19)
    sky = grad_stops(w, h, [(0, "#02040C"), (0.6, "#0A1430"), (1, "#16234A")])
    st = np.zeros_like(sky)
    stars(st, rng(7), 400, 0.7, 0.02, (60, 200), k=1.2)
    sky += st
    base = to_img(sky)
    city = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dc = ImageDraw.Draw(city)
    x = 0
    lights = []
    while x < w:
        bw = r.uniform(18, 46)
        bh = r.uniform(0.10, 0.34) * h
        dc.rectangle([x, h - bh, x + bw, h], fill=(4, 6, 14, 255))
        for wy in range(int(h - bh + 6), h - 4, 7):
            for wx in range(int(x + 4), int(x + bw - 4), 6):
                if r.random() < 0.28:
                    lights.append((wx, wy))
        x += bw + r.uniform(0, 6)
    for (wx, wy) in lights:
        dc.rectangle([wx, wy, wx + 2, wy + 3], fill=(255, 214, 120, 200))
    bursts = []
    palette = [(255, 212, 90), (255, 245, 220), (120, 180, 255), (255, 120, 160), (180, 255, 210)]
    for i in range(7):
        bursts.append(dict(x=r.uniform(0.12, 0.88) * w, y=r.uniform(0.12, 0.45) * h, t0=i / 7 + r.uniform(-0.03, 0.03), col=r.choice(palette),
                           n=r.randint(36, 60), sp=r.uniform(0.18, 0.28) * h, seed=r.randint(0, 999)))
    frames = []
    for f in range(GN):
        t = f / GN
        img = to_arr(base)
        for b in bursts:
            age = (t - b["t0"]) % 1
            if age > 0.55:
                continue
            k = age / 0.55
            rr = random.Random(b["seed"])
            fade = (1 - k) ** 1.3
            for _ in range(b["n"]):
                ang = rr.random() * 2 * math.pi
                spd = b["sp"] * (0.6 + 0.4 * rr.random())
                ease = 1 - (1 - k) ** 2.2
                px = b["x"] + math.cos(ang) * spd * ease
                py = b["y"] + math.sin(ang) * spd * ease + 0.12 * h * k * k
                c = np.array(b["col"], np.float32) / 255
                splat(img, px, py, 1.4, c, 255 * fade)
                splat(img, px, py, 5.0, c, 60 * fade)
            if k < 0.15:
                splat(img, b["x"], b["y"], 18, np.array(b["col"], np.float32) / 255, 160 * (1 - k / 0.15))
        im = to_img(img)
        im.paste(city, (0, 0), city)
        frames.append(down(bloom(im, 5, 0.55, 140), GW, GH))
    return frames


def gif_frames_lunar():
    """Lunar New Year: red lanterns swaying on a string, warm light, gold sparks drifting up."""
    w, h = AW, AH
    bg = grad_stops(w, h, [(0, "#2A0505"), (0.6, "#4A0B0A"), (1, "#1A0303")])
    bg += (radial(w, h, w * 0.5, h * 0.35, w * 0.7) ** 2)[..., None] * hexrgb("#B02010") * 0.45
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = (x + y) / 26.0, (x - y) / 26.0                                                  # a faint diamond lattice in thin lines
    lat = np.clip(1 - np.minimum(np.abs(u - np.round(u)), np.abs(v - np.round(v))) * 18, 0, 1)
    bg += lat[..., None] * hexrgb("#FFB040") * 0.07
    base = to_img(bg)
    r = random.Random(23)
    lanterns = [dict(x=w * (0.10 + 0.80 * i / 4), y=h * r.uniform(0.36, 0.56), s=r.uniform(40, 58), ph=r.random()) for i in range(5)]
    sparks = [dict(x=r.random() * w, ph=r.random(), k=r.choice([1, 2]), sw=r.uniform(3, 10)) for _ in range(60)]
    frames = []
    for f in range(GN):
        t = f / GN
        im = base.copy()
        lay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        d.line([(0, h * 0.18), (w, h * 0.16)], fill=(70, 20, 15, 255), width=3)
        glow = np.zeros((h, w, 3), np.float32)
        for L in lanterns:
            sway = 0.12 * math.sin(2 * math.pi * (t + L["ph"]))
            top = (L["x"], h * 0.17)
            cx = L["x"] + math.sin(sway) * (L["y"] - top[1])
            cy = top[1] + math.cos(sway) * (L["y"] - top[1])
            d.line([top, (cx, cy - L["s"] * 0.6)], fill=(90, 30, 20, 255), width=2)
            d.polygon(ellipse_poly(cx, cy, L["s"] * 0.62, L["s"] * 0.5, sway), fill=(214, 40, 30, 255))
            for k in (-0.5, 0, 0.5):
                d.line([(cx + k * L["s"] * 0.45, cy - L["s"] * 0.45), (cx + k * L["s"] * 0.45, cy + L["s"] * 0.45)], fill=(150, 20, 15, 255), width=2)
            d.rectangle([cx - L["s"] * 0.22, cy - L["s"] * 0.58, cx + L["s"] * 0.22, cy - L["s"] * 0.46], fill=(240, 180, 60, 255))
            d.rectangle([cx - L["s"] * 0.22, cy + L["s"] * 0.46, cx + L["s"] * 0.22, cy + L["s"] * 0.58], fill=(240, 180, 60, 255))
            d.line([(cx, cy + L["s"] * 0.58), (cx, cy + L["s"] * 0.95)], fill=(240, 180, 60, 255), width=2)
            splat(glow, cx, cy, L["s"] * 0.9, np.array([1.0, 0.45, 0.15]), 120 + 20 * math.sin(2 * math.pi * (2 * t + L["ph"])))
        a = to_arr(im) + glow
        for s in sparks:
            life = (t * s["k"] + s["ph"]) % 1
            sy = h * (1.02 - life * 1.05)
            sx = s["x"] + s["sw"] * math.sin(2 * math.pi * (life * 2 + s["ph"]))
            splat(a, sx, sy, 1.2, np.array([1.0, 0.8, 0.35]), 230 * math.sin(math.pi * life))
        im = to_img(a)
        im.paste(lay, (0, 0), lay)
        frames.append(down(bloom(im, 5, 0.45, 150), GW, GH))
    return frames


def gif_frames_valentine():
    """hearts rising through a rose glow: soft out-of-focus ones behind, a few sharp ones in front."""
    w, h = AW, AH
    bg = grad_stops(w, h, [(0, "#2A0614"), (0.55, "#52102A"), (1, "#1C0410")])
    bg += (radial(w, h, w * 0.5, h * 0.55, w * 0.65) ** 2)[..., None] * hexrgb("#FF4F7E") * 0.30
    base = to_img(bg)
    r = random.Random(29)
    hearts = [dict(x=r.random() * w, ph=r.random(), k=r.choice([1, 1, 2]), s=r.uniform(6, 22), sw=r.uniform(6, 20), z=r.random(), rot=r.uniform(-0.4, 0.4)) for _ in range(34)]
    frames = []
    for f in range(GN):
        t = f / GN
        im = base.copy()
        back = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        front = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        db, df = ImageDraw.Draw(back), ImageDraw.Draw(front)
        for hh in hearts:
            life = (t * hh["k"] + hh["ph"]) % 1
            y = h * 1.1 - (h * 1.25) * life
            x = hh["x"] + hh["sw"] * math.sin(2 * math.pi * (life * 2 + hh["ph"]))
            a = int(255 * math.sin(math.pi * life) ** 0.6)
            if hh["z"] < 0.6:
                db.polygon(heart_poly(x, y, hh["s"] * 1.6, hh["rot"]), fill=(255, 110, 150, int(a * 0.55)))
            else:
                df.polygon(heart_poly(x, y, hh["s"], hh["rot"] + 0.2 * math.sin(2 * math.pi * life)), fill=(255, 70, 120, a))
        back = back.filter(ImageFilter.GaussianBlur(5))
        front = front.filter(ImageFilter.GaussianBlur(0.6))
        im.paste(back, (0, 0), back)
        im.paste(front, (0, 0), front)
        frames.append(down(bloom(im, 5, 0.35, 170), GW, GH))
    return frames


def gif_frames_halloween():
    """a big moon behind a bare tree, bats crossing it, pumpkins glowing and flickering, mist drifting."""
    w, h = AW, AH
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    sky = grad_stops(w, h, [(0, "#0A0414"), (0.55, "#2A0F3A"), (1, "#140818")])
    mx, my, mr = w * 0.68, h * 0.34, h * 0.20
    sky += (radial(w, h, mx, my, mr * 3.2) ** 2.2)[..., None] * hexrgb("#FFB86A") * 0.35
    moon = np.clip((mr - np.sqrt((x - mx) ** 2 + (y - my) ** 2)) / 1.5, 0, 1)
    crat = value_noise(w, h, 9, rng(3), 3)
    sky = sky * (1 - moon[..., None]) + (hexrgb("#FFE2A8") * (0.85 + 0.15 * crat[..., None])) * moon[..., None]
    st = np.zeros_like(sky)
    stars(st, rng(5), 260, 0.6, 0.02, (60, 200), k=1.1)
    sky += st
    base = to_img(sky)
    tree = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(tree)
    rr = random.Random(41)

    def limb(x0, y0, ang, ln, wd, depth):
        x1, y1 = x0 + math.cos(ang) * ln, y0 + math.sin(ang) * ln
        d.line([(x0, y0), (x1, y1)], fill=(8, 4, 10, 255), width=max(1, int(wd)))
        if depth > 0:
            for da in (-0.55, 0.45):
                limb(x1, y1, ang + da + rr.uniform(-0.25, 0.25), ln * rr.uniform(0.62, 0.8), wd * 0.66, depth - 1)
    limb(w * 0.16, h * 0.98, -math.pi / 2 - 0.08, h * 0.30, 12, 6)
    d.polygon([(0, h), (0, h * 0.90), (w * 0.3, h * 0.93), (w * 0.6, h * 0.89), (w, h * 0.93), (w, h)], fill=(8, 4, 10, 255))
    pumpkins = [(w * 0.38, h * 0.90, 26, 0.1), (w * 0.52, h * 0.93, 18, 0.5), (w * 0.84, h * 0.91, 24, 0.8)]
    bats = [dict(ph=i / 5 + 0.03 * i, y=h * (0.14 + 0.09 * i), s=13 + 3 * (i % 3), dir=(1 if i % 2 == 0 else -1), k=1 + (i % 2)) for i in range(5)]
    mist_n = tile_noise(w, h // 3, 3, rng(9), 4)
    frames = []
    for f in range(GN):
        t = f / GN
        im = base.copy()
        im.paste(tree, (0, 0), tree)
        a = to_arr(im)
        shift = int(t * w) % w                                                      # mist drifting sideways (wraps: no seam)
        mist = np.roll(mist_n, shift, axis=1)
        mm = np.zeros((h, w), np.float32)
        ramp = np.clip(np.linspace(0, 1.4, h // 3, dtype=np.float32), 0, 1)[:, None] ** 1.5
        mm[h - h // 3:] = np.clip(mist - 0.25, 0, 1)[: h // 3] * 0.6 * ramp
        a = a * (1 - mm[..., None]) + hexrgb("#7A5A9A") * mm[..., None]
        lay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        dl = ImageDraw.Draw(lay)
        for (px, py, ps, ph) in pumpkins:
            flick = 0.75 + 0.25 * math.sin(2 * math.pi * (3 * t + ph)) + 0.08 * math.sin(2 * math.pi * (7 * t + ph * 3))
            splat(a, px, py - ps * 0.2, ps * 2.2, np.array([1.0, 0.5, 0.1]), 90 * flick)
            for k in (-1, 0, 1):
                dl.polygon(ellipse_poly(px + k * ps * 0.45, py, ps * 0.55, ps * 0.62, 0, 24), fill=(232, 110, 20, 255))
            dl.line([(px, py - ps * 0.6), (px + 2, py - ps * 0.9)], fill=(60, 90, 30, 255), width=3)
            gl = (255, int(200 * flick), 60, 255)
            dl.polygon([(px - ps * 0.35, py - ps * 0.12), (px - ps * 0.12, py - ps * 0.12), (px - ps * 0.24, py - ps * 0.32)], fill=gl)
            dl.polygon([(px + ps * 0.12, py - ps * 0.12), (px + ps * 0.35, py - ps * 0.12), (px + ps * 0.24, py - ps * 0.32)], fill=gl)
            dl.polygon([(px - ps * 0.35, py + ps * 0.1), (px + ps * 0.35, py + ps * 0.1), (px, py + ps * 0.32)], fill=gl)
        for b in bats:
            life = (t * b["k"] + b["ph"]) % 1
            bx = (-0.1 + 1.2 * life) * w if b["dir"] > 0 else (1.1 - 1.2 * life) * w
            by = b["y"] + 10 * math.sin(2 * math.pi * life * 2)
            flap = math.sin(2 * math.pi * (t * 8 + b["ph"]))
            s = b["s"]
            wing = s * (0.4 + 0.6 * abs(flap))
            dl.polygon([(bx, by), (bx - s * 1.4, by - wing), (bx - s * 0.9, by - wing * 0.2), (bx - s * 0.5, by - wing * 0.5), (bx, by + s * 0.25),
                        (bx + s * 0.5, by - wing * 0.5), (bx + s * 0.9, by - wing * 0.2), (bx + s * 1.4, by - wing)], fill=(6, 2, 8, 255))
        im = to_img(a)
        im.paste(lay, (0, 0), lay)
        frames.append(down(bloom(im, 5, 0.35, 170), GW, GH))
    return frames


def gif_frames_christmas():
    """a snowy night: pines with blinking lights on a snow field, snow falling in three depths, a warm glow."""
    w, h = AW, AH
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    sky = grad_stops(w, h, [(0, "#040A18"), (0.6, "#0E2034"), (1, "#16304A")])
    sky += (radial(w, h, w * 0.5, h * 1.05, w * 0.8) ** 2)[..., None] * hexrgb("#FFB070") * 0.25
    st = np.zeros_like(sky)
    stars(st, rng(13), 300, 0.55, 0.03, (60, 210), k=1.1)
    sky += st
    ground = h * 0.80 + h * 0.03 * np.sin(x[0] / w * 2 * np.pi * 1.5)
    gm = np.clip((y - ground[None, :]) / 1.5, 0, 1)
    sky = sky * (1 - gm[..., None]) + (hexrgb("#D8E6F4") * (0.75 + 0.25 * (1 - (y - h * 0.8) / (h * 0.2))[..., None])) * gm[..., None]
    base = to_img(sky)
    r = random.Random(31)
    trees = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(trees)
    lights = []
    for (tx, th, shade) in [(w * 0.08, 0.42, 0.9), (w * 0.22, 0.30, 0.7), (w * 0.80, 0.46, 1.0), (w * 0.93, 0.32, 0.75), (w * 0.62, 0.22, 0.55), (w * 0.38, 0.20, 0.5)]:
        top = h * (0.80 - th)
        base_y = h * 0.82
        tw = (base_y - top) * 0.55
        col = (int(10 + 14 * shade), int(36 + 40 * shade), int(26 + 26 * shade), 255)
        for k in range(4):                                                          # four tiers
            ty0 = top + (base_y - top) * k / 4.2
            ty1 = top + (base_y - top) * (k + 1.3) / 4.2
            half = tw * (0.35 + 0.65 * (k + 1) / 4) / 2
            d.polygon([(tx, ty0), (tx - half, ty1), (tx + half, ty1)], fill=col)
            d.polygon([(tx - half * 0.9, ty1 - 2), (tx + half * 0.9, ty1 - 2), (tx + half * 0.6, ty1 - 6), (tx - half * 0.6, ty1 - 6)], fill=(225, 235, 245, 230))
            for _ in range(int(3 + 4 * shade)):
                lx = tx + r.uniform(-half * 0.8, half * 0.8) * 0.9
                ly = ty1 - r.uniform(2, 10)
                lights.append((lx, ly, r.choice([(1.0, 0.25, 0.25), (1.0, 0.8, 0.3), (0.3, 0.9, 0.5), (0.4, 0.6, 1.0)]), r.random()))
        d.rectangle([tx - 3, base_y - 4, tx + 3, base_y + 4], fill=(40, 24, 14, 255))
        star = [(tx + math.cos(-math.pi / 2 + i * math.pi / 5) * (7 if i % 2 == 0 else 3), top - 4 + math.sin(-math.pi / 2 + i * math.pi / 5) * (7 if i % 2 == 0 else 3)) for i in range(10)]
        d.polygon(star, fill=(255, 220, 110, 255))
    flakes = []
    for depth, n, size, spd in ((0, 70, 1.0, 1), (1, 45, 1.6, 1), (2, 18, 2.6, 2)):
        for _ in range(n):
            flakes.append(dict(x=r.random() * w, ph=r.random(), s=size * r.uniform(0.8, 1.2), k=spd, sw=r.uniform(4, 14) * (1 + depth * 0.5), z=depth))
    frames = []
    for f in range(GN):
        t = f / GN
        im = base.copy()
        im.paste(trees, (0, 0), trees)
        a = to_arr(im)
        for (lx, ly, col, ph) in lights:
            on = 0.55 + 0.45 * math.sin(2 * math.pi * (2 * t + ph))
            splat(a, lx, ly, 1.3, np.array(col, np.float32), 255 * on)
            splat(a, lx, ly, 4.0, np.array(col, np.float32), 70 * on)
        for fl in flakes:
            life = (t * fl["k"] + fl["ph"]) % 1
            fy = -10 + (h + 20) * life
            fx = (fl["x"] + fl["sw"] * math.sin(2 * math.pi * (life * 2 + fl["ph"]))) % w
            splat(a, fx, fy, fl["s"], np.array([1.0, 1.0, 1.0]), 150 + 50 * fl["z"])
        frames.append(down(bloom(to_img(a), 5, 0.40, 170), GW, GH))
    return frames


MAKERS = {"carbon": carbon, "velvet": velvet, "nebula": nebula, "midnight": midnight, "jade": jade, "mono": mono, "abyss": abyss, "dusk": dusk,
          "royal": royal, "glacier": glacier, "clover": clover, "easter": easter,
          "ember": gif_frames_ember, "synthwave": gif_frames_synthwave, "aurora": gif_frames_aurora, "sakura": gif_frames_sakura, "terminal": gif_frames_terminal,
          "newyear": gif_frames_newyear, "lunar": gif_frames_lunar, "valentine": gif_frames_valentine, "halloween": gif_frames_halloween, "christmas": gif_frames_christmas}


def save_gif(frames, path):
    # one shared palette from all frames keeps the loop free of palette flicker
    strip = Image.new("RGB", (GW, GH * len(frames)))
    for i, f in enumerate(frames):
        strip.paste(f, (0, i * GH))
    pal = strip.quantize(colors=255, method=Image.MEDIANCUT, kmeans=2)
    q = [f.quantize(palette=pal, dither=Image.FLOYDSTEINBERG) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=GD, loop=0, optimize=False, disposal=1)


def main():
    args = sys.argv[1:]
    out = args[0] if args and not args[0] in MAKERS else os.path.join(os.environ.get("LOCALAPPDATA", "."), "HyperionDeploy", "hyperion-looks")
    only = [a for a in args if a in MAKERS]
    os.makedirs(os.path.join(out, "looks"), exist_ok=True)
    firsts = {}
    for lid, fname, *_ in LOOKS:
        path = os.path.join(out, "looks", fname)
        if only and lid not in only:
            if os.path.exists(path):
                im = Image.open(path)
                if fname.endswith(".gif"):
                    im.seek(GN // 3)
                firsts[lid] = im.convert("RGB")
            continue
        res = MAKERS[lid]()
        if fname.endswith(".gif"):
            save_gif(res, path)
            firsts[lid] = res[GN // 3].convert("RGB")
        else:
            if fname.endswith(".jpg"):
                res.save(path, quality=95, optimize=True, progressive=False, subsampling=0)
            else:
                res.save(path, optimize=True)
            firsts[lid] = res
        print(f"{fname:20s} {os.path.getsize(path) / 1024:8.1f} KB", flush=True)
    if len(firsts) < len(LOOKS):
        print("atlas skipped: not every picture exists yet")
        return
    rows = math.ceil(len(LOOKS) / TCOLS)
    atlas = Image.new("RGB", (TW * TCOLS, TH * rows), (0, 0, 0))
    for i, (lid, fname, panel, accent, card, tr, kind) in enumerate(LOOKS):
        im = firsts[lid]
        sc = max(TW / im.width, TH / im.height)
        im = im.resize((max(TW, round(im.width * sc)), max(TH, round(im.height * sc))), Image.LANCZOS)
        x0, y0 = (im.width - TW) // 2, (im.height - TH) // 2
        im = im.crop((x0, y0, x0 + TW, y0 + TH))
        cell = Image.blend(Image.new("RGB", (TW, TH), tuple(int(v) for v in hexrgb(panel))), im, 1 - tr * 0.6)
        atlas.paste(cell, ((i % TCOLS) * TW, (i // TCOLS) * TH))
    atlas.save(os.path.join(out, "thumbs_v3.png"), optimize=True)
    print(f"thumbs_v3.png        {os.path.getsize(os.path.join(out, 'thumbs_v3.png')) / 1024:8.1f} KB")
    cw, ch = 384, 288                                                                # a contact sheet for a person
    cols = 4
    sheet = Image.new("RGB", (cw * cols, ch * math.ceil(len(LOOKS) / cols)), (8, 8, 10))
    for i, (lid, *_rest) in enumerate(LOOKS):
        sheet.paste(firsts[lid].resize((cw, ch), Image.LANCZOS), ((i % cols) * cw, (i // cols) * ch))
    sheet.save(os.path.join(out, "contact.png"), optimize=True)


if __name__ == "__main__":
    main()
