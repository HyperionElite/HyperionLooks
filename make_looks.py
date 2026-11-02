r"""looks_art.py: draws the background pictures of the Ready-made looks (Quick Access > Theme, Settings > Theme & Sharing).

Every picture is generated here from code (gradients, noise, particles, lines): nothing is copied from anywhere, so the public repo
HyperionElite/HyperionLooks can host them. The hub downloads them from raw.githubusercontent.com when a look is applied.

  python tools/looks_art.py [out_dir]        default out_dir = %LOCALAPPDATA%\HyperionDeploy\hyperion-looks

Writes   looks/<id>.jpg | .png | .gif     one background per look (GIFs are 160 x 120 x 24 frames: the hub decodes GIF backgrounds at 160 px)
         thumbs.png                       one atlas of every look's preview (192 x 120 cells, 3 columns), the order of LOOKS below
         contact.png                      a contact sheet to look at (not used by the hub)

The colours of each look live in the hub (Features.Looks.LIST); the ones here only tint the thumbnails and must match it.
"""
import math, os, random, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1024, 768                    # static pictures (Roblox draws up to 1024)
GW, GH, GN, GD = 160, 120, 24, 90   # GIF size, frames, ms per frame
TW, TH, TCOLS = 192, 120, 3         # thumbnail cell, atlas columns

# id, file, panel hex, accent hex, card hex, image transparency on windows (the thumbnail previews the window)
LOOKS = [
    ("carbon",    "carbon.jpg",    "#111113", "#E8B84A", "#19191C", 0.30),
    ("ember",     "ember.gif",     "#140C0A", "#FF7A3D", "#1E1210", 0.30),
    ("velvet",    "velvet.jpg",    "#14080B", "#E5484D", "#1E0D11", 0.28),
    ("synthwave", "synthwave.gif", "#120A1A", "#FF4FA3", "#1C1026", 0.32),
    ("nebula",    "nebula.jpg",    "#0F0B1A", "#9B6BFF", "#171127", 0.28),
    ("midnight",  "midnight.jpg",  "#0A0F1E", "#4C8DFF", "#111A2E", 0.25),
    ("aurora",    "aurora.gif",    "#08121A", "#5AD1E6", "#0F1D26", 0.30),
    ("jade",      "jade.png",      "#08130F", "#30C48D", "#0E1D17", 0.30),
    ("mono",      "mono.png",      "#0F0F11", "#D8D8E0", "#18181B", 0.30),
]


def hexrgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def rng(seed):
    return np.random.default_rng(seed)


def value_noise(w, h, cells, r, octaves=5, gain=0.5):
    """fBm made of upscaled random grids (bicubic): smooth, 0..1."""
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    c = cells
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


def grad_v(w, h, top, bottom, power=1.0):
    t = (np.linspace(0, 1, h, dtype=np.float32) ** power)[:, None, None]
    return np.broadcast_to(top * (1 - t) + bottom * t, (h, w, 3)).copy()


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


def to_img(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def stars(canvas, r, n, ymax=1.0, big=0.02, bright=(120, 255), tint=(1, 1, 1), pw=1.4):
    h, w = canvas.shape[:2]
    for _ in range(n):
        x, y = r.random() * w, (r.random() ** pw) * h * ymax
        b = r.uniform(*bright) * (r.random() ** 2.2)
        s = 0.55 + (1.4 if r.random() < big else 0.35) * r.random()
        c = np.array(tint, np.float32) * np.array([0.85 + 0.15 * r.random(), 0.9, 1.0], np.float32)
        splat(canvas, x, y, s, c, b)


# ── the pictures ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def carbon():
    r = rng(11)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    s = 14.0
    cx, cy = np.floor(x / s), np.floor(y / s)
    fx, fy = (x / s) - cx, (y / s) - cy
    horiz = ((cx + cy) % 4) < 2                                   # 2 x 2 twill: diagonal steps of horizontal / vertical tows
    across = np.where(horiz, fy, fx)
    along = np.where(horiz, fx, fy)
    tow = np.sin(np.pi * across) ** 0.8                           # each tow is round: bright in its middle
    fibre = 0.9 + 0.1 * np.sin(across * np.pi * 9 + along * 0.6)
    light = (x / W) * 0.6 + (1 - y / H) * 0.4                     # the sheen flips between the two directions, like real carbon
    lit = np.where(horiz, 0.45 + 0.55 * light, 0.45 + 0.55 * (1 - light))
    v = 11 + 34 * tow * fibre * lit
    img = np.stack([v, v, v * 1.06], -1)
    gold = hexrgb("#E8B84A")
    g = radial(W, H, W * 0.92, H * 0.02, W * 0.75) ** 2.2
    img += g[..., None] * gold * 0.30
    g2 = radial(W, H, W * 0.05, H * 1.0, W * 0.55) ** 2.5
    img += g2[..., None] * gold * 0.10
    img = vignette(img, 0.5)
    img = grain(img, r, 1.5)
    return to_img(img)


def velvet():
    r = rng(23)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    n = value_noise(W, H, 3, r, 4) * 2 - 1
    n2 = value_noise(W, H, 2, r, 3) * 2 - 1
    th = math.radians(28)
    u = (x * math.cos(th) + y * math.sin(th)) / W
    u = u + 0.10 * n + 0.05 * n2
    hgt = 0.62 * np.sin(u * 2 * np.pi * 3.2) + 0.30 * np.sin(u * 2 * np.pi * 6.1 + 1.3 + 2 * n2) + 0.08 * np.sin(u * 2 * np.pi * 9.0 + 2 * n)
    hgt = np.asarray(Image.fromarray(((hgt + 1) * 127.5).astype(np.uint8), 'L').filter(ImageFilter.GaussianBlur(3)), np.float32) / 127.5 - 1
    gy, gx = np.gradient(hgt * 40)
    nz = np.ones_like(gx)
    norm = np.sqrt(gx * gx + gy * gy + nz)
    nx, ny, nz = -gx / norm, -gy / norm, nz / norm
    L = np.array([-0.45, -0.55, 0.70]); L /= np.linalg.norm(L)
    diff = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
    hv = L + np.array([0, 0, 1.0]); hv /= np.linalg.norm(hv)
    spec = np.clip(nx * hv[0] + ny * hv[1] + nz * hv[2], 0, 1) ** 28
    base = hexrgb("#2E0812")
    hi = hexrgb("#B0283A")
    img = base * (0.35 + 0.9 * diff[..., None] ** 1.6) + hi * (spec[..., None] * 0.55)
    glow = radial(W, H, W * 0.15, H * 0.1, W * 0.8) ** 2
    img = img * (0.75 + 0.45 * glow[..., None])
    img = vignette(img, 0.6)
    img = grain(img, r, 1.8)
    return to_img(img)


def nebula():
    r = rng(37)
    a = value_noise(W, H, 3, r, 6, 0.55)
    b = value_noise(W, H, 4, r, 6, 0.5)
    c = value_noise(W, H, 2, r, 4, 0.5)
    dens = np.clip((a - 0.35) / 0.65, 0, 1) ** 1.6
    deep = hexrgb("#0A0716")
    purple = hexrgb("#5A2C9E")
    magenta = hexrgb("#B0347E")
    blue = hexrgb("#2848A8")
    img = deep + dens[..., None] * purple * 0.85
    img += (np.clip(b - 0.45, 0, 1) * dens)[..., None] * magenta * 1.2
    img += (np.clip(c - 0.55, 0, 1) * (1 - dens * 0.5))[..., None] * blue * 0.9
    lane = np.clip(0.55 - value_noise(W, H, 6, r, 4), 0, 1) * 1.4                      # dark dust lanes
    img *= (1 - 0.6 * lane[..., None] * dens[..., None])
    st = np.zeros_like(img)
    stars(st, r, 1400, 1.0, 0.015, (90, 255))
    img += st
    for _ in range(5):                                                               # a few young stars with a halo
        x, y = r.random() * W, r.random() * H
        splat(img, x, y, 2.2, np.array([1.0, 0.9, 1.0]), 210)
        splat(img, x, y, 14, np.array([0.7, 0.5, 1.0]), 22)
    img = vignette(img, 0.45)
    img = grain(img, r, 1.6)
    return to_img(img)


def midnight():
    r = rng(53)
    img = grad_v(W, H, hexrgb("#03060F"), hexrgb("#13284F"), 1.6)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    band = np.exp(-((y - (H * 0.95 - x * 0.55)) / (H * 0.16)) ** 2)                  # a faint Milky Way across the sky
    mw = value_noise(W, H, 5, r, 5)
    img += (band * mw)[..., None] * hexrgb("#4A5F96") * 0.35
    st = np.zeros_like(img)
    stars(st, r, 1700, 0.95, 0.02, (80, 255), (0.92, 0.96, 1.0))
    img += st * (1 - 0.6 * (y / H))[..., None]
    img += (radial(W, H, W * 0.62, H * 1.05, W * 0.75) ** 2)[..., None] * hexrgb("#3A6BD0") * 0.35
    # two ridge lines: the far one blue-grey, the near one almost black
    def ridge(base_y, amp, seed, cells):
        rr = rng(seed)
        line = value_noise(W, 4, cells, rr, 5)[0] * 2 - 1
        return base_y + amp * line
    far = ridge(H * 0.80, H * 0.07, 61, 6)
    near = ridge(H * 0.90, H * 0.06, 67, 4)
    far_mask = (y > far[None, :]).astype(np.float32)
    near_mask = (y > near[None, :]).astype(np.float32)
    img = img * (1 - far_mask[..., None]) + far_mask[..., None] * hexrgb("#0C1630")
    img = img * (1 - near_mask[..., None]) + near_mask[..., None] * hexrgb("#04070E")
    im = to_img(grain(img, r, 1.2)).filter(ImageFilter.GaussianBlur(0.4))
    return im


def jade():
    r = rng(71)
    hgt = value_noise(W, H, 3, r, 4, 0.45)
    N = 26
    v = hgt * N * 0.97 + 0.37                                         # (clipped flat areas fall between two levels: no filled blobs)
    d = np.abs(v - np.round(v))
    gy, gx = np.gradient(v)
    gm = np.sqrt(gx * gx + gy * gy) + 1e-4
    px = d / gm                                                       # distance to the nearest contour in pixels
    line = np.clip(1.15 - px, 0, 1)
    idx = (np.round(v) % 5 == 0)
    a = np.where(idx, line * 0.75, line * 0.32)
    bg = grad_v(W, H, hexrgb("#06120E"), hexrgb("#040C09"))
    bg += (radial(W, H, W * 0.3, H * 0.35, W * 0.8) ** 2)[..., None] * hexrgb("#12382A") * 0.9
    col = hexrgb("#30C48D")
    img = bg * (1 - a[..., None]) + col * a[..., None] * 0.9 + bg * a[..., None] * 0.1
    img = vignette(img, 0.45)
    return to_img(img)


def mono():
    r = rng(83)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    img = 12 + 10 * ((x / W) * 0.5 + (1 - y / H) * 0.5)
    img = np.stack([img, img, img * 1.05], -1)
    s = 18.0
    cx, cy = (np.floor(x / s) + 0.5) * s, (np.floor(y / s) + 0.5) * s
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    field = 0.25 + 0.75 * np.clip(1 - np.sqrt(((cx - W * 0.1) / W) ** 2 + ((cy - H * 0.05) / H) ** 2) / 1.1, 0, 1) ** 1.5
    rad = 0.5 + 3.4 * field
    dot = np.clip(rad - d + 0.5, 0, 1)
    img += dot[..., None] * np.array([34, 34, 38], np.float32)
    img += (radial(W, H, W * 0.1, H * 0.0, W * 0.9) ** 2.5)[..., None] * np.array([40, 40, 46], np.float32)
    img = vignette(img, 0.4)
    return to_img(img)


def gif_frames_ember():
    r = random.Random(97)
    parts = []
    for _ in range(70):
        parts.append(dict(x=r.random() * GW, ph=r.random(), k=r.choice([1, 1, 2]), trav=r.uniform(0.35, 1.0), sw=r.uniform(2, 7),
                          wph=r.random(), sig=r.uniform(0.45, 1.25), br=r.uniform(120, 260)))
    frames = []
    yy, xx = np.mgrid[0:GH, 0:GW].astype(np.float32)
    for f in range(GN):
        t = f / GN
        img = grad_v(GW, GH, hexrgb("#0A0605"), hexrgb("#2C0B05"), 1.4)
        flick = 1 + 0.10 * math.sin(2 * math.pi * t) + 0.05 * math.sin(2 * math.pi * 3 * t + 1)
        img += (radial(GW, GH, GW * 0.5, GH * 1.15, GW * 0.8) ** 1.8)[..., None] * hexrgb("#8A2208") * 0.65 * flick
        img += (radial(GW, GH, GW * 0.2, GH * 1.1, GW * 0.5) ** 2)[..., None] * hexrgb("#A0400A") * 0.4 * (2 - flick)
        for p in parts:
            life = (t * p["k"] + p["ph"]) % 1
            y = GH * (1.05 - life * p["trav"] * 1.1)
            x = p["x"] + p["sw"] * math.sin(2 * math.pi * (life + p["wph"]))
            a = math.sin(math.pi * life) ** 1.5
            col = np.array([1.0, 0.75 - 0.4 * life, 0.35 - 0.25 * life], np.float32)
            splat(img, x, y, p["sig"], col, p["br"] * a)
            splat(img, x, y, p["sig"] * 3.2, col, p["br"] * a * 0.12)
        frames.append(to_img(img))
    return frames


def gif_frames_synthwave():
    S = 4                                                              # drawn 4x larger, then reduced (smooth lines)
    w, h = GW * S, GH * S
    hz = int(h * 0.60)
    r = random.Random(5)
    starl = [(r.random() * w, r.random() * hz * 0.8, r.random(), r.uniform(0.4, 1)) for _ in range(40)]
    frames = []
    for f in range(GN):
        t = f / GN
        sky = grad_v(w, h, hexrgb("#0E0420"), hexrgb("#4A0E4E"), 1.3)
        img = sky.copy()
        img[hz:] = grad_v(w, h - hz, hexrgb("#16061F"), hexrgb("#07020C"))
        for (sx, sy, ph, b) in starl:
            a = 0.55 + 0.45 * math.sin(2 * math.pi * (t * 2 + ph))
            splat(img, sx, sy, 1.6, np.array([1, 0.85, 1.0]), 220 * a * b)
        # the sun: pink-to-gold, its lower half cut by stripes that slide down
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
        glow = radial(w, h, cx, cy, rad * 2.2) ** 2.4
        img += glow[..., None] * hexrgb("#FF3A8C") * 0.22
        im = to_img(img)
        dr = ImageDraw.Draw(im)
        line = (255, 64, 190)
        # horizontal grid lines rush toward the viewer, one cell per loop
        for k in range(0, 40):
            z = 1.0 + (k - t) * 0.6
            if z <= 0.15:
                continue
            yv = hz + (h - hz) * (0.9 / z) ** 1.0 * 0.55
            if yv >= h:
                continue
            fade = min(1, (yv - hz) / (h * 0.05))
            c = tuple(int(v * (0.35 + 0.65 * fade)) for v in line)
            dr.line([(0, yv), (w, yv)], fill=c, width=max(1, int(S * 0.6 * (0.5 + fade))))
        for k in range(-16, 17):
            xb = w * 0.5 + k * w * 0.11
            dr.line([(w * 0.5 + k * w * 0.012, hz), (xb, h)], fill=line, width=max(1, int(S * 0.7)))
        dr.line([(0, hz), (w, hz)], fill=(255, 140, 220), width=S)
        im = Image.blend(im, im.filter(ImageFilter.GaussianBlur(S * 2.2)), 0.35)
        frames.append(im.resize((GW, GH), Image.LANCZOS))
    return frames


def gif_frames_aurora():
    S = 2
    w, h = GW * S, GH * S
    r = rng(131)
    st = np.zeros((h, w, 3), np.float32)
    stars(st, r, 300, 0.92, 0.02, (60, 230), (0.9, 0.95, 1.0), 1.0)
    xs = np.arange(w, dtype=np.float32) / w
    yy = np.arange(h, dtype=np.float32)[:, None]
    ray_seed = r.random(64).astype(np.float32)
    frames = []
    trees = []
    rr = random.Random(9)
    x = 0
    while x < w:
        th = rr.uniform(0.08, 0.20) * h
        tw = th * rr.uniform(0.35, 0.5)
        trees.append((x, th, tw))
        x += tw * rr.uniform(0.45, 0.9)
    for f in range(GN):
        t = f / GN
        img = grad_v(w, h, hexrgb("#02060C"), hexrgb("#0A1B26"), 1.2) + st
        for (base, amp, cyc, spd, colb, colt, strength) in ((0.52, 0.08, 1.2, 1, "#3CF0B4", "#7A5CFF", 1.0), (0.36, 0.06, 2.1, -1, "#46D8F0", "#C060FF", 0.55)):
            yc = h * (base + amp * np.sin(2 * np.pi * (xs * cyc + spd * t)) + 0.04 * np.sin(2 * np.pi * (xs * 3.3 - 2 * spd * t) + 1.1))
            dy = yc[None, :] - yy                                        # > 0 above the lower edge
            above = np.exp(-np.clip(dy, 0, None) / (h * 0.11)) * (dy >= 0)
            below = np.exp(-(np.clip(-dy, 0, None) / (h * 0.025)) ** 2) * (dy < 0)
            prof = above + below
            idx = (xs * 64).astype(int) % 64
            rays = 0.55 + 0.45 * np.sin(2 * np.pi * (xs * 23 + ray_seed[idx] + t * 2 * spd))
            inten = prof * rays[None, :] * strength
            mix = np.clip(dy / (h * 0.16), 0, 1)[..., None]
            col = hexrgb(colb)[None, None, :] * (1 - mix) + hexrgb(colt)[None, None, :] * mix
            img += inten[..., None] * col * 0.75
        im = to_img(img).filter(ImageFilter.GaussianBlur(S * 0.8))
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, h * 0.94, w, h], fill=(2, 5, 8))
        for (tx, th, tw) in trees:
            dr.polygon([(tx - tw / 2, h * 0.95), (tx + tw / 2, h * 0.95), (tx, h * 0.95 - th)], fill=(2, 5, 8))
        frames.append(im.resize((GW, GH), Image.LANCZOS))
    return frames


def save_gif(frames, path):
    # one shared palette from all frames keeps the loop free of palette flicker
    strip = Image.new("RGB", (GW, GH * len(frames)))
    for i, f in enumerate(frames):
        strip.paste(f, (0, i * GH))
    pal = strip.quantize(colors=255, method=Image.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.FLOYDSTEINBERG) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=GD, loop=0, optimize=False, disposal=1)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("LOCALAPPDATA", "."), "HyperionDeploy", "hyperion-looks")
    os.makedirs(os.path.join(out, "looks"), exist_ok=True)
    makers = {"carbon": carbon, "velvet": velvet, "nebula": nebula, "midnight": midnight, "jade": jade, "mono": mono,
              "ember": gif_frames_ember, "synthwave": gif_frames_synthwave, "aurora": gif_frames_aurora}
    firsts = {}
    for lid, fname, *_ in LOOKS:
        res = makers[lid]()
        path = os.path.join(out, "looks", fname)
        if fname.endswith(".gif"):
            save_gif(res, path)
            firsts[lid] = res[GN // 3].convert("RGB")
        else:
            if fname.endswith(".jpg"):
                res.save(path, quality=90, optimize=True, progressive=False, subsampling=0)
            else:
                res.save(path, optimize=True)
            firsts[lid] = res
        print(f"{fname:16s} {os.path.getsize(path) / 1024:7.1f} KB")
    # thumbnails: the picture over the look's panel colour at the window transparency, cropped to the cell
    rows = math.ceil(len(LOOKS) / TCOLS)
    atlas = Image.new("RGB", (TW * TCOLS, TH * rows), (0, 0, 0))
    for i, (lid, fname, panel, accent, card, tr) in enumerate(LOOKS):
        im = firsts[lid]
        sc = max(TW / im.width, TH / im.height)
        im = im.resize((max(TW, round(im.width * sc)), max(TH, round(im.height * sc))), Image.LANCZOS)
        x0, y0 = (im.width - TW) // 2, (im.height - TH) // 2
        im = im.crop((x0, y0, x0 + TW, y0 + TH))
        base = Image.new("RGB", (TW, TH), tuple(int(v) for v in hexrgb(panel)))
        cell = Image.blend(base, im, 1 - tr)
        atlas.paste(cell, ((i % TCOLS) * TW, (i // TCOLS) * TH))
    atlas.save(os.path.join(out, "thumbs.png"), optimize=True)
    print(f"thumbs.png       {os.path.getsize(os.path.join(out, 'thumbs.png')) / 1024:7.1f} KB")
    # contact sheet for a person
    cw, ch = 512, 384
    sheet = Image.new("RGB", (cw * 3, ch * 3), (0, 0, 0))
    for i, (lid, *_r) in enumerate(LOOKS):
        sheet.paste(firsts[lid].resize((cw, ch), Image.LANCZOS), ((i % 3) * cw, (i // 3) * ch))
    sheet.save(os.path.join(out, "contact.png"))


if __name__ == "__main__":
    main()
