#!/usr/bin/env python3
"""Draw the particle sprites the renderer embeds.

Wallpaper Engine's stock particle textures ship with the program, not with any
wallpaper, so they are not ours to redistribute. These are our own drawings,
calibrated against measurements of the real ones — because for particles the
energy matters far more than the artwork.

Corpus sprites fall into two classes, and conflating them is what made every
particle a bright disc:

    sprite                  mean alpha   mean RGB
    particle/halo                0.152      1.000
    particle/halo_2              0.076      1.000
    particle/halo_4              0.040      1.000
    particle/halo_6              0.465      1.000
    particle/drop                0.189      0.992
    particle/smoke/smoke2        0.462      0.921
    particle/light/shafts        0.107      1.000
    particle/chromaticdot        1.000      0.153
    particle/fog/fog1            1.000      0.052
    particle/misc/wave           1.000      0.017
    particle/nature/rain1        1.000      0.006

The first class carries its picture in alpha over white. The second is
alpha-opaque and carries it in *RGB*, drawn additively, where black contributes
nothing — `nature/rain1` is 99.4% black with a few bright streaks. Each sprite
below is generated, then rescaled so its mean lands on the measured value.

Run from the repo root; writes src/scene/sprites/*.png. Pure stdlib, so there
is nothing to install.
"""

import math
import os
import struct
import zlib

OUT = os.path.join("src", "scene", "sprites")

# name -> (side, channel the picture lives in, target mean)
SPRITES = {
    "halo":         (64,  "alpha", 0.152),
    "halo_2":       (64,  "alpha", 0.076),
    "halo_4":       (128, "alpha", 0.040),
    "halo_6":       (128, "alpha", 0.465),
    "drop":         (64,  "alpha", 0.189),
    "smoke":        (128, "alpha", 0.462),
    "shafts":       (128, "alpha", 0.107),
    "chromaticdot": (64,  "rgb",   0.153),
    "fog":          (128, "rgb",   0.052),
    "wave":         (128, "rgb",   0.017),
    "rain":         (128, "rgb",   0.006),
}


# --- value noise -----------------------------------------------------------

def _hash2(ix, iy, seed):
    h = (ix * 374761393 + iy * 668265263 + seed * 2147483647) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFFFF) / 0xFFFFFFFF


def _smooth(t):
    return t * t * (3.0 - 2.0 * t)


def value_noise(x, y, seed):
    ix, iy = math.floor(x), math.floor(y)
    fx, fy = _smooth(x - ix), _smooth(y - iy)
    a = _hash2(ix, iy, seed)
    b = _hash2(ix + 1, iy, seed)
    c = _hash2(ix, iy + 1, seed)
    d = _hash2(ix + 1, iy + 1, seed)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def fbm(x, y, seed, octaves=4, frequency=3.0):
    total, amplitude, scale, norm = 0.0, 1.0, frequency, 0.0
    for _ in range(octaves):
        total += amplitude * value_noise(x * scale, y * scale, seed)
        norm += amplitude
        amplitude *= 0.5
        scale *= 2.0
    return total / norm


# --- shapes ----------------------------------------------------------------
# Each returns an unnormalised field in 0..1 over x, y in [-1, 1].

def shape_halo(x, y, mean):
    # Gaussian; over the unit square exp(-k r^2) integrates to
    # (sqrt(pi/k) erf(sqrt k))^2 / 4, and erf(sqrt k) is 1 to four decimals for
    # every k we use here, so k = pi / (4 mean) inverts it directly.
    k = max(math.pi / (4.0 * mean), 0.5)
    r2 = x * x + y * y
    return 0.0 if r2 > 1.0 else math.exp(-k * r2)


def shape_drop(x, y, mean):
    # A teardrop: narrow in x, tapering along +y.
    taper = max(1.0 + 0.6 * y, 0.2)
    sx = x / (0.62 * taper)
    r2 = sx * sx + y * y
    return 0.0 if r2 > 1.0 else math.exp(-2.4 * r2)


def shape_smoke(x, y, mean):
    fade = max(0.0, min(1.0, 1.0 - math.hypot(x, y)))
    return fbm(x, y, 1337, octaves=4, frequency=2.0) * fade


def shape_fog(x, y, mean):
    fade = max(0.0, min(1.0, 1.0 - math.hypot(x, y)))
    return fbm(x, y, 7717, octaves=4, frequency=3.0) * fade * fade


def shape_shafts(x, y, mean):
    r = math.hypot(x, y)
    if r > 1.0:
        return 0.0
    angle = math.atan2(y, x)
    # A handful of wedges of uneven width, so they do not look mechanical.
    wedge = 0.5 + 0.5 * math.sin(angle * 5.0 + 1.3 * math.sin(angle * 2.0))
    return (wedge ** 3.0) * ((1.0 - r) ** 1.5)


def shape_rain(x, y, mean):
    # Shear so the streaks lean, then keep only the crests of a high-frequency
    # ridge across the sheared axis.
    u = x * 12.0 + y * 2.0
    ridge = abs(math.sin(u * math.pi))
    along = fbm(x, y, 4242, octaves=2, frequency=2.0)
    fade = max(0.0, min(1.0, 1.0 - abs(y)))
    return (ridge ** 24.0) * along * fade


def shape_wave(x, y, mean):
    r = math.hypot(x, y)
    if r > 1.0:
        return 0.0
    edge = 1.0 - min(1.0, abs((r - 0.78) / 0.16))
    return edge * edge


def shape_dot(x, y, mean):
    return max(0.0, min(1.0, 1.0 - math.hypot(x, y) / 0.5))


SHAPES = {
    "halo": shape_halo, "halo_2": shape_halo, "halo_4": shape_halo, "halo_6": shape_halo,
    "drop": shape_drop, "smoke": shape_smoke, "shafts": shape_shafts,
    "chromaticdot": shape_dot, "fog": shape_fog, "wave": shape_wave, "rain": shape_rain,
}


# --- rendering -------------------------------------------------------------

def render(name, side, channel, mean):
    shape = SHAPES[name]
    field = []
    for py in range(side):
        row = []
        for px in range(side):
            x = (px + 0.5) / side * 2.0 - 1.0
            y = (py + 0.5) / side * 2.0 - 1.0
            row.append(max(0.0, min(1.0, shape(x, y, mean))))
        field.append(row)

    average = sum(sum(r) for r in field) / (side * side)
    gain = 1.0 if average <= 1e-6 else mean / average

    pixels = bytearray()
    for row in field:
        for v in row:
            value = int(round(max(0.0, min(1.0, v * gain)) * 255))
            if channel == "alpha":
                # White, with the picture in alpha. Stored straight, not
                # premultiplied; the renderer premultiplies on load.
                pixels += bytes((255, 255, 255, value))
            else:
                # Opaque, with the picture in RGB over black.
                pixels += bytes((value, value, value, 255))
    return pixels


def write_png(path, side, pixels):
    raw = bytearray()
    stride = side * 4
    for y in range(side):
        raw.append(0)  # filter type 0 (None)
        raw += pixels[y * stride:(y + 1) * stride]

    def chunk(tag, data):
        out = struct.pack(">I", len(data)) + tag + data
        return out + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", side, side, 8, 6, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", header)
           + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
           + chunk(b"IEND", b""))
    with open(path, "wb") as handle:
        handle.write(png)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, (side, channel, mean) in SPRITES.items():
        pixels = render(name, side, channel, mean)
        path = os.path.join(OUT, name + ".png")
        write_png(path, side, pixels)

        total_a = sum(pixels[i + 3] for i in range(0, len(pixels), 4))
        total_rgb = sum(pixels[i] + pixels[i + 1] + pixels[i + 2] for i in range(0, len(pixels), 4))
        n = side * side
        print(f"{name:<14} {side:>4}px  meanA={total_a / n / 255:.3f}  "
              f"meanRGB={total_rgb / 3 / n / 255:.3f}  (target {channel} {mean})")


if __name__ == "__main__":
    main()
