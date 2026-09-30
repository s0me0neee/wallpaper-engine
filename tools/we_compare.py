#!/usr/bin/env python3
"""Compare our renderer against a `we_capture.py` recording of native Wallpaper Engine.

    tools/we_compare.py papers/scene_example2 papers/_we_captures/proton_4k/2619088810.mkv

Renders our frames with `simulate`'s dump hook at a grid of `g_Time`s, pairs each
with the capture at the same time, and writes <out>/report.json plus pictures:

  heat.png    per-tile PSNR over the whole run, worst tiles red, on WE's frame
  motion.png  tiles that animate in one renderer and not the other
              (red: WE moves, we don't; blue: we move, WE doesn't)
  worst.png   the worst tiles side by side, WE | ours | 3x difference

Timing: WE's `g_Time` starts at the clip's first real frame, found as the
largest jump in its first seconds. WE may draw at half the capture rate, and its
start sits ~30 ms from that jump, so each of our frames is scored against the
best of the capture frames within [-100, +100] ms — a time error of that size is
the capture's, not the animation's. The chosen lag is reported per frame; a lag
that drifts over the run is a speed mismatch.

Stock assets (fonts, `util/noise`, ...) come from the local WE install via
`WE_ASSETS`, so what remains is the renderer, not our stand-ins for WE's files;
`--stand-ins` compares against those stand-ins instead.

Needs numpy and Pillow, ffmpeg/ffprobe, and a release build.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

TILE = 120  # px at the capture's size; 32x18 tiles at 4K


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, **kw)


def scene_info(binary, wallpaper):
    out = run([binary, "info", wallpaper], text=True).stdout
    width, height = map(int, re.search(r"canvas\s+(\d+)x(\d+)", out).group(1, 2))
    return out.splitlines()[0].strip(), (width, height)


def clip_size(clip):
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
               "-of", "csv=p=0", clip], text=True).stdout
    width, height = map(int, out.strip().split(","))
    return width, height


def clip_pts(clip):
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time",
               "-of", "csv=p=0", clip], text=True).stdout
    return np.array([float(line.strip().rstrip(",")) for line in out.split()])


def decode(clip, start, duration, size):
    """Frames in [start, start+duration) as (pts, uint8 HxWx3) pairs, at `size` = (w, h)."""
    width, height = size
    raw = run(["ffmpeg", "-v", "info", "-ss", f"{max(start, 0):.4f}", "-i", clip, "-t", f"{duration:.4f}",
               "-fps_mode", "passthrough", "-vf", f"scale={width}:{height}:flags=area,showinfo",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    pts = [max(start, 0) + float(m) for m in re.findall(rb"pts_time:([0-9.]+)", raw.stderr)]
    frames = np.frombuffer(raw.stdout, np.uint8).reshape(-1, height, width, 3)
    return list(zip(pts, frames))


def first_frame_time(clip):
    """WE's first real frame: the largest frame-to-frame jump in the clip's first 4 s."""
    frames = decode(clip, 0, 4, (192, 108))
    jumps = [np.abs(b.astype(np.int16) - a).mean() for (_, a), (_, b) in zip(frames, frames[1:])]
    return frames[int(np.argmax(jumps)) + 1][0]


WE_ASSETS = Path.home() / ".local/share/Steam/steamapps/common/wallpaper_engine/assets"


def render_ours(binary, wallpaper, time, scale, dump, assets, clock):
    # SIMULATE_CLOCK pins the scripts' wall clock (clocks, time-of-day scenes) to when WE's g_Time was 0.
    env = {"SIMULATE_DUMP": str(dump), "SIMULATE_TIME": f"{time:.4f}", "SIMULATE_SCALE": f"{scale:.6f}",
           "SIMULATE_CLOCK": f"{clock:.3f}"}
    if assets:
        env["WE_ASSETS"] = str(assets)
    subprocess.run([binary, "simulate", wallpaper], env={**os.environ, **env},
                   check=True, capture_output=True, timeout=120)
    return np.asarray(Image.open(dump).convert("RGB"))


def cover_crop(image, size):
    """Centre-crop `image` to `size`, the way WE covers a window with a canvas of another aspect."""
    width, height = size
    if image.shape[1] < width or image.shape[0] < height:
        # A rounding pixel short, or a canvas `simulate` will not upscale: stretch to match.
        image = np.asarray(Image.fromarray(image).resize((max(width, image.shape[1]), max(height, image.shape[0])), Image.LANCZOS))
    top, left = (image.shape[0] - height) // 2, (image.shape[1] - width) // 2
    return image[top:top + height, left:left + width]


def psnr(mse):
    return 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-6))


def tile_mse(a, b):
    diff = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    rows, cols = a.shape[0] // TILE, a.shape[1] // TILE
    diff = diff[:rows * TILE, :cols * TILE].mean(axis=2)
    return diff.reshape(rows, TILE, cols, TILE).mean(axis=(1, 3))


def luminance(image):
    return float((image.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722])).mean())


def tile_motion(stack):
    """Mean temporal std per tile, from quarter-size frames."""
    std = np.stack(stack).astype(np.float32).std(axis=0).mean(axis=2)
    tile = TILE // 4
    rows, cols = std.shape[0] // tile, std.shape[1] // tile
    return std[:rows * tile, :cols * tile].reshape(rows, tile, cols, tile).mean(axis=(1, 3))


def hide_our_windows(title):
    # Each dump opens a window for a moment; send it where nobody sees it flash.
    escaped = re.sub(r"([\\^$.|?*+()\[\]{}])", r"\\\1", title)
    subprocess.run(["hyprctl", "keyword", "windowrule", f"match:title ^{escaped}$, workspace special:wecompare silent"],
                   capture_output=True)


def draw_tiles(base, values, colour, label):
    image = Image.fromarray(base).convert("RGBA")
    overlay = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(overlay)
    for (row, col), value in np.ndenumerate(values):
        if value > 0:
            draw.rectangle([col * TILE, row * TILE, (col + 1) * TILE - 1, (row + 1) * TILE - 1],
                           fill=(*colour(value), int(40 + 150 * min(value, 1))))
    image = Image.alpha_composite(image, overlay).convert("RGB")
    ImageDraw.Draw(image).text((8, 8), label, fill=(255, 255, 255))
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("wallpaper")
    parser.add_argument("clip")
    parser.add_argument("--times", default="0:10:0.5", help="g_Time grid, start:stop:step")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--binary", default="target/release/wallpaper-engine")
    parser.add_argument("--stand-ins", action="store_true", help="render with our stand-ins, not WE's stock assets")
    args = parser.parse_args()
    args.assets = None if args.stand_ins or not WE_ASSETS.is_dir() else WE_ASSETS

    title, canvas = scene_info(args.binary, args.wallpaper)
    size = clip_size(args.clip)
    scale = max(size[0] / canvas[0], size[1] / canvas[1])
    out = args.out or Path("papers/_we_captures/compare") / Path(args.clip).stem
    out.mkdir(parents=True, exist_ok=True)
    hide_our_windows(title)
    try:
        compare(args, title, canvas, size, scale, out)
    finally:
        subprocess.run(["hyprctl", "reload"], capture_output=True)


def compare(args, title, canvas, size, scale, out):
    t0 = first_frame_time(args.clip)
    last = clip_pts(args.clip)[-1]
    # The clip is written as it records, so its mtime is its end.
    clock = os.path.getmtime(args.clip) - last + t0
    start, stop, step = map(float, args.times.split(":"))
    times = [t for t in np.arange(start, stop + step / 2, step) if t0 + t + 0.1 <= last]
    print(f"{title}: canvas {canvas[0]}x{canvas[1]} -> {size[0]}x{size[1]} (scale {scale:.4f}), WE starts at {t0:.3f}s, "
          f"{'WE stock assets' if args.assets else 'our stand-ins'}")

    small = (size[0] // 4, size[1] // 4)
    rows = []
    tile_sum = None
    worst_tile = None
    we_stack, our_stack = [], []
    for time in times:
        ours = cover_crop(render_ours(args.binary, args.wallpaper, time, scale, out / "ours.png", args.assets, clock), (size[0], size[1]))
        candidates = decode(args.clip, t0 + time - 0.1, 0.2, size)
        scores = [(((we.astype(np.float32) - ours) ** 2).mean(), pts, we) for pts, we in candidates]
        mse, pts, we = min(scores, key=lambda s: s[0])
        tiles = tile_mse(we, ours)
        tile_sum = tiles if tile_sum is None else tile_sum + tiles
        index = np.unravel_index(np.argmax(tiles), tiles.shape)
        if worst_tile is None or tiles[index] > worst_tile[0]:
            worst_tile = (tiles[index], time, we, ours)
        we_stack.append(np.asarray(Image.fromarray(we).resize(small, Image.BOX)))
        our_stack.append(np.asarray(Image.fromarray(ours).resize(small, Image.BOX)))
        row = {"time": round(float(time), 3), "lag_ms": round((pts - t0 - time) * 1000),
               "psnr": round(float(psnr(mse)), 2), "lum_we": round(luminance(we), 1), "lum_ours": round(luminance(ours), 1)}
        rows.append(row)
        print(f"  t={row['time']:5.2f}  lag {row['lag_ms']:+4d} ms  PSNR {row['psnr']:6.2f}  "
              f"luminance ours {row['lum_ours']:6.1f} / WE {row['lum_we']:6.1f}", flush=True)

    tile_psnr = psnr(tile_sum / len(times))
    reference = we_stack[len(we_stack) // 2]
    heat = np.clip((30 - tile_psnr) / 15, 0, 1)
    base = np.asarray(Image.fromarray(reference).resize(size, Image.BILINEAR))
    draw_tiles(base, heat, lambda v: (255, int(255 * (1 - v)), 0), "tile PSNR: yellow 30 dB -> red 15 dB").save(out / "heat.png")

    we_motion, our_motion = tile_motion(we_stack), tile_motion(our_stack)
    missing = np.clip((we_motion - 2 * our_motion - 1.5) / 6, 0, 1)
    extra = np.clip((our_motion - 2 * we_motion - 1.5) / 6, 0, 1)
    signed = np.where(missing > 0, missing, -extra)
    image = Image.fromarray(base).convert("RGBA")
    overlay = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(overlay)
    for (row, col), value in np.ndenumerate(signed):
        if value != 0:
            colour = (255, 40, 40) if value > 0 else (60, 120, 255)
            draw.rectangle([col * TILE, row * TILE, (col + 1) * TILE - 1, (row + 1) * TILE - 1],
                           fill=(*colour, int(60 + 150 * abs(value))))
    image = Image.alpha_composite(image, overlay).convert("RGB")
    ImageDraw.Draw(image).text((8, 8), "red: WE moves, we don't   blue: we move, WE doesn't", fill=(255, 255, 255))
    image.save(out / "motion.png")

    order = np.dstack(np.unravel_index(np.argsort(tile_psnr, axis=None), tile_psnr.shape))[0][:8]
    _, worst_time, we, ours = worst_tile
    crops = []
    for row, col in order[:6]:
        box = (slice(max(row - 1, 0) * TILE, (row + 2) * TILE), slice(max(col - 1, 0) * TILE, (col + 2) * TILE))
        a, b = we[box], ours[box]
        diff = np.clip(np.abs(a.astype(np.int16) - b) * 3, 0, 255).astype(np.uint8)
        crops.append(np.concatenate([a, b, diff], axis=1))
    width = max(c.shape[1] for c in crops)
    crops = [np.pad(c, ((0, 4), (0, width - c.shape[1]), (0, 0))) for c in crops]
    Image.fromarray(np.concatenate(crops)).save(out / "worst.png")

    report = {
        "title": title, "wallpaper": args.wallpaper, "clip": str(args.clip), "canvas": canvas, "size": size,
        "scale": scale, "t0": t0, "stock_assets": bool(args.assets), "frames": rows,
        "mean_psnr": round(float(np.mean([r["psnr"] for r in rows])), 2),
        "worst_tiles": [{"x": int(c) * TILE, "y": int(r) * TILE, "psnr": round(float(tile_psnr[r, c]), 2)} for r, c in order],
        "worst_tiles_at_time": worst_time,
        "missing_motion_tiles": [{"x": int(c) * TILE, "y": int(r) * TILE, "we": round(float(we_motion[r, c]), 1),
                                  "ours": round(float(our_motion[r, c]), 1)} for (r, c) in zip(*np.nonzero(missing))],
        "extra_motion_tiles": [{"x": int(c) * TILE, "y": int(r) * TILE, "we": round(float(we_motion[r, c]), 1),
                                "ours": round(float(our_motion[r, c]), 1)} for (r, c) in zip(*np.nonzero(extra))],
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    (out / "ours.png").unlink(missing_ok=True)
    print(f"  mean PSNR {report['mean_psnr']}, {len(report['missing_motion_tiles'])} tiles move only in WE, "
          f"{len(report['extra_motion_tiles'])} only in ours -> {out}")


if __name__ == "__main__":
    sys.exit(main())
