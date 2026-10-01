#!/usr/bin/env python3
"""Measure native Wallpaper Engine (under Proton) and our `simulate` on the same scene, side by side.

    tools/we_bench.py papers/scene_example2 3484246124 --seconds 10

Both play on the same headless Hyprland output `we_capture.py` uses, at the same size (the canvas
fitted inside `--max`) and the same frame cap, on the same GPU (`--gpu`, Intel by default: DXVK is
otherwise free to pick a discrete card, and did). Per renderer it reports:

  fps  frames presented a second (WE: Wine's own `fps` trace channel; ours: `simulate`'s fps line)
  gpu  the render engine's busy time over wall time, for that process alone (DRM fdinfo, Intel only)
  cpu  process CPU time over wall time, in cores

At a frame cap both can hold, `gpu` is the efficiency number: the same pictures for less GPU time.
Results append to `--out` as JSON lines.

Needs: Steam with Proton, a release build, Hyprland.
"""

import argparse
import glob
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import we_capture as wc  # noqa: E402

CLK_TCK = os.sysconf("SC_CLK_TCK")


def render_ns(pid):
    """The process's summed render-engine time, deduplicated by DRM client (one fd may be dup'd)."""
    clients = {}
    for path in glob.glob(f"/proc/{pid}/fdinfo/*"):
        try:
            text = open(path).read()
        except OSError:
            continue
        client = re.search(r"drm-client-id:\s*(\d+)", text)
        busy = re.search(r"drm-engine-render:\s*(\d+)", text)
        if client and busy:
            clients[client.group(1)] = int(busy.group(1))
    return sum(clients.values())


def cpu_seconds(pid):
    try:
        fields = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    except OSError:
        return 0.0
    return (int(fields[11]) + int(fields[12])) / CLK_TCK


def sample(pids, seconds):
    """GPU busy fraction and CPU cores, summed over `pids`, across `seconds` of wall time."""
    start = time.monotonic(), sum(map(render_ns, pids)), sum(map(cpu_seconds, pids))
    time.sleep(seconds)
    end = time.monotonic(), sum(map(render_ns, pids)), sum(map(cpu_seconds, pids))
    wall = end[0] - start[0]
    return (end[1] - start[1]) / 1e9 / wall, (end[2] - start[2]) / wall


def we_pids():
    out = subprocess.run(["pgrep", "-f", r"wallpaper64\.exe"], capture_output=True, text=True).stdout
    return [int(pid) for pid in out.split()]


def gpu_env(gpu):
    if gpu == "nvidia":
        return {"DXVK_FILTER_DEVICE_NAME": "NVIDIA"}, {
            "__NV_PRIME_RENDER_OFFLOAD": "1", "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
            "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"}
    return {"DXVK_FILTER_DEVICE_NAME": "Intel"}, {"__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/50_mesa.json"}


def bench_we(project, size, args, scratch):
    wc.stop_we(args.proton)
    log_path = scratch / "we.log"
    env = {**wc.proton_env(), **gpu_env(args.gpu)[0], "WINEDEBUG": "fps"}
    with open(log_path, "w") as log:
        subprocess.Popen(
            [wc.STEAM / "steamapps/common" / args.proton / "proton", "run", wc.WE / "wallpaper64.exe",
             "-control", "openWallpaper", "-file", wc.wine_path(project),
             "-playInWindow", wc.TITLE, "-width", str(size[0]), "-height", str(size[1])],
            cwd=wc.WE, env=env, stdout=log, stderr=log, start_new_session=True,
        )
    titles = wc.wait_for_window(wc.TITLE, 90)
    if wc.TITLE not in titles or "Error" in titles:
        wc.stop_we(args.proton)
        return {"error": "no window" if wc.TITLE not in titles else "error dialog"}
    time.sleep(args.settle)
    offset = log_path.stat().st_size
    gpu, cpu = sample(we_pids(), args.seconds)
    with open(log_path) as log:
        log.seek(offset)
        rates = [float(m) for m in re.findall(r"@ approx ([\d.]+)fps", log.read())]
    wc.stop_we(args.proton)
    return {"fps": statistics.median(rates) if rates else None, "gpu": gpu, "cpu": cpu}


def bench_ours(wallpaper, title, scale, fps, args):
    escaped = re.sub(r"([\\^$.|?*+()\[\]{}])", r"\\\1", title)
    wc.hypr("--batch", f"keyword windowrule match:title ^{escaped}$, workspace name:{wc.TITLE} silent ; "
                       f"keyword windowrule match:title ^{escaped}$, fullscreen on")
    command = [args.binary, "simulate", wallpaper, "--fps", str(fps)]
    if scale < 1:
        command += ["--scale", f"{scale:.6f}"]
    env = {**os.environ, **gpu_env(args.gpu)[1]}
    if args.assets:
        env["WE_ASSETS"] = str(args.assets)
    if args.profile:
        env["SIMULATE_PROFILE"] = "1"
    proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    reader = threading.Thread(target=lambda: lines.extend(iter(proc.stdout.readline, "")), daemon=True)
    reader.start()
    deadline = time.monotonic() + 120
    while not any(" fps  (" in line for line in lines):
        if proc.poll() is not None or time.monotonic() > deadline:
            proc.kill()
            return {"error": "no frames: " + "".join(lines[-5:]).strip()}
        time.sleep(0.2)
    time.sleep(args.settle)
    first = len(lines)
    gpu, cpu = sample([proc.pid], args.seconds)
    window = lines[first:]
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()
    rates = [float(m.group(1)) for line in window if (m := re.match(r"\s+([\d.]+) fps  \(", line))]
    stats = [line.strip() for line in window if " fps  (" in line]
    profile = [line.strip() for line in window if line.strip().startswith("gpu ")]
    cpu_profile = [line.strip() for line in window if line.strip().startswith("cpu by layer")]
    return {"fps": statistics.median(rates) if rates else None, "gpu": gpu, "cpu": cpu,
            "line": stats[len(stats) // 2] if stats else None,
            "profile": profile[len(profile) // 2] if profile else None,
            "cpu_profile": cpu_profile[len(cpu_profile) // 2] if cpu_profile else None}


def scene_canvas(binary, wallpaper):
    out = subprocess.run([binary, "info", wallpaper], capture_output=True, text=True, check=True).stdout
    match = re.search(r"canvas\s+(\d+)x(\d+)", out)
    return out.splitlines()[0].strip(), (int(match.group(1)), int(match.group(2))) if match else None


def fmt(result):
    if "error" in result:
        return result["error"]
    fps = f"{result['fps']:5.1f}" if result.get("fps") is not None else "    ?"
    return f"{fps} fps  gpu {result['gpu'] * 100:5.1f}%  cpu {result['cpu']:4.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("wallpapers", nargs="+", help="workshop ids or wallpaper directories")
    parser.add_argument("--seconds", type=float, default=10, help="how long each renderer is measured")
    parser.add_argument("--settle", type=float, default=4, help="seconds to let each renderer warm up first")
    parser.add_argument("--max", default="1920x1080", help="largest window, WxH; the canvas is fitted inside it")
    parser.add_argument("--fps", type=int, default=60, help="output refresh rate and our cap (WE's is config.json's)")
    parser.add_argument("--uncapped", action="store_true", help="also run ours with no frame cap")
    parser.add_argument("--gpu", choices=["intel", "nvidia"], default="intel")
    parser.add_argument("--only", choices=["we", "ours"])
    parser.add_argument("--profile", action="store_true", help="print ours' GPU time per layer (SIMULATE_PROFILE)")
    parser.add_argument("--binary", default="target/release/wallpaper-engine")
    parser.add_argument("--label", default="", help="tag for this run in the JSON lines (e.g. a commit)")
    parser.add_argument("--out", type=Path, default=Path("papers/_we_captures/bench.jsonl"))
    parser.add_argument("--proton", default="Proton 11.0")
    args = parser.parse_args()

    args.max = tuple(int(v) for v in args.max.split("x"))
    assets = wc.WE / "assets"
    args.assets = assets if assets.exists() else None
    scratch = Path(os.environ.get("TMPDIR", "/tmp")) / "we_bench"
    scratch.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    try:
        wc.setup_output()
        for wallpaper in args.wallpapers:
            ident, project = wc.resolve_wallpaper(wallpaper)
            title, canvas = scene_canvas(args.binary, str(project.parent))
            if canvas is None:
                print(f"{ident}: not a scene", flush=True)
                continue
            general = wc.scene_general(project)
            size = wc.window_size(general, args.max)
            scale = size[0] / canvas[0]
            row = {"id": ident, "title": title, "size": size, "gpu_device": args.gpu, "label": args.label,
                   "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            print(f"{ident} {title}: {size[0]}x{size[1]}", flush=True)
            if args.only != "ours":
                wc.size_output(*size, args.fps)
                row["we"] = bench_we(project, size, args, scratch)
                print(f"  WE    {fmt(row['we'])}", flush=True)
            if args.only != "we":
                wc.size_output(*size, args.fps)
                row["ours"] = bench_ours(str(project.parent), title, scale, args.fps, args)
                print(f"  ours  {fmt(row['ours'])}", flush=True)
                for key in ("line", "profile", "cpu_profile"):
                    if row["ours"].get(key):
                        print(f"        {row['ours'][key]}", flush=True)
                if args.uncapped:
                    wc.size_output(*size, 240)
                    row["ours_uncapped"] = bench_ours(str(project.parent), title, scale, 0, args)
                    print(f"  ours uncapped  {fmt(row['ours_uncapped'])}", flush=True)
            with open(args.out, "a") as out:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        wc.stop_we(args.proton)
        wc.teardown_output()


if __name__ == "__main__":
    main()
