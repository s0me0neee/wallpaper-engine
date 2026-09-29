#!/usr/bin/env python3
"""Record native Wallpaper Engine, run under Proton, as a reference video.

    tools/we_capture.py papers/scene_example2 2862539849 --seconds 12

Each wallpaper (a workshop id or a wallpaper directory) gets a fresh WE process
and a recording, <out>/<id>.mkv: x264 at `--crf` (0 is lossless, ~1 GB per 4K
clip; the default 18 at 1080p45 keeps compression noise far below the
differences a comparison looks for). WE's own `g_Time` starts at its first
presented frame, which `we_compare.py` finds in the clip.

WE plays in its `-playInWindow` mode on a headless Hyprland output, so nothing
shows on a real screen. The window is the scene's canvas, fitted inside `--max`
with its aspect kept, so WE draws the whole canvas natively and nothing is
cropped or upscaled relative to `simulate`.

A scene with `cameraparallax` follows the cursor, and Wine reads the pointer
from XWayland, which only updates while the pointer is over an X window. So
once WE's window is up the real cursor visits its centre for ~0.15 s and goes
straight back: WE keeps that position, the parallax at rest, which is what
`simulate` renders. The output gets a named workspace declared before it
exists: left to itself it claims the next numbered one, and switching to that
number then lands on a monitor nobody can see. Every change to the compositor
is undone on exit, by `hyprctl reload`.

One-time setup, both done by running WE through Steam once, or by hand:
  - `launcher.exe` must have installed `distribution/` into the install root;
    run from `distribution/` directly, WE cannot find its `assets/` and renders
    flat grey (or crashes in scenescript64 on a scripted scene).
  - WE's `config.json` needs `hasshownwelcomedialog: true`: the welcome dialog
    is a CEF window, and CEF crashes under Wine before it draws.
Needs: Steam running, Proton, wf-recorder, ffprobe.
"""

import argparse
import json
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

STEAM = Path.home() / ".local/share/Steam"
WE = STEAM / "steamapps/common/wallpaper_engine"
PREFIX = STEAM / "steamapps/compatdata/431960"
WORKSHOP = STEAM / "steamapps/workshop/content/431960"
OUTPUT = "WECAP"
TITLE = "wecap"


def hypr(*args):
    return subprocess.run(["hyprctl", *args], check=True, capture_output=True, text=True).stdout


def hypr_json(what):
    return json.loads(hypr("-j", what))


def wine_path(path):
    return "Z:" + str(Path(path).resolve()).replace("/", "\\")


def resolve_wallpaper(arg):
    path = Path(arg)
    if not path.exists():
        path = WORKSHOP / arg
    project = path.resolve() / "project.json"
    if not project.exists():
        sys.exit(f"{arg}: no project.json (not a wallpaper directory or workshop id)")
    return project.parent.name, project


def read_pkg(path):
    """A scene.pkg's files by path (see src/pkg.rs for the layout)."""
    data = path.read_bytes()
    pos = 0

    def i32():
        nonlocal pos
        pos += 4
        return struct.unpack_from("<i", data, pos - 4)[0]

    def string():
        nonlocal pos
        length = i32()
        pos += length
        return data[pos - length:pos].decode("utf-8", "replace")

    if not string().startswith("PKGV"):
        pos = 0
    entries = [(string(), i32(), i32()) for _ in range(i32())]
    return {name: data[pos + offset:pos + offset + length] for name, offset, length in entries}


def scene_general(project):
    """`scene.json`'s `general` block, or {} for a scene that ships it unpacked or not at all."""
    pkg = project.parent / "scene.pkg"
    scene = read_pkg(pkg).get("scene.json") if pkg.exists() else None
    if scene is None and (project.parent / "scene.json").exists():
        scene = (project.parent / "scene.json").read_bytes()
    try:
        return json.loads(scene or b"{}").get("general", {})
    except ValueError:
        return {}


def user_value(value):
    # A user-overridable setting is stored as {"user": ..., "value": ...}.
    return value.get("value") if isinstance(value, dict) else value


def window_size(general, limit):
    ortho = general.get("orthogonalprojection")
    if not isinstance(ortho, dict) or not ortho.get("width") or not ortho.get("height"):
        return limit
    width, height = ortho["width"], ortho["height"]
    scale = min(1.0, limit[0] / width, limit[1] / height)
    return round(width * scale / 2) * 2, round(height * scale / 2) * 2


def proton_env():
    return {
        **os.environ,
        "STEAM_COMPAT_CLIENT_INSTALL_PATH": str(STEAM),
        "STEAM_COMPAT_DATA_PATH": str(PREFIX),
        "SteamAppId": "431960",
        "SteamGameId": "431960",
    }


def stop_we(proton):
    wineserver = STEAM / "steamapps/common" / proton / "files/bin/wineserver"
    subprocess.run([wineserver, "-k"], env={**os.environ, "WINEPREFIX": str(PREFIX / "pfx")}, capture_output=True)
    for _ in range(40):
        if subprocess.run(["pgrep", "-f", r"wallpaper(32|64)\.exe|wineserver"], capture_output=True).returncode:
            return
        time.sleep(0.5)


def output_left():
    # Right of every real monitor in X11 pixels too: XWayland is zero-scaled, so a scaled
    # monitor is wider there than in layout space, and an overlap misplaces Wine's window.
    return max(m["x"] + m["width"] for m in hypr_json("monitors") if m["name"] != OUTPUT)


def setup_output():
    hypr(
        "--batch",
        f"keyword workspace name:{TITLE}, monitor:{OUTPUT}, default:true, persistent:true ; "
        f"keyword windowrule match:class steam_app_431960, workspace name:{TITLE} silent ; "
        "keyword windowrule match:class steam_app_431960, match:title ^$, workspace special:wehide silent ; "
        f"keyword windowrule match:title {TITLE}, fullscreen on",
    )
    if not any(m["name"] == OUTPUT for m in hypr_json("monitors all")):
        hypr("output", "create", "headless", OUTPUT)


def size_output(width, height, fps):
    hypr("keyword", "monitor", f"{OUTPUT},{width}x{height}@{fps},{output_left()}x0,1")


def park_cursor_on(width, height, at):
    x, y = (int(float(v)) for v in hypr("cursorpos").replace(" ", "").split(","))
    hypr("dispatch", "movecursor", str(output_left() + round(width * at[0])), str(round(height * at[1])))
    time.sleep(0.15)
    hypr("dispatch", "movecursor", str(x), str(y))


def teardown_output():
    subprocess.run(["hyprctl", "output", "remove", OUTPUT], capture_output=True)
    subprocess.run(["hyprctl", "reload"], capture_output=True)


def wait_for_window(title, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        titles = {c["title"] for c in hypr_json("clients")}
        if title in titles or "Error" in titles:
            return titles
        time.sleep(0.25)
    return set()


def capture(wallpaper, args):
    ident, project = resolve_wallpaper(wallpaper)
    general = scene_general(project)
    width, height = window_size(general, args.max)
    stop_we(args.proton)
    size_output(width, height, args.fps)
    log = open(args.out / f"{ident}.we.log", "w")
    subprocess.Popen(
        [STEAM / "steamapps/common" / args.proton / "proton", "run", WE / "wallpaper64.exe",
         "-control", "openWallpaper", "-file", wine_path(project),
         "-playInWindow", TITLE, "-width", str(width), "-height", str(height)],
        cwd=WE, env=proton_env(), stdout=log, stderr=log, start_new_session=True,
    )
    titles = wait_for_window(TITLE, 90)
    if "Error" in titles or TITLE not in titles:
        print(f"{ident}: WE {'showed an Error dialog' if 'Error' in titles else 'opened no window'}", flush=True)
        return
    if user_value(general.get("cameraparallax")):
        park_cursor_on(width, height, args.cursor)
    clip = args.out / f"{ident}.mkv"
    with open(args.out / f"{ident}.wfr.log", "w") as wfr_log:
        subprocess.run(
            ["timeout", "--signal=INT", str(args.seconds), "wf-recorder", "-y", "-o", OUTPUT, "-r", str(args.fps),
             "-c", "libx264rgb", "-p", f"crf={args.crf}", "-p", "preset=veryfast", "-f", str(clip)],
            stdout=wfr_log, stderr=wfr_log,
        )
    if "Error" in {c["title"] for c in hypr_json("clients")}:
        print(f"{ident}: WE showed an Error dialog while recording", flush=True)
    frames = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", clip],
        capture_output=True, text=True,
    ).stdout.strip()
    print(f"{ident}: {frames} frames at {width}x{height} -> {clip}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("wallpapers", nargs="+", help="workshop ids or wallpaper directories")
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--fps", type=int, default=45)
    parser.add_argument("--crf", type=int, default=18, help="x264 quality, 0 for lossless")
    parser.add_argument("--max", default="1920x1080", help="largest window, WxH; the canvas is fitted inside it")
    parser.add_argument("--out", type=Path, default=Path("papers/_we_captures/proton_1080p"))
    parser.add_argument("--cursor", default="0.5,0.5",
                        help="where a parallax scene's cursor is parked, as fractions of the window")
    parser.add_argument("--proton", default="Proton 11.0", help="directory under steamapps/common")
    args = parser.parse_args()

    if not (WE / "wallpaper64.exe").exists():
        sys.exit(f"{WE}/wallpaper64.exe missing: run WE's launcher.exe once so it installs distribution/")
    args.max = tuple(int(v) for v in args.max.split("x"))
    args.cursor = tuple(float(v) for v in args.cursor.split(","))
    args.out.mkdir(parents=True, exist_ok=True)

    # SIGTERM too, so a killed capture still hands the compositor back.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    try:
        setup_output()
        for wallpaper in args.wallpapers:
            capture(wallpaper, args)
    finally:
        stop_we(args.proton)
        teardown_output()


if __name__ == "__main__":
    main()
