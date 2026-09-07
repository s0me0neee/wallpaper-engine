"""Unpacker for Wallpaper Engine `.pkg` scene archives.

Layout (little-endian throughout):

    int32   len(version)      # e.g. 8
    bytes   version           # e.g. b"PKGV0023"
    int32   entry_count
    repeat entry_count times:
        int32 len(path)
        bytes path            # utf-8, '/' separated
        int32 offset          # relative to the start of the data blob
        int32 length
    bytes   data_blob         # entries are slices of this

Some very old packages have no version string; there the first int32 is the
entry count itself. We detect that by checking the string for a "PKGV" prefix.
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path


class PkgError(Exception):
    pass


@dataclass(frozen=True)
class Entry:
    path: str
    offset: int
    length: int


class Reader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self.pos = 0

    def i32(self) -> int:
        if self.pos + 4 > len(self._data):
            raise PkgError(f"unexpected end of file at {self.pos}")
        (value,) = struct.unpack_from("<i", self._data, self.pos)
        self.pos += 4
        return value

    def string(self) -> str:
        length = self.i32()
        if not 0 <= length <= 4096 or self.pos + length > len(self._data):
            raise PkgError(f"implausible string length {length} at {self.pos - 4}")
        raw = self._data[self.pos : self.pos + length]
        self.pos += length
        return raw.decode("utf-8")


def read_header(data: bytes) -> tuple[str, list[Entry], int]:
    """Return (version, entries, data_blob_start)."""
    reader = Reader(data)
    version = reader.string()

    if version.startswith("PKGV"):
        entry_count = reader.i32()
    else:
        # No version string: what we just read was the first entry's path, so
        # rewind and treat the leading int32 as the entry count.
        reader.pos = 0
        version = ""
        entry_count = reader.i32()

    if not 0 <= entry_count <= 1_000_000:
        raise PkgError(f"implausible entry count {entry_count}")

    entries = []
    for index in range(entry_count):
        path = reader.string()
        offset = reader.i32()
        length = reader.i32()
        if offset < 0 or length < 0:
            raise PkgError(f"entry {index} ({path}) has negative offset/length")
        entries.append(Entry(path, offset, length))

    return version, entries, reader.pos


def sanitize(path: str, root: Path) -> Path:
    """Resolve an archive path under `root`, refusing traversal outside it."""
    target = (root / path.replace("\\", "/")).resolve()
    if not target.is_relative_to(root.resolve()):
        raise PkgError(f"entry escapes output directory: {path!r}")
    return target


def unpack(pkg: Path, out_dir: Path, *, list_only: bool = False) -> None:
    data = pkg.read_bytes()
    version, entries, blob_start = read_header(data)

    label = version or "<no version>"
    print(f"{pkg}: {label}, {len(entries)} entries, data blob at 0x{blob_start:x}")

    for entry in sorted(entries, key=lambda e: e.offset):
        start = blob_start + entry.offset
        end = start + entry.length
        if end > len(data):
            raise PkgError(
                f"entry {entry.path!r} runs past end of file "
                f"({end} > {len(data)})"
            )

        print(f"  {entry.length:>10,}  {entry.path}")
        if list_only:
            continue

        target = sanitize(entry.path, out_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data[start:end])

    if not list_only:
        print(f"\nextracted to {out_dir}/")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pkg", nargs="?", default="papers/scene_example/scene.pkg", type=Path)
    parser.add_argument("-o", "--out", default=Path("unpacked"), type=Path)
    parser.add_argument(
        "-l", "--list", action="store_true", help="list contents without extracting"
    )
    args = parser.parse_args(argv)

    try:
        unpack(args.pkg, args.out, list_only=args.list)
    except (PkgError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
