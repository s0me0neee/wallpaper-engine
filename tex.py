"""Decoder for Wallpaper Engine `.tex` textures.

A .tex is a container of containers, all little-endian:

    "TEXV0005\\0"                    format version
    "TEXI0001\\0"                    image header:
        int32 format                 TexFormat (see Format below)
        int32 flags                  1=NoInterpolation 2=ClampUVs 4=IsGif
        int32 texture_width          padded/power-of-two size
        int32 texture_height
        int32 image_width            actual visible size
        int32 image_height
        uint32 dominant_color        ARGB
    "TEXB000n\\0"                    image container:
        int32 image_count
        int32 free_image_format      FreeImage FIF_* enum; -1 = raw pixels
        repeat image_count times:
            int32 unknown            always 0 in samples seen
            int32 mipmap_count
            repeat mipmap_count times:
                int32 width
                int32 height
                int32 lz4_compressed        \\
                int32 decompressed_size     |- containers v2+ only
                int32 byte_count
                bytes data

`free_image_format` is the crux: when it is not -1 each mipmap payload is a
complete encoded image file (PNG/JPEG/...) that we can write straight to disk.
When it is -1 the payload is raw pixel data in `format`, optionally wrapped in
an LZ4 block, which we decode and re-encode as PNG.

Pure stdlib: LZ4 block decompression, BC1/BC2/BC3 (DXT1/3/5) decoding and PNG
encoding are all implemented here so the tool has no install step.
"""

from __future__ import annotations

import argparse
import struct
import sys
import zlib
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path


class TexError(Exception):
    pass


class Format(IntEnum):
    RGBA8888 = 0
    DXT5 = 4
    DXT3 = 6
    DXT1 = 7
    RG88 = 8
    R8 = 9


# FreeImage FIF_* values we may meet as embedded files, mapped to an extension.
FREE_IMAGE_EXT = {
    0: "bmp",
    1: "ico",
    2: "jpg",
    3: "jng",
    10: "pcx",
    13: "png",
    17: "tga",
    18: "tiff",
    19: "wbmp",
    20: "psd",
    25: "gif",
    26: "hdr",
    32: "jp2",
}


# --------------------------------------------------------------------------
# LZ4 block decompression
# --------------------------------------------------------------------------


def lz4_decompress(src: bytes, expected: int) -> bytearray:
    """Decode a raw LZ4 block (no frame header) of known output size."""
    dst = bytearray()
    pos = 0
    end = len(src)

    while pos < end:
        token = src[pos]
        pos += 1

        literal_len = token >> 4
        if literal_len == 15:
            while True:
                byte = src[pos]
                pos += 1
                literal_len += byte
                if byte != 255:
                    break

        dst += src[pos : pos + literal_len]
        pos += literal_len

        # The final sequence stops after its literals, with no match.
        if pos >= end:
            break

        offset = src[pos] | (src[pos + 1] << 8)
        pos += 2
        if offset == 0:
            raise TexError("corrupt LZ4 stream: zero match offset")

        match_len = token & 0x0F
        if match_len == 15:
            while True:
                byte = src[pos]
                pos += 1
                match_len += byte
                if byte != 255:
                    break
        match_len += 4  # minmatch

        start = len(dst) - offset
        if start < 0:
            raise TexError("corrupt LZ4 stream: match before start of output")

        if offset >= match_len:
            # Non-overlapping: copy the whole run at once.
            dst += dst[start : start + match_len]
        else:
            # Overlapping run (RLE-style); must copy byte by byte.
            for index in range(match_len):
                dst.append(dst[start + index])

    if len(dst) != expected:
        raise TexError(f"LZ4 output was {len(dst)} bytes, expected {expected}")
    return dst


# --------------------------------------------------------------------------
# Block-compressed texture decoding (BC1/BC2/BC3)
# --------------------------------------------------------------------------


def _rgb565(value: int) -> tuple[int, int, int]:
    red = (value >> 11) & 0x1F
    green = (value >> 5) & 0x3F
    blue = value & 0x1F
    return (red << 3) | (red >> 2), (green << 2) | (green >> 4), (blue << 3) | (blue >> 2)


def _color_block(data: bytes, offset: int, opaque_only: bool) -> list[tuple[int, ...]]:
    """Decode one BC1 colour block into 16 RGBA tuples."""
    c0, c1, bits = struct.unpack_from("<HHI", data, offset)
    r0, g0, b0 = _rgb565(c0)
    r1, g1, b1 = _rgb565(c1)

    palette = [(r0, g0, b0, 255), (r1, g1, b1, 255)]
    if c0 > c1 or opaque_only:
        palette.append(((2 * r0 + r1) // 3, (2 * g0 + g1) // 3, (2 * b0 + b1) // 3, 255))
        palette.append(((r0 + 2 * r1) // 3, (g0 + 2 * g1) // 3, (b0 + 2 * b1) // 3, 255))
    else:
        palette.append(((r0 + r1) // 2, (g0 + g1) // 2, (b0 + b1) // 2, 255))
        palette.append((0, 0, 0, 0))  # transparent black

    return [palette[(bits >> (2 * i)) & 0x3] for i in range(16)]


def _alpha_block_bc3(data: bytes, offset: int) -> list[int]:
    """Decode one BC3 (DXT5) interpolated-alpha block into 16 alpha values."""
    a0, a1 = data[offset], data[offset + 1]
    alphas = [a0, a1]
    if a0 > a1:
        alphas += [((7 - i) * a0 + (1 + i) * a1) // 7 for i in range(6)]
    else:
        alphas += [((5 - i) * a0 + (1 + i) * a1) // 5 for i in range(4)]
        alphas += [0, 255]

    # 16 three-bit indices packed into the next 6 bytes.
    packed = int.from_bytes(data[offset + 2 : offset + 8], "little")
    return [alphas[(packed >> (3 * i)) & 0x7] for i in range(16)]


def decode_dxt(data: bytes, width: int, height: int, fmt: Format) -> bytearray:
    """Decode BC1/BC2/BC3 data to a tightly packed RGBA8888 buffer."""
    block_size = 8 if fmt is Format.DXT1 else 16
    blocks_x = (width + 3) // 4
    blocks_y = (height + 3) // 4

    needed = blocks_x * blocks_y * block_size
    if len(data) < needed:
        raise TexError(f"{fmt.name} needs {needed} bytes, got {len(data)}")

    out = bytearray(width * height * 4)

    for by in range(blocks_y):
        for bx in range(blocks_x):
            offset = (by * blocks_x + bx) * block_size

            if fmt is Format.DXT1:
                texels = _color_block(data, offset, opaque_only=False)
            elif fmt is Format.DXT3:
                # 16 explicit 4-bit alphas, then the colour block.
                raw = int.from_bytes(data[offset : offset + 8], "little")
                colors = _color_block(data, offset + 8, opaque_only=True)
                texels = [
                    (*colors[i][:3], ((raw >> (4 * i)) & 0xF) * 17) for i in range(16)
                ]
            else:  # DXT5
                alphas = _alpha_block_bc3(data, offset)
                colors = _color_block(data, offset + 8, opaque_only=True)
                texels = [(*colors[i][:3], alphas[i]) for i in range(16)]

            for i, texel in enumerate(texels):
                x = bx * 4 + (i % 4)
                y = by * 4 + (i // 4)
                if x >= width or y >= height:
                    continue  # padding texel in an edge block
                pos = (y * width + x) * 4
                out[pos : pos + 4] = bytes(texel)

    return out


# --------------------------------------------------------------------------
# Minimal PNG writer
# --------------------------------------------------------------------------

# PNG colour type -> bytes per pixel, for the formats we emit.
_COLOR_TYPE_CHANNELS = {0: 1, 4: 2, 6: 4}


def write_png(path: Path, pixels: bytes, width: int, height: int, color_type: int) -> None:
    channels = _COLOR_TYPE_CHANNELS[color_type]
    stride = width * channels
    if len(pixels) < stride * height:
        raise TexError(
            f"pixel buffer too small: {len(pixels)} < {stride * height} "
            f"for {width}x{height}"
        )

    # Filter type 0 (None) in front of every scanline.
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw += pixels[y * stride : (y + 1) * stride]

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


# --------------------------------------------------------------------------
# Container parsing
# --------------------------------------------------------------------------


@dataclass
class Mipmap:
    width: int
    height: int
    data: bytes
    lz4_compressed: bool
    decompressed_size: int


@dataclass
class Tex:
    version: str
    container_version: str
    format: Format | int
    flags: int
    texture_width: int
    texture_height: int
    image_width: int
    image_height: int
    dominant_color: int
    free_image_format: int
    images: list[list[Mipmap]]
    bytes_consumed: int

    @property
    def is_gif(self) -> bool:
        return bool(self.flags & 4)

    @property
    def embedded_ext(self) -> str | None:
        """Extension of the embedded file format, or None for raw pixels."""
        if self.free_image_format < 0:
            return None
        return FREE_IMAGE_EXT.get(self.free_image_format, "bin")


class Reader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self.pos = 0

    def i32(self) -> int:
        (value,) = struct.unpack_from("<i", self._data, self.pos)
        self.pos += 4
        return value

    def u32(self) -> int:
        (value,) = struct.unpack_from("<I", self._data, self.pos)
        self.pos += 4
        return value

    def magic(self) -> str:
        """Read a NUL-terminated magic string (8 chars + NUL in practice)."""
        end = self._data.index(b"\0", self.pos)
        value = self._data[self.pos : end].decode("ascii")
        self.pos = end + 1
        return value

    def take(self, count: int) -> bytes:
        if self.pos + count > len(self._data):
            raise TexError(f"truncated: want {count} bytes at {self.pos}")
        value = self._data[self.pos : self.pos + count]
        self.pos += count
        return value


def parse(data: bytes) -> Tex:
    reader = Reader(data)

    version = reader.magic()
    if not version.startswith("TEXV"):
        raise TexError(f"not a .tex file (magic {version!r})")

    image_magic = reader.magic()
    if not image_magic.startswith("TEXI"):
        raise TexError(f"expected TEXI header, got {image_magic!r}")

    raw_format = reader.i32()
    try:
        fmt: Format | int = Format(raw_format)
    except ValueError:
        fmt = raw_format

    flags = reader.i32()
    texture_width = reader.i32()
    texture_height = reader.i32()
    image_width = reader.i32()
    image_height = reader.i32()
    dominant_color = reader.u32()

    container = reader.magic()
    if not container.startswith("TEXB"):
        raise TexError(f"expected TEXB container, got {container!r}")
    container_version = int(container[4:])

    image_count = reader.i32()

    free_image_format = -1
    if container_version >= 4:
        free_image_format = reader.i32()

    images: list[list[Mipmap]] = []
    for _ in range(image_count):
        # Purpose unknown; 0 in every sample inspected. Both samples have a
        # single image, so it could equally be a one-off field before the loop.
        reader.i32()

        mipmap_count = reader.i32()
        if not 0 <= mipmap_count <= 64:
            raise TexError(f"implausible mipmap count {mipmap_count}")

        mipmaps = []
        for _ in range(mipmap_count):
            width = reader.i32()
            height = reader.i32()

            if container_version >= 2:
                lz4_compressed = bool(reader.i32())
                decompressed_size = reader.i32()
            else:
                lz4_compressed = False
                decompressed_size = 0

            byte_count = reader.i32()
            payload = reader.take(byte_count)
            mipmaps.append(
                Mipmap(width, height, payload, lz4_compressed, decompressed_size)
            )
        images.append(mipmaps)

    return Tex(
        version=version,
        container_version=container,
        format=fmt,
        flags=flags,
        texture_width=texture_width,
        texture_height=texture_height,
        image_width=image_width,
        image_height=image_height,
        dominant_color=dominant_color,
        free_image_format=free_image_format,
        images=images,
        bytes_consumed=reader.pos,
    )


def mipmap_pixels(tex: Tex, mipmap: Mipmap) -> bytes:
    """Return decompressed payload bytes for a mipmap."""
    if not mipmap.lz4_compressed:
        return mipmap.data
    return bytes(lz4_decompress(mipmap.data, mipmap.decompressed_size))


def save_mipmap(tex: Tex, mipmap: Mipmap, stem: Path) -> Path:
    """Write one mipmap to disk, returning the path actually written."""
    payload = mipmap_pixels(tex, mipmap)

    # Already an encoded image file: pass it straight through untouched.
    ext = tex.embedded_ext
    if ext is not None:
        target = stem.with_suffix(f".{ext}")
        target.write_bytes(payload)
        return target

    target = stem.with_suffix(".png")
    width, height = mipmap.width, mipmap.height

    if tex.format is Format.RGBA8888:
        write_png(target, payload, width, height, color_type=6)
    elif tex.format is Format.RG88:
        # Two channels; greyscale+alpha is a byte-exact PNG match.
        write_png(target, payload, width, height, color_type=4)
    elif tex.format is Format.R8:
        write_png(target, payload, width, height, color_type=0)
    elif tex.format in (Format.DXT1, Format.DXT3, Format.DXT5):
        rgba = decode_dxt(payload, width, height, tex.format)
        write_png(target, bytes(rgba), width, height, color_type=6)
    else:
        raise TexError(f"unsupported pixel format {tex.format!r}")

    return target


def describe(path: Path, tex: Tex, total: int) -> None:
    fmt = tex.format.name if isinstance(tex.format, Format) else f"unknown({tex.format})"
    embedded = tex.embedded_ext or "raw pixels"
    flags = []
    if tex.flags & 1:
        flags.append("NoInterpolation")
    if tex.flags & 2:
        flags.append("ClampUVs")
    if tex.flags & 4:
        flags.append("IsGif")

    print(f"{path}")
    print(f"  {tex.version} / {tex.container_version}   format={fmt}   payload={embedded}")
    print(f"  image {tex.image_width}x{tex.image_height}  texture {tex.texture_width}x{tex.texture_height}")
    print(f"  flags={tex.flags} [{', '.join(flags) or 'none'}]  dominant=#{tex.dominant_color:08x}")

    for image_index, mipmaps in enumerate(tex.images):
        for level, mipmap in enumerate(mipmaps):
            kind = "lz4" if mipmap.lz4_compressed else "store"
            size = (
                f"{mipmap.decompressed_size:,}"
                if mipmap.lz4_compressed
                else f"{len(mipmap.data):,}"
            )
            print(
                f"    image{image_index} mip{level}  {mipmap.width:>5}x{mipmap.height:<5}"
                f"  {kind:>5}  {len(mipmap.data):>12,} -> {size:>12} bytes"
            )

    trailing = total - tex.bytes_consumed
    note = "exact" if trailing == 0 else f"{trailing:,} trailing bytes (sprite/gif data?)"
    print(f"  consumed {tex.bytes_consumed:,} of {total:,} bytes: {note}")


def convert(path: Path, out_dir: Path, *, all_mipmaps: bool, info_only: bool) -> None:
    data = path.read_bytes()
    tex = parse(data)
    describe(path, tex, len(data))

    if info_only:
        print()
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for image_index, mipmaps in enumerate(tex.images):
        selected = list(enumerate(mipmaps)) if all_mipmaps else [(0, mipmaps[0])]
        for level, mipmap in selected:
            suffix = ""
            if len(tex.images) > 1:
                suffix += f"_img{image_index}"
            if all_mipmaps and len(mipmaps) > 1:
                suffix += f"_mip{level}"
            written = save_mipmap(tex, mipmap, out_dir / (path.stem + suffix))
            print(f"    -> {written}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths",
        nargs="*",
        default=[Path("unpacked")],
        type=Path,
        help="tex files, or directories to search recursively (default: unpacked/)",
    )
    parser.add_argument("-o", "--out", default=Path("textures"), type=Path)
    parser.add_argument("-i", "--info", action="store_true", help="describe only")
    parser.add_argument(
        "-a", "--all-mipmaps", action="store_true", help="export every mip level"
    )
    args = parser.parse_args(argv)

    targets: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            targets += sorted(path.rglob("*.tex"))
        else:
            targets.append(path)

    if not targets:
        print("no .tex files found", file=sys.stderr)
        return 1

    failures = 0
    for target in targets:
        try:
            convert(target, args.out, all_mipmaps=args.all_mipmaps, info_only=args.info)
        except (TexError, OSError, struct.error) as error:
            print(f"error: {target}: {error}", file=sys.stderr)
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
