#!/usr/bin/env python3
"""contact_sheet.py - QA helpers for the LTX-2.3 quality-tuning A/Bs.

Two modes:
  midframe  <in.mp4> <out.png> [--label TEXT]   extract the middle frame (PNG), optional caption
  sheet     <out.png> <in1.mp4[:LABEL]> ...      tile middle frames side-by-side with captions

No torch needed; uses PyAV (already installed) + PIL.
"""
import argparse
import os
import sys

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_frames(path):
    container = av.open(path)
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    return frames


def mid_frame(path):
    frames = read_frames(path)
    if not frames:
        raise ValueError(f"no frames in {path}")
    return Image.fromarray(frames[len(frames) // 2]), len(frames)


def _font(size=16):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.isfile(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def caption(img, text):
    bar_h = 22
    out = Image.new("RGB", (img.width, img.height + bar_h), (0, 0, 0))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    d.text((4, img.height + 3), text, fill=(255, 255, 255), font=_font(14))
    return out


def cmd_midframe(a):
    img, n = mid_frame(a.input)
    if a.label:
        img = caption(img, a.label)
    img.save(a.output)
    print(f"wrote {a.output} (mid of {n} frames, {img.width}x{img.height})")


def cmd_sheet(a):
    tiles = []
    maxh = 0
    for spec in a.inputs:
        path, _, label = spec.partition(":")
        img, n = mid_frame(path)
        label = label or os.path.basename(path)
        img = caption(img, f"{label} [{img.width}x{img.height} {n}f]")
        tiles.append(img)
        maxh = max(maxh, img.height)
    pad = 6
    total_w = sum(t.width for t in tiles) + pad * (len(tiles) + 1)
    sheet = Image.new("RGB", (total_w, maxh + 2 * pad), (32, 32, 32))
    x = pad
    for t in tiles:
        sheet.paste(t, (x, pad))
        x += t.width + pad
    sheet.save(a.output)
    print(f"wrote {a.output} ({len(tiles)} tiles, {sheet.width}x{sheet.height})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("midframe"); m.add_argument("input"); m.add_argument("output")
    m.add_argument("--label", default=None); m.set_defaults(fn=cmd_midframe)
    s = sub.add_parser("sheet"); s.add_argument("output"); s.add_argument("inputs", nargs="+")
    s.set_defaults(fn=cmd_sheet)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
