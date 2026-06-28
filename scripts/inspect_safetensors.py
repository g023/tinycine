#!/usr/bin/env python3
"""inspect_safetensors.py — torch-free safetensors header dumper.

Reads only the JSON header of a .safetensors file (the first 8 bytes give the
header length, then that many bytes of JSON). Prints tensor names, dtypes,
shapes and byte offsets, plus a dtype histogram and any __metadata__ (which for
the LTX-2.3 DiT contains the real model config.json).

Usage:
    python3 scripts/inspect_safetensors.py FILE [FILE ...] [--names] [--meta KEY]
No torch, no safetensors lib required — stdlib + json only.
"""
import json
import struct
import sys
from collections import Counter


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    h = json.loads(raw)
    meta = h.pop("__metadata__", {})
    return h, meta, n


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    show_names = "--names" in argv
    meta_key = None
    for a in argv:
        if a.startswith("--meta="):
            meta_key = a.split("=", 1)[1]
    for path in args:
        try:
            h, meta, hlen = read_header(path)
        except Exception as e:  # noqa: BLE001
            print(f"!! {path}: {e}")
            continue
        print("=" * 78)
        print(path)
        print(f"  header bytes: {hlen}   tensors: {len(h)}")
        dt = Counter(v["dtype"] for v in h.values())
        print(f"  dtypes: {dict(dt)}")
        if meta:
            mk = list(meta.keys())
            print(f"  metadata keys: {mk}")
        if meta_key and meta_key in meta:
            val = meta[meta_key]
            try:
                val = json.dumps(json.loads(val), indent=2)
            except Exception:  # noqa: BLE001
                pass
            print(f"  --- metadata['{meta_key}'] ---")
            print(val)
        if show_names:
            for nm in sorted(h):
                v = h[nm]
                print(f"    {nm}\t{v['dtype']}\t{v['shape']}")
        else:
            names = sorted(h)
            for nm in names[:8]:
                v = h[nm]
                print(f"    {nm}\t{v['dtype']}\t{v['shape']}")
            if len(names) > 8:
                print(f"    ... ({len(names)} tensors total; --names to list all)")


if __name__ == "__main__":
    main(sys.argv[1:])
