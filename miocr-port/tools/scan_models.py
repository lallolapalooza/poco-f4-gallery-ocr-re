#!/usr/bin/env python3
"""Scan a byte range of a .so for MNN-loadable models, with a cheap flatbuffer pre-filter.

Usage: scan_models.py <so> <lo> <hi> [end] [max_tries]
"""
import os
import pathlib
import struct
import sys
import tempfile

import MNN

SO = sys.argv[1]
LO = int(sys.argv[2], 0)
HI = int(sys.argv[3], 0)
END = int(sys.argv[4], 0) if len(sys.argv) > 4 else 0
MAXT = int(sys.argv[5]) if len(sys.argv) > 5 else 3000

data = pathlib.Path(SO).read_bytes()
end = END or len(data)
tmp = tempfile.mkdtemp(prefix="scan")
probe = os.path.join(tmp, "p.mnn")


def plausible(o):
    """Cheap flatbuffer root sanity check (root offset + vtable shape)."""
    if o + 8 > end:
        return False
    root = struct.unpack_from("<I", data, o)[0]
    if root < 4 or root > 1 << 24 or root % 4:
        return False
    p = o + root
    if p + 8 > end or p % 4:
        return False
    soff = struct.unpack_from("<i", data, p)[0]
    vt = p - soff
    if vt < o or vt + 4 > end or vt % 2:
        return False
    vsz, tsz = struct.unpack_from("<HH", data, vt)
    if vsz < 6 or vsz > 512 or vsz % 2 or tsz < 4 or tsz > 4096:
        return False
    if vt + vsz > end or p + tsz > end:
        return False
    for i in range(4, vsz, 2):
        if struct.unpack_from("<H", data, vt + i)[0] >= tsz + 8:
            return False
    return True


tries = 0
hits = []
for o in range(LO - LO % 4, HI, 4):
    if not plausible(o):
        continue
    tries += 1
    if tries > MAXT:
        print("max tries reached")
        break
    with open(probe, "wb") as fh:
        fh.write(data[o:end])
    try:
        interp = MNN.Interpreter(probe)
    except Exception:
        continue
    sess = interp.createSession({})
    ins = {k: tuple(v.getShape()) for k, v in interp.getSessionInputAll(sess).items()}
    outs = {k: tuple(v.getShape()) for k, v in interp.getSessionOutputAll(sess).items()}
    print(f"*** LOADED 0x{o:x} (file 0x{o:x}) in={ins} out={outs}")
    hits.append(o)

print(f"scanned {tries} candidates, models: {[hex(h) for h in hits]}")
