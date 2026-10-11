#!/usr/bin/env python3
"""Carve the embedded miocr models out of libmiocr.so.

Model extents are found by bisecting the smallest buffer length that MNN still accepts
(MNN's flatbuffer verifier is size sensitive), which coincides with the true model size.

Outputs into <outdir>: recognizer.mnn, orientation.mnn, charset.txt, models.json
"""
import json
import os
import pathlib
import shutil
import sys
import tempfile

import MNN

SO = (sys.argv[1] if len(sys.argv) > 1
      else "/home/asdf/Downloads/poco-f4-gallery-ocr-re/gallery/lib/arm64-v8a/libmiocr.so")
OUT = sys.argv[2] if len(sys.argv) > 2 else "models"
data = pathlib.Path(SO).read_bytes()
os.makedirs(OUT, exist_ok=True)
tmp = tempfile.mkdtemp(prefix="carve")
probe = os.path.join(tmp, "probe.mnn")


def loads(start, end):
    with open(probe, "wb") as fh:
        fh.write(data[start:end])
    try:
        MNN.Interpreter(probe)
        return True
    except Exception:
        return False


def bisect_end(start, lo, hi):
    """Smallest end >= lo that MNN accepts."""
    while lo + 4 < hi:
        mid = ((lo + hi) // 2) & ~3
        if loads(start, mid):
            hi = mid
        else:
            lo = mid
    return hi


def report(path):
    interp = MNN.Interpreter(path)
    sess = interp.createSession({})
    ins = {k: tuple(v.getShape()) for k, v in interp.getSessionInputAll(sess).items()}
    outs = {k: tuple(v.getShape()) for k, v in interp.getSessionOutputAll(sess).items()}
    return ins, outs


models = {
    "recognizer": {"start": 0x4539C8, "end_guess": 0x7344F4},
    "orientation": {"start": 0x73BDD8, "end_guess": 0x740000},
}
for name, m in models.items():
    end = bisect_end(m["start"], m["start"] + 4096, m["end_guess"] + 0x300000)
    m["end"] = end
    m["size"] = end - m["start"]
    path = os.path.join(OUT, f"{name}.mnn")
    with open(path, "wb") as fh:
        fh.write(data[m["start"]:end])
    ins, outs = report(path)
    m["inputs"], m["outputs"] = ins, outs
    print(f"{name}: 0x{m['start']:x}..0x{end:x} size={m['size']:,} in={ins} out={outs}")

# charset: NUL-separated UTF-8 entries, starting right after the recognizer model
cstart = models["recognizer"]["end"]
entries = []
p = cstart
while p < len(data):
    z = data.find(b"\x00", p)
    if z < 0 or z == p:
        break
    try:
        s = data[p:z].decode("utf-8")
    except UnicodeDecodeError:
        break
    if len(s) > 8 or "\n" in s:
        break
    entries.append(s)
    p = z + 1
with open(os.path.join(OUT, "charset.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(entries))
print(f"charset: 0x{cstart:x}..0x{p:x} entries={len(entries)} (written to charset.txt)")

with open(os.path.join(OUT, "models.json"), "w") as fh:
    json.dump({"source": SO, "charset": {"start": cstart, "end": p, "entries": len(entries)},
               "models": models}, fh, indent=2)
shutil.rmtree(tmp, ignore_errors=True)      # probe dir is scratch
print("wrote models.json")
