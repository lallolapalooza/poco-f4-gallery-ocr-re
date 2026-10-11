#!/usr/bin/env python3
"""Minimal flatbuffers reader + MNN Net walker for carving models out of libmiocr.so.

Field orders follow MNN's schema/default/MNN.fbs (flatbuffers sorts table fields
alphabetically into vtable slots).

Usage:
  mnnfb.py find  <libmiocr.so> [lo] [hi]     -> offsets that parse as a non-empty Net
  mnnfb.py dump  <libmiocr.so> <offset> [n]  -> ops / tensors / IO of the Net at offset
"""
import struct
import sys


class Buf:
    def __init__(self, data, base=0):
        self.d = data
        self.base = base

    def u8(self, o):
        return self.d[o]

    def u16(self, o):
        return struct.unpack_from("<H", self.d, o)[0]

    def u32(self, o):
        return struct.unpack_from("<I", self.d, o)[0]

    def i32(self, o):
        return struct.unpack_from("<i", self.d, o)[0]

    def ok(self, o, n=1):
        return 0 <= o and o + n <= len(self.d)


NET_FIELDS = ["bizCode", "extraTensorDescribe", "extraInfo", "oplists", "outputName",
              "preferForwardType", "sourceType", "tensorName", "tensorNumber", "usage",
              "subgraphs", "mnn_uuid"]
OP_FIELDS = ["defaultDimentionFormat", "externalPath", "inputIndexes", "main", "name",
             "outputIndexes", "type"]
TENSORDESC_FIELDS = ["blob", "index", "name", "quantInfo", "regions"]


def table(buf, pos):
    """Return dict field_name -> absolute value position (or None)."""
    if not buf.ok(pos, 4):
        return None
    soff = buf.i32(pos)
    vt = pos - soff
    if not buf.ok(vt, 4) or vt % 2 or vt < 0:
        return None
    vsz = buf.u16(vt)
    tsz = buf.u16(vt + 2)
    if vsz < 4 or vsz > 1024 or vsz % 2 or tsz < 4 or tsz > 8192:
        return None
    if not buf.ok(vt, vsz) or not buf.ok(pos, tsz):
        return None
    return {"__vt": vt, "__vsz": vsz, "__tsz": tsz, "__pos": pos}


def field(buf, tbl, index):
    vt, vsz, pos = tbl["__vt"], tbl["__vsz"], tbl["__pos"]
    slot = vt + 4 + 2 * index
    if slot + 2 > vt + vsz:
        return None
    off = buf.u16(slot)
    if off == 0 or pos + off >= len(buf.d):
        return None
    return pos + off


def string(buf, pos):
    if pos is None or not buf.ok(pos, 4):
        return None
    n = buf.u32(pos)
    if n > 1 << 20 or not buf.ok(pos + 4, n):
        return None
    raw = buf.d[pos + 4:pos + 4 + n]
    try:
        return raw.decode("utf-8", "replace")
    except Exception:
        return None


def vector(buf, pos, elem=4):
    if pos is None or not buf.ok(pos, 4):
        return None
    n = buf.u32(pos)
    if n > 200000 or not buf.ok(pos + 4, n * elem):
        return None
    return (pos + 4, n)


def vec_strings(buf, pos):
    e = vector(buf, pos)
    if not e:
        return []
    start, n = e
    out = []
    for i in range(n):
        el = start + i * 4
        t = el + buf.u32(el)
        out.append(string(buf, t))
    return out


def vec_tables(buf, pos):
    e = vector(buf, pos)
    if not e:
        return []
    start, n = e
    out = []
    for i in range(n):
        el = start + i * 4
        t = el + buf.u32(el)
        tb = table(buf, t)
        if tb:
            out.append(tb)
    return out


def parse_net(buf, off):
    """Parse a Net at absolute offset off. Returns dict or None."""
    root = buf.u32(off)
    if root < 4 or root > 1 << 26 or root % 4:
        return None
    tbl = table(buf, off + root)
    if not tbl:
        return None
    ops = vec_tables(buf, field(buf, tbl, 3))
    names = vec_strings(buf, field(buf, tbl, 7))
    outs = vec_strings(buf, field(buf, tbl, 4))
    if not ops or not names:
        return None
    return {"off": off, "tbl": tbl, "ops": ops, "tensorName": names, "outputName": outs,
            "bizCode": string(buf, field(buf, tbl, 0)),
            "tensorNumber": (buf.i32(field(buf, tbl, 8)) if field(buf, tbl, 8) else None)}


def main():
    cmd = sys.argv[1]
    data = open(sys.argv[2], "rb").read()
    buf = Buf(data)
    if cmd == "find":
        lo = int(sys.argv[3], 0) if len(sys.argv) > 3 else 0x450000
        hi = int(sys.argv[4], 0) if len(sys.argv) > 4 else len(data)
        hi = min(hi, len(data) - 8)
        hits = []
        for o in range(lo - lo % 4, hi, 4):
            n = parse_net(buf, o)
            if n:
                hits.append((o, len(n["ops"]), len(n["tensorName"]), n["bizCode"], n["tensorNumber"]))
        print(f"nets found: {len(hits)}")
        for o, nops, ntn, biz, tnum in hits:
            print(f"  0x{o:07x} ops={nops:6} tensorNames={ntn:6} tensorNumber={tnum} bizCode={biz!r}")
    elif cmd == "dump":
        off = int(sys.argv[3], 0)
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 25
        net = parse_net(buf, off)
        if not net:
            print("no Net at", hex(off))
            return
        print(f"Net @0x{off:x}: ops={len(net['ops'])} tensors={len(net['tensorName'])} "
              f"outputs={net['outputName']} bizCode={net['bizCode']!r}")
        from collections import Counter
        types = Counter()
        for op in net["ops"][:200000]:
            t = field(buf, op, 6)
            types[buf.i32(t) if t else None] += 1
        print("op type histogram:", dict(types))
        print("first tensors:", net["tensorName"][:n])
        print("--- first ops ---")
        for op in net["ops"][:n]:
            nm = string(buf, field(buf, op, 4))
            t = field(buf, op, 6)
            ins = vector(buf, field(buf, op, 2))
            outs = vector(buf, field(buf, op, 5))
            ii = [buf.i32(ins[0] + 4 * i) for i in range(ins[1])] if ins else []
            oo = [buf.i32(outs[0] + 4 * i) for i in range(outs[1])] if outs else []
            print(f"  type={buf.i32(t) if t else None:<4} name={nm!r:32} in={ii} out={oo}")


if __name__ == "__main__":
    main()
