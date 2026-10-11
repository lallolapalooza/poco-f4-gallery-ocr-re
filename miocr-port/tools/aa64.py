#!/usr/bin/env python3
"""AArch64 static-analysis helper for libmiocr.so (stripped, file-offset == vaddr).

Commands:
  refs <so> [lo] [hi]        -> list data addresses in [lo,hi) that are referenced by code,
                                with the referring instruction addresses ('adrp+add' resolved)
  func <so> <codeaddr> [n]   -> disassemble around a code address, walking back to a
                                function prologue so argument setup is visible
"""
import sys
from capstone import Cs, CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN
from capstone.arm64 import ARM64_OP_IMM, ARM64_OP_REG, ARM64_OP_MEM
from elftools.elf.elffile import ELFFile


def load(path):
    data = open(path, "rb").read()
    with open(path, "rb") as fh:
        elf = ELFFile(fh)
        segs = [(s["p_vaddr"], s["p_offset"], s["p_filesz"]) for s in elf.iter_segments()
                if s["p_type"] == "PT_LOAD"]
    return data, segs


def vaddr_ok(segs, va):
    for v, o, sz in segs:
        if v <= va < v + sz:
            return True
    return False


def disasm_all(data, start=0, end=None):
    md = Cs(CS_ARCH_ARM64, CS_MODE_LITTLE_ENDIAN)
    md.detail = True
    end = end or len(data)
    return list(md.disasm(data[start:end], start))


LOADS = ("ldr", "ldrb", "ldrh", "ldrsw", "ldrsb", "ldrsh", "str", "strb", "strh", "ldur", "stur")


def resolve(insns):
    """Resolve adrp(+add|+ldr/str) pairs into absolute data addresses."""
    pending = {}
    out = {}
    for ins in insns:
        ops = ins.operands
        if ins.mnemonic == "adrp" and len(ops) >= 2 and ops[1].type == ARM64_OP_IMM:
            pending[ops[0].reg] = ops[1].imm & ~0xFFF
            continue
        if (ins.mnemonic in ("add", "sub") and len(ops) == 3 and ops[1].type == ARM64_OP_REG
                and ops[2].type == ARM64_OP_IMM and ops[1].reg in pending):
            t = pending[ops[1].reg] + (ops[2].imm if ins.mnemonic == "add" else -ops[2].imm)
            out.setdefault(t, []).append(ins.address)
            continue
        if ins.mnemonic in LOADS and len(ops) >= 2 and ops[1].type == ARM64_OP_MEM and ops[1].mem.base in pending:
            t = pending[ops[1].mem.base] + ops[1].mem.disp
            out.setdefault(t, []).append(ins.address)
    return out


def refs(path, lo, hi):
    data, segs = load(path)
    insns = disasm_all(data, 0x54DE0, 0x4508D0)
    out = resolve(insns)
    return {t: v for t, v in out.items() if lo <= t < hi}


def func_around(path, addr, n=60):
    data, _ = load(path)
    insns = disasm_all(data)
    idx = None
    for i, ins in enumerate(insns):
        if ins.address >= addr:
            idx = i
            break
    if idx is None:
        return []
    # walk back to a plausible function start: stp x29, x30 prelude or a ret
    start = max(0, idx - n)
    for j in range(idx, max(0, idx - 400), -1):
        if insns[j].mnemonic in ("stp", "sub") and "x29" in insns[j].op_str and "x30" in insns[j].op_str:
            start = j
            break
        if insns[j].mnemonic == "ret":
            start = j + 1
            break
    return insns[start:idx + n]


if __name__ == "__main__":
    cmd = sys.argv[1]
    so = sys.argv[2]
    if cmd == "refs":
        lo = int(sys.argv[3], 0) if len(sys.argv) > 3 else 0x450000
        hi = int(sys.argv[4], 0) if len(sys.argv) > 4 else 0x460000
        r = refs(so, lo, hi)
        for t in sorted(r):
            print(f"0x{t:07x}  refs@{[hex(x) for x in r[t][:4]]}{' ...' if len(r[t]) > 4 else ''}  n={len(r[t])}")
    elif cmd == "func":
        addr = int(sys.argv[3], 0)
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 60
        for ins in func_around(so, addr, n):
            mark = " <== " if ins.address == addr else "      "
            print(f"{ins.address:08x}: {ins.mnemonic:8}{ins.op_str}{mark}")
