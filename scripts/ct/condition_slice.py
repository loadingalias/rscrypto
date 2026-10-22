#!/usr/bin/env -S uv run python
"""Backward slices of conditional-branch conditions in linked CT disassembly.

The slice of a branch is the set of instructions whose results reach the
branch condition through registers, flags, or memory inside one function.
Reaching definitions follow the function's local control-flow graph; a load
reaches the stores that can write the loaded bytes (see "Memory" below). A
closed slice ends only at function inputs, entry memory, immediates, and
read-only data bound by contents. A slice is open when a value it reads may
come from:

- a call result, a call-clobbered register, or memory a call can write;
- a store to an address the model cannot locate;
- an instruction whose register effects this model does not know.

The model is conservative. An unknown instruction defines and reads every
register it names. The slice is the identity that an asm waiver binds
(`asm_heuristics.condition_identity`); it never classifies a branch by itself.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from typing import Any

import asm_heuristics as scanner

MAX_SLICE_NODES = 2048
ENTRY = -1
FRAME = -2


@dataclass(frozen=True)
class Effects:
  uses: frozenset[str]
  defs: frozenset[str]
  load: bool = False
  call: bool = False
  unknown: bool = False


def effects(uses: set[str] | frozenset[str], defs: set[str] | frozenset[str], **flags: bool) -> Effects:
  return Effects(frozenset(uses), frozenset(defs), **flags)


def split_operands(operands: str) -> list[str]:
  """Split top-level comma-separated operands, keeping brackets intact."""
  operands = re.sub(r"\s+<.*>$", "", operands)
  parts, depth, current = [], 0, []
  for character in operands:
    if character in "([{":
      depth += 1
    elif character in ")]}":
      depth -= 1
    if character == "," and depth == 0:
      parts.append("".join(current).strip())
      current = []
    else:
      current.append(character)
  if "".join(current).strip():
    parts.append("".join(current).strip())
  return parts


# x86-64 ---------------------------------------------------------------------

X86_ALIASES = {}
for wide, names in {
  "rax": ("eax", "ax", "al", "ah"), "rbx": ("ebx", "bx", "bl", "bh"), "rcx": ("ecx", "cx", "cl", "ch"),
  "rdx": ("edx", "dx", "dl", "dh"), "rsi": ("esi", "si", "sil"), "rdi": ("edi", "di", "dil"),
  "rbp": ("ebp", "bp", "bpl"), "rsp": ("esp", "sp", "spl"),
}.items():
  X86_ALIASES[wide] = wide
  for name in names:
    X86_ALIASES[name] = wide
for number in range(8, 16):
  for suffix in ("", "d", "w", "b"):
    X86_ALIASES[f"r{number}{suffix}"] = f"r{number}"
X86_CALLER_SAVED = {"rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "flags", *(f"v{n}" for n in range(32))}
# Win64 callees preserve rsi and rdi. They preserve only the low 128 bits of xmm6-xmm15, and the
# model does not split a vector register, so every vector register stays call-clobbered.
X86_WIN64_CALLER_SAVED = X86_CALLER_SAVED - {"rsi", "rdi"}
X86_FLAG_SETTERS = {
  "add", "sub", "and", "or", "xor", "adc", "sbb", "inc", "dec", "neg", "cmp", "test", "shl", "sal", "shr", "sar",
  "rol", "ror", "rcl", "rcr", "shld", "shrd", "imul", "mul", "bt", "bts", "btr", "btc", "bsf", "bsr", "popcnt",
  "lzcnt", "tzcnt", "andn", "bzhi", "bextr", "blsi", "blsr", "blsmsk", "adcx", "adox", "xadd", "cmpxchg",
}
X86_FLAG_READERS = {"adc", "sbb", "rcl", "rcr", "adcx", "adox"}
X86_WRITE_ONLY = {"mov", "movabs", "movzx", "movsx", "movsxd", "lea", "movd", "movq", "pop", "andn", "bzhi",
                  "bextr", "blsi", "blsr", "blsmsk", "shlx", "shrx", "sarx", "rorx", "popcnt", "lzcnt", "tzcnt",
                  "bsf", "bsr", "pext", "pdep", "crc32"}
X86_COMPARES = {"cmp", "test", "bt"}
X86_VECTOR_MOVE = re.compile(r"v?mov(?:dq[au](?:8|16|32|64)?|[au]p[sd])|vmov[qd]")
X86_ZERO_IDIOM = re.compile(r"v?pxor[dq]?|v?xorp[sd]")
X86_CONDITIONS = r"(?:o|no|b|nae|c|nb|ae|nc|e|z|ne|nz|be|na|nbe|a|s|ns|p|pe|np|po|l|nge|nl|ge|le|ng|nle|g)"


def x86_base(mnemonic: str) -> str:
  if mnemonic.startswith(("movz", "movs")) and mnemonic not in {"movsd", "movss", "movsq"}:
    return "movsxd" if mnemonic in {"movslq", "movsxd"} else ("movzx" if mnemonic.startswith("movz") else "movsx")
  for base in sorted(X86_FLAG_SETTERS | X86_WRITE_ONLY | {"cmov", "set", "xchg", "not", "push", "nop", "cqto",
                                                         "cltq", "cqo", "cdqe", "cltd", "cdq", "bswap", "div",
                                                         "idiv", "mulx"}, key=len, reverse=True):
    if mnemonic == base or (mnemonic.startswith(base) and mnemonic[len(base):] in {"b", "w", "l", "q"}):
      return base
  if re.fullmatch(rf"cmov{X86_CONDITIONS}[wlq]?", mnemonic):
    return "cmov"
  if re.fullmatch(rf"set{X86_CONDITIONS}b?", mnemonic):
    return "set"
  return mnemonic


X86_PARTIAL = re.compile(r"%(?:[abcd][lhx]|[sd]il?|[sb]pl?|r(?:[89]|1[0-5])[bw])\b")


def x86_registers(operand: str) -> set[str]:
  registers = set()
  for name in re.findall(r"%([a-z0-9]+)", operand):
    if name in X86_ALIASES:
      registers.add(X86_ALIASES[name])
    elif match := re.fullmatch(r"[xyz]mm(\d+)", name):
      registers.add(f"v{match.group(1)}")
    elif name != "rip":
      registers.add(name)
  return registers


def x86_effects(row: scanner.DecodedLine, identity: str, caller_saved: set[str] = X86_CALLER_SAVED) -> Effects:
  mnemonic = row.mnemonic
  base = x86_base(mnemonic)
  operands = split_operands(row.operands)
  memory = [operand for operand in operands if "(" in operand]
  constant = "<const:" in identity
  load_from_memory = bool(memory) and not constant
  address_uses = set().union(*(x86_registers(operand) for operand in memory)) if memory else set()
  if base in {"nop", "endbr64"} or mnemonic in {"nopw", "nopl", "data16"}:
    return effects(set(), set())
  if scanner.is_conditional_branch("x86_64-unknown-linux-gnu", mnemonic):
    return effects({"flags"}, set())
  if mnemonic in {"call", "callq"}:
    return effects(x86_registers(row.operands) | {"rsp"}, caller_saved, call=True)
  if mnemonic in {"ret", "retq", "jmp", "jmpq", "ud2", "int3", "hlt"}:
    return effects(x86_registers(row.operands), set())
  if not operands:
    if base in {"cqto", "cqo"}:
      return effects({"rax"}, {"rdx"})
    if base in {"cltq", "cdqe"}:
      return effects({"rax"}, {"rax"})
    if base in {"cltd", "cdq"}:
      return effects({"rax"}, {"rdx"})
    return effects(set(), set(), unknown=True)
  destination = operands[-1]
  destination_registers = set() if "(" in destination else x86_registers(destination)
  # Writing an 8- or 16-bit register keeps the other bits, so the write also reads the register.
  partial = destination_registers if X86_PARTIAL.search(destination) else set()
  sources = set().union(*(x86_registers(operand) for operand in operands[:-1])) if len(operands) > 1 else set()
  is_store = "(" in destination
  # A memory destination is a store; it defines no register but its address registers are read.
  if base in X86_COMPARES:
    return effects(sources | x86_registers(destination), {"flags"}, load=load_from_memory)
  if base in {"div", "idiv"}:
    return effects({"rax", "rdx"} | sources | x86_registers(destination), {"rax", "rdx", "flags"},
                   load=load_from_memory)
  if base == "mul" and len(operands) == 1:
    return effects({"rax"} | x86_registers(destination), {"rax", "rdx", "flags"}, load=load_from_memory)
  if base == "mulx":
    return effects({"rdx"} | sources, x86_registers(operands[-1]) | x86_registers(operands[-2]),
                   load=load_from_memory)
  if base == "push":
    return effects(x86_registers(destination) | {"rsp"}, {"rsp"}, load=load_from_memory)
  if base == "pop":
    return effects({"rsp"}, destination_registers | {"rsp"}, load=True)
  if base == "xchg":
    return effects(sources | x86_registers(destination), sources | destination_registers, load=load_from_memory)
  if base in {"set", "cmov"}:
    uses = {"flags"} | sources | address_uses | partial | (destination_registers if base == "cmov" else set())
    return effects(uses, destination_registers, load=load_from_memory)
  if base == "xor" and len(operands) == 2 and operands[0] == operands[1] and not memory and not partial:
    return effects(set(), destination_registers | {"flags"})
  if base == "lea":
    return effects(sources | partial, destination_registers)
  if base in X86_WRITE_ONLY:
    defs = destination_registers | ({"flags"} if base in X86_FLAG_SETTERS else set())
    return effects(sources | address_uses | partial, defs, load=load_from_memory and not is_store)
  if base in X86_FLAG_SETTERS or base in {"not", "bswap"}:
    uses = sources | x86_registers(destination) | ({"flags"} if base in X86_FLAG_READERS else set())
    if base == "imul" and len(operands) == 3:
      uses -= destination_registers
    defs = destination_registers | ({"flags"} if base in X86_FLAG_SETTERS else set())
    return effects(uses, defs, load=load_from_memory)
  if X86_VECTOR_MOVE.fullmatch(mnemonic):
    # A whole-register vector copy, load, or store.
    return effects(sources | address_uses, destination_registers, load=load_from_memory and not is_store)
  if X86_ZERO_IDIOM.fullmatch(mnemonic) and not memory and len(set(operands)) == 1:
    # `xor` of a register with itself sets it to zero.
    return effects(set(), destination_registers)
  # Vector and other instructions: AT&T puts the destination last. Two
  # operands read the destination too. The model does not know them.
  uses = sources | address_uses | (x86_registers(destination) if len(operands) == 2 else set())
  return effects(uses, destination_registers | {"flags"}, load=load_from_memory, unknown=True)


# AArch64 --------------------------------------------------------------------

AARCH64_CALLER_SAVED = {f"x{n}" for n in range(19)} | {"x30", "nzcv"} | {f"v{n}" for n in range(32)}
AARCH64_FLAG_SETTERS = {"cmp", "cmn", "tst", "adds", "subs", "ands", "bics", "negs", "adcs", "sbcs", "ngcs",
                        "fcmp", "fcmpe", "ccmp", "ccmn", "fccmp", "fccmpe"}
AARCH64_FLAG_READERS = {"csel", "csinc", "csinv", "csneg", "cset", "csetm", "cinc", "cinv", "cneg", "ccmp", "ccmn",
                        "adc", "adcs", "sbc", "sbcs", "ngc", "ngcs", "fcsel", "fccmp", "fccmpe"}
AARCH64_DATA = {
  "add", "sub", "and", "orr", "eor", "orn", "eon", "bic", "lsl", "lsr", "asr", "ror", "mul", "madd", "msub",
  "mneg", "umulh", "smulh", "umull", "smull", "umaddl", "smaddl", "umsubl", "smsubl", "udiv", "sdiv", "mov",
  "mvn", "neg", "sxtw", "sxth", "sxtb", "uxtw", "uxth", "uxtb", "ubfx", "sbfx", "ubfiz", "sbfiz", "ubfm", "sbfm",
  "extr", "rbit", "rev", "rev16", "rev32", "clz", "cls", "adr", "adrp", "movz", "movn", "csel", "csinc", "csinv",
  "csneg", "cset", "csetm", "cinc", "cinv", "cneg", "adc", "sbc", "ngc", "adds", "subs", "ands", "bics", "negs",
  "adcs", "sbcs", "ngcs", "lslv", "lsrv", "asrv", "rorv", "fmov", "umov", "smov", "dup", "ins", "cnt", "addv",
  "uaddlv", "xtn", "uzp1", "uzp2", "zip1", "zip2", "trn1", "trn2", "cmeq", "cmhi", "cmhs", "cmgt", "cmge", "cmtst",
  "ushr", "sshr", "shl", "ushll", "ushll2", "umlal", "umlal2", "umull2", "mla", "fadd", "not", "bsl", "bit", "bif",
  "pmull", "pmull2", "ext", "tbl", "movi", "rev64", "addp", "umaxv", "uminv", "uzp", "shrn", "sli", "sri", "usra",
}
AARCH64_READS_DESTINATION = {"movk", "bfi", "bfxil", "bfm", "bfc", "ins", "mla", "umlal", "umlal2", "bsl", "bit",
                             "bif", "sli", "sri", "usra"}


AARCH64_ATOMIC = re.compile(r"^(?:ld(?:add|clr|eor|set|smax|smin|umax|umin)|swp|cas)")


def aarch64_registers(text: str) -> list[str]:
  names = []
  for match in re.finditer(r"\b(?:([xw])(\d+)|(sp|wsp)|([qdsbhv])(\d+)(?:\.\w+)?)\b", text):
    if match.group(1):
      names.append(f"x{match.group(2)}")
    elif match.group(3):
      names.append("sp")
    else:
      names.append(f"v{match.group(5)}")
  return names


def aarch64_effects(row: scanner.DecodedLine, identity: str) -> Effects:
  mnemonic = row.mnemonic.split(".")[0] if not row.mnemonic.startswith("b.") else row.mnemonic
  operands = split_operands(row.operands)
  constant = "<const:" in identity
  # Brackets are a memory operand only for loads, stores, and atomics; elsewhere they index a vector lane.
  memory = re.search(r"\[(?P<inside>[^\]]*)\](?P<writeback>!?)(?:, #(?P<post>[-+]?\w+))?", row.operands) \
    if mnemonic.startswith(("ld", "st", "prfm")) or AARCH64_ATOMIC.match(mnemonic) else None
  writeback = memory is not None and (memory.group("writeback") or memory.group("post"))
  if mnemonic.startswith("b."):
    return effects({"nzcv"}, set())
  if mnemonic in {"cbz", "cbnz", "tbz", "tbnz"}:
    return effects(set(aarch64_registers(operands[0])), set())
  if mnemonic in {"bl", "blr", "blraa", "blrab"}:
    return effects(set(aarch64_registers(row.operands)) | {"sp"}, AARCH64_CALLER_SAVED, call=True)
  if mnemonic in {"b", "br", "ret", "retaa", "retab", "brk", "udf", "hint", "nop", "prfm", "dmb", "dsb", "isb",
                  "bti", "paciasp", "autiasp", "pacibsp", "autibsp", "csdb", "sb", "msr"}:
    if mnemonic == "msr" and "dit" not in row.operands.lower():
      return effects(set(aarch64_registers(row.operands)), set(), unknown=True)
    return effects(set(aarch64_registers(row.operands)), set())
  if mnemonic == "mrs":
    return effects(set(), set(aarch64_registers(operands[0])), unknown="dit" not in row.operands.lower())
  if mnemonic in {"ldr", "ldrsw"} and memory is None:
    # A PC-relative literal load reads only its target, which the identity binds.
    return effects(set(), set(aarch64_registers(operands[0])), load=not constant)
  if memory is not None:
    base = aarch64_registers(memory.group("inside"))
    data = [register for operand in operands if "[" not in operand and not operand.startswith("#")
            for register in aarch64_registers(operand)]
    if mnemonic.startswith(("st", "stur")):
      defs = {base[0]} if writeback and base else set()
      if mnemonic.startswith(("stxr", "stlxr", "stxp", "stlxp")):
        defs |= {data[0]}
        data = data[1:]
      return effects(set(data) | set(base), defs)
    if mnemonic.startswith(("ld", "ldur")):
      defs = set(data) | ({base[0]} if writeback and base else set())
      return effects(set(base), defs, load=not constant)
    return effects(set(base) | set(data), set(data), load=not constant, unknown=True)
  registers = aarch64_registers(row.operands)
  if mnemonic in {"cmp", "cmn", "tst", "fcmp", "fcmpe"}:
    return effects(set(registers), {"nzcv"})
  if mnemonic in {"ccmp", "ccmn", "fccmp", "fccmpe"}:
    return effects(set(registers) | {"nzcv"}, {"nzcv"})
  if mnemonic in AARCH64_DATA or mnemonic in AARCH64_READS_DESTINATION:
    if not registers:
      return effects(set(), set(), unknown=True)
    destination, sources = registers[0], registers[1:]
    uses = set(sources) | ({"nzcv"} if mnemonic in AARCH64_FLAG_READERS else set())
    if mnemonic in AARCH64_READS_DESTINATION or (mnemonic == "mov" and "[" in operands[0]):
      uses.add(destination)
    defs = {destination} | ({"nzcv"} if mnemonic in AARCH64_FLAG_SETTERS else set())
    return effects(uses, defs)
  return effects(set(registers), set(registers) | {"nzcv"}, unknown=True)


# RISC-V ---------------------------------------------------------------------

RISCV_CALLER_SAVED = {"ra", *(f"t{n}" for n in range(7)), *(f"a{n}" for n in range(8)),
                      *(f"ft{n}" for n in range(12)), *(f"fa{n}" for n in range(8))}
RISCV_ALU = {
  "add", "addi", "addw", "addiw", "sub", "subw", "and", "andi", "or", "ori", "xor", "xori", "sll", "slli", "srl",
  "srli", "sra", "srai", "sllw", "slliw", "srlw", "srliw", "sraw", "sraiw", "slt", "slti", "sltu", "sltiu", "seqz",
  "snez", "sltz", "sgtz", "mv", "li", "lui", "auipc", "not", "neg", "negw", "sext.w", "sext.b", "sext.h", "zext.b",
  "zext.h", "zext.w", "mul", "mulh", "mulhu", "mulhsu", "mulw", "div", "divu", "divw", "divuw", "rem", "remu",
  "remw", "remuw", "c.mv", "c.li", "c.addi", "c.add", "c.sub", "c.and", "c.or", "c.xor", "c.slli", "c.srli",
  "c.srai", "c.andi", "c.lui", "c.addiw", "c.addw", "c.subw", "c.addi16sp", "c.addi4spn", "fmv.x.d", "fmv.d.x",
}
RISCV_LOADS = {"lb", "lh", "lw", "ld", "lbu", "lhu", "lwu", "flw", "fld", "c.lw", "c.ld", "c.lwsp", "c.ldsp",
               "c.fld", "c.fldsp"}
RISCV_BRANCH_TWO = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "bgt", "ble", "bgtu", "bleu"}
RISCV_BRANCH_ONE = {"beqz", "bnez", "blez", "bgez", "bltz", "bgtz", "c.beqz", "c.bnez"}


def riscv_registers(text: str) -> list[str]:
  return [name for name in scanner.RISCV_REGISTER.findall(re.sub(r"\s+<.*>$", "", text)) if name != "zero"]


def riscv_effects(row: scanner.DecodedLine, identity: str) -> Effects:
  mnemonic = row.mnemonic
  registers = riscv_registers(row.operands)
  if mnemonic in RISCV_BRANCH_TWO | RISCV_BRANCH_ONE:
    return effects(set(registers), set())
  if mnemonic in {"jal", "c.jal"} or (mnemonic in {"jalr", "c.jalr"} and (not registers or registers[0] == "ra"
                                                                            or "(" in row.operands)):
    return effects(set(registers) | {"sp"}, RISCV_CALLER_SAVED, call=True)
  if mnemonic in {"j", "c.j", "jr", "c.jr", "ret", "nop", "c.nop", "ebreak", "c.ebreak", "unimp", "c.unimp",
                  "fence", "fence.i"}:
    return effects(set(registers), set())
  if mnemonic in scanner.RISCV_STORES:
    return effects(set(registers), set())
  if mnemonic in RISCV_LOADS:
    return effects(set(registers[1:]), set(registers[:1]), load="<const:" not in identity)
  if mnemonic in RISCV_ALU:
    if mnemonic in {"auipc", "lui", "li", "c.li", "c.lui"}:
      return effects(set(), set(registers[:1]))
    return effects(set(registers[1:]), set(registers[:1]))
  return effects(set(registers), set(registers[:1]), unknown=True)


# POWER ----------------------------------------------------------------------

POWER_CALLER_SAVED = {"r0", *(f"r{n}" for n in range(3, 13)), "lr", "ctr", "xer", "cr0", "cr1", "cr5", "cr6", "cr7"}
POWER_ARITHMETIC = {
  "add", "subf", "and", "or", "xor", "nand", "nor", "andc", "orc", "eqv", "mullw", "mulld", "mulhdu", "mulhd",
  "mulhw", "mulhwu", "divd", "divdu", "divw", "divwu", "sld", "srd", "slw", "srw", "modud", "modsd", "moduw",
  "modsw", "isel",
}
POWER_CARRY = {"addc", "adde", "subfc", "subfe", "addze", "subfze", "addme", "subfme", "srad", "sraw", "sradi",
               "srawi", "addic", "subfic"}
POWER_UNARY = {"neg", "extsw", "extsh", "extsb", "cntlzd", "cntlzw", "popcntd", "popcntw", "cnttzd", "cnttzw",
               "mr", "not", "extswsli"}
POWER_ROTATE = {"rldicl", "rldicr", "rldic", "rlwinm", "rlwnm", "rldcl", "sldi", "srdi", "slwi", "srwi", "clrldi",
                "clrlwi", "clrrdi", "rotldi", "rotlwi", "extldi", "extrdi", "rotld"}
POWER_ROTATE_INSERT = {"rldimi", "rlwimi", "insrdi", "insrwi"}
POWER_IMMEDIATE = {"addi", "addis", "mulli", "ori", "oris", "xori", "xoris", "andi.", "andis."}
POWER_LOADS = {"lbz", "lhz", "lha", "lwz", "lwa", "ld", "lbzu", "lhzu", "lwzu", "ldu", "lbzx", "lhzx", "lhax",
               "lwzx", "lwax", "ldx", "ldbrx", "lwbrx", "lhbrx", "ldarx", "lwarx"}
POWER_STORES = {"stb", "sth", "stw", "std", "stbu", "sthu", "stwu", "stdu", "stbx", "sthx", "stwx", "stdx",
                "stdbrx", "stwbrx", "sthbrx", "stdcx.", "stwcx."}
POWER_VECTOR_MEMORY = {"lxv": 1, "lxvd2x": 2, "lxvw4x": 2, "lxsd": 1, "lxssp": 1, "lxsdx": 2, "lvx": 2, "lfd": 1,
                       "lfs": 1, "lfdx": 2, "lxvx": 2, "lxvdsx": 2, "lxsiwzx": 2}
POWER_VECTOR_STORE = {"stxv": 1, "stxvd2x": 2, "stxvw4x": 2, "stxsd": 1, "stxssp": 1, "stxsdx": 2, "stvx": 2,
                      "stfd": 1, "stfs": 1, "stfdx": 2, "stxvx": 2}
POWER_COMPARES = {"cmpd", "cmpw", "cmpld", "cmplw", "cmpdi", "cmpwi", "cmpldi", "cmplwi", "cmpb"}


def power_numbers(text: str) -> list[str]:
  return re.findall(r"(?<![\w.])-?\d+(?![\w.])", re.sub(r"\s+<.*>$", "", text))


def power_condition_field(row: scanner.DecodedLine) -> str:
  """Return the CR field a conditional branch reads."""
  first = split_operands(row.operands)[0] if row.operands else ""
  if row.mnemonic.rstrip("+-") in {"bt", "bf", "bdnzt", "bdnzf", "bdzt", "bdzf"}:
    return f"cr{int(first) // 4}" if first.isdigit() else "cr0"
  if row.mnemonic.rstrip("+-") == "bc":
    numbers = power_numbers(row.operands)
    return f"cr{int(numbers[1]) // 4}" if len(numbers) >= 2 else "cr0"
  return f"cr{first}" if first.isdigit() and len(split_operands(row.operands)) > 1 else "cr0"


def power_effects(row: scanner.DecodedLine, identity: str) -> Effects:
  mnemonic = row.mnemonic.rstrip("+-")
  record = mnemonic.endswith(".") and mnemonic not in {"andi.", "andis.", "stdcx.", "stwcx."}
  plain = mnemonic if mnemonic in {"andi.", "andis."} else mnemonic.rstrip(".")
  numbers = power_numbers(row.operands)
  gprs = [f"r{number}" for number in numbers]
  constant = "<const:" in identity
  cr0 = {"cr0"} if record or mnemonic in {"andi.", "andis.", "stdcx.", "stwcx."} else set()
  if scanner.is_conditional_branch("powerpc64le-unknown-linux-gnu", mnemonic):
    if mnemonic in {"bdnz", "bdz"}:
      return effects({"ctr"}, {"ctr"})
    if mnemonic.startswith("bd"):
      return effects({power_condition_field(row), "ctr"}, {"ctr"})
    return effects({power_condition_field(row)}, set())
  if mnemonic in {"bl", "bla", "bctrl", "blrl"}:
    return effects({f"r{n}" for n in range(3, 11)} | {"r1", "r2", "ctr"}, POWER_CALLER_SAVED, call=True)
  if mnemonic in {"b", "blr", "bctr", "nop", "trap", "sync", "lwsync", "isync", "attn"}:
    return effects({"lr"} if mnemonic == "blr" else ({"ctr"} if mnemonic == "bctr" else set()), set())
  if plain in POWER_COMPARES:
    operands = split_operands(row.operands)
    field = f"cr{operands[0]}" if len(operands) == 3 else "cr0"
    sources = numbers[1:] if len(operands) == 3 else numbers
    if plain.endswith("i"):
      sources = sources[:-1]
    return effects({f"r{n}" for n in sources}, {field})
  if plain in {"li", "lis"}:
    return effects(set(), {gprs[0]})
  if plain in POWER_IMMEDIATE or plain in {"addic", "subfic"}:
    uses = set() if plain in {"addi", "addis"} and numbers[1] == "0" else {gprs[1]}
    return effects(uses, {gprs[0]} | cr0 | ({"xer"} if plain in POWER_CARRY else set()))
  if plain in {"sub", "subc"}:
    # Extended mnemonics: sub rD, rA, rB computes rA - rB.
    return effects(set(gprs[1:3]), {gprs[0]} | cr0 | ({"xer"} if plain == "subc" else set()))
  if plain in {"iseleq", "isellt", "iselgt", "iselso"}:
    uses = {gprs[2]} | ({gprs[1]} if numbers[1] != "0" else set()) | {"cr0"}
    return effects(uses, {gprs[0]})
  if plain == "mcrf":
    return effects({f"cr{numbers[1]}"}, {f"cr{numbers[0]}"})
  if plain in {"cror", "crorc", "crand", "crandc", "crxor", "crnor", "crnand", "creqv", "crset", "crclr", "crnot",
               "crmove"}:
    fields = [f"cr{int(number) // 4}" for number in numbers]
    # One CR bit changes, so the destination field is also read.
    return effects(set(fields), set(fields[:1]))
  if plain in POWER_ARITHMETIC:
    uses = set(gprs[1:3])
    if plain == "isel":
      uses = {gprs[2]} | ({gprs[1]} if numbers[1] != "0" else set()) | {f"cr{int(numbers[3]) // 4}"}
    return effects(uses, {gprs[0]} | cr0)
  if plain in POWER_CARRY:
    uses = set(gprs[1:3]) if plain not in {"sradi", "srawi", "addze", "subfze", "addme", "subfme"} else {gprs[1]}
    if plain in {"adde", "subfe", "addze", "subfze", "addme", "subfme"}:
      uses.add("xer")
    return effects(uses, {gprs[0], "xer"} | cr0)
  if plain in POWER_UNARY:
    return effects({gprs[1]}, {gprs[0]} | cr0)
  if plain in POWER_ROTATE:
    uses = {gprs[1]} | ({gprs[2]} if plain in {"rlwnm", "rldcl", "rotld"} else set())
    return effects(uses, {gprs[0]} | cr0)
  if plain in POWER_ROTATE_INSERT:
    return effects({gprs[0], gprs[1]}, {gprs[0]} | cr0)
  if plain in POWER_LOADS:
    if "(" in row.operands:
      base = {gprs[2]} if numbers[2] != "0" else set()
      defs = {gprs[0]} | ({gprs[2]} if plain.endswith("u") else set())
    else:
      base = ({gprs[1]} if numbers[1] != "0" else set()) | {gprs[2]}
      defs = {gprs[0]}
    return effects(base, defs | cr0, load=not constant)
  if plain in POWER_STORES:
    if "(" in row.operands:
      uses = {gprs[0]} | ({gprs[2]} if numbers[2] != "0" else set())
      defs = {gprs[2]} if plain.endswith("u") else set()
    else:
      uses = {gprs[0], gprs[2]} | ({gprs[1]} if numbers[1] != "0" else set())
      defs = set()
    return effects(uses, defs | cr0)
  if plain in POWER_VECTOR_MEMORY or plain in POWER_VECTOR_STORE:
    count = POWER_VECTOR_MEMORY.get(plain) or POWER_VECTOR_STORE[plain]
    vector = f"vs{numbers[0]}"
    if count == 1:
      bases = {gprs[2]} if len(numbers) > 2 and numbers[2] != "0" else set()
    else:
      bases = ({gprs[1]} if numbers[1] != "0" else set()) | {gprs[2]}
    if plain in POWER_VECTOR_MEMORY:
      return effects(bases, {vector}, load=not constant)
    return effects(bases | {vector}, set())
  if plain in {"mtvsrd", "mtvsrwz", "mtvsrdd", "mtvsrws"}:
    return effects(set(gprs[1:]), {f"vs{numbers[0]}"})
  if plain in {"mfvsrd", "mfvsrwz", "mfvsrld"}:
    return effects({f"vs{numbers[1]}"}, {gprs[0]})
  if plain == "mflr":
    return effects({"lr"}, {gprs[0]})
  if plain == "mtlr":
    return effects({gprs[0]}, {"lr"})
  if plain == "mfctr":
    return effects({"ctr"}, {gprs[0]})
  if plain == "mtctr":
    return effects({gprs[0]}, {"ctr"})
  if plain in {"mfocrf", "mfcr"}:
    return effects({f"cr{n}" for n in range(8)}, {gprs[0]})
  if plain in {"mtocrf", "mtcrf"}:
    return effects({gprs[-1]}, {f"cr{n}" for n in range(8)})
  if plain.startswith(("x", "v", "f")) and not plain.startswith(("xor", "xori", "xoris")):
    # VSX, VMX, and FP arithmetic use only vector registers.
    vectors = [f"vs{number}" for number in numbers]
    return effects(set(vectors[1:]), set(vectors[:1]), unknown=plain not in POWER_KNOWN_VECTOR)
  return effects(set(gprs), set(gprs[:1]) | {"cr0"}, unknown=True)


POWER_KNOWN_VECTOR = {
  "xxlor", "xxland", "xxlxor", "xxlnor", "xxlandc", "xxlorc", "xxsel", "xxpermdi", "xxswapd", "xxspltw", "xxmrghw",
  "xxmrglw", "xxmrghd", "xxmrgld", "xxleqv", "xxlnand", "xxspltib", "xxbrd", "xxbrw", "vadduwm", "vaddudm",
  "vsubuwm", "vsubudm", "vmuluwm", "vmulouw", "vmuleuw", "vperm", "vsldoi", "vsl", "vsr", "vsld", "vsrd", "vslw",
  "vsrw", "vsraw", "vsrad", "vspltisw", "vspltisb", "vsplth", "vspltb", "vspltw", "vcmpequw", "vcmpequd",
  "vcmpgtuw", "vcmpgtud", "vand", "vor", "vxor", "vandc", "vnor", "vmrghw", "vmrglw", "vpkudum", "vupklsw",
  "vrlw", "vrld", "vaddcuw", "vsububm", "vaddubm", "vcipher", "vcipherlast", "vncipher", "vncipherlast",
  "vpmsumd", "vpmsumw", "vsbox", "vshasigmaw", "vshasigmad", "fmr",
}


# IBM Z ----------------------------------------------------------------------

S390X_CALLER_SAVED = {f"r{n}" for n in range(6)} | {"r14", "cc"} | {f"f{n}" for n in range(8)} | {
  f"v{n}" for n in range(32)}
S390X_NO_CC = {
  "l", "lg", "ly", "lgf", "lgfr", "llgf", "llgfr", "llgc", "llgcr", "llc", "llcr", "llh", "llhr", "llgh", "llghr",
  "lh", "lhy", "lgh", "lghr", "lb", "lbr", "lgb", "lgbr", "lr", "lgr", "la", "lay", "larl", "lhi", "lghi", "lgfi",
  "llilf", "llihf", "llill", "llilh", "llihl", "llihh", "iilf", "iihf", "iill", "iilh", "iihl", "iihh", "lgrl",
  "lrl", "llgfrl", "lgfrl", "llghrl", "lghrl", "lhrl", "llhrl", "sll", "srl", "sllg", "srlg", "sllk", "srlk",
  "rll", "rllg", "risbg", "risbgn", "risbhg", "risblg", "msgr", "msr", "msg", "ms", "msgf", "msgfr", "msfi",
  "msgfi", "mlgr", "mlg", "lgdr", "ldgr", "locr", "locgr", "loc", "locg", "lochi", "locghi", "lgat", "bras", "risbgz",
  "risbgnz",
  "lrvgr", "lrvr", "lrvg", "lrv", "llgt", "llgtr", "lax", "sar", "ear",
}
S390X_CC_SETTERS = {
  "a", "ag", "agr", "ar", "ahi", "aghi", "agf", "agfr", "afi", "agfi", "al", "alg", "algr", "alr", "alfi", "algfi",
  "alc", "alcr", "alcg", "alcgr", "s", "sg", "sgr", "sr", "sgf", "sgfr", "sl", "slg", "slgr", "slr", "slb", "slbr",
  "slbg", "slbgr", "slfi", "slgfi", "n", "ng", "ngr", "nr", "nilf", "nihf", "nill", "nilh", "nihl", "nihh", "o",
  "og", "ogr", "or", "oilf", "oihf", "oill", "oilh", "oihl", "oihh", "x", "xg", "xgr", "xr", "xilf", "xihf", "c",
  "cg", "cgr", "cr", "cgf", "cgfr", "chi", "cghi", "cfi", "cgfi", "cl", "clg", "clgr", "clr", "clgf", "clgfr",
  "clfi", "clgfi", "tm", "tmll", "tmlh", "tmhl", "tmhh", "ltr", "ltgr", "lt", "ltg", "ltgf", "ltgfr", "lcr",
  "lcgr", "lpr", "lpgr", "lnr", "lngr", "ark", "agrk", "sgrk", "srk", "alrk", "algrk", "slrk", "slgrk", "nrk",
  "ngrk", "ork", "ogrk", "xrk", "xgrk", "ahik", "aghik", "alhsik", "alghsik", "sla", "slag", "sra", "srag", "slak",
  "srak", "flogr", "popcnt", "icm", "icmh", "icmy", "rnsbg", "rosbg", "rxsbg", "clc", "xc", "nc", "oc", "chhsi",
  "cghsi", "clhhsi", "clghsi", "clfhsi", "cgrl", "crl", "clgrl", "clrl", "cghrl", "chrl", "clhrl", "clghrl",
  "cgfrl", "clgfrl", "clgt", "clt", "cy", "cly", "msrkc", "msgrkc", "msc", "msgc", "cli", "cliy",
}
# Instructions that replace all 64 bits of their first operand. Every other
# instruction may keep some bits, so it also reads the first operand.
S390X_FULL_DEFS = {
  "lg", "lgr", "lgf", "lgfr", "llgf", "llgfr", "llgc", "llgcr", "llgh", "llghr", "lgh", "lghr", "lgb", "lgbr", "la",
  "lay", "larl", "lghi", "lgfi", "llilf", "llihf", "llill", "llilh", "llihl", "llihh", "lgrl", "llgfrl", "lgfrl",
  "llghrl", "lghrl", "ltg", "ltgr", "ltgf", "ltgfr", "lcgr", "lpgr", "lngr", "sllg", "srlg", "srag", "slag", "rllg",
  "agrk", "sgrk", "algrk", "slgrk", "ngrk", "ogrk", "xgrk", "aghik", "alghsik", "msgrkc", "lgdr", "lrvgr", "lrvg",
  "llgt", "llgtr", "popcnt", "lgat",
}
S390X_COMPARES = {"cli", "cliy", "clhhsi", "c", "cg", "cgr", "cr", "cgf", "cgfr", "chi", "cghi", "cfi", "cgfi", "cl", "clg", "clgr", "clr",
                  "clgf", "clgfr", "clfi", "clgfi", "tm", "tmll", "tmlh", "tmhl", "tmhh", "chhsi", "cghsi",
                  "clhhsi", "clghsi", "clfhsi", "cgrl", "crl", "clgrl", "clrl", "cghrl", "chrl", "clhrl", "clghrl",
                  "cgfrl", "clgfrl", "cy", "cly", "clc"}
S390X_VECTOR_LOADS = {"vl", "vlrepg", "vlrepf", "vlreph", "vlrepb", "vleg", "vlef", "vleh", "vleb", "vlbb", "vll",
                      "vlm", "vlrl", "vllezg", "vllezf", "vllezb", "vllezh"}
S390X_VECTOR_STORES = {"vst", "vsteg", "vstef", "vsteh", "vsteb", "vstl", "vstm", "vstrl"}
S390X_KNOWN_VECTOR = {
  "vzero", "vone", "vrepib", "vrepih", "vrepif", "vrepig", "vrepi", "vrepg", "vrepf", "vreph", "vrepb", "vlvgg",
  "vlvgf", "vlvgh", "vlvgb", "vlvgp", "vlgvg", "vlgvf", "vlgvh", "vlgvb", "vaq", "vacq", "vaccq", "vacccq", "vsq",
  "vsbiq", "vscbiq", "vsbcbiq", "vag", "vaf", "vah", "vab", "vsg", "vsf", "vn", "vo", "vx", "vnc", "vnn", "vno",
  "vnx", "voc", "vsel", "vperm", "vpdi", "vsrlb", "vslb", "vsrl", "vsl", "vsra", "vsrab", "vsldb", "vesrlg",
  "veslg", "vesrag", "vesrlf", "veslf", "vmrhg", "vmrlg", "vmrhf", "vmrlf", "vlr", "vgbm", "vgm", "vceqg",
  "vceqf", "vchlg", "vchlf", "vchg", "vchf", "vmlf", "vmlhf", "vmalf", "vmalhf", "vmlof", "vmlef", "vmalof",
  "vmalef", "vuplf", "vupllf", "vpkg", "vpkf", "vsumqg", "vsumqf", "vsumg", "vsum", "vmxlg", "vmnlg",
}
S390X_STORES = {"st", "stg", "sty", "sth", "sthy", "stc", "stcy", "stgrl", "strl", "sthrl", "stmg", "mvc", "mvi",
                "mvhi", "mvghi", "mvhhi", "stoc", "stocg", "stfh", "std", "ste", "vst", "vsteg", "vstef", "stam"}


def s390x_registers(text: str) -> list[str]:
  return [f"{kind}{number}" for kind, number in re.findall(r"%([rfva])(\d+)", re.sub(r"\s+<.*>$", "", text))]


def s390x_effects(row: scanner.DecodedLine, identity: str) -> Effects:
  mnemonic = row.mnemonic
  registers = s390x_registers(row.operands)
  memory = "(" in row.operands
  constant = "<const:" in identity
  operands = split_operands(row.operands)
  first = s390x_registers(operands[0]) if operands else []
  if mnemonic == "brasl":
    return effects({f"r{n}" for n in range(2, 7)} | {"r15"}, S390X_CALLER_SAVED, call=True)
  if mnemonic in {"j", "jg", "br", "bcr", "nopr", "nop", "brcl"} and not scanner.is_conditional_branch(
      "s390x-unknown-linux-gnu", mnemonic):
    return effects(set(registers), set())
  if scanner.S390X_COMPARE_AND_BRANCH.match(mnemonic):
    return effects(set(registers), set())
  if mnemonic in {"brct", "brctg"}:
    return effects(set(first), set(first))
  if scanner.is_conditional_branch("s390x-unknown-linux-gnu", mnemonic):
    return effects({"cc"}, set())
  if mnemonic == "exrl":
    return effects(set(registers), set(registers), unknown=True)
  if mnemonic in {"lmg", "lm"} and len(first) == 1:
    last = s390x_registers(operands[1]) if len(operands) > 1 else []
    if last and last[0].startswith("r"):
      defs = {f"r{number}" for number in range(int(first[0][1:]), int(last[0][1:]) + 1)}
      return effects(set(s390x_registers(operands[-1])), defs, load=True)
  if mnemonic == "ipm":
    return effects({"cc"} | set(first), set(first))
  if mnemonic.startswith("v") and registers:
    address = set(s390x_registers("".join(operand for operand in operands if "(" in operand)))
    if mnemonic in S390X_VECTOR_STORES:
      return effects(set(registers), set())
    if mnemonic in S390X_VECTOR_LOADS:
      # Element loads keep the other elements, so they also read the destination.
      reads = set(first) if mnemonic.startswith("vle") else set()
      return effects(address | reads, set(first), load=not constant)
    # Register-to-register vector operations; reading every named register over-approximates uses.
    return effects(set(registers), set(first), unknown=mnemonic not in S390X_KNOWN_VECTOR)
  if mnemonic in S390X_STORES:
    return effects(set(registers) | ({"cc"} if mnemonic in {"stoc", "stocg"} else set()), set())
  # Address registers inside parentheses are read; only the first operand can be defined.
  address = set(s390x_registers("".join(operand for operand in operands if "(" in operand)))
  sources = set(s390x_registers(", ".join(operands[1:])))
  load = memory and not constant and mnemonic not in {"la", "lay", "larl"}
  if mnemonic in S390X_COMPARES:
    return effects(set(registers), {"cc"}, load=load)
  if mnemonic in {"mlgr", "mlg", "dlgr", "dlg", "dsgr", "dsg"}:
    pair = {first[0], f"r{int(first[0][1:]) + 1}"} if first else set()
    return effects(pair | sources | address, pair, load=load)
  if mnemonic in S390X_NO_CC or mnemonic in S390X_CC_SETTERS:
    uses = sources | address | (set() if mnemonic in S390X_FULL_DEFS else set(first))
    if mnemonic.startswith("loc") or mnemonic in {"alc", "alcr", "alcg", "alcgr", "slb", "slbr", "slbg", "slbgr"}:
      uses.add("cc")
    defs = set(first) | ({"cc"} if mnemonic in S390X_CC_SETTERS else set())
    return effects(uses, defs, load=load)
  return effects(set(registers), set(first) | {"cc"}, load=load, unknown=True)


# Memory ---------------------------------------------------------------------
#
# A slice follows a loaded value back to the stores that can write it. Each ISA
# model below reports an instruction's memory accesses (base and index
# registers, displacement, width when known) and its pointer steps
# (`destination = source + constant`, including base-register writeback). An
# access it cannot locate is a store to an unknown address.


@dataclass(frozen=True)
class Access:
  store: bool
  base: str | None
  index: str | None = None
  displacement: int | None = 0
  width: int | None = None


@dataclass(frozen=True)
class Memory:
  accesses: tuple[Access, ...] = ()
  # `(destination, source, delta)`: destination = source + delta; a `None` delta keeps the source's root at an
  # unknown offset.
  steps: tuple[tuple[str, str, int | None], ...] = ()
  wild_store: bool = False


NO_MEMORY = Memory()


def immediate(text: str) -> int | None:
  text = text.strip().lstrip("#$")
  try:
    return int(text, 0)
  except ValueError:
    return None


X86_MEMORY = re.compile(r"^(?:%[a-z]s:)?(?P<displacement>-?(?:0x)?[0-9a-f]+)?\((?P<base>%[a-z0-9]+)?"
                        r"(?:,\s*(?P<index>%[a-z0-9]+)?(?:,\s*\d+)?)?\)$")
X86_SUFFIX_WIDTH = {"b": 1, "w": 2, "l": 4, "q": 8}
X86_STORE_ONLY = {"mov", "movabs", "set", "movnti"}
X86_LOAD_ONLY = X86_COMPARES | {"movzx", "movsx", "movsxd", "push", "imul", "mul", "div", "idiv", "crc32", "popcnt",
                                "lzcnt", "tzcnt", "bsf", "bsr", "cmov", "andn", "bextr", "bzhi", "pext", "pdep", "mulx"}
X86_STRING = re.compile(r"^(?:stos|movs|cmps|scas|lods)[bwlq]?$")


def x86_register_width(operand: str) -> int | None:
  name = operand.lstrip("%")
  if re.fullmatch(r"[xyz]mm\d+", name):
    return {"x": 16, "y": 32, "z": 64}[name[0]]
  if re.fullmatch(r"r(?:[abcd]x|[sd]i|[sb]p|\d+)", name):
    return 8
  if re.fullmatch(r"e(?:[abcd]x|[sd]i|[sb]p)|r\d+d", name):
    return 4
  if re.fullmatch(r"[abcd]x|[sd]i|[sb]p|r\d+w", name):
    return 2
  if re.fullmatch(r"[abcd][lh]|[sd]il|[sb]pl|r\d+b", name):
    return 1
  return None


def x86_memory(row: scanner.DecodedLine) -> Memory:
  mnemonic = row.mnemonic
  # `rep stosq` decodes as mnemonic `rep` with the string operation first in its operands.
  string = next((word for word in f"{mnemonic} {row.operands}".split()[:2] if X86_STRING.match(word)), None)
  if string is not None:
    return Memory(wild_store=string.startswith(("stos", "movs")))
  base_name = x86_base(mnemonic)
  operands = split_operands(row.operands)
  registers = [operand for operand in operands if operand.startswith("%")]
  if base_name == "push":
    loads = x86_memory_operands(operands, store=False)
    return Memory((*loads, Access(True, "rsp", None, -8, 8)), (("rsp", "rsp", -8),))
  if base_name == "pop":
    stores = x86_memory_operands(operands, store=True)
    return Memory((Access(False, "rsp", None, 0, 8), *stores), (("rsp", "rsp", 8),))
  if base_name == "lea" and len(operands) == 2:
    match = X86_MEMORY.match(operands[0])
    destination = X86_ALIASES.get(operands[1].lstrip("%"))
    if match and destination and x86_register_width(operands[1]) == 8:
      base = X86_ALIASES.get((match.group("base") or "").lstrip("%"))
      if base and not match.group("index"):
        return Memory(steps=((destination, base, immediate(match.group("displacement") or "0")),))
      if base:
        return Memory(steps=((destination, base, None),))
    return NO_MEMORY
  if not any("(" in operand for operand in operands):
    steps: tuple[tuple[str, str, int | None], ...] = ()
    destination = X86_ALIASES.get(operands[-1].lstrip("%")) if operands else None
    wide = bool(operands) and x86_register_width(operands[-1]) == 8
    if destination and wide and base_name in {"mov", "movq"} and len(operands) == 2 and operands[0].startswith("%"):
      source = X86_ALIASES.get(operands[0].lstrip("%"))
      if source:
        steps = ((destination, source, 0),)
    elif destination and wide and base_name in {"add", "sub"} and len(operands) == 2 and operands[0].startswith("$"):
      delta = immediate(operands[0])
      steps = ((destination, destination, None if delta is None else (delta if base_name == "add" else -delta)),)
    elif destination and wide and base_name in {"inc", "dec"}:
      steps = ((destination, destination, 1 if base_name == "inc" else -1),)
    return Memory(steps=steps)
  width = X86_SUFFIX_WIDTH.get(mnemonic[-1]) if mnemonic != base_name else None
  if base_name in {"movzx", "movsx"} and len(mnemonic) > 4:
    width = X86_SUFFIX_WIDTH.get(mnemonic[4])
  elif base_name == "movsxd":
    width = 4
  if width is None and registers:
    width = x86_register_width(registers[0])
  destination_is_memory = "(" in operands[-1]
  accesses = []
  for position, operand in enumerate(operands):
    if "(" not in operand:
      continue
    last = position == len(operands) - 1
    if not last or base_name in X86_LOAD_ONLY:
      accesses.extend(x86_memory_operands([operand], store=False, width=width))
    elif base_name in X86_STORE_ONLY or (destination_is_memory and mnemonic.startswith(("vmov", "mov"))):
      accesses.extend(x86_memory_operands([operand], store=True, width=width))
    else:
      # Read-modify-write: arithmetic with a memory destination.
      accesses.extend(x86_memory_operands([operand], store=False, width=width))
      accesses.extend(x86_memory_operands([operand], store=True, width=width))
  return Memory(tuple(accesses))


def x86_memory_operands(operands: list[str], store: bool, width: int | None = 8) -> tuple[Access, ...]:
  accesses = []
  for operand in operands:
    if "(" not in operand:
      continue
    match = X86_MEMORY.match(operand)
    if match is None or operand.startswith("%"):
      accesses.append(Access(store, None, None, None, width))
      continue
    base = match.group("base")
    if base == "%rip" or base is None:
      accesses.append(Access(store, None, None, None, width))
      continue
    index = match.group("index")
    accesses.append(Access(store, X86_ALIASES.get(base.lstrip("%"), base.lstrip("%")),
                           X86_ALIASES.get(index.lstrip("%"), index.lstrip("%")) if index else None,
                           immediate(match.group("displacement") or "0"), width))
  return tuple(accesses)


AARCH64_WIDTH = {"x": 8, "w": 4, "q": 16, "d": 8, "s": 4, "h": 2, "b": 1}


def aarch64_memory(row: scanner.DecodedLine) -> Memory:
  mnemonic = row.mnemonic.split(".")[0]
  operands = split_operands(row.operands)
  memory = re.search(r"\[(?P<inside>[^\]]*)\](?P<writeback>!?)(?:, #(?P<post>[-+]?\w+))?", row.operands) \
    if mnemonic.startswith(("ld", "st")) or AARCH64_ATOMIC.match(mnemonic) else None
  if mnemonic == "prfm":
    return NO_MEMORY
  if memory is None:
    if mnemonic in {"mov", "add", "sub"} and len(operands) >= 2:
      registers = [aarch64_registers(operand) for operand in operands[:2]]
      if not all(len(names) == 1 and names[0].startswith(("x", "sp")) and
                 not operand.startswith("w") for names, operand in zip(registers, operands[:2])):
        return NO_MEMORY
      destination, source = registers[0][0], registers[1][0]
      if mnemonic == "mov" and len(operands) == 2:
        return Memory(steps=((destination, source, 0),))
      if len(operands) >= 3 and operands[2].startswith("#"):
        delta = immediate(operands[2])
        if delta is not None and len(operands) == 4:
          shift = re.fullmatch(r"lsl #(\d+)", operands[3])
          delta = delta << int(shift.group(1)) if shift else None
        if delta is not None:
          return Memory(steps=((destination, source, delta if mnemonic == "add" else -delta),))
    return NO_MEMORY
  inside = split_operands(memory.group("inside"))
  base = aarch64_registers(inside[0])[0] if inside and aarch64_registers(inside[0]) else None
  index = None
  displacement: int | None = 0
  if len(inside) > 1:
    if inside[1].startswith("#"):
      displacement = immediate(inside[1])
    else:
      index = (aarch64_registers(inside[1]) or [None])[0]
  data = [operand for operand in operands if "[" not in operand and "]" not in operand and not operand.startswith("#")]
  width = AARCH64_WIDTH.get(data[0][0]) if data and data[0][:1] in AARCH64_WIDTH else None
  if mnemonic.endswith(("rb", "rsb", "urb", "ursb")):
    width = 1
  elif mnemonic.endswith(("rh", "rsh", "urh", "ursh")):
    width = 2
  elif mnemonic in {"ldrsw", "ldursw"}:
    width = 4
  if mnemonic.startswith(("ldp", "stp", "ldnp", "stnp")) and width is not None:
    width *= 2
  if mnemonic.startswith(("ld1", "ld2", "ld3", "ld4", "st1", "st2", "st3", "st4")) or "xr" in mnemonic \
      or "xp" in mnemonic:
    # Structure accesses and exclusive pairs: the first data register does not give the width.
    width = None
  steps: tuple[tuple[str, str, int | None], ...] = ()
  if base is not None and memory.group("writeback"):
    steps = ((base, base, displacement),)
  elif base is not None and memory.group("post"):
    steps = ((base, base, immediate(memory.group("post"))),)
    displacement = 0
  if AARCH64_ATOMIC.match(mnemonic):
    return Memory((Access(False, base, index, displacement, width), Access(True, base, index, displacement, width)),
                  steps)
  if mnemonic.startswith("st"):
    return Memory((Access(True, base, index, displacement, width),), steps)
  if mnemonic.startswith("ld"):
    return Memory((Access(False, base, index, displacement, width),), steps)
  return Memory(wild_store=True)


RISCV_MEMORY = re.compile(r"(?P<displacement>-?(?:0x)?[0-9a-f]+)?\((?P<base>[a-z0-9]+)\)$")
RISCV_WIDTH = {"b": 1, "h": 2, "w": 4, "d": 8}


def riscv_memory(row: scanner.DecodedLine) -> Memory:
  mnemonic = row.mnemonic
  registers = riscv_registers(row.operands)
  operands = split_operands(row.operands)
  core = mnemonic.removeprefix("c.").removesuffix("sp")
  if mnemonic in RISCV_LOADS or mnemonic in scanner.RISCV_STORES or core.startswith(("lr.", "sc.", "amo")):
    match = RISCV_MEMORY.search(operands[-1]) if operands else None
    if match is None:
      return Memory(wild_store=True)
    letter = core.split(".")[1][0] if "." in core else core.lstrip("f").lstrip("ls")[:1]
    access = functools.partial(Access, base=match.group("base"), index=None,
                               displacement=immediate(match.group("displacement") or "0"),
                               width=RISCV_WIDTH.get(letter))
    if core.startswith("amo"):
      return Memory((access(False), access(True)))
    return Memory((access(mnemonic in scanner.RISCV_STORES or core.startswith("sc.")),))
  if mnemonic in {"addi", "c.addi", "c.addi16sp", "c.addi4spn"} and len(registers) == 2:
    return Memory(steps=((registers[0], registers[1], immediate(operands[-1])),))
  if mnemonic in {"mv", "c.mv"} and len(registers) == 2:
    return Memory(steps=((registers[0], registers[1], 0),))
  return NO_MEMORY


POWER_WIDTH = {"b": 1, "h": 2, "w": 4, "d": 8}
POWER_VECTOR_WIDTH = {"lxv": 16, "stxv": 16, "lxvd2x": 16, "stxvd2x": 16, "lxvw4x": 16, "stxvw4x": 16, "lvx": 16,
                      "stvx": 16, "lxvx": 16, "stxvx": 16, "lxsd": 8, "stxsd": 8, "lxsdx": 8, "stxsdx": 8, "lfd": 8,
                      "stfd": 8, "lfdx": 8, "stfdx": 8, "lxssp": 4, "stxssp": 4, "lfs": 4, "stfs": 4, "lxsiwzx": 4,
                      "lxvdsx": 8}


def power_memory(row: scanner.DecodedLine) -> Memory:
  mnemonic = row.mnemonic.rstrip("+-")
  plain = mnemonic.rstrip(".")
  numbers = power_numbers(row.operands)
  vector = plain in POWER_VECTOR_MEMORY or plain in POWER_VECTOR_STORE
  if plain in POWER_LOADS or plain in POWER_STORES or vector:
    store = plain in POWER_STORES or plain in POWER_VECTOR_STORE
    if vector:
      width = POWER_VECTOR_WIDTH.get(plain)
    else:
      letter = plain[2] if plain.startswith("st") else plain[1]
      width = POWER_WIDTH.get(letter)
    if "(" in row.operands and len(numbers) >= 3:
      base = None if numbers[2] == "0" else f"r{numbers[2]}"
      access = Access(store, base, None, int(numbers[1]) if base else None, width)
      steps = ((base, base, int(numbers[1])),) if base and plain.endswith("u") else ()
      return Memory((access,), steps)
    if len(numbers) >= 3:
      base = None if numbers[1] == "0" else f"r{numbers[1]}"
      index = f"r{numbers[2]}"
      access = Access(store, base or index, index if base else None, 0, width)
      steps = ((base, base, None),) if base and plain.endswith("ux") else ()
      return Memory((access,), steps)
    return Memory(wild_store=True)
  if plain in {"addi", "addis"} and len(numbers) == 3 and numbers[1] != "0":
    delta = int(numbers[2]) << (16 if plain == "addis" else 0)
    return Memory(steps=((f"r{numbers[0]}", f"r{numbers[1]}", delta),))
  if plain == "mr" and len(numbers) == 2:
    return Memory(steps=((f"r{numbers[0]}", f"r{numbers[1]}", 0),))
  if plain in {"stmw", "lmw", "stswi", "lswi", "dcbz"}:
    return Memory(wild_store=True)
  return NO_MEMORY


S390X_STORAGE = re.compile(r"(?P<displacement>-?\d+)\((?:(?P<inner>%r\d+|\d+),)?(?P<base>%r\d+)\)")
S390X_WIDTH = {
  **dict.fromkeys(("lg", "stg", "ltg", "alg", "ag", "sg", "slg", "cg", "clg", "ng", "og", "xg", "lrvg", "alcg",
                   "slbg", "msg", "mlg", "dlg", "dsg", "mvghi", "lgat", "stocg", "locg", "vleg", "vsteg", "vlrepg",
                   "cghsi", "clghsi"), 8),
  **dict.fromkeys(("l", "ly", "st", "sty", "a", "al", "s", "sl", "c", "cl", "n", "o", "x", "lt", "llgf", "lgf",
                   "ltgf", "lrv", "agf", "algf", "sgf", "slgf", "cgf", "clgf", "msgf", "mvhi", "ms", "alc", "slb",
                   "llgt", "vlef", "vstef", "vlrepf", "chsi", "clfhsi"), 4),
  **dict.fromkeys(("lh", "lhy", "sth", "sthy", "llh", "llgh", "lgh", "mvhhi", "ch", "chy", "chhsi", "clhhsi"), 2),
  **dict.fromkeys(("llc", "llgc", "lgb", "lb", "stc", "stcy", "ic", "icy", "cli", "cliy", "mvi", "tm"), 1),
  **dict.fromkeys(("vl", "vst"), 16),
}


S390X_STORAGE_COMPARES = {"chsi", "chhsi", "cghsi", "clhhsi", "clghsi", "clfhsi", "cli", "cliy", "tm", "tmy"}


def s390x_memory(row: scanner.DecodedLine) -> Memory:
  mnemonic = row.mnemonic
  if mnemonic in {"pfd", "pfdrl"}:
    # A prefetch reads no value and writes no memory.
    return NO_MEMORY
  operands = split_operands(row.operands)
  storage = [S390X_STORAGE.search(operand) for operand in operands]
  located = [(position, match) for position, match in enumerate(storage) if match]
  if not located:
    if mnemonic == "lgr" and len(operands) == 2:
      registers = [s390x_registers(operand) for operand in operands]
      return Memory(steps=((registers[0][0], registers[1][0], 0),))
    if mnemonic in {"aghi", "agfi"} and len(operands) == 2:
      register = s390x_registers(operands[0])[0]
      return Memory(steps=((register, register, immediate(operands[1])),))
    return NO_MEMORY

  def access(match: re.Match[str], store: bool, width: int | None) -> Access:
    inner = match.group("inner")
    index = inner.lstrip("%") if inner and inner.startswith("%r") and inner != "%r0" else None
    return Access(store, match.group("base").lstrip("%"), index, int(match.group("displacement")), width)

  if mnemonic in {"la", "lay"}:
    match = located[0][1]
    destination = s390x_registers(operands[0])[0]
    if match.group("inner") and match.group("inner") != "%r0":
      return Memory(steps=((destination, match.group("base").lstrip("%"), None),))
    return Memory(steps=((destination, match.group("base").lstrip("%"), int(match.group("displacement"))),))
  if mnemonic in {"mvc", "xc", "nc", "oc", "clc"} and len(located) == 2:
    first, second = located[0][1], located[1][1]
    length = int(first.group("inner")) if first.group("inner") and first.group("inner").isdigit() else None
    destination = Access(True, first.group("base").lstrip("%"), None, int(first.group("displacement")), length)
    reads = [access(second, False, length)]
    if mnemonic != "mvc":
      reads.append(Access(False, destination.base, None, destination.displacement, length))
    return Memory((*reads, destination) if mnemonic != "clc" else tuple(reads))
  if mnemonic in {"stmg", "lmg", "stm", "lm"} and len(operands) == 3:
    first, last = (int(name[1:]) for name in (s390x_registers(operands[0])[0], s390x_registers(operands[1])[0]))
    width = (last - first + 1) * (8 if mnemonic.endswith("g") else 4) if last >= first else None
    return Memory((access(located[0][1], mnemonic.startswith("st"), width),))
  match = located[0][1]
  width = S390X_WIDTH.get(mnemonic)
  if mnemonic in S390X_STORES or mnemonic in S390X_VECTOR_STORES:
    return Memory((access(match, True, width),))
  if (mnemonic in S390X_VECTOR_LOADS or mnemonic in S390X_NO_CC or mnemonic in S390X_CC_SETTERS
      or mnemonic in S390X_STORAGE_COMPARES):
    return Memory((access(match, False, width),))
  return Memory(wild_store=True)


def isa_memory(target: str) -> Any:
  if target.startswith(("x86_64-", "i686-")):
    return x86_memory
  if target.startswith("aarch64-"):
    return aarch64_memory
  if target.startswith("riscv64"):
    return riscv_memory
  if target.startswith(("powerpc64", "ppc64")):
    return power_memory
  return s390x_memory


def argument_registers(target: str) -> set[str]:
  """Registers that can carry a pointer argument into a call."""
  if target.startswith("x86_64-") and "-windows-" in target:
    return {"rcx", "rdx", "r8", "r9"}
  if target.startswith(("x86_64-", "i686-")):
    return {"rdi", "rsi", "rdx", "rcx", "r8", "r9"}
  if target.startswith("aarch64-"):
    return {f"x{n}" for n in range(8)}
  if target.startswith("riscv64"):
    return {f"a{n}" for n in range(8)}
  if target.startswith(("powerpc64", "ppc64")):
    return {f"r{n}" for n in range(3, 11)}
  return {f"r{n}" for n in range(2, 7)}


def stack_pointer(target: str) -> str:
  """The register whose value is the frame address; a slice records it as a leaf."""
  if target.startswith(("x86_64-", "i686-")):
    return "rsp"
  if target.startswith(("powerpc64", "ppc64")):
    return "r1"
  if target.startswith("s390x-"):
    return "r15"
  return "sp"


def isa_effects(target: str) -> Any:
  if target.startswith("x86_64-") and "-windows-" in target:
    return functools.partial(x86_effects, caller_saved=X86_WIN64_CALLER_SAVED)
  if target.startswith(("x86_64-", "i686-")):
    return x86_effects
  if target.startswith("aarch64-"):
    return aarch64_effects
  if target.startswith("riscv64"):
    return riscv_effects
  if target.startswith(("powerpc64", "ppc64")):
    return power_effects
  if target.startswith("s390x-"):
    return s390x_effects
  raise ValueError(f"no condition-slice model for {target}")


@dataclass(frozen=True)
class Value:
  """A register value as `root + offset`.

  `root` is the definition that produced the value, `(ENTRY, register)` for a function input, or `None` after
  paths with different roots merge; `offset` is `None` when unknown. `frame` and `other` record whether the
  value may point into this function's frame or anywhere else.
  """

  root: tuple[int, str] | None
  offset: int | None
  frame: bool
  other: bool


def entry_value(register: str, frame: str) -> Value:
  return Value((ENTRY, register), 0, register == frame, register != frame)


def join_values(left: Value, right: Value) -> Value:
  if left == right:
    return left
  root = left.root if left.root == right.root else None
  return Value(root, None, left.frame or right.frame, left.other or right.other)


def pointer_values(rows: list[scanner.DecodedLine], successors: list[list[int] | None], row_effects: list[Effects],
                   row_memory: list[Memory], frame: str) -> list[dict[str, Value] | None]:
  """The value of each register before each row; an absent register still holds its entry value."""
  states: list[dict[str, Value] | None] = [None] * len(rows)
  if not rows:
    return states
  states[0] = {}
  pending = [0]
  while pending:
    index = pending.pop()
    state = states[index]
    assert state is not None
    after = dict(state)
    stepped = set()
    for destination, source, delta in row_memory[index].steps:
      value = state.get(source) or entry_value(source, frame)
      offset = None if value.offset is None or delta is None else value.offset + delta
      after[destination] = Value(value.root, offset, value.frame, value.other)
      stepped.add(destination)
    for register in row_effects[index].defs - stepped:
      after[register] = Value((index, register), 0, False, True)
    for successor in successors[index] or []:
      if successor >= len(rows):
        continue
      current = states[successor]
      if current is None:
        states[successor] = after
        pending.append(successor)
        continue
      merged = dict(current)
      for register in set(current) | set(after):
        merged[register] = join_values(current.get(register) or entry_value(register, frame),
                                       after.get(register) or entry_value(register, frame))
      if merged != current:
        states[successor] = merged
        pending.append(successor)
  return states


def cyclic_rows(successors: list[list[int] | None], count: int) -> set[int]:
  """Rows on a control-flow cycle, by Tarjan's strongly connected components."""
  index_of: dict[int, int] = {}
  low: dict[int, int] = {}
  stack: list[int] = []
  on_stack: set[int] = set()
  cyclic: set[int] = set()
  counter = 0
  for start in range(count):
    if start in index_of:
      continue
    work = [(start, 0)]
    while work:
      node, position = work.pop()
      if position == 0:
        index_of[node] = low[node] = counter
        counter += 1
        stack.append(node)
        on_stack.add(node)
      following = [successor for successor in successors[node] or [] if successor < count]
      if position < len(following):
        work.append((node, position + 1))
        successor = following[position]
        if successor not in index_of:
          work.append((successor, 0))
        elif successor in on_stack:
          low[node] = min(low[node], index_of[successor])
        continue
      for successor in following:
        if successor in on_stack:
          low[node] = min(low[node], low[successor])
      if low[node] == index_of[node]:
        component = []
        while True:
          member = stack.pop()
          on_stack.discard(member)
          component.append(member)
          if member == node:
            break
        if len(component) > 1 or node in following:
          cyclic.update(component)
  return cyclic


FLAG_REGISTERS = {"flags", "nzcv", "cc", *(f"cr{n}" for n in range(8))}


def frame_escapes(target: str, rows: list[scanner.DecodedLine], row_effects: list[Effects],
                  row_memory: list[Memory], states: list[dict[str, Value] | None], frame: str) -> bool:
  """Whether a frame address can reach memory or code that this function's slices cannot see.

  An address of the locals escapes when it is stored as data, passed in an argument register to a call, or used
  by an instruction other than an address computation, a pointer step, or a comparison.
  """
  arguments = argument_registers(target)
  for index, effect in enumerate(row_effects):
    state = states[index]
    if state is None:
      continue

    def framed(register: str) -> bool:
      # A value at or above the entry stack pointer, such as the POWER or IBM Z back chain, addresses the
      # caller's area, not this function's locals.
      value = state.get(register) or entry_value(register, frame)
      return value.frame and (value.offset is None or value.offset < 0)

    if effect.call:
      if any(framed(register) for register in arguments):
        return True
      continue
    memory = row_memory[index]
    allowed = {register for access in memory.accesses for register in (access.base, access.index) if register}
    allowed |= {source for _, source, _ in memory.steps}
    data = {register for register in effect.uses - allowed if framed(register)}
    if not data:
      continue
    stores = any(access.store for access in memory.accesses)
    if stores or memory.wild_store or not effect.defs <= FLAG_REGISTERS:
      return True
  return False


@dataclass(frozen=True)
class Location:
  frame: bool
  other: bool
  root: tuple[int, str] | None
  start: int | None
  width: int | None


def locate(access: Access, state: dict[str, Value] | None, frame: str) -> Location:
  if access.base is None:
    return Location(False, True, None, None, access.width)
  value = (state or {}).get(access.base) or entry_value(access.base, frame)
  known = value.offset is not None and access.displacement is not None and access.index is None
  return Location(value.frame, value.other, value.root, value.offset + access.displacement if known else None,
                  access.width)


def alias(store: Location, load: Location, escaped: bool, unique: Any) -> str | None:
  """`must` when the store writes every loaded byte, `may` when it can overlap them, else None."""
  if not escaped and not ((store.frame and load.frame) or (store.other and load.other)):
    return None
  definite = store.frame != store.other and load.frame != load.other
  if (not definite or store.root is None or store.root != load.root or not unique(store.root)
      or None in (store.start, load.start, store.width, load.width)):
    return "may"
  if store.start <= load.start and load.start + load.width <= store.start + store.width:
    # The store writes every loaded byte, so earlier stores cannot reach the load.
    return "must"
  if store.start + store.width <= load.start or load.start + load.width <= store.start:
    return None
  return "may"


@dataclass
class Slice:
  """A canonical backward slice of one branch condition."""

  canonical: str
  nodes: list[dict[str, Any]] = field(default_factory=list)
  reasons: list[str] = field(default_factory=list)


def strip_offset(identity: str) -> str:
  return re.sub(r"^[0-9a-f]+: ", "", identity)


def branch_signature(identity: str) -> str:
  """Branch identity without position, encoding, and local destination.

  A select-shaped branch always targets its own function, so the destination
  offset and the displacement bytes only record where the arms were placed.
  """
  text = strip_offset(identity)
  if exact := re.fullmatch(r"((?:[0-9a-fA-F]{2,8}\s+)+)(\S.*)", text):
    length = sum(len(word) // 2 for word in exact.group(1).split())
    text = f"{length}B {' '.join(exact.group(2).split())}"
  return re.sub(r"\s*(?:0x[0-9a-fA-F]+\s+)?<.*>$|\s+0x[0-9a-fA-F]+$", " <local>", text)


def condition_slice(target: str, body: scanner.FunctionBody, image: Any, line_number: int,
                    local_symbols: set[str]) -> Slice:
  """Compute the canonical backward slice of the branch on `line_number`."""
  return FunctionSlices(target, body, image, local_symbols).slice(line_number)


class FunctionSlices:
  """The per-function analysis that every slice of one function shares."""

  def __init__(self, target: str, body: scanner.FunctionBody, image: Any, local_symbols: set[str]) -> None:
    rows = self.rows = scanner.decode_lines(body)
    self.identities = scanner.instruction_identities(target, body, local_symbols, image)
    self.frame = stack_pointer(target)
    self.index_by_line = {row.number: index for index, row in enumerate(rows)}
    reasons = self.reasons = []
    if any(scanner.is_register_jump(target, row) for row in rows):
      reasons.append("function contains an indirect jump; its control-flow graph is incomplete")
    successors = scanner.row_successors(target, body, rows)
    predecessors = self.predecessors = {index: [] for index in range(len(rows))}
    if rows:
      predecessors[0].append(ENTRY)
    for index, following in enumerate(successors):
      if following is None:
        if not scanner.is_register_jump(target, rows[index]):
          reasons.append(f"unresolved local branch at +0x{rows[index].offset:x}")
        continue
      for successor in following:
        if successor < len(rows):
          predecessors[successor].append(index)
    model = isa_effects(target)
    self.row_effects = [model(row, self.identities[row.number]) for row in rows]
    self.row_memory = [isa_memory(target)(row) for row in rows]
    self.states = pointer_values(rows, successors, self.row_effects, self.row_memory, self.frame)
    self.escaped = frame_escapes(target, rows, self.row_effects, self.row_memory, self.states, self.frame)
    self.cyclic = cyclic_rows(successors, len(rows))

  def slice(self, line_number: int) -> Slice:
    rows, identities, frame = self.rows, self.identities, self.frame
    predecessors, row_effects, row_memory = self.predecessors, self.row_effects, self.row_memory
    states, escaped, cyclic = self.states, self.escaped, self.cyclic
    if line_number not in self.index_by_line:
      return Slice("", reasons=["branch line not decoded"])
    reasons = list(self.reasons)
    index_by_line = self.index_by_line

    def unique(root: tuple[int, str]) -> bool:
      # A root defined outside every loop holds one value per call.
      return root[0] == ENTRY or root[0] not in cyclic

    def memory_reaching(index: int, location: Location) -> tuple[list[int], list[str]]:
      """The stores that can write the loaded bytes, and `ENTRY` when entry memory can reach the load."""
      found, open_reasons, pending, seen = set(), [], list(predecessors[index]), set()
      while pending:
        current = pending.pop()
        if current in seen:
          continue
        seen.add(current)
        if current == ENTRY:
          found.add(ENTRY)
          continue
        memory = row_memory[current]
        where = f"+0x{rows[current].offset:x}"
        if row_effects[current].call and (location.other or (location.frame and escaped)):
          open_reasons.append(f"memory may be written by the call at {where}")
        elif memory.wild_store:
          open_reasons.append(f"memory may be written at an unknown address at {where}")
        else:
          killed = False
          for access in memory.accesses:
            if not access.store:
              continue
            relation = alias(locate(access, states[current], frame), location, escaped, unique)
            if relation:
              found.add(current)
              killed = killed or relation == "must"
          if killed:
            continue
        pending.extend(predecessors[current])
      if ENTRY in found and location.frame and (location.start is None or location.start < 0):
        open_reasons.append(f"frame memory without a reaching store at +0x{rows[index].offset:x}")
      return sorted(found), open_reasons

    def reaching(index: int, register: str) -> list[int]:
      found, pending, seen = set(), list(predecessors[index]), set()
      while pending:
        current = pending.pop()
        if current in seen:
          continue
        seen.add(current)
        if current == ENTRY:
          found.add(ENTRY)
          continue
        if register in row_effects[current].defs:
          found.add(current)
          continue
        pending.extend(predecessors[current])
      return sorted(found)

    order: dict[int, int] = {}
    nodes: list[dict[str, Any]] = []
    branch_index = index_by_line[line_number]
    pending = [branch_index]
    while pending:
      index = pending.pop(0)
      if index in order:
        continue
      if len(order) >= MAX_SLICE_NODES:
        reasons.append("slice exceeds the node limit")
        break
      order[index] = len(order)
      row = rows[index]
      effect = row_effects[index]
      inputs = {}
      for register in sorted(effect.uses):
        if register == frame:
          # Stack addresses follow the frame layout, which the instruction identities already bind.
          inputs[register] = [FRAME]
          continue
        definitions = reaching(index, register)
        inputs[register] = definitions
        pending.extend(definition for definition in definitions if definition >= 0)
      if effect.load:
        loads = [access for access in row_memory[index].accesses if not access.store]
        if not loads:
          reasons.append(f"non-constant load at +0x{row.offset:x}: {row.mnemonic} {row.operands}")
        else:
          # The loaded value is a leaf only at entry memory; otherwise the reaching stores join the slice.
          definitions = set()
          for access in loads:
            found, open_reasons = memory_reaching(index, locate(access, states[index], frame))
            definitions.update(found)
            reasons.extend(open_reasons)
          inputs["mem"] = sorted(definitions)
          pending.extend(definition for definition in definitions if definition >= 0)
      nodes.append({"index": index, "offset": row.offset, "identity": identities[row.number], "inputs": inputs,
                    "load": effect.load, "call": effect.call, "unknown": effect.unknown})
      if index != branch_index:
        if effect.call:
          reasons.append(f"call result at +0x{row.offset:x}: {row.mnemonic} {row.operands}")
        if effect.unknown:
          reasons.append(f"unknown register effects at +0x{row.offset:x}: {row.mnemonic} {row.operands}")
      elif effect.unknown:
        reasons.append(f"unknown branch condition at +0x{row.offset:x}: {row.mnemonic} {row.operands}")

    lines = []
    for node in nodes:
      text = branch_signature(node["identity"]) if node["index"] == branch_index else strip_offset(node["identity"])
      inputs = ";".join(
        f"{register}<-" + ",".join({ENTRY: "entry", FRAME: "frame"}.get(definition) or f"#{order[definition]}"
                                    for definition in definitions if definition < 0 or definition in order)
        for register, definitions in sorted(node["inputs"].items()))
      lines.append(f"#{order[node['index']]} {text} [{inputs}]")
    for node in nodes:
      node["inputs"] = {register: [{ENTRY: "entry", FRAME: "frame"}.get(definition) or f"+0x{rows[definition].offset:x}"
                                   for definition in definitions] for register, definitions in node["inputs"].items()}
    return Slice("\n".join(lines), nodes, sorted(set(reasons)))
