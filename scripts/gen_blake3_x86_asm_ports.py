#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASM_DIR = ROOT / "crates" / "hashes" / "src" / "crypto" / "blake3" / "x86_64" / "asm"

SRC_AVX2 = ASM_DIR / "rscrypto_blake3_avx2_x86-64_unix_linux.s"
SRC_AVX512 = ASM_DIR / "rscrypto_blake3_avx512_x86-64_unix_linux.s"

DST_DARWIN_AVX2 = ASM_DIR / "rscrypto_blake3_avx2_x86-64_apple_darwin.s"
DST_DARWIN_AVX512 = ASM_DIR / "rscrypto_blake3_avx512_x86-64_apple_darwin.s"

DST_WIN_AVX2 = ASM_DIR / "rscrypto_blake3_avx2_x86-64_windows_msvc.s"
DST_WIN_AVX512 = ASM_DIR / "rscrypto_blake3_avx512_x86-64_windows_msvc.s"


def _strip_gnu_stack_note(s: str) -> str:
  return re.sub(r"^\\.section \\.note\\.GNU-stack.*\\n", "", s, flags=re.M)


def _darwin_port(s: str) -> str:
  s = _strip_gnu_stack_note(s)
  # Mach-O assembler rejects ELF-style `.section .text/.rodata`.
  s = s.replace("\n.section .text\n", "\n.text\n")
  s = s.replace("\n.section .rodata\n", "\n.section __TEXT,__const\n")

  # We assemble via Rust `global_asm!`, which uses the `asm!` template parser.
  # The upstream-derived sources already escape EVEX opmask braces as `{{k1}}`
  # so they round-trip correctly. Do not unescape them here.
  s = ".intel_syntax noprefix\n" + s

  # Mach-O uses a leading '_' for external symbols.
  globals_ = re.findall(r"^\\.global\\s+([A-Za-z0-9_]+)\\s*$", s, flags=re.M)

  def repl_global(m: re.Match[str]) -> str:
    sym = m.group(1)
    return f".globl _{sym}"

  s = re.sub(r"^\\.global\\s+([A-Za-z0-9_]+)\\s*$", repl_global, s, flags=re.M)
  for sym in sorted(set(globals_), key=len, reverse=True):
    s = re.sub(rf"^(?P<indent>\\s*){re.escape(sym)}:\\s*$", rf"\\g<indent>_{sym}:", s, flags=re.M)
  return s


def _windows_port(s: str) -> str:
  s = _strip_gnu_stack_note(s)
  s = s.replace("\n.section .text\n", "\n.text\n")
  s = s.replace("\n.section .rodata\n", '\n.section .rdata,"dr"\n')
  # On Windows/COFF, Intel syntax is the default; explicitly switching syntax
  # trips `bad_asm_style` under `-D warnings` in some CI configurations.
  # Clang's COFF assembler accepts both, but normalize for consistency.
  s = re.sub(r"^\\.global\\s+", ".globl ", s, flags=re.M)
  return s


def _write(dst: Path, contents: str) -> None:
  dst.write_text(contents)
  print(f"wrote {dst.relative_to(ROOT)}")


def main() -> None:
  _write(DST_DARWIN_AVX2, _darwin_port(SRC_AVX2.read_text()))
  _write(DST_DARWIN_AVX512, _darwin_port(SRC_AVX512.read_text()))
  _write(DST_WIN_AVX2, _windows_port(SRC_AVX2.read_text()))
  _write(DST_WIN_AVX512, _windows_port(SRC_AVX512.read_text()))


if __name__ == "__main__":
  main()
