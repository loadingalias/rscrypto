#!/usr/bin/env python3
"""Regression tests for architecture-specific DudeCT evidence parsing."""

from __future__ import annotations

from dudect_report import linker_driver, owner_call_site_counts, owner_symbol_evidence


def expect_failure(action) -> None:
  try:
    action()
  except ValueError:
    return
  raise AssertionError("malformed DudeCT evidence unexpectedly passed")


def main() -> None:
  power_linker_command = (
    'LC_ALL="C" PATH="/usr/local/bin:/usr/bin" VSLANG="1033" "cc" "-m64" '
    '"/tmp/rustc/first.o" "input.o" "-o" "/tmp/rscrypto-ct-dudect"'
  )
  assert linker_driver(power_linker_command) == "cc"
  assert linker_driver('LC_ALL="C" "/usr/bin/clang" "input.o" "-o" "output"') == "/usr/bin/clang"
  expect_failure(lambda: linker_driver(""))
  expect_failure(lambda: linker_driver('LC_ALL="C" "-m64" "input.o"'))

  expected = {
    "ct_entry_owner_eq_16",
    "ct_entry_owner_eq_32",
    "ct_entry_owner_eq_48",
    "ct_entry_owner_eq_64",
  }
  symbols = """\
00000000001dad60 T ct_entry_owner_eq_16
00000000001dadf0 T ct_entry_owner_eq_32
00000000001daed0 T ct_entry_owner_eq_48
00000000001daf60 T ct_entry_owner_eq_64
"""
  symbol_counts, symbols_by_address = owner_symbol_evidence(symbols, expected)
  assert symbol_counts == {symbol: 1 for symbol in sorted(expected)}
  assert symbols_by_address[0x1DAD60] == "ct_entry_owner_eq_16"
  malformed_counts, malformed_addresses = owner_symbol_evidence(
    "ct_entry_owner_eq_16\n",
    expected,
  )
  assert malformed_counts == {symbol: 0 for symbol in sorted(expected)}
  assert malformed_addresses == {}

  s390x_disassembly = """\
00000000000382c0 <rscrypto_ct_dudect::main>:
   382f0: c0 e5 00 0d 15 38     brasl %r14, 0x1dad60
   3873a: c0 e5 00 0d 13 5b     brasl %r14, 0x1dadf0
   38b7a: c0 e5 00 0d 11 ab     brasl %r14, 0x1daed0
   38fba: c0 e5 00 0d 0f d3     brasl %r14, 0x1daf60
"""
  assert owner_call_site_counts(s390x_disassembly, expected, symbols_by_address) == {
    symbol: 1 for symbol in sorted(expected)
  }

  unresolved = s390x_disassembly.replace("0x1dad60", "0x1dad61")
  unresolved_counts = owner_call_site_counts(unresolved, expected, symbols_by_address)
  assert unresolved_counts["ct_entry_owner_eq_16"] == 0

  symbolic_disassembly = """\
0000000000001000 <caller>:
    1000: e8 0b 00 00 00 callq 0x1010 <ct_entry_owner_eq_16>
"""
  symbolic_counts = owner_call_site_counts(symbolic_disassembly, expected, {})
  assert symbolic_counts["ct_entry_owner_eq_16"] == 1

  x86_got_disassembly = """\
DYNAMIC RELOCATION RECORDS
00000000003733a0 R_X86_64_RELATIVE        *ABS*+0x1dad60
00000000003733a8 R_X86_64_RELATIVE        *ABS*+0x1dadf0
00000000003733b0 R_X86_64_RELATIVE        *ABS*+0x1daed0
00000000003733b8 R_X86_64_RELATIVE        *ABS*+0x1daf60

0000000000010000 <caller>:
   10010: ff 15 8a 33 36 00       callq *0x36338a(%rip) # 0x3733a0 <writev+0x3733a0>
   10016: ff 15 8c 33 36 00       callq *0x36338c(%rip) # 0x3733a8 <writev+0x3733a8>
   1001c: ff 15 8e 33 36 00       callq *0x36338e(%rip) # 0x3733b0 <writev+0x3733b0>
   10022: ff 15 90 33 36 00       callq *0x363390(%rip) # 0x3733b8 <writev+0x3733b8>
"""
  assert owner_call_site_counts(x86_got_disassembly, expected, symbols_by_address) == {
    symbol: 1 for symbol in sorted(expected)
  }

  missing_relocation = x86_got_disassembly.replace(
    "00000000003733a0 R_X86_64_RELATIVE        *ABS*+0x1dad60\n",
    "",
  )
  missing_relocation_counts = owner_call_site_counts(missing_relocation, expected, symbols_by_address)
  assert missing_relocation_counts["ct_entry_owner_eq_16"] == 0


if __name__ == "__main__":
  main()
