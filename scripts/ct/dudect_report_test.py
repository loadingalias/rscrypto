#!/usr/bin/env python3
"""Regression tests for architecture-specific DudeCT evidence parsing."""

from __future__ import annotations

from pathlib import Path

from dudect_report import (
  dudect_case_rows,
  linker_driver,
  manifest_dudect_cases,
  owner_call_site_counts,
  owner_symbol_evidence,
)
from full import validate_dudect_case_report, validated_dudect_case_report
from toml_compat import tomllib


def expect_failure(action) -> None:
  try:
    action()
  except ValueError:
    return
  raise AssertionError("malformed DudeCT evidence unexpectedly passed")


def main() -> None:
  root = Path(__file__).resolve().parents[2]
  with (root / "ct.toml").open("rb") as source:
    manifest_cases = manifest_dudect_cases(tomllib.load(source))

  baseline_t_values = {
    "aes128_header_protection_fixed_vs_random_key": 516.75757,
    "aes256_header_protection_fixed_vs_random_key": 424.57239,
    "aes_siv_cmac256_fixed_vs_random_key_seal": 418.20301,
    "aes_siv_cmac256_fixed_vs_random_key_open": 377.36551,
    "aes_siv_cmac256_portable_s2v_fixed_vs_random_key": 85.85045,
    "aes_siv_cmac256_portable_s2v_seal_fixed_vs_random_key": 97.96790,
    "aes_siv_cmac256_portable_open_fixed_vs_random_key": 102.87448,
    "ecdsa_p256_diag_reduce_wide_fixed_vs_random_input": 609.64484,
    "ecdsa_p256_diag_nonce_inverse_blinded_fixed_vs_random_secret": 36.62068,
    "ecdsa_p384_diag_reduce_wide_fixed_vs_random_input": 660.27823,
  }
  baseline_rows = dudect_case_rows(
    {
      name: {
        "abs_max_t": abs(max_t),
        "max_t": max_t,
      }
      for name, max_t in baseline_t_values.items()
    },
    {},
    {},
    manifest_cases,
    threshold=10.0,
    requested_samples=20_000,
  )
  assert len(baseline_rows) == 10
  assert all(row["gate"] == "required" for row in baseline_rows)
  assert sum(row["status"] == "fail" for row in baseline_rows) == 10
  assert sum(row["status"] == "diagnostic-fail" for row in baseline_rows) == 0

  required_case = manifest_cases["ecdsa_p256_diag_reduce_wide_fixed_vs_random_input"]
  validate_dudect_case_report(
    required_case,
    {"name": required_case["name"], "gate": "required", "status": "fail"},
  )
  expect_failure(
    lambda: validate_dudect_case_report(
      required_case,
      {"name": required_case["name"], "gate": "diagnostic", "status": "diagnostic-fail"},
    )
  )
  expect_failure(
    lambda: validate_dudect_case_report(
      required_case,
      {"name": required_case["name"], "gate": "required", "status": "diagnostic-fail"},
    )
  )
  valid_required_row = {"name": required_case["name"], "gate": "required", "status": "fail"}
  assert validated_dudect_case_report(required_case, {"cases": [valid_required_row]}) == valid_required_row
  expect_failure(lambda: validated_dudect_case_report(required_case, {"cases": []}))
  expect_failure(lambda: validated_dudect_case_report(required_case, {"cases": [valid_required_row] * 2}))
  expect_failure(
    lambda: dudect_case_rows(
      {"undeclared_case": {"abs_max_t": 11.0}},
      {},
      {},
      manifest_cases,
      threshold=10.0,
      requested_samples=20_000,
    )
  )

  power_linker_command = (
    'LC_ALL="C" PATH="/usr/local/bin:/usr/bin" VSLANG="1033" "cc" "-m64" '
    '"/tmp/rustc/first.o" "input.o" "-o" "/tmp/rscrypto-ct-dudect"'
  )
  assert linker_driver(power_linker_command) == "cc"
  assert linker_driver('LC_ALL="C" "/usr/bin/clang" "input.o" "-o" "output"') == "/usr/bin/clang"
  darwin_linker_command = (
    'LC_ALL="C" "/usr/bin/env" "-u" "IPHONEOS_DEPLOYMENT_TARGET" '
    '"-u" "TVOS_DEPLOYMENT_TARGET" ZERO_AR_DATE="1" "/usr/bin/cc" '
    '"input.o" "-o" "output"'
  )
  assert linker_driver(darwin_linker_command) == "/usr/bin/cc"
  expect_failure(lambda: linker_driver(""))
  expect_failure(lambda: linker_driver('LC_ALL="C" "-m64" "input.o"'))
  expect_failure(lambda: linker_driver('"/usr/bin/env" "-u"'))

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

  riscv_disassembly = """\
0000000000029000 <rscrypto_ct_dudect::main>:
   292f2: 0015b097      auipc ra, 0x15b
   292f6: 47a080e7      jalr 0x47a(ra) <ct_entry_owner_eq_16>
   299be: 0015b097      auipc ra, 0x15b
   299c2: f2c080e7      jalr -0xd4(ra) <ct_entry_owner_eq_32>
   2a2f6: 0015b097      auipc ra, 0x15b
   2a2fa: 8ee080e7      jalr -0x712(ra) <ct_entry_owner_eq_48>
   2abcc: 0015a097      auipc ra, 0x15a
   2abd0: 446080e7      jalr 0x446(ra) <ct_entry_owner_eq_64>
   2abd4: 8082          ret
"""
  assert owner_call_site_counts(riscv_disassembly, expected, symbols_by_address) == {
    symbol: 1 for symbol in sorted(expected)
  }

  unresolved_riscv = riscv_disassembly.replace(
    "<ct_entry_owner_eq_16>",
    "<unresolved_indirect_target>",
  )
  unresolved_riscv_counts = owner_call_site_counts(
    unresolved_riscv,
    expected,
    {0x47A: "ct_entry_owner_eq_16"},
  )
  assert unresolved_riscv_counts["ct_entry_owner_eq_16"] == 0

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
