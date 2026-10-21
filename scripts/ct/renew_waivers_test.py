#!/usr/bin/env -S uv run python
"""Regression tests for applying proposed waiver renewals to ct.toml."""

import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory

from renew_waivers import apply_renewals


def waiver(locator: str, digest: str) -> str:
  return (
    "[[asm_waiver]]\n"
    'target = "riscv64gc-unknown-linux-gnu"\nprimitive = "rsa.private_ops"\nsymbol = "f"\nkind = "select_branch"\n'
    'artifact = "artifacts/rscrypto-ct-evidence.binary.disasm.txt"\n'
    f'locator = "{locator}"\nfunction_sha256 = "{digest}"\nsource = "src/auth/rsa.rs:1"\n'
    'classification = "public"\nrationale = "public length"\nreviewer = "review"\nreviewed_at = "2026-10-15"\n'
  )


def main() -> None:
  with TemporaryDirectory() as temporary:
    path = Path(temporary) / "ct.toml"
    # Two waivers in one function share the old hash; only the proposed one changes.
    path.write_text("[manifest]\nschema_version = 1\n\n" + waiver("f+0x10:select_branch:aa", "11" * 32) + "\n"
                    + waiver("f+0x20:select_branch:bb", "11" * 32))
    reviewed = tomllib.loads(path.read_text())["asm_waiver"][1]
    proposal = {"index": 1, "waiver": reviewed,
                "new": {"locator": "f+0x24:select_branch:bb", "function_sha256": "22" * 32}}
    assert apply_renewals(path, [proposal], "2026-10-21") == 1
    first, second = tomllib.loads(path.read_text())["asm_waiver"]
    assert (first["locator"], first["function_sha256"], first["reviewed_at"]) == (
      "f+0x10:select_branch:aa", "11" * 32, "2026-10-15")
    assert (second["locator"], second["function_sha256"], second["reviewed_at"]) == (
      "f+0x24:select_branch:bb", "22" * 32, "2026-10-21")
    assert second["rationale"] == "public length" and second["reviewer"] == "review"
    # A proposal for a waiver that changed since the record was written must not rewrite another line.
    try:
      apply_renewals(path, [proposal], "2026-10-22")
    except SystemExit as error:
      assert "asm_waiver[1] changed since the renewal record was written" in str(error)
    else:
      raise AssertionError("a stale proposal rewrote ct.toml")


if __name__ == "__main__":
  main()
