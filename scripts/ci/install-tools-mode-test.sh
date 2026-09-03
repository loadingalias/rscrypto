#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLER="$SCRIPT_DIR/install-tools.sh"

for mode in \
  ci \
  supply-chain \
  bench \
  structural-bench \
  profile \
  fuzz \
  coverage \
  ct-linux \
  minimal \
  none; do
  "$INSTALLER" --check-mode "$mode"
done

if "$INSTALLER" --check-mode unsupported >/dev/null 2>&1; then
  echo "install-tools mode check accepted an unsupported profile" >&2
  exit 1
fi
if "$INSTALLER" --check-mode >/dev/null 2>&1; then
  echo "install-tools mode check accepted a missing profile" >&2
  exit 1
fi

for mode in bench profile coverage; do
  mode_body="$(sed -n "/^  $mode)/,/^    ;;/p" "$INSTALLER")"
  if ! grep -q 'ensure_llvm_tools' <<<"$mode_body"; then
    echo "install-tools $mode mode omits LLVM disassembly tools" >&2
    exit 1
  fi
done

if ! grep -Fq 'command -v rustup.exe' "$INSTALLER"; then
  echo "install-tools does not invoke rustup.exe explicitly on Windows" >&2
  exit 1
fi
if ! grep -Fq 'unzip -q "$archive" "$binary" -d "$RSCRYPTO_CARGO_BIN"' "$INSTALLER"; then
  echo "install-tools does not extract Windows just release zip archives with unzip" >&2
  exit 1
fi

ensure_just_body="$(sed -n '/^ensure_just()/,/^}/p' "$INSTALLER")"
if ! grep -q 'install_just_release' <<<"$ensure_just_body"; then
  echo "install-tools does not prefer the pinned just release binary" >&2
  exit 1
fi
for digest in \
  4a5cc2f53e6f0f8c59092a6cc38291eb729d46a7dd95d3ae582008881b84931d \
  748237128c4c40cbdabc65e841d05ceba13cc23a91eaba395495894c1d9764df \
  1cbca0ce9880d5d1050115a6e2ced510927f85d1797a204ef6bccb319d923d8d \
  9a09cfef66aaa79da58203970103a0684307716caaabd3e9844cacc4dc0f4023 \
  50ae3e996c974a0bf32ea7d10f495070df33f1b43e0616b2769e3d4821ed8f48 \
  759f16fb7aa17c5c8b9594b6d4a8c1a6630dfd042cf2b3ff84841454d3d188dc \
  3a39ed629eb67678976c811a4da46f7985a2c22f4dbabe017b8b2eb5ceb5d01c; do
  grep -Fq "$digest" "$INSTALLER" || {
    echo "install-tools omits a pinned just release digest: $digest" >&2
    exit 1
  }
done

echo "Install-tools mode contract tests passed"
