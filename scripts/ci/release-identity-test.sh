#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGER="$SCRIPT_DIR/package-release-source.sh"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

fixture="$TMP_ROOT/repository"
git -C "$TMP_ROOT" init -q -b main repository
git -C "$fixture" config user.email test@example.com
git -C "$fixture" config user.name "Release Source Test"
git -C "$fixture" config commit.gpgsign false
git -C "$fixture" config tag.gpgsign false
cat >"$fixture/Cargo.toml" <<'EOF'
[package]
name = "rscrypto"
version = "1.2.3"
edition = "2024"
EOF
printf 'version = 4\n' >"$fixture/Cargo.lock"
mkdir -p "$fixture/.github/workflows"
printf '[toolchain]\nchannel = "1.98.0"\n' >"$fixture/rust-toolchain.toml"
printf 'name: Release\n' >"$fixture/.github/workflows/release.yaml"
git -C "$fixture" add .
git -C "$fixture" commit -qm release
commit=$(git -C "$fixture" rev-parse HEAD)
git -C "$fixture" tag -a v1.2.3 -m "release v1.2.3"

"$PACKAGER" --root "$fixture" --version 1.2.3 --tag v1.2.3 \
  --commit "$commit" --out "$TMP_ROOT/first" >/dev/null
"$PACKAGER" --root "$fixture" --version 1.2.3 --tag v1.2.3 \
  --commit "$commit" --out "$TMP_ROOT/second" >/dev/null
cmp "$TMP_ROOT/first/rscrypto-1.2.3-source.tar.gz" \
  "$TMP_ROOT/second/rscrypto-1.2.3-source.tar.gz"

git -C "$fixture" commit --allow-empty -qm moved
git -C "$fixture" tag -fa v1.2.3 -m moved >/dev/null
if "$PACKAGER" --root "$fixture" --version 1.2.3 --tag v1.2.3 \
  --commit "$commit" --out "$TMP_ROOT/moved" >/dev/null 2>&1; then
  echo "source packager accepted a moved release tag" >&2
  exit 1
fi

echo "Release source identity regression tests passed"
