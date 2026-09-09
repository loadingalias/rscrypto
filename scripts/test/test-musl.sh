#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
host=$(scripts/lib/toolchain.sh --print-host)
case "$host" in
  x86_64-unknown-linux-gnu|aarch64-unknown-linux-gnu) ;;
  *) echo "musl runtime tests require a matching native Linux host" >&2; exit 2 ;;
esac
export CARGO_BUILD_TARGET="${host%-gnu}-musl"
# musl-gcc supplies the matching native musl headers and linker specification.
target_key=${CARGO_BUILD_TARGET//-/_}
export "CARGO_TARGET_${target_key^^}_LINKER"=musl-gcc
export "CC_$target_key"=musl-gcc
scripts/test/test.sh --all
scripts/test/test.sh --all --portable
