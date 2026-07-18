#!/usr/bin/env bash
set -euo pipefail

crate="${1:-rscrypto}"

status=0
cargo rail release run "$crate" --bump auto --pr --check || status=$?

# cargo-rail uses 1 for check-mode pending mutations. For this local gate,
# "a release plan exists" is success; genuine tool errors still fail.
if [[ "$status" -eq 0 || "$status" -eq 1 ]]; then
  exit 0
fi

exit "$status"
