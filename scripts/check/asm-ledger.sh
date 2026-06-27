#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

ASM_FILES="$TMP_DIR/asm-files.txt"

find "$REPO_ROOT/src" -type f \( -name '*.s' -o -name '*.S' \) \
  | sed "s#^$REPO_ROOT/##" \
  | sort >"$ASM_FILES"

actual_count="$(wc -l <"$ASM_FILES" | tr -d ' ')"

status=0

error() {
  printf 'asm-ledger: %s\n' "$*" >&2
  status=1
}

external_count=0
owned_count=0

while IFS= read -r file; do
  path="$REPO_ROOT/$file"

  if ! grep -q 'SPDX-License-Identifier:' "$path"; then
    error "$file is missing an SPDX-License-Identifier header"
  fi

  owned=false
  external=false

  if grep -Eq '^(//|[[:space:]]*\*) rscrypto-owned' "$path"; then
    owned=true
  fi

  if grep -Eq '^(//|[[:space:]]*\*) Adapted for rscrypto|^[[:space:]]*\* The butterfly schedule is auto-derived from' "$path"; then
    external=true
  fi

  if [[ "$owned" == true && "$external" == true ]]; then
    error "$file has both owned and external-derived file-level markers"
  elif [[ "$owned" == true ]]; then
    owned_count=$((owned_count + 1))
  elif [[ "$external" == true ]]; then
    external_count=$((external_count + 1))
  else
    error "$file is missing a file-level ownership marker"
  fi
done <"$ASM_FILES"

if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

printf 'asm-ledger: %s assembly files, %s owned, %s external-derived\n' \
  "$actual_count" "$owned_count" "$external_count"
