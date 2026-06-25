#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LEDGER="$REPO_ROOT/private/tasks/asm.md"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

ASM_FILES="$TMP_DIR/asm-files.txt"
LEDGER_FILES="$TMP_DIR/ledger-files.txt"

find "$REPO_ROOT/src" -type f \( -name '*.s' -o -name '*.S' \) \
  | sed "s#^$REPO_ROOT/##" \
  | sort >"$ASM_FILES"

grep -oE '`src/[^`]+[.][sS]`' "$LEDGER" \
  | tr -d '`' \
  | sort >"$LEDGER_FILES"

actual_count="$(wc -l <"$ASM_FILES" | tr -d ' ')"
declared_count="$(awk '$3 == "tracked" && $4 == "`.s`" && $6 == "`.S`" { print $2; exit }' "$LEDGER")"
declared_external="$(awk '/explicitly external-derived today/ { print $2; exit }' "$LEDGER")"
declared_owned="$(awk '/explicitly marked `rscrypto-owned`/ { print $2; exit }' "$LEDGER")"
declared_candidates="$(awk '/first-party candidates awaiting headers/ { print $2; exit }' "$LEDGER")"

status=0

error() {
  printf 'asm-ledger: %s\n' "$*" >&2
  status=1
}

if [[ -z "$declared_count" ]]; then
  error "private/tasks/asm.md is missing the tracked .s/.S inventory count"
elif [[ "$actual_count" != "$declared_count" ]]; then
  error "private/tasks/asm.md declares $declared_count assembly files, found $actual_count"
fi

if [[ -z "$declared_external" ]]; then
  error "private/tasks/asm.md is missing the external-derived assembly count"
fi

if [[ -z "$declared_owned" ]]; then
  error "private/tasks/asm.md is missing the rscrypto-owned assembly count"
fi

if [[ -z "$declared_candidates" ]]; then
  error "private/tasks/asm.md is missing the first-party candidate count"
fi

external_count=0
owned_count=0
candidate_count=0

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
    candidate_count=$((candidate_count + 1))
  fi

  ledger_occurrences="$(grep -Fxc "$file" "$LEDGER_FILES" || true)"
  if [[ "$ledger_occurrences" != "1" ]]; then
    error "$file appears $ledger_occurrences times in private/tasks/asm.md; expected exactly once"
  fi
done <"$ASM_FILES"

while IFS= read -r file; do
  if [[ ! -f "$REPO_ROOT/$file" ]]; then
    error "private/tasks/asm.md references missing assembly file $file"
  fi
done <"$LEDGER_FILES"

if [[ -n "$declared_external" && "$external_count" != "$declared_external" ]]; then
  error "private/tasks/asm.md declares $declared_external external-derived files, classified $external_count"
fi

if [[ -n "$declared_owned" && "$owned_count" != "$declared_owned" ]]; then
  error "private/tasks/asm.md declares $declared_owned owned files, classified $owned_count"
fi

if [[ -n "$declared_candidates" && "$candidate_count" != "$declared_candidates" ]]; then
  error "private/tasks/asm.md declares $declared_candidates candidate files, classified $candidate_count"
fi

if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

printf 'asm-ledger: %s assembly files, %s owned, %s external-derived, %s candidates\n' \
  "$actual_count" "$owned_count" "$external_count" "$candidate_count"
