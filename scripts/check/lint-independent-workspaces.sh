#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
POLICY_MANIFEST="$REPO_ROOT/Cargo.toml"
TARGET_DIR="${RSCRYPTO_INDEPENDENT_LINT_TARGET_DIR:-$REPO_ROOT/target/independent-lints}"
MESSAGE_FORMAT="${RSCRYPTO_LINT_MESSAGE_FORMAT:-human}"
LINT_CAP="${RSCRYPTO_LINT_CAP:-}"

case "$MESSAGE_FORMAT" in
  human | json) ;;
  *) echo "RSCRYPTO_LINT_MESSAGE_FORMAT must be human or json" >&2; exit 2 ;;
esac
case "$LINT_CAP" in
  "" | warn) ;;
  *) echo "RSCRYPTO_LINT_CAP must be empty or warn" >&2; exit 2 ;;
esac

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to resolve independent Cargo workspaces" >&2
  exit 1
fi

manifests=()
while IFS= read -r package_manifest; do
  if ! metadata=$(cargo metadata \
    --locked \
    --no-deps \
    --format-version 1 \
    --manifest-path "$package_manifest"); then
    echo "Failed to resolve Cargo workspace for $package_manifest" >&2
    exit 1
  fi
  workspace_root=$(printf '%s\n' "$metadata" | jq -er '.workspace_root')
  workspace_root=$(cd "$workspace_root" && pwd -P)
  if [[ "$workspace_root" == "$REPO_ROOT" ]]; then
    continue
  fi

  workspace_manifest="$workspace_root/Cargo.toml"
  already_listed=false
  for manifest in "${manifests[@]:+${manifests[@]}}"; do
    if [[ "$manifest" == "$workspace_manifest" ]]; then
      already_listed=true
      break
    fi
  done
  if [[ "$already_listed" == false ]]; then
    manifests+=("$workspace_manifest")
  fi
done < <(find "$REPO_ROOT" -type d -name target -prune -o -type f -name Cargo.toml -print | sort)

if [[ ${#manifests[@]} -eq 0 ]]; then
  echo "No independent Cargo workspaces found through Cargo metadata"
  exit 0
fi

lint_names() {
  local namespace=$1
  awk -v section="[lints.$namespace]" '
    /^\[/ {
      active = ($0 == section)
      next
    }
    active {
      line = $0
      sub(/^[[:space:]]*/, "", line)
      if (line ~ /^[a-z0-9_]+[[:space:]]*=/ && line ~ /"deny"/) {
        sub(/[[:space:]]*=.*/, "", line)
        print line
      }
    }
  ' "$POLICY_MANIFEST"
}

lint_flags=()
while IFS= read -r lint; do
  [[ -n "$lint" ]] && lint_flags+=("-D$lint")
done < <(lint_names rust)
while IFS= read -r lint; do
  [[ -n "$lint" ]] && lint_flags+=("-Dclippy::$lint")
done < <(lint_names clippy)

if [[ ${#lint_flags[@]} -eq 0 ]]; then
  echo "No deny-level Rust or Clippy policy found in $POLICY_MANIFEST" >&2
  exit 1
fi

check_cfg_flags=(
  '--check-cfg=cfg(miri)'
  '--check-cfg=cfg(fuzzing)'
  '--check-cfg=cfg(rscrypto_internal_fuzzing)'
  '--check-cfg=cfg(target_feature,values("movdiri","movdir64b","serialize"))'
)

failed=0
for manifest in "${manifests[@]}"; do
  relative_manifest=${manifest#"$REPO_ROOT/"}
  if [[ "$MESSAGE_FORMAT" == json ]]; then
    echo "Linting independent workspace: $relative_manifest" >&2
  else
    echo "Linting independent workspace: $relative_manifest"
  fi

  cargo_args=(
    clippy
    --locked
    --manifest-path "$manifest"
    --workspace
    --all-targets
    --all-features
    --no-deps
  )
  if [[ "$MESSAGE_FORMAT" == json ]]; then
    cargo_args+=(--message-format=json)
  fi

  compiler_flags=("${lint_flags[@]}" "${check_cfg_flags[@]}")
  if [[ -n "$LINT_CAP" ]]; then
    compiler_flags+=(--cap-lints "$LINT_CAP")
  fi

  if ! CARGO_TARGET_DIR="$TARGET_DIR" cargo "${cargo_args[@]}" -- "${compiler_flags[@]}"; then
    failed=1
  fi
done

exit "$failed"
