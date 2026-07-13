#!/usr/bin/env bash
set -euo pipefail

# Fast dependency-update gate for Dependabot PRs. This intentionally avoids the
# full platform matrix; expensive architecture evidence belongs to normal CI,
# weekly CI, and manual runs.

BASE_SHA="${GITHUB_BASE_SHA:-${1:-}}"

changed_files() {
  if [[ -n "$BASE_SHA" ]] && git rev-parse --verify "$BASE_SHA^{commit}" >/dev/null 2>&1; then
    git diff --name-only "$BASE_SHA...HEAD"
    return 0
  fi

  if [[ -n "${GITHUB_BASE_REF:-}" ]] && git rev-parse --verify "origin/${GITHUB_BASE_REF}^{commit}" >/dev/null 2>&1; then
    git diff --name-only "origin/${GITHUB_BASE_REF}...HEAD"
    return 0
  fi

  git diff --name-only HEAD^..HEAD
}

add_manifest() {
  local manifest=$1

  [[ -f "$manifest" ]] || return 0

  local existing
  for existing in "${MANIFESTS[@]-}"; do
    [[ "$existing" == "$manifest" ]] && return 0
  done

  MANIFESTS+=("$manifest")
}

run_root_smoke() {
  echo "Root dependency smoke"
  cargo fetch --locked
  cargo check --locked --workspace --all-targets --all-features
  cargo test --locked --workspace --all-features --no-run
  cargo deny check advisories
}

run_manifest_smoke() {
  local manifest=$1
  local lockfile
  lockfile="$(dirname "$manifest")/Cargo.lock"

  echo "Standalone dependency smoke: $manifest"
  if [[ ! -f "$lockfile" ]]; then
    cargo generate-lockfile --manifest-path "$manifest"
  fi
  cargo fetch --manifest-path "$manifest" --locked
  cargo check --manifest-path "$manifest" --locked --all-targets --all-features
  cargo test --manifest-path "$manifest" --locked --all-features --no-run
}

run_automation_smoke() {
  echo "GitHub Actions dependency smoke"
  scripts/ci/pin-actions.sh --verify-only
  scripts/ci/pin-actions-test.sh
  scripts/ci/check-ci-ownership.sh
  scripts/ci/check-ci-ownership-test.sh
  actionlint
  zizmor .github/workflows .github/actions
}

classify_changed_files() {
  RUN_ROOT=false
  RUN_AUTOMATION=false
  MANIFESTS=()
  UNSUPPORTED=()

  local path
  for path in "${CHANGED[@]}"; do
    case "$path" in
      Cargo.toml | Cargo.lock)
        RUN_ROOT=true
        ;;
      fuzz/Cargo.toml | fuzz/Cargo.lock | fuzz/support/Cargo.toml)
        add_manifest "fuzz/Cargo.toml"
        ;;
      fuzz-packages/*/Cargo.toml | fuzz-packages/*/Cargo.lock)
        add_manifest "${path%/*}/Cargo.toml"
        ;;
      tools/*/Cargo.toml | tools/*/Cargo.lock)
        add_manifest "${path%/*}/Cargo.toml"
        ;;
      .github/actions-lock.yaml | .github/dependabot.yaml | .github/workflows/* | .github/actions/*)
        RUN_AUTOMATION=true
        ;;
      *)
        UNSUPPORTED+=("$path")
        ;;
    esac
  done
}

main() {
  mapfile -t CHANGED < <(changed_files)

  echo "Changed files:"
  printf '  %s\n' "${CHANGED[@]:-}"

  classify_changed_files

  if [[ ${#UNSUPPORTED[@]} -gt 0 ]]; then
    echo "Unsupported Dependabot changes:" >&2
    printf '  %s\n' "${UNSUPPORTED[@]}" >&2
    return 1
  fi

  if [[ "$RUN_ROOT" != true && "$RUN_AUTOMATION" != true && "${#MANIFESTS[@]}" -eq 0 ]]; then
    echo "Dependabot change set has no recognized dependency surface" >&2
    return 1
  fi

  if [[ "$RUN_ROOT" == true ]]; then
    run_root_smoke
  fi

  local manifest
  for manifest in "${MANIFESTS[@]}"; do
    run_manifest_smoke "$manifest"
  done

  if [[ "$RUN_AUTOMATION" == true ]]; then
    run_automation_smoke
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
