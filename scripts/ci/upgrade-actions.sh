#!/usr/bin/env bash
set -euo pipefail

# Upgrade .github/actions-lock.yaml refs to the latest stable tags that are safe
# for current GitHub-hosted runners, then let pin-actions.sh resolve SHAs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOCK_FILE="$REPO_ROOT/.github/actions-lock.yaml"

CHECK_ONLY=false

usage() {
  cat <<'USAGE'
Usage: scripts/ci/upgrade-actions.sh [--check]

Options:
  --check    Print the refs that would change without editing the lock file.
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --check)
      CHECK_ONLY=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $arg"
      usage
      exit 1
      ;;
  esac
done

check_dependencies() {
  local missing=()

  if ! command -v yq >/dev/null 2>&1; then
    missing+=("yq (install: brew install yq)")
  fi

  if ! command -v jq >/dev/null 2>&1; then
    missing+=("jq (install: brew install jq)")
  fi

  if ! command -v curl >/dev/null 2>&1; then
    missing+=("curl")
  fi

  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "ERROR: Missing required dependencies:"
    printf '  - %s\n' "${missing[@]}"
    exit 1
  fi
}

github_api() {
  local url="$1"
  local headers=(
    -H "Accept: application/vnd.github+json"
    -H "X-GitHub-Api-Version: 2022-11-28"
    -H "User-Agent: rscrypto-action-upgrader"
  )

  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    headers+=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
  fi

  curl -fsSL "${headers[@]}" "$url"
}

raw_github() {
  local url="$1"
  local headers=(-H "User-Agent: rscrypto-action-upgrader")

  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    headers+=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
  fi

  curl -fsSL "${headers[@]}" "$url"
}

list_tags() {
  local action="$1"

  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    gh api --paginate "repos/$action/tags?per_page=100" --jq '.[].name'
    return 0
  fi

  local page=1
  while true; do
    local response names
    response=$(github_api "https://api.github.com/repos/$action/tags?per_page=100&page=$page")
    names=$(jq -r '.[].name' <<<"$response")

    if [[ -z "$names" ]]; then
      break
    fi

    printf '%s\n' "$names"
    page=$((page + 1))
  done
}

stable_semver_tags() {
  jq -R -s -r '
    split("\n")
    | map(select(length > 0))
    | map(select(test("^v?[0-9]+(\\.[0-9]+){0,2}$")))
    | map(
        capture("^(?<prefix>v?)(?<major>[0-9]+)(\\.(?<minor>[0-9]+))?(\\.(?<patch>[0-9]+))?$") as $v
        | {
            tag: .,
            major: ($v.major | tonumber),
            minor: (($v.minor // "0") | tonumber),
            patch: (($v.patch // "0") | tonumber),
            specificity: ((if $v.minor then 1 else 0 end) + (if $v.patch then 1 else 0 end))
          }
      )
    | sort_by(.major, .minor, .patch, .specificity)
    | reverse
    | .[].tag
  '
}

action_runtime() {
  local action="$1"
  local ref="$2"
  local tmp
  tmp=$(mktemp)

  for filename in action.yml action.yaml; do
    if raw_github "https://raw.githubusercontent.com/$action/$ref/$filename" >"$tmp" 2>/dev/null; then
      yq eval '.runs.using // ""' "$tmp" 2>/dev/null || true
      rm -f "$tmp"
      return 0
    fi
  done

  rm -f "$tmp"
  return 1
}

runtime_is_supported() {
  local using="$1"

  case "$using" in
    composite|docker|node24)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

select_latest_supported_ref() {
  local action="$1"
  local current_ref="$2"
  local tags semver_tags

  tags=$(list_tags "$action")
  semver_tags=$(stable_semver_tags <<<"$tags")

  if [[ -z "$semver_tags" ]]; then
    local current_runtime
    current_runtime=$(action_runtime "$action" "$current_ref" || true)
    if runtime_is_supported "$current_runtime"; then
      printf '%s\n' "$current_ref"
      return 0
    fi

    echo "ERROR: $action has no stable semver tags and $current_ref uses unsupported runtime '$current_runtime'" >&2
    return 1
  fi

  local tag runtime
  while IFS= read -r tag; do
    [[ -z "$tag" ]] && continue

    runtime=$(action_runtime "$action" "$tag" || true)
    if runtime_is_supported "$runtime"; then
      printf '%s\n' "$tag"
      return 0
    fi

    echo "  Skipping $action@$tag (runs.using=${runtime:-unknown})" >&2
  done <<<"$semver_tags"

  echo "ERROR: no stable Node 24-compatible/composite/docker tag found for $action" >&2
  return 1
}

main() {
  check_dependencies

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  if [[ "$CHECK_ONLY" == true ]]; then
    echo "Checking GitHub Action refs for supported latest stable tags"
  else
    echo "Upgrading GitHub Action refs to supported latest stable tags"
  fi
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""

  local temp_lock
  temp_lock=$(mktemp)
  cp "$LOCK_FILE" "$temp_lock"

  local changed=false
  local action
  while IFS= read -r action; do
    [[ -z "$action" ]] && continue

    local current_ref next_ref
    current_ref=$(yq eval ".\"$action\".ref" "$LOCK_FILE")

    if [[ "$current_ref" == "null" || -z "$current_ref" ]]; then
      echo "Processing: $action"
      echo "  Skipping: no ref defined"
      echo ""
      continue
    fi

    echo "Processing: $action@$current_ref"
    next_ref=$(select_latest_supported_ref "$action" "$current_ref")

    if [[ "$next_ref" == "$current_ref" ]]; then
      echo "  Up to date: $current_ref"
    else
      changed=true
      echo "  Upgrade: $current_ref -> $next_ref"
      if [[ "$CHECK_ONLY" == false ]]; then
        yq eval -i ".\"$action\".ref = \"$next_ref\"" "$temp_lock"
      fi
    fi
    echo ""
  done < <(yq eval 'keys | .[]' "$LOCK_FILE")

  if [[ "$CHECK_ONLY" == true ]]; then
    rm -f "$temp_lock"
    if [[ "$changed" == true ]]; then
      echo "GitHub Action ref updates are available."
    else
      echo "All GitHub Action refs are current under the Node 24 policy."
    fi
    return 0
  fi

  if [[ "$changed" == true ]]; then
    mv "$temp_lock" "$LOCK_FILE"
    echo "Action refs upgraded. Run scripts/ci/pin-actions.sh --update-lock to refresh SHAs."
  else
    rm -f "$temp_lock"
    echo "No action ref upgrades needed."
  fi
}

main
