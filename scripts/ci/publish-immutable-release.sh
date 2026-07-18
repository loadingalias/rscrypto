#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: scripts/ci/publish-immutable-release.sh [options]

Required:
  --tag TAG            Existing annotated release tag
  --title TITLE        GitHub Release title
  --notes PATH         Release notes file
  --asset PATH         Asset to publish; repeat for every asset
  --stable-asset PATH  Asset that must match on a published-release rerun; repeat as needed

An absent release is assembled as a draft and then published. An existing
draft is repaired only when its final asset set is exact. An existing published
release is never modified and must have a valid immutable-release attestation.
EOF
}

tag=""
title=""
notes=""
assets=()
stable_assets=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) tag=${2:?missing value for --tag}; shift 2 ;;
    --title) title=${2:?missing value for --title}; shift 2 ;;
    --notes) notes=${2:?missing value for --notes}; shift 2 ;;
    --asset) assets+=("${2:?missing value for --asset}"); shift 2 ;;
    --stable-asset) stable_assets+=("${2:?missing value for --stable-asset}"); shift 2 ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$tag" && -n "$title" && -f "$notes" ]] || {
  usage
  exit 2
}
(( ${#assets[@]} > 0 )) || {
  echo "immutable release error: at least one --asset is required" >&2
  exit 2
}
(( ${#stable_assets[@]} > 0 )) || {
  echo "immutable release error: at least one --stable-asset is required" >&2
  exit 2
}

asset_names=()
for asset in "${assets[@]}"; do
  [[ -f "$asset" ]] || {
    echo "immutable release error: asset is missing: $asset" >&2
    exit 1
  }
  asset_names+=("$(basename "$asset")")
done
duplicate_name=$(printf '%s\n' "${asset_names[@]}" | LC_ALL=C sort | uniq -d | head -n 1)
[[ -z "$duplicate_name" ]] || {
  echo "immutable release error: duplicate asset name: $duplicate_name" >&2
  exit 1
}

for asset in "${stable_assets[@]}"; do
  [[ -f "$asset" ]] || {
    echo "immutable release error: stable asset is missing: $asset" >&2
    exit 1
  }
  stable_present=false
  for published_asset in "${assets[@]}"; do
    if [[ "$published_asset" == "$asset" ]]; then
      stable_present=true
      break
    fi
  done
  [[ "$stable_present" == true ]] || {
    echo "immutable release error: stable asset must also be a published asset: $asset" >&2
    exit 1
  }
done

verify_attempts=${RSCRYPTO_RELEASE_VERIFY_ATTEMPTS:-18}
verify_delay=${RSCRYPTO_RELEASE_VERIFY_DELAY:-10}
[[ "$verify_attempts" =~ ^[1-9][0-9]*$ && "$verify_delay" =~ ^[0-9]+$ ]] || {
  echo "immutable release error: invalid verification retry configuration" >&2
  exit 2
}

expected_assets="$(printf '%s\n' "${asset_names[@]}" | LC_ALL=C sort)"

verify_asset_set() {
  local actual_assets
  actual_assets="$(gh release view "$tag" --json assets --jq '.assets[].name' | LC_ALL=C sort)"
  if [[ "$actual_assets" != "$expected_assets" ]]; then
    echo "immutable release error: published asset set differs from the expected artifacts" >&2
    diff -u <(printf '%s\n' "$expected_assets") <(printf '%s\n' "$actual_assets") >&2 || true
    return 1
  fi
}

verify_immutable_release() {
  local attempt
  for ((attempt = 1; attempt <= verify_attempts; attempt++)); do
    if gh release verify "$tag" >/dev/null; then
      return 0
    fi
    echo "immutable release attestation not available yet; retry $attempt/$verify_attempts"
    if (( attempt < verify_attempts )); then
      sleep "$verify_delay"
    fi
  done
  echo "immutable release error: release is mutable or its attestation was not produced" >&2
  return 1
}

release_state=""
if gh release view "$tag" >/dev/null 2>&1; then
  release_state=$(gh release view "$tag" --json isDraft --jq .isDraft)
fi

if [[ "$release_state" == "false" ]]; then
  verify_immutable_release
  verify_asset_set
  for asset in "${stable_assets[@]}"; do
    gh release verify-asset "$tag" "$asset" >/dev/null
  done
  echo "Published immutable release already matches the stable assets"
  exit 0
fi

if [[ -z "$release_state" ]]; then
  gh release create "$tag" \
    --draft \
    --verify-tag \
    --title "$title" \
    --notes-file "$notes"
fi

gh release upload "$tag" "${assets[@]}" --clobber
verify_asset_set
gh release edit "$tag" \
  --title "$title" \
  --notes-file "$notes" \
  --draft=false

verify_immutable_release
verify_asset_set
for asset in "${assets[@]}"; do
  gh release verify-asset "$tag" "$asset" >/dev/null
done
