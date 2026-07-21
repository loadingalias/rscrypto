#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: check-action-pins.sh [--root PATH]" >&2
  exit 2
}

root=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      root=${2:?missing path after --root}
      shift 2
      ;;
    *) usage ;;
  esac
done

if [[ -z "$root" ]]; then
  root=$(git rev-parse --show-toplevel)
fi
root=$(cd "$root" && pwd)

for dependency in curl git sed sort yq; do
  command -v "$dependency" >/dev/null 2>&1 || {
    echo "action pin error: missing dependency: $dependency" >&2
    exit 1
  }
done

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
inventory="$tmp/inventory.tsv"
: >"$inventory"
status=0

fail() {
  echo "action pin error: $*" >&2
  status=1
}

workflow_files() {
  local roots=()
  [[ -d "$root/.github/workflows" ]] && roots+=("$root/.github/workflows")
  [[ -d "$root/.github/actions" ]] && roots+=("$root/.github/actions")
  [[ ${#roots[@]} -gt 0 ]] || return 0
  find "${roots[@]}" \( -name '*.yaml' -o -name '*.yml' \) -type f | sort
}

while IFS= read -r file; do
  line_number=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    line_number=$((line_number + 1))
    use=$(sed -nE 's/^[[:space:]-]*uses:[[:space:]]*([^[:space:]#]+).*/\1/p' <<<"$line")
    [[ -n "$use" ]] || continue

    case "$use" in
      ./* | docker://*) continue ;;
    esac

    parsed=$(sed -nE \
      's/^[[:space:]-]*uses:[[:space:]]*([^@[:space:]#]+)@([0-9a-f]{40})[[:space:]]*#[[:space:]]*(v?[0-9]+(\.[0-9]+){0,2})[[:space:]]*$/\1\	\2\	\3/p' \
      <<<"$line")
    if [[ -z "$parsed" ]]; then
      fail "external action must use a lowercase 40-character SHA and same-line semantic ref: ${file#"$root"/}:$line_number"
      continue
    fi

    action=${parsed%%$'\t'*}
    remainder=${parsed#*$'\t'}
    sha=${remainder%%$'\t'*}
    ref=${remainder#*$'\t'}
    if [[ ! "$action" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$ ]]; then
      fail "invalid external action name '$action': ${file#"$root"/}:$line_number"
      continue
    fi

    printf '%s\t%s\t%s\t%s:%s\n' \
      "$action" "$sha" "$ref" "${file#"$root"/}" "$line_number" >>"$inventory"
  done <"$file"
done < <(workflow_files)

if [[ ! -s "$inventory" ]]; then
  fail "no external GitHub Actions were found"
fi

sorted="$tmp/inventory.sorted.tsv"
sort -t $'\t' -k1,1 -k2,2 -k3,3 "$inventory" >"$sorted"
unique="$tmp/actions.tsv"
: >"$unique"
previous_action=""
previous_sha=""
previous_ref=""
previous_location=""
while IFS=$'\t' read -r action sha ref location; do
  if [[ "$action" == "$previous_action" ]]; then
    if [[ "$sha" != "$previous_sha" || "$ref" != "$previous_ref" ]]; then
      fail "$action has inconsistent pins at $previous_location and $location"
    fi
    continue
  fi
  printf '%s\t%s\t%s\n' "$action" "$sha" "$ref" >>"$unique"
  previous_action=$action
  previous_sha=$sha
  previous_ref=$ref
  previous_location=$location
done <"$sorted"

resolve_ref() {
  local repository=$1
  local ref=$2
  local refs
  refs=$(git ls-remote \
    "https://github.com/${repository}.git" \
    "refs/tags/$ref" \
    "refs/tags/$ref^{}") || return 1
  awk '
    $2 ~ /\^\{\}$/ { peeled = $1 }
    $2 !~ /\^\{\}$/ { direct = $1 }
    END { print (peeled != "" ? peeled : direct) }
  ' <<<"$refs"
}

fetch_definition() {
  local repository=$1
  local sha=$2
  local path=$3
  local output=$4
  curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location \
    --output "$output" \
    "https://raw.githubusercontent.com/$repository/$sha/$path"
}

while IFS=$'\t' read -r action sha ref; do
  owner=${action%%/*}
  remainder=${action#*/}
  repository_name=${remainder%%/*}
  repository="$owner/$repository_name"
  resolved=$(resolve_ref "$repository" "$ref" || true)
  if [[ -z "$resolved" ]]; then
    fail "$action@$ref does not resolve to a commit"
    continue
  fi
  if [[ "$resolved" != "$sha" ]]; then
    fail "$action@$ref resolves to $resolved, not pinned SHA $sha"
    continue
  fi

  if [[ "$action" == "$repository" ]]; then
    path_prefix=""
  else
    path_prefix="${action#"$repository"/}/"
  fi

  case "$path_prefix" in
    .github/workflows/*.yml/ | .github/workflows/*.yaml/)
      workflow_path=${path_prefix%/}
      if ! fetch_definition "$repository" "$sha" "$workflow_path" "$tmp/definition"; then
        fail "$action@$sha does not contain $workflow_path"
      fi
      ;;
    *)
      runtime=""
      for filename in action.yml action.yaml; do
        if fetch_definition "$repository" "$sha" "${path_prefix}${filename}" "$tmp/definition" 2>/dev/null; then
          runtime=$(yq eval -r '.runs.using // ""' "$tmp/definition" 2>/dev/null || true)
          [[ -n "$runtime" ]] && break
        fi
      done
      case "$runtime" in
        composite | docker | node24) ;;
        "") fail "$action@$sha has no readable action definition" ;;
        *) fail "$action@$sha uses unsupported runtime '$runtime'" ;;
      esac
      ;;
  esac
done <"$unique"

if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

action_count=$(wc -l <"$unique" | tr -d ' ')
use_count=$(wc -l <"$inventory" | tr -d ' ')
echo "Action pins verified: $use_count use(s), $action_count action(s)"
