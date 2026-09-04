#!/usr/bin/env bash
# Execute the repository-owned feature contracts without embedding product
# feature knowledge in Just.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"
# shellcheck source=../lib/feature-profiles.sh
source "$SCRIPT_DIR/../lib/feature-profiles.sh"
FEATURE_CATALOG="$SCRIPT_DIR/../../.config/feature-matrix.json"

usage() {
  cat >&2 <<'EOF'
usage: feature-contracts.sh [all | compile [N/M] | runtime [N/M] | selected DOMAIN N/M PROFILES | list | matrix]

N/M is a one-based deterministic shard. Compile and runtime shard counts are
independent; omitting a shard runs the complete selected domain. N/M and
PROFILES in selected mode are emitted together by matrix mode.
EOF
}

compile_profile_id() {
  local feature_set=$1
  if [[ -z "$feature_set" ]]; then
    printf 'compile.none\n'
  else
    printf 'compile.%s\n' "${feature_set//,/.}"
  fi
}

has_independent_compile_contract() {
  local feature=$1
  local feature_set alias
  for feature_set in "${COMPILE_FEATURE_SETS[@]}"; do
    [[ "$feature_set" == "$feature" ]] && return 0
  done
  for alias in "${COMPILE_FEATURE_ALIASES[@]}"; do
    [[ "${alias#*|}" == "$feature" ]] && return 0
  done
  return 1
}

profile_is_known() {
  local needle=$1
  local item
  for item in "${COMPILE_PROFILE_IDS[@]}" "${RUNTIME_PROFILE_IDS_CANONICAL[@]}"; do
    [[ "$needle" == "$item" ]] && return 0
  done
  return 1
}

append_unique() {
  local needle=$1
  local array_name=$2
  local -a current=()
  local item
  eval "current=(\"\${${array_name}[@]:+\${${array_name}[@]}}\")"
  for item in "${current[@]:+${current[@]}}"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  eval "$array_name+=(\"\$needle\")"
}

validate_profiles() {
  local expected=${#RUNTIME_PROFILE_IDS[@]}
  [[ "$expected" -eq "${#RUNTIME_FEATURE_SETS[@]}" ]] || {
    echo "feature runtime catalog columns have different lengths" >&2
    return 2
  }

  local i j feature_set alias_entry canonical alias canonical_known
  for i in "${!COMPILE_FEATURE_SETS[@]}"; do
    for ((j = i + 1; j < ${#COMPILE_FEATURE_SETS[@]}; j++)); do
      [[ "${COMPILE_FEATURE_SETS[$i]}" != "${COMPILE_FEATURE_SETS[$j]}" ]] || {
        echo "duplicate compile feature root: ${COMPILE_FEATURE_SETS[$i]:-<none>}" >&2
        return 2
      }
    done
  done
  for i in "${!COMPILE_FEATURE_ALIASES[@]}"; do
    alias_entry=${COMPILE_FEATURE_ALIASES[$i]}
    canonical=${alias_entry%%|*}
    alias=${alias_entry#*|}
    [[ -n "$canonical" && -n "$alias" && "$canonical" != "$alias_entry" ]] || {
      echo "malformed compile alias: $alias_entry" >&2
      return 2
    }
    canonical_known=false
    for feature_set in "${COMPILE_FEATURE_SETS[@]}"; do
      [[ "$feature_set" == "$canonical" ]] && canonical_known=true
      [[ "$feature_set" != "$alias" ]] || {
        echo "compile alias is also a unique graph: $alias" >&2
        return 2
      }
    done
    [[ "$canonical_known" == true ]] || {
      echo "compile alias names unknown canonical graph: $canonical" >&2
      return 2
    }
    for ((j = i + 1; j < ${#COMPILE_FEATURE_ALIASES[@]}; j++)); do
      [[ "$alias" != "${COMPILE_FEATURE_ALIASES[$j]#*|}" ]] || {
        echo "duplicate compile alias: $alias" >&2
        return 2
      }
    done
  done
  for i in "${!RUNTIME_PROFILE_IDS[@]}"; do
    [[ -n "${RUNTIME_PROFILE_IDS[$i]}" \
      && -n "${RUNTIME_FEATURE_SETS[$i]}" ]] || {
      echo "runtime profile $i has an empty field" >&2
      return 2
    }
    for ((j = i + 1; j < ${#RUNTIME_PROFILE_IDS[@]}; j++)); do
      [[ "${RUNTIME_PROFILE_IDS[$i]}" != "${RUNTIME_PROFILE_IDS[$j]}" ]] || {
        echo "duplicate runtime profile ID: ${RUNTIME_PROFILE_IDS[$i]}" >&2
        return 2
      }
      [[ "${RUNTIME_FEATURE_SETS[$i]}" != "${RUNTIME_FEATURE_SETS[$j]}" ]] || {
        echo "duplicate runtime feature root: ${RUNTIME_FEATURE_SETS[$i]}" >&2
        return 2
      }
    done
  done

  local case_entry case_profile case_target case_filter known count profile_id
  for i in "${!RUNTIME_PROFILE_IDS[@]}"; do
    count=0
    for case_entry in "${RUNTIME_TEST_CASES[@]}"; do
      IFS='|' read -r case_profile case_target case_filter <<<"$case_entry"
      [[ "$case_profile" == "${RUNTIME_PROFILE_IDS[$i]}" ]] && count=$((count + 1))
    done
    ((count > 0)) || {
      echo "runtime profile ${RUNTIME_PROFILE_IDS[$i]} has no test cases" >&2
      return 2
    }
  done
  for i in "${!RUNTIME_TEST_CASES[@]}"; do
    case_entry=${RUNTIME_TEST_CASES[$i]}
    IFS='|' read -r case_profile case_target case_filter <<<"$case_entry"
    [[ -n "$case_profile" && -n "$case_target" ]] || {
      echo "runtime test case $i has an empty profile or target" >&2
      return 2
    }
    known=false
    for profile_id in "${RUNTIME_PROFILE_IDS[@]}"; do
      [[ "$case_profile" == "$profile_id" ]] && known=true
    done
    [[ "$known" == true ]] || {
      echo "runtime test case $i names unknown profile $case_profile" >&2
      return 2
    }
    for ((j = i + 1; j < ${#RUNTIME_TEST_CASES[@]}; j++)); do
      [[ "$case_entry" != "${RUNTIME_TEST_CASES[$j]}" ]] || {
        echo "duplicate runtime test case: $case_entry" >&2
        return 2
      }
    done
  done

  COMPILE_PROFILE_IDS=()
  for feature_set in "${COMPILE_FEATURE_SETS[@]}"; do
    COMPILE_PROFILE_IDS+=("$(compile_profile_id "$feature_set")")
  done
  RUNTIME_PROFILE_IDS_CANONICAL=()
  for profile_id in "${RUNTIME_PROFILE_IDS[@]}"; do
    RUNTIME_PROFILE_IDS_CANONICAL+=("runtime.$profile_id")
  done
}

validate_variant_catalog() {
  FEATURE_GRAPH=$(cargo metadata --locked --format-version 1 --no-deps \
    | jq -ce '[.packages[] | select(.name == "rscrypto") | .features] | if length == 1 then .[0] else error("rscrypto feature graph is ambiguous") end') || {
    echo "cannot resolve the Cargo feature graph" >&2
    return 2
  }

  jq -e '
    .variant_catalog_version == 2
    and .work == "contracts.features"
    and (.variants | type == "array" and length > 0)
    and ([.variants[].id] | length == (unique | length))
    and any(.variants[]; .dimensions.full == true)
    and all(.variants[];
      (.id | test("^[a-z][a-z0-9.-]*$"))
      and (.dimensions | keys | sort) == ["feature_roots", "full", "group", "runtime_profiles"]
      and (.dimensions.group | type == "string" and length > 0)
      and (.dimensions.feature_roots | type == "string")
      and (.dimensions.runtime_profiles | type == "string")
      and (.dimensions.full | type == "boolean")
      and (if .dimensions.full then
        .dimensions.feature_roots == "" and .dimensions.runtime_profiles == ""
      else
        (.dimensions.feature_roots | length > 0) or (.dimensions.runtime_profiles | length > 0)
      end)
      and (.external_paths | type == "array" and length > 0 and length == (unique | length))
      and all(.external_paths[]; type == "string" and length > 0)
    )
  ' "$FEATURE_CATALOG" >/dev/null || {
    echo "feature variant catalog is malformed" >&2
    return 2
  }

  local catalog_id
  while IFS= read -r catalog_id; do
    [[ -n "$catalog_id" ]] || continue
    [[ "$catalog_id" == runtime.* ]] || {
      echo "feature variant catalog carries malformed runtime profile $catalog_id" >&2
      return 2
    }
    profile_is_known "$catalog_id" || {
      echo "feature variant catalog names unknown runtime profile $catalog_id" >&2
      return 2
    }
  done < <(jq -r '
    .variants[].dimensions
    | .runtime_profiles
    | select(length > 0)
    | split(",")[]
  ' "$FEATURE_CATALOG")

  local feature_root
  while IFS= read -r feature_root; do
    [[ -n "$feature_root" ]] || continue
    jq -e --arg feature "$feature_root" 'has($feature)' <<<"$FEATURE_GRAPH" >/dev/null || {
      echo "feature variant catalog names unknown Cargo feature $feature_root" >&2
      return 2
    }
    has_independent_compile_contract "$feature_root" || {
      echo "feature variant catalog root lacks an independent compile contract: $feature_root" >&2
      return 2
    }
  done < <(jq -r '
    .variants[].dimensions.feature_roots
    | select(length > 0)
    | split(",")[]
  ' "$FEATURE_CATALOG")

  while IFS= read -r feature_root; do
    has_independent_compile_contract "$feature_root" || {
      echo "Cargo feature lacks an independent compile contract: $feature_root" >&2
      return 2
    }
  done < <(jq -r 'keys[]' <<<"$FEATURE_GRAPH")
}

parse_shard() {
  local value=${1:-}
  SHARD_NUMBER=1
  SHARD_COUNT=1
  [[ -n "$value" ]] || return 0
  [[ "$value" =~ ^([1-9][0-9]*)/([1-9][0-9]*)$ ]] || {
    echo "invalid shard '$value'; expected one-based N/M" >&2
    return 2
  }
  SHARD_NUMBER=${BASH_REMATCH[1]}
  SHARD_COUNT=${BASH_REMATCH[2]}
  ((SHARD_NUMBER <= SHARD_COUNT)) || {
    echo "invalid shard '$value'; N must not exceed M" >&2
    return 2
  }
}

selected_by_shard() {
  local index=$1
  ((index % SHARD_COUNT == SHARD_NUMBER - 1))
}

selected_profile() {
  local domain=$1
  local index=$2
  local id item
  if [[ "$USE_PROFILE_FILTER" == false ]]; then
    selected_by_shard "$index"
    return
  fi
  if [[ "$domain" == compile ]]; then
    id=${COMPILE_PROFILE_IDS[$index]}
  else
    id=${RUNTIME_PROFILE_IDS_CANONICAL[$index]}
  fi
  for item in "${PROFILE_FILTER_IDS[@]}"; do
    [[ "$item" == "$id" ]] && return 0
  done
  return 1
}

parse_profile_filter() {
  local domain=$1
  local value=$2
  local -a requested=()
  local id
  [[ "$domain" == compile || "$domain" == runtime ]] || usage
  [[ -n "$value" ]] || {
    echo "selected feature profile list must not be empty" >&2
    return 2
  }
  IFS=',' read -r -a requested <<<"$value"
  PROFILE_FILTER_IDS=()
  for id in "${requested[@]}"; do
    profile_is_known "$id" || {
      echo "unknown selected feature profile: $id" >&2
      return 2
    }
    [[ "$id" == "$domain."* ]] || {
      echo "selected $domain execution cannot consume $id" >&2
      return 2
    }
    local before=${#PROFILE_FILTER_IDS[@]}
    append_unique "$id" PROFILE_FILTER_IDS
    [[ "${#PROFILE_FILTER_IDS[@]}" -gt "$before" ]] || {
      echo "duplicate selected feature profile: $id" >&2
      return 2
    }
  done
  USE_PROFILE_FILTER=true
}

quoted_command() {
  local arg rendered=""
  for arg in "$@"; do
    printf -v rendered '%s%q ' "$rendered" "$arg"
  done
  printf '%s' "${rendered% }"
}

run_logged() {
  local label=$1
  local log_path=$2
  shift 2
  local reproduction
  reproduction=$(quoted_command "$@")

  step "$label"
  if ! "$@" >"$log_path" 2>&1; then
    fail
    show_error "$log_path"
    echo "  Reproduce: $reproduction" >&2
    return 1
  fi
  ok
}

resolved_graph() {
  local feature_set=$1
  cargo metadata --locked --format-version 1 --no-default-features \
    --features "$feature_set" \
    | jq -S -c '[.resolve.nodes[] | {id, features}]'
}

verify_compile_aliases() {
  local canonical=$1
  local entry alias alias_graph canonical_graph

  for entry in "${COMPILE_FEATURE_ALIASES[@]}"; do
    [[ "${entry%%|*}" == "$canonical" ]] || continue
    alias=${entry#*|}
    canonical_graph=$(resolved_graph "$canonical") || return 1
    alias_graph=$(resolved_graph "$alias") || return 1
    [[ "$canonical_graph" == "$alias_graph" ]] || {
      echo "compile contracts '$canonical' and '$alias' no longer resolve identically" >&2
      echo "promote '$alias' back to COMPILE_FEATURE_SETS" >&2
      return 1
    }
    echo "    alias: $alias resolves identically"
  done
}

compile_alias_count() {
  local canonical=$1
  local entry count=0
  for entry in "${COMPILE_FEATURE_ALIASES[@]}"; do
    [[ "${entry%%|*}" == "$canonical" ]] && count=$((count + 1))
  done
  printf '%s\n' "$count"
}

run_compile_contracts() {
  local total_unique=${#COMPILE_FEATURE_SETS[@]}
  local total_named=$((total_unique + ${#COMPILE_FEATURE_ALIASES[@]}))
  local selected=0 started_at=$SECONDS
  local i feature_set display log_path profile_started aliases

  echo "Compile feature contracts ($total_named named, $total_unique unique; shard $SHARD_NUMBER/$SHARD_COUNT)"
  for i in "${!COMPILE_FEATURE_SETS[@]}"; do
    selected_profile compile "$i" || continue
    selected=$((selected + 1))
    feature_set=${COMPILE_FEATURE_SETS[$i]}
    display=${feature_set:-no-features}
    log_path="$LOG_DIR/compile-${display//,/_}.log"
    profile_started=$SECONDS
    aliases=$(compile_alias_count "$feature_set")

    if ((aliases > 0)); then
      step "[$((i + 1))/$total_unique] verify aliases for $display"
      if ! verify_compile_aliases "$feature_set" >"$log_path" 2>&1; then
        fail
        show_error "$log_path"
        return 1
      fi
      ok
    fi

    local args=(cargo check --locked --workspace --lib --tests --no-default-features)
    [[ -n "$feature_set" ]] && args+=(--features "$feature_set")
    run_logged "[$((i + 1))/$total_unique] compile $display" "$log_path" "${args[@]}" || return 1
    echo "    elapsed: $((SECONDS - profile_started))s"
  done

  ((selected > 0)) || {
    echo "compile shard $SHARD_NUMBER/$SHARD_COUNT selects no profiles" >&2
    return 2
  }
  echo "${GREEN}✓${RESET} Compile feature contracts passed: $selected unique graphs in $((SECONDS - started_at))s"
}

runtime_args() {
  local feature_set=$1
  local target=$2
  local filter=$3

  if [[ "$target" == all ]] && command -v cargo-nextest >/dev/null 2>&1; then
    RUNTIME_ARGS=(cargo nextest run --locked --workspace --no-default-features
      --features "$feature_set" --config-file .config/nextest.toml -P default)
    return 0
  fi

  RUNTIME_ARGS=(cargo test --locked --workspace --no-default-features --features "$feature_set")
  case "$target" in
    all) RUNTIME_ARGS+=(--lib --tests) ;;
    lib) RUNTIME_ARGS+=(--lib) ;;
    *) RUNTIME_ARGS+=(--test "$target") ;;
  esac
  [[ -n "$filter" ]] && RUNTIME_ARGS+=(-- "$filter")
  return 0
}

runtime_test_runner() {
  local profile_id=$1
  local case_entry case_profile case_target case_filter
  for case_entry in "${RUNTIME_TEST_CASES[@]}"; do
    IFS='|' read -r case_profile case_target case_filter <<<"$case_entry"
    if [[ "$case_profile" == "$profile_id" && "$case_target" == all ]]; then
      printf 'nextest\n'
      return
    fi
  done
  printf 'cargo\n'
}

run_runtime_contracts() {
  local total=${#RUNTIME_PROFILE_IDS[@]}
  local selected=0 started_at=$SECONDS
  local i profile_id feature_set log_path profile_started
  local case_entry case_profile case_target case_filter case_number cases_run

  echo "Runtime feature contracts ($total profiles; shard $SHARD_NUMBER/$SHARD_COUNT)"
  for i in "${!RUNTIME_PROFILE_IDS[@]}"; do
    selected_profile runtime "$i" || continue
    selected=$((selected + 1))
    profile_id=${RUNTIME_PROFILE_IDS[$i]}
    feature_set=${RUNTIME_FEATURE_SETS[$i]}
    profile_started=$SECONDS
    case_number=0
    cases_run=0
    echo "  profile [$((i + 1))/$total] $profile_id ($feature_set)"
    for case_entry in "${RUNTIME_TEST_CASES[@]}"; do
      IFS='|' read -r case_profile case_target case_filter <<<"$case_entry"
      [[ "$case_profile" == "$profile_id" ]] || continue
      case_number=$((case_number + 1))
      cases_run=$((cases_run + 1))
      log_path="$LOG_DIR/runtime-$profile_id-$case_number.log"
      runtime_args "$feature_set" "$case_target" "$case_filter"
      run_logged "case $case_number: $case_target${case_filter:+ ($case_filter)}" \
        "$log_path" "${RUNTIME_ARGS[@]}" || return 1
    done
    ((cases_run > 0)) || return 2
    echo "    elapsed: $((SECONDS - profile_started))s"
  done

  ((selected > 0)) || {
    echo "runtime shard $SHARD_NUMBER/$SHARD_COUNT selects no profiles" >&2
    return 2
  }
  echo "${GREEN}✓${RESET} Runtime feature contracts passed: $selected profiles in $((SECONDS - started_at))s"
}

list_contracts() {
  local feature_set entry
  echo "compile (${#COMPILE_FEATURE_SETS[@]} unique graphs, $(( ${#COMPILE_FEATURE_SETS[@]} + ${#COMPILE_FEATURE_ALIASES[@]} )) named contracts)"
  for feature_set in "${COMPILE_FEATURE_SETS[@]}"; do
    printf '  %s\n' "${feature_set:-no-features}"
    for entry in "${COMPILE_FEATURE_ALIASES[@]}"; do
      [[ "${entry%%|*}" == "$feature_set" ]] && printf '    alias: %s\n' "${entry#*|}"
    done
  done
  echo "runtime (${#RUNTIME_PROFILE_IDS[@]} profiles)"
  local i
  for i in "${!RUNTIME_PROFILE_IDS[@]}"; do
    printf '  %s: %s\n' "${RUNTIME_PROFILE_IDS[$i]}" "${RUNTIME_FEATURE_SETS[$i]}"
    local case_entry case_profile case_target case_filter
    for case_entry in "${RUNTIME_TEST_CASES[@]}"; do
      IFS='|' read -r case_profile case_target case_filter <<<"$case_entry"
      [[ "$case_profile" == "${RUNTIME_PROFILE_IDS[$i]}" ]] || continue
      printf '    %s%s\n' "$case_target" "${case_filter:+: $case_filter}"
    done
  done
}

print_matrix() {
  local selected_rows
  if [[ -n "${RAIL_PLAN_FILE:-}" || -n "${RAIL_PLAN_READER:-}" ]]; then
    selected_rows=$(rail_variant_matrix contracts.features)
  else
    selected_rows=all
  fi
  if [[ "$selected_rows" == all ]]; then
    selected_rows=$(jq -c '{include: [.variants[] | {id: .id} + .dimensions]}' "$FEATURE_CATALOG")
  fi
  jq -e '
    (.include | type == "array")
    and all(.include[];
      (.id | type == "string")
      and (.feature_roots | type == "string")
      and (.runtime_profiles | type == "string")
      and (.full | type == "boolean")
    )
  ' <<<"$selected_rows" >/dev/null || {
    echo "Cargo Rail emitted an invalid feature variant matrix" >&2
    return 2
  }

  local id domain count shard index label planned_id profiles separator="" test_runner
  local -a planned=()
  if jq -e 'any(.include[]; .full == true)' <<<"$selected_rows" >/dev/null; then
    for id in "${COMPILE_PROFILE_IDS[@]}" "${RUNTIME_PROFILE_IDS_CANONICAL[@]}"; do
      append_unique "$id" planned
    done
  else
    local roots profile_rows="[" row_separator=""
    roots=$(jq -r '[.include[].feature_roots | select(length > 0) | split(",")[]] | unique | join(",")' \
      <<<"$selected_rows")
    for index in "${!COMPILE_PROFILE_IDS[@]}"; do
      profile_rows+="${row_separator}{\"id\":\"${COMPILE_PROFILE_IDS[$index]}\",\"features\":\"${COMPILE_FEATURE_SETS[$index]}\"}"
      row_separator=,
    done
    profile_rows+="]"
    while IFS= read -r id; do
      [[ -n "$id" ]] && append_unique "$id" planned
    done < <(jq -nr \
      --argjson graph "$FEATURE_GRAPH" \
      --arg roots "$roots" \
      --argjson profiles "$profile_rows" '
        def local_edge($graph):
          select(startswith("dep:") | not)
          | split("/")[0]
          | sub("\\?$"; "")
          | select($graph[.] != null);
        def closure($graph; $pending; $seen):
          if ($pending | length) == 0 then $seen
          else $pending[0] as $next
          | if ($seen | index($next)) != null then
              closure($graph; $pending[1:]; $seen)
            else
              [$graph[$next][]? | local_edge($graph)] as $edges
              | closure($graph; $pending[1:] + $edges; $seen + [$next])
            end
          end;
        ($roots | split(",") | map(select(length > 0))) as $selected
        | $profiles[]
        | . as $profile
        | closure($graph; ($profile.features | split(",") | map(select(length > 0))); []) as $resolved
        | select(any($selected[]; . as $feature | ($resolved | index($feature)) != null))
        | $profile.id
      ')
    while IFS= read -r id; do
      [[ -n "$id" ]] || continue
      profile_is_known "$id" || {
        echo "Cargo Rail selected unknown runtime feature profile $id" >&2
        return 2
      }
      append_unique "$id" planned
    done < <(jq -r '
      .include[].runtime_profiles
      | select(length > 0)
      | split(",")[]
    ' <<<"$selected_rows")
  fi

  printf '{"include":['
  for domain in compile runtime; do
    if [[ "$domain" == compile ]]; then
      count=$FEATURE_COMPILE_SHARDS
    else
      count=$FEATURE_RUNTIME_SHARDS
    fi
    for ((shard = 1; shard <= count; shard++)); do
      profiles=""
      if [[ "$domain" == compile ]]; then
        for index in "${!COMPILE_PROFILE_IDS[@]}"; do
          ((index % count == shard - 1)) || continue
          id=${COMPILE_PROFILE_IDS[$index]}
          for planned_id in "${planned[@]:+${planned[@]}}"; do
            if [[ "$planned_id" == "$id" ]]; then
              profiles="${profiles:+$profiles,}$id"
              break
            fi
          done
        done
      else
        for index in "${!RUNTIME_PROFILE_IDS_CANONICAL[@]}"; do
          ((index % count == shard - 1)) || continue
          id=${RUNTIME_PROFILE_IDS_CANONICAL[$index]}
          for planned_id in "${planned[@]:+${planned[@]}}"; do
            if [[ "$planned_id" == "$id" ]]; then
              profiles="${profiles:+$profiles,}$id"
              break
            fi
          done
        done
      fi
      [[ -n "$profiles" ]] || continue
      if [[ "$domain" == compile ]]; then
        label="Compile $shard/$count"
        test_runner=cargo
      else
        label="Runtime ${profiles#runtime.}"
        test_runner=$(runtime_test_runner "${profiles#runtime.}")
      fi
      printf '%s{"domain":"%s","shard":"%s/%s","profiles":"%s","label":"%s","test_runner":"%s"}' \
        "$separator" "$domain" "$shard" "$count" "$profiles" "$label" "$test_runner"
      separator=,
    done
  done
  printf ']}\n'
}

domain=${1:-all}
shard=${2:-}
profile_filter=${3:-}
USE_PROFILE_FILTER=false
PROFILE_FILTER_IDS=()
case "$domain" in
  all)
    [[ $# -le 1 ]] || { usage; exit 2; }
    ;;
  compile | runtime)
    [[ $# -le 2 ]] || { usage; exit 2; }
    ;;
  selected)
    [[ $# -eq 4 ]] || { usage; exit 2; }
    domain=$shard
    shard=$profile_filter
    profile_filter=$4
    ;;
  list | matrix)
    [[ $# -eq 1 ]] || { usage; exit 2; }
    ;;
  *) usage; exit 2 ;;
esac

validate_profiles
validate_variant_catalog
if [[ "$domain" == list ]]; then
  list_contracts
  exit 0
fi
if [[ "$domain" == matrix ]]; then
  print_matrix
  exit 0
fi

if [[ -n "$profile_filter" ]]; then
  parse_profile_filter "$domain" "$profile_filter"
  parse_shard "$shard"
else
  parse_shard "$shard"
fi
LOG_DIR=$(mktemp -d)
trap 'rm -rf "$LOG_DIR"' EXIT

case "$domain" in
  compile) run_compile_contracts ;;
  runtime) run_runtime_contracts ;;
  all)
    run_compile_contracts
    run_runtime_contracts
    ;;
esac
