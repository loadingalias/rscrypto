#!/usr/bin/env bash
set -euo pipefail

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Coverage Reporting for rscrypto
#
# Generates Rust source coverage from two deterministic sources:
#   1. Unit/integration/property tests via cargo-nextest.
#   2. Committed fuzz corpora via corpus replay integration tests.
#
# Usage:
#   ./scripts/test/test-coverage.sh                # Total LCOV: nextest + fuzz replay
#   ./scripts/test/test-coverage.sh --nextest      # Test suite LCOV only
#   ./scripts/test/test-coverage.sh --fuzz         # Fuzz corpus replay LCOV only
#   ./scripts/test/test-coverage.sh --html         # Total LCOV + HTML report
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

maybe_disable_sccache

COV_DIR="$REPO_ROOT/coverage"
FUZZ_COVERAGE_SCOPE=${RSCRYPTO_FUZZ_COVERAGE_SCOPE:-all}
LLVM_COV_TARGET_DIR="${RSCRYPTO_COV_TARGET_DIR:-$REPO_ROOT/target/llvm-cov-target}"
FUZZ_REPLAY_CORPUS_FILES=0
FUZZ_REPLAY_STATUS="not run"

RUN_NEXTEST=false
RUN_FUZZ=false
GEN_HTML=false

case "${1:-}" in
  --nextest)  RUN_NEXTEST=true ;;
  --fuzz)     RUN_FUZZ=true ;;
  --html)     RUN_NEXTEST=true; RUN_FUZZ=true; GEN_HTML=true ;;
  -h|--help)
    echo "Usage: $0 [--nextest|--fuzz|--html]"
    echo "  (no args)    Both nextest + fuzz coverage"
    echo "  --nextest    Test suite coverage only"
    echo "  --fuzz       Fuzz corpus coverage only"
    echo "  --html       Both + HTML report"
    exit 0
    ;;
  *)          RUN_NEXTEST=true; RUN_FUZZ=true ;;
esac

mkdir -p "$COV_DIR"

clean_coverage_outputs() {
  rm -f "$COV_DIR"/nextest.lcov "$COV_DIR"/fuzz.lcov "$COV_DIR"/total.lcov "$COV_DIR"/merged.lcov
  rm -f "$COV_DIR"/fuzz-*.lcov "$COV_DIR"/fuzz-profile-manifest.txt "$COV_DIR"/SUMMARY.md
  rm -rf "$COV_DIR"/html-nextest "$COV_DIR"/html-fuzz "$COV_DIR"/html-total
  rm -rf "$LLVM_COV_TARGET_DIR"
  clear_profile_data
}

# ── Coverage capture ─────────────────────────────────────────────────────────

require_coverage_tools() {
  if ! command -v cargo-llvm-cov &>/dev/null; then
    echo "Error: cargo-llvm-cov not found"
    echo "Install with: cargo install cargo-llvm-cov"
    exit 1
  fi

  if [ "$RUN_NEXTEST" = true ] && ! cargo nextest --version &>/dev/null; then
    echo "Error: cargo-nextest not found"
    echo "Install with: cargo install cargo-nextest"
    exit 1
  fi
}

start_coverage_capture() {
  export CARGO_INCREMENTAL=0
  export CARGO_TARGET_DIR="$LLVM_COV_TARGET_DIR"
  mkdir -p "$LLVM_COV_TARGET_DIR"

  cargo llvm-cov clean --workspace

  # shellcheck disable=SC2046
  eval "$(cargo llvm-cov show-env --sh)"

  # Keep coverage profile output coupled to the target directory we report from.
  # Some cargo-llvm-cov versions infer this from CARGO_TARGET_DIR, but weekly CI
  # runs multiple isolated fuzz manifests, so make the handoff explicit.
  export CARGO_TARGET_DIR="$LLVM_COV_TARGET_DIR"
  set_profile_output nextest
}

set_profile_output() {
  local phase=$1

  mkdir -p "$LLVM_COV_TARGET_DIR"
  export LLVM_PROFILE_FILE="$LLVM_COV_TARGET_DIR/rscrypto-${phase}-%p-%10m.profraw"
}

run_nextest_capture() {
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Test Suite Coverage Capture (cargo-nextest)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  cargo nextest run --all-features --workspace
}

run_fuzz_replay_capture() {
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Fuzz Corpus Coverage Capture (deterministic replay)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  case "$FUZZ_COVERAGE_SCOPE" in
    all) ;;
    full) ;;
    scoped) ;;
    *)
      echo "Error: unknown fuzz coverage scope: $FUZZ_COVERAGE_SCOPE"
      return 1
      ;;
  esac

  echo "Fuzz coverage scope: $FUZZ_COVERAGE_SCOPE"

  FUZZ_REPLAY_CORPUS_FILES=$(count_fuzz_corpus_files "$FUZZ_COVERAGE_SCOPE")
  if [ "$FUZZ_REPLAY_CORPUS_FILES" -eq 0 ]; then
    FUZZ_REPLAY_STATUS="skipped: no corpus files"
    if [ "$RUN_NEXTEST" = true ]; then
      echo "No fuzz corpus files found for scope '$FUZZ_COVERAGE_SCOPE'; unified coverage will use nextest only"
    else
      echo "No fuzz corpus files found for scope '$FUZZ_COVERAGE_SCOPE'; no fuzz coverage report will be generated"
    fi
    return 0
  fi

  export RSCRYPTO_FUZZ_REPLAY_MISSING=skip
  case "$FUZZ_COVERAGE_SCOPE" in
    all)
      run_full_fuzz_replay
      run_scoped_fuzz_replay
      ;;
    full)
      run_full_fuzz_replay
      ;;
    scoped)
      run_scoped_fuzz_replay
      ;;
  esac

  echo ""
  FUZZ_REPLAY_STATUS="captured: ${FUZZ_REPLAY_CORPUS_FILES} corpus file(s)"
  echo "Fuzz corpus replay captured (${FUZZ_REPLAY_CORPUS_FILES} corpus file(s))"
}

run_full_fuzz_replay() {
  echo "Full fuzz workspace replay"
  cargo test --manifest-path "$REPO_ROOT/fuzz/Cargo.toml" --all-features --test corpus_replay -- --nocapture
  assert_profile_data_after "full fuzz workspace replay"
}

run_scoped_fuzz_replay() {
  echo "Scoped fuzz package replay"

  local after before manifest
  while IFS= read -r manifest; do
    echo "  -> ${manifest#"$REPO_ROOT"/}"
    before=$(profile_count)
    cargo test --manifest-path "$manifest" --all-features --test corpus_replay -- --nocapture
    after=$(profile_count)
    if [ "$after" -le "$before" ]; then
      echo "Error: no new coverage profile data after ${manifest#"$REPO_ROOT"/}"
      echo "LLVM_PROFILE_FILE=$LLVM_PROFILE_FILE"
      echo "CARGO_TARGET_DIR=$CARGO_TARGET_DIR"
      return 1
    fi
  done < <(find "$REPO_ROOT/fuzz-packages" -mindepth 2 -maxdepth 2 -name Cargo.toml | sort)
}

count_fuzz_corpus_files() {
  local scope=$1
  local roots=()
  local count=0

  case "$scope" in
    all)
      roots=("$REPO_ROOT/fuzz/corpus" "$REPO_ROOT/fuzz-packages")
      ;;
    full)
      roots=("$REPO_ROOT/fuzz/corpus")
      ;;
    scoped)
      roots=("$REPO_ROOT/fuzz-packages")
      ;;
  esac

  local root
  for root in "${roots[@]}"; do
    if [ -d "$root" ]; then
      while IFS= read -r -d '' _; do
        count=$((count + 1))
      done < <(find "$root" -path "*/corpus/*" -type f -print0 2>/dev/null)
    fi
  done

  echo "$count"
}

coverage_output_path() {
  if [ "$RUN_NEXTEST" = true ] && [ "$RUN_FUZZ" = true ]; then
    echo "$COV_DIR/total.lcov"
  elif [ "$RUN_NEXTEST" = true ]; then
    echo "$COV_DIR/nextest.lcov"
  else
    echo "$COV_DIR/fuzz.lcov"
  fi
}

coverage_html_dir() {
  if [ "$RUN_NEXTEST" = true ] && [ "$RUN_FUZZ" = true ]; then
    echo "$COV_DIR/html-total"
  elif [ "$RUN_NEXTEST" = true ]; then
    echo "$COV_DIR/html-nextest"
  else
    echo "$COV_DIR/html-fuzz"
  fi
}

clear_profile_data() {
  local profile

  while IFS= read -r profile; do
    rm -f "$profile"
  done < <(
    find_profile_files '*.profraw'
    find_profile_files '*.profdata'
    find_profile_files '*-profraw-list'
  )
}

find_profile_roots() {
  local root

  for root in "$LLVM_COV_TARGET_DIR" "$REPO_ROOT" "$REPO_ROOT/fuzz" "$REPO_ROOT/fuzz-packages"; do
    if [ -d "$root" ]; then
      printf '%s\n' "$root"
    fi
  done
}

write_profile_manifest() {
  local output=$1

  find_profile_files '*.profraw' | sort -u > "$output"
}

find_profile_files() {
  local name_pattern=$1

  if [ -d "$LLVM_COV_TARGET_DIR" ]; then
    find "$LLVM_COV_TARGET_DIR" -maxdepth 4 -name "$name_pattern" -type f 2>/dev/null
  fi
  if [ -d "$REPO_ROOT" ]; then
    find "$REPO_ROOT" -maxdepth 1 -name "$name_pattern" -type f 2>/dev/null
  fi
  if [ -d "$REPO_ROOT/fuzz" ]; then
    find "$REPO_ROOT/fuzz" -maxdepth 1 -name "$name_pattern" -type f 2>/dev/null
  fi
  if [ -d "$REPO_ROOT/fuzz-packages" ]; then
    find "$REPO_ROOT/fuzz-packages" -mindepth 2 -maxdepth 2 -name "$name_pattern" -type f 2>/dev/null
  fi
}

profile_count() {
  local manifest
  manifest=$(mktemp)
  write_profile_manifest "$manifest"
  wc -l < "$manifest" | tr -d ' '
  rm -f "$manifest"
}

assert_profile_data_after() {
  local label=$1

  if [ "$(profile_count)" -eq 0 ]; then
    echo "Error: no coverage profile data after $label"
    echo "LLVM_PROFILE_FILE=$LLVM_PROFILE_FILE"
    echo "CARGO_TARGET_DIR=$CARGO_TARGET_DIR"
    echo "Profile search roots:"
    find_profile_roots | sed 's/^/  - /'
    return 1
  fi
}

llvm_tool() {
  local tool=$1
  local host path sysroot

  sysroot="$(rustc --print sysroot)"
  host="$(rustc -vV | awk '/^host:/{print $2}')"
  path="$sysroot/lib/rustlib/$host/bin/$tool"
  if [ -x "$path" ]; then
    echo "$path"
    return 0
  fi

  if command -v "$tool" &>/dev/null; then
    command -v "$tool"
    return 0
  fi

  echo "Error: $tool not found; install rustup component llvm-tools-preview" >&2
  return 1
}

generate_root_report() {
  local lcov_path=$1
  local html_dir

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Generating test-suite coverage report"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  cargo llvm-cov report --lcov --output-path "$lcov_path"
  echo "LCOV report: $lcov_path"

  if [ "$GEN_HTML" = true ] && { [ "$RUN_NEXTEST" = false ] || [ "$RUN_FUZZ" = false ]; }; then
    html_dir="$(coverage_html_dir)"
    cargo llvm-cov report --html --output-dir "$html_dir"
    echo "HTML report: $html_dir/index.html"
  fi
}

generate_fuzz_report() {
  local lcov_path=$1
  local llvm_cov llvm_profdata profdata profraw_list raw_lcov
  local replay_binaries=()
  local candidate

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Generating fuzz replay coverage report"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  llvm_cov="$(llvm_tool llvm-cov)"
  llvm_profdata="$(llvm_tool llvm-profdata)"
  profdata="$LLVM_COV_TARGET_DIR/rscrypto.profdata"
  profraw_list="$LLVM_COV_TARGET_DIR/rscrypto-profraw-list"
  raw_lcov="$COV_DIR/fuzz.raw.lcov"

  write_profile_manifest "$profraw_list"
  if [ ! -s "$profraw_list" ]; then
    echo "Error: no fuzz profile data found"
    echo "LLVM_PROFILE_FILE=$LLVM_PROFILE_FILE"
    echo "Profile search roots:"
    find_profile_roots | sed 's/^/  - /'
    return 1
  fi

  "$llvm_profdata" merge -sparse -f "$profraw_list" -o "$profdata"

  while IFS= read -r -d '' candidate; do
    if [ -x "$candidate" ]; then
      replay_binaries+=(--object "$candidate")
    fi
  done < <(find "$LLVM_COV_TARGET_DIR/debug/deps" -maxdepth 1 -type f -name "corpus_replay-*" -print0)

  if [ "${#replay_binaries[@]}" -eq 0 ]; then
    echo "Error: no fuzz corpus_replay test binaries found in $LLVM_COV_TARGET_DIR/debug/deps"
    return 1
  fi

  "$llvm_cov" export \
    --format=lcov \
    --instr-profile "$profdata" \
    --ignore-filename-regex "(/\\.cargo/registry/|/rustc/|/fuzz/|/fuzz-packages/|/tests/)" \
    "${replay_binaries[@]}" > "$raw_lcov"

  filter_lcov_sources "$raw_lcov" "$lcov_path"
  rm -f "$raw_lcov"
  echo "LCOV report: $lcov_path"
}

filter_lcov_sources() {
  local input=$1
  local output=$2

  awk -v source_root="$REPO_ROOT/src/" '
    BEGIN { keep = 0 }
    /^SF:/ {
      path = substr($0, 4)
      keep = (index(path, source_root) == 1)
    }
    keep { print }
    /^end_of_record/ { keep = 0 }
  ' "$input" > "$output"

  if [ ! -s "$output" ]; then
    echo "Error: filtered LCOV report is empty: $output"
    return 1
  fi
}

merge_lcov_reports() {
  local output=$1
  shift

  awk '
    /^SF:/ {
      file = substr($0, 4)
      files[file] = 1
      next
    }
    /^FN:/ {
      if (file == "") next
      rest = substr($0, 4)
      comma = index(rest, ",")
      line = substr(rest, 1, comma - 1)
      name = substr(rest, comma + 1)
      key = file SUBSEP name
      fn_line[key] = line
      funcs[key] = 1
      next
    }
    /^FNDA:/ {
      if (file == "") next
      rest = substr($0, 6)
      comma = index(rest, ",")
      count = substr(rest, 1, comma - 1) + 0
      name = substr(rest, comma + 1)
      key = file SUBSEP name
      fn_hits[key] += count
      funcs[key] = 1
      next
    }
    /^DA:/ {
      if (file == "") next
      rest = substr($0, 4)
      split(rest, parts, ",")
      key = file SUBSEP parts[1]
      line_hits[key] += parts[2] + 0
      lines[key] = 1
      next
    }
    /^end_of_record/ {
      file = ""
      next
    }
    END {
      for (file_name in files) {
        print "SF:" file_name

        fn_found = 0
        fn_hit = 0
        for (key in funcs) {
          split(key, parts, SUBSEP)
          if (parts[1] != file_name) continue
          name = parts[2]
          count = fn_hits[key] + 0
          if (fn_line[key] != "") print "FN:" fn_line[key] "," name
          print "FNDA:" count "," name
          fn_found++
          if (count > 0) fn_hit++
        }
        print "FNF:" fn_found
        print "FNH:" fn_hit

        line_found = 0
        line_hit = 0
        for (key in lines) {
          split(key, parts, SUBSEP)
          if (parts[1] != file_name) continue
          line = parts[2]
          count = line_hits[key] + 0
          print "DA:" line "," count
          line_found++
          if (count > 0) line_hit++
        }
        print "LF:" line_found
        print "LH:" line_hit
        print "end_of_record"
      }
    }
  ' "$@" > "$output"

  if [ ! -s "$output" ]; then
    echo "Error: merged LCOV report is empty: $output"
    return 1
  fi

  echo "Merged LCOV report: $output"
}

generate_summary() {
  local lcov_path=$1

  local lines_found lines_hit fn_found fn_hit
  lines_found=$(awk -F: '/^LF:/{s+=$2} END{print s+0}' "$lcov_path")
  lines_hit=$(awk -F: '/^LH:/{s+=$2} END{print s+0}' "$lcov_path")
  fn_found=$(awk -F: '/^FNF:/{s+=$2} END{print s+0}' "$lcov_path")
  fn_hit=$(awk -F: '/^FNH:/{s+=$2} END{print s+0}' "$lcov_path")

  local line_pct="0.0" fn_pct="0.0"
  if [ "$lines_found" -gt 0 ]; then
    line_pct=$(awk "BEGIN{printf \"%.1f\", 100*$lines_hit/$lines_found}")
  fi
  if [ "$fn_found" -gt 0 ]; then
    fn_pct=$(awk "BEGIN{printf \"%.1f\", 100*$fn_hit/$fn_found}")
  fi

  local sources_list=""
  if [ "$RUN_NEXTEST" = true ]; then
    sources_list="${sources_list}
- \`nextest\`"
  fi
  if [ "$RUN_FUZZ" = true ]; then
    sources_list="${sources_list}
- \`fuzz corpus replay (${FUZZ_COVERAGE_SCOPE}; ${FUZZ_REPLAY_STATUS})\`"
  fi

  echo ""
  echo "  Lines:     ${lines_hit}/${lines_found} (${line_pct}%)"
  echo "  Functions: ${fn_hit}/${fn_found} (${fn_pct}%)"
  echo "  Report:    $lcov_path"

  cat > "$COV_DIR/SUMMARY.md" <<SUMMARY_EOF
# Coverage Summary

| Metric    | Hit | Total | Coverage |
|-----------|-----|-------|----------|
| Lines     | ${lines_hit} | ${lines_found} | ${line_pct}% |
| Functions | ${fn_hit} | ${fn_found} | ${fn_pct}% |

Generated: $(date -u +"%Y-%m-%d %H:%M UTC")

## Sources
${sources_list}
SUMMARY_EOF

  echo "  Summary:   $COV_DIR/SUMMARY.md"
}

# ── Main ──────────────────────────────────────────────────────────────────────

clean_coverage_outputs
require_coverage_tools
start_coverage_capture

LCOV_PATH="$(coverage_output_path)"

if [ "$RUN_NEXTEST" = true ] && [ "$RUN_FUZZ" = true ]; then
  run_nextest_capture
  generate_root_report "$COV_DIR/nextest.lcov"
  clear_profile_data

  set_profile_output fuzz
  run_fuzz_replay_capture
  if [ "$FUZZ_REPLAY_CORPUS_FILES" -eq 0 ]; then
    cp "$COV_DIR/nextest.lcov" "$LCOV_PATH"
  else
    generate_fuzz_report "$COV_DIR/fuzz.lcov"
    merge_lcov_reports "$LCOV_PATH" "$COV_DIR/nextest.lcov" "$COV_DIR/fuzz.lcov"
  fi
elif [ "$RUN_NEXTEST" = true ]; then
  run_nextest_capture
  generate_root_report "$LCOV_PATH"
elif [ "$RUN_FUZZ" = true ]; then
  set_profile_output fuzz
  run_fuzz_replay_capture

  if [ "$FUZZ_REPLAY_CORPUS_FILES" -eq 0 ]; then
    echo ""
    echo "No fuzz coverage report generated because no corpus files were available"
    exit 0
  fi

  generate_fuzz_report "$LCOV_PATH"
fi

generate_summary "$LCOV_PATH"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Coverage reports in: $COV_DIR/"
if find "$COV_DIR" -maxdepth 1 \( -name "*.lcov" -o -name "SUMMARY.md" \) -type f -print -quit | grep -q .; then
  find "$COV_DIR" -maxdepth 1 \( -name "*.lcov" -o -name "SUMMARY.md" \) -type f -exec ls -la {} +
else
  echo "(no coverage files generated)"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
