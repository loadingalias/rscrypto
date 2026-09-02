#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

fail() {
  echo "check worktree regression failure: $*" >&2
  exit 1
}

fixture="$TMP_ROOT/repository"
fake_bin="$TMP_ROOT/bin"
fake_home="$TMP_ROOT/home"
command_log="$TMP_ROOT/commands.log"
preflight_marker="$TMP_ROOT/locked-metadata-preflight"
mkdir -p \
  "$fixture/.config" \
  "$fixture/scripts/check" \
  "$fixture/scripts/ci" \
  "$fixture/scripts/lib" \
  "$fixture/scripts/test" \
  "$fixture/src" \
  "$fake_bin" \
  "$fake_home/.cargo/bin"

cp \
  "$REPO_ROOT/scripts/check/asm-ledger.sh" \
  "$REPO_ROOT/scripts/check/check-all.sh" \
  "$REPO_ROOT/scripts/check/feature-contracts.sh" \
  "$REPO_ROOT/scripts/check/lint-independent-workspaces.sh" \
  "$REPO_ROOT/scripts/check/check.sh" \
  "$fixture/scripts/check/"
cp "$REPO_ROOT/scripts/lib/common.sh" "$REPO_ROOT/scripts/lib/rail-plan.sh" \
  "$REPO_ROOT/scripts/lib/feature-profiles.sh" "$REPO_ROOT/scripts/lib/toolchain.sh" \
  "$fixture/scripts/lib/"
cp "$REPO_ROOT/.config/toolchains.toml" "$fixture/.config/toolchains.toml"
cp "$REPO_ROOT/.config/feature-matrix.json" "$fixture/.config/feature-matrix.json"

cat >"$fixture/.config/target-matrix.json" <<'EOF'
{"variants":[{"id":"mock-cross","dimensions":{"operation":"cross"}}]}
EOF

cat >"$fixture/scripts/ci/target-contracts.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == run && "$2" == mock-cross && "$3" == deep ]]
cargo check --locked --lib
EOF

cat >"$fixture/scripts/lib/python.sh" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == --print ]]; then
  printf '%s\n' "$0"
fi
exit 0
EOF

cat >"$fixture/scripts/check/zeroize-evidence.sh" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF

cat >"$fixture/scripts/check/msrv.sh" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF

cat >"$fixture/scripts/test/test-examples.sh" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF

cat >"$fixture/scripts/check/affected.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/check.sh" --all
EOF

cat >"$fixture/scripts/check/policy.sh" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF

cat >"$fixture/scripts/check/rsa-asm-provenance.sh" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF

chmod +x \
  "$fixture/scripts/ci/target-contracts.sh" \
  "$fixture/scripts/lib/python.sh" \
  "$fixture/scripts/check/affected.sh" \
  "$fixture/scripts/check/msrv.sh" \
  "$fixture/scripts/check/policy.sh" \
  "$fixture/scripts/check/rsa-asm-provenance.sh" \
  "$fixture/scripts/check/zeroize-evidence.sh" \
  "$fixture/scripts/test/test-examples.sh"

cat >"$fake_bin/cargo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

printf 'cargo %s\n' "$*" >>"$MOCK_LOG"

mutate_source() {
  printf '%s\n' '// rewritten by a mutating format check' >>"$MOCK_REPO_ROOT/src/lib.rs"
}

mutate_lockfile() {
  printf '%s\n' '# rewritten by an unlocked Cargo command' >>"$MOCK_REPO_ROOT/Cargo.lock"
}

require_locked() {
  if [[ " $* " != *" --locked "* ]]; then
    mutate_lockfile
    return 0
  fi
  if [[ "${MOCK_LOCK_DRIFT:-0}" == "1" ]]; then
    echo "error: the lock file needs to be updated but --locked was passed" >&2
    exit 41
  fi
}

case "${1:-}" in
  fmt)
    if [[ " $* " != *" --check "* ]]; then
      mutate_source
    elif [[ "${MOCK_FORMAT_DRIFT:-0}" == "1" ]]; then
      echo "Diff in src/lib.rs" >&2
      exit 40
    fi
    ;;
  metadata)
    require_locked "$@"
    : >"$MOCK_PREFLIGHT_MARKER"
    if [[ " $* " == *" --no-deps "* ]]; then
      printf '{"workspace_root":"%s","packages":[{"name":"rscrypto","features":{"aead":[],"aegis256":[],"aes-gcm":[],"aes-gcm-siv":[],"aes-siv":[],"argon2":[],"ascon-aead":[],"ascon-hash":[],"blake2b":[],"blake2s":[],"blake3":[],"chacha20poly1305":[],"crc16":[],"crc24":[],"crc32":[],"crc64":[],"ecdsa-p256":[],"ecdsa-p384":[],"ed25519":[],"hkdf":[],"hmac":[],"hmac-sha3":[],"kmac":[],"ml-kem":[],"parallel":[],"pbkdf2":[],"phc-strings":[],"poly1305":[],"rapidhash":[],"rsa":[],"scrypt":[],"serde":[],"serde-secrets":[],"sha2":[],"sha3":[],"websocket-sha1":[],"x25519":[],"xchacha20poly1305":[],"xxh3":[]}}]}\n' "$MOCK_REPO_ROOT"
    else
      printf '{"workspace_root":"%s","resolve":{"nodes":[{"id":"rscrypto","features":["alloc","auth"]}]}}\n' "$MOCK_REPO_ROOT"
    fi
    ;;
  rail)
    if [[ "${2:-}" == "plan" ]]; then
      if [[ "${3:-}" == "--verify" ]]; then
        exit 0
      fi
      printf '%s\n' '{"plan_contract_version":8,"work":{"cargo.build":{"state":"required","cause":"changed_input","evidence":["evidence:sha256:test"],"scope":{"kind":"cargo","selection":{"kind":"workspace","cargo_args":[],"targets":[]}}},"cargo.test":{"state":"required","cause":"changed_input","evidence":["evidence:sha256:test"],"scope":{"kind":"cargo","selection":{"kind":"workspace","cargo_args":[],"targets":[]}}}}}'
    fi
    ;;
  check | clippy | doc | test | build | rustc)
    require_locked "$@"
    ;;
  xwin)
    if [[ "${2:-}" != "--version" ]]; then
      require_locked "$@"
    fi
    ;;
  deny)
    require_locked "$@"
    ;;
  audit | clean)
    ;;
esac
EOF

cat >"$fake_bin/rustup" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == "target list --installed" ]]; then
  printf '%s\n' mock-win mock-linux mock-ibm mock-nostd x86_64-pc-windows-msvc
fi
EOF

chmod +x "$fake_bin/cargo" "$fake_bin/rustup"
ln -s /bin/bash "$fake_bin/bash"
cp "$fake_bin/cargo" "$fake_home/.cargo/bin/cargo"

cat >"$fixture/Cargo.toml" <<'EOF'
[package]
name = "rscrypto"
version = "0.0.0"
edition = "2024"
EOF
cat >"$fixture/Cargo.lock" <<'EOF'
version = 4

[[package]]
name = "rscrypto"
version = "0.0.0"
EOF
cat >"$fixture/src/lib.rs" <<'EOF'
#![no_std]
EOF
cat >"$fixture/.gitignore" <<'EOF'
/target/
EOF

git -C "$fixture" init --quiet --initial-branch=main
git -C "$fixture" config user.email "ci@example.invalid"
git -C "$fixture" config user.name "CI"
git -C "$fixture" config commit.gpgsign false
git -C "$fixture" add .
git -C "$fixture" commit --quiet -m baseline

printf '%s\n' 'pub const DIRTY_SOURCE: bool = true;' >>"$fixture/src/lib.rs"
printf '%s\n' '# pre-existing lockfile note' >>"$fixture/Cargo.lock"
printf '%s\n' 'pre-existing untracked file' >"$fixture/local-note.txt"

snapshot_worktree() {
  local output=$1
  {
    git -C "$fixture" status --short --untracked-files=all
    git -C "$fixture" diff --binary
    git -C "$fixture" diff --cached --binary
    git -C "$fixture" hash-object "$fixture/local-note.txt"
  } >"$output"
}

assert_worktree_unchanged() {
  local before=$1
  local after=$2
  local description=$3
  if ! cmp -s "$before" "$after"; then
    diff -u "$before" "$after" >&2 || true
    fail "$description changed the pre-existing worktree"
  fi
}

run_recipe() {
  local recipe=$1
  local expected_status=$2
  local format_drift=$3
  local lock_drift=$4
  local description=$5
  local before="$TMP_ROOT/$description.before"
  local after="$TMP_ROOT/$description.after"
  local output="$TMP_ROOT/$description.out"
  local status=0

  : >"$command_log"
  rm -f "$preflight_marker"
  snapshot_worktree "$before"

  (
    unset RAIL_PLAN_JSON_CACHE RAIL_SCOPE_JSON RAIL_SCOPE_JSON_CACHE RAIL_SURFACES_JSON
    export HOME="$fake_home"
    export PATH="$fake_bin:$PATH"
    export MOCK_FORMAT_DRIFT="$format_drift"
    export MOCK_LOCK_DRIFT="$lock_drift"
    export MOCK_LOG="$command_log"
    export MOCK_PREFLIGHT_MARKER="$preflight_marker"
    export MOCK_REPO_ROOT="$fixture"
    just --justfile "$REPO_ROOT/justfile" --working-directory "$fixture" "$recipe"
  ) >"$output" 2>&1 || status=$?

  if [[ "$expected_status" == success && "$status" -ne 0 ]]; then
    cat "$output" >&2
    fail "$description unexpectedly failed"
  fi
  if [[ "$expected_status" == failure && "$status" -eq 0 ]]; then
    cat "$output" >&2
    fail "$description unexpectedly succeeded"
  fi

  if [[ "$format_drift" == 1 ]]; then
    grep -Fq 'Diff in src/lib.rs' "$output" || fail "$description did not report formatting drift"
  fi
  if [[ "$lock_drift" == 1 ]]; then
    grep -Fq 'lock file needs to be updated' "$output" || fail "$description did not report lockfile drift"
  fi

  snapshot_worktree "$after"
  assert_worktree_unchanged "$before" "$after" "$description"
}

run_recipe check success 0 0 check-success
run_recipe check failure 1 0 check-format-failure
run_recipe check-all success 0 0 check-all-success
run_recipe check-all failure 0 1 check-all-lock-failure

sed -n '/cargo rustc \\/,/--manifest-path/p' "$REPO_ROOT/scripts/check/zeroize-evidence.sh" \
  | grep -Fq -- '--locked' \
  || fail "zeroize evidence build is not locked"

echo "Check worktree regression tests passed"
