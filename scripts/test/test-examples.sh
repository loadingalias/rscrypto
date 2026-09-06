#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

examples=(
  "aead_seal_open alloc,chacha20poly1305,getrandom"
  "argon2id_password_hashing argon2,phc-strings,getrandom"
  "ed25519_sign_verify ed25519,getrandom"
  "introspect crc32,sha2,chacha20poly1305,diag"
  "mlkem_encapsulation ml-kem,getrandom"
  "p256_ecdh p256-ecdh,getrandom"
  "rsa_pss_verify rsa"
  "x25519_key_agreement x25519,getrandom"
)

expected=$(printf '%s\n' "${examples[@]%% *}")
actual=$(
  cargo metadata --locked --no-deps --format-version 1 \
    | jq -r '.packages[] | select(.name == "rscrypto") | .targets[] | select(.kind | index("example")) | .name' \
    | sort
)
total=${#examples[@]}

if [[ "$actual" != "$expected" ]]; then
  echo "example test error: Cargo example targets and test-examples.sh differ" >&2
  diff -u <(printf '%s\n' "$expected") <(printf '%s\n' "$actual") >&2 || true
  exit 1
fi

index=0
run_example() {
  local name=$1
  local features=$2
  index=$((index + 1))
  printf '[%d/%d] %s\n' "$index" "$total" "$name"
  cargo run --locked --quiet --no-default-features --example "$name" --features "$features"
}

for example in "${examples[@]}"; do
  run_example "${example%% *}" "${example#* }"
done

echo "All $total examples passed"
