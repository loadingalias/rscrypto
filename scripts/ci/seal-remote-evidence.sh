#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: scripts/ci/seal-remote-evidence.sh KIND RUN_ID RELATIVE_PATH [...]" >&2
}

if [[ $# -lt 3 ]]; then
  usage
  exit 2
fi

KIND="$1"
RUN_ID="$2"
shift 2

if [[ ! "$KIND" =~ ^[a-z][a-z0-9-]*$ ]]; then
  echo "invalid evidence kind: $KIND" >&2
  exit 2
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || "$RUN_ID" == *..* ]]; then
  echo "invalid evidence run ID: $RUN_ID" >&2
  exit 2
fi

ROOT="${RSCRYPTO_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[[ "$ROOT" == /* && -d "$ROOT" ]] || {
  echo "evidence repository root must be an absolute directory: $ROOT" >&2
  exit 2
}
RESULTS_ROOT="$ROOT/benchmark_results"
DESTINATION="$RESULTS_ROOT/$KIND/$RUN_ID"
TRANSFER_DIR="$RESULTS_ROOT/.transfers"
[[ ! -e "$DESTINATION" ]] || {
  echo "evidence run already exists: $KIND/$RUN_ID" >&2
  exit 2
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    echo "sha256sum or shasum is required to seal evidence" >&2
    return 1
  fi
}

mkdir -p "$RESULTS_ROOT/$KIND" "$TRANSFER_DIR"
STAGING="$(mktemp -d "$RESULTS_ROOT/$KIND/.$RUN_ID.staging.XXXXXX")"
trap 'rm -rf "$STAGING"' EXIT
mkdir -p "$STAGING/artifacts"
: > "$STAGING/artifacts.txt"
: > "$STAGING/.labels"

for relative_path in "$@"; do
  case "$relative_path" in
    "" | /* | .. | ../* | */../* | */..)
      echo "evidence path must stay below the repository root: $relative_path" >&2
      exit 2
      ;;
  esac
  SOURCE="$ROOT/$relative_path"
  [[ -e "$SOURCE" && ! -L "$SOURCE" ]] || {
    echo "evidence path is missing or a symbolic link: $relative_path" >&2
    exit 1
  }
  LABEL="${relative_path//\//__}"
  LABEL="${LABEL// /_}"
  if grep -Fqx "$LABEL" "$STAGING/.labels"; then
    echo "evidence paths collide after labeling: $relative_path" >&2
    exit 2
  fi
  echo "$LABEL" >> "$STAGING/.labels"
  cp -R "$SOURCE" "$STAGING/artifacts/$LABEL"
  printf '%s\t%s\n' "$LABEL" "$relative_path" >> "$STAGING/artifacts.txt"
done
rm "$STAGING/.labels"

git -C "$ROOT" status --short > "$STAGING/git-status.txt"
rustc -Vv > "$STAGING/rustc.txt"
cargo -V > "$STAGING/cargo.txt"
uname -a > "$STAGING/uname.txt"
if command -v lscpu >/dev/null 2>&1; then
  lscpu > "$STAGING/lscpu.txt"
fi

: > "$STAGING/source-files.sha256"
while IFS= read -r -d '' source_file; do
  digest="$(sha256_file "$ROOT/$source_file")"
  printf '%s  %q\n' "$digest" "$source_file" >> "$STAGING/source-files.sha256"
done < <(
  git -C "$ROOT" ls-files --cached --others --exclude-standard -z -- \
    Cargo.toml Cargo.lock rust-toolchain.toml ct.toml src tools/ct-dudect \
    tools/ct-harness scripts/check/zeroize-evidence.sh scripts/ct \
    scripts/ci/seal-remote-evidence.sh \
    | sort -z
)
SOURCE_ID="$(sha256_file "$STAGING/source-files.sha256")"
{
  echo "run_id=$RUN_ID"
  echo "kind=$KIND"
  echo "target=${DEV_MACHINE_TARGET:-unknown}"
  echo "instance_type=${DEV_MACHINE_INSTANCE_TYPE:-unknown}"
  echo "commit=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "source_identity=sha256:$SOURCE_ID"
} > "$STAGING/remote-run.txt"

mv "$STAGING" "$DESTINATION"
trap - EXIT
tar -cf "$TRANSFER_DIR/$RUN_ID.tar" -C "$RESULTS_ROOT" "$KIND/$RUN_ID"
TRANSFER_DIGEST="$(sha256_file "$TRANSFER_DIR/$RUN_ID.tar")"
printf '%s  %s.tar\n' "$TRANSFER_DIGEST" "$RUN_ID" > "$TRANSFER_DIR/$RUN_ID.tar.sha256"
echo "Remote evidence run ID: $RUN_ID"
