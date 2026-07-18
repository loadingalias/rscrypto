#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLISHER="$SCRIPT_DIR/publish-immutable-release.sh"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

mkdir -p "$TMP_ROOT/bin" "$TMP_ROOT/artifacts"
cat > "$TMP_ROOT/bin/gh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

printf '%q ' "$@" >> "$FAKE_GH_LOG"
printf '\n' >> "$FAKE_GH_LOG"

[[ ${1:-} == "release" ]] || exit 2
operation=${2:-}
shift 2

case "$operation" in
  view)
    [[ $(cat "$FAKE_GH_STATE") != "absent" ]] || exit 1
    if [[ " $* " == *" --json isDraft "* ]]; then
      [[ $(cat "$FAKE_GH_STATE") == "draft" ]] && echo true || echo false
    elif [[ " $* " == *" --json assets "* ]]; then
      cat "$FAKE_GH_ASSETS"
    fi
    ;;
  create)
    echo draft > "$FAKE_GH_STATE"
    ;;
  upload)
    [[ $(cat "$FAKE_GH_STATE") == "draft" ]] || exit 1
    shift
    while [[ $# -gt 0 ]]; do
      if [[ $1 != "--clobber" ]]; then
        basename "$1" >> "$FAKE_GH_ASSETS"
      fi
      shift
    done
    LC_ALL=C sort -u "$FAKE_GH_ASSETS" -o "$FAKE_GH_ASSETS"
    ;;
  edit)
    [[ $(cat "$FAKE_GH_STATE") == "draft" ]] || exit 1
    [[ " $* " == *" --draft=false "* ]] || exit 1
    echo published > "$FAKE_GH_STATE"
    ;;
  verify)
    [[ $(cat "$FAKE_GH_STATE") == "published" && ${FAKE_GH_VERIFY_FAIL:-false} != true ]]
    ;;
  verify-asset)
    [[ $(cat "$FAKE_GH_STATE") == "published" ]] || exit 1
    asset_name=$(basename "${2:?missing asset path}")
    grep -Fxq "$asset_name" "$FAKE_GH_ASSETS"
    [[ "$asset_name" != "${FAKE_GH_BAD_ASSET:-}" ]]
    ;;
  *)
    exit 2
    ;;
esac
EOF
chmod +x "$TMP_ROOT/bin/gh"

export PATH="$TMP_ROOT/bin:$PATH"
export FAKE_GH_LOG="$TMP_ROOT/gh.log"
export FAKE_GH_STATE="$TMP_ROOT/state"
export FAKE_GH_ASSETS="$TMP_ROOT/assets"
export RSCRYPTO_RELEASE_VERIFY_ATTEMPTS=1
export RSCRYPTO_RELEASE_VERIFY_DELAY=0

for name in rscrypto-1.2.3.crate rscrypto-1.2.3-source.tar.gz SHA256SUMS; do
  printf '%s\n' "$name" > "$TMP_ROOT/artifacts/$name"
done
printf 'notes\n' > "$TMP_ROOT/notes.md"

publish() {
  /bin/bash "$PUBLISHER" \
    --tag v1.2.3 \
    --title "rscrypto v1.2.3" \
    --notes "$TMP_ROOT/notes.md" \
    --asset "$TMP_ROOT/artifacts/rscrypto-1.2.3.crate" \
    --asset "$TMP_ROOT/artifacts/rscrypto-1.2.3-source.tar.gz" \
    --asset "$TMP_ROOT/artifacts/SHA256SUMS" \
    --stable-asset "$TMP_ROOT/artifacts/rscrypto-1.2.3.crate" \
    --stable-asset "$TMP_ROOT/artifacts/rscrypto-1.2.3-source.tar.gz"
}

if /bin/bash "$PUBLISHER" \
  --tag v1.2.3 \
  --title "rscrypto v1.2.3" \
  --notes "$TMP_ROOT/notes.md" \
  --asset "$TMP_ROOT/artifacts/rscrypto-1.2.3.crate" \
  --asset "$TMP_ROOT/artifacts/rscrypto-1.2.3.crate" \
  --stable-asset "$TMP_ROOT/artifacts/rscrypto-1.2.3.crate" >/dev/null 2>&1; then
  echo "immutable release publisher accepted duplicate asset names" >&2
  exit 1
fi

: > "$FAKE_GH_LOG"
: > "$FAKE_GH_ASSETS"
echo absent > "$FAKE_GH_STATE"
publish >/dev/null
[[ $(cat "$FAKE_GH_STATE") == "published" ]]
diff -u \
  <(printf '%s\n' SHA256SUMS rscrypto-1.2.3-source.tar.gz rscrypto-1.2.3.crate) \
  "$FAKE_GH_ASSETS"
grep -Fq 'release create' "$FAKE_GH_LOG"
grep -Fq 'release upload' "$FAKE_GH_LOG"
grep -Fq 'release edit' "$FAKE_GH_LOG"

: > "$FAKE_GH_LOG"
printf 'unexpected.bin\n' > "$FAKE_GH_ASSETS"
echo draft > "$FAKE_GH_STATE"
if publish >/dev/null 2>&1; then
  echo "immutable release publisher accepted an unexpected draft asset" >&2
  exit 1
fi
[[ $(cat "$FAKE_GH_STATE") == "draft" ]]
if grep -Fq 'release edit' "$FAKE_GH_LOG"; then
  echo "immutable release publisher published an invalid draft" >&2
  exit 1
fi

: > "$FAKE_GH_LOG"
printf '%s\n' SHA256SUMS rscrypto-1.2.3-source.tar.gz rscrypto-1.2.3.crate > "$FAKE_GH_ASSETS"
echo published > "$FAKE_GH_STATE"
publish >/dev/null
if grep -Eq 'release (create|upload|edit)' "$FAKE_GH_LOG"; then
  echo "immutable release publisher modified an existing published release" >&2
  exit 1
fi
[[ $(grep -c 'release verify-asset' "$FAKE_GH_LOG") -eq 2 ]]

: > "$FAKE_GH_LOG"
printf '%s\n' rscrypto-1.2.3-source.tar.gz rscrypto-1.2.3.crate > "$FAKE_GH_ASSETS"
echo published > "$FAKE_GH_STATE"
if publish >/dev/null 2>&1; then
  echo "immutable release publisher accepted a missing published asset" >&2
  exit 1
fi

: > "$FAKE_GH_LOG"
printf '%s\n' SHA256SUMS rscrypto-1.2.3-source.tar.gz rscrypto-1.2.3.crate > "$FAKE_GH_ASSETS"
echo published > "$FAKE_GH_STATE"
if FAKE_GH_BAD_ASSET=rscrypto-1.2.3.crate publish >/dev/null 2>&1; then
  echo "immutable release publisher accepted a mismatched stable asset" >&2
  exit 1
fi

: > "$FAKE_GH_LOG"
echo published > "$FAKE_GH_STATE"
if FAKE_GH_VERIFY_FAIL=true publish >/dev/null 2>&1; then
  echo "immutable release publisher accepted a mutable published release" >&2
  exit 1
fi

echo "Immutable release publication regression tests passed"
