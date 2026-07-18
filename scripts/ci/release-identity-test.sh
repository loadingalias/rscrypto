#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SOURCE_PACKAGER="$SCRIPT_DIR/package-release-source.sh"
MANIFEST_WRITER="$SCRIPT_DIR/write-release-manifest.sh"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

fixture="$TMP_ROOT/repository"
mkdir -p "$fixture/.github/workflows"
git -C "$TMP_ROOT" init -q -b main repository
git -C "$fixture" config user.email test@example.com
git -C "$fixture" config user.name "Release Identity Test"

cat > "$fixture/Cargo.toml" <<'EOF'
[package]
name = "rscrypto"
version = "1.2.3"
edition = "2024"
EOF
cat > "$fixture/Cargo.lock" <<'EOF'
# release identity fixture
version = 4
EOF
cp "$REPO_ROOT/rust-toolchain.toml" "$fixture/rust-toolchain.toml"
cp "$REPO_ROOT/.github/workflows/release.yaml" "$fixture/.github/workflows/release.yaml"
git -C "$fixture" add .
git -C "$fixture" commit -qm "release fixture"
commit=$(git -C "$fixture" rev-parse HEAD)
git -C "$fixture" tag -a v1.2.3 -m "release v1.2.3"

artifacts="$TMP_ROOT/artifacts"
mkdir -p "$artifacts" "$TMP_ROOT/package/rscrypto-1.2.3" "$TMP_ROOT/ct"

"$SOURCE_PACKAGER" \
  --root "$fixture" \
  --version 1.2.3 \
  --tag v1.2.3 \
  --commit "$commit" \
  --out "$artifacts" >/dev/null
"$SOURCE_PACKAGER" \
  --root "$fixture" \
  --version 1.2.3 \
  --tag v1.2.3 \
  --commit "$commit" \
  --out "$TMP_ROOT/reproduced" >/dev/null
cmp "$artifacts/rscrypto-1.2.3-source.tar.gz" "$TMP_ROOT/reproduced/rscrypto-1.2.3-source.tar.gz"

jq -n --arg commit "$commit" \
  '{git: {sha1: $commit, dirty: false}, path_in_vcs: ""}' \
  > "$TMP_ROOT/package/rscrypto-1.2.3/.cargo_vcs_info.json"
tar -czf "$artifacts/rscrypto-1.2.3.crate" -C "$TMP_ROOT/package" rscrypto-1.2.3

jq -n --arg commit "$commit" '{
  schema_version: 1,
  kind: "rscrypto.ct.release-evidence",
  crate: "rscrypto",
  crate_version: "1.2.3",
  git_commit: $commit,
  evidence_git_commit: $commit,
  evidence_mode: "exact_commit"
}' > "$TMP_ROOT/ct/CT-EVIDENCE-BUNDLE.json"
tar -czf "$artifacts/rscrypto-1.2.3-ct-evidence.tar.gz" -C "$TMP_ROOT/ct" CT-EVIDENCE-BUNDLE.json

jq -n --arg commit "$commit" '{
  schema_version: 3,
  kind: "rscrypto.repository-controls",
  release_commit: $commit
}' > "$artifacts/rscrypto-1.2.3-repository-controls.json"

write_manifest() {
  "$MANIFEST_WRITER" \
    --root "$fixture" \
    --version 1.2.3 \
    --tag v1.2.3 \
    --commit "$commit" \
    --source "${SOURCE_PATH:-$artifacts/rscrypto-1.2.3-source.tar.gz}" \
    --crate "$artifacts/rscrypto-1.2.3.crate" \
    --ct-evidence "${CT_PATH:-$artifacts/rscrypto-1.2.3-ct-evidence.tar.gz}" \
    --repository-controls "${CONTROLS_PATH:-$artifacts/rscrypto-1.2.3-repository-controls.json}" \
    --evidence-commit "$commit" \
    --evidence-mode exact_commit \
    --output "${MANIFEST_PATH:-$artifacts/rscrypto-1.2.3-release-manifest.json}"
}

github_output="$TMP_ROOT/github-output"
GITHUB_OUTPUT="$github_output" write_manifest >/dev/null
manifest="$artifacts/rscrypto-1.2.3-release-manifest.json"
jq -e --arg commit "$commit" '
  .schema_version == 1
  and .kind == "rscrypto.release-manifest"
  and .crate_version == "1.2.3"
  and .release.tag == "v1.2.3"
  and (.release.tag_object | test("^[0-9a-f]{40}$"))
  and .release.git_commit == $commit
  and (.release.git_tree | test("^[0-9a-f]{40}$"))
  and .toolchain.channel == "nightly-2026-04-27"
  and (.toolchain.manifest.sha256 | test("^[0-9a-f]{64}$"))
  and .evidence.git_commit == $commit
  and .evidence.mode == "exact_commit"
  and .artifacts.source_archive.name == "rscrypto-1.2.3-source.tar.gz"
  and .artifacts.source_archive.prefix == "rscrypto-1.2.3/"
  and .artifacts.crate_package.name == "rscrypto-1.2.3.crate"
' "$manifest" >/dev/null
grep -Fxq "manifest_path=$manifest" "$github_output"
grep -Fxq "manifest_name=$(basename "$manifest")" "$github_output"
grep -Eq '^manifest_sha256=[0-9a-f]{64}$' "$github_output"

tampered_source="$TMP_ROOT/rscrypto-1.2.3-source.tar.gz"
cp "$artifacts/rscrypto-1.2.3-source.tar.gz" "$tampered_source"
printf 'tampered' >> "$tampered_source"
if SOURCE_PATH="$tampered_source" MANIFEST_PATH="$TMP_ROOT/tampered-source.json" write_manifest >/dev/null 2>&1; then
  echo "release manifest accepted a source archive not reproduced from the release commit" >&2
  exit 1
fi

bad_controls="$TMP_ROOT/rscrypto-1.2.3-repository-controls.json"
jq '.release_commit = "0000000000000000000000000000000000000000"' \
  "$artifacts/rscrypto-1.2.3-repository-controls.json" > "$bad_controls"
if CONTROLS_PATH="$bad_controls" MANIFEST_PATH="$TMP_ROOT/bad-controls.json" write_manifest >/dev/null 2>&1; then
  echo "release manifest accepted repository controls for another commit" >&2
  exit 1
fi

bad_ct_dir="$TMP_ROOT/bad-ct"
mkdir -p "$bad_ct_dir"
jq '.git_commit = "0000000000000000000000000000000000000000"' \
  "$TMP_ROOT/ct/CT-EVIDENCE-BUNDLE.json" > "$bad_ct_dir/CT-EVIDENCE-BUNDLE.json"
bad_ct="$TMP_ROOT/rscrypto-1.2.3-ct-evidence.tar.gz"
tar -czf "$bad_ct" -C "$bad_ct_dir" CT-EVIDENCE-BUNDLE.json
if CT_PATH="$bad_ct" MANIFEST_PATH="$TMP_ROOT/bad-ct.json" write_manifest >/dev/null 2>&1; then
  echo "release manifest accepted CT evidence for another commit" >&2
  exit 1
fi

git -C "$fixture" commit --allow-empty -qm "move release"
moved_commit=$(git -C "$fixture" rev-parse HEAD)
git -C "$fixture" tag -fa v1.2.3 -m "moved release" >/dev/null
if "$SOURCE_PACKAGER" \
  --root "$fixture" \
  --version 1.2.3 \
  --tag v1.2.3 \
  --commit "$commit" \
  --out "$TMP_ROOT/moved" >/dev/null 2>&1; then
  echo "source packager accepted moved tag $moved_commit for release commit $commit" >&2
  exit 1
fi

echo "Release identity regression tests passed"
