#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: scripts/ci/write-release-manifest.sh [options]

Required:
  --version VERSION
  --tag TAG
  --commit SHA
  --source PATH
  --crate PATH
  --ct-evidence PATH
  --repository-controls PATH
  --evidence-commit SHA
  --evidence-mode exact_commit
  --output PATH

Optional:
  --root PATH   Repository root (default: current repository)
EOF
}

version=""
tag=""
commit=""
source_archive=""
crate_package=""
ct_evidence=""
repository_controls=""
evidence_commit=""
evidence_mode=""
output=""
root="$(git rev-parse --show-toplevel)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) version=${2:?}; shift 2 ;;
    --tag) tag=${2:?}; shift 2 ;;
    --commit) commit=${2:?}; shift 2 ;;
    --source) source_archive=${2:?}; shift 2 ;;
    --crate) crate_package=${2:?}; shift 2 ;;
    --ct-evidence) ct_evidence=${2:?}; shift 2 ;;
    --repository-controls) repository_controls=${2:?}; shift 2 ;;
    --evidence-commit) evidence_commit=${2:?}; shift 2 ;;
    --evidence-mode) evidence_mode=${2:?}; shift 2 ;;
    --output) output=${2:?}; shift 2 ;;
    --root) root=${2:?}; shift 2 ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

fail() {
  echo "release manifest error: $*" >&2
  exit 1
}

[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)?$ ]] || fail "invalid version: $version"
[[ "$tag" == "v$version" ]] || fail "tag $tag does not match version $version"
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || fail "release commit must be a full lowercase Git commit"
[[ "$evidence_commit" =~ ^[0-9a-f]{40}$ ]] || fail "evidence commit must be a full lowercase Git commit"
[[ "$evidence_mode" == "exact_commit" ]] || fail "invalid evidence mode: $evidence_mode"
[[ "$evidence_commit" == "$commit" ]] || fail "evidence commit must equal the release commit"
[[ -n "$output" ]] || fail "--output is required"

for path in "$source_archive" "$crate_package" "$ct_evidence" "$repository_controls"; do
  [[ -f "$path" ]] || fail "artifact is missing: $path"
done

git -C "$root" rev-parse -q --verify "$tag^{tag}" >/dev/null || fail "release tag is not annotated: $tag"
tag_object=$(git -C "$root" rev-parse "$tag^{tag}")
tag_commit=$(git -C "$root" rev-parse "$tag^{commit}")
[[ "$tag_commit" == "$commit" ]] || fail "tag $tag resolves to $tag_commit, not $commit"
git_tree=$(git -C "$root" rev-parse "$commit^{tree}")

expected_source_name="rscrypto-${version}-source.tar.gz"
[[ $(basename "$source_archive") == "$expected_source_name" ]] \
  || fail "unexpected source archive name: $(basename "$source_archive")"
reproduced_dir=$(mktemp -d)
committed_toolchain=$(mktemp)
committed_lock=$(mktemp)
committed_workflow=$(mktemp)
trap 'rm -rf "$reproduced_dir"; rm -f "$committed_toolchain" "$committed_lock" "$committed_workflow"' EXIT
"$(dirname "$0")/package-release-source.sh" \
  --root "$root" \
  --version "$version" \
  --tag "$tag" \
  --commit "$commit" \
  --out "$reproduced_dir" >/dev/null
cmp -s "$source_archive" "$reproduced_dir/$expected_source_name" \
  || fail "source archive is not the deterministic archive for $commit"

expected_crate_name="rscrypto-${version}.crate"
[[ $(basename "$crate_package") == "$expected_crate_name" ]] \
  || fail "unexpected crate package name: $(basename "$crate_package")"
vcs_json=$(tar -xOf "$crate_package" "rscrypto-${version}/.cargo_vcs_info.json" 2>/dev/null) \
  || fail "crate package lacks .cargo_vcs_info.json"
jq -e --arg commit "$commit" '
  .git.sha1 == $commit
  and (.git.dirty // false) == false
  and ((.path_in_vcs // "") == "")
' <<< "$vcs_json" >/dev/null || fail "crate package is not bound to release commit $commit"

expected_ct_name="rscrypto-${version}-ct-evidence.tar.gz"
[[ $(basename "$ct_evidence") == "$expected_ct_name" ]] \
  || fail "unexpected CT evidence name: $(basename "$ct_evidence")"
ct_metadata=$(tar -xOf "$ct_evidence" CT-EVIDENCE-BUNDLE.json 2>/dev/null) \
  || fail "CT evidence lacks CT-EVIDENCE-BUNDLE.json"
jq -e \
  --arg version "$version" \
  --arg commit "$commit" \
  --arg evidence_commit "$evidence_commit" \
  --arg evidence_mode "$evidence_mode" '
  .schema_version == 1
  and .kind == "rscrypto.ct.release-evidence"
  and .crate == "rscrypto"
  and .crate_version == $version
  and .git_commit == $commit
  and .evidence_git_commit == $evidence_commit
  and .evidence_mode == $evidence_mode
' <<< "$ct_metadata" >/dev/null || fail "CT evidence identity does not match the release"

expected_controls_name="rscrypto-${version}-repository-controls.json"
[[ $(basename "$repository_controls") == "$expected_controls_name" ]] \
  || fail "unexpected repository-controls name: $(basename "$repository_controls")"
jq -e --arg commit "$commit" '
  .kind == "rscrypto.repository-controls"
  and .release_commit == $commit
' "$repository_controls" >/dev/null || fail "repository controls are not bound to release commit $commit"

git -C "$root" show "$commit:rust-toolchain.toml" > "$committed_toolchain"
git -C "$root" show "$commit:Cargo.lock" > "$committed_lock"
git -C "$root" show "$commit:.github/workflows/release.yaml" > "$committed_workflow"
toolchain_channel=$(awk -F'"' '/^channel/ {print $2}' "$committed_toolchain")
[[ -n "$toolchain_channel" ]] || fail "committed rust-toolchain.toml has no channel"
active_toolchain=$(cd "$root" && rustup show active-toolchain | awk '{print $1}')
[[ "$active_toolchain" == "$toolchain_channel"* ]] \
  || fail "active toolchain $active_toolchain does not match pinned channel $toolchain_channel"
rustc_version=$(cd "$root" && rustc -Vv)
cargo_version=$(cd "$root" && cargo -V)

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

source_sha256=$(sha256_file "$source_archive")
crate_sha256=$(sha256_file "$crate_package")
ct_sha256=$(sha256_file "$ct_evidence")
controls_sha256=$(sha256_file "$repository_controls")
toolchain_sha256=$(sha256_file "$committed_toolchain")
lock_sha256=$(sha256_file "$committed_lock")
workflow_sha256=$(sha256_file "$committed_workflow")

mkdir -p "$(dirname "$output")"
output_tmp="${output}.tmp.$$"
trap 'rm -rf "$reproduced_dir"; rm -f "$committed_toolchain" "$committed_lock" "$committed_workflow" "$output_tmp"' EXIT
jq -nS \
  --arg version "$version" \
  --arg tag "$tag" \
  --arg tag_object "$tag_object" \
  --arg commit "$commit" \
  --arg tree "$git_tree" \
  --arg evidence_commit "$evidence_commit" \
  --arg evidence_mode "$evidence_mode" \
  --arg toolchain_channel "$toolchain_channel" \
  --arg active_toolchain "$active_toolchain" \
  --arg rustc "$rustc_version" \
  --arg cargo "$cargo_version" \
  --arg toolchain_sha256 "$toolchain_sha256" \
  --arg lock_sha256 "$lock_sha256" \
  --arg workflow_sha256 "$workflow_sha256" \
  --arg source_name "$(basename "$source_archive")" \
  --arg source_sha256 "$source_sha256" \
  --arg crate_name "$(basename "$crate_package")" \
  --arg crate_sha256 "$crate_sha256" \
  --arg ct_name "$(basename "$ct_evidence")" \
  --arg ct_sha256 "$ct_sha256" \
  --arg controls_name "$(basename "$repository_controls")" \
  --arg controls_sha256 "$controls_sha256" '
  {
    schema_version: 1,
    kind: "rscrypto.release-manifest",
    crate: "rscrypto",
    crate_version: $version,
    release: {
      tag: $tag,
      tag_object: $tag_object,
      git_commit: $commit,
      git_tree: $tree
    },
    toolchain: {
      channel: $toolchain_channel,
      active: $active_toolchain,
      rustc: $rustc,
      cargo: $cargo,
      manifest: {path: "rust-toolchain.toml", sha256: $toolchain_sha256}
    },
    inputs: {
      cargo_lock: {path: "Cargo.lock", sha256: $lock_sha256},
      release_workflow: {path: ".github/workflows/release.yaml", sha256: $workflow_sha256}
    },
    evidence: {
      git_commit: $evidence_commit,
      mode: $evidence_mode
    },
    artifacts: {
      source_archive: {
        name: $source_name,
        sha256: $source_sha256,
        format: "git-archive+tar+gzip-n",
        prefix: ("rscrypto-" + $version + "/")
      },
      crate_package: {name: $crate_name, sha256: $crate_sha256},
      ct_evidence: {name: $ct_name, sha256: $ct_sha256},
      repository_controls: {name: $controls_name, sha256: $controls_sha256}
    }
  }
' > "$output_tmp"
mv "$output_tmp" "$output"
trap 'rm -rf "$reproduced_dir"; rm -f "$committed_toolchain" "$committed_lock" "$committed_workflow"' EXIT

manifest_sha256=$(sha256_file "$output")
if [[ -n ${GITHUB_OUTPUT:-} ]]; then
  {
    echo "manifest_path=$output"
    echo "manifest_name=$(basename "$output")"
    echo "manifest_sha256=$manifest_sha256"
  } >> "$GITHUB_OUTPUT"
fi

echo "release manifest: $output"
echo "sha256:          $manifest_sha256"
