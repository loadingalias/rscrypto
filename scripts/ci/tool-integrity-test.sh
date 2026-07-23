#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

fail() {
  echo "tool integrity test failure: $*" >&2
  exit 1
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

make_direct_fixture() {
  local fixture=$1
  mkdir -p "$fixture/.config" "$fixture/scripts/ci" "$fixture/scripts/lib"
  cp "$REPO_ROOT/.config/ci-tool-archives.tsv" "$fixture/.config/"
  cp "$REPO_ROOT/scripts/lib/ci-tool-integrity.sh" "$fixture/scripts/lib/"
  cp "$REPO_ROOT/scripts/ci/install-codecov.sh" "$REPO_ROOT/scripts/ci/nostd-wasm-suite.sh" \
    "$fixture/scripts/ci/"
}

set_manifest_digest() {
  local fixture=$1
  local tool=$2
  local os=$3
  local architecture=$4
  local digest=$5
  local manifest="$fixture/.config/ci-tool-archives.tsv"
  awk -F '\t' -v OFS='\t' \
    -v tool="$tool" -v os="$os" -v architecture="$architecture" -v digest="$digest" '
      $1 == tool && $3 == os && $4 == architecture { $7 = digest }
      { print }
    ' "$manifest" >"$manifest.tmp"
  mv "$manifest.tmp" "$manifest"
}

direct_bin="$TMP_ROOT/direct-bin"
direct_log="$TMP_ROOT/direct.log"
mkdir -p "$direct_bin"

cat >"$direct_bin/uname" <<'SH'
#!/usr/bin/env bash
case "${1:-}" in
  -s) printf '%s\n' "${MOCK_UNAME_S:-Linux}" ;;
  -m) printf '%s\n' "${MOCK_UNAME_M:-x86_64}" ;;
  *) printf '%s %s\n' "${MOCK_UNAME_S:-Linux}" "${MOCK_UNAME_M:-x86_64}" ;;
esac
SH

cat >"$direct_bin/curl" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'curl %s\n' "$*" >>"$MOCK_COMMAND_LOG"
[[ "${MOCK_CURL_FAIL:-0}" != 1 ]] || exit 22
output=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output | -o)
      output=$2
      shift 2
      ;;
    *) shift ;;
  esac
done
[[ -n "$output" ]]
cp "$MOCK_DOWNLOAD_FILE" "$output"
SH

cat >"$direct_bin/tar" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'tar %s\n' "$*" >>"$MOCK_COMMAND_LOG"
destination=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -C)
      destination=$2
      shift 2
      ;;
    *) shift ;;
  esac
done
[[ -n "$destination" ]]
root="$destination/wasmtime-v46.0.1-x86_64-linux"
mkdir -p "$root"
cat >"$root/wasmtime" <<'EOF'
#!/usr/bin/env bash
printf 'wasmtime 46.0.1 (mock)\n'
printf 'wasmtime executed\n' >>"$MOCK_EXEC_LOG"
EOF
chmod +x "$root/wasmtime"
SH

cat >"$direct_bin/cargo" <<'SH'
#!/usr/bin/env bash
printf 'cargo %s\n' "$*" >>"$MOCK_COMMAND_LOG"
SH

cat >"$direct_bin/rustup" <<'SH'
#!/usr/bin/env bash
printf 'rustup %s\n' "$*" >>"$MOCK_COMMAND_LOG"
SH
chmod +x "$direct_bin"/*

codecov_artifact="$TMP_ROOT/codecov"
cat >"$codecov_artifact" <<'SH'
#!/usr/bin/env bash
printf 'codecovcli version 11.3.1\n'
printf 'codecov executed\n' >>"$MOCK_EXEC_LOG"
SH
codecov_digest=$(sha256_file "$codecov_artifact")

direct_fixture="$TMP_ROOT/direct-fixture"
make_direct_fixture "$direct_fixture"
set_manifest_digest "$direct_fixture" codecov linux x86_64 "$codecov_digest"

codecov_temp="$TMP_ROOT/codecov-valid"
codecov_output="$TMP_ROOT/codecov.output"
codecov_exec="$TMP_ROOT/codecov.exec"
mkdir -p "$codecov_temp"
PATH="$direct_bin:$PATH" \
  RUNNER_TEMP="$codecov_temp" \
  GITHUB_OUTPUT="$codecov_output" \
  MOCK_COMMAND_LOG="$direct_log" \
  MOCK_DOWNLOAD_FILE="$codecov_artifact" \
  MOCK_EXEC_LOG="$codecov_exec" \
  "$direct_fixture/scripts/ci/install-codecov.sh"
installed_codecov=$(sed -n 's/^binary=//p' "$codecov_output")
[[ -x "$installed_codecov" ]] || fail "valid Codecov artifact was not installed"
grep -Fqx 'codecov executed' "$codecov_exec" \
  || fail "verified Codecov artifact did not execute"

invalid_artifact="$TMP_ROOT/invalid-codecov"
cat >"$invalid_artifact" <<'SH'
#!/usr/bin/env bash
printf 'invalid artifact executed\n' >>"$MOCK_EXEC_LOG"
printf 'codecovcli version 11.3.1\n'
SH
invalid_temp="$TMP_ROOT/codecov-invalid"
invalid_exec="$TMP_ROOT/codecov-invalid.exec"
mkdir -p "$invalid_temp"
if PATH="$direct_bin:$PATH" \
  RUNNER_TEMP="$invalid_temp" \
  GITHUB_OUTPUT="$TMP_ROOT/codecov-invalid.output" \
  MOCK_COMMAND_LOG="$direct_log" \
  MOCK_DOWNLOAD_FILE="$invalid_artifact" \
  MOCK_EXEC_LOG="$invalid_exec" \
  "$direct_fixture/scripts/ci/install-codecov.sh" >/dev/null 2>&1; then
  fail "checksum mismatch installed Codecov"
fi
[[ ! -e "$invalid_exec" ]] || fail "checksum mismatch executed Codecov"
[[ ! -s "$TMP_ROOT/codecov-invalid.output" ]] \
  || fail "checksum mismatch published an installed Codecov path"

missing_temp="$TMP_ROOT/codecov-missing"
mkdir -p "$missing_temp"
if PATH="$direct_bin:$PATH" \
  RUNNER_TEMP="$missing_temp" \
  GITHUB_OUTPUT="$TMP_ROOT/codecov-missing.output" \
  MOCK_COMMAND_LOG="$direct_log" \
  MOCK_DOWNLOAD_FILE="$codecov_artifact" \
  MOCK_CURL_FAIL=1 \
  MOCK_EXEC_LOG="$TMP_ROOT/codecov-missing.exec" \
  "$direct_fixture/scripts/ci/install-codecov.sh" >/dev/null 2>&1; then
  fail "missing download was accepted"
fi
[[ ! -e "$TMP_ROOT/codecov-missing.exec" ]] \
  || fail "missing download reached executable fallback"

: >"$direct_log"
unsupported_temp="$TMP_ROOT/codecov-unsupported"
mkdir -p "$unsupported_temp"
if PATH="$direct_bin:$PATH" \
  RUNNER_TEMP="$unsupported_temp" \
  GITHUB_OUTPUT="$TMP_ROOT/codecov-unsupported.output" \
  MOCK_COMMAND_LOG="$direct_log" \
  MOCK_DOWNLOAD_FILE="$codecov_artifact" \
  MOCK_UNAME_M=riscv64 \
  MOCK_EXEC_LOG="$TMP_ROOT/codecov-unsupported.exec" \
  "$direct_fixture/scripts/ci/install-codecov.sh" >/dev/null 2>&1; then
  fail "unsupported direct-tool architecture was accepted"
fi
[[ ! -s "$direct_log" ]] || fail "unsupported architecture attempted a download"

missing_platform_fixture="$TMP_ROOT/missing-platform-fixture"
make_direct_fixture "$missing_platform_fixture"
awk -F '\t' '$1 != "codecov"' \
  "$missing_platform_fixture/.config/ci-tool-archives.tsv" \
  >"$missing_platform_fixture/.config/ci-tool-archives.tsv.tmp"
mv "$missing_platform_fixture/.config/ci-tool-archives.tsv.tmp" \
  "$missing_platform_fixture/.config/ci-tool-archives.tsv"
: >"$direct_log"
mkdir -p "$TMP_ROOT/codecov-no-platform"
if PATH="$direct_bin:$PATH" \
  RUNNER_TEMP="$TMP_ROOT/codecov-no-platform" \
  GITHUB_OUTPUT="$TMP_ROOT/codecov-no-platform.output" \
  MOCK_COMMAND_LOG="$direct_log" \
  MOCK_DOWNLOAD_FILE="$codecov_artifact" \
  MOCK_EXEC_LOG="$TMP_ROOT/codecov-no-platform.exec" \
  "$missing_platform_fixture/scripts/ci/install-codecov.sh" >/dev/null 2>&1; then
  fail "missing platform contract was accepted"
fi
[[ ! -s "$direct_log" ]] || fail "missing platform contract attempted a download"

wasmtime_artifact="$TMP_ROOT/wasmtime.tar.xz"
printf 'authenticated mock Wasmtime archive\n' >"$wasmtime_artifact"
wasmtime_digest=$(sha256_file "$wasmtime_artifact")
set_manifest_digest "$direct_fixture" wasmtime linux x86_64 "$wasmtime_digest"
wasmtime_home="$TMP_ROOT/wasmtime-home"
wasmtime_exec="$TMP_ROOT/wasmtime.exec"
: >"$direct_log"
(
  cd "$direct_fixture"
  PATH="$direct_bin:$PATH" \
    WASMTIME_HOME="$wasmtime_home" \
    MOCK_COMMAND_LOG="$direct_log" \
    MOCK_DOWNLOAD_FILE="$wasmtime_artifact" \
    MOCK_EXEC_LOG="$wasmtime_exec" \
    scripts/ci/nostd-wasm-suite.sh wasm32-wasip1 shallow
) >/dev/null
[[ -x "$wasmtime_home/bin/wasmtime" ]] || fail "verified Wasmtime was not installed"
grep -Fq 'tar -xJf' "$direct_log" || fail "verified Wasmtime was not extracted"
grep -Fqx 'wasmtime executed' "$wasmtime_exec" || fail "verified Wasmtime was not executed"

: >"$direct_log"
bad_wasmtime_exec="$TMP_ROOT/bad-wasmtime.exec"
if (
  cd "$direct_fixture"
  PATH="$direct_bin:$PATH" \
    WASMTIME_HOME="$TMP_ROOT/bad-wasmtime-home" \
    MOCK_COMMAND_LOG="$direct_log" \
    MOCK_DOWNLOAD_FILE="$invalid_artifact" \
    MOCK_EXEC_LOG="$bad_wasmtime_exec" \
    scripts/ci/nostd-wasm-suite.sh wasm32-wasip1 shallow
) >/dev/null 2>&1; then
  fail "invalid Wasmtime archive was accepted"
fi
if grep -Fq 'tar ' "$direct_log"; then
  fail "invalid Wasmtime archive reached extraction"
fi
[[ ! -e "$bad_wasmtime_exec" ]] || fail "invalid Wasmtime archive reached execution"

zig_artifact="$TMP_ROOT/zig.tar.xz"
printf 'authenticated mock Zig archive\n' >"$zig_artifact"
zig_digest=$(sha256_file "$zig_artifact")
set_manifest_digest "$direct_fixture" zig linux x86_64 "$zig_digest"
zig_download="$TMP_ROOT/zig-download"
mkdir -p "$zig_download"
(
  cd "$direct_fixture"
  export PATH="$direct_bin:$PATH"
  export MOCK_COMMAND_LOG="$direct_log"
  export MOCK_DOWNLOAD_FILE="$zig_artifact"
  source scripts/lib/ci-tool-integrity.sh
  ci_tool_download zig "$zig_download"
)
cmp "$zig_artifact" \
  "$zig_download/zig-x86_64-linux-0.17.0-dev.1282+c0f9b51d8.tar.xz" \
  || fail "verified Zig artifact did not match the authenticated download"

package_bin="$TMP_ROOT/package-bin"
package_log="$TMP_ROOT/package.log"
package_state="$TMP_ROOT/package.state"
mkdir -p "$package_bin"
: >"$package_state"

cat >"$package_bin/cargo" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'cargo %s\n' "$*" >>"$MOCK_PACKAGE_LOG"
if [[ "${1:-}" == install && "${2:-}" == --list ]]; then
  while read -r package version; do
    [[ -n "$package" ]] || continue
    printf '%s v%s:\n    %s\n' "$package" "$version" "$package"
  done <"$MOCK_CARGO_STATE"
  exit 0
fi
[[ "${1:-}" == install ]] || exit 0
shift
package=""
required=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry)
      [[ "$2" == crates-io ]]
      shift 2
      ;;
    --version)
      required=$2
      shift 2
      ;;
    --locked | --force) shift ;;
    -*) exit 91 ;;
    *)
      [[ -z "$package" ]] || exit 92
      package=$1
      shift
      ;;
  esac
done
[[ -n "$package" && "$required" == =* ]]
case "$CARGO_HOME" in
  "$RUNNER_TEMP"/rscrypto-ci-tools.*/cargo) ;;
  *) exit 93 ;;
esac
version=${required#=}
awk -v package="$package" '$1 != package' "$MOCK_CARGO_STATE" >"$MOCK_CARGO_STATE.tmp"
printf '%s %s\n' "$package" "$version" >>"$MOCK_CARGO_STATE.tmp"
mv "$MOCK_CARGO_STATE.tmp" "$MOCK_CARGO_STATE"
binary="$CARGO_HOME/bin/$package"
mkdir -p "$(dirname "$binary")"
printf '#!/usr/bin/env bash\nprintf "%%s %%s\\n" %q %q\n' "$package" "$version" >"$binary"
chmod +x "$binary"
SH

cat >"$package_bin/go" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'go %s\n' "$*" >>"$MOCK_PACKAGE_LOG"
[[ "$1" == install && "$2" == github.com/rhysd/actionlint/cmd/actionlint@v1.7.12 ]]
case "$GOMODCACHE" in
  "$RUNNER_TEMP"/rscrypto-ci-tools.*/go/pkg/mod) ;;
  *) exit 93 ;;
esac
case "$GOCACHE" in
  "$RUNNER_TEMP"/rscrypto-ci-tools.*/go-build) ;;
  *) exit 94 ;;
esac
mkdir -p "$GOBIN"
cat >"$GOBIN/actionlint" <<'EOF'
#!/usr/bin/env bash
printf '1.7.12\n'
EOF
chmod +x "$GOBIN/actionlint"
SH

cat >"$package_bin/rustup" <<'SH'
#!/usr/bin/env bash
printf 'rustup %s\n' "$*" >>"$MOCK_PACKAGE_LOG"
SH

cat >"$package_bin/rustc" <<'SH'
#!/usr/bin/env bash
printf 'rustc 1.97.0-nightly\ncommit-date: 2026-04-26\n'
SH

cp "$direct_bin/uname" "$package_bin/uname"
chmod +x "$package_bin"/*

package_home="$TMP_ROOT/package-home"
package_temp="$TMP_ROOT/package-temp"
mkdir -p "$package_home/.cargo/bin"
mkdir -p "$package_temp"
: >"$package_log"
for mode in standard quality release rail ci supply-chain ibm bench fuzz coverage minimal none; do
  HOME="$package_home" \
    RUNNER_TEMP="$package_temp" \
    PATH="$package_bin:$PATH" \
    MOCK_PACKAGE_LOG="$package_log" \
    MOCK_CARGO_STATE="$package_state" \
    "$REPO_ROOT/scripts/ci/install-tools.sh" "$mode" >/dev/null
done

cat >"$package_bin/sudo" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
"$@"
SH

cat >"$package_bin/apt-get" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'apt-get %s\n' "$*" >>"$MOCK_PACKAGE_LOG"
SH

cat >"$package_bin/dpkg-query" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
case "${*: -1}" in
  build-essential) printf '12.10ubuntu1' ;;
  git) printf '1:2.43.0-1ubuntu7.3' ;;
  libgmp-dev) printf '2:6.3.0+dfsg-2ubuntu6' ;;
  libmpfr-dev) printf '4.2.1-1build1' ;;
  m4) printf '1.4.19-4build1' ;;
  opam) printf '2.1.5-1' ;;
  pkg-config) printf '1.8.1-2build1' ;;
  zlib1g-dev) printf '1:1.3.dfsg-3.1ubuntu2' ;;
  musl-tools) printf '1.2.4-2' ;;
  *) exit 96 ;;
esac
SH

cat >"$package_bin/git" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'git %s\n' "$*" >>"$MOCK_PACKAGE_LOG"
[[ "$1" == -C && "$3" == rev-parse && "$4" == HEAD ]]
printf '49f6d620cf20ae0168cfcbeb2c33932e06cb4b74\n'
SH

cat >"$package_bin/opam" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'opamroot=%s opam %s\n' "$OPAMROOT" "$*" >>"$MOCK_PACKAGE_LOG"
case "$1" in
  init)
    mkdir -p "$OPAMROOT/repo/default/.git"
    ;;
  switch)
    [[ "$2" == create ]]
    ;;
  install)
    mkdir -p "$OPAMROOT/$OPAMSWITCH/bin"
    cat >"$OPAMROOT/$OPAMSWITCH/bin/binsec" <<'EOF'
#!/usr/bin/env bash
printf 'BINSEC version 0.11.1\n'
EOF
    chmod +x "$OPAMROOT/$OPAMSWITCH/bin/binsec"
    ;;
  reinstall) ;;
  list)
    printf '%s\n' \
      ocaml-base-compiler.5.2.1 \
      unisim_archisec.0.0.14 \
      bitwuzla.1.0.6 \
      bitwuzla-cxx.0.9.0 \
      binsec.0.11.1
    ;;
  var)
    [[ "$2" == bin ]]
    printf '%s\n' "$OPAMROOT/$OPAMSWITCH/bin"
    ;;
  *) exit 97 ;;
esac
SH
chmod +x "$package_bin/sudo" "$package_bin/apt-get" \
  "$package_bin/dpkg-query" "$package_bin/git" "$package_bin/opam"

ct_home="$TMP_ROOT/ct-home"
ct_temp="$TMP_ROOT/ct-temp"
ct_state="$TMP_ROOT/ct.state"
ct_log="$TMP_ROOT/ct.log"
ct_os_release="$TMP_ROOT/ct-os-release"
ct_installer="$TMP_ROOT/install-tools-ct.sh"
mkdir -p "$ct_home" "$ct_temp"
: >"$ct_state"
: >"$ct_log"
printf 'ID=ubuntu\nVERSION_ID="24.04"\n' >"$ct_os_release"
cp "$REPO_ROOT/scripts/ci/install-tools.sh" "$ct_installer"
sed -i.bak "s#/etc/os-release#$ct_os_release#g" "$ct_installer"
rm -f "$ct_installer.bak"
HOME="$ct_home" \
  RUNNER_TEMP="$ct_temp" \
  PATH="$package_bin:$PATH" \
  MOCK_PACKAGE_LOG="$ct_log" \
  MOCK_CARGO_STATE="$ct_state" \
  "$ct_installer" ct-linux >/dev/null
grep -Fq \
  'apt-get install -y --no-install-recommends --allow-downgrades build-essential=12.10ubuntu1 git=1:2.43.0-1ubuntu7.3' \
  "$ct_log" || fail "ct-linux did not select exact Ubuntu package versions"
grep -Fq \
  'opam init --bare --disable-sandboxing --no-setup --no-opamrc -y default git+https://github.com/ocaml/opam-repository.git#49f6d620cf20ae0168cfcbeb2c33932e06cb4b74' \
  "$ct_log" || fail "ct-linux did not select the commit-pinned OPAM repository"
grep -Eq '^opamroot=.*/ct-temp/rscrypto-ci-tools\.[^/]+/opam opam switch create rscrypto-ct ocaml-base-compiler\.5\.2\.1 ' \
  "$ct_log" || fail "ct-linux did not use a fresh exact OPAM switch"

for contract in \
  'cargo-nextest =0.9.140' \
  'cargo-deny =0.20.2' \
  'cargo-audit =0.22.2' \
  'cargo-rail =0.18.0' \
  'cargo-semver-checks =0.48.0' \
  'just =1.57.0' \
  'zizmor =1.26.1' \
  'cargo-criterion =1.1.0' \
  'critcmp =0.1.8' \
  'cargo-fuzz =0.13.2' \
  'cargo-llvm-cov =0.8.7'; do
  package=${contract%% *}
  version=${contract#* }
  grep -Fq "cargo install --registry crates-io $package --locked --version $version --force" \
    "$package_log" || fail "$package was not installed through its exact Cargo contract"
done
grep -Fq 'go install github.com/rhysd/actionlint/cmd/actionlint@v1.7.12' "$package_log" \
  || fail "actionlint was not installed at an exact Go module version"
grep -Fq 'rustup component add llvm-tools-preview' "$package_log" \
  || fail "coverage did not use the pinned rustup toolchain boundary"
if grep -Eq 'binstall|latest' "$package_log"; then
  fail "tool mode selected a mutable or Cargo-binstall path"
fi

cached_home="$TMP_ROOT/cached-home"
cached_state="$TMP_ROOT/cached.state"
malicious_exec="$TMP_ROOT/malicious.exec"
github_path_file="$TMP_ROOT/github.path"
mkdir -p "$cached_home/.cargo/bin"
: >"$cached_state"
: >"$github_path_file"
cat >"$cached_home/.cargo/bin/just" <<'SH'
#!/usr/bin/env bash
printf 'forged exact-version cache executed\n' >>"$MALICIOUS_EXEC_LOG"
printf 'just 1.57.0\n'
SH
chmod +x "$cached_home/.cargo/bin/just"
cat >"$cached_home/.cargo/.crates.toml" <<'EOF'
[v1]
"just 1.57.0 (registry+https://github.com/rust-lang/crates.io-index)" = ["just"]
EOF
: >"$package_log"
HOME="$cached_home" \
  RUNNER_TEMP="$package_temp" \
  GITHUB_PATH="$github_path_file" \
  PATH="$package_bin:$PATH" \
  MOCK_PACKAGE_LOG="$package_log" \
  MOCK_CARGO_STATE="$cached_state" \
  MALICIOUS_EXEC_LOG="$malicious_exec" \
  "$REPO_ROOT/scripts/ci/install-tools.sh" minimal >/dev/null
[[ ! -e "$malicious_exec" ]] \
  || fail "forged exact-version cached binary executed before authenticated replacement"
grep -Fq 'cargo install --registry crates-io just --locked --version =1.57.0 --force' "$package_log" \
  || fail "forged cache did not trigger a fresh authenticated install"
trusted_bin=$(tail -n 1 "$github_path_file")
case "$trusted_bin" in
  "$package_temp"/rscrypto-ci-tools.*/cargo/bin) ;;
  *) fail "authenticated Cargo tool root was not exported to later CI steps" ;;
esac

: >"$package_log"
MOCK_PACKAGE_LOG="$package_log" PATH="$package_bin:$PATH" \
  "$REPO_ROOT/scripts/ci/setup-toolchain.sh" \
  nightly-2026-04-27 'clippy, rustfmt' >/dev/null
grep -Fq \
  'rustup toolchain install nightly-2026-04-27 --profile minimal --no-self-update --component clippy --component rustfmt' \
  "$package_log" || fail "rustup toolchain command was not exact"
grep -Fq 'rustup default nightly-2026-04-27' "$package_log" \
  || fail "rustup did not select the exact toolchain"
if MOCK_PACKAGE_LOG="$package_log" PATH="$package_bin:$PATH" \
  "$REPO_ROOT/scripts/ci/setup-toolchain.sh" nightly clippy >/dev/null 2>&1; then
  fail "mutable rustup channel was accepted"
fi

echo "CI tool integrity regression tests passed"
