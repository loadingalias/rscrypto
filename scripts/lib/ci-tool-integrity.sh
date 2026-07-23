#!/usr/bin/env bash

CI_TOOL_INTEGRITY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CI_TOOL_REPO_ROOT="$(cd "$CI_TOOL_INTEGRITY_DIR/../.." && pwd)"
CI_TOOL_ARCHIVES="$CI_TOOL_REPO_ROOT/.config/ci-tool-archives.tsv"

ci_tool_fail() {
  echo "CI tool integrity error: $*" >&2
  return 1
}

ci_tool_detect_host() {
  case "$(uname -s)" in
    Linux) CI_TOOL_HOST_OS=linux ;;
    Darwin) CI_TOOL_HOST_OS=macos ;;
    *) ci_tool_fail "unsupported host OS: $(uname -s)" || return ;;
  esac

  case "$(uname -m)" in
    x86_64 | amd64) CI_TOOL_HOST_ARCH=x86_64 ;;
    aarch64 | arm64) CI_TOOL_HOST_ARCH=aarch64 ;;
    *) ci_tool_fail "unsupported host architecture: $(uname -m)" || return ;;
  esac
}

ci_tool_validate_record() {
  local tool=$1
  local version=$2
  local os=$3
  local architecture=$4
  local filename=$5
  local url=$6
  local digest=$7

  case "$tool" in
    wasmtime | zig | codecov) ;;
    *) ci_tool_fail "unknown direct CI tool: $tool" || return ;;
  esac
  [[ "$version" =~ ^v?[0-9]+\.[0-9]+\.[0-9]+[-+A-Za-z0-9.]*$ ]] \
    || ci_tool_fail "invalid version for $tool: $version" || return
  case "$version" in
    *[Ll][Aa][Tt][Ee][Ss][Tt]*) ci_tool_fail "mutable version for $tool: $version" || return ;;
  esac
  [[ "$os" == linux || "$os" == macos ]] \
    || ci_tool_fail "invalid host OS for $tool: $os" || return
  [[ "$architecture" == x86_64 || "$architecture" == aarch64 ]] \
    || ci_tool_fail "invalid host architecture for $tool: $architecture" || return
  [[ "$filename" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]] \
    || ci_tool_fail "invalid filename for $tool: $filename" || return
  [[ "$url" == https://* ]] \
    || ci_tool_fail "invalid URL for $tool: $url" || return
  case "$url" in
    *[Ll][Aa][Tt][Ee][Ss][Tt]*) ci_tool_fail "mutable URL for $tool: $url" || return ;;
  esac
  [[ "$url" != *\?* && "$url" != *\#* && "${url##*/}" == "$filename" ]] \
    || ci_tool_fail "URL and filename disagree for $tool" || return
  [[ "$url" == *"$version"* ]] \
    || ci_tool_fail "URL does not contain the exact $tool version" || return
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] \
    || ci_tool_fail "invalid SHA-256 for $tool $os/$architecture" || return
}

ci_tool_resolve() {
  local requested_tool=$1
  local entry_tool entry_version entry_os entry_arch entry_filename entry_url entry_digest extra
  local selector seen_selectors=$'\n'
  local matches=0

  [[ -f "$CI_TOOL_ARCHIVES" ]] \
    || ci_tool_fail "missing archive manifest: $CI_TOOL_ARCHIVES" || return
  ci_tool_detect_host || return

  while IFS=$'\t' read -r \
    entry_tool entry_version entry_os entry_arch entry_filename entry_url entry_digest extra \
    || [[ -n "$entry_tool$entry_version$entry_os$entry_arch$entry_filename$entry_url$entry_digest$extra" ]]; do
    [[ -z "$entry_tool" || "$entry_tool" == \#* ]] && continue
    [[ -z "$extra" ]] || ci_tool_fail "unexpected archive manifest field for $entry_tool" || return
    ci_tool_validate_record \
      "$entry_tool" "$entry_version" "$entry_os" "$entry_arch" \
      "$entry_filename" "$entry_url" "$entry_digest" || return

    selector="$entry_tool/$entry_os/$entry_arch"
    case "$seen_selectors" in
      *$'\n'"$selector"$'\n'*) ci_tool_fail "duplicate archive contract: $selector" || return ;;
    esac
    seen_selectors+="$selector"$'\n'

    if [[ "$entry_tool" == "$requested_tool" \
      && "$entry_os" == "$CI_TOOL_HOST_OS" \
      && "$entry_arch" == "$CI_TOOL_HOST_ARCH" ]]; then
      matches=$((matches + 1))
      CI_TOOL_NAME=$entry_tool
      CI_TOOL_VERSION=$entry_version
      CI_TOOL_FILENAME=$entry_filename
      CI_TOOL_URL=$entry_url
      CI_TOOL_SHA256=$entry_digest
    fi
  done <"$CI_TOOL_ARCHIVES"

  [[ "$matches" -eq 1 ]] \
    || ci_tool_fail "no unique $requested_tool archive for $CI_TOOL_HOST_OS/$CI_TOOL_HOST_ARCH" || return
}

ci_tool_validate_manifest() {
  local tool
  for tool in wasmtime zig codecov; do
    case "$tool" in
      wasmtime)
        local expected_platforms=(linux:x86_64 linux:aarch64 macos:x86_64 macos:aarch64)
        ;;
      zig | codecov)
        local expected_platforms=(linux:x86_64)
        ;;
    esac

    local platform
    for platform in "${expected_platforms[@]}"; do
      local expected_os=${platform%%:*}
      local expected_arch=${platform#*:}
      local count
      count=$(awk -F '\t' \
        -v tool="$tool" -v os="$expected_os" -v arch="$expected_arch" \
        '$1 == tool && $3 == os && $4 == arch { count++ } END { print count + 0 }' \
        "$CI_TOOL_ARCHIVES")
      [[ "$count" -eq 1 ]] \
        || ci_tool_fail "expected one $tool archive for $expected_os/$expected_arch" || return
    done
  done

  local entry_tool entry_version entry_os entry_arch entry_filename entry_url entry_digest extra
  local selector seen_selectors=$'\n'
  while IFS=$'\t' read -r \
    entry_tool entry_version entry_os entry_arch entry_filename entry_url entry_digest extra \
    || [[ -n "$entry_tool$entry_version$entry_os$entry_arch$entry_filename$entry_url$entry_digest$extra" ]]; do
    [[ -z "$entry_tool" || "$entry_tool" == \#* ]] && continue
    [[ -z "$extra" ]] || ci_tool_fail "unexpected archive manifest field for $entry_tool" || return
    ci_tool_validate_record \
      "$entry_tool" "$entry_version" "$entry_os" "$entry_arch" \
      "$entry_filename" "$entry_url" "$entry_digest" || return
    selector="$entry_tool/$entry_os/$entry_arch"
    case "$seen_selectors" in
      *$'\n'"$selector"$'\n'*) ci_tool_fail "duplicate archive contract: $selector" || return ;;
    esac
    seen_selectors+="$selector"$'\n'
  done <"$CI_TOOL_ARCHIVES"
}

ci_tool_sha256() {
  local path=$1
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    ci_tool_fail "sha256sum or shasum is required"
  fi
}

ci_tool_download() {
  local tool=$1
  local destination_dir=$2
  local partial actual

  ci_tool_resolve "$tool" || return
  [[ -d "$destination_dir" ]] \
    || ci_tool_fail "download destination is not a directory: $destination_dir" || return

  CI_TOOL_ARCHIVE_PATH="$destination_dir/$CI_TOOL_FILENAME"
  [[ ! -e "$CI_TOOL_ARCHIVE_PATH" ]] \
    || ci_tool_fail "refusing to overwrite archive: $CI_TOOL_ARCHIVE_PATH" || return
  partial=$(mktemp "$destination_dir/.${CI_TOOL_FILENAME}.part.XXXXXX")

  if ! curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location \
    --retry 3 --retry-delay 2 --output "$partial" "$CI_TOOL_URL"; then
    ci_tool_fail "download failed for $CI_TOOL_NAME $CI_TOOL_VERSION" || return
  fi

  actual=$(ci_tool_sha256 "$partial") || return
  if [[ "$actual" != "$CI_TOOL_SHA256" ]]; then
    ci_tool_fail \
      "SHA-256 mismatch for $CI_TOOL_FILENAME (expected $CI_TOOL_SHA256, got $actual)" || return
  fi

  mv "$partial" "$CI_TOOL_ARCHIVE_PATH"
}
