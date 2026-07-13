#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="$SCRIPT_DIR/upgrade-actions.sh"
TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

fixture="$TMP_ROOT/repository"
fake_bin="$TMP_ROOT/bin"
mkdir -p "$fixture/.github" "$fake_bin"

cat >"$fixture/.github/actions-lock.yaml" <<'YAML'
github/codeql-action/upload-sarif:
  ref: v4.37.0
  sha: "99df26d4f13ea111d4ec1a7dddef6063f76b97e9"
YAML

cat >"$fake_bin/gh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

if [[ "$1" == "auth" && "$2" == "status" ]]; then
  exit 0
fi

[[ "$1" == "api" ]]
[[ "$2" == "--paginate" ]]
[[ "$3" == "repos/github/codeql-action/tags?per_page=100" ]]
printf '%s\n' v4.37.0
SH
chmod +x "$fake_bin/gh"

cat >"$fake_bin/curl" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

url=${!#}
[[ "$url" == "https://raw.githubusercontent.com/github/codeql-action/v4.37.0/upload-sarif/action.yml" ]]
cat <<'YAML'
runs:
  using: node24
YAML
SH
chmod +x "$fake_bin/curl"

output=$(PATH="$fake_bin:$PATH" "$CHECKER" --check --root "$fixture")
grep -Fq "Up to date: v4.37.0" <<<"$output"
grep -Fq "All GitHub Action refs are current under the Node 24 policy." <<<"$output"

echo "Action upgrade regression tests passed"
