#!/usr/bin/env bash
set -euo pipefail

# extract-results.sh — Generate benchmark_results/OVERVIEW.md from extracted results.
#
# Usage:
#   scripts/bench/extract-results.sh                     Latest date
#   scripts/bench/extract-results.sh --date 2026-04-15   Specific date
#   scripts/bench/extract-results.sh <run-id>            Download + extract + generate
#   scripts/bench/extract-results.sh --filter crc         Filter groups
#
# Requires: python3, gh (for run-id mode)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_ROOT="$REPO_ROOT/benchmark_results"
REPO="loadingalias/rscrypto"

usage() {
  cat <<'USAGE'
Usage: extract-results.sh [<run-id> | --date YYYY-MM-DD | --latest] [--filter <pattern>]

Modes:
  <run-id>              Download CI artifacts, extract, generate OVERVIEW.md
  --date YYYY-MM-DD     Generate from existing results for that date
  --latest (default)    Generate from the most recent date directory

Options:
  --filter <pattern>    Only include groups containing <pattern>
  --output <path>       Output path (default: benchmark_results/OVERVIEW.md)
USAGE
  exit "${1:-0}"
}

MODE="latest"
RUN_ID=""
DATE=""
FILTER=""
OUTPUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --date)   DATE="${2:-}"; MODE="date"; shift 2 ;;
    --latest) MODE="latest"; shift ;;
    --filter) FILTER="${2:-}"; shift 2 ;;
    --output) OUTPUT="${2:-}"; shift 2 ;;
    -h|--help) usage 0 ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        RUN_ID="$1"; MODE="run"
      else
        echo "error: unknown argument '$1'" >&2; usage 1
      fi
      shift ;;
  esac
done

# ── Download + extract (run-id mode) ─────────────────────────────────────────
if [[ "$MODE" == "run" ]]; then
  echo "Fetching metadata for run $RUN_ID..."
  CREATED=$(gh run view "$RUN_ID" --repo "$REPO" --json createdAt --jq '.createdAt')
  DATE=$(echo "$CREATED" | sed -E 's/T.*//')
  TIME=$(echo "$CREATED" | sed -E 's/.*T([0-9]{2}):([0-9]{2}):([0-9]{2})Z/\1_\2_\3/')
  COMMIT=$(gh run view "$RUN_ID" --repo "$REPO" --json headSha --jq '.headSha')

  echo "  date=$DATE time=$TIME commit=${COMMIT:0:12}"
  echo "Downloading artifacts..."
  WORK="/tmp/rscrypto-bench-artifacts-$$"
  rm -rf "$WORK"
  gh run download "$RUN_ID" --repo "$REPO" --dir "$WORK"

  RESULTS_BASE="$RESULTS_ROOT/$DATE"

  python3 - "$WORK" "$RESULTS_BASE" "$DATE" "$TIME" "$COMMIT" <<'EXTRACT_PY'
import os, re, sys

work, base, date, time_str, commit = sys.argv[1:6]
ansi_re = re.compile(r'\x1b\[[0-9;]*m')
running_re = re.compile(r'^Running: cargo bench .+--bench (\S+)')
noise_re = re.compile(r'^\s*(Updating|Downloading|Downloaded|Compiling|Compiled|Finished|Locking|Running (unittests|benches))\b|^warning\[')
SEP = '\u2501' * 74

def write_normalized_results(path, runner, body):
    with open(path, 'w') as out:
        out.write(f'date={date}\n')
        out.write(f'time={time_str}\n')
        out.write('mode=ci\n')
        out.write(f'platform={runner}\n')
        out.write(f'commit={commit}\n\n')
        body = body.strip()
        if body:
            out.write(body)
            if not body.endswith('\n'):
                out.write('\n')

for entry in sorted(os.listdir(work)):
    if not entry.startswith('benchmark-'):
        continue
    runner = entry.replace('benchmark-', '')
    dst_dir = os.path.join(base, 'linux', runner)
    os.makedirs(dst_dir, exist_ok=True)
    results_path = os.path.join(dst_dir, 'results.txt')

    # Prefer pre-built results.txt
    pre = os.path.join(work, entry, 'results.txt')
    if os.path.isfile(pre):
        with open(pre) as f:
            pre_lines = f.readlines()

        body_start = 0
        for i, line in enumerate(pre_lines):
            if not line.strip():
                body_start = i + 1
                break

        body = ''.join(pre_lines[body_start:])
        write_normalized_results(results_path, runner, body)
        print(f'  {runner}: normalized results.txt')
        continue

    # Fall back to parsing output.txt
    src = os.path.join(work, entry, 'output.txt')
    if not os.path.isfile(src):
        continue

    with open(src) as f:
        lines = f.readlines()

    sections = []
    for i, line in enumerate(lines):
        cleaned = ansi_re.sub('', line).strip()
        m = running_re.match(cleaned)
        if m:
            bench_name = m.group(1)
            filt = ''
            if ' -- ' in cleaned:
                after = cleaned.split(' -- ', 1)[1]
                for token in after.split():
                    if not token.startswith('--'):
                        filt = token
                        break
            sections.append((i, bench_name, filt))

    section_chunks = []
    for idx, (start_line, bench, filt) in enumerate(sections):
        end_line = sections[idx + 1][0] if idx + 1 < len(sections) else len(lines)
        content_lines = [ansi_re.sub('', l) for l in lines[start_line + 1 : end_line] if not noise_re.match(ansi_re.sub('', l))]
        content = ''.join(content_lines).strip()
        chunk = [SEP, f'bench={bench}']
        if filt:
            chunk.append(f'filter={filt}')
        chunk.append(SEP)
        chunk.append(content)
        section_chunks.append('\n'.join(chunk).strip())

    write_normalized_results(results_path, runner, '\n\n'.join(section_chunks))

    print(f'  {runner}: parsed {len(sections)} sections')

EXTRACT_PY
  rm -rf "$WORK"
fi

# ── Resolve results directory ────────────────────────────────────────────────
if [[ -z "$DATE" ]]; then
  DATE="$(ls -1 "$RESULTS_ROOT" 2>/dev/null | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' | sort -r | head -1)"
  if [[ -z "$DATE" ]]; then
    echo "error: no benchmark results found in $RESULTS_ROOT" >&2
    exit 1
  fi
fi

RESULTS_BASE="$RESULTS_ROOT/$DATE"
if [[ ! -d "$RESULTS_BASE" ]]; then
  echo "error: results directory not found: $RESULTS_BASE" >&2
  exit 1
fi

OUTPUT="${OUTPUT:-$RESULTS_ROOT/OVERVIEW.md}"
echo "Source: $RESULTS_BASE"

# ── Generate OVERVIEW.md ─────────────────────────────────────────────────────
python3 - "$RESULTS_BASE" "$OUTPUT" "$FILTER" "$RUN_ID" <<'OVERVIEW_PY'
import os, re, sys
from collections import defaultdict

results_base, output_path, filter_pattern, run_id = sys.argv[1], sys.argv[2], sys.argv[3] or None, sys.argv[4] or None

# ── Constants ────────────────────────────────────────────────────────────────

PLATFORM_ORDER = [
    'amd-zen4', 'amd-zen5', 'intel-spr', 'intel-icl',
    'graviton3', 'graviton4', 'ibm-s390x', 'ibm-power10', 'rise-riscv',
]
PLATFORM_SHORT = {
    'amd-zen4': 'Zen4', 'amd-zen5': 'Zen5', 'intel-spr': 'SPR', 'intel-icl': 'ICL',
    'graviton3': 'Grav3', 'graviton4': 'Grav4', 'ibm-s390x': 's390x',
    'ibm-power10': 'POWER10', 'rise-riscv': 'RISE',
}
PLATFORM_FULL = {
    'amd-zen4': ('AMD EPYC (Zen 4)', 'x86-64'), 'amd-zen5': ('AMD EPYC (Zen 5)', 'x86-64'),
    'intel-spr': ('Intel Xeon (Sapphire Rapids)', 'x86-64'), 'intel-icl': ('Intel Xeon (Ice Lake)', 'x86-64'),
    'graviton3': ('AWS Graviton 3', 'aarch64'), 'graviton4': ('AWS Graviton 4', 'aarch64'),
    'ibm-s390x': ('IBM Z s390x', 's390x'), 'ibm-power10': ('IBM POWER10', 'ppc64le'),
    'rise-riscv': ('RISE RISC-V riscv64', 'riscv64'),
}
CATEGORY_ORDER = ['Checksums', 'SHA-2', 'SHA-3', 'SHAKE', 'Ascon', 'Blake3', 'XXH3', 'RapidHash', 'Auth', 'AEAD']
TIE_LO, TIE_HI = 0.97, 1.03
SIZE_LABELS = {0:'0B',1:'1B',32:'32B',48:'48B',64:'64B',96:'96B',256:'256B',
               1024:'1KiB',4096:'4KiB',16384:'16KiB',65536:'64KiB',1048576:'1MiB'}

def fmt_size(n):
    return SIZE_LABELS.get(n, f'{n}B')

def to_ns(val, unit):
    m = {'ps': 1e-3, 'ns': 1, 'us': 1e3, '\u00b5s': 1e3, '\u03bcs': 1e3, 'ms': 1e6, 's': 1e9}
    return val * m[unit] if unit in m else None

def category_for(group):
    g = group.split('/')[0]
    if g.startswith('crc'): return 'Checksums'
    if g in ('sha224','sha256','sha384','sha512','sha512-256'): return 'SHA-2'
    if g.startswith('sha3-'): return 'SHA-3'
    if g.startswith('shake'): return 'SHAKE'
    if g.startswith('ascon-hash') or g.startswith('ascon-xof'): return 'Ascon'
    if g.startswith('blake3'): return 'Blake3'
    if g.startswith('xxh3'): return 'XXH3'
    if g.startswith('rapidhash'): return 'RapidHash'
    if any(g.startswith(p) for p in ('hmac-','hkdf-','pbkdf2-','ed25519','x25519','kmac')): return 'Auth'
    if any(g.startswith(p) for p in ('chacha20-poly1305','xchacha20-poly1305','aes-256-gcm','aegis-256','ascon-aead')): return 'AEAD'
    return 'Other'

# ── Parse results.txt files ──────────────────────────────────────────────────

bench_id_re = re.compile(r'^([A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)+)')
time_re = re.compile(r'time:\s+\[([^\]]+)\]')
ansi_re = re.compile(r'\x1b\[[0-9;]*m')

# data[group][size][platform][impl] = mid_ns
data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
meta = {}
platforms_found = []

for root, dirs, files in os.walk(results_base):
    if 'results.txt' not in files:
        continue
    path = os.path.join(root, 'results.txt')
    rel_parts = os.path.relpath(root, results_base).split(os.sep)  # e.g. ['linux', 'amd-zen4']
    platform_id = rel_parts[-1] if len(rel_parts) >= 2 else rel_parts[0]

    with open(path) as f:
        lines = f.readlines()

    # Read header
    if not meta:
        for line in lines:
            s = line.strip()
            if not s: break
            if '=' in s:
                k, v = s.split('=', 1)
                meta[k] = v

    if platform_id not in platforms_found:
        platforms_found.append(platform_id)

    current_id = None
    for line in lines:
        line = ansi_re.sub('', line)
        stripped = line.strip()
        m = bench_id_re.match(stripped)
        if m:
            current_id = m.group(1)
        tm = time_re.search(stripped)
        if tm and current_id:
            tokens = tm.group(1).split()
            if len(tokens) >= 4:
                try:
                    mid, unit = float(tokens[2]), tokens[3]
                    mid_ns = to_ns(mid, unit)
                    if mid_ns is not None:
                        parts = current_id.split('/')
                        if len(parts) >= 3 and parts[-1].isdigit():
                            size, impl_name = int(parts[-1]), parts[-2]
                            group = '/'.join(parts[:-2])
                            if filter_pattern and filter_pattern not in group:
                                current_id = None; continue
                            data[group][size][platform_id][impl_name] = mid_ns
                except (ValueError, IndexError):
                    pass
            current_id = None

platforms = [p for p in PLATFORM_ORDER if p in platforms_found]
# Add any non-CI platforms (e.g., local macOS) at the end
for p in platforms_found:
    if p not in platforms:
        platforms.append(p)

if not data:
    print('error: no benchmark data parsed', file=sys.stderr); sys.exit(1)

# ── Compute speedups + W/T/L ─────────────────────────────────────────────────

def primary_competitor(group_data):
    counts = defaultdict(int)
    for size_data in group_data.values():
        for plat_data in size_data.values():
            for impl in plat_data:
                if impl != 'rscrypto':
                    counts[impl] += 1
    return max(counts, key=lambda k: (counts[k], k)) if counts else None

speedups = defaultdict(lambda: defaultdict(dict))
wtl_group = defaultdict(lambda: [0, 0, 0])   # [W, T, L]
wtl_cat   = defaultdict(lambda: [0, 0, 0])
wtl_plat  = defaultdict(lambda: [0, 0, 0])
total_wtl = [0, 0, 0]

for group, sizes in data.items():
    primary = primary_competitor(sizes)
    if not primary: continue
    cat = category_for(group)

    for size, plats in sizes.items():
        for plat, impls in plats.items():
            ours = impls.get('rscrypto')
            if not ours or ours == 0: continue

            if primary in impls:
                speedups[group][size][plat] = impls[primary] / ours

            for impl, theirs in impls.items():
                if impl == 'rscrypto' or theirs == 0: continue
                ratio = theirs / ours
                idx = 0 if ratio >= TIE_HI else (2 if ratio < TIE_LO else 1)
                wtl_group[group][idx] += 1
                wtl_cat[cat][idx] += 1
                wtl_plat[plat][idx] += 1
                total_wtl[idx] += 1

# ── Generate OVERVIEW.md ─────────────────────────────────────────────────────

o = []
w = o.append

date_str = meta.get('date', os.path.basename(results_base))
commit = meta.get('commit', '?')
commit_short = commit[:12]

w('# Benchmark Overview\n')
if run_id:
    w(f'Source: CI run [#{run_id}](https://github.com/loadingalias/rscrypto/actions/runs/{run_id}) on {date_str} across {len(platforms)} platforms.')
else:
    w(f'Source: {meta.get("mode","local")} run on {date_str} across {len(platforms)} platforms.')
w(f'Commit: `{commit_short}`\n')
w('This file covers the size-indexed competitive groups from the full `comp` sweep.')
w('Fixed-cost benches without a size axis, such as `ed25519/public-key-from-secret`, `ed25519/keypair-from-secret`, and `x25519`, are intentionally excluded from this canonical report.\n')

# Platforms
w('## Platforms\n')
w('| Short | Full | Architecture |')
w('|-------|------|-------------|')
for p in platforms:
    short = PLATFORM_SHORT.get(p, p)
    full, arch = PLATFORM_FULL.get(p, (p, '?'))
    w(f'| {short} | {full} | {arch} |')
w('')

# How to Read
w('## How to Read\n')
w('- **Speedup** = competitor_time / rscrypto_time. Values above `1.00x` mean `rscrypto` is faster.')
w('- **WIN**: speedup >= `1.03x`. **TIE**: `0.97x`--`1.03x`. **LOSS**: < `0.97x`.')
w('- Group W/T/L totals count every competitor in the canonical size-indexed bench for that algorithm.')
w('')

# Scoreboard
total = sum(total_wtl)
overall_pct = total_wtl[0] * 100 // total if total else 0
w('## Overall Scoreboard\n')
w(f'**{total_wtl[0]}W / {total_wtl[1]}T / {total_wtl[2]}L = {overall_pct}% win rate** ({total} comparisons)\n')

# By Category
w('### By Category\n')
w('| Category | W | T | L | Total | Win % |')
w('|----------|--:|--:|--:|------:|------:|')
for cat in CATEGORY_ORDER:
    if cat not in wtl_cat: continue
    d = wtl_cat[cat]; t = sum(d)
    w(f'| {cat} | {d[0]} | {d[1]} | {d[2]} | {t} | {d[0]*100//t if t else 0}% |')
w('')

# By Platform
w('### By Platform\n')
w('| Platform | W | T | L | Total | Win % |')
w('|----------|--:|--:|--:|------:|------:|')
for p in platforms:
    if p not in wtl_plat: continue
    d = wtl_plat[p]; t = sum(d)
    w(f'| {PLATFORM_SHORT.get(p,p)} | {d[0]} | {d[1]} | {d[2]} | {t} | {d[0]*100//t if t else 0}% |')
w('')

# Per-group tables
def fmt_ratio(r):
    s = f'{r:.2f}x'
    return f'**{s}**' if r < TIE_LO else s

groups_by_cat = defaultdict(list)
for group in sorted(data.keys()):
    groups_by_cat[category_for(group)].append(group)

for cat in CATEGORY_ORDER:
    if cat not in groups_by_cat: continue
    w(f'\n---\n## {cat}\n')

    for group in groups_by_cat[cat]:
        if group not in wtl_group: continue
        d = wtl_group[group]; t = sum(d)
        group_pct = f'{d[0]*100.0/t:.1f}' if t else '0.0'
        w(f'### {group} ({d[0]}W/{d[1]}T/{d[2]}L = {group_pct}%)\n')

        hdrs = ' | '.join(f'{PLATFORM_SHORT.get(p,p)}' for p in platforms)
        w(f'| Size | {hdrs} |')
        aligns = ' | '.join('----:' for _ in platforms)
        w(f'|------| {aligns} |')

        for size in sorted(speedups[group].keys()):
            cells = []
            for p in platforms:
                r = speedups[group][size].get(p)
                cells.append(fmt_ratio(r) if r is not None else '-')
            w(f'| {fmt_size(size)} | {" | ".join(cells)} |')
        w('')

with open(output_path, 'w') as f:
    f.write('\n'.join(o) + '\n')

# Summary to stdout
print(f'\n  Overall: {total_wtl[0]}W / {total_wtl[1]}T / {total_wtl[2]}L = {overall_pct}% win rate ({total} comparisons)')
for cat in CATEGORY_ORDER:
    if cat not in wtl_cat: continue
    d = wtl_cat[cat]; t = sum(d)
    print(f'    {cat:12s}  {d[0]:4d}W {d[1]:3d}T {d[2]:3d}L  ({d[0]*100//t if t else 0}%)')
print(f'\n  Generated: {output_path}')
OVERVIEW_PY

echo "Done."
