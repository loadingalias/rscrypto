#!/usr/bin/env -S uv run python
"""Exercise the runner with isolated Cargo and Criterion process fixtures."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from execution import execute, hardware

ROOT = Path(__file__).resolve().parents[2]

CARGO = r'''
if sys.argv[1:] == ['-V']:
  print('cargo fixture'); sys.exit(0)
with open('builds.jsonl', 'a') as log:
  log.write(json.dumps(sys.argv[1:]) + '\n')
if hasattr(os, 'getsid'):
  # The watchdog starts each bounded invocation in its own session.
  with open('sessions.jsonl', 'a') as log: log.write(json.dumps(os.getsid(0)) + '\n')
assert '--no-run' in sys.argv and '--message-format=json' in sys.argv
if os.environ.get('FAIL_BUILD'):
  print('fixture compiler detail: rejected benchmark source', file=sys.stderr)
  print(json.dumps({'reason': 'compiler-message', 'message': {'rendered': 'fixture compiler JSON diagnostic'}}))
  sys.exit(8)
if os.environ.get('STALL_BUILD'):
  import time; time.sleep(30)
features = sys.argv[sys.argv.index('--features') + 1]
flags = os.environ.get('CARGO_ENCODED_RUSTFLAGS', '').split('\x1f')
if 'diag' in features.split(','): assert 'rscrypto_internal' in flags
with open('build-flags.jsonl', 'a') as log: log.write(json.dumps(flags) + '\n')
for index, value in enumerate(sys.argv):
  if value != '--bench': continue
  binary = sys.argv[index + 1]
  artifact = pathlib.Path('bin') / (binary + '--' + hashlib.sha256(features.encode()).hexdigest())
  artifact.write_bytes(pathlib.Path('bin/criterion-fixture').read_bytes())
  artifact.chmod(0o755)
  artifact.with_suffix('.features').write_text(features)
  print(json.dumps({'reason': 'compiler-artifact', 'target': {'name': binary, 'kind': ['bench']},
                   'features': features.split(','), 'executable': str(artifact.resolve())}))
'''

CRITERION = r'''
if sys.argv[1:2] == ['--rscrypto-filter-cases']:
  request = json.loads(pathlib.Path(sys.argv[2]).read_text())
  with open('filters.jsonl', 'a') as log: log.write(json.dumps(request) + '\n')
  print(json.dumps({pattern: [case for case in request['cases'] if re.search(pattern, case)] for pattern in request['patterns']}))
  sys.exit(0)
binary = pathlib.Path(sys.argv[0]).name.split('--')[0]
cases = {
  'sha2': ['sha256/rscrypto/64', 'sha256/other/64', 'sha512/rscrypto/64', 'sha512-256/rscrypto/64'],
  'auth': ['p256-ecdh/rscrypto/64'],
  'password_hashing': ['argon2id-small/salt16-raw32/rscrypto/m=8_t=1_p=1',
                       'argon2id-owasp/salt16-raw32/rscrypto/m=19MiB_t=2_p=1',
                       'argon2id-parallel/rscrypto/p=4'],
  'blake2': ['blake2/rscrypto/blake2b256/64', 'blake2/dryoc/blake2b256/64', 'blake2/keyed/dryoc/blake2b256/64'],
  'crc': [name + '/rscrypto/64' for name in ('crc16-ccitt', 'crc16-ibm', 'crc24-openpgp', 'crc32', 'crc32c', 'crc64-xz', 'crc64-nvme')],
  'slhdsa': ['slhdsa-sha2-128s/keygen/fixed-seeds/rscrypto',
             'slhdsa-sha2-128s/sign/deterministic-32/rscrypto', 'slhdsa-sha2-128s/sign/deterministic-32/rustcrypto',
             'slhdsa-sha2-128s/sign/deterministic-32/fips205', 'slhdsa-sha2-128s/verify/32/rscrypto',
             'slhdsa-sha2-128s/import/raw-secret/rscrypto', 'slhdsa-sha2-128s/hash-sign/sha256/deterministic-32/fips205',
             'slhdsa-sha2-128s/hash-verify/sha256/32/fips205',
             'slhdsa-shake-256f/verify/32/rustcrypto', *filter(None, [os.environ.get('EXTRA_CASE')])],
}.get(binary, [binary + '/rscrypto/64'])
if binary == 'blake3' and 'diag' in pathlib.Path(sys.argv[0]).with_suffix('.features').read_text().split(','):
  cases.append('blake3/rscrypto-scalar/64')
with open('executor.jsonl', 'a') as log:
  log.write(json.dumps({'binary': binary, 'args': sys.argv[1:], 'home': os.environ['CRITERION_HOME']}) + '\n')
if '--list' in sys.argv:
  if os.environ.get('FAIL_LIST'):
    print('fixture discovery detail: cannot enumerate cases', file=sys.stderr)
    sys.exit(6)
  pattern = sys.argv[sys.argv.index('--') + 1] if '--' in sys.argv else ''
  cases = [case for case in cases if re.search(pattern, case)]
  if os.environ.get('DUPLICATE_CASES'): cases *= 2
  for case in cases: print(case + ': benchmark')
  sys.exit(0)
plan = json.loads(pathlib.Path(os.environ['RSCRYPTO_BENCH_CASES']).read_text())
selected = plan['cases']
print(f"Benchmark harness budget: {plan['budget_seconds']}s", flush=True)
if os.environ.get('STALL_MEASURE'):
  import time; time.sleep(30)
assert set(selected) <= set(cases)
if '--profile-time' in sys.argv: sys.exit(0)
for case in selected:
  if os.environ.get('NO_MEASUREMENT'): sys.exit(0)
  if os.environ.get('SKIP_CASE') == case: continue
  directory = pathlib.Path(os.environ['CRITERION_HOME']) / case
  previous = directory / 'base/estimates.json'
  old = json.loads(previous.read_text())['mean']['point_estimate'] if previous.exists() else None
  print('baseline=' + str(old), flush=True)
  value = float(os.environ.get('ESTIMATE', '10'))
  estimate = {'point_estimate': value, 'standard_error': 0.1,
              'confidence_interval': {'confidence_level': 0.95, 'lower_bound': value - 0.1, 'upper_bound': value + 0.1}}
  count = 2 if '--quick' in sys.argv else int(sys.argv[sys.argv.index('--sample-size') + 1])
  data = {'benchmark.json': {'full_id': case}, 'sample.json': {'iters': [1] * count, 'times': [value] * count},
          'estimates.json': {'mean': estimate, 'median': estimate}}
  if os.environ.get('WRONG_CASE'): data['benchmark.json']['full_id'] = 'wrong'
  if os.environ.get('WRONG_SAMPLE_COUNT'):
    wrong = 3 if '--quick' in sys.argv else 2
    data['sample.json'] = {'iters': [1] * wrong, 'times': [value] * wrong}
  if os.environ.get('EMPTY_SAMPLES'): data['sample.json'] = {'iters': [], 'times': []}
  if os.environ.get('BAD_ESTIMATE'): data['estimates.json']['mean']['point_estimate'] = None
  for name in ('new', 'base'):
    target = directory / name; target.mkdir(parents=True, exist_ok=True)
    for filename, contents in data.items():
      if filename != os.environ.get('MISSING_FILE'):
        (target / filename).write_text(json.dumps(contents))
  if old is not None and not os.environ.get('NO_COMPARISON'):
    target = directory / 'change'; target.mkdir()
    (target / 'estimates.json').write_text(json.dumps({'mean': estimate, 'median': estimate}))
  print('benchmark output', flush=True)
  if os.environ.get('SIGNAL_PARENT'):
    import signal; os.kill(os.getppid(), signal.SIGTERM)

sys.exit(int(os.environ.get('MEASURE_STATUS', '0')))
'''


class HardwareTests(unittest.TestCase):
  def test_power_identity_keeps_model_and_revision_without_clock_noise(self):
    cpuinfo = (
      "processor\t: 0\ncpu\t\t: POWER10, altivec supported\n"
      "revision\t: 2.0 (pvr 0080 0200)\nclock\t\t: 3500.000000MHz\n"
      "platform\t: PowerNV\nmodel\t\t: IBM,9009-42G\n"
    )
    with patch('platform.system', return_value='Linux'), patch('platform.machine', return_value='ppc64le'), \
         patch('os.cpu_count', return_value=4), patch.object(Path, 'is_file', return_value=True), \
         patch.object(Path, 'read_text', return_value=cpuinfo) as read:
      identity = hardware()
      self.assertEqual(identity['cpu'], [
        'cpu\t\t: POWER10, altivec supported',
        'model\t\t: IBM,9009-42G',
        'platform\t: PowerNV',
        'processor\t: 0',
        'revision\t: 2.0 (pvr 0080 0200)',
      ])
      read.return_value = cpuinfo.replace('3500.000000', '3000.000000')
      self.assertEqual(hardware(), identity)
      for before, after in [('POWER10', 'POWER9'), ('2.0 (pvr 0080 0200)', '2.1 (pvr 0080 0201)')]:
        with self.subTest(change=after):
          read.return_value = cpuinfo.replace(before, after)
          self.assertNotEqual(hardware(), identity)


class RunnerTests(unittest.TestCase):
  def setUp(self):
    temporary = tempfile.TemporaryDirectory(prefix="rscrypto-runner-test-")
    self.addCleanup(temporary.cleanup)
    self.root = Path(temporary.name)
    shutil.copytree(ROOT / "scripts/bench", self.root / "scripts/bench", ignore=shutil.ignore_patterns("__pycache__"))
    (self.root / "scripts/ct").mkdir()
    for name in ("internal.py", "provenance.py"):
      shutil.copy2(ROOT / "scripts/ct" / name, self.root / "scripts/ct" / name)
    (self.root / "scripts/lib").mkdir()
    for name in ("python.sh", "toolchain.py"):
      shutil.copy2(ROOT / "scripts/lib" / name, self.root / "scripts/lib" / name)
    (self.root / ".config").mkdir()
    for name in ("criterion.json", "benchmark-matrix.json"):
      shutil.copy2(ROOT / ".config" / name, self.root / ".config" / name)
    for name in ("Cargo.toml", "justfile", "rust-toolchain.toml"):
      shutil.copy2(ROOT / name, self.root / name)
    (self.root / "benches").symlink_to(ROOT / "benches", target_is_directory=True)
    (self.root / 'tools/provider-bench').mkdir(parents=True)
    shutil.copy2(ROOT / 'tools/provider-bench/Cargo.toml', self.root / 'tools/provider-bench/Cargo.toml')
    shutil.copytree(ROOT / 'tools/provider-bench/benches', self.root / 'tools/provider-bench/benches')
    (self.root / "bin").mkdir()
    self.env = {key: value for key, value in os.environ.items()
                if not key.startswith(("BENCH_", "CARGO_", "RUST", "DEV_MACHINE_", "RSCRYPTO_", "GIT_"))
                and key not in {"BASH_ENV", "ENV", "GITHUB_STEP_SUMMARY"}}
    self.env.update(PATH=f"{self.root / 'bin'}:{os.environ['PATH']}", PYTHON=sys.executable,
                    GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    self.tool("cargo", "with open('toolchains.jsonl', 'a') as log: log.write(json.dumps(os.environ.get('RUSTUP_TOOLCHAIN')) + '\\n')\n" + CARGO)
    self.tool("criterion-fixture", CRITERION)
    self.tool("rustc", "print(os.environ.get('COMPILER', 'rustc fixture')); print('host: ' + os.environ.get('CHECK_HOST', 'x86_64-unknown-linux-gnu'))")
    self.tool("samply", """
if '--version' in sys.argv: print('samply fixture'); sys.exit(0)
if os.environ.get('FAIL_CAPTURE'): sys.exit(7)
output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])
subprocess.run(sys.argv[sys.argv.index('--output') + 2:], check=True)
output.write_text('profile fixture')
""")
    self.tool("perf", """
if sys.argv[1:] == ['--version']: print('perf fixture'); sys.exit(0)
mode = sys.argv[1]
args = sys.argv[2:]
command = args[args.index('--') + 1:] if '--' in args else []
probe = any('perf-probe-' in arg for arg in args)
if mode == 'record' and os.environ.get('REQUIRE_PRIVILEGED_PERF') and not os.environ.get('PERF_VIA_SUDO'):
  sys.exit(5)
if mode == 'record':
  if probe and os.environ.get('FAIL_PERF_PROBE'):
    print('Access to performance monitoring is limited. perf_event_paranoid setting is 4', file=sys.stderr)
    sys.exit(5)
  if probe and '--call-graph' in args and os.environ.get('FAIL_PERF_DWARF'): sys.exit(5)
  if not probe and os.environ.get('FAIL_PERF_CAPTURE'): sys.exit(7)
  status = subprocess.run(command).returncode
  if status == 0 and not (probe and os.environ.get('ZERO_PERF_PROBE_SAMPLES')):
    pathlib.Path(args[args.index('--output') + 1]).write_text(
      'dwarf' if '--call-graph' in args else 'flat')
  sys.exit(status)
if mode == 'script':
  data = pathlib.Path(args[args.index('--input') + 1]).read_text()
  if data == 'dwarf' and os.environ.get('POWER_DWARF_LOST'):
    print('Processed 5 samples and lost 100.00%!', file=sys.stderr)
    sys.exit(0)
  print('criterion-fixture 1234 cycles:u: rscrypto::production_frame')
  sys.exit(0)
if mode == 'report':
  if os.environ.get('FAIL_PERF_REPORT'): sys.exit(6)
  if '--no-inline' not in args:
    sys.stderr.buffer.write(b'addr2line configuration failed: \\xff\\n')
    sys.exit(255)
  if os.environ.get('HEADER_ONLY_PERF_REPORT'):
    print('# To display the perf.data header info, please use --header/--header-only options.')
    print('The perf.data data has no samples!', file=sys.stderr)
    sys.exit(0)
  if os.environ.get('ZERO_PERF_SAMPLES'): print('# Samples: 0 of event cycles:u'); sys.exit(0)
  print('# Samples: 5 of event cycles:u')
  if os.environ.get('RISCV_MAPPING_SYMBOLS'): print(' 99.00% auth auth [.] $xrv64i2p1_m2p0'); sys.exit(0)
  print(' 99.00% criterion-fixture criterion-fixture [.] rscrypto::production_frame')
  sys.exit(0)
sys.exit(64)
""")
    for command in (["git", "init", "-q"], ["git", "add", "scripts", ".config", "Cargo.toml"],
                    ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"]):
      subprocess.run(command, cwd=self.root, env=self.env, check=True)

  def test_native_toolchains(self):
    catalog = tomllib.loads((ROOT / '.config/tooling.toml').read_text())
    hosts = {row['rust-host'] for row in catalog.values() if isinstance(row, dict) and 'rust-host' in row}
    hosts.add('aarch64-apple-darwin')
    channel = tomllib.loads((ROOT / 'rust-toolchain.toml').read_text())['toolchain']['channel']
    for host in sorted(hosts):
      with self.subTest(host=host):
        log = self.root / 'toolchains.jsonl'
        log.write_text('')
        result = self.bench('sha256', '--list', CHECK_HOST=host, RUSTUP_TOOLCHAIN='wrong-ambient-channel')
        self.assertEqual(result.returncode, 0, result.stderr)
        channels = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertTrue(channels)
        self.assertEqual(set(channels), {channel})

  def tool(self, name, body):
    path = self.root / "bin" / name
    path.write_text(f"#!{sys.executable}\nimport hashlib, json, os, pathlib, re, subprocess, sys\n" + body)
    path.chmod(0o755)

  def invoke(self, *args, **env):
    command = (["--bench", *args[1:]] if args[0] == "bench" else
               [sys.executable, "scripts/bench/runner.py", *args])
    return subprocess.run([sys.executable, "scripts/bench/bounded.py", *command],
                          cwd=self.root, env=self.env | env, text=True, capture_output=True, timeout=30)

  def bench(self, *args, **env):
    return self.invoke("bench", *args, **env)

  def runs(self):
    return sorted((self.root / "benchmark_results/criterion").glob("*"))

  def calls(self, listing=False):
    path = self.root / "executor.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [row for row in rows if ("--list" in row["args"]) == listing]

  def plan(self, root=None):
    return json.loads(((root or self.runs()[-1]) / "plan.json").read_text())

  def ok(self, result):
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_grouped_deduplicated_comparison(self):
    self.ok(self.bench("sha256"))
    previous = self.runs()[0]
    before = {str(path): path.read_bytes() for path in previous.rglob('*') if path.is_file()}
    self.ok(self.bench("sha256", "filter=sha256", "filter=sha256/rscrypto", f"baseline={previous}", ESTIMATE="20"))
    current = next(path for path in self.runs() if path != previous)
    self.assertEqual(len(self.calls()), 2)  # One process for both cases in each run.
    self.assertEqual(self.plan(current)[0]['baselines'], ['sha256/rscrypto/64', 'sha256/other/64'])
    self.assertEqual((current / 'output.txt').read_text().count('baseline=10.0'), 2)
    self.assertEqual(before, {str(path): path.read_bytes() for path in previous.rglob('*') if path.is_file()})

  def test_step_summary_reports_checked_rows(self):
    summary = self.root / 'summary.md'
    self.ok(self.bench('sha256', GITHUB_STEP_SUMMARY=str(summary)))
    self.assertEqual(summary.read_text(), '- Output checks: sha2, 2 of 2 measured rows passed\n')

  def test_configuration_and_runtime_compatibility(self):
    self.ok(self.bench('sha256'))
    previous = self.runs()[0]
    for env in ({'RAYON_NUM_THREADS': '2'}, {'RSCRYPTO_BLAKE3_BENCH_ISA': 'avx2'},
                {'COMPILER': 'different compiler'}, {'RUSTFLAGS': '-C opt-level=1'}):
      with self.subTest(env=env):
        result = self.bench('sha256', f'baseline={previous}', **env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('no matching', result.stderr)
    (self.root / '.cargo').mkdir()
    config = self.root / '.cargo/config.toml'
    config.write_text('[build]\nrustflags = ["-C", "opt-level=1"]\n')
    self.assertNotEqual(self.bench('sha256', f'baseline={previous}').returncode, 0)
    config.unlink()
    manifest = self.root / 'Cargo.toml'; original = manifest.read_text()
    manifest.write_text(original.replace('[profile.bench]', '[profile.bench]\nopt-level = 1'))
    self.assertNotEqual(self.bench('sha256', f'baseline={previous}').returncode, 0)
    manifest.write_text(original)
    config = self.root / '.config/criterion.json'; data = json.loads(config.read_text())
    data['max_run_seconds'] = 599; config.write_text(json.dumps(data))
    self.ok(self.bench('sha256', f'baseline={previous}'))

  def test_external_b3sum_change_cannot_reuse_baseline(self):
    binary = self.root / 'b3sum'
    receipt = self.root / 'b3sum.json'
    binary.write_bytes(b'first executable')
    receipt.write_text('{"source": "first"}\n')
    env = {'RSCRYPTO_BLAKE3_B3SUM': str(binary), 'RSCRYPTO_BLAKE3_B3SUM_RECEIPT': str(receipt)}
    self.ok(self.bench('sha256', **env))
    previous = self.runs()[0]
    for path in (binary, receipt):
      original = path.read_bytes()
      path.write_bytes(original + b'changed')
      result = self.bench('sha256', f'baseline={previous}', **env)
      self.assertNotEqual(result.returncode, 0)
      self.assertIn('no matching', result.stderr)
      path.write_bytes(original)

  def test_split_target_cases_each_belong_to_one_algorithm(self):
    # Explicit run IDs avoid relying on the ordering of random suffixes within one timestamp second.
    runs = self.root / 'benchmark_results/criterion'
    self.ok(self.bench('slhdsa-sha2-128s', RSCRYPTO_BENCH_RUN_ID='pure'))
    self.assertEqual(self.plan(runs / 'pure')[0]['cases'], [
      'slhdsa-sha2-128s/keygen/fixed-seeds/rscrypto',
      'slhdsa-sha2-128s/sign/deterministic-32/rscrypto', 'slhdsa-sha2-128s/sign/deterministic-32/rustcrypto',
      'slhdsa-sha2-128s/sign/deterministic-32/fips205', 'slhdsa-sha2-128s/verify/32/rscrypto',
      'slhdsa-sha2-128s/import/raw-secret/rscrypto'])
    self.ok(self.bench('hash-slhdsa-sha2-128s', RSCRYPTO_BENCH_RUN_ID='hash'))
    self.assertEqual(self.plan(runs / 'hash')[0]['cases'], ['slhdsa-sha2-128s/hash-sign/sha256/deterministic-32/fips205',
                                                         'slhdsa-sha2-128s/hash-verify/sha256/32/fips205'])
    self.ok(self.bench('slhdsa-sha2-128s-sign', RSCRYPTO_BENCH_RUN_ID='sign'))
    self.assertEqual(self.plan(runs / 'sign')[0]['cases'], [f'slhdsa-sha2-128s/sign/deterministic-32/{implementation}'
                                                         for implementation in ('rscrypto', 'rustcrypto', 'fips205')])
    # CI runs each SLH-DSA algorithm on its own, so a case that no algorithm claims would never run there.
    extra = 'slhdsa-sha2-128s/sign-randomized/32/rscrypto'
    for selection in ('slhdsa-sha2-128s', 'slhdsa'):
      with self.subTest(selection=selection):
        result = self.bench(selection, '--list', EXTRA_CASE=extra)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f'exactly one algorithm: {extra} (no algorithm)', result.stderr)
    self.ok(self.bench('sha256', '--list', EXTRA_CASE=extra))
    # Overlapping filters would run a case twice.
    import runner
    with self.assertRaisesRegex(ValueError, 'case/x \\(first, second\\)'):
      runner.require_partition(['case/x'], {'first': 'x', 'second': '/x'}, {'x': ['case/x'], '/x': ['case/x']})

  def test_literal_filters_and_zero_matches(self):
    for pattern in (r'^sha256/rscrypto/\d+$', r'^sha256/rscrypto/[0-9]{1,3}$'):
      self.ok(self.bench('sha256', 'filter=' + pattern))
      self.assertEqual(json.loads((self.root / 'filters.jsonl').read_text().splitlines()[-1])['patterns'][-1], pattern)
    result = self.bench('sha256', 'filter=sha256', 'filter=missing')
    self.assertNotEqual(result.returncode, 0)
    self.assertEqual(len(self.calls()), 2)
    result = self.bench('sha256', 'filter=$(touch injected)')
    self.assertNotEqual(result.returncode, 0)
    self.assertFalse((self.root / 'injected').exists())
    self.assertNotEqual(self.bench('sha256', '--filter', 'parameter=value').returncode, 0)
    self.assertEqual(json.loads((self.root / 'filters.jsonl').read_text().splitlines()[-1])['patterns'][-1], 'parameter=value')

  def test_explicit_filters_narrow_algorithm_scope(self):
    result = self.bench('sha256', 'filter=rscrypto', '--list')
    self.ok(result)
    listed = [line.split()[-1] for line in result.stdout.splitlines() if line.startswith('[')]
    self.assertEqual(listed, ['sha256/rscrypto/64'])
    self.ok(self.bench('sha256', 'filter=rscrypto'))
    self.assertEqual(self.plan()[0]['cases'], ['sha256/rscrypto/64'])
    self.ok(self.bench('sha256', BENCH_FILTER='rscrypto'))
    self.assertEqual(self.plan()[0]['cases'], ['sha256/rscrypto/64'])
    result = self.bench('sha256', 'filter=sha512', '--list')
    self.assertNotEqual(result.returncode, 0)
    self.assertIn('no cases matched', result.stderr)
    result = self.bench('sha2', 'filter=rscrypto', '--list')
    self.ok(result)
    listed = [line.split()[-1] for line in result.stdout.splitlines() if line.startswith('[')]
    self.assertEqual(listed, ['sha256/rscrypto/64', 'sha512/rscrypto/64', 'sha512-256/rscrypto/64'])
    self.ok(self.bench('sha256', 'sha512', 'filter=rscrypto', 'filter=other'))
    self.assertEqual(set(self.plan()[0]['cases']), {'sha256/rscrypto/64', 'sha256/other/64', 'sha512/rscrypto/64'})

  def test_failed_command_replays_only_its_output(self):
    log = self.root / 'captured.log'
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as diagnostic:
      self.assertEqual(execute([sys.executable, '-c', "print('earlier successful output')"], log, capture=True),
                       'earlier successful output\n')
      with self.assertRaises(subprocess.CalledProcessError) as failure:
        execute([sys.executable, '-c',
                 "import sys; print('failed stdout'); "
                 "sys.stderr.buffer.write(b'failed stderr: \\xff\\n'); sys.exit(9)"],
                log, capture=True)
    self.assertEqual(failure.exception.returncode, 9)
    self.assertNotIn('earlier successful output', diagnostic.getvalue())
    self.assertIn('failed stdout\n', diagnostic.getvalue())
    self.assertIn('failed stderr: \ufffd\n', diagnostic.getvalue())
    self.assertIn('earlier successful output', log.read_text())
    self.assertIn('failed stderr: \ufffd', log.read_text())

  def test_discovery_surfaces_failed_command_diagnostics(self):
    for env, code, diagnostic in (
      ({'FAIL_BUILD': '1'}, 8, 'fixture compiler detail: rejected benchmark source'),
      ({'FAIL_LIST': '1'}, 6, 'fixture discovery detail: cannot enumerate cases'),
    ):
      with self.subTest(env=env):
        result = self.bench('sha256', '--list', **env)
        self.assertEqual(result.returncode, code)
        self.assertIn(diagnostic, result.stderr)
        if 'FAIL_BUILD' in env:
          self.assertIn('fixture compiler JSON diagnostic', result.stderr)
    self.assertFalse(self.runs())

  def test_codegen_uses_target_configuration_and_literal_arguments(self):
    self.tool('cargo', "pathlib.Path('inspection.json').write_text(json.dumps(sys.argv[1:]))")
    for mode in ('codegen', 'llvm-lines'):
      self.ok(self.invoke(mode, 'blake3', '--diag', '--', '--filter', 'symbol with spaces'))
      command = json.loads((self.root / 'inspection.json').read_text())
      self.assertEqual(command[-2:], ['--filter', 'symbol with spaces'])
      self.assertIn('--no-default-features', command)
      self.assertEqual(command[command.index('--profile') + 1], 'bench')
      self.assertEqual(command[command.index('--features') + 1], 'blake3,diag,parallel,std')

  def test_completion_requires_all_measurement_evidence(self):
    for env in ({'NO_MEASUREMENT': '1'}, {'MISSING_FILE': 'sample.json'}, {'EMPTY_SAMPLES': '1'},
                {'WRONG_SAMPLE_COUNT': '1'}, {'WRONG_CASE': '1'}, {'BAD_ESTIMATE': '1'}, {'SKIP_CASE': 'sha256/other/64'}):
      with self.subTest(env=env):
        result = self.bench('sha256', **env)
        self.assertNotEqual(result.returncode, 0)
    self.assertTrue(all('state=failed' in (root / 'status.txt').read_text() for root in self.runs()))

  def test_baseline_is_not_measurement_evidence(self):
    self.ok(self.bench('sha256'))
    previous = self.runs()[0]
    for env in ({'NO_MEASUREMENT': '1'}, {'NO_COMPARISON': '1'}):
      self.assertNotEqual(self.bench('sha256', f'baseline={previous}', **env).returncode, 0)

  def test_discovery_preserves_evidence_and_build_scope(self):
    self.ok(self.bench('--diag', 'blake3', '--list'))
    result = self.bench('crc64-nvme', '--list')
    self.ok(result)
    self.assertIn('crc64-nvme/rscrypto/64', result.stdout)
    self.assertNotIn('crc64-xz/rscrypto/64', result.stdout)
    self.assertFalse(self.runs())
    self.assertFalse(self.calls())
    self.ok(self.bench('crc', 'sha2'))
    self.assertEqual(len(self.calls()), 2)
    plan = self.plan()
    self.assertEqual(len(plan), 2)
    self.assertNotEqual(plan[0]['home'], plan[1]['home'])
    self.assertTrue(all('--no-default-features' in entry['artifact']['command'] for entry in plan))

  def test_discovery_runs_once_per_configuration(self):
    self.ok(self.bench('sha256', 'sha512', 'filter=rscrypto', 'filter=other', '--list'))
    self.assertEqual(len(self.calls(listing=True)), 1)
    self.assertEqual(len((self.root / 'filters.jsonl').read_text().splitlines()), 1)
    self.assertNotIn('--', self.calls(listing=True)[0]['args'])
    self.ok(self.bench('sha2', 'crc', 'filter=rscrypto', 'filter=64', '--list'))
    self.assertEqual(len(self.calls(listing=True)), 3)

  @unittest.skipUnless(os.name == 'posix', 'process sessions identify bounded invocations')
  def test_listing_bounds_each_target_in_its_own_invocation(self):
    # A cold listing builds every selected target; one shared budget could not cover all of them.
    result = self.bench('sha2', 'crc', 'filter=rscrypto', '--list')
    self.assertEqual(result.returncode, 0, result.stderr)
    listed = [line.split()[1:] for line in result.stdout.splitlines() if line.startswith('[')]
    self.assertEqual(list(dict.fromkeys(binary for binary, _ in listed)), ['sha2', 'crc'])
    self.assertTrue(all('rscrypto' in case for _, case in listed))
    sessions = [json.loads(line) for line in (self.root / 'sessions.jsonl').read_text().splitlines()]
    self.assertEqual(len(sessions), 2)
    self.assertEqual(len(set(sessions)), 2)
    self.assertNotIn(os.getsid(0), sessions)
    result = self.bench('sha2', '--list-target', 'sha2')
    self.assertNotEqual(result.returncode, 0)
    self.assertIn('--list-target applies only to --list', result.stderr)

  def test_explicit_options_override_environment(self):
    self.ok(self.bench('sha256', 'sample_size=12', 'measure_ms=250', BENCH_SAMPLE_SIZE='10', BENCH_MEASURE_MS='333'))
    self.assertEqual(self.plan()[0]['settings']['sample_size'], 12)
    self.assertEqual(self.plan()[0]['settings']['measure_ms'], 250)
    self.assertNotIn('max_run_seconds', self.plan()[0]['settings'])
    self.assertNotIn('max_comparison_seconds', self.plan()[0]['settings'])

  def test_quick_mode_has_distinct_evidence_and_no_performance_baseline(self):
    self.ok(self.bench('sha256', 'quick=false', BENCH_QUICK='true'))
    regular = self.runs()[0]
    original = self.plan()[0]
    self.assertNotIn('quick', original['settings'])
    self.ok(self.bench('--quick', 'sha256', 'measure_ms=10'))
    quick = self.plan()[0]
    self.assertTrue(quick['settings']['quick'])
    self.assertNotIn('sample_size', quick['settings'])
    self.assertNotIn('warmup_ms', quick['settings'])
    self.assertIn('--quick', quick['command'])
    self.assertNotIn('--sample-size', quick['command'])
    self.assertNotEqual(quick['id'], original['id'])
    result = self.bench('sha256', 'quick=true', f'baseline={regular}')
    self.assertIn('quick mode does not compare performance baselines', result.stderr)
    result = self.bench('sha256', '--quick', WRONG_SAMPLE_COUNT='1')
    self.assertNotEqual(result.returncode, 0)
    self.assertIn('observed sample count differs', result.stderr)

  def test_diagnostic_boolean_options(self):
    for option, enabled in ((['--diag'], True), (['--diag', 'false'], False), (['--diag', 'TRUE'], True)):
      for arguments in ((*option, 'blake3'), ('blake3', *option)):
        with self.subTest(arguments=arguments):
          result = self.bench(*arguments, '--list', BENCH_DIAG=str(not enabled).lower())
          self.ok(result)
          self.assertEqual('blake3/rscrypto-scalar/64' in result.stdout, enabled)

  def test_invalid_requests_do_not_build(self):
    # Target names are plain selectors; the old `bench=` and BENCH_BENCH forms must fail.
    for args in (('unknown-selector',), ('bench=sha2',), ('--bench', 'sha2'), ('sha256', 'filter='), ('sample_size=9',),
                 ('diag=typo',), ('quick=typo',), ('sha256', '--quick', 'sample_size=10'),
                 ('sha256', 'quick=true', 'warmup_ms=10'), ('structural',), ('tls-provider', '--diag')):
      self.assertNotEqual(self.bench(*args).returncode, 0)
    for env in ({'BENCH_UNKNOWN': '1'}, {'BENCH_BENCH': 'sha2'}):
      self.assertNotEqual(self.bench('sha256', **env).returncode, 0)
    self.assertFalse((self.root / 'builds.jsonl').exists())
    self.assertFalse(self.runs())

  def test_failures_keep_status_and_partial_logs(self):
    for env, code in (({'FAIL_BUILD': '1'}, 8), ({'FAIL_LIST': '1'}, 6), ({'DUPLICATE_CASES': '1'}, 1), ({'MEASURE_STATUS': '7'}, 7)):
      self.assertEqual(self.bench('sha256', **env).returncode, code)
    self.assertTrue(all('state=failed' in (root / 'status.txt').read_text() for root in self.runs()))

  def test_excessive_budget_does_not_build(self):
    path = self.root / '.config/criterion.json'
    config = json.loads(path.read_text())
    config['max_run_seconds'] = 601
    path.write_text(json.dumps(config))
    result = self.bench('sha256')
    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
    self.assertIn('cannot exceed 10 minutes', result.stderr)
    self.assertFalse((self.root / 'builds.jsonl').exists())
    self.assertFalse(self.runs())

  def test_slow_comparison_carries_budget_without_changing_samples_or_peers(self):
    self.ok(self.bench('slhdsa-sha2-128s-sign', RSCRYPTO_BENCH_RUN_ID='slow'))
    root = self.runs()[0]
    entry, = self.plan(root)
    cases = [f'slhdsa-sha2-128s/sign/deterministic-32/{peer}' for peer in ('rscrypto', 'rustcrypto', 'fips205')]
    self.assertEqual(set(entry['cases']), set(cases))
    self.assertEqual(json.loads((root / 'requests.json').read_text())['budget_seconds'], 1800)
    self.assertEqual(json.loads((root / (entry['id'] + '.cases.json')).read_text()),
                     {'cases': entry['cases'], 'budget_seconds': 1800})
    self.assertIn('Benchmark harness budget: 1800s', (root / 'output.txt').read_text())
    self.assertEqual(entry['settings']['sample_size'], 20)
    self.assertFalse({'max_run_seconds', 'max_comparison_seconds'} & entry['settings'].keys())
    samples = list((root / entry['home']).glob('**/new/sample.json'))
    self.assertEqual(len(samples), 3)
    self.assertTrue(all(len(json.loads(path.read_text())['iters']) == 20 for path in samples))

  def test_run_id_and_export_are_immutable(self):
    self.ok(self.bench('sha256', RSCRYPTO_BENCH_RUN_ID='fixed'))
    root = self.runs()[0]
    self.assertFalse((root.parent.parent / '.transfers').exists())
    self.ok(self.invoke('export', str(root)))
    archive = root.parent.parent / '.transfers/fixed.tar'
    self.assertEqual(archive.with_suffix('.tar.sha256').read_text(), f'{hashlib.sha256(archive.read_bytes()).hexdigest()}  fixed.tar\n')
    self.assertNotEqual(self.invoke('export', str(root)).returncode, 0)
    self.assertNotEqual(self.bench('sha256', RSCRYPTO_BENCH_RUN_ID='fixed').returncode, 0)
    shutil.rmtree(root)
    self.assertNotEqual(self.bench('sha256', RSCRYPTO_BENCH_RUN_ID='fixed').returncode, 0)

  def test_timeout_keeps_failed_run(self):
    path = self.root / '.config/criterion.json'; config = json.loads(path.read_text()); config['max_run_seconds'] = 3
    path.write_text(json.dumps(config))
    # A stalled build times out in its own bounded build step, before any run directory exists.
    result = self.bench('sha256', STALL_BUILD='1')
    self.assertEqual(result.returncode, 124, result.stdout + result.stderr)
    self.assertFalse(self.runs())
    # A stalled measurement keeps its failed run.
    result = self.bench('sha256', STALL_MEASURE='1')
    self.assertEqual(result.returncode, 124, result.stdout + result.stderr)
    self.assertTrue(self.runs())
    self.assertNotIn('state=complete', (self.runs()[0] / 'status.txt').read_text())

  def test_catalog_rustflags_build_a_separate_configuration(self):
    import runner
    from benchmark_catalog import load_catalog
    catalog = load_catalog()
    rows = [runner.target(catalog, name, False) | {'bench': name, 'scope': '', 'pattern': ''} for name in ('xxh3', 'xxh3_v4')]
    with patch.dict(os.environ, self.env, clear=True), contextlib.chdir(self.root), \
         contextlib.redirect_stdout(io.StringIO()):
      plan = runner.resolve(rows, {'sample_size': 10}, self.root / 'output.txt',
                            self.env | {'CRITERION_HOME': str(self.root / 'discovery')})
    self.assertEqual([entry['bench'] for entry in plan], ['xxh3', 'xxh3_v4'])
    self.assertNotEqual(plan[0]['home'], plan[1]['home'])
    self.assertNotIn('rustflags', plan[0])
    self.assertEqual(plan[1]['rustflags'], ['-C', 'target-cpu=x86-64-v4'])
    self.assertEqual(plan[1]['artifact']['rustflags'], ['-C', 'target-cpu=x86-64-v4'])
    # A configuration without catalog flags keeps the identity that earlier baselines recorded.
    command = plan[0]['artifact']['command']
    self.assertEqual(plan[0]['id'], runner.identity([command, plan[0]['compatibility'], {'sample_size': 10}]))
    flags = [json.loads(line) for line in (self.root / 'build-flags.jsonl').read_text().splitlines()]
    self.assertEqual(flags, [[''], ['-C', 'target-cpu=x86-64-v4']])

  def test_catalog_rustflags_extend_inherited_flags(self):
    from execution import flagged
    extra = ['-C', 'target-cpu=x86-64-v4']
    self.assertEqual(flagged({'RUSTFLAGS': '-C  opt-level=3'}, extra), {'CARGO_ENCODED_RUSTFLAGS': '-C\x1fopt-level=3\x1f-C\x1ftarget-cpu=x86-64-v4'})
    # Cargo ignores RUSTFLAGS when the encoded form is set, so only the encoded flags carry forward.
    self.assertEqual(flagged({'RUSTFLAGS': '-C opt-level=1', 'CARGO_ENCODED_RUSTFLAGS': '--cfg\x1frscrypto_internal'}, extra),
                     {'CARGO_ENCODED_RUSTFLAGS': '--cfg\x1frscrypto_internal\x1f-C\x1ftarget-cpu=x86-64-v4'})

  def test_architecture_limited_configurations(self):
    import runner
    from benchmark_catalog import load_catalog
    catalog = load_catalog()

    def benches(*selectors, target, every=False):
      args = runner.parse(['bench', *selectors, '--target', target])
      return [row['bench'] for row in runner.requests(args, catalog, every_architecture=every)]

    self.assertIn('xxh3_v4', benches('hashes', target='x86_64-unknown-linux-gnu'))
    self.assertNotIn('xxh3_v4', benches('hashes', target='riscv64gc-unknown-linux-gnu'))
    self.assertIn('xxh3_v4', benches('hashes', target='riscv64gc-unknown-linux-gnu', every=True))
    with self.assertRaisesRegex(ValueError, 'xxh3_v4 does not run on powerpc64'):
      benches('xxh3_v4', target='powerpc64le-unknown-linux-gnu')
    for mode in ('profile', 'codegen'):
      with self.subTest(mode=mode):
        result = self.invoke(*((mode, 'xxh3_v4', '--list') if mode == 'profile' else (mode, 'xxh3_v4')))
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(result.stderr, 'does not run on|catalog rustflags')

  def test_just_preserves_arguments(self):
    result = subprocess.run(['just', 'bench', 'sha256', r'filter=^sha256/rscrypto/\d+$'], cwd=self.root, env=self.env,
                            capture_output=True, text=True, timeout=30)
    self.ok(result)
    self.assertEqual(self.plan()[0]['cases'], ['sha256/rscrypto/64'])


if __name__ == '__main__':
  unittest.main()
