#!/usr/bin/env python3
"""Exercise smoke budgets and overrides through the real shell and executor."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def main():
  with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    for name in ('scripts/ct/dudect.sh', 'scripts/ct/dudect_execute.py', 'scripts/ct/manifest.py',
                 'scripts/ct/provenance.py', 'scripts/lib/python.sh'):
      path = root / name
      path.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(ROOT / name, path)
    binary = root / 'bin'
    binary.mkdir()
    def tool(path, body):
      path.write_text(f'#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n' + body)
      path.chmod(0o755)
    tool(root / 'scripts/lib/toolchain.sh', "print('fixture-toolchain')")
    tool(binary / 'rustc', "print('host: fixture-host')")
    tool(binary / 'llvm', "print('fixture symbols')")
    tool(binary / 'cargo', '''
args = sys.argv
build = Path(args[args.index('--target-dir') + 1]) / args[args.index('--target') + 1] / 'release'
build.mkdir(parents=True, exist_ok=True)
path = build / 'rscrypto-ct-dudect'
path.write_text('unused fixture')
print('linker "-o" fixture')
''')
    # Reporting and the timing binary are substitutes; selection, budgets, shell
    # precedence, isolation, and summary publication execute production code.
    (root / 'scripts/ct/dudect_report.py').write_text('''
import json, sys
from pathlib import Path

def write_report(path, report):
  path.write_text(json.dumps(report))

def case_report(prepared, args):
  with Path('budgets.jsonl').open('a') as log:
    log.write(json.dumps([args.filter, args.samples]) + '\\n')
  return {'cases': [{'name': args.filter, 'requested_samples': args.samples}],
          'case_count': 1, 'failure_count': 0, 'diagnostic_failure_count': 0}

if __name__ == '__main__':
  path = Path(sys.argv[sys.argv.index('--out') + 1])
  path.write_text(json.dumps({'metadata': {'binary': {'path': sys.executable}}, 'manifest_cases': {
    'cheap': {'smoke_samples': 2000}, 'expensive': {'smoke_samples': 16}}}))
'''.replace("'path': sys.executable", "'path': str(Path('bin/measurement').resolve())"))
    tool(binary / 'measurement', "print('fixture timing output')")
    env = {key: value for key, value in os.environ.items()
           if key not in ('BASH_ENV', 'ENV') and not key.startswith('RSCRYPTO_CT_DUDECT_')}
    env.update(PATH=str(binary) + os.pathsep + os.environ['PATH'], PYTHON=sys.executable,
               LLVM_OBJDUMP=str(binary / 'llvm'), LLVM_NM=str(binary / 'llvm'))
    for args, override, expected in (
      (['--smoke'], {}, [['cheap', 2000], ['expensive', 16]]),
      (['--smoke', '--filter', 'expensive'], {}, [['expensive', 16]]),
      (['--smoke', '--samples', '20000'], {}, [['cheap', 20000], ['expensive', 20000]]),
      (['--samples', '24', '--smoke'], {'RSCRYPTO_CT_DUDECT_SAMPLES': '48'}, [['cheap', 24], ['expensive', 24]]),
      (['--smoke'], {'RSCRYPTO_CT_DUDECT_SAMPLES': '48'}, [['cheap', 48], ['expensive', 48]]),
      (['--filter', 'expensive'], {}, [['expensive', 20000]]),
    ):
      (root / 'budgets.jsonl').write_text('')
      result = subprocess.run(['bash', 'scripts/ct/dudect.sh', '--target', 'fixture-host', *args],
                              cwd=root, env={**env, **override}, capture_output=True, text=True, timeout=20)
      assert result.returncode == 0, result.stdout + result.stderr
      actual = [json.loads(line) for line in (root / 'budgets.jsonl').read_text().splitlines()]
      assert actual == expected, (args, actual)
      if '--smoke' in args:
        report = json.loads((root / 'target/ct/fixture-host/release/dudect/dudect-report.json').read_text())
        assert report['requested_samples_by_case'] == dict(expected)
    # Foreign code may be prepared but must never be timed by this host.
    (root / 'budgets.jsonl').write_text('')
    cross = ['bash', 'scripts/ct/dudect.sh', '--target', 'riscv64gc-unknown-linux-gnu']
    result = subprocess.run(cross, cwd=root, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 2 and 'physical host' in result.stderr, result.stderr
    result = subprocess.run([*cross, '--prepare-only'], cwd=root, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (root / 'budgets.jsonl').read_text()
  manifest = tomllib.loads((ROOT / 'ct.toml').read_text())
  assert all(isinstance(case['smoke_samples'], int) and case['smoke_samples'] >= 2 for case in manifest['dudect_case'])
  print('DudeCT smoke policy regressions passed')


if __name__ == '__main__':
  main()
