#!/usr/bin/env python3
"""Execute Just recipes against recorders to verify exact argument boundaries."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
  with tempfile.TemporaryDirectory(prefix="just arguments ") as temporary:
    root = Path(temporary)
    shutil.copy2(ROOT / "justfile", root / "justfile")
    recorder = root / 'executor "quoted"'
    recorder.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
if sys.argv[1:] == ['--print']:
  print(os.environ['PYTHON_RECORDER'])
  sys.exit(0)
with open(os.environ['ARGUMENT_LOG'], 'a') as log:
  log.write(json.dumps(sys.argv[1:]) + '\\n')
''')
    recorder.chmod(0o755)
    source = (root / "justfile").read_text()
    for name in set(re.findall(r'scripts/[\w/.-]+\.(?:sh|py)', source)) | {'bin/cargo'}:
      path = root / name
      path.parent.mkdir(parents=True, exist_ok=True)
      path.symlink_to(recorder)
    log = root / 'arguments.jsonl'
    env = {key: value for key, value in os.environ.items()
           if key not in ('BASH_ENV', 'ENV') and not key.startswith('BASH_FUNC_')}
    env.update(DEV_MACHINE_BIN=str(recorder), PYTHON_RECORDER=str(recorder), ARGUMENT_LOG=str(log),
               PATH=str(root / 'bin') + os.pathsep + os.environ['PATH'],
               CARGO_RAIL_CACHE_REMOTE='remote value', CARGO_RAIL_CACHE_MODE='read-write')
    words = ['two words', "single'quote", 'double"quote', r'^(foo|bar)\s+[0-9].*$',
             '$(touch SHOULD_NOT_EXIST)', '`touch ALSO_NOT`', '*', '']

    def run(recipe, args, expected):
      log.write_text('')
      result = subprocess.run(['just', '--justfile', str(root / 'justfile'), recipe, *args],
                              cwd=root, env=env, text=True, capture_output=True, timeout=20)
      assert result.returncode == 0, (recipe, result.stderr)
      actual = [json.loads(line) for line in log.read_text().splitlines()]
      assert actual == expected, (recipe, actual, expected)
      assert not (root / 'SHOULD_NOT_EXIST').exists() and not (root / 'ALSO_NOT').exists()

    prefixes = {
      '_remote-cargo': ['--exec', 'cargo'],
      'build': ['--exec', 'cargo', 'build', '--locked', '--workspace', '--all-targets', '--all-features'],
      'plan': ['rail', 'plan', '--explain'],
      'test': [], 'test-miri': [], 'test-fuzz': [], 'test-fuzz-asan': [],
      'ct-dudect': [], 'ct-artifacts': [], 'update': [],
      'ct-full': ['scripts/ct/full.py'], 'ct-binsec': ['scripts/ct/binsec.py'],
      'ct-replay': ['scripts/ct/replay.py'],
      'ct-validate': ['scripts/ct/validate.py'],
      'bench': ['scripts/bench/bounded.py', str(recorder), 'scripts/bench/runner.py', 'bench'],
      'profile': ['scripts/bench/bounded.py', str(recorder), 'scripts/bench/runner.py', 'profile'],
      'perf-codegen': ['scripts/bench/runner.py', 'codegen'],
      'perf-llvm-lines': ['scripts/bench/runner.py', 'llvm-lines'],
    }
    for recipe, prefix in prefixes.items():
      run(recipe, words, [[*prefix, *words]])
      if recipe not in ('perf-codegen', 'perf-llvm-lines'):
        run(recipe, [], [prefix])
    for recipe, operation in (('ssh', 'ssh'), ('ssh-create', 'create'), ('ssh-just', 'just')):
      run(recipe, words, [[operation, 'rscrypto', *words]])
      run(recipe, [words[0]], [[operation, 'rscrypto', words[0]]])
    run('ssh-cargo', words, [['just', 'rscrypto', words[0], '_remote-cargo', *words[1:]]])
    for recipe, operation in (('ssh-check', 'ssh'), ('ssh-preflight', 'preflight'),
                              ('ssh-start', 'start'), ('ssh-deallocate', 'deallocate'), ('ssh-kill', 'kill')):
      run(recipe, [words[2]], [[operation, 'rscrypto', words[2], *(['--check'] if recipe == 'ssh-check' else [])]])
    run('ssh-status', [], [['status', 'rscrypto']])
    run('ssh-status', [words[2]], [['status', 'rscrypto', words[2]]])
    run('ssh-bootstrap', [words[2]], [['bootstrap', 'rscrypto', words[2]]])
    run('ssh-bootstrap', words[:2], [['bootstrap', 'rscrypto', *words[:2]]])
    run('ssh-collect-bench', words[:3], [['just', 'rscrypto', words[0], 'bench-export', 'benchmark_results/criterion/' + words[1]],
                                      ['collect-results', 'rscrypto', words[0], 'criterion', *words[1:3]]])
    run('bench-export', [words[2]], [['scripts/bench/runner.py', 'export', words[2]]])
    cache = ['--remote', 'remote value', '--remote-mode', 'read-write', '--root-portability', 'remap']
    run('rail-cache-setup', words, [['rail', 'cache', 'setup', '--check', *cache, *words],
                                  ['rail', 'cache', 'setup', *cache, *words], ['rail', 'cache', 'probe', '--json']])
    env.pop('HOME', None)
    run('ci-check', [], [['native']])
    covered = set(prefixes) | {'ssh', 'ssh-create', 'ssh-just', 'ssh-cargo', 'rail-cache-setup'}
    assert set(re.findall(r'^([\w-]+).*\*args:', source, re.M)) == covered
  print('Just argument forwarding regressions passed')


if __name__ == '__main__':
  main()
