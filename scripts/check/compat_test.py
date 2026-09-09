#!/usr/bin/env python3
"""Check compatibility coverage and prove failures terminate sibling work."""
import importlib.util
import os
import json
import shutil
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('compat', Path(__file__).with_name('compat.py'))
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


@unittest.skipUnless(os.name == "posix", "compatibility workers execute on Linux")
class Compatibility(unittest.TestCase):
    def test_musl_runs_both_profiles_and_preserves_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'scripts/test').mkdir(parents=True)
            (root / 'scripts/lib').mkdir(parents=True)
            shutil.copy2(compat.ROOT / 'scripts/test/test-musl.sh', root / 'scripts/test/test-musl.sh')
            selector = root / 'scripts/lib/toolchain.sh'
            selector.write_text('#!/bin/sh\nprintf "%s\\n" "$FIXTURE_HOST"\n')
            selector.chmod(0o755)
            runner = root / 'scripts/test/test.sh'
            runner.write_text('#!' + sys.executable + '\n' + """import json,os,sys
with open(os.environ['LOG'], 'a') as log:
    log.write(json.dumps([sys.argv[1:], os.environ['CARGO_BUILD_TARGET'],
                         os.environ['CC_' + os.environ['CARGO_BUILD_TARGET'].replace('-', '_')],
                         os.environ['CARGO_TARGET_' + os.environ['CARGO_BUILD_TARGET'].replace('-', '_').upper() + '_LINKER']]) + '\\n')
sys.exit(int(os.environ['FAIL']))
""")
            runner.chmod(0o755)
            for arch in ('x86_64', 'aarch64'):
                for fail in (0, 7):
                    log = root / 'log'
                    log.write_text('')
                    result = subprocess.run([shutil.which('bash'), 'scripts/test/test-musl.sh'], cwd=root,
                                            env={**os.environ, 'FIXTURE_HOST': arch + '-unknown-linux-gnu',
                                                 'LOG': str(log), 'FAIL': str(fail)}, capture_output=True, text=True)
                    self.assertEqual(result.returncode, fail, result.stderr)
                    rows = [json.loads(line) for line in log.read_text().splitlines()]
                    profiles = [['--all']] if fail else [['--all'], ['--all', '--portable']]
                    self.assertEqual(rows, [[profile, arch + '-unknown-linux-musl', 'musl-gcc', 'musl-gcc']
                                            for profile in profiles])

    def test_plan_covers_contracts(self):
        plan = list(compat.cases())
        names = [name for name, _, _ in plan]
        self.assertEqual(len(names), len(set(names)))
        manifest = compat.read('Cargo.toml')
        for channel in (compat.toolchain.stable(), manifest['package']['rust-version']):
            for feature in manifest['features']:
                self.assertIn(f'{channel}-{feature}', names)
            for boundary in ('core', 'alloc'):
                self.assertIn(f'{channel}-thumb-{boundary}', names)
        self.assertTrue(set(compat.targets()) <= set(names))
        wasm = [(commands, env) for name, commands, env in plan if name.endswith(('-scalar', '-simd'))]
        self.assertEqual(len(wasm), 4)
        self.assertTrue(all(commands[1][:2] == ['wasmtime', 'run'] for commands, _ in wasm))
        self.assertEqual(sum('--invoke' in commands[1] for commands, _ in wasm), 2)

    def test_failure_kills_running_sibling_and_skips_pending_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pid = root / 'pid'
            later = root / 'later'
            sibling = [sys.executable, '-c',
                       'import os,time,pathlib; pathlib.Path(' + repr(str(pid)) + ').write_text(str(os.getpid())); time.sleep(60)']
            failure = [sys.executable, '-c',
                       'import pathlib,time,sys\np=pathlib.Path(' + repr(str(pid)) + ')\n'
                       'while not p.exists(): time.sleep(0.01)\nsys.exit(7)']
            pending = [sys.executable, '-c', 'import pathlib; pathlib.Path(' + repr(str(later)) + ').touch()']
            with self.assertRaisesRegex(SystemExit, 'failed \\(7\\)'):
                compat.execute([('sibling', [sibling], {}), ('failure', [failure], {}),
                                ('pending', [pending], {})], 2, root / 'logs')
            self.assertFalse(later.exists())
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pid.read_text()), 0)

    def test_command_failure_prevents_runtime_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / 'ran'
            with self.assertRaises(SystemExit):
                compat.execute([('build', [[sys.executable, '-c', 'raise SystemExit(9)'],
                                          [sys.executable, '-c', 'open(' + repr(str(marker)) + ', "w").close()']], {})],
                               1, root / 'logs')
            self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main()
