#!/usr/bin/env python3
"""Exercise shared preparation actions and protect their qualification boundaries."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
ACTION = './.github/actions/cross-build/'


def read_yaml(path):
    return yaml.load((ROOT / path).read_text(), Loader=yaml.BaseLoader)


class CrossWorkflow(unittest.TestCase):
    def setUp(self):
        self.setup = read_yaml('.github/actions/cross-build/setup/action.yml')['runs']['steps']
        self.upload = read_yaml('.github/actions/cross-build/upload/action.yml')['runs']['steps']
        temporary = tempfile.TemporaryDirectory(prefix='cross workflow ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / 'output'
        self.output.touch()
        self.env = {key: value for key, value in os.environ.items()
                    if key not in ('BASH_ENV', 'ENV') and not key.startswith('BASH_FUNC_')}
        self.env.update(GITHUB_OUTPUT=str(self.output), GITHUB_PATH=str(self.output))

    def execute(self, step, **env):
        return subprocess.run(['bash', '--noprofile', '--norc', '-eo', 'pipefail', '-c', step['run']],
                              cwd=self.root, env={**self.env, **env}, text=True, capture_output=True)

    def test_setup_preserves_target_and_stops_on_installation_failure(self):
        restore, install, save = self.setup
        self.assertEqual(restore['with']['id'], 'cross-build-${{ inputs.target }}')
        self.assertEqual(save['if'], 'always() && steps.apt-state.outputs.key')
        self.assertEqual(save['with'], {'key': '${{ steps.apt-state.outputs.key }}',
                                       'hit': '${{ steps.apt-state.outputs.hit }}'})
        installer = self.root / 'scripts/tooling/x86_64-linux.sh'
        installer.parent.mkdir(parents=True)
        installer.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > arguments\nexit "${INSTALL_EXIT:-0}"\n')
        installer.chmod(0o755)
        for target in ('s390x-unknown-linux-gnu', 'target with spaces'):
            with self.subTest(target=target):
                self.output.write_text('')
                result = self.execute(install, TARGET=target)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((self.root / 'arguments').read_text().splitlines(), ['--ci-cross-build', target])
                self.assertEqual(self.output.read_text(), str(Path.home() / '.cargo/bin') + '\n')
        self.output.write_text('')
        result = self.execute(install, TARGET='s390x-unknown-linux-gnu', INSTALL_EXIT='42')
        self.assertEqual(result.returncode, 42)
        self.assertEqual(self.output.read_text(), '')

    def test_upload_requires_both_files_and_keeps_consumer_names(self):
        require, upload = self.upload
        self.assertNotIn('if', upload)  # The transfer must require successful preparation.
        self.assertNotIn('continue-on-error', upload)
        self.assertEqual(upload['with'], {
            'name': '${{ steps.bundle.outputs.artifact }}',
            'path': '${{ steps.bundle.outputs.archive }}\ntarget/${{ inputs.target }}-tools.tar.gz\n',
            'if-no-files-found': 'error', 'retention-days': '2',
        })
        target = 'riscv64gc-unknown-linux-gnu'
        cases = [('tests', 'tests.tar.xz', 'tests'), ('ct', 'ct.tar.gz', 'ct-prepared'),
                 ('bench', 'bench.tar.gz', 'bench-prepared'), ('profile', 'profile.tar.gz', 'profile-prepared')]
        (self.root / 'target').mkdir()
        tools = self.root / 'target' / f'{target}-tools.tar.gz'
        for suite, filename, artifact in cases:
            with self.subTest(suite=suite):
                archive = self.root / 'target' / f'{target}-{filename}'
                archive.write_bytes(b'prepared suite')
                tools.write_bytes(b'runner tools')
                self.output.write_text('')
                result = self.execute(require, TARGET=target, SUITE=suite)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.output.read_text(),
                                 f'archive=target/{target}-{filename}\nartifact={target}-{artifact}\n')
                for missing in (archive, tools):
                    missing.unlink()
                    self.output.write_text('')
                    result = self.execute(require, TARGET=target, SUITE=suite)
                    self.assertEqual(result.returncode, 1)
                    self.assertIn('Missing prepared artifact:', result.stderr)
                    self.assertEqual(self.output.read_text(), '')
                    missing.write_bytes(b'restored fixture')
        self.output.write_text('')
        result = self.execute(require, TARGET=target, SUITE='unknown')
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.output.read_text(), '')

    def test_callers_keep_preparation_separate_from_native_qualification(self):
        for filename, producer, consumer, suite, command, suffix in (
            ('ci', 'cross-build', 'native', 'tests', 'just test-cross prepare', 'tests'),
            ('ct', 'cross-build', 'ct', 'ct', 'just ct-full --target', 'ct-prepared'),
            ('bench', 'cross-build', 'bench', 'bench', 'python3 scripts/bench/ci.py prepare', 'bench-prepared'),
            ('profile', 'prepare', 'capture', 'profile', 'python3 scripts/bench/profile_ci.py prepare', 'profile-prepared'),
        ):
            with self.subTest(workflow=filename):
                jobs = read_yaml(f'.github/workflows/{filename}.yml')['jobs']
                build, native = jobs[producer], jobs[consumer]
                target = '${{ needs.plan.outputs.target }}' if suite == 'profile' else '${{ matrix.target }}'
                steps = build['steps']
                setup, = [step for step in steps if step.get('uses') == ACTION + 'setup']
                upload, = [step for step in steps if step.get('uses') == ACTION + 'upload']
                prepare, = [step for step in steps if command in step.get('run', '')]
                self.assertEqual(setup['with'], {'target': target})
                self.assertEqual(upload['with'], {'target': target, 'suite': suite})
                self.assertLess(steps.index(setup), steps.index(prepare))
                self.assertLess(steps.index(prepare), steps.index(upload))
                self.assertNotIn('if', setup)
                self.assertNotIn('if', upload)
                self.assertTrue(all('continue-on-error' not in step for step in (setup, prepare, upload)))
                self.assertIn('source "$HOME/.local/share/rscrypto-tooling/environment.sh"', prepare['run'])
                needs = native['needs']
                self.assertIn(producer, needs if isinstance(needs, list) else [needs])
                download, = [step for step in native['steps']
                             if step.get('uses', '').startswith('actions/download-artifact@')]
                self.assertEqual(download['with']['name'], f'{target}-{suffix}')
                self.assertNotIn('run-id', download['with'])
                if suite != 'profile':
                    self.assertEqual(build['strategy']['fail-fast'], 'true')
                    self.assertEqual(native['strategy']['fail-fast'], 'true')
                if suite in ('tests', 'ct'):
                    self.assertIn('just test-transfer', prepare['run'])


if __name__ == '__main__':
    unittest.main()
