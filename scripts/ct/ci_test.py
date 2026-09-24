"""Keep full CT selection complete."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from manifest import required_dudect_cases

spec = importlib.util.spec_from_file_location('ct_ci', Path(__file__).with_name('ci.py'))
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


class Selection(unittest.TestCase):
    def test_power_diagnostic_preserves_target_bound_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'output'
            with patch.dict(os.environ, INPUT_ARCHITECTURES='powerpc64le-linux',
                            DIAGNOSTIC_CASE='mldsa_inverse_ntt_fixed_vs_random',
                            GITHUB_RUN_ID='123', GITHUB_OUTPUT=str(output)), patch.object(sys, 'argv', ['ci.py', 'plan']):
                ci.main()
            values = dict(line.split('=', 1) for line in output.read_text().splitlines())
            self.assertEqual(values['cross'], 'true')
            self.assertEqual(json.loads(values['builds']),
                             {'include': [{'target': 'powerpc64le-unknown-linux-gnu'}]})
            rows = json.loads(values['matrix'])['include']
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['target'], 'powerpc64le-unknown-linux-gnu')

    def test_invalid_diagnostic_emits_no_runnable_matrix(self):
        for architectures, case, error in (
            ('powerpc64le-linux', 'mldsa_*', 'unknown CT diagnostic case'),
            ('x86_64-linux', 'mldsa_inverse_ntt_fixed_vs_random', 'requires POWER, IBM Z, or RISC-V'),
            ('all', 'mldsa_inverse_ntt_fixed_vs_random', 'requires POWER, IBM Z, or RISC-V'),
        ):
            with self.subTest(architectures=architectures, case=case), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / 'output'
                output.write_text('')
                with patch.dict(os.environ, INPUT_ARCHITECTURES=architectures, DIAGNOSTIC_CASE=case,
                                GITHUB_RUN_ID='123', GITHUB_OUTPUT=str(output)), patch.object(sys, 'argv', ['ci.py', 'plan']):
                    with self.assertRaisesRegex(ValueError, error):
                        ci.main()
                self.assertEqual(output.read_text(), '')

    def test_riscv_job_outlasts_required_case_budget(self):
        manifest = tomllib.loads((Path(__file__).resolve().parents[2] / 'ct.toml').read_text())
        cases = required_dudect_cases(manifest, 'riscv64gc-unknown-linux-gnu')
        longest_case = max(case.get('timeout_seconds', 300) for case in cases)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'output'
            with patch.dict(os.environ, INPUT_ARCHITECTURES='riscv64-linux',
                            DIAGNOSTIC_CASE='', GITHUB_RUN_ID='123', GITHUB_OUTPUT=str(output)), patch.object(sys, 'argv', ['ci.py', 'plan']):
                ci.main()
            values = dict(line.split('=', 1) for line in output.read_text().splitlines())
            row = json.loads(values['matrix'])['include'][0]
            self.assertGreater(row['timeout'] * 60, longest_case)

    def test_one_many_all_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'output'
            for selection, count in (('x86_64-linux', 1), ('powerpc64le-linux', 1), ('s390x-linux', 1), ('x86_64-win,riscv64-linux', 2), ('all', 6)):
                output.write_text('')
                with patch.dict(os.environ, INPUT_ARCHITECTURES=selection,
                                DIAGNOSTIC_CASE='', GITHUB_RUN_ID='123', GITHUB_OUTPUT=str(output)), patch.object(sys, 'argv', ['ci.py', 'plan']):
                    ci.main()
                values = dict(line.split('=', 1) for line in output.read_text().splitlines())
                rows = json.loads(values['matrix'])['include']
                self.assertEqual(len(rows), count)
                self.assertTrue(all(row['timeout'] == 360 for row in rows))
                expected_runners = {
                    'x86_64-linux': 'runs-on=123/runner=ct-x86_64-linux-intel/env=production',
                    'aarch64-linux': 'runs-on=123/runner=ct-aarch64-linux/env=production',
                    'x86_64-win': 'runs-on=123/runner=ct-x86_64-win-intel/env=production',
                    'riscv64-linux': 'ubuntu-24.04-riscv',
                    'powerpc64le-linux': 'ubuntu-24.04-ppc64le-p10',
                    's390x-linux': 'ubuntu-24.04-s390x',
                }
                self.assertEqual([row['runner'] for row in rows],
                                 [expected_runners[row['platform']] for row in rows])
                expected = {
                    'riscv64-linux': 'riscv64gc-unknown-linux-gnu',
                    'powerpc64le-linux': 'powerpc64le-unknown-linux-gnu',
                    's390x-linux': 's390x-unknown-linux-gnu',
                }
                builds = [{'target': expected[row['platform']]} for row in rows if row['platform'] in expected]
                self.assertEqual(json.loads(values['builds']), {'include': builds})
                self.assertEqual(values['cross'], str(bool(builds)).lower())
                self.assertEqual([row['target'] for row in rows if 'target' in row], [row['target'] for row in builds])


if __name__ == '__main__':
    unittest.main()
