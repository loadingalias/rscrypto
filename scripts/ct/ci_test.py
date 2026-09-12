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
    def test_riscv_job_outlasts_required_case_budget(self):
        manifest = tomllib.loads((Path(__file__).resolve().parents[2] / 'ct.toml').read_text())
        cases = required_dudect_cases(manifest, 'riscv64gc-unknown-linux-gnu')
        longest_case = max(case.get('timeout_seconds', 300) for case in cases)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'output'
            with patch.dict(os.environ, INPUT_ARCHITECTURES='riscv64-linux',
                            GITHUB_RUN_ID='123', GITHUB_OUTPUT=str(output)), patch.object(sys, 'argv', ['ci.py', 'plan']):
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
                                GITHUB_RUN_ID='123', GITHUB_OUTPUT=str(output)), patch.object(sys, 'argv', ['ci.py', 'plan']):
                    ci.main()
                values = dict(line.split('=', 1) for line in output.read_text().splitlines())
                rows = json.loads(values['matrix'])['include']
                self.assertEqual(len(rows), count)
                self.assertTrue(all(row['timeout'] == 360 for row in rows))
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
