"""Keep smoke/full selection distinct and stop CT execution on the first failure."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('ct_ci', Path(__file__).with_name('ci.py'))
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


class Selection(unittest.TestCase):
    def test_one_many_all_and_invalid_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'output'
            for selection, count in (('x86_64-linux', 1), ('x86_64-win,riscv64-linux', 2), ('all', 6)):
                output.write_text('')
                with patch.dict(os.environ, INPUT_MODE='full', INPUT_ARCHITECTURES=selection,
                                GITHUB_RUN_ID='123', GITHUB_OUTPUT=str(output)), patch.object(sys, 'argv', ['ci.py', 'plan']):
                    ci.main()
                rows = json.loads(output.read_text().removeprefix('matrix='))['include']
                self.assertEqual(len(rows), count)
                self.assertTrue(all(row['timeout'] == 360 for row in rows))
            with patch.dict(os.environ, INPUT_MODE='invalid'), self.assertRaises(ValueError):
                ci.main()

    def test_full_does_not_repeat_smoke_or_filter_required_cases(self):
        with patch.dict(os.environ, INPUT_MODE='full', PLATFORM='aarch64-linux'), \
             patch.object(sys, 'argv', ['ci.py', 'run']), patch.object(ci.subprocess, 'run') as run:
            ci.main()
            self.assertEqual([call.args[0] for call in run.call_args_list], [['just', 'ct-full']])

    def test_failed_artifacts_prevent_timing(self):
        with patch.dict(os.environ, INPUT_MODE='smoke', PLATFORM='x86_64-linux'), \
             patch.object(sys, 'argv', ['ci.py', 'run']), \
             patch.object(ci.subprocess, 'run', side_effect=[None, subprocess.CalledProcessError(7, 'artifacts')]) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                ci.main()
            self.assertEqual([call.args[0] for call in run.call_args_list], [['just', 'ct-test'], ['just', 'ct-artifacts']])


if __name__ == '__main__':
    unittest.main()
