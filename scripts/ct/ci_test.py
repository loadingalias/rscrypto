"""Keep full CT selection complete."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('ct_ci', Path(__file__).with_name('ci.py'))
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


class Selection(unittest.TestCase):
    def test_one_many_all_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'output'
            for selection, count in (('x86_64-linux', 1), ('x86_64-win,riscv64-linux', 2), ('all', 6)):
                output.write_text('')
                with patch.dict(os.environ, INPUT_ARCHITECTURES=selection,
                                GITHUB_RUN_ID='123', GITHUB_OUTPUT=str(output)), patch.object(sys, 'argv', ['ci.py', 'plan']):
                    ci.main()
                values = dict(line.split('=', 1) for line in output.read_text().splitlines())
                rows = json.loads(values['matrix'])['include']
                self.assertEqual(len(rows), count)
                self.assertTrue(all(row['timeout'] == (60 if row['platform'] == 'riscv64-linux' else 360) for row in rows))
                self.assertEqual(values['riscv'], str(any(row['platform'] == 'riscv64-linux' for row in rows)).lower())


if __name__ == '__main__':
    unittest.main()
