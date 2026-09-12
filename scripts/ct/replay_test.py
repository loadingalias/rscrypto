"""A diagnostic replay must retain failures and never stop at the first pass."""

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import replay


class Replay(unittest.TestCase):
    def test_fixed_campaign_retains_failure_and_original_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            args = argparse.Namespace(out=out, prepared=out / 'prepared.json', case='p384', repetitions=3)
            prepared = {'manifest_cases': {'p384': {'samples': 20000, 'timeout_seconds': 7200}},
                        'metadata': {'transfer': {'source': {'commit': 'original'}},
                                     'binary': {'sha256': 'original-binary'}}}
            # Substitute the external measurement boundary; this is controller evidence only.
            with patch.object(replay, 'snapshot', return_value={'cpu': 2}), \
                 patch.object(replay, 'measure', side_effect=[1, 0, 0]) as measure:
                self.assertEqual(replay.repeat(prepared, args, 2), 1)
            self.assertEqual(measure.call_count, 3)
            for index, call in enumerate(measure.call_args_list, 1):
                invocation = call.args[1]
                self.assertEqual((invocation.samples, invocation.threshold, invocation.timeout),
                                 (20000, 10.0, 7200))
                self.assertEqual(invocation.evidence_dir, out / f'repetition-{index}')
                self.assertTrue((invocation.evidence_dir / 'after.json').is_file())
            report = json.loads((out / 'replay.json').read_text())
            self.assertEqual([row['exit_code'] for row in report['repetitions']], [1, 0, 0])
            self.assertEqual(report['source'], {'commit': 'original'})
            self.assertTrue(report['diagnostic_only'])

    def test_tooling_failure_stops_and_remains_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            args = argparse.Namespace(out=out, prepared=out / 'prepared.json', case='p384', repetitions=3)
            prepared = {'manifest_cases': {'p384': {'samples': 20000}},
                        'metadata': {'transfer': {'source': {}}, 'binary': {}}}
            with patch.object(replay, 'snapshot', return_value={}), \
                 patch.object(replay, 'measure', return_value=124) as measure:
                self.assertEqual(replay.repeat(prepared, args, 0), 124)
            self.assertEqual(measure.call_count, 1)
            report = json.loads((out / 'replay.json').read_text())
            self.assertEqual(report['repetitions'], [{'repetition': 1, 'exit_code': 124}])

    def test_single_candidate_retains_failure_without_extra_measurements(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            args = argparse.Namespace(out=out, prepared=out / 'prepared.json', case='p384', repetitions=1)
            prepared = {'manifest_cases': {'p384': {'samples': 20000, 'timeout_seconds': 7200}},
                        'metadata': {'transfer': {'source': {'commit': 'candidate'}}, 'binary': {}}}
            with patch.object(replay, 'snapshot', return_value={}), \
                 patch.object(replay, 'measure', return_value=1) as measure:
                self.assertEqual(replay.repeat(prepared, args, 0), 1)
            self.assertEqual(measure.call_count, 1)
            invocation = measure.call_args.args[1]
            self.assertEqual((invocation.samples, invocation.threshold, invocation.timeout), (20000, 10.0, 7200))
            report = json.loads((out / 'replay.json').read_text())
            self.assertEqual(report['planned_repetitions'], 1)
            self.assertEqual(report['repetitions'], [{'repetition': 1, 'exit_code': 1}])
            self.assertEqual(report['source'], {'commit': 'candidate'})


if __name__ == '__main__':
    unittest.main()
