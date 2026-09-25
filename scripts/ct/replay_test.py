"""A diagnostic replay must retain failures and never stop at the first pass."""

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch

import replay


class Replay(unittest.TestCase):
    def test_suite_keeps_each_budget_and_continues_after_timing_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            args = argparse.Namespace(out=out, prepared=out / 'prepared.json', case='mldsa', repetitions=1)
            primitive = 'signature.mldsa.secret_kernels'
            prepared = {
                'manifest_cases': {
                    'mldsa_a': {'primitive': primitive, 'samples': 20000},
                    'mldsa_b': {'primitive': primitive, 'samples': 24000, 'timeout_seconds': 7200},
                    'mldsa_diagnostic': {'primitive': primitive, 'samples': 512, 'gate': 'diagnostic'},
                    'unrelated': {'primitive': 'another', 'samples': 20000},
                },
                'metadata': {'transfer': {'source': {'commit': 'candidate'}}, 'binary': {'sha256': 'binary'}},
            }
            with patch.object(replay, 'snapshot', return_value={}), \
                 patch.object(replay, 'measure', side_effect=[1, 0]) as measure:
                self.assertEqual(replay.campaign(prepared, args, 0), 1)
            self.assertEqual([(call.args[1].filter, call.args[1].samples, call.args[1].timeout,
                               call.args[1].threshold) for call in measure.call_args_list],
                             [('mldsa_a', 20000, 300, 10.0), ('mldsa_b', 24000, 7200, 10.0)])
            report = json.loads((out / 'campaign.json').read_text())
            self.assertEqual(report['planned_cases'], ['mldsa_a', 'mldsa_b'])
            self.assertEqual(report['cases'], [{'name': 'mldsa_a', 'exit_code': 1},
                                              {'name': 'mldsa_b', 'exit_code': 0}])
            self.assertTrue(report['complete'])
            for name in report['planned_cases']:
                self.assertTrue((out / name / 'repetition-1/after.json').is_file())

    def test_suite_tooling_failure_records_incomplete_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            args = argparse.Namespace(out=out, prepared=out / 'prepared.json', case='mldsa', repetitions=1)
            prepared = {'manifest_cases': {
                name: {'primitive': 'signature.mldsa.secret_kernels', 'samples': 20000}
                for name in ('mldsa_a', 'mldsa_b')
            }, 'metadata': {'transfer': {'source': {}}, 'binary': {}}}
            with patch.object(replay, 'snapshot', return_value={}), \
                 patch.object(replay, 'measure', return_value=124) as measure:
                self.assertEqual(replay.campaign(prepared, args, 0), 124)
            self.assertEqual(measure.call_count, 1)
            report = json.loads((out / 'campaign.json').read_text())
            self.assertFalse(report['complete'])
            self.assertEqual(report['planned_cases'], ['mldsa_a', 'mldsa_b'])
            self.assertEqual(report['cases'], [{'name': 'mldsa_a', 'exit_code': 124}])

    def test_suite_rejects_an_archive_without_required_mldsa_cases(self):
        with self.assertRaisesRegex(ValueError, 'required kernel cases'):
            replay.replay_cases({'mldsa_old': {'primitive': 'signature.mldsa', 'gate': 'diagnostic'}}, 'mldsa')

    def test_native_target_reaches_archive_verification_and_affinity_is_restored(self):
        for target in ('powerpc64le-unknown-linux-gnu', 's390x-unknown-linux-gnu', None):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                prepared_path = root / 'prepared.json'
                prepared_path.write_text(json.dumps({
                    'manifest_cases': {'mldsa_inverse': {'samples': 20000}},
                    'metadata': {'transfer': {'source': {'commit': 'source'}}, 'binary': {}},
                }))
                args = ['replay.py', '--source-root', str(root), '--archive', str(root / 'sealed.tar.gz'),
                        '--out', str(root / 'result'), '--case', 'mldsa_inverse', '--repetitions', '1']
                if target:
                    args += ['--target', target]
                # Substitute archive IO, OS affinity, and the external measurement process.
                # The production controller still selects the target and retains the failed result.
                with patch('sys.argv', args), \
                     patch.object(replay, 'consume', return_value=([], prepared_path)) as consume, \
                     patch.object(replay.os, 'sched_getaffinity', return_value={2, 4}, create=True), \
                     patch.object(replay.os, 'sched_setaffinity', create=True) as affinity, \
                     patch.object(replay, 'snapshot', return_value={}), \
                     patch.object(replay, 'measure', return_value=1) as measure:
                    self.assertEqual(replay.main(), 1)
                consume.assert_called_once_with(root, root / 'result', root / 'sealed.tar.gz',
                                                target or 'riscv64gc-unknown-linux-gnu')
                self.assertEqual(affinity.call_args_list, [call(0, {2}), call(0, {2, 4})])
                self.assertEqual(measure.call_count, 1)
                report = json.loads((root / 'result/replay.json').read_text())
                self.assertEqual(report['repetitions'], [{'repetition': 1, 'exit_code': 1}])

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
