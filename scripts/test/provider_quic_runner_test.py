#!/usr/bin/env -S uv run python
"""Incomplete and unsupported runner results must not qualify a campaign."""

import copy
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from provider_quic_runner import GO, GO_ECN_ENTRYPOINT, campaign_budget, go_image_for_bundle, summarize


class CampaignResults(unittest.TestCase):
    def test_ecn_binding_keeps_other_cases_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / 'adapter.sh'
            adapter.write_text(GO_ECN_ENTRYPOINT)
            endpoint = root / 'run_endpoint.sh'
            endpoint.write_text('#!/bin/sh\nprintf "%s\\n" "$TESTCASE"\n')
            endpoint.chmod(0o755)
            for case, expected in [('ecn', 'transfer'), ('handshake', 'handshake'),
                                   ('connectionmigration', 'connectionmigration')]:
                with self.subTest(case=case):
                    result = subprocess.run(['sh', str(adapter)], cwd=root,
                                            env=dict(os.environ, TESTCASE=case),
                                            capture_output=True, text=True, check=True)
                    self.assertEqual(result.stdout, expected + '\n')

    def test_go_adapter_requires_the_pinned_base_and_image_identity(self):
        self.assertEqual(go_image_for_bundle({}), GO)
        with self.assertRaises(RuntimeError):
            go_image_for_bundle({'go_image': 'unbound:fixture'})
        self.assertEqual(go_image_for_bundle({'go_image': 'bound:fixture', 'go_base_image': GO,
                                             'go_image_id': 'sha256:fixture'}), 'bound:fixture')

    def test_budget_cannot_exceed_campaign_cap(self):
        self.assertEqual(campaign_budget('360'), 360)
        for value in ('0', '-1', '601', '1.5'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                campaign_budget(value)

    def test_requires_each_requested_case_for_each_pair(self):
        cases = [{'name': 'handshake', 'result': 'succeeded'},
                 {'name': 'retry', 'result': 'succeeded'}]
        complete = {'results': [cases, copy.deepcopy(cases)]}
        self.assertEqual(summarize(complete, ['handshake', 'retry'], 2),
                         {'complete': True, 'passed': 4, 'other': []})
        for pairs in ([], [cases], [cases, []], [cases, [cases[0], cases[0]]]):
            with self.subTest(pairs=pairs):
                self.assertFalse(summarize({'results': pairs}, ['handshake', 'retry'], 2)['complete'])

    def test_failure_and_unsupported_are_retained(self):
        rows = [{'name': 'handshake', 'result': 'failed'},
                {'name': 'retry', 'result': 'unsupported'}]
        self.assertEqual(summarize({'results': [rows]}, ['handshake', 'retry'], 1),
                         {'complete': True, 'passed': 0, 'other': rows})


if __name__ == '__main__':
    unittest.main()
