#!/usr/bin/env -S uv run python
"""Incomplete and unsupported runner results must not qualify a campaign."""

import copy
import unittest

from provider_quic_runner import summarize


class CampaignResults(unittest.TestCase):
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
