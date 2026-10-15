#!/usr/bin/env -S uv run python
"""Keep BoGo profile filtering tied to declared protocol requirements."""
import unittest
from provider_bogo import exclusions


class ProfileExclusions(unittest.TestCase):
    def test_tls13_profiles_remain_eligible_independent_of_case_names(self):
        base = dict(Protocol='TLS', MaxVersion=0x0304, ExpectedVersion=0,
                    ResumeMaxVersion=0, ResumeExpectedVersion=0)
        profiles = [dict(base, Name='failure-looking-name'),
                    dict(base, Name='unconstrained', MaxVersion=0),
                    dict(base, Name='tls12', ResumeExpectedVersion=0x0303),
                    dict(base, Name='quic', Protocol='QUIC')]
        self.assertEqual(set(exclusions(profiles)), {'tls12', 'quic'})


if __name__ == '__main__':
    unittest.main()
