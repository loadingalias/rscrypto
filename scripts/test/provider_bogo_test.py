#!/usr/bin/env -S uv run python
"""Keep BoGo profile filtering tied to declared protocol requirements."""
import unittest
from provider_bogo import exclusions


class ProfileExclusions(unittest.TestCase):
    def profile(self, name, **fields):
        return dict(dict(Name=name, Protocol='TLS', MaxVersion=0x0304, ExpectedVersion=0,
                         ResumeMaxVersion=0, ResumeExpectedVersion=0, ShouldFail=False,
                         ResumeSession=False, NoExtendedMasterSecret=False,
                         ResumeNoExtendedMasterSecret=False), **fields)

    def test_supported_profiles_remain_eligible_independent_of_case_names(self):
        profiles = [self.profile('failure-looking-name'),
                    self.profile('unconstrained', MaxVersion=0),
                    self.profile('tls12', MaxVersion=0x0303),
                    self.profile('cross-version', ResumeExpectedVersion=0x0303),
                    self.profile('quic', Protocol='QUIC'),
                    self.profile('dtls', Protocol='DTLS')]
        self.assertEqual(set(exclusions(profiles)), {'quic', 'dtls'})

    def test_older_protocol_requirements_are_excluded(self):
        fields = ('MaxVersion', 'ExpectedVersion', 'ResumeMaxVersion', 'ResumeExpectedVersion')
        profiles = [self.profile(field, **{field: 0x0302}) for field in fields]
        self.assertEqual(set(exclusions(profiles)), set(fields))

    def test_ems_policy_keeps_rejection_cases_and_tls13_eligible(self):
        profiles = [
            self.profile('initial-positive', MaxVersion=0x0303, NoExtendedMasterSecret=True),
            self.profile('legacy-before-resumption', MaxVersion=0x0303, NoExtendedMasterSecret=True,
                         ResumeSession=True, ShouldFail=True),
            self.profile('resumed-positive', ResumeMaxVersion=0x0303,
                         ResumeNoExtendedMasterSecret=True, ResumeSession=True),
            self.profile('initial-negative', MaxVersion=0x0303, NoExtendedMasterSecret=True,
                         ShouldFail=True),
            self.profile('resumed-negative', ResumeMaxVersion=0x0303,
                         ResumeNoExtendedMasterSecret=True, ResumeSession=True, ShouldFail=True),
            self.profile('tls13', NoExtendedMasterSecret=True, ResumeNoExtendedMasterSecret=True,
                         ResumeSession=True, ResumeMaxVersion=0x0304),
        ]
        self.assertEqual(set(exclusions(profiles)),
                         {'initial-positive', 'legacy-before-resumption', 'resumed-positive'})


if __name__ == '__main__':
    unittest.main()
