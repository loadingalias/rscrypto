#!/usr/bin/env -S uv run python
"""Keep BoGo profile filtering tied to declared protocol requirements, and the shim on the linked rustls."""
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from provider_bogo import exclusions, linked_rustls, registry_version, vcs_commit


def dependency(entry):
    return tomllib.loads(f'rustls = {entry}')['rustls']


class RustlsPin(unittest.TestCase):
    def test_git_pin_with_a_version_stays_a_git_pin(self):
        # A registry reading of this entry would build the shim from the crates.io
        # 0.24.0-dev.2 commit, not from the revision that the provider links.
        pinned = dependency('{ version = "=0.24.0-dev.2", git = "https://github.com/rustls/rustls", '
                            'rev = "0419f479ff5c99a471428ee197e899ccd1629aa8", default-features = false }')
        self.assertIsNone(registry_version(pinned))

    def test_exact_registry_pins_name_their_version(self):
        self.assertEqual(registry_version(dependency('"=0.24.0"')), '0.24.0')
        self.assertEqual(registry_version(dependency(
            '{ version = "=0.24.0-rc.1", default-features = false, features = ["webpki"] }')), '0.24.0-rc.1')

    def test_pins_that_name_no_single_commit_fail_with_their_cause(self):
        for entry, cause in (('"0.24.0"', 'range'), ('"=0.24"', 'range'), ('">=0.24.0, <0.25"', 'range'),
                             ('{ default-features = false }', 'no version'),
                             ('{ version = "=0.24.0", path = "../rustls" }', 'path dependency'),
                             ('{ git = "https://github.com/rustls/rustls", branch = "main" }', 'no `rev`')):
            with self.subTest(entry), self.assertRaisesRegex(ValueError, cause):
                registry_version(dependency(entry))

    def test_the_provider_dependency_selects_the_crate(self):
        # Another workspace member may link a second rustls; only the provider's names the shim's source.
        ids = ('rustls 0.23.32', 'rustls 0.24.0', 'rscrypto-rustls 0.1.0', 'other 0.1.0')
        metadata = {'packages': [dict(zip(('id', 'name', 'version'), (id, *id.split()))) for id in ids],
                    'resolve': {'nodes': [
                        {'id': 'other 0.1.0', 'deps': [{'name': 'rustls', 'pkg': 'rustls 0.23.32'}]},
                        {'id': 'rscrypto-rustls 0.1.0', 'deps': [{'name': 'rustls', 'pkg': 'rustls 0.24.0'}]}]}}
        self.assertEqual(linked_rustls(metadata)['version'], '0.24.0')

    def vcs_record(self, record):
        crate = Path(self.enterContext(tempfile.TemporaryDirectory()))
        if record is not None:
            (crate / '.cargo_vcs_info.json').write_text(json.dumps(record))
        return crate

    def test_registry_commit_comes_from_the_packaged_vcs_record(self):
        # The form that cargo package wrote into the published rustls 0.23.32.
        crate = self.vcs_record({'git': {'sha1': '6a188a70a0e1bf4547884a9cdd7710936d653a22'}, 'path_in_vcs': 'rustls'})
        self.assertEqual(vcs_commit(crate), '6a188a70a0e1bf4547884a9cdd7710936d653a22')

    def test_crates_without_a_clean_vcs_record_fail(self):
        for record, cause in ((None, 'missing'),
                              ({'git': {'sha1': '6a188a70a0e1bf4547884a9cdd7710936d653a22', 'dirty': True},
                                'path_in_vcs': 'rustls'}, 'modified checkout')):
            with self.subTest(cause), self.assertRaisesRegex(ValueError, cause):
                vcs_commit(self.vcs_record(record))


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
