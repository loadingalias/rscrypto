#!/usr/bin/env -S uv run python
"""Regressions for provider graph isolation, feature forwarding, and target scope."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import provider


class Provider(unittest.TestCase):
    def test_std_scope_preserves_core_only_catalog_targets(self):
        targets = json.loads((provider.ROOT / '.config/target-matrix.json').read_text())['targets']
        excluded = {target for target in targets if not provider.supports_std(target)}
        self.assertEqual(excluded, {'aarch64-unknown-none', 'x86_64-unknown-none',
                                    'thumbv6m-none-eabi', 'riscv32imac-unknown-none-elf',
                                    'wasm32-unknown-unknown'})
        self.assertTrue(provider.supports_std('wasm32-wasip1'))

    def test_feature_contract_rejects_msrv_drift_and_extra_forwarding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'rscrypto-rustls').mkdir()
            (root / 'Cargo.toml').write_text('[package]\nrust-version="1.100.0"\n[features]\nhkdf=[]\n')
            manifest = root / 'rscrypto-rustls/Cargo.toml'
            valid = ('[package]\nrust-version="1.100.0"\n'
                     '[features]\ndefault=["hkdf"]\nhkdf=["rscrypto/hkdf"]\nportable-only=["rscrypto/portable-only"]\n')
            with (root / 'Cargo.toml').open('a') as output:
                output.write('portable-only=[]\ngetrandom=[]\n')
            valid += 'getrandom=["rscrypto/getrandom", "dep:getrandom"]\n'
            manifest.write_text(valid)
            self.assertIn(('hkdf', ['hkdf']), provider.feature_matrix(root))
            for invalid in (valid.replace('1.100.0', '1.99.0'),
                            valid.replace('["rscrypto/hkdf"]', '["rscrypto/hkdf", "rscrypto/full"]'),
                            valid.replace('"dep:getrandom"', '"dep:other-rng"')):
                manifest.write_text(invalid)
                with self.assertRaises(ValueError):
                    provider.feature_matrix(root)

    def test_real_cargo_tree_excludes_dev_but_rejects_transitive_and_foreign_build_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ['rscrypto-rustls', 'relay', 'ring', 'cc', 'openssl-sys', 'getrandom']
            (root / 'Cargo.toml').write_text('[workspace]\nresolver="3"\nmembers=' + json.dumps(names) + '\n')
            for name in names:
                directory = root / name
                directory.mkdir()
                (directory / 'lib.rs').write_text('// Dependency graph fixture.\n')
                (directory / 'Cargo.toml').write_text(
                    '[package]\nname=' + json.dumps(name) + '\nversion="0.1.0"\nedition="2024"\n'
                    '[lib]\npath="lib.rs"\n')
            with (root / 'relay/Cargo.toml').open('a') as output:
                output.write('[dependencies]\nbackend={package="ring",path="../ring"}\n')
            with (root / 'rscrypto-rustls/Cargo.toml').open('a') as output:
                output.write('[features]\ncrypto=["dep:relay"]\nbuilder=["dep:compiler"]\n'
                             'getrandom=["dep:getrandom"]\nentropy=["dep:getrandom"]\n'
                             '[dependencies]\nrelay={path="../relay",optional=true}\n'
                             'getrandom={path="../getrandom",optional=true}\n'
                             '[target.\'cfg(target_os="windows")\'.build-dependencies]\n'
                             'compiler={package="cc",path="../cc",optional=true}\n'
                             '[dev-dependencies]\nopenssl-sys={path="../openssl-sys"}\n')
            subprocess.run(['cargo', '+' + provider.toolchain.channel(), 'generate-lockfile', '--offline'],
                           cwd=root, check=True, capture_output=True)
            # The shared lockfile contains every forbidden fixture, even for the
            # empty feature graph. Only normal/build reachability decides the gate.
            with contextlib.redirect_stdout(io.StringIO()):
                provider.no_c([], root)
                provider.no_c(['getrandom'], root)
                for features, name in ((['crypto'], 'ring'), (['builder'], 'cc'), (['entropy'], 'getrandom')):
                    with self.subTest(features=features), self.assertRaisesRegex(ValueError, name):
                        provider.no_c(features, root)


GIT = 'version = "=0.24.0-dev.2", git = "https://github.com/rustls/rustls", rev = "0419f479ff5c99a471428ee197e899ccd1629aa8"'


class ToolPins(unittest.TestCase):
    def tool(self, edit):
        """Write a provider and a passing tool fixture, then apply `edit` to the tool manifest."""
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (root / 'rscrypto-rustls').mkdir()
        (root / 'rscrypto-rustls/Cargo.toml').write_text(
            f'[dependencies]\nrustls = {{ {GIT}, default-features = false, features = ["webpki"] }}\n'
            'rustls-webpki = { version = "=0.104.0-alpha.8", default-features = false }\n'
            f'[dev-dependencies]\nrustls-ring = {{ {GIT} }}\n')
        # Features, renames, and target tables may differ from the provider's.
        valid = (f'[dependencies]\nrustls = {{ {GIT}, features = ["webpki", "std"] }}\n'
                 f"[target.'cfg(unix)'.dev-dependencies]\nrustls-ring = {{ {GIT} }}\n"
                 'webpki = { package = "rustls-webpki", version = "=0.104.0-alpha.8" }\n')
        (root / 'tools/bench').mkdir(parents=True)
        (root / 'tools/bench/Cargo.toml').write_text(valid)
        provider.check_tool_pins(root)
        (root / 'tools/bench/Cargo.toml').write_text(edit(valid))
        return root

    def test_a_tool_on_another_rustls_revision_fails(self):
        root = self.tool(lambda valid: valid.replace('rev = "0419f479', 'rev = "e7506724', 1))
        with self.assertRaisesRegex(ValueError, r'tools/bench/Cargo\.toml pins rustls as'):
            provider.check_tool_pins(root)

    def test_a_tool_on_another_registry_version_fails(self):
        root = self.tool(lambda valid: valid.replace('=0.104.0-alpha.8', '=0.104.0-alpha.7'))
        with self.assertRaisesRegex(ValueError, r'tools/bench/Cargo\.toml pins rustls-webpki as'):
            provider.check_tool_pins(root)

    def test_a_rustls_crate_that_the_provider_does_not_pin_fails(self):
        root = self.tool(lambda valid: valid + f'rustls-util = {{ {GIT} }}\n')
        with self.assertRaisesRegex(ValueError, r'tools/bench/Cargo\.toml pins rustls-util, which'):
            provider.check_tool_pins(root)


if __name__ == '__main__':
    unittest.main()
