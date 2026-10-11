#!/usr/bin/env python3
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
            valid += 'quinn=["dep:quinn-proto", "dep:bytes", "aes-gcm", "hkdf", "getrandom"]\n'
            manifest.write_text(valid)
            self.assertIn(('hkdf', ['hkdf']), provider.feature_matrix(root))
            for invalid in (valid.replace('1.100.0', '1.99.0'),
                            valid.replace('["rscrypto/hkdf"]', '["rscrypto/hkdf", "rscrypto/full"]'),
                            valid.replace('"dep:getrandom"', '"dep:other-rng"'),
                            valid.replace('"dep:quinn-proto"', '"dep:quinn-proto", "quinn-proto/ring"')):
                manifest.write_text(invalid)
                with self.assertRaises(ValueError):
                    provider.feature_matrix(root)

    def test_real_cargo_tree_excludes_dev_but_rejects_transitive_and_foreign_build_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ['rscrypto-rustls', 'relay', 'ring', 'cc', 'openssl-sys']
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
                             '[dependencies]\nrelay={path="../relay",optional=true}\n'
                             '[target.\'cfg(target_os="windows")\'.build-dependencies]\n'
                             'compiler={package="cc",path="../cc",optional=true}\n'
                             '[dev-dependencies]\nopenssl-sys={path="../openssl-sys"}\n')
            subprocess.run(['cargo', '+' + provider.toolchain.channel(), 'generate-lockfile', '--offline'],
                           cwd=root, check=True, capture_output=True)
            # The shared lockfile contains every forbidden fixture, even for the
            # empty feature graph. Only normal/build reachability decides the gate.
            with contextlib.redirect_stdout(io.StringIO()):
                provider.no_c([], root)
                for features, name in ((['crypto'], 'ring'), (['builder'], 'cc')):
                    with self.subTest(features=features), self.assertRaisesRegex(ValueError, name):
                        provider.no_c(features, root)


if __name__ == '__main__':
    unittest.main()
