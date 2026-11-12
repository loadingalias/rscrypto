#!/usr/bin/env -S uv run python
"""Regressions for the provider archive's content check and its Git-pin patches."""

from pathlib import Path
import tempfile
import tomllib
import unittest

import package


class Contents(unittest.TestCase):
    def archive(self, edit=None):
        """Write a source package and a matching archive with Cargo's own files, then apply `edit`."""
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        source, archive = root / 'rscrypto-rustls', root / 'rscrypto-rustls-0.1.0'
        for directory, files in ((source, {'Cargo.toml': '[package]\nname = "rscrypto-rustls"\n'}),
                                 (archive, {'Cargo.toml': '# normalized by Cargo\n', 'Cargo.lock': 'version = 4\n',
                                            '.cargo_vcs_info.json': '{}\n',
                                            'Cargo.toml.orig': '[package]\nname = "rscrypto-rustls"\n'})):
            for name, text in {**files, 'README.md': 'provider\n', 'src/lib.rs': '//! provider\n'}.items():
                (directory / name).parent.mkdir(parents=True, exist_ok=True)
                (directory / name).write_text(text)
        package.check_contents(archive, source)
        if edit:
            edit(archive)
        return archive, source

    def test_cargo_generated_files_and_the_original_manifest_are_not_failures(self):
        self.archive()

    def test_an_archive_file_from_outside_the_package_fails(self):
        archive, source = self.archive(lambda archive: (archive / 'LICENSE').write_text('root license\n'))
        with self.assertRaisesRegex(SystemExit, 'holds LICENSE, which is not rscrypto-rustls/LICENSE'):
            package.check_contents(archive, source)

    def test_an_archive_file_that_differs_from_its_source_fails(self):
        archive, source = self.archive(lambda archive: (archive / 'src/lib.rs').write_text('//! changed\n'))
        with self.assertRaisesRegex(SystemExit, 'holds src/lib.rs'):
            package.check_contents(archive, source)


class GitPatches(unittest.TestCase):
    def test_renamed_git_dependencies_in_target_tables_are_patched_and_registry_pins_are_not(self):
        manifest = tomllib.loads(
            '[dependencies]\nrustls-webpki = "=0.104.0-alpha.8"\n'
            "[target.'cfg(unix)'.dev-dependencies]\n"
            'tls = { package = "rustls", git = "https://github.com/rustls/rustls", rev = "0419f479" }\n')
        self.assertEqual(package.git_patches(manifest),
                         ['--config', 'patch.crates-io.rustls.git="https://github.com/rustls/rustls"',
                          '--config', 'patch.crates-io.rustls.rev="0419f479"'])


if __name__ == '__main__':
    unittest.main()
