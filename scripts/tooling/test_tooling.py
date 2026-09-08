"""Offline regression checks for native provisioning and release selection."""
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import tomlkit
import catalog
import update


class Tooling(unittest.TestCase):
    def test_windows_x64_requires_nasm(self):
        data = catalog.read()
        del data['x86_64-win']['assets']['nasm']
        with self.assertRaisesRegex(ValueError, 'missing NASM'):
            catalog.validate(data)

    def test_nasm_release_excludes_prereleases_and_sorts_numerically(self):
        index = b'<a href="3.02/">3.02</a><a href="3.10/">3.10</a><a href="4.00rc1/">rc</a>'
        with patch.object(update, 'fetch', return_value=(index, '')):
            self.assertEqual(update.nasm_release(), '3.10')
        with patch.object(update, 'fetch', return_value=(b'<a href="4.00rc1/">rc</a>', '')):
            with self.assertRaisesRegex(ValueError, 'no stable NASM'):
                update.nasm_release()

    def test_catalog_roundtrip_and_profile_boundaries(self):
        data = catalog.read()
        catalog.validate(data)
        self.assertEqual(tomlkit.parse(update.catalog_text(data)), data)
        for platform in catalog.PLATFORMS:
            broken = copy.deepcopy(data)
            if platform in catalog.PROFILING_PLATFORMS:
                broken[platform]['cargo'].remove('samply')
            else:
                broken[platform]['cargo'].append('samply')
            with self.assertRaises(ValueError):
                catalog.validate(broken)
        for platform in catalog.NATIVE_SOURCE_PLATFORMS:
            self.assertEqual(set(data[platform]['assets']),
                             {'rustup', 'cargo-binstall'} if platform == 'riscv64-linux' else {'rustup'})
            self.assertEqual(data[platform]['components'], [])

    def test_stable_release_selection_respects_msrv_and_yanks(self):
        versions = [{'num': '2.0.0', 'created_at': '2099-01-01T00:00:00Z', 'rust_version': '1.98'},
                    {'num': '3.0.0', 'yanked': True}, {'num': '4.0.0-rc.1'},
                    {'num': '1.0.0', 'rust_version': '1.80'}]
        self.assertEqual(update.eligible_release(versions), '2.0.0')
        self.assertEqual(update.eligible_release(versions, rust_version='1.91.0'), '1.0.0')
        with self.assertRaises(ValueError):
            update.eligible_release([{'num': '1.0.0', 'yanked': True}])

    def test_manifest_updates_preserve_structure_and_cover_dependency_tables(self):
        source = '''# preserve
[package]
name="fixture"
version="0.1.0"
rust-version="1.91.0"
[dependencies]
alias={package="dep",version="=1",features=["std"],default-features=false} # keep
local={path="../local",version="1"}
inherited.workspace=true
[workspace.dependencies]
dep="1"
[dev-dependencies]
rscrypto="1"
[build-dependencies]
cargo-rail="1"
[target.'cfg(unix)'.dependencies]
dep="1"
'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'Cargo.toml'
            path.write_text(source)
            with patch.object(update, 'ROOT', root), patch.object(update, 'crate_version', return_value='2.0.0'):
                update.update_manifests([path])
            result = tomlkit.parse(path.read_text())
            self.assertEqual(result['dependencies']['alias']['version'], '=1')
            self.assertEqual(result['dependencies']['alias']['features'], ['std'])
            self.assertFalse(result['dependencies']['alias']['default-features'])
            self.assertEqual(result['dependencies']['local']['version'], '1')
            self.assertTrue(result['dependencies']['inherited']['workspace'])
            self.assertEqual(result['workspace']['dependencies']['dep'], '2.0.0')
            self.assertEqual(result['dev-dependencies']['rscrypto'], '2.0.0')
            self.assertEqual(result['build-dependencies']['cargo-rail'], '2.0.0')
            self.assertEqual(result['target']['cfg(unix)']['dependencies']['dep'], '2.0.0')
            self.assertIn('# keep', path.read_text())
            self.assertIn('# preserve', path.read_text())

    def test_compatibility_alias_does_not_collapse_into_current_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'Cargo.toml'
            path.write_text('[dependencies]\nsha2="0.11.0"\n'
                            'sha2_010={package="sha2",version="0.10.9"}\n')
            with patch.object(update, 'ROOT', root), patch.object(update, 'api', return_value={
                    'versions': [{'num': '0.11.1'}, {'num': '0.10.9'}]}):
                update.update_manifests([path])
            dependencies = catalog.read(path)['dependencies']
            self.assertEqual(dependencies['sha2'], '0.11.1')
            self.assertEqual(dependencies['sha2_010']['version'], '0.10.9')

    def test_failed_lookup_preserves_all_manifests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / 'one.toml', root / 'two.toml']
            originals = ['[dependencies]\none="1"\n', '[dependencies]\ntwo="1"\n']
            for path, text in zip(paths, originals):
                path.write_text(text)
            def select(name, *_):
                if name == 'two':
                    raise ValueError('release lookup failed')
                return '2.0.0'
            with patch.object(update, 'crate_version', side_effect=select):
                with self.assertRaises(ValueError):
                    update.update_manifests(paths)
            self.assertEqual([p.read_text() for p in paths], originals)

    def test_discovers_support_untracked_and_lockless_workspaces(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            manifests = []
            for name in ('', 'fuzz/support', 'tools/lockless', 'fuzz-packages/new'):
                directory = root / name
                (directory / 'src').mkdir(parents=True)
                (directory / 'src/lib.rs').write_text('')
                path = directory / 'Cargo.toml'
                path.write_text('[package]\nname="fixture"\nversion="0.1.0"\n[workspace]\n')
                manifests.append(path)
            (root / 'target').mkdir()
            (root / 'target/Cargo.toml').write_text('generated')
            (root / 'tools/lockless/vendor/dependency').mkdir(parents=True)
            (root / 'tools/lockless/vendor/dependency/Cargo.toml').write_text('upstream')
            with patch.object(update, 'ROOT', root):
                self.assertEqual(set(update.manifest_paths()), set(manifests))
                self.assertEqual(set(update.cargo_roots(manifests)), set(manifests))

    def test_runner_tracks_resolved_gungraun(self):
        data = catalog.read()
        def read(path=None):
            return {'package': [{'name': 'gungraun', 'version': '0.19.4'}]} if path else data
        with patch.object(update, 'read', side_effect=read), patch.object(update, 'write_catalog') as write:
            update.sync_gungraun_runner()
        self.assertEqual(write.call_args.args[0]['cargo']['gungraun-runner'], '0.19.4')

    def test_stable_update_checks_only_selected_native_components(self):
        data = catalog.read()
        hosts = [data[p]['rust-host'] for p in catalog.PLATFORMS] + ['aarch64-apple-darwin']
        packages = {name: {'target': {h: {'available': True} for h in hosts}}
                    for name in ('rustc', 'cargo', 'rust-std', 'clippy-preview', 'rustfmt-preview',
                                 'rust-src', 'llvm-tools-preview', 'rust-analyzer-preview')}
        packages['rust'] = {'version': '1.98.1 (fixture)'}
        manifest = tomlkit.dumps({'date': '2026-09-03', 'pkg': packages}).encode()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'rust-toolchain.toml'
            path.write_text('[toolchain]\nchannel="1.98.0"\ncomponents=["clippy","rustfmt","rust-src","rust-analyzer"]\n')
            with patch.object(update, 'ROOT', root), patch.object(update, 'read', return_value=data), \
                 patch.object(update, 'fetch', return_value=(manifest, '')) as fetch:
                update.update_rust()
            fetch.assert_called_once_with('https://static.rust-lang.org/dist/channel-rust-stable.toml')
            self.assertEqual(catalog.read(path)['toolchain']['channel'], '1.98.1')

    def test_update_stops_on_resolution_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = json.dumps({'packages': [{'name': 'rscrypto', 'source': None,
                                                 'manifest_path': str(root / 'Cargo.toml')}]})
            with patch.object(update, 'ROOT', root), patch.object(update.sys, 'argv', ['update.py']), \
                 patch.object(update.sys, 'platform', 'darwin'), patch.object(update, 'resolve_catalog'), \
                 patch.object(update, 'write_catalog'), patch.object(update, 'update_rust'), \
                 patch.object(update, 'rust_channel', return_value='1.98.1'), \
                 patch.object(update, 'manifest_paths', return_value=[root / 'Cargo.toml']), \
                 patch.object(update, 'cargo_roots', return_value=[root / 'Cargo.toml']), \
                 patch.object(update, 'update_manifests'), \
                 patch.object(update.subprocess, 'check_output', return_value=metadata), \
                 patch.object(update, 'update_actions') as actions:
                def run(command, **kwargs):
                    if command[0] == 'cargo':
                        self.assertEqual(command[3], 'update')
                        self.assertEqual(catalog.read(command[2])['patch']['crates-io'],
                                         {'rscrypto': {'path': str(root)}})
                        raise subprocess.CalledProcessError(1, command)
                with patch.object(update.subprocess, 'run', side_effect=run):
                    with self.assertRaises(subprocess.CalledProcessError):
                        update.main()
                actions.assert_not_called()


class ActionPins(unittest.TestCase):
    def test_only_yaml_action_values_change(self):
        source = '''jobs:
  test:
    steps:
      - uses: owner/action/subpath@v1 # keep this comment
      - uses: ./local
      - uses: docker://ubuntu:26.04
      - run: |
          uses: this/is-shell@v1
'''
        with patch.object(update, 'release', return_value={'tag_name':'v2.0.0'}), patch.object(update, 'api', return_value={'sha':'a'*40}):
            result = update.action_edits(source)
        self.assertIn('owner/action/subpath@'+'a'*40, result)
        self.assertIn('# keep this comment', result)
        self.assertIn('uses: ./local', result)
        self.assertIn('uses: docker://ubuntu:26.04', result)
        self.assertIn('uses: this/is-shell@v1', result)

    def test_branch_only_actions_keep_their_update_reference(self):
        source = 'steps:\n  - uses: owner/action@stable\n'
        with patch.object(update, 'release', side_effect=ValueError('HTTP 404')), patch.object(update, 'api', return_value={'sha':'a'*40}) as lookup:
            result = update.action_edits(source)
            self.assertIn('# stable', result)
            update.action_edits(result)
            self.assertTrue(all(call.args[0].endswith('/commits/stable') for call in lookup.call_args_list))


class Archives(unittest.TestCase):
    def test_single_archive_cli_preserves_paths_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix='rscrypto tooling ') as temporary:
            prefix = Path(temporary)
            asset = catalog.read()['x86_64-win']['assets']['llvm']
            destination = prefix / 'llvm' / asset['sha256'][:16]
            destination.mkdir(parents=True)
            (destination / '.rscrypto-installed').write_text(asset['sha256'] + '\n')
            result = subprocess.run(
                [sys.executable, str(Path(catalog.__file__)), 'install-archive',
                 'x86_64-win', 'llvm', str(prefix)],
                capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), str(destination))

    def test_extraction_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); archive=root/'archive.tar'
            with tarfile.open(archive,'w') as bundle:
                info=tarfile.TarInfo('../escape');info.size=4
                bundle.addfile(info,io.BytesIO(b'fail'))
            with self.assertRaises(tarfile.TarError): catalog.unpack(archive,root/'output')
            self.assertFalse((root/'escape').exists())

    def test_complete_tool_layout_survives_install_and_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);archive=root/'archive.tar'
            with tarfile.open(archive,'w') as bundle:
                for name,content in [('tool/bin/compiler',b'compiler'),('tool/lib/runtime',b'runtime')]:
                    info=tarfile.TarInfo(name);info.size=len(content);info.mode=0o755
                    bundle.addfile(info,io.BytesIO(content))
            asset={'url':'https://example.test/tool.tar','sha256':'a'*64}
            def download(url,path,checksum): Path(path).write_bytes(archive.read_bytes())
            with patch.object(catalog,'download',side_effect=download) as fetch:
                directory=catalog.install_archive('tool',asset,root/'prefix')
                self.assertEqual((directory/'lib/runtime').read_bytes(),b'runtime')
                self.assertEqual(catalog.install_archive('tool',asset,root/'prefix'),directory)
                self.assertEqual(fetch.call_count,1)

    def test_zip_extraction_preserves_executable_permissions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'tool.zip'
            info = zipfile.ZipInfo('tool')
            info.external_attr = 0o100755 << 16
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr(info, b'#!/bin/sh\nexit 0\n')
            catalog.unpack(archive, root / 'output')
            self.assertEqual(subprocess.run([str(root / 'output/tool')]).returncode, 0)

    def test_bad_checksum_does_not_publish_download(self):
        response=io.BytesIO(b'bad payload');response.url='https://example.test/asset'
        with tempfile.TemporaryDirectory() as temporary, patch.object(catalog.urllib.request,'urlopen',return_value=response):
            path=Path(temporary)/'download'
            with self.assertRaises(ValueError): catalog.download(response.url,path,'a'*64)
            self.assertFalse(path.exists())



if __name__ == '__main__': unittest.main()
