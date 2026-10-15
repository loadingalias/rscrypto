"""Offline regression checks for native provisioning and release selection."""
import copy
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import shutil
import tempfile
import tomllib
import unittest
import zipfile
from unittest.mock import patch

import tomlkit
import catalog
import fingerprint
import update


class Tooling(unittest.TestCase):
    def test_windows_x64_requires_nasm(self):
        data = catalog.read()
        del data['x86_64-win']['assets']['nasm']
        with self.assertRaisesRegex(ValueError, 'missing NASM'):
            catalog.validate(data)

    def test_windows_x64_requires_native_perl(self):
        data = catalog.read()
        del data['x86_64-win']['assets']['strawberry-perl']
        with self.assertRaisesRegex(ValueError, 'missing native Perl'):
            catalog.validate(data)

    def test_strawberry_perl_release_reads_the_portable_asset_version(self):
        names = ['strawberry-perl-5.42.3.1-64bit.msi', 'strawberry-perl-5.42.3.1-64bit-portable.zip',
                 'strawberry-perl-5.42.3.1-64bit-portable.zip.sha256.txt']
        with patch.object(update, 'release', return_value={'assets': [{'name': name} for name in names]}):
            self.assertEqual(update.strawberry_perl_release(), '5.42.3.1')
        with patch.object(update, 'release', return_value={'assets': [{'name': names[0]}]}):
            with self.assertRaisesRegex(ValueError, 'no unique'):
                update.strawberry_perl_release()

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
        self.assertEqual(set(data['ci-cross-build']['assets']), set(data['ci-cross-build']['cargo']))
        self.assertNotIn('cargo-rail', data['aarch64-win']['assets'])
        for platform in ('aarch64-linux', 'x86_64-linux', 'x86_64-win'):
            self.assertIn('cargo-rail', data[platform]['assets'])
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
                             {'rustup', 'uv', 'cargo-binstall'} if platform == 'riscv64-linux' else {'rustup', 'uv'})
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
fork={version="=0.24.0-dev.1",git="https://github.com/owner/fork",rev="2e56badaef3fde0a0164b51a86ba117831f63c54"}
ahead={version="=3.0.0",git="https://github.com/owner/ahead",rev="9d729c1b6b7ad30d45d40265037eeeb9d98a7420"}
alpha="=0.104.0-alpha.7"
'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'Cargo.toml'
            path.write_text(source)
            with patch.object(update, 'ROOT', root), patch.object(update, 'crate_version', return_value='2.0.0') as version:
                update.update_manifests([path])
            result = tomlkit.parse(path.read_text())
            # A prerelease pin has no eligible release: no lookup, and no move to an older stable.
            self.assertNotIn(('fork', '1.91.0'), [call.args for call in version.call_args_list])
            self.assertEqual(result['target']['cfg(unix)']['dependencies']['fork']['version'], '=0.24.0-dev.1')
            self.assertEqual(result['target']['cfg(unix)']['dependencies']['fork']['rev'],
                             '2e56badaef3fde0a0164b51a86ba117831f63c54')
            self.assertEqual(result['target']['cfg(unix)']['dependencies']['alpha'], '=0.104.0-alpha.7')
            # A Git revision ahead of the newest registry release keeps its version and revision.
            self.assertEqual(result['target']['cfg(unix)']['dependencies']['ahead']['version'], '=3.0.0')
            self.assertEqual(result['target']['cfg(unix)']['dependencies']['ahead']['rev'],
                             '9d729c1b6b7ad30d45d40265037eeeb9d98a7420')
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

    # `just update` runs on the workstation, which requires ripgrep; CI images do not carry it.
    @unittest.skipUnless(shutil.which('rg'), 'manifest discovery uses ripgrep')
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

    def test_nightly_update_pins_newest_complete_manifest(self):
        data = catalog.read()
        hosts = [data[p]['rust-host'] for p in catalog.PLATFORMS] + ['aarch64-apple-darwin']
        targets = json.loads((catalog.ROOT / '.config/target-matrix.json').read_text())['targets']
        def manifest(complete):
            packages = {name: {'target': {h: {'available': True} for h in hosts}}
                        for name in ('rustc', 'cargo', 'clippy-preview', 'rustfmt-preview', 'rust-src',
                                     'llvm-tools-preview', 'rust-analyzer-preview', 'miri-preview')}
            packages['rust-std'] = {'target': {t: {'available': True} for t in {*hosts, *targets}}}
            if not complete:
                packages['rust-std']['target']['s390x-unknown-linux-gnu']['available'] = False
            return tomlkit.dumps({'date': 'fixture', 'pkg': packages}).encode()
        responses = [OSError('not yet published'), (manifest(False), ''), (manifest(True), '')]
        def fetch(url):
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / '.config').mkdir()
            shutil.copy2(catalog.ROOT / '.config/target-matrix.json', root / '.config/target-matrix.json')
            path = root / 'rust-toolchain.toml'
            path.write_text('[toolchain]\nchannel="nightly-2026-01-01"\ncomponents=["clippy","rustfmt","rust-src","rust-analyzer"]\n')
            with patch.object(update, 'ROOT', root), patch.object(update, 'read', return_value=data), \
                 patch.object(update, 'fetch', side_effect=fetch) as fetched:
                update.update_rust()
            urls = [c.args[0] for c in fetched.call_args_list]
            self.assertEqual(len(urls), 3)
            self.assertTrue(all(u.endswith('/channel-rust-nightly.toml') for u in urls))
            third = urls[2].split('/dist/')[1].split('/')[0]
            self.assertEqual(catalog.read(path)['toolchain']['channel'], f'nightly-{third}')

    def test_nextest_recommendation_preserves_minimum_and_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / '.config').mkdir()
            path = root / '.config/nextest.toml'
            source = ('# keep\nnextest-version = { required = "0.9.100", recommended = "0.9.143" }\n'
                      '[profile.default]\nretries = 0\n')
            path.write_text(source)
            with patch.object(update, 'ROOT', root), patch.object(update, 'read', return_value={
                    'cargo': {'cargo-nextest': '0.9.144'}}):
                update.sync_nextest_recommendation()
                self.assertEqual(path.read_text(), source.replace('recommended = "0.9.143"', 'recommended = "0.9.144"'))

    def test_update_stops_on_resolution_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = json.dumps({'packages': [{'name': 'rscrypto', 'source': None,
                                                 'manifest_path': str(root / 'Cargo.toml')}]})
            with patch.object(update, 'ROOT', root), patch.object(update.sys, 'argv', ['update.py']), \
                 patch.object(update.sys, 'platform', 'darwin'), patch.object(update, 'resolve_catalog'), \
                 patch.object(update, 'write_catalog'), patch.object(update, 'update_python'), \
                 patch.object(update, 'update_rust'), \
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


class Fingerprint(unittest.TestCase):
    PLATFORMS = ('x86_64-linux', 'aarch64-linux', 'x86_64-win')

    def test_each_platform_counts_only_its_installer_inputs(self):
        data = catalog.read()
        source = catalog.ROOT
        with tempfile.TemporaryDirectory() as temporary, patch.object(catalog, 'ROOT', Path(temporary)):
            root = Path(temporary)
            names = {name for platform in self.PLATFORMS for operation in ('dev', 'ci-compat', 'ci-package')
                     if operation in fingerprint.OPERATIONS[platform.rsplit('-', 1)[1]]
                     for name in fingerprint.inputs(platform, operation)}
            for name in names | {'Cargo.toml', 'scripts/tooling/test_tooling.py'}:
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_bytes((source / name).read_bytes())

            def current():
                return {**{platform: fingerprint.fingerprint(data, platform, 'dev') for platform in self.PLATFORMS},
                        'compat': fingerprint.fingerprint(data, 'x86_64-linux', 'ci-compat')}
            original = current()
            self.assertEqual(len(set(original.values())), len(original))
            self.assertNotEqual(fingerprint.fingerprint(data, 'x86_64-linux', 'ci'), original['x86_64-linux'])
            for platform, operation in (('x86_64-linux', 'ci-cross-run'), ('x86_64-win', 'ci-fuzz')):
                with self.assertRaisesRegex(ValueError, 'no reusable'):
                    fingerprint.fingerprint(data, platform, operation)
            linux = {'x86_64-linux', 'aarch64-linux', 'compat'}
            everything = set(original)
            manifest = (root / 'Cargo.toml').read_text()
            msrv = tomllib.loads(manifest)['package']['rust-version']
            for name, before, after, changed in [
                    ('scripts/tooling/linux.sh', None, None, linux),
                    ('scripts/tooling/uv.sh', None, None, linux),
                    ('scripts/tooling/x86_64-linux.sh', None, None, {'x86_64-linux', 'compat'}),
                    ('scripts/tooling/windows.ps1', None, None, {'x86_64-win'}),
                    ('scripts/lib/toolchain.py', None, None, everything),
                    ('rust-toolchain.toml', None, None, everything),
                    ('.python-version', None, None, everything),
                    ('scripts/check/compat.py', None, None, {'compat'}),
                    ('Cargo.toml', '[package]', '[package]\n# changed', set()),
                    ('Cargo.toml', f'rust-version = "{msrv}"', 'rust-version = "1.0"', {'compat'}),
                    ('scripts/tooling/test_tooling.py', None, None, set())]:
                with self.subTest(name=name, change=after):
                    path = root / name
                    original_bytes = path.read_bytes()
                    if before is None:
                        path.write_bytes(original_bytes + b'\n# changed\n')
                    else:
                        self.assertIn(before, original_bytes.decode())
                        path.write_text(original_bytes.decode().replace(before, after, 1))
                    updated = current()
                    self.assertEqual({key for key in original if updated[key] != original[key]}, changed)
                    path.write_bytes(original_bytes)
            self.assertEqual(current(), original)

    def test_inputs_cover_every_script_an_installer_reads(self):
        # Only cross-build and cross-run installations, which never reuse a fingerprint, read these.
        cross = {'scripts/tooling/transfer.py', 'scripts/tooling/perf-riscv-mapping-symbols.patch',
                 'scripts/lib/cross_build.py'}
        scripts = {path.relative_to(catalog.ROOT).as_posix()
                   for directory in ('scripts/tooling', 'scripts/lib', 'scripts/check')
                   for path in (catalog.ROOT / directory).iterdir() if path.is_file()}
        for platform in ('x86_64-linux', 'aarch64-linux', 'riscv64-linux', 'x86_64-win', 'aarch64-win'):
            family = platform.rsplit('-', 1)[1]
            inputs = {name for operation in fingerprint.OPERATIONS[family]
                      for name in fingerprint.inputs(platform, operation)}
            # Shell and PowerShell installers run their helpers; the Python helpers
            # import nothing beyond what the inputs already name.
            text = '\n'.join((catalog.ROOT / name).read_text() for name in inputs
                             if name.endswith(('.sh', '.ps1')))
            for name in scripts:
                if re.search(r'(?<![\w.-])' + re.escape(Path(name).name) + r'\b', text):
                    self.assertIn(name, inputs | cross, f'{platform} reads {name}')


class ActionPins(unittest.TestCase):
    def test_version_annotations_follow_resolved_release(self):
        for annotation in ('# v1.0.0', '# v1.0.0  pinned for CI'):
            with self.subTest(annotation=annotation):
                source = f'steps:\n  - uses: "owner/action@{"b" * 40}" {annotation}\n'
                with patch.object(update, 'release', return_value={'tag_name': 'v2.0.0'}), patch.object(update, 'api', return_value={'sha': 'a' * 40}):
                    result = update.action_edits(source)
                    self.assertEqual(result, f'steps:\n  - uses: "owner/action@{"a" * 40}" {annotation.replace("v1.0.0", "v2.0.0")}\n')
                    self.assertEqual(update.action_edits(result), result)

    def test_only_yaml_action_values_change(self):
        source = '''jobs:
  test:
    steps:
      - uses: owner/action/subpath@v1 # keep this comment
      - uses: owner/action@v1 # 2 workers
      - uses: ./local
      - uses: docker://ubuntu:26.04
      - run: |
          uses: this/is-shell@v1
'''
        with patch.object(update, 'release', return_value={'tag_name':'v2.0.0'}), patch.object(update, 'api', return_value={'sha':'a'*40}):
            result = update.action_edits(source)
        self.assertIn('owner/action/subpath@'+'a'*40, result)
        self.assertIn('# keep this comment', result)
        self.assertIn('# 2 workers', result)
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
