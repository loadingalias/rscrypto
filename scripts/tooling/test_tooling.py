"""Offline regression checks for native provisioning."""
import copy
import hashlib
import io
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import unittest
import zipfile
from unittest.mock import patch

import catalog
import fingerprint


class Tooling(unittest.TestCase):
    def test_nextest_version_bump_requires_matching_archive(self):
        data = catalog.read()
        data['cargo']['cargo-nextest'] = '0.0.0'
        with self.assertRaisesRegex(ValueError, 'cargo-nextest archive must match'):
            catalog.validate(data)

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

    def test_catalog_profile_boundaries(self):
        data = catalog.read()
        catalog.validate(data)
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

    def test_macos_pins_only_catalog_uv_and_zig(self):
        data = catalog.read()
        self.assertEqual(set(data[catalog.MACOS]['assets']), {'uv', 'zig'})
        for name in ('uv', 'zig'):
            for field, value in (('url', 'https://example.invalid/0.0.1/archive.tar.xz'), ('sha256', 'f' * 63)):
                with self.subTest(name=name, field=field):
                    broken = copy.deepcopy(data)
                    broken[catalog.MACOS]['assets'][name][field] = value
                    with self.assertRaisesRegex(ValueError, f'verified {name}'):
                        catalog.validate(broken)
        broken = copy.deepcopy(data)
        broken[catalog.MACOS]['assets']['cmake'] = data['x86_64-linux']['assets']['cmake']
        with self.assertRaisesRegex(ValueError, 'exactly uv and zig'):
            catalog.validate(broken)


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
        with tempfile.TemporaryDirectory() as temporary, patch.object(catalog.urllib.request,'urlopen',return_value=response) as fetch:
            path=Path(temporary)/'download'
            with self.assertRaises(ValueError): catalog.download(response.url,path,'a'*64)
            self.assertFalse(path.exists())
            self.assertEqual(fetch.call_count,1,'a checksum mismatch is not retried')

    def test_download_retries_only_transient_failures_and_names_the_url(self):
        url='https://example.test/asset'
        payload=b'pinned payload'
        checksum=hashlib.sha256(payload).hexdigest()

        class Stalled(io.BytesIO):
            """Delivers one short block, then times out mid-download."""
            def read(self, size=-1):
                if self.tell(): raise TimeoutError('read timed out')
                return super().read(4)

        def response(stalls=False):
            stream=(Stalled if stalls else io.BytesIO)(payload);stream.url=url
            return stream

        def http(code): return catalog.urllib.error.HTTPError(url,code,'status',{},None)

        cases=[
            ('5xx then success',[http(504),response()],2,[2],None),
            ('network and timeout then success',[catalog.urllib.error.URLError('reset'),TimeoutError('timed out'),response()],3,[2,4],None),
            ('partial read then success',[response(stalls=True),response()],2,[2],None),
            ('5xx exhaustion',[http(504),http(502),http(503),http(504),http(500),http(504)],6,[2,4,8,16,32],'after 6 attempt'),
            ('partial reads exhaust',[response(stalls=True) for _ in range(6)],6,[2,4,8,16,32],'after 6 attempt'),
            ('4xx is not retried',[http(404),response()],1,[],'after 1 attempt'),
        ]
        for name,effects,calls,delays,failure in cases:
            with self.subTest(name), tempfile.TemporaryDirectory() as temporary, \
                    patch.object(catalog.urllib.request,'urlopen',side_effect=effects) as fetch, \
                    patch.object(catalog.time,'sleep') as sleep, patch('sys.stderr',new=io.StringIO()):
                path=Path(temporary)/'download'
                if failure is None:
                    catalog.download(url,path,checksum)
                    self.assertEqual(path.read_bytes(),payload)
                else:
                    with self.assertRaises(OSError) as raised: catalog.download(url,path,checksum)
                    self.assertIn(failure,str(raised.exception))
                    self.assertIn(url,str(raised.exception))
                    self.assertFalse(path.exists(),'no partial download survives')
                self.assertEqual(fetch.call_count,calls)
                self.assertEqual([call.args[0] for call in sleep.call_args_list],delays)



if __name__ == '__main__': unittest.main()
