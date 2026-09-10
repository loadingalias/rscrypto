#!/usr/bin/env python3
"""Reject mixed-source, corrupt, incomplete, and unsafe transferred evidence."""

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/lib'), str(ROOT / 'scripts/ct'), str(ROOT / 'scripts/test')]
import evidence_bundle as bundle
import doctest_bundle
import dudect_execute
import transfer
import riscv
from riscv_build import environment


class Bundles(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        (self.root / '.gitignore').write_text('target/\n')
        (self.root / 'source.rs').write_text('original\n')
        subprocess.run(['git', 'add', '.'], cwd=self.root, check=True)
        subprocess.run(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                        'commit', '-qm', 'fixture'], cwd=self.root, check=True)
        self.out = self.root / 'target/bundle'
        self.out.mkdir(parents=True)
        (self.out / 'program').write_bytes(b'program')
        (self.out / 'program').chmod(0o755)
        self.identity = bundle.source_identity(self.root)

    def seal(self):
        return bundle.seal(self.root, self.out, 'test', 'target', self.identity, {})

    def test_round_trip_and_corruption(self):
        self.seal()
        archive = self.root / 'target/test.tar.gz'
        bundle.pack(self.out, archive)
        incoming = self.root / 'target/incoming'
        bundle.unpack(archive, incoming)
        bundle.verify(self.root, incoming, 'test', 'target')
        self.assertTrue((incoming / 'program').stat().st_mode & 0o111)
        for mutation in ('content', 'missing', 'extra', 'mode'):
            with self.subTest(mutation=mutation):
                path = incoming / 'program'
                if mutation == 'content': path.write_bytes(b'changed')
                elif mutation == 'missing': path.unlink()
                elif mutation == 'extra': (incoming / 'extra').touch()
                else: path.chmod(0o644)
                with self.assertRaises(ValueError): bundle.verify(self.root, incoming, 'test', 'target')
                shutil.rmtree(incoming)
                bundle.unpack(archive, incoming)

    def test_source_and_target_binding(self):
        self.seal()
        with self.assertRaises(ValueError): bundle.verify(self.root, self.out, 'test', 'other')
        (self.root / 'source.rs').write_text('edited\n')
        with self.assertRaises(ValueError): bundle.verify(self.root, self.out, 'test', 'target')
        with self.assertRaises(ValueError): self.seal()

    def test_archive_rejects_traversal_links_and_duplicates(self):
        for index, names in enumerate((['../escape'], ['/absolute'], ['a', 'a'], ['link'], ['a\\b'])):
            archive = self.root / f'target/bad-{index}.tar.gz'
            with tarfile.open(archive, 'w:gz') as output:
                for name in names:
                    member = tarfile.TarInfo(name)
                    if name == 'link': member.type, member.linkname = tarfile.SYMTYPE, '/tmp'
                    output.addfile(member, io.BytesIO())
            with self.assertRaises(ValueError): bundle.unpack(archive, self.root / f'target/unpack-{index}')

    def test_ct_transfer_relocates_but_preserves_producer_identity(self):
        ct = self.root / 'target/ct/riscv64gc-unknown-linux-gnu/release'
        (ct / 'artifacts').mkdir(parents=True)
        (ct / 'artifacts/evidence').write_bytes(b'exact code')
        logs = ct / 'full/logs'; logs.mkdir(parents=True)
        steps = []
        for name in transfer.GATES:
            path = logs / (name + '.log'); path.write_text('passed')
            steps.append({'name': name, 'status': 'pass', 'stdout': str(path), 'stderr': str(path)})
        for name in ('evidence-index.json', 'artifact-hashes.txt', 'asm-heuristics.json',
                     'asm-heuristics.md', 'zeroization.json'):
            (ct / name).write_text('{}')
        (ct / 'provenance.json').write_text(json.dumps({'target': transfer.TARGET, 'profile': 'release'}))
        shared = ct / 'build/shared'; shared.mkdir(parents=True)
        metadata = {'target': transfer.TARGET, 'profile': 'release', 'host': {'machine': 'x86_64'}}
        for key in ('binary', 'binary_disassembly', 'binary_symbols', 'linker_command_log'):
            path = shared / key
            path.write_bytes(b'\x7fELF\x02\x01' + bytes(12) + b'\xf3\x00' if key == 'binary' else b'evidence')
            metadata[key] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': bundle.digest(path)}
        (shared / 'prepared.json').write_text(json.dumps({'metadata': metadata, 'manifest_cases': {}}))
        archive = self.root / 'target/ct.tar.gz'
        transfer.export(self.root, ct, shared, steps, self.identity, archive)
        with patch('platform.system', return_value='Linux'), patch('platform.machine', return_value='riscv64'):
            imported_steps, path = transfer.consume(self.root, ct, archive)
            prepared = json.loads(path.read_text())
            self.assertEqual(prepared['metadata']['build_host']['machine'], 'x86_64')
            self.assertEqual(prepared['metadata']['host']['machine'], 'riscv64')
            self.assertTrue(all(Path(s['stdout']).is_file() for s in imported_steps))
            dudect_execute.verify_transferred(prepared)
            binary = Path(prepared['metadata']['binary']['path'])
            binary.write_bytes(b'changed executable')
            with self.assertRaises(ValueError): dudect_execute.verify_transferred(prepared)
        with patch('platform.machine', return_value='x86_64'):
            with self.assertRaises(ValueError): transfer.consume(self.root, ct, archive)
        with self.assertRaises(ValueError): transfer.export(self.root, ct, shared, steps[:-1], self.identity, archive)

    def test_overrides_are_rejected(self):
        for key in ('RUSTFLAGS', 'CARGO_PROFILE_RELEASE_LTO', 'NEXTEST_FILTERSET'):
            with patch.dict(os.environ, {key: 'override'}):
                with self.assertRaises(ValueError): environment()

    def test_prepare_keeps_both_full_release_modes_and_fails_closed(self):
        for name in ('Cargo.toml', '.config/tooling.toml'):
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        commands = self.root / 'commands'; commands.mkdir()
        log = self.root / 'target/commands.jsonl'
        tool = commands / 'tool'
        tool.write_text('#!' + sys.executable + '''
import json,os,sys
from pathlib import Path
name=Path(sys.argv[0]).name
args=sys.argv[1:]
with Path(os.environ['TRANSFER_TEST_LOG']).open('a') as log:
    log.write(json.dumps([name,*args])+'\\n')
if name=='cargo' and args==['nextest','--version']:
    print('cargo-nextest '+os.environ['TRANSFER_NEXTEST'])
elif name=='cargo':
    if os.environ.get('TRANSFER_FAIL'): sys.exit(23)
    Path(args[args.index('--archive-file')+1]).write_bytes(b'archive fixture')
elif name=='rustc': print('rustc pinned fixture')
elif name.endswith('gcc'): print('gcc pinned fixture')
''')
        tool.chmod(0o755)
        for name in ('just', 'cargo', 'rustc', 'riscv64-linux-gnu-gcc'):
            (commands / name).symlink_to(tool)
        # Do not include fixture executable symlinks in the effective source digest.
        with (self.root / '.gitignore').open('a') as output: output.write('commands/\n')
        pin = __import__('tomllib').loads((ROOT / '.config/tooling.toml').read_text())['cargo']['cargo-nextest']
        def docs(root, directory, args, env):
            directory.mkdir()
            (directory / 'fixture').write_text('prepared')
            return {'total': 232}
        with patch.object(riscv, 'ROOT', self.root), patch('platform.system', return_value='Linux'), \
             patch('platform.machine', return_value='x86_64'), patch.object(doctest_bundle, 'prepare', side_effect=docs) as prepare_docs, \
             patch.dict(os.environ, PATH=str(commands) + os.pathsep + os.environ['PATH'],
                        TRANSFER_TEST_LOG=str(log), TRANSFER_NEXTEST=pin):
            archive = self.root / 'target/test-suites.tar.gz'
            riscv.prepare(archive)
            calls = [json.loads(row) for row in log.read_text().splitlines()]
            self.assertIn(['just', 'ci-check-target', riscv.TARGET], calls)
            builds = [row for row in calls if row[:3] == ['cargo', 'nextest', 'archive']]
            self.assertEqual(len(builds), 2)
            self.assertTrue(all('--workspace' in row and '--locked' in row and '--release' in row for row in builds))
            self.assertTrue(all(row[row.index('--target') + 1] == riscv.TARGET for row in builds))
            self.assertNotIn('portable-only', builds[0][builds[0].index('--features') + 1].split(','))
            self.assertIn('--all-features', builds[1])
            self.assertEqual(prepare_docs.call_count, 2)
            with patch.dict(os.environ, TRANSFER_FAIL='1'):
                failed = self.root / 'target/failed.tar.gz'
                with self.assertRaises(subprocess.CalledProcessError): riscv.prepare(failed)
                self.assertFalse(failed.exists())


class Doctests(unittest.TestCase):
    def test_pinned_rustdoc_compile_and_native_execution_contract(self):
        # A small independent fixture exercises rustdoc itself, including negative
        # compilation, no_run, expected panic, and a failing runtime assertion.
        channel = __import__('toolchain').contracts()['nightly']
        with tempfile.TemporaryDirectory(prefix='rscrypto doctest transfer ') as temporary:
            root = Path(temporary)
            for failing in (False, True):
                directory = root / str(failing); directory.mkdir()
                source = directory / 'lib.rs'
                source.write_text('//! ```\n//! assert!(' + str(not failing).lower() + ');\n//! ```\n'
                    '//! ```should_panic\n//! panic!("expected");\n//! ```\n'
                    '//! ```compile_fail,E0308\n//! let x: u8 = "no";\n//! ```\n'
                    '//! ```no_run\n//! panic!("must never execute");\n//! ```\n')
                (directory / 'Cargo.toml').write_text('[package]\nname="doctest-transfer-fixture"\nversion="0.0.0"\nedition="2024"\n[lib]\npath="lib.rs"\n[workspace]\n')
                env = {**os.environ, 'RUSTUP_TOOLCHAIN': channel, 'CARGO_RAIL_CACHE': 'off'}
                subprocess.run(['cargo', 'generate-lockfile', '--offline'], cwd=directory, env=env, check=True,
                               capture_output=True)
                evidence = directory / 'evidence'
                plan = doctest_bundle.prepare(directory, evidence, [], env)
                inventory = json.loads((evidence / 'inventory.json').read_text())
                if failing:
                    with self.assertRaises(ValueError): doctest_bundle.execute(directory, evidence, directory / 'results')
                else:
                    summary = doctest_bundle.execute(directory, evidence, directory / 'results')
                    self.assertEqual(summary, {'status': 'pass', 'total': 4, 'compile_fail': 1, 'no_run': 1, 'executed': 2})
                    incomplete = {**plan, 'programs': plan['programs'][:-1]}
                    (evidence / 'plan.json').write_text(json.dumps(incomplete))
                    with self.assertRaises(ValueError): doctest_bundle.execute(directory, evidence, directory / 'incomplete')
                    (evidence / 'plan.json').write_text(json.dumps(plan))
                    (evidence / 'programs' / plan['programs'][0]['binary']).unlink()
                    with self.assertRaises(ValueError): doctest_bundle.plan_tests(inventory, evidence / 'programs')


class NativeArchive(unittest.TestCase):
    def test_real_nextest_archive_executes_after_transfer(self):
        # Test the complete consumer locally with a tiny host-native archive.
        # Only host identification is substituted; this is not RISC-V evidence.
        channel = __import__('toolchain').contracts()['nightly']
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'Cargo.toml').write_text('[package]\nname="archive-fixture"\nversion="0.0.0"\nedition="2024"\n'
                '[lib]\npath="lib.rs"\n[workspace]\n')
            (root / 'lib.rs').write_text('//! ```\n//! assert_eq!(2 + 2, 4);\n//! ```\n'
                '#[test] fn reads_fixture() { assert_eq!(std::fs::read_to_string("fixture.txt").unwrap(), "oracle"); }\n')
            (root / 'fixture.txt').write_text('oracle')
            (root / '.gitignore').write_text('target/\n')
            for name in ('.config/nextest.toml', '.config/tooling.toml'):
                path = root / name; path.parent.mkdir(exist_ok=True)
                shutil.copy2(ROOT / name, path)
            # The tiny fixture has no RSA/AEAD integration binaries to which the
            # product's timeout overrides apply. Retain its exact runner pin.
            config = __import__('tomllib').loads((ROOT / '.config/nextest.toml').read_text())
            (root / '.config/nextest.toml').write_text('nextest-version = ' +
                '{ required = "' + config['nextest-version']['required'] + '" }\n')
            env = {**os.environ, 'RUSTUP_TOOLCHAIN': channel, 'CARGO_RAIL_CACHE': 'off'}
            subprocess.run(['cargo', 'generate-lockfile', '--offline'], cwd=root, env=env, check=True, capture_output=True)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            subprocess.run(['git', 'add', '.'], cwd=root, check=True)
            subprocess.run(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                            'commit', '-qm', 'fixture'], cwd=root, check=True)
            directory = root / 'target/transfer'; directory.mkdir(parents=True)
            metadata = {'nextest': riscv.nextest_version(), 'modes': {}}
            for mode in ('native', 'portable'):
                built = subprocess.run(['cargo', 'nextest', 'archive', '--locked', '--workspace', '--release',
                                '--archive-file', str(directory / (mode + '.tar.zst'))], cwd=root, env=env,
                               capture_output=True, text=True)
                self.assertEqual(built.returncode, 0, built.stderr)
                plan = doctest_bundle.prepare(root, directory / (mode + '-docs'), [], env)
                metadata['modes'][mode] = {'doctests': plan['total']}
            bundle.seal(root, directory, 'rscrypto.riscv.tests', riscv.TARGET, bundle.source_identity(root), metadata)
            archive = root / 'target/transfer.tar.gz'; bundle.pack(directory, archive)
            # Removing the original build tree catches hidden dependencies on it.
            shutil.rmtree(root / 'target/release')
            with patch.object(riscv, 'ROOT', root), patch('platform.system', return_value='Linux'), \
                 patch('platform.machine', return_value='riscv64'), patch('sys.stdout', new=io.StringIO()):
                riscv.execute(archive)
            summaries = list((root / 'target/riscv-results').glob('*/summary.json'))
            self.assertEqual(len(summaries), 1)
            self.assertEqual(json.loads(summaries[0].read_text())['status'], 'pass')


if __name__ == '__main__':
    unittest.main()
