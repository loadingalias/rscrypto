#!/usr/bin/env -S uv run python
"""Manual selection must neither launch extra machines nor broaden measured work."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci

ROOT = Path(__file__).resolve().parents[2]


def request(benchmarks='sha256', **values):
    return {'DISPATCH_INPUTS': json.dumps({'benchmarks': benchmarks, 'x86_64-linux': True, **values})}


def plan(benchmarks='sha256', **values):
    measurements, builds, _ = ci.plan(ci.dispatch_inputs(request(benchmarks, **values)), '42')
    return measurements['include'], builds['include']


class ManualBench(unittest.TestCase):
    def test_rescore_requires_the_complete_successful_native_attempt(self):
        source = {'path': '.github/workflows/bench.yml', 'event': 'workflow_dispatch',
                  'repository': {'full_name': 'owner/repo'}, 'head_branch': 'main', 'status': 'completed',
                  'run_attempt': 1, 'head_sha': 'a' * 40, 'html_url': 'https://github.com/owner/repo/actions/runs/42'}
        jobs = [{'name': name, 'conclusion': conclusion} for name, conclusion in (
            ('Plan', 'success'), ('Measure s390x-linux / auth', 'success'),
            ('Measure x86_64-linux-intel / hashes', 'success'), ('Score comparisons', 'failure'))]
        api = Mock(repository='owner/repo')
        def prepare(run=source, rows=jobs):
            api.request.side_effect = [run, {'jobs': rows[:2], 'total_count': len(rows)},
                                      {'jobs': rows[2:], 'total_count': len(rows)}]
            return ci.rescore_source('42/1', api)
        result = prepare()
        self.assertEqual(result['artifacts'], ['bench-s390x-linux-auth-42-1.tar.gz',
                                               'bench-x86_64-linux-intel-hashes-42-1.tar.gz'])
        self.assertEqual(result['revision'], 'a' * 40)
        self.assertEqual(api.request.call_args.args[1], '/actions/runs/42/attempts/1/jobs?per_page=100&page=2')
        for field, value in (('status', 'in_progress'), ('path', '.github/workflows/ct.yml'),
                             ('head_branch', 'other'), ('run_attempt', 2),
                             ('repository', {'full_name': 'other/repo'})):
            with self.subTest(field=field), self.assertRaises(ValueError):
                prepare(source | {field: value})
        for conclusion in ('failure', 'cancelled', 'skipped', None):
            with self.subTest(conclusion=conclusion), self.assertRaisesRegex(ValueError, 'native measurement'):
                prepare(rows=[jobs[0], jobs[1] | {'conclusion': conclusion}, *jobs[2:]])
        with self.assertRaisesRegex(ValueError, 'only the source comparison report'):
            prepare(rows=[jobs[0] | {'conclusion': 'failure'}, *jobs[1:]])
        with self.assertRaisesRegex(ValueError, 'native measurement'):
            prepare(rows=[jobs[0], jobs[-1]])
        for value in ('42', '0/1', '42/0', '42/1/2', '42/1\n', '../1', '42/1;true'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ci.rescore_source(value, api)

    def test_rescore_requires_every_artifact_and_preserves_source_identity(self):
        source = {'run_id': '42', 'attempt': '1', 'revision': 'a' * 40, 'url': 'https://example.test/run/42',
                  'artifacts': ['bench-s390x-linux-auth-42-1.tar.gz', 'bench-x86_64-linux-intel-hashes-42-1.tar.gz']}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'GITHUB_SHA': 'b' * 40}):
            root = Path(directory)
            output = root / 'report.json'
            for name in source['artifacts']:
                (root / name).write_bytes(b'fixture')
            def score(directory, output, inputs, **kwargs):
                output.write_text('{"mode":"quick"}')
                return 0
            with patch.object(ci, 'score', side_effect=score) as scoring:
                self.assertEqual(ci.rescore(root, output, source), 0)
                scoring.assert_called_once_with(root, output, {}, source=('42', '1'))
                report = json.loads(output.read_text())
                self.assertEqual(report['measurement_source'], source)
                self.assertEqual(report['report_revision'], 'b' * 40)
                (root / source['artifacts'][0]).unlink()
                with self.assertRaisesRegex(ValueError, 'every successful source measurement job'):
                    ci.rescore(root, output, source)
                (root / source['artifacts'][0]).write_bytes(b'fixture')
                (root / 'bench-s390x-linux-auth-42-2.tar.gz').write_bytes(b'wrong attempt')
                with self.assertRaisesRegex(ValueError, 'every successful source measurement job'):
                    ci.rescore(root, output, source)
                self.assertEqual(scoring.call_count, 1)

    def test_one_many_all_platforms(self):
        self.assertEqual([r['platform'] for r in ci.platforms('riscv64-linux', '42')['include']], ['riscv64-linux'])
        rows = ci.platforms('s390x-linux, x86_64-win s390x-linux', '42')['include']
        self.assertEqual([r['platform'] for r in rows], ['s390x-linux', 'x86_64-win', 'x86_64-win'])
        self.assertIn('measure-x86_64-win-intel', rows[1]['runner'])
        self.assertIn('measure-x86_64-win-amd', rows[2]['runner'])
        self.assertNotEqual(rows[1]['name'], rows[2]['name'])
        self.assertEqual(len(ci.platforms('all', '42')['include']), 8)
        for bad in ('', 'all,x86_64-linux', 'apple-arm64', 'x86_64-linux,typo'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ci.platforms(bad, '42')

    def test_checkboxes_select_platforms(self):
        measurements, _ = plan(**{'x86_64-linux': False, 'aarch64-linux': True, 's390x-linux': True})
        self.assertEqual([row['name'] for row in measurements], ['aarch64-linux', 's390x-linux'])
        with self.assertRaisesRegex(ValueError, 'at least one platform'):
            plan(**{'x86_64-linux': False})
        everything = {name: True for name in ci.PLATFORMS}
        self.assertEqual(len(plan(**everything)[0]), 8)

    def test_one_job_per_platform_and_family(self):
        measurements, builds = plan('sha256,sha512,blake3,crc64', **{'riscv64-linux': True})
        jobs = [(row['name'], row['family']) for row in measurements]
        self.assertEqual(jobs, [(name, family) for family in ('hashes', 'checksum')
                                for name in ('x86_64-linux-intel', 'x86_64-linux-amd', 'riscv64-linux')])
        self.assertEqual([(row['target'], row['family'], row['timeout']) for row in builds],
                         [('riscv64gc-unknown-linux-gnu', 'hashes', 40), ('riscv64gc-unknown-linux-gnu', 'checksum', 30)])
        # Each further target adds its ten-minute budget to the job.
        self.assertEqual([row['timeout'] for row in measurements], [130, 130, 190, 120, 120, 180])
        # RunsOn lets a job take any host whose label matches, so each RunsOn job has its own label.
        hosted = [row['runner'] for row in measurements if row['runner'].startswith('runs-on=')]
        self.assertEqual(hosted, ['runs-on=42-hashes/runner=measure-x86_64-linux-intel/env=production',
                                  'runs-on=42-hashes/runner=measure-x86_64-linux-amd/env=production',
                                  'runs-on=42-checksum/runner=measure-x86_64-linux-intel/env=production',
                                  'runs-on=42-checksum/runner=measure-x86_64-linux-amd/env=production'])
        self.assertEqual({row['runner'] for row in measurements if row['name'] == 'riscv64-linux'}, {'ubuntu-24.04-riscv'})
        families = ci.jobs(ci.dispatch_inputs(request('sha256,sha512,blake3,crc64')))
        self.assertEqual({family: list(runs) for family, runs in families.items()},
                         {'hashes': ['sha2', 'blake3'], 'checksum': ['crc']})
        runs = ci.selection('sha256,sha512,blake3,crc64')
        self.assertEqual(runs, {'sha2': ['sha256', 'sha512'], 'blake3': ['blake3'], 'crc': ['crc64-xz', 'crc64-nvme']})
        # A whole target subsumes the algorithms it contains.
        self.assertEqual(ci.selection('auth,mlkem,ed25519'), {'auth': ['auth']})

    def test_all_runs_every_default_target_without_external_peers(self):
        catalog = ci.load_catalog()
        defaults = [name for name, bench in catalog['benches'].items()
                    if bench['kind'] == 'criterion' and bench['required']]
        self.assertEqual(ci.selection('all'), {name: [name] for name in defaults})
        self.assertNotIn('tls_connections', defaults)
        families = ci.jobs(ci.dispatch_inputs(request('all')))
        self.assertEqual(list(families), ['auth', 'checksum', 'hashes', 'aead'])
        # A split target runs one catalog algorithm at a time. Auth must not combine its 15 algorithms with a cold
        # Windows build in one budget. SLH-DSA has 24 variants; PBKDF2 has SHA-256 and SHA-512.
        split = {name: [algorithm for algorithm, entry in catalog['algorithms'].items() if entry['bench'] == name]
                 for name in defaults if catalog['benches'][name].get('split') == 'algorithms'}
        self.assertEqual({name: len(runs) for name, runs in split.items()}, {'slhdsa': 24, 'auth': 15, 'pbkdf2': 2})
        expected = [run for name in defaults for run in split.get(name, [name])]
        self.assertEqual(sorted(run for runs in families.values() for run in runs), sorted(expected))
        everything = {name: True for name in ci.PLATFORMS}
        measurements, builds = plan('all', **everything)
        self.assertEqual(len(measurements), 8 * len(families))
        self.assertEqual(len(builds), 3 * len(families))

    def test_split_target_runs_each_algorithm_alone(self):
        inputs = ci.dispatch_inputs(request('slhdsa-shake-192s,hash-slhdsa-shake-192s,mldsa65,mldsa87'))
        self.assertEqual({name: args[:-2] for name, args in ci.jobs(inputs)['auth'].items()},
                         {'slhdsa-shake-192s': ['slhdsa-shake-192s'], 'hash-slhdsa-shake-192s': ['hash-slhdsa-shake-192s'],
                          'mldsa': ['mldsa65', 'mldsa87']})
        env = request('slhdsa-shake-192s,hash-slhdsa-shake-192s')
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, env, clear=True), \
             patch.object(sys, 'argv', ['ci.py', 'prepare', 'auth', 'riscv64gc-unknown-linux-gnu', directory]), \
             patch.object(ci.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as execute:
            self.assertEqual(ci.main(), 0)
            self.assertEqual([[arg for arg in call.args[0] if arg.startswith('prepare_archive=')]
                              for call in execute.call_args_list],
                             [[f'prepare_archive={Path(directory) / name}.tar.gz']
                              for name in ('slhdsa-shake-192s', 'hash-slhdsa-shake-192s')])
        measurements, builds = plan('slhdsa', **{'riscv64-linux': True})
        self.assertEqual({row['timeout'] for row in builds}, {30 + 10 * 23})
        self.assertEqual([row['timeout'] for row in measurements], [120 + 10 * 23] * 2 + [180 + 10 * 23])

    def test_external_targets_fail_before_runners_start(self):
        for selection in ('tls-provider', 'tls_connections', 'sha256,tls-provider'):
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, 'OpenSSL 4.0.3'):
                plan(selection)

    def test_catalog_selection_preserves_algorithm_scope(self):
        inputs = ci.dispatch_inputs(request('sha256, blake3-mixed', filter='^.*/rscrypto/64$'))
        for bench, names in ci.selection(inputs['benchmarks']).items():
            args = ci.arguments(names, ci.options(inputs))
            rows = ci.runner.requests(ci.runner.parse(['bench', *args]), ci.load_catalog())
            self.assertEqual({r['binary'] for r in rows}, {bench})
            self.assertTrue(all(r['scope'] for r in rows))
            self.assertEqual({r['pattern'] for r in rows}, {'^.*/rscrypto/64$'})
        for selection in ('hashes', 'checksums', 'auth', 'aead', 'mldsa', 'password-hashing', 'all'):
            with self.subTest(selection=selection):
                self.assertTrue(plan(selection)[0])

    def test_rejects_unknown_selection_options_and_bad_inputs(self):
        for selection in ('', 'sha999', '--output-dir=/tmp', 'bench=sha2', 'sha256;echo pwned', 'structural'):
            with self.subTest(selection=selection), self.assertRaises((ValueError, SystemExit)):
                plan(selection)
        with self.assertRaises(ValueError):
            plan(sample_size='1')
        for inputs in ({'benchmarks': 'sha256', 'architectures': 'all'}, {'benchmarks': 'sha256', 'x86_64-linux': 'true'},
                       {'benchmarks': 'sha256', 'diagnostic': 'false'}, {'benchmarks': 7}, []):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                ci.dispatch_inputs({'DISPATCH_INPUTS': json.dumps(inputs)})

    def test_quick_mode_reaches_every_run_and_rejects_ignored_sampling(self):
        inputs = ci.dispatch_inputs(request('sha256,ed25519', quick=True, measure_ms='10'))
        for runs in ci.jobs(inputs).values():
            for args in runs.values():
                self.assertIn('quick=true', args)
                self.assertIn('measure_ms=10', args)
        for values in ({'sample_size': '10'}, {'warmup_ms': '10'}):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, 'quick mode takes no'):
                plan(quick=True, **values)
        with self.assertRaisesRegex(ValueError, 'Boolean'):
            ci.dispatch_inputs(request(quick='true'))

    def test_run_forwards_regex_literally_and_propagates_failure(self):
        env = request(filter="$(touch /tmp/not-executed);'|.*")
        with patch.dict(os.environ, env, clear=True), patch.object(sys, 'argv', ['ci.py', 'run', 'hashes']), \
             patch.object(ci.subprocess, 'run', return_value=subprocess.CompletedProcess([], 7)) as execute:
            self.assertEqual(ci.main(), 7)
            command = execute.call_args.args[0]
            self.assertEqual(command[:3], ['just', 'bench', 'sha256'])
            self.assertIn('filter=' + json.loads(env['DISPATCH_INPUTS'])['filter'], command)
            self.assertNotIn('shell', execute.call_args.kwargs)
        for family in ('auth', 'sha2'):
            with self.subTest(family=family), patch.dict(os.environ, env, clear=True), \
                 patch.object(sys, 'argv', ['ci.py', 'run', family]), \
                 self.assertRaisesRegex(ValueError, 'not a family of this selection'):
                ci.main()

    def test_family_runs_each_target_separately_and_stops_at_failure(self):
        env = request('sha256,blake3,crc32c')
        for statuses, expected in (([0, 0], 0), ([5, 0], 5)):
            with self.subTest(statuses=statuses), patch.dict(os.environ, env, clear=True), \
                 patch.object(sys, 'argv', ['ci.py', 'run', 'hashes']), \
                 patch.object(ci.subprocess, 'run',
                              side_effect=[subprocess.CompletedProcess([], status) for status in statuses]) as execute:
                self.assertEqual(ci.main(), expected)
                commands = [call.args[0][:3] for call in execute.call_args_list]
                self.assertEqual(commands, [['just', 'bench', 'sha256'], ['just', 'bench', 'blake3']][:len(commands)])
                self.assertEqual(len(commands), 2 if expected == 0 else 1)

    def test_platform_jobs_skip_runs_for_other_architectures(self):
        # Every platform receives the same plan; an x86-64-only configuration runs only on x86-64.
        env = request('xxh3,xxh3_v4')
        for machine, expected in (('x86_64', ['xxh3', 'xxh3_v4']), ('AMD64', ['xxh3', 'xxh3_v4']),
                                  ('aarch64', ['xxh3'])):
            with self.subTest(machine=machine), patch.dict(os.environ, env, clear=True), \
                 patch.object(sys, 'argv', ['ci.py', 'run', 'hashes']), \
                 patch('platform.machine', return_value=machine), \
                 patch.object(ci.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as execute:
                self.assertEqual(ci.main(), 0)
                self.assertEqual([call.args[0][2] for call in execute.call_args_list], expected)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, env, clear=True), \
             patch.object(sys, 'argv', ['ci.py', 'prepare', 'hashes', 'riscv64gc-unknown-linux-gnu', directory]), \
             patch.object(ci.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as execute:
            self.assertEqual(ci.main(), 0)
            self.assertEqual([call.args[0][2] for call in execute.call_args_list], ['xxh3'])

    def test_score_reads_each_measurement_archive_of_this_attempt(self):
        import compare
        for values, partial in (({}, []), ({'filter': '^sha256/'}, ['--partial'])):
            env = request('sha256', **values) | {'GITHUB_RUN_ID': '42', 'GITHUB_RUN_ATTEMPT': '1'}
            with self.subTest(values=values), tempfile.TemporaryDirectory() as directory, \
                 patch.dict(os.environ, env, clear=True), patch.object(compare, 'main', return_value=0) as score:
                root = Path(directory)
                source = root / 'source'
                (source / 'criterion/run').mkdir(parents=True)
                (source / 'criterion/run/status.txt').write_text('state=complete\n')
                # download-artifact saves each raw archive inside a directory with the artifact's name.
                for name in ('bench-x86_64-linux-intel-hashes-42-1', 'bench-aarch64-linux-auth-42-1',
                             'bench-prepare-riscv64-linux-hashes-42-1', 'bench-x86_64-linux-intel-hashes-42-2',
                             'bench-score-42-1'):
                    (root / 'artifacts' / f'{name}.tar.gz').mkdir(parents=True)
                    ci.evidence(source, root / 'artifacts' / f'{name}.tar.gz' / f'{name}.tar.gz')
                artifacts = root / 'artifacts'
                self.assertEqual(ci.score(artifacts, root / 'score.json', ci.dispatch_inputs(env)), 0)
                self.assertEqual(score.call_args.args[0], [
                    f'aarch64-linux={artifacts}/bench-aarch64-linux-auth-42-1/criterion/run',
                    f'x86_64-linux-intel={artifacts}/bench-x86_64-linux-intel-hashes-42-1/criterion/run',
                    '--json', str(root / 'score.json'), *partial])
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {'GITHUB_RUN_ID': '42', 'GITHUB_RUN_ATTEMPT': '1'}), \
             self.assertRaisesRegex(ValueError, 'no measurement artifacts'):
            ci.score(Path(directory), Path(directory) / 'score.json', {})

    def test_evidence_archive_omits_only_transferred_binaries(self):
        import tarfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'bench'
            files = ['criterion/run/status.txt', 'criterion/run/input/bundle.json', 'criterion/run/input/bin/1/sha2',
                     'criterion/run/criterion/sha256/new/estimates.json', 'preparation/riscv/bin/1/sha2']
            for name in files:
                (source / name).parent.mkdir(parents=True, exist_ok=True)
                (source / name).write_text(name)
            self.assertEqual(ci.evidence(source, root / 'evidence.tar.gz'), 0)
            with tarfile.open(root / 'evidence.tar.gz') as archive:
                self.assertEqual(archive.getnames(), sorted(set(files) - {'criterion/run/input/bin/1/sha2'}))
                self.assertTrue(all(member.isfile() for member in archive.getmembers()))
            # An existing archive is never replaced, and a missing or linked tree is an error.
            for source_dir, archive in ((source, root / 'evidence.tar.gz'), (root / 'missing', root / 'other.tar.gz')):
                with self.subTest(source=source_dir), self.assertRaises((FileExistsError, ValueError)):
                    ci.evidence(source_dir, archive)
            (source / 'criterion/run/link').symlink_to(source / 'criterion/run/status.txt')
            with self.assertRaisesRegex(ValueError, 'unexpected link'):
                ci.evidence(source, root / 'linked.tar.gz')

    def test_transfer_dispatch_keeps_selection_and_separate_invocations(self):
        env = request('sha256,sha512,crc32c', filter='^sha256/rscrypto/64$', sample_size='24')
        for operation, flag in (('measure', 'run_archive'), ('prepare', 'prepare_archive')):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as parent, \
                 patch.dict(os.environ, env, clear=True), \
                 patch.object(sys, 'argv', ['ci.py', operation, 'hashes', 's390x-unknown-linux-gnu',
                                            str(Path(parent) / 'archives with spaces')]), \
                 patch.object(ci.subprocess, 'run', return_value=subprocess.CompletedProcess([], 9)) as execute:
                self.assertEqual(ci.main(), 9)
                command = execute.call_args.args[0]
                self.assertEqual(command[:4], ['just', 'bench', 'sha256', 'sha512'])
                self.assertNotIn('crc32c', command)
                self.assertIn(f'{flag}={Path(parent) / "archives with spaces" / "sha2.tar.gz"}', command)
                self.assertIn('sample_size=24', command)
                self.assertIn('target=s390x-unknown-linux-gnu', command)
                # Preparation creates the transfer directory; measurement only reads it.
                self.assertEqual((Path(parent) / 'archives with spaces').is_dir(), operation == 'prepare')
        for name in ('riscv64-linux', 's390x-linux', 'powerpc64le-linux'):
            row, = ci.platforms(name, '42')['include']
            self.assertIn('target', row)
            self.assertGreaterEqual(row['timeout'], 120)
        self.assertTrue(all('target' not in row for row in ci.platforms('x86_64-linux', '42')['include']))

    def test_isolated_python_entry_points(self):
        for command in ([sys.executable, '-I', str(ROOT / 'scripts/bench/runner.py'), '--help'],
                        [sys.executable, '-I', str(ROOT / 'scripts/bench/bounded.py'), sys.executable, '-c', 'pass']):
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            inputs = {'benchmarks': 'sha256,blake3', 'filter': '"rscrypto" 64$',
                      **{name: True for name in ci.PLATFORMS}}
            # GitHub's toJSON spans lines, like this request.
            env = {**os.environ, 'DISPATCH_INPUTS': json.dumps(inputs, indent=2), 'GITHUB_RUN_ID': '42',
                   'GITHUB_OUTPUT': str(output)}
            result = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/bench/ci.py'), 'plan'],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            values = dict(line.split('=', 1) for line in output.read_text().splitlines())
            # Later jobs read this one-line copy; Enter-VsDevShell would keep only the first line of the original.
            self.assertEqual(json.loads(values['request']), inputs)
            self.assertEqual(len(json.loads(values['measurements'])['include']), 8)
            self.assertEqual(values['cross'], 'true')
            self.assertEqual({(row['target'], row['family']) for row in json.loads(values['builds'])['include']},
                             {(target, 'hashes') for target in
                              ('riscv64gc-unknown-linux-gnu', 'powerpc64le-unknown-linux-gnu', 's390x-unknown-linux-gnu')})
            self.assertIn('hashes / sha2: just bench sha256 output_dir=target/bench diag=false filter="rscrypto" 64$',
                          result.stdout)
            self.assertIn('hashes / blake3: just bench blake3 output_dir=target/bench diag=false', result.stdout)


if __name__ == '__main__':
    unittest.main()
