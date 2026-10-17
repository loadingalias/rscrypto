#!/usr/bin/env -S uv run python
"""Exercise shared preparation actions and protect their qualification boundaries."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
ACTION = './.github/actions/cross-build/'
sys.path.insert(0, str(ROOT / 'scripts/bench'))
import ci as bench_ci  # noqa: E402


def read_yaml(path):
    return yaml.load((ROOT / path).read_text(), Loader=yaml.BaseLoader)


class CrossWorkflow(unittest.TestCase):
    def setUp(self):
        self.setup = read_yaml('.github/actions/cross-build/setup/action.yml')['runs']['steps']
        self.upload = read_yaml('.github/actions/cross-build/upload/action.yml')['runs']['steps']
        temporary = tempfile.TemporaryDirectory(prefix='cross workflow ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / 'output'
        self.output.touch()
        self.env = {key: value for key, value in os.environ.items()
                    if key not in ('BASH_ENV', 'ENV') and not key.startswith('BASH_FUNC_')}
        self.env.update(GITHUB_OUTPUT=str(self.output), GITHUB_PATH=str(self.output))

    def execute(self, step, **env):
        return subprocess.run(['bash', '--noprofile', '--norc', '-eo', 'pipefail', '-c', step['run']],
                              cwd=self.root, env={**self.env, **env}, text=True, capture_output=True)

    def test_setup_preserves_target_and_stops_on_installation_failure(self):
        [install] = self.setup
        installer = self.root / 'scripts/tooling/x86_64-linux.sh'
        installer.parent.mkdir(parents=True)
        installer.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > arguments\nexit "${INSTALL_EXIT:-0}"\n')
        installer.chmod(0o755)
        for target in ('s390x-unknown-linux-gnu', 'target with spaces'):
            with self.subTest(target=target):
                self.output.write_text('')
                result = self.execute(install, TARGET=target)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((self.root / 'arguments').read_text().splitlines(), ['--ci-cross-build', target])
                self.assertEqual(self.output.read_text(), str(Path.home() / '.cargo/bin') + '\n')
        self.output.write_text('')
        result = self.execute(install, TARGET='s390x-unknown-linux-gnu', INSTALL_EXIT='42')
        self.assertEqual(result.returncode, 42)
        self.assertEqual(self.output.read_text(), '')

    def test_upload_requires_both_files_and_keeps_consumer_names(self):
        require, upload = self.upload
        self.assertNotIn('if', upload)  # The transfer must require successful preparation.
        self.assertNotIn('continue-on-error', upload)
        self.assertEqual(upload['with'], {
            'name': '${{ steps.bundle.outputs.artifact }}',
            'path': '${{ steps.bundle.outputs.archive }}\ntarget/${{ inputs.target }}-tools.tar.gz\n',
            'if-no-files-found': 'error', 'retention-days': '2',
        })
        target = 'riscv64gc-unknown-linux-gnu'
        # Benchmark preparation transfers a directory with one sealed archive per catalog target.
        cases = [('tests', 'tests.tar.xz', 'tests.tar.xz', 'tests'), ('ct', 'ct.tar.gz', 'ct.tar.gz', 'ct-prepared'),
                 ('bench', 'bench', 'bench/sha2.tar.gz', 'bench-prepared'),
                 ('profile', 'profile.tar.gz', 'profile.tar.gz', 'profile-prepared')]
        (self.root / 'target').mkdir()
        tools = self.root / 'target' / f'{target}-tools.tar.gz'
        for suite, filename, prepared, artifact in cases:
            with self.subTest(suite=suite):
                archive = self.root / 'target' / f'{target}-{prepared}'
                archive.parent.mkdir(exist_ok=True)
                archive.write_bytes(b'prepared suite')
                tools.write_bytes(b'runner tools')
                self.output.write_text('')
                result = self.execute(require, TARGET=target, SUITE=suite)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.output.read_text(),
                                 f'archive=target/{target}-{filename}\nartifact={target}-{artifact}\n')
                for missing in (archive, tools):
                    missing.unlink()
                    self.output.write_text('')
                    result = self.execute(require, TARGET=target, SUITE=suite)
                    self.assertEqual(result.returncode, 1)
                    self.assertIn('Missing prepared artifact:', result.stderr)
                    self.assertEqual(self.output.read_text(), '')
                    missing.write_bytes(b'restored fixture')
        archive = self.root / 'target' / f'{target}-bench' / 'sha2.tar.gz'
        archive.write_bytes(b'prepared suite')
        self.output.write_text('')
        result = self.execute(require, TARGET=target, SUITE='bench', ARTIFACT='bench-prepared-riscv64-linux-hashes')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.output.read_text(),
                         f'archive=target/{target}-bench\nartifact=bench-prepared-riscv64-linux-hashes\n')
        self.output.write_text('')
        result = self.execute(require, TARGET=target, SUITE='unknown')
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.output.read_text(), '')

    def test_callers_keep_preparation_separate_from_native_qualification(self):
        bench_artifact = 'bench-prepared-${{ matrix.platform }}-${{ matrix.family }}'
        for filename, producer, consumer, suite, command, artifact in (
            ('ci', 'cross-build', 'native', 'tests', 'just test-cross prepare', '${{ matrix.target }}-tests'),
            ('ct', 'cross-build', 'ct', 'ct', 'just ct-full --target', '${{ matrix.target }}-ct-prepared'),
            ('bench', 'cross-build', 'bench', 'bench', 'scripts/lib/python.sh scripts/bench/ci.py prepare', bench_artifact),
            ('profile', 'prepare', 'capture', 'profile', 'scripts/lib/python.sh scripts/bench/profile_ci.py prepare',
             '${{ needs.plan.outputs.target }}-profile-prepared'),
        ):
            with self.subTest(workflow=filename):
                jobs = read_yaml(f'.github/workflows/{filename}.yml')['jobs']
                build, native = jobs[producer], jobs[consumer]
                target = '${{ needs.plan.outputs.target }}' if suite == 'profile' else '${{ matrix.target }}'
                steps = build['steps']
                setup, = [step for step in steps if step.get('uses') == ACTION + 'setup']
                upload, = [step for step in steps if step.get('uses') == ACTION + 'upload']
                prepare, = [step for step in steps if command in step.get('run', '')]
                self.assertEqual(setup['with'], {'target': target})
                expected = {'target': target, 'suite': suite}
                if suite == 'bench':
                    # One transfer per platform and primitive family; the default name would collide.
                    expected['artifact'] = bench_artifact
                self.assertEqual(upload['with'], expected)
                self.assertLess(steps.index(setup), steps.index(prepare))
                self.assertLess(steps.index(prepare), steps.index(upload))
                self.assertNotIn('if', setup)
                self.assertNotIn('if', upload)
                self.assertTrue(all('continue-on-error' not in step for step in (setup, prepare, upload)))
                self.assertIn('source "$HOME/.local/share/rscrypto-tooling/environment.sh"', prepare['run'])
                needs = native['needs']
                self.assertIn(producer, needs if isinstance(needs, list) else [needs])
                download, = [step for step in native['steps']
                             if step.get('uses', '').startswith('actions/download-artifact@')]
                self.assertEqual(download['with']['name'], artifact)
                self.assertNotIn('run-id', download['with'])
                if suite != 'profile':
                    self.assertEqual(build['strategy']['fail-fast'], 'true')
                    self.assertEqual(native['strategy']['fail-fast'], 'true')
                if suite in ('tests', 'ct'):
                    self.assertIn('just test-transfer', prepare['run'])

    def test_bench_form_matches_planner_inputs(self):
        workflow = read_yaml('.github/workflows/bench.yml')
        inputs = workflow['on']['workflow_dispatch']['inputs']
        self.assertEqual(set(inputs), bench_ci.INPUTS)
        self.assertEqual([name for name in inputs if inputs[name]['type'] == 'boolean'],
                         [*bench_ci.PLATFORMS, 'diagnostic'])
        self.assertEqual([name for name in bench_ci.PLATFORMS if inputs[name].get('default') == 'true'],
                         ['x86_64-linux'])
        self.assertEqual(workflow['env']['DISPATCH_INPUTS'], '${{ toJSON(inputs) }}')
        self.assertEqual(workflow['jobs']['plan']['outputs']['request'], '${{ steps.plan.outputs.request }}')
        for job, label in (('cross-build', 'Prepare ${{ matrix.platform }} / ${{ matrix.family }}'),
                           ('bench', 'Measure ${{ matrix.name }} / ${{ matrix.family }}')):
            with self.subTest(job=job):
                self.assertEqual(workflow['jobs'][job]['name'], label)
                self.assertEqual(workflow['jobs'][job]['env']['FAMILY'], '${{ matrix.family }}')
                # Enter-VsDevShell keeps only the first line of toJSON's multi-line request.
                self.assertEqual(workflow['jobs'][job]['env']['DISPATCH_INPUTS'], '${{ needs.plan.outputs.request }}')
                self.assertEqual(workflow['jobs'][job]['timeout-minutes'], '${{ matrix.timeout }}')
                # RunsOn ignores max-parallel when it starts hosts; the planner's job count bounds them.
                self.assertNotIn('max-parallel', workflow['jobs'][job]['strategy'])
                runs = [step['run'] for step in workflow['jobs'][job]['steps'] if 'scripts/bench/ci.py' in step.get('run', '')]
                self.assertTrue(runs)
                self.assertTrue(all('$FAMILY' in run or '$env:FAMILY' in run for run in runs))
                self.assertTrue(all('${{' not in run for run in runs))

    def test_bench_scorecard_reads_only_this_attempts_measurements(self):
        jobs = read_yaml('.github/workflows/bench.yml')['jobs']
        score = jobs['score']
        self.assertEqual(score['needs'], ['plan', 'bench'])
        self.assertEqual(score['env']['DISPATCH_INPUTS'], '${{ needs.plan.outputs.request }}')
        labels = [job['runs-on'] for job in jobs.values() if isinstance(job.get('runs-on'), str)]
        self.assertEqual(labels.count(score['runs-on']), 1)
        download, = [step for step in score['steps'] if 'download-artifact' in step.get('uses', '')]
        self.assertEqual(download['with']['pattern'], 'bench-*-${{ github.run_id }}-${{ github.run_attempt }}')
        run, = [step['run'] for step in score['steps'] if 'scripts/bench/ci.py score' in step.get('run', '')]
        self.assertIn(download['with']['path'], run)
        self.assertIn('$GITHUB_STEP_SUMMARY', run)

    def test_bench_evidence_keeps_each_transferred_binary_once(self):
        jobs = read_yaml('.github/workflows/bench.yml')['jobs']
        evidence, = [step for step in jobs['bench']['steps'] if step.get('name') == 'Retain benchmark evidence']
        preparation, = [step for step in jobs['cross-build']['steps']
                        if step.get('name') == 'Retain benchmark preparation evidence']
        # The measurement artifact drops the unpacked transfer binaries that the preparation artifact keeps.
        self.assertEqual(evidence['with']['path'].split(), ['target/bench/', '!target/bench/criterion/*/input/bin/**'])
        self.assertEqual(preparation['with']['path'], 'target/bench/preparation/')
        self.assertEqual(evidence['with']['retention-days'], preparation['with']['retention-days'])

    def test_largest_bench_run_fits_aws_quotas_with_one_label_per_job(self):
        # Standard-family vCPU quotas of the RunsOn AWS account in us-east-1, read 2026-10-16 from
        # Service Quotas L-34B43A08 (Spot) and L-1216C47A (On-Demand). Update them when AWS changes them.
        quotas = {'spot': 256, 'on-demand': 366}
        runners = read_yaml('.github/runs-on.yml')['runners']
        jobs = read_yaml('.github/workflows/bench.yml')['jobs']
        benches = bench_ci.load_catalog()['benches']
        # Every Criterion target that CI can run, on every platform.
        names = [name for name, bench in benches.items() if 'family' in bench and 'external' not in bench]
        inputs = {'benchmarks': ','.join(names), **{name: True for name in bench_ci.PLATFORMS}}
        measurements, builds, _ = bench_ci.plan(inputs, '42')
        for job, rows in (('cross-build', builds['include']), ('bench', measurements['include'])):
            with self.subTest(job=job):
                labels = [re.sub(r'\$\{\{ (?:github\.run_id|matrix\.(\w+)) \}\}',
                                 lambda match: row[match[1]] if match[1] else '42', jobs[job]['runs-on'])
                          for row in rows]
                hosted = [label for label in labels if label.startswith('runs-on=')]
                self.assertTrue(hosted)
                # RunsOn starts a host for every queued job, and a shared label lets one job take another's host.
                self.assertEqual(len(hosted), len(set(hosted)))
                demand = dict.fromkeys(quotas, 0)
                for label in hosted:
                    runner = runners[re.fullmatch(r'runs-on=[^/]+/runner=([^/]+)/env=production', label)[1]]
                    demand['on-demand' if runner['spot'] == 'false' else 'spot'] += int(runner['cpu'])
                for pool, vcpus in demand.items():
                    self.assertLessEqual(vcpus, quotas[pool], pool)

    def test_ct_preparation_failure_keeps_diagnostics_out_of_transfer(self):
        jobs = read_yaml('.github/workflows/ct.yml')['jobs']
        steps = jobs['cross-build']['steps']
        prepare, = [step for step in steps if 'just ct-full --target' in step.get('run', '')]
        failure, = [step for step in steps if step.get('name') == 'Retain failed CT preparation evidence']
        self.assertLess(steps.index(prepare), steps.index(failure))
        self.assertEqual(failure['if'], 'failure()')
        self.assertNotIn('continue-on-error', prepare)
        self.assertEqual(failure['with']['path'], 'target/ct/${{ matrix.target }}/release/')
        self.assertEqual(failure['with']['name'],
                         '${{ matrix.target }}-ct-preparation-failure-${{ github.run_attempt }}')
        download, = [step for step in jobs['ct']['steps']
                     if step.get('uses', '').startswith('actions/download-artifact@')]
        self.assertNotEqual(failure['with']['name'], download['with']['name'])
        transfer, = [step for step in steps if step.get('uses') == ACTION + 'upload']
        self.assertNotIn('if', transfer)  # Failed preparation cannot produce a qualified transfer.

    def test_riscv_executes_its_callers_artifacts_in_a_dispatched_run_releases_require(self):
        child = read_yaml('.github/workflows/riscv.yml')
        execute = child['jobs']['execute']
        self.assertEqual(child['env']['TARGET'], 'riscv64gc-unknown-linux-gnu')
        self.assertEqual(execute['runs-on'], 'ubuntu-24.04-riscv')
        # Run 38501560929 failed to start: a dispatched '120' is not a number.
        self.assertEqual(child['on']['workflow_dispatch']['inputs']['timeout_minutes']['type'], 'string')
        self.assertEqual(execute['timeout-minutes'], '${{ fromJSON(inputs.timeout_minutes) }}')
        self.assertEqual(execute['permissions'], {'contents': 'read', 'actions': 'read'})
        self.assertEqual(execute['steps'][0]['with']['ref'], '${{ inputs.sha }}')
        download, = [step for step in execute['steps']
                     if step.get('uses', '').startswith('actions/download-artifact@')]
        self.assertEqual(download['with']['run-id'], '${{ inputs.run_id }}')
        # The names cross-build/upload gives the tests and CT bundles.
        self.assertEqual(download['with']['name'],
                         "${{ env.TARGET }}-${{ inputs.suite == 'ct' && 'ct-prepared' || 'tests' }}")
        ci_rows = read_yaml('.github/workflows/ci.yml')['jobs']['native']['strategy']['matrix']['include']
        self.assertNotIn('riscv64-linux', [row['platform'] for row in ci_rows])
        for filename, suite in (('ci', 'tests'), ('ct', 'ct')):
            with self.subTest(workflow=filename):
                workflow = read_yaml(f'.github/workflows/{filename}.yml')
                self.assertEqual(workflow['permissions'], {'contents': 'read'})
                # Only the dispatching job, which builds nothing, may change Actions state.
                writers = [name for name, job in workflow['jobs'].items()
                           if job.get('permissions', {}).get('actions') == 'write']
                self.assertEqual(writers, ['riscv64-linux'])
                proxy = workflow['jobs']['riscv64-linux']
                self.assertEqual(proxy['runs-on'], 'ubuntu-24.04')
                install, start, stop = proxy['steps'][1:]
                # The pinned uv runs the dispatcher; the token never reaches its installation.
                self.assertEqual(install['run'], 'scripts/tooling/uv.sh x86_64-linux >>"$GITHUB_PATH"')
                self.assertNotIn('env', install)
                self.assertIn('scripts/lib/dispatched_run.py run --workflow riscv.yml', start['run'])
                self.assertIn(f'--input suite={suite}', start['run'])
                self.assertEqual(start['env']['QUEUE_MINUTES'], "${{ inputs.require_riscv && '0' || '30' }}")
                self.assertEqual(stop['if'], 'always()')
                self.assertIn('scripts/lib/dispatched_run.py cancel', stop['run'])
                self.assertEqual(workflow['on']['workflow_call']['inputs']['require_riscv']['default'], 'false')
        release = read_yaml('.github/workflows/release.yml')['jobs']
        for name in ('ci', 'ct'):
            with self.subTest(release=name):
                self.assertEqual(release[name]['with']['require_riscv'], 'true')
                self.assertEqual(release[name]['permissions'], {'contents': 'read', 'actions': 'write'})


if __name__ == '__main__':
    unittest.main()
