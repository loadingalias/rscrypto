#!/usr/bin/env -S uv run python
"""Check that a dispatched run's runner wait is bounded without hiding its result."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('dispatched_run', Path(__file__).with_name('dispatched_run.py'))
dispatched_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dispatched_run)

MINUTE = 60
RUN = 7
PARENT = 1
URL = 'https://github.com/owner/repo/actions/runs/7'


ERRORS = []


def http_error(code):
    error = urllib.error.HTTPError('https://api.github.com', code, 'error', {}, None)
    ERRORS.append(error)
    return error


def tearDownModule():
    for error in ERRORS:
        error.close()


class FakeGitHub:
    """Serve the dispatched run and its caller's jobs as functions of elapsed time."""

    def __init__(self, *, started_at=None, completed_at=None, conclusion='success',
                 sibling_failed_at=None, failures=()):
        self.now = 0
        self.started_at = started_at
        self.completed_at = completed_at
        self.conclusion = conclusion
        self.sibling_failed_at = sibling_failed_at
        self.failures = list(failures)
        self.cancelled_at = None
        self.calls = []

    def reached(self, moment):
        return moment is not None and self.now >= moment

    def request(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        if self.failures and method == 'GET':
            failure = self.failures.pop(0)
            if failure:
                raise failure
        if path.endswith('/dispatches'):
            # GitHub answers 204 with no body unless the request opts in to run details.
            if payload.get('return_run_details') is not True:
                return None
            return {'workflow_run_id': RUN, 'run_url': 'https://api.github.com/runs/7', 'html_url': URL}
        if path in (f'/actions/runs/{RUN}/cancel', f'/actions/runs/{RUN}/force-cancel'):
            self.cancelled_at = self.now
            return None
        if path == f'/actions/runs/{RUN}':
            if self.cancelled_at is not None:
                return {'status': 'completed', 'conclusion': 'cancelled'}
            if self.reached(self.completed_at):
                return {'status': 'completed', 'conclusion': self.conclusion}
            return {'status': 'in_progress' if self.reached(self.started_at) else 'queued', 'conclusion': None}
        if path.startswith(f'/actions/runs/{RUN}/jobs'):
            return {'jobs': [{'name': 'execute', 'conclusion': None,
                              'status': 'in_progress' if self.reached(self.started_at) else 'queued'}]}
        if path.startswith(f'/actions/runs/{PARENT}/jobs'):
            failed = self.reached(self.sibling_failed_at)
            return {'jobs': [{'name': 'riscv64-linux', 'status': 'in_progress', 'conclusion': None},
                             {'name': 'x86_64-linux', 'status': 'completed',
                              'conclusion': 'failure' if failed else 'success'}]}
        raise AssertionError(f'unexpected request {method} {path}')

    def sleep(self, seconds):
        self.now += seconds

    def cancels(self):
        return [call for call in self.calls if call[1].endswith('/cancel')]


class DispatchedRun(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.state = Path(temporary.name) / 'run'
        self.summary = Path(temporary.name) / 'summary'
        patcher = patch.dict(os.environ, GITHUB_STEP_SUMMARY=str(self.summary))
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_with(self, github, queue_minutes=30):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = dispatched_run.run(
                github, workflow='riscv.yml', ref='main', inputs={'suite': 'tests'}, queue_minutes=queue_minutes,
                state=self.state, parent=PARENT, label='RISC-V', clock=lambda: github.now, sleep=github.sleep)
        return status, output.getvalue()

    def test_dispatches_the_requested_ref_and_inputs(self):
        github = FakeGitHub(started_at=0, completed_at=MINUTE)
        self.run_with(github)
        self.assertEqual(github.calls[0], ('POST', '/actions/workflows/riscv.yml/dispatches',
                                           {'ref': 'main', 'inputs': {'suite': 'tests'},
                                            'return_run_details': True}))

    def test_limit_bounds_only_the_runner_wait(self):
        # Started at minute 29 and finished at minute 120: the run outlasts the limit and still decides.
        for conclusion, expected in (('success', 0), ('failure', 1)):
            with self.subTest(conclusion=conclusion):
                github = FakeGitHub(started_at=29 * MINUTE, completed_at=120 * MINUTE, conclusion=conclusion)
                status, output = self.run_with(github)
                self.assertEqual(status, expected)
                self.assertEqual(github.cancels(), [])
                self.assertFalse(self.state.exists())
                self.assertIn(f'concluded {conclusion}' if expected else 'passed', output)

    def test_unstarted_run_is_cancelled_and_skipped_at_the_limit(self):
        github = FakeGitHub()
        status, output = self.run_with(github)
        self.assertEqual(status, 0)
        self.assertEqual(github.cancelled_at, 30 * MINUTE)
        self.assertEqual(len(github.cancels()), 1)
        self.assertIn('::notice title=RISC-V skipped::', output)
        self.assertIn(f'no runner started {URL} within 30 minutes', self.summary.read_text())
        self.assertFalse(self.state.exists())

    def test_zero_limit_waits_for_a_runner(self):
        github = FakeGitHub(started_at=300 * MINUTE, completed_at=400 * MINUTE)
        status, _ = self.run_with(github, queue_minutes=0)
        self.assertEqual(status, 0)
        self.assertEqual(github.cancels(), [])

    def test_failed_caller_job_stops_the_dispatched_run(self):
        for started_at in (None, 5 * MINUTE):
            with self.subTest(started_at=started_at):
                github = FakeGitHub(started_at=started_at, completed_at=200 * MINUTE, sibling_failed_at=10 * MINUTE)
                status, output = self.run_with(github, queue_minutes=0)
                self.assertEqual(status, 1)
                self.assertEqual(github.cancelled_at, 10 * MINUTE)
                self.assertIn('stopped (cancelled) after x86_64-linux failed', output)

    def test_transient_api_failures_are_retried_and_others_are_not(self):
        github = FakeGitHub(started_at=0, completed_at=5 * MINUTE,
                            failures=[http_error(502), urllib.error.URLError('reset'), http_error(429)])
        status, output = self.run_with(github)
        self.assertEqual(status, 0)
        self.assertEqual(output.count('::warning::'), 3)
        github = FakeGitHub(failures=[http_error(503)] * dispatched_run.MAX_POLL_ERRORS)
        with self.assertRaises(urllib.error.HTTPError):
            self.run_with(github)
        self.assertEqual(github.now, (dispatched_run.MAX_POLL_ERRORS - 1) * dispatched_run.POLL_SECONDS)
        github = FakeGitHub(failures=[http_error(403)])
        with self.assertRaises(urllib.error.HTTPError):
            self.run_with(github)
        self.assertEqual(github.now, 0)
        # The cleanup step still holds the recorded run.
        self.assertEqual(self.state.read_text(), f'{RUN}\n')

    def test_new_child_run_and_jobs_can_be_temporarily_missing(self):
        for missing in (f'/actions/runs/{RUN}', f'/actions/runs/{RUN}/jobs?per_page=100'):
            with self.subTest(path=missing):
                github = FakeGitHub(completed_at=MINUTE)
                request = github.request
                absent = [True]

                def delayed(method, path, payload=None):
                    if path == missing and absent:
                        absent.pop()
                        raise http_error(404)
                    return request(method, path, payload)

                github.request = delayed
                status, output = self.run_with(github)
                self.assertEqual(status, 0)
                self.assertEqual(github.now, MINUTE)
                self.assertEqual(output.count('::warning::'), 1)
                self.assertEqual(github.cancels(), [])

    def test_missing_child_retries_are_bounded_and_parent_errors_are_fatal(self):
        for missing, expected_time in (
                (f'/actions/runs/{RUN}', 9 * MINUTE),
                (f'/actions/runs/{RUN}/jobs?per_page=100', 9 * MINUTE),
                (f'/actions/runs/{PARENT}/jobs?per_page=100', 0)):
            with self.subTest(path=missing):
                github = FakeGitHub()
                request = github.request

                def absent(method, path, payload=None):
                    if path == missing:
                        raise http_error(404)
                    return request(method, path, payload)

                github.request = absent
                with self.assertRaises(urllib.error.HTTPError):
                    self.run_with(github)
                self.assertEqual(github.now, expected_time)
                self.assertEqual(self.state.read_text(), f'{RUN}\n')

    def test_cancel_conflict_does_not_confirm_a_queued_run_stopped(self):
        github = FakeGitHub()
        request = github.request

        def conflict(method, path, payload=None):
            if path.endswith('/cancel') and not github.cancels():
                github.calls.append((method, path, payload))
                raise http_error(409)
            return request(method, path, payload)

        github.request = conflict
        status, _ = self.run_with(github)
        self.assertEqual(status, 0)
        self.assertEqual(len(github.cancels()), 2)
        self.assertIsNotNone(github.cancelled_at)
        self.assertFalse(self.state.exists())

    def test_started_child_cannot_disappear_silently(self):
        github = FakeGitHub(started_at=0)
        request = github.request

        def disappear(method, path, payload=None):
            if path == f'/actions/runs/{RUN}' and github.now:
                raise http_error(404)
            return request(method, path, payload)

        github.request = disappear
        with self.assertRaises(urllib.error.HTTPError):
            self.run_with(github)
        self.assertEqual(github.now, MINUTE)
        self.assertEqual(self.state.read_text(), f'{RUN}\n')

    def test_accepted_cancellation_waits_for_terminal_state(self):
        github = FakeGitHub()
        request = github.request
        requested = []

        def delayed(method, path, payload=None):
            if path.endswith('/cancel'):
                requested.append(github.now)
                return None  # HTTP 202 acknowledges the request, not completion.
            if path == f'/actions/runs/{RUN}' and requested:
                self.assertTrue(self.state.exists())
                if github.now >= requested[0] + 10:
                    github.cancelled_at = github.now
            return request(method, path, payload)

        github.request = delayed
        status, _ = self.run_with(github)
        self.assertEqual(status, 0)
        self.assertEqual(github.cancelled_at, 30 * MINUTE + 10)
        self.assertFalse(self.state.exists())

    def test_completion_racing_the_queue_timeout_keeps_its_actual_result(self):
        for conclusion, expected in (('success', 0), ('failure', 1)):
            with self.subTest(conclusion=conclusion):
                github = FakeGitHub()
                request = github.request
                reads = []

                def finish(method, path, payload=None):
                    if path == f'/actions/runs/{RUN}' and github.now >= 30 * MINUTE:
                        reads.append(path)
                        if len(reads) > 1:
                            return {'status': 'completed', 'conclusion': conclusion}
                    return request(method, path, payload)

                github.request = finish
                status, output = self.run_with(github)
                self.assertEqual(status, expected)
                self.assertNotIn('skipped', output)
                self.assertEqual(github.cancels(), [])
                self.assertFalse(self.state.exists())

    def test_dispatch_without_a_run_id_fails(self):
        github = FakeGitHub()
        github.request = lambda method, path, payload=None: None
        with self.assertRaisesRegex(RuntimeError, 'did not identify'):
            self.run_with(github)
        self.assertFalse(self.state.exists())

    def test_stop_cancels_only_a_recorded_run(self):
        github = FakeGitHub()
        self.assertEqual(dispatched_run.stop(github, self.state), 0)
        self.assertEqual(github.calls, [])
        self.state.write_text(f'{RUN}\n')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(dispatched_run.stop(github, self.state,
                             clock=lambda: github.now, sleep=github.sleep), 0)
        self.assertEqual(github.cancels(), [('POST', f'/actions/runs/{RUN}/cancel', None)])
        self.assertFalse(self.state.exists())

    def test_stop_accepts_an_already_completed_run_without_cancelling_it(self):
        github = FakeGitHub(completed_at=0)
        self.state.write_text(f'{RUN}\n')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(dispatched_run.stop(github, self.state), 0)
        self.assertEqual(github.cancels(), [])
        self.assertFalse(self.state.exists())
        self.assertIn('completed (success)', output.getvalue())

    def test_stop_forces_unresponsive_cancellation_with_a_fixed_deadline(self):
        for force_works, missing_reads in ((True, 0), (False, 0), (True, 3)):
            with self.subTest(force_works=force_works, missing_reads=missing_reads):
                self.state.write_text(f'{RUN}\n')
                github = FakeGitHub(failures=[http_error(404)] * missing_reads)
                request = github.request
                forced = []

                def unresponsive(method, path, payload=None):
                    self.assertTrue(self.state.exists())
                    if path.endswith('/cancel'):
                        return None
                    if path.endswith('/force-cancel'):
                        forced.append(github.now)
                        if not force_works:
                            return None
                    return request(method, path, payload)

                github.request = unresponsive
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    result = dispatched_run.stop(github, self.state,
                                                 clock=lambda: github.now, sleep=github.sleep)
                self.assertEqual(forced[0], MINUTE + 5 * missing_reads)
                self.assertEqual(result, 0 if force_works else 1)
                self.assertEqual(self.state.exists(), not force_works)
                self.assertEqual('::error' in output.getvalue(), not force_works)
                self.assertEqual(github.now, 65 + 5 * missing_reads if force_works else 120)

    def test_cleanup_api_errors_keep_the_run_id_and_fail_within_the_budget(self):
        for code, elapsed in ((403, 0), (404, 120), (503, 120)):
            with self.subTest(code=code):
                github = FakeGitHub(failures=[http_error(code)] * 30)
                self.state.write_text(f'{RUN}\n')
                with contextlib.redirect_stdout(io.StringIO()):
                    result = dispatched_run.stop(github, self.state,
                                                 clock=lambda: github.now, sleep=github.sleep)
                self.assertEqual(result, 1)
                self.assertEqual(github.now, elapsed)
                self.assertEqual(self.state.read_text(), f'{RUN}\n')


if __name__ == '__main__':
    unittest.main()
