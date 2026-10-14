#!/usr/bin/env python3
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
            return {'workflow_run_id': RUN, 'run_url': 'https://api.github.com/runs/7', 'html_url': URL}
        if path == f'/actions/runs/{RUN}/cancel':
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
                                           {'ref': 'main', 'inputs': {'suite': 'tests'}}))

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
                self.assertIn('because x86_64-linux failed', output)

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

    def test_dispatch_without_a_run_id_fails(self):
        github = FakeGitHub()
        github.request = lambda method, path, payload=None: None
        with self.assertRaisesRegex(RuntimeError, 'did not identify'):
            self.run_with(github)
        self.assertFalse(self.state.exists())

    def test_stop_cancels_only_a_recorded_run(self):
        github = FakeGitHub()
        dispatched_run.stop(github, self.state)
        self.assertEqual(github.calls, [])
        self.state.write_text(f'{RUN}\n')
        with contextlib.redirect_stdout(io.StringIO()):
            dispatched_run.stop(github, self.state)
        self.assertEqual(github.cancels(), [('POST', f'/actions/runs/{RUN}/cancel', None)])
        self.assertFalse(self.state.exists())
        for error, cancelled in ((http_error(409), True), (http_error(500), False)):
            with self.subTest(code=error.code):
                self.state.write_text(f'{RUN}\n')
                github = FakeGitHub()
                def refuse(method, path, payload=None, error=error):
                    raise error
                github.request = refuse
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    dispatched_run.stop(github, self.state)
                self.assertEqual(self.state.exists(), not cancelled)
                self.assertEqual('::warning::' in output.getvalue(), not cancelled)


if __name__ == '__main__':
    unittest.main()
