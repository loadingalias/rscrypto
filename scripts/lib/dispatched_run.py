#!/usr/bin/env -S uv run python
"""Run one workflow job as a dispatched run whose runner wait the caller bounds.

GitHub Actions cannot time out a queued job or cancel one job of a run. A caller
that must not wait indefinitely for a runner pool dispatches the work as its own
run, reports that run's result, and cancels it when no runner starts it in time.
"""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

# This version answers a dispatch with an empty 204 unless `return_run_details` is set.
API_VERSION = '2022-11-28'
POLL_SECONDS = 60
# Consecutive transient API failures to tolerate, about ten minutes of polling.
MAX_POLL_ERRORS = 10
CANCEL_SECONDS = 120
FORCE_CANCEL_AFTER = 60
CANCEL_POLL_SECONDS = 5
FAILED = {'failure', 'timed_out'}


class GitHub:
    def __init__(self, repository, token):
        self.repository = repository
        self.token = token

    def request(self, method, path, payload=None):
        request = urllib.request.Request(
            f'https://api.github.com/repos/{self.repository}{path}', method=method,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={'Authorization': f'Bearer {self.token}', 'Accept': 'application/vnd.github+json',
                     'Content-Type': 'application/json', 'X-GitHub-Api-Version': API_VERSION})
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
        return json.loads(body) if body else None


def annotate(level, message, title=None):
    print(f'::{level} title={title}::{message}' if title else f'::{level}::{message}', flush=True)
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a', encoding='utf-8') as file:
            file.write(message + '\n')


def cancel(api, run_id, *, clock=time.monotonic, sleep=time.sleep):
    """Return the terminal run, or fail without claiming that cancellation finished."""
    deadline = clock() + CANCEL_SECONDS
    requested_at = None
    while True:
        try:
            result = api.request('GET', f'/actions/runs/{run_id}')
            if result['status'] == 'completed':
                return result
            if clock() >= deadline:
                break
            operation = ('force-cancel' if requested_at is not None and clock() - requested_at >= FORCE_CANCEL_AFTER
                         else 'cancel')
            if requested_at is None:
                requested_at = clock()
            try:
                api.request('POST', f'/actions/runs/{run_id}/{operation}')
            except urllib.error.HTTPError as error:
                # A newly queued run can reject cancellation before it is ready.
                # Neither 409 nor 202 establishes that the run has stopped.
                if error.code != 409:
                    raise
        except Exception as error:
            if not transient(error, allow_missing=True):
                raise
            print(f'::warning::Cancellation API request failed ({error}); retrying', flush=True)
        remaining = deadline - clock()
        if remaining <= 0:
            break
        sleep(min(CANCEL_POLL_SECONDS, remaining))
    raise RuntimeError(f'dispatched run {run_id} did not stop within {CANCEL_SECONDS} seconds')


def jobs(api, run_id):
    return api.request('GET', f'/actions/runs/{run_id}/jobs?per_page=100')['jobs']


def transient(error, *, allow_missing=False):
    if isinstance(error, urllib.error.HTTPError):
        return (allow_missing and error.code == 404) or error.code == 429 or error.code >= 500
    return isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError))


def report_conclusion(result, label, url):
    if result['conclusion'] == 'success':
        annotate('notice', f'{label} passed: {url}', title=f'{label} passed')
        return 0
    annotate('error', f'{label} concluded {result["conclusion"]}: {url}')
    return 1


def run(api, *, workflow, ref, inputs, queue_minutes, state, parent, label,
        clock=time.monotonic, sleep=time.sleep):
    """Return 0 when the dispatched run passes or never starts in time, 1 when it fails."""
    dispatched = api.request('POST', f'/actions/workflows/{workflow}/dispatches',
                             {'ref': ref, 'inputs': inputs, 'return_run_details': True})
    if not dispatched or 'workflow_run_id' not in dispatched:
        raise RuntimeError(f'GitHub did not identify the {workflow} run it dispatched')
    run_id, url = dispatched['workflow_run_id'], dispatched['html_url']
    # The caller's cleanup step cancels this run if the wait below is interrupted.
    state.write_text(f'{run_id}\n')
    print(f'{label}: dispatched {url}', flush=True)
    # The limit bounds only the wait for a runner; a started run may take its full timeout.
    deadline = clock() + queue_minutes * 60 if queue_minutes else None
    started = False
    errors = 0
    while True:
        pending_child = False
        try:
            failed = sorted(job['name'] for job in jobs(api, parent) if job['conclusion'] in FAILED)
            if failed:
                result = cancel(api, run_id, clock=clock, sleep=sleep)
                state.unlink()
                annotate('error', f'{label} stopped ({result["conclusion"]}) after {", ".join(failed)} failed: {url}')
                return 1
            # Only this known child's initial run/jobs reads may return a transient
            # 404. A 404 from parent lookup, dispatch, or a started run remains fatal.
            pending_child = not started
            dispatched_run = api.request('GET', f'/actions/runs/{run_id}')
            if dispatched_run['status'] == 'completed':
                state.unlink()
                return report_conclusion(dispatched_run, label, url)
            if not started:
                started = any(job['status'] in ('in_progress', 'completed') for job in jobs(api, run_id))
                if started:
                    print(f'{label}: a runner started {url}', flush=True)
                elif deadline is not None and clock() >= deadline:
                    result = cancel(api, run_id, clock=clock, sleep=sleep)
                    state.unlink()
                    if result['conclusion'] != 'cancelled':
                        return report_conclusion(result, label, url)
                    annotate('notice', f'{label} skipped: no runner started {url} within {queue_minutes} minutes',
                             title=f'{label} skipped')
                    return 0
            errors = 0
        except Exception as error:
            if not transient(error, allow_missing=pending_child):
                raise
            errors += 1
            if errors >= MAX_POLL_ERRORS:
                raise
            print(f'::warning::GitHub API request failed ({error}); retrying', flush=True)
        sleep(POLL_SECONDS)


def stop(api, state, *, clock=time.monotonic, sleep=time.sleep):
    """Cancel a dispatched run whose caller stopped before it reported a result."""
    if not state.exists():
        return 0
    run_id = int(state.read_text())
    try:
        result = cancel(api, run_id, clock=clock, sleep=sleep)
    except Exception as error:
        annotate('error', f'Could not confirm cancellation of dispatched run {run_id}: {error}')
        return 1
    state.unlink()
    print(f'Dispatched run {run_id} completed ({result["conclusion"]})', flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    start = commands.add_parser('run', help='dispatch a workflow run and report its result')
    start.add_argument('--workflow', required=True)
    start.add_argument('--ref', required=True)
    start.add_argument('--label', required=True)
    start.add_argument('--queue-minutes', type=int, required=True,
                       help='minutes to wait for a runner before cancelling and skipping; 0 waits indefinitely')
    start.add_argument('--input', action='append', default=[], metavar='NAME=VALUE')
    start.add_argument('--state', type=Path, required=True)
    end = commands.add_parser('cancel', help='cancel the run recorded in --state, if any')
    end.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    api = GitHub(os.environ['GITHUB_REPOSITORY'], os.environ['GH_TOKEN'])
    if args.command == 'cancel':
        return stop(api, args.state)
    if args.queue_minutes < 0:
        parser.error('--queue-minutes must not be negative')
    if any('=' not in value for value in args.input):
        parser.error('--input must be NAME=VALUE')
    inputs = dict(value.split('=', 1) for value in args.input)
    return run(api, workflow=args.workflow, ref=args.ref, inputs=inputs, queue_minutes=args.queue_minutes,
               state=args.state, parent=os.environ['GITHUB_RUN_ID'], label=args.label)


if __name__ == '__main__':
    raise SystemExit(main())
