#!/usr/bin/env python3
"""Queue physical Mac qualification in an isolated worktree and publish its commit status."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import qualified

ROOT = qualified.ROOT


def directory():
    return Path(qualified.git('rev-parse', '--path-format=absolute', '--git-common-dir')) / 'rscrypto-macos'


def repository(remote):
    match = re.fullmatch(r'(?:git@github\.com:|https://github\.com/|ssh://git@github\.com/)([\w.-]+/[\w.-]+?)(?:\.git)?/?', remote)
    return match[1] if match else None


def save(job, state, phase, **fields):
    state.update(phase=phase, updated=time.time(), **fields)
    temporary = job / 'state.tmp'
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(job / 'state.json')


def queue(remote, expected=None):
    if subprocess.check_output(['uname', '-sm'], text=True).strip() != 'Darwin arm64':
        raise ValueError('Queue macOS qualification on the physical Apple Silicon Mac')
    if qualified.git('status', '--porcelain'):
        raise ValueError('Commit or set aside every change before queuing qualification')
    sha = qualified.git('rev-parse', 'HEAD')
    if expected is not None and sha != expected:
        raise ValueError('Checkout changed while preparing the push; retry with the intended commit checked out')
    tree = qualified.git('rev-parse', f'{sha}^{{tree}}')
    rustc = qualified.compiler()
    if '\nhost: aarch64-apple-darwin\n' not in '\n' + rustc + '\n':
        raise ValueError('macOS qualification requires an aarch64-apple-darwin compiler')
    qualified.macos_description(tree, rustc)
    repo = repository(remote)
    key = hashlib.sha256((rustc + '\n' + (repo or '')).encode()).hexdigest()[:16]
    job = directory() / f'{sha}-{key}'
    job.mkdir(parents=True, exist_ok=True)
    with (job / 'lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f'Mac qualification is already queued for {sha[:12]}. Log: {job / "log"}')
            return
        state = {'commit': sha, 'tree': tree, 'compiler': rustc, 'repository': repo}
        save(job, state, 'queued')
        # Hook-local Git variables must not redirect commands in the detached worktree
        # back to the developer's index, HEAD, or working directory.
        local_git = set(qualified.git('rev-parse', '--local-env-vars').splitlines())
        env = {key: value for key, value in os.environ.items() if key not in local_git}
        # Freeze the runner too: an edit just after the push must not change code
        # that the background Python process has not loaded yet.
        for name in ('macos_async.py', 'qualified.py'):
            (job / name).write_text(qualified.git('show', f'{sha}:scripts/check/{name}') + '\n')
        # The child inherits the lock, so duplicate pushes cannot start a second job.
        # Closing the parent's descriptor does not unlock the child's open file description.
        with (job / 'log').open('a') as log:
            try:
                subprocess.Popen([sys.executable, str(job / 'macos_async.py'), 'worker', str(job), str(lock.fileno()), str(ROOT)],
                                 cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, pass_fds=(lock.fileno(),), start_new_session=True)
            except OSError as error:
                save(job, state, 'failed', error=str(error))
                raise
    print(f'Mac qualification queued for {sha[:12]}; inspect it with just macos-status.')
    print(f'Log: {job / "log"}')
    if repo is None:
        print('This destination is not GitHub; the result will remain local.')


def publish(state, success):
    repo, sha = state['repository'], state['commit']
    if repo is None:
        return
    # An inert-only change can finish before the push makes its commit visible.
    # Retry only that visibility race; authentication and other failures need operator repair.
    for attempt in range(24):
        visible = subprocess.run(['gh', 'api', f'repos/{repo}/commits/{sha}/status', '--silent', '--hostname', 'github.com'],
                                 capture_output=True, text=True, timeout=30)
        if visible.returncode == 0:
            break
        if 'HTTP 404' not in visible.stderr or attempt == 23:
            raise ValueError(f'Cannot find the pushed commit on GitHub: {visible.stderr.strip()}')
        time.sleep(5)
    payload = {'context': qualified.MACOS_CONTEXT, 'state': 'success' if success else 'failure',
               'description': qualified.macos_description(state['tree'], state['compiler']) if success
               else 'Mac qualification failed; inspect just macos-status and its local log'}
    subprocess.run(['gh', 'api', '--method', 'POST', f'repos/{repo}/statuses/{sha}', '--input', '-', '--silent', '--hostname', 'github.com'],
                   input=json.dumps(payload), text=True, check=True, timeout=30)


def worker(job, descriptor):
    state = json.loads((job / 'state.json').read_text())
    print(f'Qualifying {state["commit"]} at {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}\n{state["compiler"]}', flush=True)
    success = False
    try:
        # Serialize heavy jobs and create fresh source mtimes only after the previous build ends.
        # Cargo may share compatible dependencies; its profiles/features/flags still separate kernels.
        with (job.parent / 'build.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            save(job, state, 'running')
            checkout = job / 'checkout'
            if checkout.exists():
                subprocess.run(['git', 'worktree', 'remove', '--force', str(checkout)], cwd=ROOT, check=True)
            subprocess.run(['git', 'worktree', 'add', '--detach', str(checkout), state['commit']], cwd=ROOT, check=True)
            try:
                qualified.ROOT = checkout
                if qualified.compiler() != state['compiler']:
                    raise ValueError('Compiler changed after qualification was queued; queue it again')
                reason = qualified.qualified('macos', state['tree'])
                if reason:
                    print(reason, flush=True)
                else:
                    env = {**os.environ, 'CARGO_TARGET_DIR': str(job.parent / 'target'), 'JUST_TIME': 'true',
                           'CARGO_TERM_COLOR': 'never'}
                    subprocess.run(['just', 'check-macos'], cwd=checkout, env=env, check=True)
                if (qualified.git('status', '--porcelain') or qualified.git('rev-parse', 'HEAD^{tree}') != state['tree']
                        or qualified.compiler() != state['compiler']):
                    raise ValueError('Source or compiler changed during qualification; no pass recorded')
                records = qualified.records('macos')
                records.mkdir(parents=True, exist_ok=True)
                (records / state['tree']).write_text(state['compiler'])
                success = True
            finally:
                qualified.ROOT = ROOT
                subprocess.run(['git', 'worktree', 'remove', '--force', str(checkout)], cwd=ROOT, check=True)
        save(job, state, 'qualified')
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        success = False
        save(job, state, 'failed', error=str(error))
        print(f'Mac qualification failed: {error}', flush=True)
    try:
        publish(state, success)
        if success:
            save(job, state, 'published' if state['repository'] else 'passed')
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        save(job, state, 'publication-failed', error=str(error), qualified=success)
        print(f'GitHub status was not published: {error}. Repair the cause and run just qualify-macos.', flush=True)
    finally:
        os.close(descriptor)


def status():
    for record in sorted(directory().glob('*/state.json'), key=lambda path: path.stat().st_mtime):
        state = json.loads(record.read_text())
        phase = state['phase']
        if phase in ('queued', 'running', 'qualified'):
            with (record.parent / 'lock').open('w') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    phase = 'interrupted; retry with just qualify-macos'
                except BlockingIOError:
                    pass
        print(f'{state["commit"][:12]} {phase} ({state["repository"] or "local only"})')
        if state.get('error'):
            print(f'  {state["error"]}')
        print(f'  Log: {record.parent / "log"}')


def main():
    global ROOT
    match sys.argv[1:]:
        case ['queue', destination, expected]:
            queue(destination, expected)
        case ['queue', *destination] if len(destination) <= 1:
            queue(destination[0] if destination else qualified.git('remote', 'get-url', '--push', 'origin'))
        case ['status']:
            status()
        case ['worker', job, descriptor, root]:
            ROOT = qualified.ROOT = Path(root)
            worker(Path(job), int(descriptor))
        case _:
            raise ValueError('usage: macos_async.py {queue [remote-url]|status}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit(f'Mac qualification: {error}') from error
