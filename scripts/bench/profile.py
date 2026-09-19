"""Capture one exact production benchmark case with Samply or native Linux perf."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
  import resource
except ImportError:  # Windows local Samply profiles do not use native perf facts.
  resource = None

from execution import (
  build,
  build_command,
  build_identity,
  discover,
  execute,
  exit_code,
  source_evidence,
  unchanged,
  write_json,
)
from settings import PROFILE_CAPTURE_MAX_SECONDS, load

PROFILE_KIND = 'rscrypto.cross.profile'
SAMPLE_FREQUENCY = 99


class ProfileUnavailable(RuntimeError):
  """The native host cannot collect the requested profile."""


class ProfileIncomplete(RuntimeError):
  """The native host retained evidence but did not complete the capture."""


CAPTURE_ERRORS = (OSError, subprocess.CalledProcessError, ProfileIncomplete)


def probe(command: list[str], env: dict | None = None) -> dict:
  try:
    result = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    return {'command': command, 'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
  except OSError as error:
    return {'command': command, 'exit_code': 127, 'stdout': '', 'stderr': str(error)}


def host_facts(perf: str | None, env: dict) -> dict:
  facts = {
    'uname': probe(['uname', '-a'], env),
    'lscpu': probe(['lscpu'], env),
    'perf': probe([perf, '--version'], env) if perf is not None else {
      'command': ['perf', '--version'],
      'exit_code': 127,
      'stdout': '',
      'stderr': 'perf was not found in PATH',
    },
    'affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
    'limits': {},
    'files': {},
  }
  if resource is not None:
    for name in ('RLIMIT_AS', 'RLIMIT_CORE', 'RLIMIT_DATA', 'RLIMIT_FSIZE', 'RLIMIT_MEMLOCK',
                 'RLIMIT_NOFILE', 'RLIMIT_NPROC', 'RLIMIT_RSS', 'RLIMIT_STACK'):
      if hasattr(resource, name):
        facts['limits'][name] = list(resource.getrlimit(getattr(resource, name)))
  for name in ('/proc/cpuinfo', '/proc/self/status', '/proc/sys/kernel/perf_event_mlock_kb',
               '/proc/sys/kernel/perf_event_paranoid', '/proc/sys/kernel/kptr_restrict'):
    path = Path(name)
    if path.is_file():
      try:
        facts['files'][name] = path.read_text()
      except OSError as error:
        facts['files'][name] = {'error': str(error)}
  return facts


def capabilities(perf: str, root: Path, env: dict,
                 allow_callchain: bool) -> tuple[str | None, bool, list[dict]]:
  commands = []
  noop = [sys.executable, '-c',
          'import time\nend = time.monotonic() + 0.1\nwhile time.monotonic() < end: pass']
  sample_event = None
  callchain = False
  with tempfile.TemporaryDirectory(prefix='perf-probe-', dir=root) as directory:
    output_path = Path(directory) / 'perf.data'
    output = str(output_path)
    for event in ('cycles:u', 'cpu-clock:u'):
      for with_callchain in ((True, False) if allow_callchain else (False,)):
        output_path.unlink(missing_ok=True)
        command = [perf, 'record', '--quiet', '--output', output, '--event', event,
                   '--freq', str(SAMPLE_FREQUENCY)]
        if with_callchain:
          command += ['--call-graph', 'dwarf']
        result = probe([*command, '--', *noop], env)
        commands.append(result)
        if result['exit_code'] == 0 and output_path.is_file() and output_path.stat().st_size > 0:
          sample_event, callchain = event, with_callchain
          break
      if sample_event is not None:
        break
  return sample_event, callchain, commands


def unavailable_cause(probes: list[dict]) -> str:
  diagnostic = '\n'.join(probe['stderr'] for probe in probes)
  paranoid = re.search(r'perf_event_paranoid(?: setting)? is\s+(-?\d+)', diagnostic)
  if 'Access to performance monitoring' in diagnostic or 'Permission denied' in diagnostic:
    setting = f'perf_event_paranoid={paranoid.group(1)}; ' if paranoid else ''
    return f'native perf access is denied ({setting}runner needs CAP_PERFMON or owner configuration)'
  if 'Segmentation fault' in diagnostic:
    return 'native perf crashed during sampling (runner needs a kernel-compatible perf build)'
  if re.search(r'not supported|No such device', diagnostic, re.IGNORECASE):
    return 'native perf exposes neither hardware cycles nor software CPU clock sampling'
  return 'native perf cannot record hardware cycles or software CPU clock'


def collector(perf: str, event: str, callchain: bool, host: dict) -> dict:
  return {
    'path': perf,
    'version': host['perf']['stdout'].strip(),
    'event': event,
    'frequency': SAMPLE_FREQUENCY,
    'callchain': 'dwarf' if callchain else 'flat',
    'access': 'runner',
  }


def perf_capture(root: Path, command: list[str], env: dict) -> tuple[str, str | None, dict]:
  perf = shutil.which('perf', path=env.get('PATH'))
  host = host_facts(perf, env)
  write_json(root / 'host.json', host)
  if perf is None:
    raise ProfileUnavailable('native runner does not provide perf')
  riscv_host = os.uname().machine.startswith('riscv')
  allow_callchain = not riscv_host
  sample_event, callchain, probes = capabilities(perf, root, env, allow_callchain)
  write_json(root / 'capabilities.json', {'sample_event': sample_event,
                                         'callchain': 'dwarf' if callchain else 'flat',
                                         'access': 'runner', 'probes': probes})
  if sample_event is None:
    raise ProfileUnavailable(unavailable_cause(probes))

  collector_info = collector(perf, sample_event, callchain, host)
  record_output = root / 'perf.data'
  record_output.touch()
  record = [perf, 'record', '--output', str(record_output), '--event', sample_event,
            '--freq', str(SAMPLE_FREQUENCY)]
  if callchain:
    record += ['--call-graph', 'dwarf']
  try:
    execute([*record, '--', *command], root / 'output.txt', env=env)
    if not (root / 'perf.data').is_file() or (root / 'perf.data').stat().st_size == 0:
      raise ProfileIncomplete('perf record produced no raw profile')
  except CAPTURE_ERRORS as error:
    cause = f'perf record failed with exit code {exit_code(error)}'
    return 'failed', cause, collector_info

  try:
    report = execute([perf, 'report', '--stdio', '--no-inline', '--percent-limit', '0.5',
                      '--input', str(root / 'perf.data')],
                     root / 'output.txt', env=env, capture=True)
    if not report.strip() or re.search(r'# Samples:\s+0\b', report):
      raise ProfileIncomplete('perf report produced no sampled text output')
    if riscv_host and re.search(r'^\s+\d+(?:\.\d+)?%.*\[\.\]\s+\$[dx]\S*', report, re.MULTILINE):
      raise ProfileIncomplete('perf report exposed RISC-V mapping symbols instead of functions')
    (root / 'perf-report.txt').write_text(report)
  except ProfileIncomplete as error:
    return 'partial', str(error), collector_info
  except (OSError, subprocess.CalledProcessError) as error:
    return 'partial', f'perf report failed with exit code {exit_code(error)}', collector_info
  return 'complete', None, collector_info


def request(args, entry) -> tuple[list[dict], dict]:
  rows = [entry]
  settings = {'seconds': args.seconds, 'collector': 'perf'}
  return rows, settings


def prepared_profile(args, entry) -> None:
  from transfer import KIND, bundle, consume, prepare

  rows, settings = request(args, entry)
  repository = Path.cwd()
  if args.prepare_archive:
    prepare(repository, args.target, args.prepare_archive.resolve(), Path('target/profiles').resolve(), rows, settings)
    return

  parent = Path('target/profiles').resolve()
  parent.mkdir(parents=True, exist_ok=True)
  root = Path(tempfile.mkdtemp(prefix=entry['binary'] + '-', dir=parent))
  source = bundle.source_identity(repository)
  capture_request = {'case': args.case, **settings}
  metadata = {
    'created': datetime.now(timezone.utc).isoformat(),
    'request': capture_request,
    'target': args.target,
    'features': entry['features'],
    'status': 'recording',
  }
  write_json(root / 'request.json', capture_request)
  failure = None
  try:
    artifacts, compatibility = consume(repository, args.target, args.run_archive.resolve(), root / 'input', rows, settings)
    artifact = artifacts[tuple(build_command(entry['binary'], entry['features']))]
    env = dict(os.environ) | {'CRITERION_HOME': str(root / 'criterion')}
    cases = discover(artifact, '', root / 'output.txt', env)
    if cases.count(args.case) != 1:
      raise ValueError('choose one exact case discovered from the native artifact')
    write_json(root / 'cases.json', [args.case])
    command = [artifact['path'], '--bench', '--profile-time', str(args.seconds), '--noplot']
    env |= {'RSCRYPTO_BENCH_CASES': str(root / 'cases.json')}
    metadata.update(artifact=artifact, compatibility=compatibility, command=command)
    unchanged(artifact)
    status, cause, collector = perf_capture(root, command, env)
    metadata.update(status=status, cause=cause, collector=collector)
    unchanged(artifact)
    bundle.verify(repository, root / 'input', KIND, args.target)
    if status != 'complete':
      failure = ProfileIncomplete(cause or 'native profile is incomplete')
  except ProfileUnavailable as error:
    metadata.update(status='unavailable', cause=str(error))
    failure = error
  except BaseException as error:  # noqa: BLE001 - retain every failed native capture outcome.
    metadata.update(status='failed', cause=str(error), exit_code=exit_code(error))
    failure = error
  finally:
    metadata['exit_code'] = 0 if failure is None else exit_code(failure)
    write_json(root / 'metadata.json', metadata)
    (root / 'status.txt').write_text(
      f"state={metadata['status']}\nexit_code={metadata['exit_code']}\n"
      + (f"cause={metadata['cause']}\n" if metadata.get('cause') else '')
    )
    try:
      bundle.seal(repository, root, PROFILE_KIND, args.target, source, metadata)
    except BaseException as error:  # noqa: BLE001 - a sealing failure must replace a false complete status.
      (root / 'bundle.json').unlink(missing_ok=True)
      metadata.update(status='failed', cause=f'evidence sealing failed: {error}', exit_code=exit_code(error))
      write_json(root / 'metadata.json', metadata)
      (root / 'status.txt').write_text(
        f"state=failed\nexit_code={metadata['exit_code']}\ncause={metadata['cause']}\n"
      )
      failure = error
    print(f'Profile directory: {root}', flush=True)
  if failure is not None:
    raise failure


def local_profile(args, entry) -> None:
  if not args.list and (not args.case or shutil.which('samply') is None):
    raise ValueError('profiling requires an exact case and samply')
  with tempfile.TemporaryDirectory(prefix='rscrypto-profile-build-') as directory:
    temporary = Path(directory)
    env = dict(os.environ) | {'CRITERION_HOME': str(temporary / 'criterion')}
    artifact = build(build_command(entry['binary'], entry['features']), temporary / 'output.txt', env)
    cases = discover(artifact, '', temporary / 'output.txt', env)
    if args.list:
      print('\n'.join(cases))
      return
    if cases.count(args.case) != 1:
      raise ValueError('choose one exact case from --list')
    parent = Path('target/profiles').resolve()
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=entry['binary'] + '-', dir=parent))
    shutil.copyfile(temporary / 'output.txt', root / 'output.txt')
    source_evidence(root)
    write_json(root / 'cases.json', [args.case])
    command = [artifact['path'], '--bench', '--profile-time', str(args.seconds), '--noplot']
    metadata = {'created': datetime.now(timezone.utc).isoformat(), 'case': args.case,
                'artifact': artifact, 'compatibility': build_identity(), 'command': command, 'status': 'recording',
                'budget_seconds': load()['max_run_seconds'],
                'samply': subprocess.check_output(['samply', '--version'], text=True).strip()}
    write_json(root / 'metadata.json', metadata)
    status = 1
    try:
      unchanged(artifact)
      execute(['samply', 'record', '--save-only', '--output', str(root / 'profile.json.gz'), *command],
              root / 'output.txt', env=dict(os.environ) | {
                'CRITERION_HOME': str(root / 'criterion'), 'RSCRYPTO_BENCH_CASES': str(root / 'cases.json'),
              })
      unchanged(artifact)
      if not (root / 'profile.json.gz').is_file():
        raise ValueError('samply produced no profile')
      status = 0
    except BaseException as error:
      status = exit_code(error)
      raise
    finally:
      metadata.update(status='complete' if status == 0 else 'failed', exit_code=status)
      write_json(root / 'metadata.json', metadata)
      print(f'Profile directory: {root}', flush=True)


def profile(args, entry) -> None:
  if not math.isfinite(args.seconds) or not 1 <= args.seconds <= load()['max_run_seconds']:
    raise ValueError('profile duration must be within the configured run budget')
  transferred = args.prepare_archive or args.run_archive or args.target
  if transferred:
    from cross_build import LINUX_TARGETS
    if args.target not in LINUX_TARGETS or not (args.prepare_archive or args.run_archive) or args.list or not args.case:
      raise ValueError('profile transfer requires an exact case, supported target, and prepare/run archive')
    if args.seconds > PROFILE_CAPTURE_MAX_SECONDS:
      raise ValueError(f'transferred profile duration must be at most {PROFILE_CAPTURE_MAX_SECONDS} seconds')
    prepared_profile(args, entry)
  else:
    local_profile(args, entry)
