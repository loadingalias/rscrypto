"""Capture one exact case using the benchmark build and discovery path."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from execution import build, build_command, build_identity, discover, execute, exit_code, unchanged, write_json, source_evidence
from settings import load


def profile(args, target) -> None:
  if not math.isfinite(args.seconds) or not 1 <= args.seconds <= load()['max_run_seconds']:
    raise ValueError('profile duration must be within the configured run budget')
  if not args.list and (not args.case or shutil.which('samply') is None):
    raise ValueError('profiling requires an exact case and samply')
  with tempfile.TemporaryDirectory(prefix='rscrypto-profile-build-') as directory:
    temporary = Path(directory)
    env = dict(os.environ) | {'CRITERION_HOME': str(temporary / 'criterion')}
    artifact = build(build_command(target['binary'], target['features']), temporary / 'output.txt', env)
    cases = discover(artifact, '', temporary / 'output.txt', env)
    if args.list:
      print('\n'.join(cases))
      return
    if cases.count(args.case) != 1:
      raise ValueError('choose one exact case from --list')
    parent = Path('target/profiles').resolve()
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=target['binary'] + '-', dir=parent))
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
