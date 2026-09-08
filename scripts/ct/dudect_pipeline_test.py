#!/usr/bin/env python3
"""Exercise shared preparation and isolated measurements through CT orchestration."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import full
from dudect_report import write_report


def main():
  repository = Path(__file__).resolve().parents[2]
  with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    runner = root / "runner.py"
    runner.write_text(f"#!{sys.executable}\n" + '''
import os, sys
from pathlib import Path
name = sys.argv[sys.argv.index('--filter') + 1]
with Path('executions').open('a') as log:
  log.write(name + '\\n')
mode = Path('mode').read_text()
if mode == 'failure':
  sys.exit(1)
if mode == 'missing':
  sys.exit(0)
if mode == 'partial':
  Path(sys.argv[2]).write_text('benchname,sequence,class,runtime_ns\\n' + name + ',0,')
  sys.exit(0)
name = 'manifest_' + name
count = int(os.environ['RSCRYPTO_CT_DUDECT_SAMPLES'])
Path(sys.argv[2]).write_text('benchname,sequence,class,runtime_ns\\n' + ''.join(
  f'{name},{i},{i % 2},100\\n' for i in range(count)))
print(f'bench {name} ... : n == +0.01M, max t = +1.00, max tau = +0.01, (5/tau)^2 = 250000')
''')
    runner.chmod(0o755)
    executable = runner
    if os.name == "nt":
      executable = root / "runner.cmd"
      executable.write_text(f'@"{sys.executable}" "{runner}" %*\n')
    (root / "runner-path").write_text(executable.name)
    cases = [
      {"name": "manifest_" + name, "filter": name, "primitive": "fixture", "samples": 4}
      for name in ("alpha", "beta")
    ]
    prep = root / "prepare.py"
    prep.write_text('''
import hashlib, json, shutil, sys
from pathlib import Path
with Path('preparations').open('a') as log:
  log.write('prepare\\n')
if Path('mode').read_text() == 'prepare-fail':
  sys.exit(2)
shared = Path(sys.argv[sys.argv.index('--shared-dir') + 1])
shared.mkdir()
executable = Path('runner-path').read_text()
binary = shared / executable
shutil.copy2(executable, binary)
metadata = {'binary': {'path': str(binary), 'sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}}
manifest = {'manifest_' + name: {'primitive': 'fixture', 'gate': 'required', 'left_class': 'left', 'right_class': 'right'} for name in ('alpha', 'beta')}
(shared / 'prepared.json').write_text(json.dumps({'metadata': metadata, 'manifest_cases': manifest}))
''')

    def invoke(mode):
      (root / "mode").write_text(mode)
      with patch.object(full, "shell_script", return_value=[sys.executable, str(prep)]), patch.object(
        full, "python_script", return_value=[sys.executable, str(repository / "scripts/ct/dudect_execute.py")],
      ):
        return full.run_dudect_cases(root, root / "out", root / "logs", "fixture", "release", cases, 10.0, 10)

    run, preparation, rows = invoke("success")
    assert preparation.status == "pass"
    assert [row["status"] for row in rows] == ["pass", "pass"], rows
    assert (root / "preparations").read_text().splitlines() == ["prepare"]
    assert (root / "executions").read_text().splitlines() == ["alpha", "beta"]
    assert rows[0]["binary"] == rows[1]["binary"]
    for row in rows:
      assert Path(row["report"]).parent.name.startswith(row["name"] + "-")
      assert {item["name"] for item in row["artifacts"]} == {"dudect-report.json", "dudect-raw.csv", "dudect.stdout.txt"}
    snapshot = {str(path): path.read_bytes() for path in run.rglob('*') if path.is_file()}
    for mode in ("failure", "missing", "partial"):
      new_run, _, failed = invoke(mode)
      assert new_run != run
      assert all(row["status"] == "tooling-fail" for row in failed), failed
      assert all(row["report"] is None for row in failed)
    _, failed_preparation, failed = invoke("prepare-fail")
    assert failed_preparation.status == "fail" and not failed
    assert len((root / "executions").read_text().splitlines()) == 8
    (root / "mode").write_text("success")
    with patch.object(full, "python_script", return_value=[sys.executable, str(repository / "scripts/ct/dudect_execute.py")]):
      timed_out = full.dudect_case_result(root, root / "logs", 4, 10.0, cases[0], 0, run / "shared/prepared.json")
    assert timed_out["status"] == "timeout" and timed_out["report"] is None
    assert all(Path(path).read_bytes() == data for path, data in snapshot.items())
    interrupted = root / "interrupted.json"
    with patch.object(Path, "replace", side_effect=OSError("interrupted publication")):
      try:
        write_report(interrupted, {"cases": [{"status": "pass"}]})
      except OSError:
        pass
      else:
        raise AssertionError("interrupted report unexpectedly published")
    assert not interrupted.exists()
    records = full.collect_artifact_records((root / "out").resolve(), run)
    assert len([record for record in records if Path(record["path"]).name == executable.name]) == 1
    assert all(str(run.relative_to((root / 'out').resolve())) in record['path'] for record in records)


if __name__ == "__main__":
  main()
