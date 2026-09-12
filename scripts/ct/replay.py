"""Repeat one prepared CT case on a pinned native CPU, retaining every result."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transfer import consume
from dudect_execute import measure
from dudect_report import write_report
from manifest import dudect_sample_count


def snapshot(cpu):
    paths = [Path('/proc/cpuinfo'), Path('/proc/loadavg'), Path('/proc/stat')]
    paths += sorted(Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq').glob('*'))
    values = {}
    for path in paths:
        if path.is_file():
            try:
                values[str(path)] = path.read_text()
            except OSError as error:
                values[str(path)] = str(error)
    return {'utc': datetime.now(timezone.utc).isoformat(), 'cpu': cpu,
            'affinity': sorted(os.sched_getaffinity(0)), 'files': values,
            'processes': subprocess.check_output(
                ['ps', '-eo', 'pid,ppid,psr,pcpu,comm'], text=True)}


def repeat(prepared, args, cpu):
    case = prepared['manifest_cases'][args.case]
    samples = dudect_sample_count(case)
    results = []
    for index in range(3):
        directory = args.out / f'repetition-{index + 1}'
        directory.mkdir()
        write_report(directory / 'before.json', snapshot(cpu))
        invocation = argparse.Namespace(
            prepared=args.prepared, evidence_dir=directory, samples=samples,
            smoke=False, threshold=10.0, filter=args.case,
            timeout=case.get('timeout_seconds', 300), latest=None)
        print(f'Repetition {index + 1}/3: {args.case}, {samples} samples, CPU {cpu}', flush=True)
        status = measure(prepared, invocation)
        write_report(directory / 'after.json', snapshot(cpu))
        results.append({'repetition': index + 1, 'exit_code': status})
        write_report(args.out / 'replay.json', {
            'diagnostic_only': True, 'case': args.case, 'repetitions': results,
            'source': prepared['metadata']['transfer']['source'],
            'binary': prepared['metadata']['binary'],
            'note': 'All three planned repetitions run even after timing failures; no result is discarded.'})
        if status not in (0, 1):
            return status
    return int(any(row['exit_code'] for row in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    _, args.prepared = consume(args.source_root.resolve(), args.out,
                               args.archive.resolve(), 'riscv64gc-unknown-linux-gnu')
    prepared = json.loads(args.prepared.read_text())
    if args.case not in prepared['manifest_cases']:
        raise ValueError('case is absent from the prepared manifest')
    # Pin the controller and its measurement child; leave system policy unchanged.
    allowed = os.sched_getaffinity(0)
    cpu = min(allowed)
    os.sched_setaffinity(0, {cpu})
    try:
        return repeat(prepared, args, cpu)
    finally:
        os.sched_setaffinity(0, allowed)


if __name__ == '__main__':
    raise SystemExit(main())
