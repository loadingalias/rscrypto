"""Transfer cross-compiled CT preparation without rebuilding the measured executable."""

import json
import platform
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
import evidence_bundle as bundle

from cross_build import require_host
KIND = 'rscrypto.cross.ct'
GATES = ('ct-artifacts', 'ct-validate-artifacts', 'ct-zeroization-sentinel')


def export(root, out_dir, shared, steps, identity, archive, target):
    if [step['name'] for step in steps] != list(GATES) or any(step['status'] != 'pass' for step in steps):
        raise ValueError('every CT preparation gate must pass before export')
    with tempfile.TemporaryDirectory(prefix='ct-export-', dir=root / 'target') as temporary:
        directory = Path(temporary)
        for name in ('artifacts', 'full/logs'):
            shutil.copytree(out_dir / name, directory / name)
        for name in ('provenance.json', 'evidence-index.json', 'artifact-hashes.txt',
                     'asm-heuristics.json', 'asm-heuristics.md', 'zeroization.json'):
            shutil.copy2(out_dir / name, directory / name)
        shutil.copytree(shared, directory / 'shared')
        bundle.seal(root, directory, KIND, target, identity,
                    {'steps': steps, 'build_root': str(root), 'out_dir': str(out_dir)})
        bundle.pack(directory, archive)


def consume(root, out_dir, archive, target):
    require_host(target)
    run = Path(tempfile.mkdtemp(prefix='import-', dir=out_dir))
    directory = run / 'original'
    bundle.unpack(archive, directory)
    manifest = bundle.verify(root, directory, KIND, target)
    steps = manifest['metadata']['steps']
    if [step['name'] for step in steps] != list(GATES) or any(step['status'] != 'pass' for step in steps):
        raise ValueError('transferred CT preparation gates are incomplete or failed')
    provenance = json.loads((directory / 'provenance.json').read_text())
    if provenance['target'] != target or provenance['profile'] != 'release':
        raise ValueError('wrong CT artifact target/profile')
    # Retain the producer bundle unchanged; only path-bearing runtime copies move.
    for name in ('provenance.json', 'evidence-index.json', 'artifact-hashes.txt',
                 'asm-heuristics.json', 'asm-heuristics.md', 'zeroization.json'):
        shutil.copy2(directory / name, out_dir / name)
    if (out_dir / 'artifacts').exists():
        shutil.rmtree(out_dir / 'artifacts')
    shutil.copytree(directory / 'artifacts', out_dir / 'artifacts')
    for step in steps:
        for field in ('stdout', 'stderr'):
            relative = Path(step[field]).relative_to(manifest['metadata']['out_dir'])
            step[field] = str(directory / relative)
    dudect_runs = out_dir / 'dudect/runs'
    dudect_runs.mkdir(parents=True, exist_ok=True)
    dudect_run = Path(tempfile.mkdtemp(prefix='transferred-', dir=dudect_runs))
    shared = dudect_run / 'shared'
    shutil.copytree(directory / 'shared', shared)
    prepared = json.loads((shared / 'prepared.json').read_text())
    metadata = prepared['metadata']
    if metadata['target'] != target or metadata['profile'] != 'release':
        raise ValueError('wrong timed binary target/profile')
    for key in ('binary', 'binary_disassembly', 'binary_symbols', 'linker_command_log'):
        row = metadata[key]
        path = shared / Path(row['path']).name
        if bundle.digest(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError(f'timed evidence changed: {key}')
        row['path'] = str(path)
    metadata['build_host'] = metadata['host']
    metadata['host'] = {'system': platform.system(), 'release': platform.release(),
                        'machine': platform.machine(), 'processor': platform.processor(),
                        'python': platform.python_version()}
    metadata['transfer'] = {'archive_sha256': bundle.digest(archive), 'source': manifest['source'],
                            'original': str(directory), 'manifest_sha256': bundle.digest(directory / 'bundle.json')}
    path = shared / 'prepared.json'
    path.chmod(0o644)
    path.write_text(json.dumps(prepared, indent=2) + '\n')
    path.chmod(0o444)
    return steps, path
