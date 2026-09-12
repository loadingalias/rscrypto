"""Transfer benchmark executables; discover and measure only on native hardware."""

import copy
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

from execution import build, build_command, build_identity, digest, exit_code, hardware
from evidence import collect
from cross_build import TARGETS, environment, require_host, verify_elf
import evidence_bundle as bundle

KIND = 'rscrypto.cross.bench'


def prepare(root, target, archive, output, rows, settings):
    if (platform.system(), platform.machine()) != ('Linux', 'x86_64'):
        raise ValueError('benchmark preparation requires Linux x86-64')
    if archive.exists():
        raise ValueError(f'refusing to overwrite benchmark evidence: {archive}')
    os.environ.update(environment(target))
    source = bundle.source_identity(root)
    parent = output / 'preparation' / target
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='build-', dir=parent))
    print(f'Preparation directory: {directory}', flush=True)
    try:
        compiler_path = shutil.which(TARGETS[target][1] + '-gcc')
        if compiler_path is None:
            raise FileNotFoundError('install ci-cross-build tooling before preparing benchmarks')
        compiler = Path(compiler_path).resolve(strict=True)
        metadata = {'rows': rows, 'settings': settings, 'build': build_identity(), 'artifacts': [],
                    'linker': {'path': str(compiler), 'sha256': digest(compiler),
                               'version': subprocess.check_output([str(compiler), '--version'], text=True)}}
        seen = set()
        for row in rows:
            command = build_command(row['binary'], row['features'])
            if tuple(command) in seen:
                continue
            seen.add(tuple(command))
            artifact = build([*command, '--target', target], directory / 'output.txt', dict(os.environ))
            binary = directory / 'bin' / str(len(seen)) / row['binary']
            binary.parent.mkdir(parents=True)
            shutil.copy2(artifact['path'], binary)
            verify_elf(binary, target)
            if digest(binary) != artifact['sha256']:
                raise ValueError('benchmark executable changed during preparation')
            metadata['artifacts'].append({'configuration': command, 'artifact': artifact,
                                          'binary': binary.relative_to(directory).as_posix()})
        (directory / 'status.txt').write_text('state=prepared\n')
        bundle.seal(root, directory, KIND, target, source, metadata)
        bundle.pack(directory, archive)
    except BaseException as error:
        (directory / 'status.txt').write_text(f'state=failed\nexit_code={exit_code(error)}\n')
        raise
    print(f'Benchmark preparation complete; native measurement remains required: {archive}', flush=True)


def consume(root, target, archive, directory, rows, settings):
    require_host(target)
    environment(target)  # Reject inherited compiler/runner overrides before discovery.
    bundle.unpack(archive, directory)
    manifest = bundle.verify(root, directory, KIND, target)
    metadata = manifest['metadata']
    if metadata['rows'] != rows or metadata['settings'] != settings:
        raise ValueError('prepared benchmark selection or sampling settings differ')
    artifacts = {}
    for row in metadata['artifacts']:
        key = tuple(row['configuration'])
        if key in artifacts:
            raise ValueError('duplicate prepared benchmark configuration')
        binary = directory / bundle.relative_path(row['binary'])
        verify_elf(binary, target)
        artifact = row['artifact']
        if (digest(binary) != artifact['sha256']
                or artifact['command'] != [*row['configuration'], '--target', target]):
            raise ValueError('prepared benchmark binary or build command differs')
        artifacts[key] = {**artifact, 'path': str(binary)}
    expected = {tuple(build_command(row['binary'], row['features'])) for row in rows}
    if set(artifacts) != expected:
        raise ValueError('prepared benchmark configurations are incomplete or unexpected')
    # Build provenance remains unchanged in the input manifest. Compatibility
    # uses the measurement host and runtime environment, never the x86 builder.
    compatibility = copy.deepcopy(metadata['build'])
    compatibility['host'] = hardware()
    compatibility['environment']['runtime'] = collect()['runtime']
    compatibility['target'] = target
    compatibility['linker'] = metadata['linker']
    return artifacts, compatibility
