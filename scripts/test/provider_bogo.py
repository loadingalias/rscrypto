#!/usr/bin/env -S uv run python
"""Run the pinned BoGo TLS 1.2/1.3 profile against the rscrypto provider."""

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[2]
BORINGSSL = '9389a4af2c6c47a59ec39bfb51e4da2fe971d1bc'


def registry_version(dependency):
    """Return None for a Git pin, or the version of an exact registry pin.

    The shim must build from the rustls source that the provider links, so a Git
    pin needs a `rev`, a registry pin needs one `=X.Y.Z` version, and every other
    form is refused before anything is fetched or built.
    """
    if isinstance(dependency, str):
        dependency = {'version': dependency}
    if 'path' in dependency:
        raise ValueError('rustls is a path dependency, which names no upstream commit')
    if 'git' in dependency:
        if 'rev' not in dependency:
            raise ValueError('the rustls Git pin has no `rev`; a branch or tag can move')
        return None
    if 'version' not in dependency:
        raise ValueError('rustls has no version')
    exact = re.fullmatch(r'=\s*(\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?)', dependency['version'])
    if exact is None:
        raise ValueError(f"rustls version {dependency['version']!r} is a range; pin one version as `=X.Y.Z`")
    return exact[1]


def linked_rustls(metadata):
    """Return the rustls package that `cargo metadata` resolves for the provider."""
    packages = {package['id']: package for package in metadata['packages']}
    provider = next(node for node in metadata['resolve']['nodes'] if packages[node['id']]['name'] == 'rscrypto-rustls')
    return packages[next(dependency['pkg'] for dependency in provider['deps'] if dependency['name'] == 'rustls')]


def vcs_commit(crate):
    """Return the commit that Cargo recorded when it packaged the registry crate in `crate`."""
    record = crate / '.cargo_vcs_info.json'
    if not record.is_file():
        raise ValueError(f'{record} is missing: the published crate records no source commit')
    git = json.loads(record.read_text()).get('git', {})
    if git.get('dirty'):
        raise ValueError(f'{record}: the crate was packaged from a modified checkout, so no commit holds its source')
    if not re.fullmatch('[0-9a-f]{40}', git.get('sha1', '')):
        raise ValueError(f'{record} records no Git commit')
    return git['sha1']


def exclusions(profiles):
    """Select scope from declared requirements, never observed outcomes."""
    result = {}
    for profile in profiles:
        reason = None
        if profile['Protocol'] != 'TLS':
            reason = f"The TLS campaign does not test {profile['Protocol']}."
        for field in ('MaxVersion', 'ExpectedVersion', 'ResumeMaxVersion', 'ResumeExpectedVersion'):
            value = profile[field]
            if value and value < 0x0303:
                reason = f'Declared {field}=0x{value:04x} requires a version below TLS 1.2.'
        # A resumption case must first establish the original session. Negative
        # full-handshake cases still run so the peer independently checks refusal.
        initial_tls12 = any(0 < profile[field] <= 0x0303
                            for field in ('MaxVersion', 'ExpectedVersion'))
        if initial_tls12 and profile['NoExtendedMasterSecret'] and (
                not profile['ShouldFail'] or profile['ResumeSession']):
            reason = 'The initial handshake requires a legacy master secret; this provider requires EMS.'
        resumed_versions = (profile['ResumeMaxVersion'] or profile['MaxVersion'],
                            profile['ResumeExpectedVersion'] or profile['ExpectedVersion'])
        resumed_tls12 = any(0 < version <= 0x0303 for version in resumed_versions)
        if (profile['ResumeSession'] and resumed_tls12 and profile['ResumeNoExtendedMasterSecret']
                and not profile['ShouldFail']):
            reason = 'The resumed handshake requires a legacy master secret; this provider requires EMS.'
        if reason:
            result[profile['Name']] = reason
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--work', type=Path, help='new directory for the upstream checkouts')
    parser.add_argument('--test', default='*', help='case pattern; default is the full profile')
    args = parser.parse_args()
    dependency = tomllib.loads((ROOT / 'rscrypto-rustls/Cargo.toml').read_text())['dependencies']['rustls']
    version = registry_version(dependency)
    if version is None:
        pin = {'rustls': dependency['rev'], 'rustls_repository': dependency['git']}
    else:
        package = linked_rustls(json.loads(subprocess.check_output(
            ['cargo', 'metadata', '--locked', '--format-version=1'], cwd=ROOT)))
        pin = {'rustls': vcs_commit(Path(package['manifest_path']).parent),
               'rustls_repository': package['repository'], 'rustls_version': version}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = (args.output or ROOT / 'benchmark_results' / f'provider-bogo-{stamp}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    work = args.work.resolve() if args.work else Path(tempfile.mkdtemp(prefix='provider-bogo-'))
    if args.work:
        work.mkdir(parents=True, exist_ok=False)
    rustls, runner = work / 'rustls', work / 'boringssl'
    commands = []
    env = dict(os.environ, BOGO_SHIM_PROVIDER='rscrypto')

    def run(name, command, cwd=ROOT, *, data=None, check=True, timeout=600):
        commands.append({'name': name, 'cwd': str(cwd), 'argv': list(map(str, command)),
                         'env': {'BOGO_SHIM_PROVIDER': 'rscrypto'}})
        (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        with (output / (name + '.log')).open('wb') as log:
            result = subprocess.run(command, cwd=cwd, env=env, input=data,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        if check and result.returncode:
            raise RuntimeError(f'{name} exited {result.returncode}; see {output}')
        return result.returncode

    versions = {'host': platform.platform(), 'source': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        **pin, 'boringssl': BORINGSSL, 'test': args.test, 'work': str(work)}
    for tool, arguments in (('go', ['version']), ('rustc', ['-Vv'])):
        versions[tool] = subprocess.check_output([tool, *arguments], cwd=ROOT, text=True).strip()
    if not versions['go'].startswith('go version go1.27.3 '):
        raise RuntimeError('this pinned campaign requires Go 1.27.3')
    (output / 'provenance.json').write_text(json.dumps(versions, indent=2) + '\n')
    (output / 'source.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=ROOT))
    shutil.copytree(ROOT / 'rscrypto-rustls', output / 'source/rscrypto-rustls')
    paths = ['Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml']
    paths += ['scripts/test/provider_bogo' + suffix for suffix in ('.py', '.patch', '_runner.patch', '_disabled.json')]
    for path in paths:
        destination = output / 'source' / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, destination)
    for name, repository, revision, directory in (
        ('rustls', pin['rustls_repository'], pin['rustls'], rustls),
        ('runner', 'https://github.com/rustls/boringssl.git', BORINGSSL, runner),
    ):
        run(name + '-init', ['git', 'init', str(directory)])
        run(name + '-fetch', ['git', 'fetch', '--depth=1', repository, revision], directory)
        run(name + '-checkout', ['git', 'checkout', '--detach', revision], directory)
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=directory, text=True).strip()
        if actual != revision:
            raise RuntimeError(f'{name} revision differs from pin')
    patch = (ROOT / 'scripts/test/provider_bogo.patch').read_text().replace('@RSCRYPTO_ROOT@', str(ROOT))
    run('shim-patch', ['git', 'apply', '-'], rustls, data=patch.encode())
    run('runner-patch', ['git', 'apply', str(ROOT / 'scripts/test/provider_bogo_runner.patch')], runner)
    compiler = tomllib.loads((ROOT / 'rust-toolchain.toml').read_text())['toolchain']['channel']
    run('shim-build', ['cargo', '+' + compiler, 'build', '--manifest-path', str(rustls / 'bogo/Cargo.toml'),
                       '--features', 'rscrypto'], rustls, timeout=1200)
    run('runner-build', ['go', 'test', '-c', '-o', str(runner / 'runner.test')], runner / 'ssl/test/runner')
    shutil.copyfile(rustls / 'Cargo.lock', output / 'shim-Cargo.lock')
    metadata = json.loads(subprocess.check_output(
        ['cargo', '+' + compiler, 'metadata', '--no-deps', '--format-version=1'], cwd=rustls))
    shim = str(Path(metadata['target_directory']) / 'debug/bogo')
    encoded = base64.b64encode((ROOT / 'rscrypto-rustls/tests/vectors/credentials/rsa3072-leaf-key.der').read_bytes()).decode()
    rsa = output / 'rsa3072.pem'
    rsa.write_text('-----BEGIN PRIVATE KEY-----\n' + '\n'.join(
        encoded[i:i + 64] for i in range(0, len(encoded), 64)) + '\n-----END PRIVATE KEY-----\n')
    command = [str(runner / 'runner.test'), '-shim-path', shim, '-rsa-key-file', str(rsa), '-require-ems-profile']
    run('profiles', [*command, '-profile-output', str(output / 'profiles.json')], runner / 'ssl/test/runner')
    disabled = exclusions(json.loads((output / 'profiles.json').read_text()))
    disabled.update(json.loads((ROOT / 'scripts/test/provider_bogo_disabled.json').read_text()))
    (output / 'profile-exclusions.json').write_text(json.dumps(disabled, indent=2) + '\n')
    config = json.loads(subprocess.check_output(
        ['cpp', '-P', '-DAWS_LC_RS', str(rustls / 'bogo/config.json.in')], cwd=rustls))
    (output / 'upstream-config.json').write_text(json.dumps(config, indent=2) + '\n')
    config['DisabledTests'].update(disabled)
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    status = run('runner', [*command, '-shim-config', str(output / 'config.json'), '-pipe',
                          '-num-workers', '2', '-test.timeout', '600s', '-allow-unimplemented',
                          '-test', args.test, '-json-output', str(output / 'results.json')],
                 runner / 'ssl/test/runner', check=False, timeout=660)
    result = json.loads((output / 'results.json').read_text())
    counts = Counter(value['actual'] for value in result['tests'].values())
    (output / 'summary.json').write_text(json.dumps({'exit_code': status, 'counts': counts}, indent=2) + '\n')
    print(f'BoGo exit={status}: {dict(counts)}; evidence: {output}', flush=True)
    # Preserve upstream failures as failures, without result-driven exclusions.
    return status


if __name__ == '__main__':
    raise SystemExit(main())
