#!/usr/bin/env python3
"""Local release selection for tooling, Cargo manifests, and action references."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import cache
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib
import urllib.error
import urllib.parse
import urllib.request

from catalog import CATALOG, PLATFORMS, NATIVE_SOURCE_PLATFORMS, ROOT, read, rust_channel, validate

REPOS = {'cargo-rail': 'loadingalias/cargo-rail', 'cargo-binstall': 'cargo-bins/cargo-binstall',
         'llvm': 'llvm/llvm-project', 'cmake': 'Kitware/CMake', 'zig': None,
         'wasmtime': 'bytecodealliance/wasmtime',
         'git': 'git-for-windows/git', 'jq': 'jqlang/jq', 'powershell': 'PowerShell/PowerShell'}


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'rscrypto tooling (local release updater)'})
    with urllib.request.urlopen(request, timeout=120) as response:
        if not response.url.startswith('https://'):
            raise ValueError(f'non-HTTPS redirect: {url}')
        return response.read(), response.url


@cache
def api(url):
    if url.startswith('https://api.github.com/'):
        # gh owns credential handling; tokens never enter URLs or update output.
        result = subprocess.run(['gh', 'api', url.removeprefix('https://api.github.com/')],
                                capture_output=True, text=True)
        if result.returncode:
            raise ValueError(f'GitHub lookup failed: {url}\n{result.stderr.strip()}')
        return json.loads(result.stdout)
    return json.loads(fetch(url)[0])


def semver(value):
    match = re.fullmatch(r'v?(\d+)\.(\d+)\.(\d+)', value)
    return tuple(map(int, match.groups())) if match else None


def eligible_release(versions, *, rust_version=None):
    candidates = []
    for version in versions:
        number = semver(version['num'])
        if number is None or version.get('yanked'):
            continue
        required = version.get('rust_version')
        if rust_version and required:
            required_tuple = tuple(map(int, required.split('.')))
            if required_tuple > tuple(map(int, rust_version.split('.'))):
                continue
        candidates.append((number, version['num']))
    if not candidates:
        raise ValueError('no eligible stable crate release')
    return max(candidates)[1]


def crate_version(name, rust_version=None):
    versions = api(f'https://crates.io/api/v1/crates/{name}')['versions']
    try:
        return eligible_release(versions, rust_version=rust_version)
    except ValueError as error:
        raise ValueError(f'{name}: {error}') from error


@cache
def release(repo):
    result = api(f'https://api.github.com/repos/{repo}/releases/latest')
    if result['prerelease'] or result['draft']:
        raise ValueError(f'{repo}: latest release is not stable')
    return result


def pinned_url(url):
    data, resolved = fetch(url)
    return {'url': resolved, 'sha256': hashlib.sha256(data).hexdigest()}


def nasm_release():
    index = fetch('https://www.nasm.us/pub/nasm/releasebuilds/')[0].decode()
    versions = re.findall(r'href="(\d+\.\d+(?:\.\d+)?)/"', index)
    if not versions:
        raise ValueError('no stable NASM release')
    return max(versions, key=lambda version: tuple(map(int, version.split('.'))))


def github_asset(repo, name):
    matches = [asset for asset in release(repo)['assets'] if asset['name'] == name]
    if len(matches) != 1:
        raise ValueError(f'{repo}: required native release asset is missing: {name}')
    asset = matches[0]
    digest = asset.get('digest', '') or ''
    if re.fullmatch(r'sha256:[0-9a-f]{64}', digest):
        return {'url': asset['browser_download_url'], 'sha256': digest.removeprefix('sha256:')}
    return pinned_url(asset['browser_download_url'])


def toml_value(value):
    if isinstance(value, dict):
        return '{ ' + ', '.join(f'{json.dumps(k)} = {toml_value(v)}' for k, v in value.items()) + ' }'
    return json.dumps(value, ensure_ascii=False)


def catalog_text(data):
    lines = ['# Updated by just update on the local macOS workstation.',
             '# Installers consume these pins; they never select latest releases.',
             '# Ubuntu system packages are pinned together by archive snapshot.',
             '# Rust and its components are owned by rust-toolchain.toml.', '']
    for section, fields in data.items():
        lines.append(f'[{section}]')
        for key, value in fields.items():
            if section in PLATFORMS and key == 'assets':
                continue
            lines.append(f'{key} = {toml_value(value)}')
        lines.append('')
        if section in PLATFORMS:
            for name, asset in fields['assets'].items():
                lines.append(f'[{section}.assets.{name}]')
                lines.extend(f'{key} = {toml_value(value)}' for key, value in asset.items())
                lines.append('')
    return '\n'.join(lines)


def resolve_catalog():
    now = datetime.now(timezone.utc)
    data = read()
    cargo_names = list(data['cargo'])
    with ThreadPoolExecutor(max_workers=6) as pool:
        data['cargo'] = dict(zip(cargo_names, pool.map(crate_version, cargo_names)))
        list(pool.map(release, [repo for repo in REPOS.values() if repo]))
    data['updater'] = {name: api(f'https://pypi.org/pypi/{name}/json')['info']['version']
                       for name in ('tomlkit', 'PyYAML')}
    versions = {name: release(repo)['tag_name'].removeprefix('v').removeprefix('llvmorg-').removeprefix('jq-')
                for name, repo in REPOS.items() if repo}
    versions['rustup'] = tomllib.loads(fetch('https://static.rust-lang.org/rustup/release-stable.toml')[0].decode())['version']
    python_releases = api('https://www.python.org/api/v2/downloads/release/')
    versions['python'] = max((semver(item['name'].removeprefix('Python ')), item['name'].removeprefix('Python '))
                             for item in python_releases if item['is_published'] and not item['pre_release']
                             and semver(item['name'].removeprefix('Python ')))[1]
    zig = api('https://ziglang.org/download/index.json')
    versions['zig'] = max((semver(v), v) for v in zig if semver(v))[1]
    versions['nasm'] = nasm_release()
    data['versions'] = versions
    # The release archive is authoritative even before the upgrade notifier enables an LTS.
    meta = fetch('https://changelogs.ubuntu.com/meta-release-lts')[0].decode()
    records = [dict(line.split(': ', 1) for line in block.splitlines() if ': ' in line)
               for block in meta.strip().split('\n\n')]
    record = max((item for item in records if 'LTS' in item.get('Version', '')
                  and parsedate_to_datetime(item['Date']).replace(tzinfo=timezone.utc) <= now),
                 key=lambda item: tuple(map(int, item['Version'].split()[0].split('.'))))
    ubuntu = '.'.join(record['Version'].split()[0].split('.')[:2])
    snapshot = now.strftime('%Y%m%dT%H%M%SZ')
    archive = f'https://snapshot.ubuntu.com/ubuntu/{snapshot}/dists/{record["Dist"]}/Release'
    release_text = fetch(archive)[0].decode()
    if f'Version: {ubuntu}\n' not in release_text:
        raise ValueError('Ubuntu snapshot does not match the selected LTS')
    ci_release = fetch(f'https://snapshot.ubuntu.com/ubuntu/{snapshot}/dists/{data["linux-ci"]["codename"]}/Release')[0].decode()
    if f'Version: {data["linux-ci"]["ubuntu"]}\n' not in ci_release:
        raise ValueError('Ubuntu snapshot does not match the CI runner release')
    data['linux'].update(ubuntu=ubuntu, codename=record['Dist'], snapshot=snapshot)
    channel_bytes, channel_url = fetch('https://aka.ms/vs/stable/channel')
    channel = json.loads(channel_bytes)
    vsman = next(item['payloads'][0] for item in channel['channelItems'] if item['id'] == 'Microsoft.VisualStudio.Manifests.VisualStudio')
    manifest = api(vsman['url'])
    sdk_ids = {item['id'] for item in manifest['packages'] if re.fullmatch(r'Microsoft.VisualStudio.Component.Windows11SDK.\d+', item['id'])}
    windows_sdk = max(sdk_ids, key=lambda item: int(item.rsplit('.', 1)[1]))
    bootstrap = pinned_url('https://aka.ms/vs/stable/vs_buildtools.exe')
    data['windows'] = {'visual-studio': channel['info']['productDisplayVersion'],
                       'build-version': channel['info']['buildVersion'],
                       'channel-url': channel_url, 'channel-sha256': hashlib.sha256(channel_bytes).hexdigest(),
                       'bootstrap-url': bootstrap['url'], 'bootstrap-sha256': bootstrap['sha256'],
                       'sdk-component': windows_sdk}
    for platform in PLATFORMS:
        arch, os_name = platform.split('-')
        windows = os_name == 'win'
        host = data[platform]['rust-host']
        native = data[platform]
        native['os'] = ('Windows 11 Enterprise ARM64' if arch == 'aarch64' else 'Windows Server 2025') if windows else f'Ubuntu {ubuntu} LTS'
        native['rust-host'] = host
        native['assets'] = {}
        assets = native['assets']
        rustup_url = f'https://static.rust-lang.org/rustup/archive/{versions["rustup"]}/{host}/rustup-init' + ('.exe' if windows else '')
        checksum = fetch(rustup_url + '.sha256')[0].decode().split()[0]
        assets['rustup'] = {'url': rustup_url, 'sha256': checksum}
        if platform == 'riscv64-linux':
            assets['cargo-binstall'] = github_asset(REPOS['cargo-binstall'], f'cargo-binstall-{host.removesuffix("gnu")}musl.tgz')
        if platform in NATIVE_SOURCE_PLATFORMS:
            continue  # Distro CMake/Clang are snapshot-pinned; missing binaries build natively.
        assets['cargo-binstall'] = github_asset(REPOS['cargo-binstall'], f'cargo-binstall-{host}.' + ('zip' if windows else 'tgz'))
        assets['cargo-rail'] = github_asset(REPOS['cargo-rail'], f'cargo-rail-{host}.' + ('zip' if windows else 'tar.gz'))
        cmake_arch = ('arm64' if arch == 'aarch64' else 'x86_64') if windows else arch
        assets['cmake'] = github_asset(REPOS['cmake'], f'cmake-{versions["cmake"]}-' + (f'windows-{cmake_arch}.zip' if windows else f'linux-{cmake_arch}.tar.gz'))
        llvm_name = f'clang+llvm-{versions["llvm"]}-{host}.tar.xz' if windows else f'LLVM-{versions["llvm"]}-Linux-' + ('ARM64' if arch == 'aarch64' else 'X64') + '.tar.xz'
        assets['llvm'] = github_asset(REPOS['llvm'], llvm_name)
        if windows:
            pyarch = 'arm64' if arch == 'aarch64' else 'amd64'
            assets['python'] = pinned_url(f'https://www.python.org/ftp/python/{versions["python"]}/python-{versions["python"]}-embed-{pyarch}.zip')
            git_name = versions['git'].replace('.windows.', '.')
            # Full Git includes Git Bash, curl, archive utilities, and certificates.
            assets['git'] = github_asset(REPOS['git'], f'Git-{git_name}-' + ('arm64' if arch == 'aarch64' else '64-bit') + '.exe')
            assets['jq'] = github_asset(REPOS['jq'], f'jq-windows-{pyarch}.exe')
            assets['powershell'] = github_asset(REPOS['powershell'], f'PowerShell-{versions["powershell"]}-win-' + ('arm64' if arch == 'aarch64' else 'x64') + '.zip')
            if platform == 'x86_64-win':
                assets['nasm'] = pinned_url(f'https://www.nasm.us/pub/nasm/releasebuilds/{versions["nasm"]}/win64/nasm-{versions["nasm"]}-win64.zip')
        else:
            entry = zig[versions['zig']][f'{arch}-linux']
            assets['zig'] = {'url': entry['tarball'], 'sha256': entry['shasum']}
        if platform == 'x86_64-linux':
            assets['wasmtime'] = github_asset(REPOS['wasmtime'], f'wasmtime-v{versions["wasmtime"]}-x86_64-linux.tar.xz')
        data[platform] = native
    validate(data)
    return data


def write_catalog(data):
    content = catalog_text(data)
    validate(tomllib.loads(content))
    CATALOG.write_text(content)


def manifest_paths():
    # Includes untracked manifests and standalone consumers, excludes generated/cache trees.
    return [ROOT / path for path in subprocess.check_output(
        ['rg', '--files', '--hidden', '-g', 'Cargo.toml', '-g', '!.git', '-g', '!.agents',
         '-g', '!target', '-g', '!node_modules', '-g', '!vendor'], cwd=ROOT, text=True).splitlines()]


def dependency_tables(document):
    for key in ('dependencies', 'dev-dependencies', 'build-dependencies'):
        if key in document:
            yield document[key]
    if 'workspace' in document and 'dependencies' in document['workspace']:
        yield document['workspace']['dependencies']
    for target in document.get('target', {}).values():
        yield from dependency_tables(target)


def update_manifests(paths):
    import tomlkit
    documents = {path: tomlkit.parse(path.read_text()) for path in paths}
    rust_versions = {}
    for path, document in documents.items():
        rust_version = document.get('package', {}).get('rust-version')
        if isinstance(rust_version, dict):
            workspace = Path(subprocess.check_output(
                ['cargo', 'locate-project', '--workspace', '--manifest-path', str(path),
                 '--message-format', 'plain'], cwd=ROOT, text=True).strip())
            rust_version = documents[workspace]['workspace']['package']['rust-version']
        rust_versions[path] = rust_version
    # Renames can hold a compatibility version beside the current crate (sha2_010).
    # Preserve their requirements; cargo update still refreshes compatible lock entries.
    requests = set()
    for path, document in documents.items():
        rust_version = rust_versions[path]
        for table in dependency_tables(document):
            for alias, dependency in table.items():
                if isinstance(dependency, str):
                    requests.add((alias, rust_version))
                elif (not dependency.get('workspace') and not dependency.get('path')
                      and dependency.get('package', alias) == alias and 'version' in dependency):
                    if dependency.get('registry'):
                        raise ValueError(f'{alias}: updater needs a version source for custom registry {dependency["registry"]}')
                    requests.add((dependency.get('package', alias), rust_version))
    def select(request):
        name, rust_version = request
        return request, crate_version(name, rust_version)
    with ThreadPoolExecutor(max_workers=6) as pool:
        selected = dict(pool.map(select, sorted(requests, key=str)))
    for path, document in documents.items():
        rust_version = rust_versions[path]
        for table in dependency_tables(document):
            for alias, dependency in list(table.items()):
                if isinstance(dependency, str):
                    version = selected[(alias, rust_version)]
                    table[alias] = ('=' if dependency.startswith('=') else '') + version
                elif (not dependency.get('workspace') and not dependency.get('path')
                      and dependency.get('package', alias) == alias and 'version' in dependency):
                    name = dependency.get('package', alias)
                    version = selected[(name, rust_version)]
                    dependency['version'] = ('=' if dependency['version'].startswith('=') else '') + version
                    if 'git' in dependency:
                        repo = dependency['git'].removeprefix('https://github.com/').removesuffix('.git')
                        if dependency['git'] != f'https://github.com/{repo}' and dependency['git'] != f'https://github.com/{repo}.git':
                            raise ValueError(f'{name}: unsupported Git release source {dependency["git"]}')
                        tags = api(f'https://api.github.com/repos/{repo}/tags?per_page=100')
                        tag = next((tag for tag in tags if tag['name'] in (version, f'v{version}')), None)
                        if tag is None:
                            raise ValueError(f'{name}: no Git release tag for eligible crate version {version}')
                        dependency.pop('branch', None)
                        dependency.pop('tag', None)
                        dependency['rev'] = tag['commit']['sha']
        # Resolve everything before making any manifest edits.
    for path, document in documents.items():
        content = tomlkit.dumps(document)
        if content != path.read_text():
            print(f'Updating {path.relative_to(ROOT)}', flush=True)
            path.write_text(content)


def action_edits(content):
    """Use YAML source spans so comments, quoting, and script bodies survive."""
    import yaml
    edits = []
    def walk(node):
        if isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                if key.value == 'uses' and isinstance(value, yaml.ScalarNode):
                    match = re.fullmatch(r'([\w.-]+/[\w.-]+)(/[^@]+)?@([^\s]+)', value.value)
                    if match:
                        repo, subpath, ref = match.groups()
                        try:
                            ref = release(repo)['tag_name']
                        except ValueError as error:
                            if '404' not in str(error):
                                raise
                            # Some actions publish only branches, such as stable/nightly.
                            if re.fullmatch('[0-9a-f]{40}', ref):
                                suffix = content[value.end_mark.index:].split('\n', 1)[0]
                                recorded = re.match(r'\s*#\s*([\w./-]+)', suffix)
                                ref = recorded[1] if recorded else api(f'https://api.github.com/repos/{repo}')['default_branch']
                        commit = api(f'https://api.github.com/repos/{repo}/commits/{urllib.parse.quote(ref, safe="")}')['sha']
                        if not re.fullmatch('[0-9a-f]{40}', commit):
                            raise ValueError(f'{repo}: invalid action commit')
                        text = f'{repo}{subpath or ""}@{commit}'
                        end = content.find('\n', value.end_mark.index)
                        if end == -1:
                            end = len(content)
                        suffix = content[value.end_mark.index:end]
                        replacement = json.dumps(text)
                        if not suffix.strip():
                            replacement += f' # {ref}'
                        edits.append((value.start_mark.index, value.end_mark.index, replacement))
                walk(value)
        elif isinstance(node, yaml.SequenceNode):
            for child in node.value:
                walk(child)
    for document in yaml.compose_all(content, Loader=yaml.BaseLoader):
        if document:
            walk(document)
    for start, end, replacement in sorted(edits, reverse=True):
        content = content[:start] + replacement + content[end:]
    return content


def update_actions():
    # No workflows are created. Existing local actions and workflow files are inputs only.
    directory = ROOT / '.github'
    for path in sorted(directory.rglob('*')) if directory.exists() else []:
        if path.suffix in ('.yaml', '.yml'):
            previous = path.read_text()
            updated = action_edits(previous)
            if updated != previous:
                path.write_text(updated)


def update_rust():
    import tomlkit
    path = ROOT / 'rust-toolchain.toml'
    document = tomlkit.parse(path.read_text())
    channel = 'stable'
    latest = tomllib.loads(fetch('https://static.rust-lang.org/dist/channel-rust-stable.toml')[0].decode())
    data = read()
    hosts = {data[platform]['rust-host']: data[platform]['components'] for platform in PLATFORMS}
    hosts['aarch64-apple-darwin'] = document['toolchain']['components']
    names = {'clippy': 'clippy-preview', 'rustfmt': 'rustfmt-preview', 'llvm-tools': 'llvm-tools-preview',
             'rust-analyzer': 'rust-analyzer-preview'}
    for host, components in hosts.items():
        for component in {'rustc', 'cargo', 'rust-std', 'clippy', 'rustfmt', *components}:
            targets = latest['pkg'][names.get(component, component)]['target']
            if not targets.get(host, targets.get('*', {})).get('available'):
                raise ValueError(f'{channel} {latest["date"]}: {component} unavailable for {host}')
    document['toolchain']['channel'] = latest['pkg']['rust']['version'].split()[0]
    path.write_text(tomlkit.dumps(document))


def sync_gungraun_runner():
    # The structural benchmark protocol requires the runner and library to match.
    versions = {p['version'] for p in read(ROOT / 'Cargo.lock')['package'] if p['name'] == 'gungraun'}
    if len(versions) != 1:
        raise ValueError(f'expected one Gungraun library version, found {sorted(versions)}')
    data = read()
    data['cargo']['gungraun-runner'] = versions.pop()
    write_catalog(data)


def cargo_roots(paths):
    roots = set()
    for path in paths:
        root = subprocess.check_output(['cargo', 'locate-project', '--workspace', '--manifest-path', str(path),
                                        '--message-format', 'plain'], cwd=ROOT, text=True).strip()
        roots.add(Path(root))
    return sorted(roots)


def main():
    if sys.platform != 'darwin':
        raise ValueError('release updates run only on the local macOS workstation')
    if len(sys.argv) != 1:
        raise ValueError('usage: just update')

    # Select releases before installing anything.
    write_catalog(resolve_catalog())
    update_rust()
    subprocess.run(['rustup', 'toolchain', 'install', rust_channel(), '--profile', 'minimal',
                    '--component', 'clippy', '--component', 'rustfmt'], cwd=ROOT, check=True)
    paths = manifest_paths()
    roots = cargo_roots(paths)
    update_manifests(paths)

    # Standalone proof/consumer workspaces resolve the same local rscrypto sources.
    metadata = json.loads(subprocess.check_output(
        ['cargo', 'metadata', '--quiet', '--format-version', '1', '--no-deps'], cwd=ROOT, text=True))
    with tempfile.TemporaryDirectory(prefix='rscrypto-update-') as temporary:
        patches = Path(temporary) / 'patches.toml'
        patches.write_text('[patch.crates-io]\n' + ''.join(
            f'{json.dumps(package["name"])} = {{ path = {json.dumps(str(Path(package["manifest_path"]).parent))} }}\n'
            for package in metadata['packages'] if package['source'] is None))
        for manifest in roots:
            subprocess.run(['cargo', '--config', str(patches), 'update',
                            '--manifest-path', str(manifest)], cwd=ROOT, check=True)
    sync_gungraun_runner()
    update_actions()

    subprocess.run(['cargo', 'deny', '--locked', 'check', 'all'], cwd=ROOT, check=True)
    subprocess.run(['cargo', 'audit'], cwd=ROOT, check=True)
    print('Update complete')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, StopIteration, subprocess.CalledProcessError) as error:
        sys.exit(f'update: {error}')
