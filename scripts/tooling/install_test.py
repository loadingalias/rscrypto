#!/usr/bin/env python3
"""Exercise Linux provisioning with substitute host and installation commands."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[2]
BASH = shutil.which('bash')
CATALOG = tomllib.loads((ROOT / '.config/tooling.toml').read_text())

STUB = r'''import json, os, pathlib, subprocess, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['INSTALL_LOG'], 'a') as log:
    log.write(json.dumps([name, *args]) + '\n')
if name == 'uname':
    print('Linux' if args == ['-s'] else os.environ['INSTALL_ARCH'])
elif name == 'id':
    print('0')
elif name == 'apt-cache':
    print(args[-1] + ' | 1.0 | snapshot')
elif name == 'apt-get':
    if os.environ.get('INSTALL_FAIL_APT'):
        sys.exit(42)
    if os.environ.get('INSTALL_REAL_APT'):
        options = dict(arg.split('=', 1) for arg in args if arg.startswith('Dir::'))
        fixture = pathlib.Path(os.environ['INSTALL_APT_FIXTURE'])
        if args[-1] == 'update':
            source = pathlib.Path(options['Dir::Etc::sourcelist']).read_text().splitlines()[0].split()
            url, suite = next((source[i], source[i + 1]) for i in range(len(source)) if source[i].startswith('https://'))
            name = url.removeprefix('https://').replace('/', '_') + '_dists_' + suite
            name += '_main_binary-' + os.environ['INSTALL_APT_ARCH'] + '_Packages'
            pathlib.Path(options['Dir::State::lists'], name).write_text((fixture / 'Packages').read_text())
        else:
            if os.environ.get('INSTALL_WITHOUT_PREFERENCE'):
                pathlib.Path(options['Dir::Etc::preferences']).write_text('')
            sys.exit(subprocess.run([os.environ['INSTALL_REAL_APT'], *args, '--simulate', '--no-remove',
                '-o', 'Dir::State::status=' + str(fixture / 'status'),
                '-o', 'Dir::Cache::pkgcache=', '-o', 'Dir::Cache::srcpkgcache=']).returncode)
elif name == 'python3':
    script = pathlib.Path(args[0]).name
    if script == 'catalog.py' and args[1] == 'download':
        pathlib.Path(args[-1]).write_text('#!/bin/sh\nexit 0\n')
    elif script == 'catalog.py' and args[1] == 'install-archive':
        directory = pathlib.Path(args[-1]) / args[-2]
        directory.mkdir(parents=True, exist_ok=True)
        executable = directory / 'cargo-binstall'
        executable.write_text('#!/bin/sh\nexit 0\n')
        executable.chmod(0o755)
        print(directory)
    elif script in ('toolchain.py', 'compat.py') and '--install' in args:
        pass
    else:
        sys.exit(subprocess.run([sys.executable, *args]).returncode)
'''


class LinuxInstall(unittest.TestCase):
    def provision(self, platform, fail=False, real_apt=False, without_preference=False, profile='ci'):
        temporary = tempfile.TemporaryDirectory(prefix='rscrypto installer ')
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        binaries = root / 'bin'
        binaries.mkdir()
        for name in ('uname', 'id', 'apt-get', 'apt-cache', 'cargo', 'clang', 'cmake', 'python3', 'rustup', 'wasmtime'):
            script = binaries / name
            script.write_text('#!' + sys.executable + '\n' + STUB)
            script.chmod(0o755)
        bash_env = root / 'bash-env'
        bash_env.write_text('''source() {
  if [[ "$1" == /etc/os-release ]]; then
    ID=ubuntu
    VERSION_ID=24.04
    PRETTY_NAME=fixture
  else
    builtin source "$@"
  fi
}
''')
        arch = platform.removesuffix('-linux').replace('powerpc64le', 'ppc64le')
        env = {**os.environ, 'HOME': str(root), 'CARGO_HOME': str(root / 'custom cargo'),
               'PATH': str(binaries) + os.pathsep + os.environ['PATH'],
               'BASH_ENV': str(bash_env), 'INSTALL_ARCH': arch,
               'INSTALL_LOG': str(root / 'commands.jsonl')}
        if fail:
            env['INSTALL_FAIL_APT'] = '1'
        if real_apt:
            def package(name, version, installed=False, depends=''):
                return (f'Package: {name}\nVersion: {version}\nArchitecture: all\n'
                        + ('Status: install ok installed\n' if installed else
                           f'Filename: pool/{name}_{version}_all.deb\nSize: 1\nSHA256: {"0" * 64}\n')
                        + (f'Depends: {depends}\n' if depends else '')
                        + 'Maintainer: Fixture <fixture@example.invalid>\nDescription: APT resolver fixture\n\n')
            (root / 'Packages').write_text(''.join(
                package(name, '1.0', depends='git-man (= 1.0)' if name == 'git' else '')
                for name in [*CATALOG['linux-ci']['packages'], *CATALOG['ci-musl']['packages'], 'git-man', 'fixture-unrelated']))
            (root / 'status').write_text(''.join(
                package(name, '2.0', installed=True, depends='git-man (= 2.0)' if name == 'git' else '')
                for name in ('git', 'git-man', 'fixture-unrelated')))
            env.update(INSTALL_REAL_APT=shutil.which('apt-get'), INSTALL_APT_FIXTURE=str(root),
                       INSTALL_APT_ARCH=subprocess.check_output(['dpkg', '--print-architecture'], text=True).strip())
            if without_preference:
                env['INSTALL_WITHOUT_PREFERENCE'] = '1'
        result = subprocess.run([BASH, str(ROOT / 'scripts/tooling/linux.sh'), platform, '--' + profile],
                                env=env, capture_output=True, text=True)
        calls = [json.loads(line) for line in (root / 'commands.jsonl').read_text().splitlines()]
        return result, calls, root

    def test_ci_profiles_install_only_required_tools(self):
        for platform in ('x86_64-linux', 'aarch64-linux', 's390x-linux', 'powerpc64le-linux', 'riscv64-linux'):
            with self.subTest(platform=platform):
                result, calls, root = self.provision(platform)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                installs = [c for c in calls if c[0] == 'cargo' and ('install' in c or 'binstall' in c)]
                expected_tools = CATALOG['ci']['cargo'] + (CATALOG['ci-policy']['cargo'] if platform == 'x86_64-linux' else [])
                self.assertEqual(len(installs), len(expected_tools))
                binary = platform in ('x86_64-linux', 'aarch64-linux', 'riscv64-linux')
                for command, tool in zip(installs, expected_tools):
                    self.assertIn('binstall' if binary else 'install', command)
                    self.assertEqual(command[-1], f"{tool}@{CATALOG['cargo'][tool]}" if binary else tool)
                    self.assertIn('--locked', command)
                    host = CATALOG[platform]['rust-host']
                    self.assertIn(host, command)
                    targets = [command[i + 1] for i, arg in enumerate(command) if arg == '--targets']
                    self.assertEqual(targets, [host, host.removesuffix('gnu') + 'musl'] if binary else [])
                apt = next(c for c in calls if c[0] == 'apt-get' and '--allow-downgrades' in c)
                self.assertIn('--no-install-recommends', apt)
                self.assertEqual([a for a in apt if a.endswith('=1.0')],
                                 [p + '=1.0' for p in CATALOG['linux-ci']['packages'] +
                                  (CATALOG['ci-musl']['packages'] if platform in ('x86_64-linux', 'aarch64-linux') else [])])
                environment = (root / '.local/share/rscrypto-tooling/environment.sh').read_text()
                self.assertIn('custom\\ cargo/bin', environment)
                self.assertFalse((root / '.bashrc').exists())
                self.assertFalse((root / '.profile').exists())

    def test_compat_profile_omits_native_test_and_policy_tools(self):
        result, calls, _ = self.provision('x86_64-linux', profile='ci-compat')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        installs = [c for c in calls if c[0] == 'cargo' and 'binstall' in c]
        self.assertEqual([c[-1] for c in installs], [f"just@{CATALOG['cargo']['just']}"])
        archives = [c[-2] for c in calls if c[0] == 'python3' and 'install-archive' in c]
        self.assertEqual(archives, ['cargo-binstall', 'wasmtime'])
        self.assertFalse(any('musl-tools=1.0' in c for c in calls))
        self.assertFalse(any('--install' in c and any(arg.endswith('toolchain.py') for arg in c) for c in calls))

    def test_focused_profiles_install_only_their_execution_dependencies(self):
        for platform, profile, components in (
                ('x86_64-linux', 'ci-fuzz', ['rust-src']),
                ('x86_64-linux', 'ci-ct', ['llvm-tools']),
                ('x86_64-linux', 'ci-bench', []),
                ('aarch64-linux', 'ci-bench', [])):
            with self.subTest(platform=platform, profile=profile):
                result, calls, _ = self.provision(platform, profile=profile)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                installs = [c[-1] for c in calls if c[0] == 'cargo' and 'binstall' in c]
                self.assertEqual(installs, [f"{t}@{CATALOG['cargo'][t]}" for t in CATALOG[profile]['cargo']])
                apt = next(c for c in calls if c[0] == 'apt-get' and '--allow-downgrades' in c)
                self.assertEqual([a for a in apt if a.endswith('=1.0')],
                                 [p + '=1.0' for p in CATALOG['linux-ci' if profile == 'ci-bench' else profile]['packages']])
                rustup = [c for c in calls if c[:3] == ['rustup', 'toolchain', 'install']]
                self.assertEqual(len(rustup), 2 if profile == 'ci-fuzz' else 1)
                self.assertEqual([c[i + 1] for c in rustup for i, arg in enumerate(c) if arg == '--component'],
                                 components)
                if profile == 'ci-fuzz':
                    policy = tomllib.loads((ROOT / '.config/toolchains.toml').read_text())
                    self.assertEqual(rustup[1][3], policy['nightly'])
                self.assertFalse(any('musl-tools=1.0' in c or 'target' in c and c[0] == 'rustup' for c in calls))
                archives = [c[-2] for c in calls if c[0] == 'python3' and 'install-archive' in c]
                self.assertEqual(archives, ['cargo-binstall'])

    def test_package_failure_stops_before_rust_installation(self):
        result, calls, _ = self.provision('x86_64-linux', fail=True)
        self.assertEqual(result.returncode, 42)
        self.assertFalse(any(c[0] == 'cargo' or 'download' in c for c in calls))

    @unittest.skipUnless(shutil.which('apt-get') and shutil.which('dpkg'), 'requires the real APT resolver')
    def test_snapshot_resolves_newer_installed_dependencies(self):
        result, _, _ = self.provision('x86_64-linux', real_apt=True, without_preference=True)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('unmet dependencies', result.stdout + result.stderr)
        result, _, _ = self.provision('x86_64-linux', real_apt=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Inst git [2.0] (1.0 ', result.stdout)
        self.assertIn('Inst git-man [2.0] (1.0 ', result.stdout)
        self.assertNotIn('Inst fixture-unrelated', result.stdout)
        self.assertNotIn('Remv ', result.stdout)


if __name__ == '__main__':
    unittest.main()
