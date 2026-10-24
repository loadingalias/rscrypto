#!/usr/bin/env -S uv run python
"""Exercise Linux provisioning and the Mac's pinned uv and zig with substitute host and installation commands."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import unittest
from unittest.mock import ANY, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/ct'))
import provenance  # noqa: E402
BASH = shutil.which('bash')
CATALOG = tomllib.loads((ROOT / '.config/tooling.toml').read_text())

STUB = r'''import json, os, pathlib, shutil, subprocess, sys, tomllib
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['INSTALL_LOG'], 'a') as log:
    log.write(json.dumps([name, *args]) + '\n')
if name == 'uv' and args[:2] == ['python', 'install']:
    sys.exit(0)
if name == 'uv' and args[:2] == ['run', 'python']:
    if args[2:3] == ['-c'] and 'sys.executable' in args[3]:
        # uv selects the pinned interpreter: here, the substitute that records each helper.
        print(os.environ['INSTALL_PYTHON'])
        sys.exit(0)
    name, args = 'python', args[2:]
if os.environ.get('INSTALL_PINNED') and args[-1:] == ['--version']:
    # A converged host: each installed tool reports its catalog pin.
    catalog = tomllib.loads(pathlib.Path(os.environ['INSTALL_CATALOG']).read_text())
    pins = {**catalog['versions'], **catalog['cargo']}
    tool = 'cargo-' + args[-2] if name == 'cargo' else {'rg': 'ripgrep'}.get(name, name)
    if tool in pins:
        print(f'{tool} {pins[tool]}')
        sys.exit(0)
if name == 'uname':
    if args == ['-s']:
        print('Linux')
    elif args == ['-r']:
        print(os.environ.get('INSTALL_KERNEL_RELEASE', 'fixture-kernel'))
    else:
        print(os.environ['INSTALL_ARCH'])
elif name == 'id':
    print('0')
elif name == 'dpkg-query':
    # Only a fixture that says so already holds every requested package.
    if os.environ.get('INSTALL_PACKAGES_PRESENT'):
        for package in args[2:]:
            print('ii ' + package)
elif name == 'apt-cache':
    if not (os.environ.get('INSTALL_NO_EXACT_PERF') and args[-1].startswith('linux-tools-')
            and args[-1] != 'linux-tools-generic'):
        print(args[-1] + ' | 1.0 | archive')
elif name == 'apt-get':
    if os.environ.get('INSTALL_FAIL_APT'):
        sys.exit(42)
elif name == 'perf':
    installed = any(call[0] == 'apt-get' and 'linux-tools-generic' in call
                    for call in map(json.loads, pathlib.Path(os.environ['INSTALL_LOG']).read_text().splitlines()))
    if os.environ.get('INSTALL_PERF_AFTER_PACKAGE') and not installed:
        sys.exit(1)
    print('perf version fixture')
elif name == 'tar':
    destination = pathlib.Path(args[args.index('-C') + 1])
    source = destination / ('linux-' + os.environ['INSTALL_PERF_SOURCE_VERSION']) / 'tools' / 'perf'
    source.mkdir(parents=True)
elif name == 'make':
    build = pathlib.Path(next(arg.removeprefix('O=') for arg in args if arg.startswith('O=')))
    build.mkdir(parents=True, exist_ok=True)
    executable = build / 'perf'
    executable.write_text('#!/bin/sh\nprintf "perf version source fixture\\n"\n')
    executable.chmod(0o755)
elif name == 'nproc':
    print('4')
elif name == 'tee':
    data = sys.stdin.read()
    sys.stdout.write(data)
    for path in args:
        if path != '/etc/sysctl.d/99-rscrypto-profiling.conf':
            pathlib.Path(path).write_text(data)
elif name == 'sysctl':
    if args == ['-n', 'kernel.perf_event_paranoid']:
        print(os.environ.get('INSTALL_PERF_PARANOID', '2'))
    elif args == ['-w', 'kernel.perf_event_paranoid=-1']:
        print('kernel.perf_event_paranoid = -1')
    elif args == ['-p', '/etc/sysctl.d/99-rscrypto-profiling.conf']:
        print('kernel.perf_event_paranoid = -1')
    else:
        sys.exit(1)
elif name == 'python':
    script = pathlib.Path(args[0]).name
    if script == 'catalog.py' and args[1] == 'download':
        pathlib.Path(args[-1]).write_text('#!/bin/sh\nexit 0\n')
    elif script == 'catalog.py' and args[1] == 'download-entry':
        pathlib.Path(args[-1]).write_text('pinned source fixture')
    elif script == 'catalog.py' and args[1] == 'install-archives':
        pass
    elif script == 'catalog.py' and args[1] == 'install-archive':
        directory = pathlib.Path(args[-1]) / args[-2]
        directory.mkdir(parents=True, exist_ok=True)
        executable = directory / args[-2]
        executable.write_text(
            '#!' + sys.executable + '\n'
            'import json, os, pathlib, sys\n'
            'with open(os.environ["INSTALL_LOG"], "a") as log:\n'
            '    log.write(json.dumps([pathlib.Path(sys.argv[0]).name, *sys.argv[1:]]) + "\\n")\n'
        )
        executable.chmod(0o755)
        print(directory)
    elif script == 'transfer.py':
        if args[1] == 'install': print(str(pathlib.Path(args[-1]) / 'bin'))
    elif script in ('compat.py', 'package.py') and '--install' in args:
        pass
    else:
        sys.exit(subprocess.run([sys.executable, *args]).returncode)
'''


def stub(path):
    path.write_text('#!' + sys.executable + '\n' + STUB)
    path.chmod(0o755)


def python(calls):
    """Python helper invocations, each with the interpreter that uv selected."""
    return [call[1:] for call in calls if call[0] == 'python']


def package_installs(calls):
    """The package lists of APT installations, after the bootstrap of curl and the trust store."""
    installs = []
    for call in calls:
        if call[0] == 'apt-get' and 'install' in call:
            packages = [arg for arg in call[call.index('install') + 1:] if not arg.startswith('-')]
            if packages != ['curl', 'ca-certificates']:
                installs.append(packages)
    return installs


class LinuxInstall(unittest.TestCase):
    def provision(self, platform, fail=False, profile='ci', target=None, extra_env=None, arguments=None, again=None):
        """Run the installer in a fixture home; `again` reruns it there with these environment changes."""
        temporary = tempfile.TemporaryDirectory(prefix='rscrypto installer ')
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        binaries = root / 'bin'
        binaries.mkdir()
        (root / 'commands.jsonl').touch()
        for name in ('uname', 'id', 'apt-get', 'apt-cache', 'dpkg-query', 'cargo', 'clang', 'cmake', 'make', 'nproc',
                     'patch', 'perf', 'rustc', 'rustup', 'sudo', 'tar', 'tee', 'valgrind', 'wasmtime', 'opam', 'just',
                     'rg', 'lychee', 'rumdl', 'samply', 'gungraun-runner', 'sysctl'):
            stub(binaries / name)
        # The pinned uv is installed, so the installer finds it by its receipt.
        uv = CATALOG[platform]['assets']['uv']['sha256']
        uv_directory = root / '.local/share/rscrypto-tooling/uv' / uv[:16]
        uv_directory.mkdir(parents=True)
        (uv_directory / '.rscrypto-installed').write_text(uv + '\n')
        stub(uv_directory / 'uv')
        (root / 'venv/bin').mkdir(parents=True)
        stub(root / 'venv/bin/python')
        bash_env = root / 'bash-env'
        bash_env.write_text('''source() {
  if [[ "$1" == /etc/os-release ]]; then
    ID=ubuntu
    VERSION_ID=${INSTALL_UBUNTU_VERSION:-24.04}
    PRETTY_NAME=fixture
  else
    builtin source "$@"
  fi
}
''')
        arch = platform.removesuffix('-linux').replace('powerpc64le', 'ppc64le')
        temporary_path = root / 'temporary files'
        temporary_path.mkdir()
        env = {**os.environ, 'HOME': str(root), 'CARGO_HOME': str(root / 'custom cargo'),
               'TMPDIR': str(temporary_path),
               'PATH': str(binaries) + os.pathsep + os.environ['PATH'],
               'BASH_ENV': str(bash_env), 'INSTALL_ARCH': arch,
               'INSTALL_LOG': str(root / 'commands.jsonl'), 'INSTALL_CATALOG': str(ROOT / '.config/tooling.toml'),
               'INSTALL_PYTHON': str(root / 'venv/bin/python')}
        env.update(extra_env or {})
        if fail:
            env['INSTALL_FAIL_APT'] = '1'
        if arguments is None:
            extra = [target] if profile == 'ci-cross-build' else ['tools.tar.gz'] if profile == 'ci-cross-run' else []
            arguments = ['--' + profile, *extra]
        command = [BASH, str(ROOT / 'scripts/tooling/linux.sh'), platform, *arguments]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        if again is not None:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            (root / 'commands.jsonl').write_text('')
            result = subprocess.run(command, env={**env, **again}, capture_output=True, text=True)
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
                apt = next(c for c in calls if c[0] == 'apt-get' and '--no-install-recommends' in c)
                self.assertNotIn('--allow-downgrades', apt)
                self.assertEqual(package_installs(calls), [
                    CATALOG['linux-ci']['packages'] +
                    (CATALOG['ci-musl']['packages'] if platform in ('x86_64-linux', 'aarch64-linux') else [])])
                environment = (root / '.local/share/rscrypto-tooling/environment.sh').read_text()
                self.assertIn('custom\\ cargo/bin', environment)
                self.assertIn('rscrypto-tooling/uv/' + CATALOG[platform]['assets']['uv']['sha256'][:16], environment)
                self.assertFalse((root / '.bashrc').exists())
                self.assertFalse((root / '.profile').exists())

    def test_python_runs_only_through_the_pinned_uv(self):
        result, calls, _ = self.provision('x86_64-linux')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(['uv', 'python', 'install', '--compile-bytecode'], calls)
        self.assertIn(['uv', 'run', 'python', '-c', 'import sys; print(sys.executable)'], calls)
        self.assertFalse(any(call[0] == 'python3' for call in calls))
        helpers = {Path(call[0]).name for call in python(calls)}
        self.assertLessEqual({'catalog.py', 'fingerprint.py', 'toolchain.py'}, helpers)

    def test_compat_profile_omits_policy_tools_and_native_suite_packages(self):
        result, calls, _ = self.provision('x86_64-linux', profile='ci-compat')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        installs = [c for c in calls if c[0] == 'cargo' and 'binstall' in c]
        # Nextest runs the provider runtime matrix that `just ci-provider` executes in this row.
        self.assertEqual([c[-1] for c in installs], [f"just@{CATALOG['cargo']['just']}",
                                                     f"cargo-nextest@{CATALOG['cargo']['cargo-nextest']}"])
        archives = [c[-2] for c in python(calls) if 'install-archive' in c]
        self.assertEqual(archives, ['cargo-binstall', 'wasmtime'])
        self.assertEqual(package_installs(calls), [CATALOG['ci-compat']['packages']])
        self.assertFalse(any('--install' in c and c[0].endswith('toolchain.py') for c in python(calls)))

    def test_focused_profiles_install_only_their_execution_dependencies(self):
        for platform, profile, components in (
                ('x86_64-linux', 'ci-fuzz', ['rust-src', 'llvm-tools']),
                ('aarch64-linux', 'ci-fuzz', ['rust-src', 'llvm-tools']),
                ('x86_64-linux', 'ci-miri', ['miri', 'rust-src']),
                ('x86_64-linux', 'ci-ct', ['llvm-tools']),
                ('aarch64-linux', 'ci-ct', ['llvm-tools']),
                ('s390x-linux', 'ci-ct', ['llvm-tools']),
                ('powerpc64le-linux', 'ci-ct', ['llvm-tools']),
                ('riscv64-linux', 'ci-ct', ['llvm-tools']),
                ('x86_64-linux', 'ci-bench', []),
                ('aarch64-linux', 'ci-bench', []),
                ('s390x-linux', 'ci-bench', []),
                ('powerpc64le-linux', 'ci-bench', []),
                ('riscv64-linux', 'ci-bench', [])):
            with self.subTest(platform=platform, profile=profile):
                result, calls, _ = self.provision(platform, profile=profile)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                installs = [c[-1] for c in calls if c[0] == 'cargo' and 'binstall' in c]
                if 'cargo-binstall' not in CATALOG[platform]['assets']:
                    installs = [f"{c[-1]}@{c[c.index('--version') + 1]}" for c in calls
                                if c[0] == 'cargo' and 'install' in c]
                self.assertEqual(installs, [f"{t}@{CATALOG['cargo'][t]}" for t in CATALOG[profile]['cargo']])
                self.assertEqual(package_installs(calls),
                                 [CATALOG['linux-ci' if profile == 'ci-bench' else profile]['packages']])
                rustup = [c for c in calls if c[:3] == ['rustup', 'toolchain', 'install']]
                # One canonical toolchain carries every lane's components.
                self.assertEqual(len(rustup), 1)
                self.assertEqual(rustup[0][3], tomllib.loads((ROOT / 'rust-toolchain.toml').read_text())['toolchain']['channel'])
                self.assertEqual([c[i + 1] for c in rustup for i, arg in enumerate(c) if arg == '--component'],
                                 components)
                self.assertFalse(any('target' in c and c[0] == 'rustup' for c in calls))
                archives = [c[-2] for c in python(calls) if 'install-archive' in c]
                self.assertEqual(archives, ['cargo-binstall'] if 'cargo-binstall' in CATALOG[platform]['assets'] else [])
                for tool in CATALOG[profile]['cargo']:
                    if tool.startswith('cargo-'):
                        self.assertTrue(any(c[0] == 'cargo' and c[-2:] == [tool.removeprefix('cargo-'), '--version']
                                            for c in calls))
                    else:
                        probe = ['rg' if tool == 'ripgrep' else tool, '--version']
                        self.assertIn(probe, calls)

    def test_package_profile_installs_consumer_prerequisites_once(self):
        result, calls, _ = self.provision('x86_64-linux', profile='ci-package')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        installs = [c for c in calls if c[0] == 'cargo' and 'binstall' in c]
        self.assertEqual([c[-1] for c in installs],
                         [f"{tool}@{CATALOG['cargo'][tool]}" for tool in ('just', 'cargo-semver-checks')])
        self.assertTrue(any(c[0] == 'cargo' and c[-2:] == ['semver-checks', '--version'] for c in calls))
        self.assertEqual(package_installs(calls), [CATALOG['linux-ci']['packages']])
        compiler_installs = [c for c in python(calls) if '--install' in c]
        self.assertEqual(len(compiler_installs), 1)
        self.assertTrue(compiler_installs[0][0].endswith('scripts/check/package.py'))

    def test_cross_build_and_execution_tooling_are_separate(self):
        channel = tomllib.loads((ROOT / 'rust-toolchain.toml').read_text())['toolchain']['channel']
        for platform, target, prefix, libc in (
            ('aarch64-linux', 'aarch64-unknown-linux-gnu', 'aarch64-linux-gnu', 'arm64'),
            ('riscv64-linux', 'riscv64gc-unknown-linux-gnu', 'riscv64-linux-gnu', 'riscv64'),
            ('powerpc64le-linux', 'powerpc64le-unknown-linux-gnu', 'powerpc64le-linux-gnu', 'ppc64el'),
            ('s390x-linux', 's390x-unknown-linux-gnu', 's390x-linux-gnu', 's390x'),
            ('x86_64-linux', 'x86_64-unknown-linux-gnu', None, None),
        ):
            for profile in ('ci-cross-build', 'ci-cross-run'):
                with self.subTest(target=target, profile=profile):
                    result, calls, root = self.provision('x86_64-linux' if profile.endswith('build') else platform,
                                                         profile=profile, target=target)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    packages = list(CATALOG[profile]['packages'])
                    if profile.endswith('build') and prefix is not None:
                        packages += ['gcc-' + prefix, 'g++-' + prefix, 'libc6-dev-' + libc + '-cross']
                    self.assertEqual(package_installs(calls), [packages])
                    installs = [c for c in calls if c[:3] == ['rustup', 'toolchain', 'install']]
                    self.assertEqual([c[3] for c in installs], [channel])
                    components = [c[i + 1] for c in installs for i, arg in enumerate(c) if arg == '--component']
                    self.assertEqual(components, ['rustfmt', 'clippy', 'llvm-tools'] if profile.endswith('build') else [])
                    if profile.endswith('build'):
                        self.assertIn(['rustup', 'target', 'add', '--toolchain', channel, target], calls)
                        archives = [c[-2] for c in python(calls) if 'install-archive' in c]
                        self.assertEqual(archives, CATALOG['ci-cross-build']['assets'])
                        self.assertFalse(any(c[0] == 'cargo' and ('install' in c or 'binstall' in c) for c in calls))
                        self.assertIn(['just', '--version'], calls)
                        self.assertIn(['cargo', f'+{channel}', 'nextest', '--version'], calls)
                    else:
                        self.assertFalse(any(c[0] == 'cargo' and ('build' in c or 'install' in c or 'binstall' in c) for c in calls))
                    transfers = [c for c in python(calls) if c[0].endswith('/tooling/transfer.py')]
                    self.assertEqual(len(transfers), 1)
                    self.assertEqual(transfers[0][1:3], ['prepare' if profile.endswith('build') else 'install', target])
                    # A target or a transferred archive decides these installations; nothing records them.
                    self.assertFalse(any(c[0].endswith('/fingerprint.py') for c in python(calls)))
                    self.assertEqual(list((root / '.local/share/rscrypto-tooling').glob('fingerprint.*')), [])

    def test_cross_run_uses_generic_perf_when_the_kernel_package_is_absent(self):
        result, calls, _ = self.provision(
            'powerpc64le-linux', profile='ci-cross-run',
            extra_env={'RSCRYPTO_REQUIRE_PERF': '1', 'INSTALL_NO_EXACT_PERF': '1',
                       'INSTALL_PERF_AFTER_PACKAGE': '1'})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        installs = package_installs(calls)
        self.assertEqual(len(installs), 2)
        self.assertEqual(installs[-1], ['linux-tools-generic'])
        self.assertIn('perf version fixture', result.stdout)

    def test_cross_run_enables_perf_when_runner_policy_denies_it(self):
        result, calls, _ = self.provision(
            'x86_64-linux', profile='ci-cross-run',
            extra_env={'RSCRYPTO_REQUIRE_PERF': '1', 'INSTALL_PERF_PARANOID': '4'})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        sysctl = [call for call in calls if call[0] == 'sysctl']
        self.assertEqual(sysctl, [
            ['sysctl', '-n', 'kernel.perf_event_paranoid'],
            ['sysctl', '-w', 'kernel.perf_event_paranoid=-1'],
        ])

    def test_riscv_cross_run_builds_perf_from_matching_pinned_kernel_source(self):
        source = CATALOG['ci-cross-run']['perf-source']
        result, calls, _ = self.provision(
            'riscv64-linux', profile='ci-cross-run',
            extra_env={'RSCRYPTO_REQUIRE_PERF': '1', 'INSTALL_NO_EXACT_PERF': '1',
                       'INSTALL_PERF_AFTER_PACKAGE': '1',
                       'INSTALL_KERNEL_RELEASE': source['version'] + '-scw1',
                       'INSTALL_PERF_SOURCE_VERSION': source['version']})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        installs = package_installs(calls)
        self.assertEqual(len(installs), 2)
        self.assertEqual(installs[-1], source['packages'])
        self.assertIn([str(ROOT / 'scripts/tooling/catalog.py'), 'download-entry',
                       'ci-cross-run', 'perf-source', ANY], python(calls))
        patch = next(call for call in calls if call[0] == 'patch')
        self.assertIn(str(ROOT / 'scripts/tooling/perf-riscv-mapping-symbols.patch'), patch)
        make = next(call for call in calls if call[0] == 'make')
        self.assertLess(calls.index(patch), calls.index(make))
        self.assertIn('ARCH=riscv', make)
        self.assertIn('NO_LIBUNWIND=1', make)
        self.assertIn('perf version source fixture', result.stdout)

    def test_riscv_cross_run_rejects_an_unpinned_custom_kernel(self):
        result, calls, _ = self.provision(
            'riscv64-linux', profile='ci-cross-run',
            extra_env={'RSCRYPTO_REQUIRE_PERF': '1', 'INSTALL_NO_EXACT_PERF': '1',
                       'INSTALL_PERF_AFTER_PACKAGE': '1', 'INSTALL_KERNEL_RELEASE': '6.6.1-custom'})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('does not match pinned perf source', result.stderr)
        self.assertFalse(any(call[0] == 'make' for call in calls))

    def test_proof_tools_only_on_supported_full_ct_hosts(self):
        for platform in ('x86_64-linux', 'aarch64-linux', 's390x-linux', 'powerpc64le-linux', 'riscv64-linux'):
            with self.subTest(platform=platform):
                result, calls, _ = self.provision(platform, profile='ci-ct-full')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertFalse(any('jq' in packages for packages in package_installs(calls)))
                opam = [c for c in calls if c[0] == 'opam']
                if platform in ('x86_64-linux', 'aarch64-linux'):
                    self.assertIn(CATALOG['ci-ct-proof']['opam-repository'], opam[0])
                    install = next(c for c in opam if c[1] == 'install')
                    self.assertEqual(install[-3:], CATALOG['ci-ct-proof']['opam'])
                else:
                    self.assertEqual(opam, [])

    def test_native_proof_tools_preserve_the_native_ubuntu_catalog(self):
        result, calls, _ = self.provision(
            'x86_64-linux', profile='ct-proof',
            extra_env={'INSTALL_UBUNTU_VERSION': CATALOG['linux']['ubuntu']})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        [packages] = package_installs(calls)
        self.assertEqual(packages[:len(CATALOG['linux']['packages'])], CATALOG['linux']['packages'])
        opam = [call for call in calls if call[0] == 'opam']
        self.assertIn(CATALOG['ci-ct-proof']['opam-repository'], opam[0])
        install = next(call for call in opam if call[1] == 'install')
        self.assertEqual(install[-3:], CATALOG['ci-ct-proof']['opam'])

    def test_package_failure_stops_before_rust_installation(self):
        result, calls, _ = self.provision('x86_64-linux', fail=True)
        self.assertEqual(result.returncode, 42)
        self.assertFalse(any(c[0] == 'cargo' or 'download' in c for c in calls))

    def test_operation_names_install_what_their_flags_install(self):
        def installation(arguments):
            result, calls, _ = self.provision('x86_64-linux', arguments=arguments)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return [c for c in calls if c[0] in ('apt-get', 'cargo', 'rustup')]
        self.assertEqual(installation(['ci']), installation(['--ci']))
        self.assertEqual(installation(['ci-ct-full']), installation(['--ci-ct-full']))
        result, _, _ = self.provision('x86_64-linux', arguments=['ci-unknown'])
        self.assertEqual(result.returncode, 64)

    def test_installed_packages_skip_the_archive(self):
        result, calls, _ = self.provision('x86_64-linux', extra_env={'INSTALL_PACKAGES_PRESENT': '1'})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Ubuntu packages are installed.', result.stdout)
        self.assertEqual(package_installs(calls), [])

    def test_converged_tools_are_kept_and_the_run_records_its_fingerprint(self):
        result, calls, root = self.provision('x86_64-linux', extra_env={'INSTALL_PINNED': '1'})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(any(c[0] == 'cargo' and ('install' in c or 'binstall' in c) for c in calls))
        self.assertFalse(any('download' in c for c in python(calls)))
        fingerprint = subprocess.check_output(
            [sys.executable, str(ROOT / 'scripts/tooling/fingerprint.py'), 'x86_64-linux', 'ci'], text=True)
        self.assertEqual((root / '.local/share/rscrypto-tooling/fingerprint.ci').read_text(), fingerprint)

    def test_a_matching_fingerprint_republishes_without_installing(self):
        result, calls, root = self.provision('x86_64-linux', extra_env={'INSTALL_PINNED': '1'}, again={})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertRegex(result.stdout, r'Tooling fingerprint [0-9a-f]{64} matches')
        self.assertEqual(package_installs(calls), [])
        self.assertFalse(any(c[0] in ('cargo', 'rustup') and ('install' in c or 'binstall' in c) for c in calls))
        environment = (root / '.local/share/rscrypto-tooling/environment.sh').read_text()
        self.assertIn('custom\\ cargo/bin', environment)

    def test_a_matching_fingerprint_with_a_missing_tool_converges(self):
        result, calls, _ = self.provision('x86_64-linux', extra_env={'INSTALL_PINNED': '1'},
                                          again={'INSTALL_PINNED': ''})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('a pinned tool is missing; converging', result.stderr)
        installs = [c[-1] for c in calls if c[0] == 'cargo' and 'binstall' in c]
        self.assertEqual(installs, [f"{tool}@{CATALOG['cargo'][tool]}"
                                    for tool in CATALOG['ci']['cargo'] + CATALOG['ci-policy']['cargo']])


class Fingerprint(unittest.TestCase):
    def test_every_host_names_the_inputs_of_each_operation(self):
        # No substitute host: the subcommand runs before any host check.
        def named(*arguments):
            return subprocess.run([BASH, str(ROOT / 'scripts/tooling/x86_64-linux.sh'), 'fingerprint', *arguments],
                                  capture_output=True, text=True)
        default, ci = named(), named('ci')
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertRegex(ci.stdout, r'^[0-9a-f]{64}\n$')
        self.assertEqual(default.stdout, named('dev').stdout)
        self.assertNotEqual(default.stdout, ci.stdout)
        self.assertEqual(ci.stdout, subprocess.check_output(
            [sys.executable, str(ROOT / 'scripts/tooling/fingerprint.py'), 'x86_64-linux', 'ci'], text=True))
        refused = named('ci-cross-run')
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn('no reusable ci-cross-run installation', refused.stderr)

    @unittest.skipUnless(shutil.which('pwsh') or shutil.which('powershell'), 'requires PowerShell')
    def test_windows_front_door_names_the_same_fingerprint(self):
        shell = shutil.which('pwsh') or shutil.which('powershell')
        for operation in ('dev', 'ci-bench'):
            with self.subTest(operation=operation):
                result = subprocess.run([shell, '-NoProfile', '-NonInteractive', '-File',
                                         str(ROOT / 'scripts/tooling/x86_64-win.ps1'), 'fingerprint', operation],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(result.stdout.strip(), subprocess.check_output(
                    [sys.executable, str(ROOT / 'scripts/tooling/fingerprint.py'), 'x86_64-win', operation],
                    text=True).strip())


class WindowsDownload(unittest.TestCase):
    @unittest.skipUnless(shutil.which('powershell') or shutil.which('pwsh'), 'requires PowerShell')
    def test_download_recovery_exhaustion_and_checksum(self):
        shell = shutil.which('powershell') or shutil.which('pwsh')
        result = subprocess.run([shell, '-NoProfile', '-NonInteractive', '-File',
                                 str(ROOT / 'scripts/tooling/windows_download_test.ps1')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class MacTools(unittest.TestCase):
    """The Apple Silicon Mac runs the catalog's uv and zig from the installers' layout, never from PATH."""

    def fixture(self, system='Darwin', machine='arm64'):
        temporary = tempfile.TemporaryDirectory(prefix='rscrypto-mac-')
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        checkout = root / 'checkout'
        for name in ('.config/tooling.toml', 'scripts/lib/python.sh', 'scripts/tooling/uv.sh',
                     'scripts/tooling/zig.sh', 'scripts/ct/zig-cc.sh'):
            (checkout / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, checkout / name)
        binaries = root / 'bin'
        binaries.mkdir()
        tools = {'uname': f'case "$1" in -s) echo {system} ;; -m) echo {machine} ;; *) echo {system} {machine} ;; esac',
                 # Homebrew's uv and zig on PATH must stay unused on the Mac.
                 'uv': 'echo \'["path-uv"]\' >> "$INSTALL_LOG"; exit 99',
                 'zig': 'echo \'["path-zig"]\' >> "$INSTALL_LOG"',
                 'curl': 'echo \'["curl"]\' >> "$INSTALL_LOG"\n'
                         'while [ $# -gt 0 ]; do [ "$1" != -o ] || output=$2; shift; done\ncp "$UV_ARCHIVE" "$output"'}
        for name, body in tools.items():
            (binaries / name).write_text('#!/bin/sh\n' + body + '\n')
            (binaries / name).chmod(0o755)
        (root / 'commands.jsonl').touch()
        env = {key: value for key, value in os.environ.items() if key != 'PYTHON'}
        env.update(HOME=str(root), PATH=str(binaries) + os.pathsep + os.environ['PATH'], BASH_ENV='/dev/null',
                   INSTALL_LOG=str(root / 'commands.jsonl'))
        return root, checkout, env

    def calls(self, root):
        return [json.loads(line) for line in (root / 'commands.jsonl').read_text().splitlines()]

    def test_python_installs_and_runs_the_pinned_uv(self):
        root, checkout, env = self.fixture()
        # The upstream archive layout: one directory holding the executables.
        archive = root / 'uv.tar.gz'
        stub(root / 'uv')
        with tarfile.open(archive, 'w:gz') as bundle:
            bundle.add(root / 'uv', 'uv-aarch64-apple-darwin/uv')
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        catalog = checkout / '.config/tooling.toml'
        pinned = CATALOG['aarch64-macos']['assets']['uv']['sha256']
        catalog.write_text(catalog.read_text().replace(pinned, digest))
        env.update(UV_ARCHIVE=str(archive), INSTALL_PYTHON=sys.executable)
        command = [str(checkout / 'scripts/lib/python.sh'), '-c', 'import os; print(os.environ["UV_PROJECT"])']
        for _ in range(2):
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(Path(result.stdout.strip()).resolve(), checkout.resolve())
        directory = root / '.local/share/rscrypto-tooling/uv' / digest[:16]
        self.assertEqual((directory / '.rscrypto-installed').read_text(), digest + '\n')
        calls = self.calls(root)
        # One verified download serves every later run; the Python that .python-version names comes with it.
        self.assertEqual(calls.count(['curl']), 1)
        self.assertEqual(calls.count(['uv', 'python', 'install', '--compile-bytecode']), 1)
        self.assertEqual(sum(call[:2] == ['uv', 'run'] for call in calls), 2)
        self.assertNotIn(['path-uv'], calls)

        # An archive that does not match its pin is never installed or run.
        catalog.write_text(catalog.read_text().replace(digest, 'f' * 64))
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((root / '.local/share/rscrypto-tooling/uv/ffffffffffffffff').exists())

    def zig_fixture(self, **host):
        root, checkout, env = self.fixture(**host)
        # Installation itself is the catalog's install-archive, which the Linux installers exercise.
        python = checkout / 'scripts/lib/python.sh'
        python.write_text('#!/bin/sh\nexec "$INSTALL_PYTHON" "$@"\n')
        stub(root / 'python')
        env['INSTALL_PYTHON'] = str(root / 'python')
        return root, checkout, env

    def test_zig_links_with_the_pinned_zig_only_on_the_mac(self):
        command = ['--target=powerpc64le-unknown-linux-gnu', 'main.o', '-o', 'main']
        root, checkout, env = self.zig_fixture()
        result = subprocess.run([str(checkout / 'scripts/ct/zig-cc.sh'), *command], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        prefix = root / '.local/share/rscrypto-tooling'
        self.assertEqual(self.calls(root), [
            ['python', str(checkout / 'scripts/tooling/catalog.py'), 'install-archive', 'aarch64-macos', 'zig', str(prefix)],
            ['zig', 'cc', '-target', 'powerpc64le-linux-gnu.2.32', 'main.o', '-o', 'main']])
        # CT provenance records the zig that linked the artifact.
        with patch.dict(os.environ, env):
            linker = provenance.resolve_executable(str(checkout / 'scripts/ct/zig-cc.sh'), checkout)
        self.assertEqual(linker, (prefix / 'zig/zig').resolve())

        # Linux hosts keep the zig that their installers put on PATH.
        root, checkout, env = self.zig_fixture(system='Linux', machine='x86_64')
        result = subprocess.run([str(checkout / 'scripts/ct/zig-cc.sh'), *command], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls(root), [['path-zig']])


if __name__ == '__main__':
    unittest.main()
