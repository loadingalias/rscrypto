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
    elif script == 'toolchain.py' and '--install' in args:
        pass
    else:
        sys.exit(subprocess.run([sys.executable, *args]).returncode)
'''


class LinuxInstall(unittest.TestCase):
    def provision(self, platform, fail=False):
        temporary = tempfile.TemporaryDirectory(prefix='rscrypto installer ')
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        binaries = root / 'bin'
        binaries.mkdir()
        for name in ('uname', 'id', 'apt-get', 'apt-cache', 'cargo', 'clang', 'cmake', 'python3'):
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
        result = subprocess.run([BASH, str(ROOT / 'scripts/tooling/linux.sh'), platform, '--ci'],
                                env=env, capture_output=True, text=True)
        calls = [json.loads(line) for line in (root / 'commands.jsonl').read_text().splitlines()]
        return result, calls, root

    def test_ci_profiles_install_only_required_tools(self):
        for platform in ('x86_64-linux', 'aarch64-linux', 's390x-linux', 'powerpc64le-linux', 'riscv64-linux'):
            with self.subTest(platform=platform):
                result, calls, root = self.provision(platform)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                installs = [c for c in calls if c[0] == 'cargo' and ('install' in c or 'binstall' in c)]
                self.assertEqual(len(installs), len(CATALOG['ci']['cargo']))
                binary = platform in ('x86_64-linux', 'aarch64-linux')
                for command, tool in zip(installs, CATALOG['ci']['cargo']):
                    self.assertIn('binstall' if binary else 'install', command)
                    self.assertEqual(command[-1], f"{tool}@{CATALOG['cargo'][tool]}" if binary else tool)
                    self.assertIn('--locked', command)
                    self.assertIn(CATALOG[platform]['rust-host'], command)
                apt = next(c for c in calls if c[0] == 'apt-get' and '--allow-downgrades' in c)
                self.assertEqual([a for a in apt if a.endswith('=1.0')],
                                 [p + '=1.0' for p in CATALOG['linux-ci']['packages']])
                environment = (root / '.local/share/rscrypto-tooling/environment.sh').read_text()
                self.assertIn('custom\\ cargo/bin', environment)
                self.assertFalse((root / '.bashrc').exists())
                self.assertFalse((root / '.profile').exists())

    def test_package_failure_stops_before_rust_installation(self):
        result, calls, _ = self.provision('x86_64-linux', fail=True)
        self.assertEqual(result.returncode, 42)
        self.assertFalse(any(c[0] == 'cargo' or 'download' in c for c in calls))


if __name__ == '__main__':
    unittest.main()
