"""Cross-build configuration for native Linux transfer runners."""

import os
import platform
import sys
import toolchain

# Target triple: (uname machine, GNU tool prefix, ELF byte order, ELF machine).
TARGETS = {
    'riscv64gc-unknown-linux-gnu': ('riscv64', 'riscv64-linux-gnu', 'little', 243),
    'powerpc64le-unknown-linux-gnu': ('ppc64le', 'powerpc64le-linux-gnu', 'little', 21),
    's390x-unknown-linux-gnu': ('s390x', 's390x-linux-gnu', 'big', 22),
}
LINUX_TARGETS = {
    'aarch64-unknown-linux-gnu': ('aarch64', 'aarch64-linux-gnu', 'little', 183),
    **TARGETS,
    'x86_64-unknown-linux-gnu': ('x86_64', 'x86_64-linux-gnu', 'little', 62),
}


def require_host(target):
    machine, _, _, _ = LINUX_TARGETS[target]
    if (platform.system(), platform.machine()) != ('Linux', machine):
        raise ValueError(f'transferred evidence requires native Linux {machine}')


def verify_elf(path, target):
    _, _, endian, machine = LINUX_TARGETS[target]
    with path.open('rb') as source:
        header = source.read(20)
    if (len(header) != 20 or header[:5] != b'\x7fELF\x02'
            or header[5] != (1 if endian == 'little' else 2)
            or int.from_bytes(header[18:20], endian) != machine):
        raise ValueError(f'not a {target} ELF64 executable: {path}')


def environment(target):
    _, compiler, _, _ = LINUX_TARGETS[target]
    conflicting = [key for key in os.environ if key in {
        'RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'RUSTDOCFLAGS', 'CARGO_ENCODED_RUSTDOCFLAGS', 'RUSTC', 'RUSTDOC',
        'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER', 'CARGO_BUILD_RUSTFLAGS', 'CARGO_BUILD_TARGET',
    } or key.startswith(('CARGO_PROFILE_', 'CARGO_TARGET_', 'NEXTEST_'))]
    if conflicting:
        raise ValueError(f'unreviewed build/test overrides: {sorted(conflicting)}')
    normalized = target.replace('-', '_')
    return {**os.environ, 'RUSTUP_TOOLCHAIN': toolchain.for_target(target),
            f'CARGO_TARGET_{normalized.upper()}_LINKER': compiler + '-gcc',
            f'CC_{normalized}': compiler + '-gcc',
            f'CXX_{normalized}': compiler + '-g++',
            f'AR_{normalized}': compiler + '-ar'}


if __name__ == '__main__':
    # The shell installer uses the same closed target set as the transfer code.
    target, = sys.argv[1:]
    print(LINUX_TARGETS[target][1])
