"""Pinned RISC-V cross-build environment shared by tests and CT."""

import os
import toolchain

TARGET = 'riscv64gc-unknown-linux-gnu'


def environment():
    conflicting = [key for key in os.environ if key in {
        'RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'RUSTDOCFLAGS', 'CARGO_ENCODED_RUSTDOCFLAGS', 'RUSTC', 'RUSTDOC',
        'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER', 'CARGO_BUILD_RUSTFLAGS',
    } or key.startswith(('CARGO_PROFILE_', 'CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_', 'NEXTEST_'))]
    if conflicting:
        raise ValueError(f'unreviewed build/test overrides: {sorted(conflicting)}')
    return {**os.environ, 'RUSTUP_TOOLCHAIN': toolchain.for_target(TARGET), 'CARGO_RAIL_CACHE': 'off',
            'CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER': 'riscv64-linux-gnu-gcc',
            'CC_riscv64gc_unknown_linux_gnu': 'riscv64-linux-gnu-gcc',
            'CXX_riscv64gc_unknown_linux_gnu': 'riscv64-linux-gnu-g++',
            'AR_riscv64gc_unknown_linux_gnu': 'riscv64-linux-gnu-ar'}
