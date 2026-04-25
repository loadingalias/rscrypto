# LLVM ppc64le miscompilation report — `core::ptr::read_volatile` on a `pub static` produces wrong SHA-256 output

**Status:** Workaround in place. Upstream issue not yet filed.

This document captures the reproducer and rscrypto's mitigation so we can
file the LLVM issue cleanly when bandwidth allows.

## Symptom

On `powerpc64le-unknown-linux-gnu`, the SHA-256 round-constant load via
`core::ptr::read_volatile` produced wrong NIST CAVP vectors despite the
read happening from a non-mutable `pub static const K: KConsts`. The same
pattern was correct on aarch64, riscv64, s390x, and wasm32.

The bug surfaced on POWER10 native CI (rscrypto commit `a70c6596` reverted
the POWER path to direct indexing as a working mitigation). After the fix
landed, every NIST CAVP vector passes.

The minimum reproducer below should be a one-file LLVM issue.

## Reproducer (rscrypto-rooted)

The pattern was:

```rust
#[repr(C, align(64))]
struct KConsts([u32; 64]);

const K: KConsts = KConsts([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, // ... 64 SHA-256 round constants
]);

#[inline(always)]
fn rk(i: usize) -> u32 {
    // SAFETY: i is always in 0..64.
    unsafe { core::ptr::read_volatile(K.0.as_ptr().add(i)) }
}

// rk(i) was called inside a fully-unrolled SHA-256 round loop.
```

After running through the SHA-256 message schedule and round function, the
final state on ppc64le diverged from the NIST published value at byte 0.
The miscompilation was deterministic — every test run produced the same
wrong bytes — so it was not a memory-ordering / threading interaction.

## Bisection

- LLVM toolchain: nightly-2026-04-18 backed by LLVM ~18.x (rscrypto MSRV 1.95.0).
- Target: `powerpc64le-unknown-linux-gnu` (gcc-toolchain v13, glibc 2.39).
- Optimization: `-C opt-level=3` (release).

Replacing `core::ptr::read_volatile(K.0.as_ptr().add(i))` with plain
`K.0[i]` produces the correct hash on the same toolchain. The volatile load
is the only differing instruction.

## Mitigation in rscrypto

We dropped `read_volatile` from `rk()` on every non-x86 target and replaced
it with a `core::hint::black_box` over the table base pointer. `black_box`
prevents constant-propagation of the pointer (so the compiler still emits a
single load instruction instead of materialising 32-bit/64-bit immediates
via multi-instruction sequences) without invoking volatile semantics:

```rust
#[inline(always)]
fn rk(i: usize) -> u32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    { unsafe { core::ptr::read(K.0.as_ptr().add(i)) } }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let base = core::hint::black_box(K.0.as_ptr());
        unsafe { core::ptr::read(base.add(i)) }
    }
}
```

This produces correct output on POWER10 plus the aarch64/riscv64/s390x/wasm32
targets the previous `read_volatile` had also covered, while removing the
volatile-related sharp edge entirely.

## Why this is an LLVM-side issue

`core::ptr::read_volatile` lowers to LLVM's `load volatile` intrinsic.
Reading volatile from a non-`mut` `pub static` should be observably
equivalent to reading the same address non-volatile (the value is
constant), but with the additional guarantee that the load is not elided
or reordered with respect to other volatile operations.

The miscompilation suggests either:

1. The ppc64le backend reorders volatile loads with adjacent arithmetic in
   a way that materialises wrong K values when the compress-loop is
   aggressively unrolled, or
2. Some pass between IR and ppc64le codegen treats `load volatile` from a
   constant address differently than expected — e.g. erroneously eliding
   to a constant despite `volatile`, or alias-analysis confusing the
   `K.0.as_ptr().add(i)` provenance.

A reduced reproducer that boils SHA-256 down to a single round picking K
constants and asserts a specific intermediate state would let LLVM
maintainers triage which pass introduces the divergence.

## Action items (when filing)

1. Reduce to a single `#[no_mangle]` function in C-via-`extern "C"` Rust
   that takes an index, calls `read_volatile(K.as_ptr().add(i))`, and
   returns the loaded value. Compare assembly output between ppc64le and
   x86_64 at `-O3`.
2. Confirm the bug reproduces with stable Rust 1.95.0 + LLVM 18 ppc64le.
3. File at https://github.com/llvm/llvm-project/issues with the reduced
   reproducer plus rscrypto's `a70c6596` commit as the contextual link.
4. Subscribe rscrypto's release process to the LLVM issue so we can drop
   the workaround when fixed.

## Related

- rscrypto commit `a70c6596` — POWER ppc64le SHA-256 mitigation.
- rscrypto commit (this branch) — `black_box`-based replacement on all
  non-x86 targets.
- LLVM upstream: `volatile` load semantics across architectures, especially
  PPC backend's `legalizeOperationsLoad`.
