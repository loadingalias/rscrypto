# LLVM ppc64le volatile-load SHA-256 wrong-code packet

**Status:** rscrypto workaround is in place. Upstream LLVM issue is not filed.

This is the working packet for filing the bug later. Do not file it until the
product reproducer below has been rerun on native `powerpc64le-unknown-linux-gnu`
and the failing log is attached. A standalone reducer is better, but the
rscrypto product repro is enough to open a useful issue if reduction stalls.

## Known Facts

On native POWER10 ppc64le CI, SHA-256 produced wrong NIST vector output when
the portable SHA-256 round constants were loaded with
`core::ptr::read_volatile(K.0.as_ptr().add(i))`.

The affected pattern was in `src/hashes/crypto/sha256/mod.rs`:

```rust
static K: Aligned64<[u32; 64]> = Aligned64([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, // ...
]);

#[inline(always)]
fn rk(i: usize) -> u32 {
    // SAFETY: i is always in 0..64, and K has exactly 64 elements.
    unsafe { core::ptr::read_volatile(K.0.as_ptr().add(i)) }
}
```

The same SHA-256 vectors passed after replacing the ppc64le volatile path with
plain direct indexing. Commit `a70c6596` made that POWER-only correctness
mitigation. Later, commit `2a769634` replaced `read_volatile` on every non-x86
target with a `core::hint::black_box` base pointer plus ordinary `ptr::read`.

Observed failure shape:

- Target: `powerpc64le-unknown-linux-gnu`.
- Hardware: native POWER10 runner.
- Optimization: release / `-C opt-level=3`.
- Symptom: deterministic SHA-256 mismatch against NIST vector corpus.
- Mitigation: direct indexing or `black_box(K.0.as_ptr())` plus ordinary load.
- Not a threading issue: the constant table is immutable and the failure was
  deterministic.

Local toolchain note: `nightly-2026-04-18` reports `rustc 1.97.0-nightly
(e9e32aca5 2026-04-17)` and `LLVM version: 22.1.2`. Do not claim LLVM 18
unless the ppc64le runner reports that exact version.

## Current rscrypto Mitigation

Current `rk()` intentionally avoids volatile loads:

```rust
#[inline(always)]
fn rk(i: usize) -> u32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // SAFETY: i is always in 0..64, and K has exactly 64 elements.
        unsafe { core::ptr::read(K.0.as_ptr().add(i)) }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let base = core::hint::black_box(K.0.as_ptr());
        // SAFETY: i is always in 0..64, K has exactly 64 elements, and
        // `black_box(ptr)` returns the same provenance / value.
        unsafe { core::ptr::read(base.add(i)) }
    }
}
```

This keeps the desired single memory-load codegen on non-x86 targets without
relying on LLVM volatile semantics for an ordinary constant table.

## Why This Looks Like Compiler Wrong-Code

For memory inside a Rust allocation, `core::ptr::read_volatile` is documented
as behaving like `read` except that the volatile access must actually happen
and must not be elided or reordered across other externally observable events.
The pointer used here is in-bounds, aligned, and points into an initialized
`static` table of `u32` values.

LLVM IR models this as `load volatile`. LLVM's language reference constrains
volatile load/store transformations and says the backend should not split or
merge target-legal volatile load/store instructions.

If a valid volatile `u32` load from an immutable table produces a different
SHA-256 result than a valid non-volatile load from the same address, the useful
default assumption is backend or optimizer wrong-code. The open question is
where the wrong-code enters: Rust lowering, LLVM IR optimization, SelectionDAG,
PowerPC address-mode selection, scheduling, or final machine code emission.

## Reproduce And File

Run this on native ppc64le Linux. Cross-compiling from another host is not
enough because the failure is a runtime wrong-code symptom.

### 1. Create a throwaway branch

```sh
git switch -c repro/ppc64le-read-volatile-sha256
rustc -Vv | tee /tmp/rscrypto-ppc64le-rustc.txt
uname -a | tee /tmp/rscrypto-ppc64le-uname.txt
```

### 2. Restore the volatile round-constant load

Apply this temporary patch only for the repro branch:

```diff
diff --git a/src/hashes/crypto/sha256/mod.rs b/src/hashes/crypto/sha256/mod.rs
@@
   #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
   {
-    let base = core::hint::black_box(K.0.as_ptr());
-    // SAFETY: i is always in 0..64, K has exactly 64 elements, and
-    // `black_box(ptr)` returns the same provenance / value, only opaque
-    // to subsequent constant propagation. The pointer arithmetic stays
-    // within the `K` allocation.
-    unsafe { core::ptr::read(base.add(i)) }
+    // SAFETY: i is always in 0..64, and K has exactly 64 elements.
+    unsafe { core::ptr::read_volatile(K.0.as_ptr().add(i)) }
   }
 }
```

Then confirm the patch is the only source change:

```sh
git diff -- src/hashes/crypto/sha256/mod.rs
git diff -- src/hashes/crypto/sha256/mod.rs > /tmp/rscrypto-volatile-sha256.patch
```

### 3. Run the product repro

```sh
cargo test --release --test sha256_official_vectors sha256_official_vectors -- --exact --nocapture \
  2>&1 | tee /tmp/rscrypto-ppc64le-sha256-volatile.log
```

Expected result if the bug still reproduces:

- The test fails with a SHA-256 vector mismatch.
- `/tmp/rscrypto-ppc64le-sha256-volatile.log` contains the failing case index,
  input length, expected digest, and actual digest.

Now restore the current mitigation and prove the same vector passes:

```sh
git restore src/hashes/crypto/sha256/mod.rs
cargo test --release --test sha256_official_vectors sha256_official_vectors -- --exact --nocapture \
  2>&1 | tee /tmp/rscrypto-ppc64le-sha256-black-box.log
```

Expected result:

- The unmodified current tree passes.
- The only behavioral difference is volatile load versus the current
  `black_box` ordinary load.

### 4. Capture compiler artifacts

With the volatile patch reapplied, collect LLVM IR and assembly:

```sh
git apply /tmp/rscrypto-volatile-sha256.patch
cargo rustc --release --lib -- --emit=llvm-ir,asm
find target/release/deps \( -name 'rscrypto-*.ll' -o -name 'rscrypto-*.s' \) -print
```

Attach the smallest generated `.ll` and `.s` files that contain SHA-256
compression. If they are too large, attach the product test log first and mark
reduction as pending.

### 5. Try a one-file reducer

First try reducing to the SHA-256 compression path with only the constant-load
strategy changed. If the reduced file fails, submit that instead of the full
rscrypto product repro. If it does not fail, keep the product repro and say the
bug currently requires the full unrolled rscrypto shape.

Minimum reducer requirements:

- One Rust file or one LLVM IR file.
- No external dependencies.
- Compiled with `rustc -O` or `llc -O3`.
- Prints or asserts expected and actual SHA-256 digest for `b"abc"`.
- Has one switch between volatile and non-volatile K loads.
- Shows volatile fails and non-volatile passes on the same ppc64le machine.

Use this command shape for the final reducer:

```sh
rustc -O /tmp/volatile_sha256_repro.rs -o /tmp/volatile_sha256_repro
/tmp/volatile_sha256_repro
rustc -O --emit=llvm-ir,asm /tmp/volatile_sha256_repro.rs
```

Do not spend days chasing a perfect reducer. If the product repro is crisp and
the artifacts show `load volatile`, file the issue and let LLVM maintainers ask
for a narrower reduction if needed.

### 6. Search for duplicates before filing

Search LLVM issues again before opening a new one:

```text
repo:llvm/llvm-project is:issue ppc64le "load volatile" miscompile
repo:llvm/llvm-project is:issue "read_volatile" ppc64le
repo:llvm/llvm-project is:issue "PowerPC" "load volatile" wrong-code
```

As of 2026-04-29, no exact duplicate was found. Related but different issue:
https://github.com/llvm/llvm-project/issues/127298.

### 7. File the LLVM issue

Open https://github.com/llvm/llvm-project/issues/new and use this body:

````markdown
## Summary

On native `powerpc64le-unknown-linux-gnu`, LLVM appears to miscompile a Rust
SHA-256 compression path when the immutable round-constant table is read with
`core::ptr::read_volatile`. Replacing the volatile load with an ordinary load
from the same `static` table produces correct NIST SHA-256 vectors.

## Environment

- Target: `powerpc64le-unknown-linux-gnu`
- Hardware / OS: <paste `uname -a`>
- rustc: <paste `rustc -Vv`>
- Optimization: release / `-C opt-level=3`
- Project: https://github.com/loadingalias/rscrypto
- Context commit: `a70c6596` first mitigated the POWER failure by avoiding the
  volatile path; current mitigation uses `black_box` plus ordinary `ptr::read`.

## Reproducer

<Prefer a standalone Rust or LLVM IR reducer. If reduction failed, paste the
temporary rscrypto patch from this document and the exact `cargo test` command.>

```sh
cargo test --release --test sha256_official_vectors sha256_official_vectors -- --exact --nocapture
```

## Expected Result

The SHA-256 digest matches the NIST vector corpus. Volatile and non-volatile
loads from the same immutable `static` `u32` table should produce the same
loaded values.

## Actual Result

The volatile-load variant deterministically produces a wrong SHA-256 digest on
ppc64le. The same source with an ordinary load passes on the same machine.

Attach:

- `/tmp/rscrypto-ppc64le-rustc.txt`
- `/tmp/rscrypto-ppc64le-uname.txt`
- `/tmp/rscrypto-ppc64le-sha256-volatile.log`
- `/tmp/rscrypto-ppc64le-sha256-black-box.log`
- LLVM IR / assembly for the volatile variant, if available.

## Notes

Rust documents `read_volatile` inside an allocation as equivalent to `read`
except that the access must not be elided or reordered across externally
observable events. LLVM IR lowers this to `load volatile`.
````

After filing, update this document with the issue URL and decide whether the
`black_box` mitigation can be removed only after a fixed LLVM has been verified
on native ppc64le CI.

## References

- Rust `read_volatile` docs: https://doc.rust-lang.org/core/ptr/fn.read_volatile.html
- LLVM LangRef volatile rules: https://llvm.org/docs/LangRef.html
- Related LLVM PowerPC volatile issue: https://github.com/llvm/llvm-project/issues/127298
- rscrypto mitigation commit: `a70c6596`
- rscrypto `black_box` replacement commit: `2a769634`
