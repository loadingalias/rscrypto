//! Forced-kernel ChaCha20 backend equivalence vs the portable oracle.
//!
//! Mirrors the `tests/argon2_kernels.rs` harness: drive every compiled SIMD
//! backend (NEON / AVX2 / AVX-512 / VSX / z/Vector / RVV / simd128) with the
//! same ChaCha20 inputs and assert byte-identical output against the portable
//! kernel. Full AEAD algorithms are covered by oracle tests that run under both
//! normal dispatch and `portable-only` feature-matrix lanes.
//!
//! This is the test that catches CRIT-1 class silent kernel divergence —
//! the bug that shipped on POWER VSX and s390x z/Vector for ChaCha20 prior
//! to commit `2631aefa` (rotate-left amounts inverted via `splat_ror(N) =
//! splat(32 - N)`). With this harness in CI gating green on every target
//! that compiles a SIMD backend, that class of regression cannot land
//! silently again.
//!
//! The harness deliberately does NOT duplicate test logic per backend.
//! A single generic loop iterates the static list of compiled backends,
//! calls each diag entry point, and compares against the portable oracle.
//! Adding a new backend requires only listing it in the `BACKENDS` table.

#![cfg(all(feature = "diag", feature = "chacha20poly1305"))]

#[cfg(target_arch = "aarch64")]
use rscrypto::platform::caps::aarch64;
#[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
use rscrypto::platform::caps::power;
#[cfg(target_arch = "riscv64")]
use rscrypto::platform::caps::riscv;
#[cfg(target_arch = "s390x")]
use rscrypto::platform::caps::s390x;
#[cfg(target_arch = "wasm32")]
use rscrypto::platform::caps::wasm;
#[cfg(target_arch = "x86_64")]
use rscrypto::platform::caps::x86;
use rscrypto::{
  aead::diag_chacha20_xor_keystream_portable,
  platform::{self, Caps},
};

/// Function pointer type for ChaCha20 XOR-keystream kernels.
type XorKeystreamFn = fn(&[u8; 32], u32, &[u8; 12], &mut [u8]);

#[derive(Clone, Copy)]
struct Backend {
  name: &'static str,
  required: Caps,
  xor_keystream: XorKeystreamFn,
}

/// Compiled backends keyed by name and required runtime capability. Backends
/// not compiled on this target are absent from the table; compiled backends
/// whose instructions are not legal on this host are skipped before invocation.
const BACKENDS: &[Backend] = &[
  #[cfg(target_arch = "aarch64")]
  Backend {
    name: "aarch64-neon",
    required: aarch64::NEON,
    xor_keystream: rscrypto::aead::diag_chacha20_xor_keystream_aarch64_neon,
  },
  #[cfg(all(target_arch = "aarch64", any(target_os = "linux", target_os = "macos")))]
  Backend {
    name: "aarch64-owned-asm-8block",
    required: aarch64::NEON,
    xor_keystream: rscrypto::aead::diag_chacha20_xor_keystream_aarch64_owned_asm,
  },
  #[cfg(target_arch = "x86_64")]
  Backend {
    name: "x86-avx2",
    required: x86::AVX2,
    xor_keystream: rscrypto::aead::diag_chacha20_xor_keystream_x86_avx2,
  },
  #[cfg(target_arch = "x86_64")]
  Backend {
    name: "x86-avx512",
    required: x86::AVX512_READY,
    xor_keystream: rscrypto::aead::diag_chacha20_xor_keystream_x86_avx512,
  },
  #[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
  Backend {
    name: "power-vsx",
    required: power::POWER8_VECTOR,
    xor_keystream: rscrypto::aead::diag_chacha20_xor_keystream_power_vsx,
  },
  #[cfg(target_arch = "s390x")]
  Backend {
    name: "s390x-vector",
    required: s390x::VECTOR,
    xor_keystream: rscrypto::aead::diag_chacha20_xor_keystream_s390x_vector,
  },
  #[cfg(target_arch = "riscv64")]
  Backend {
    name: "riscv64-vector",
    required: riscv::V,
    xor_keystream: rscrypto::aead::diag_chacha20_xor_keystream_riscv64_vector,
  },
  #[cfg(target_arch = "wasm32")]
  Backend {
    name: "wasm-simd128",
    required: wasm::SIMD128,
    xor_keystream: rscrypto::aead::diag_chacha20_xor_keystream_wasm_simd128,
  },
];

/// Boundary-aware test sizes. Each ChaCha20 block is 64 bytes; SIMD
/// kernels typically batch 4–16 blocks at a time. Hit:
///   - 0 (empty input)
///   - 1, 32, 63 (sub-block tails)
///   - 64 (exactly one block)
///   - 65, 127 (sub-batch tails)
///   - 128, 256, 512 (multi-block batches across SIMD batch boundaries)
///   - 1024, 4096 (large enough to exercise main loops fully)
///   - 4097 (large + boundary tail)
const TEST_SIZES: &[usize] = &[
  0, 1, 32, 63, 64, 65, 127, 128, 256, 511, 512, 1023, 1024, 4095, 4096, 4097,
];

fn deterministic_buffer(seed: u8, len: usize) -> Vec<u8> {
  // Reproducible "random-ish" bytes from a tiny LCG so test failures are
  // deterministic and re-runnable. seed varies per call site so different
  // tests pick different starting bytes.
  let mut buf = Vec::with_capacity(len);
  let mut x = seed as u32;
  for _ in 0..len {
    x = x.wrapping_mul(1664525).wrapping_add(1013904223);
    buf.push((x >> 24) as u8);
  }
  buf
}

/// All compiled backends produce byte-identical output for every test size
/// at counter=0 (the most common entry point).
#[test]
fn all_chacha20_backends_match_portable_at_counter_zero() {
  let key = [0xA5u8; 32];
  let nonce = [0x5Au8; 12];

  for &len in TEST_SIZES {
    let plain = deterministic_buffer(0x11, len);

    let mut expected = plain.clone();
    diag_chacha20_xor_keystream_portable(&key, 0, &nonce, &mut expected);

    for backend in runnable_backends() {
      let mut actual = plain.clone();
      (backend.xor_keystream)(&key, 0, &nonce, &mut actual);
      assert_eq!(
        actual, expected,
        "ChaCha20 backend {} diverged from portable at len={len}, counter=0",
        backend.name
      );
    }
  }
}

/// Backends must produce byte-identical output across mid-stream counter
/// values too. ChaCha20 is counter-driven; the rotation-amount bug fixed in
/// 2631aefa surfaced on every counter, but a hypothetical regression that
/// only affected one counter parity (e.g. odd-counter blocks) would slip
/// through `counter=0` testing alone.
#[test]
fn all_chacha20_backends_match_portable_at_arbitrary_counters() {
  let key = [0x33u8; 32];
  let nonce = [0xCCu8; 12];
  // Counter values to exercise: 0 (already covered above), 1, 7 (small
  // single-block), 64, 1023, 65536 (mid-range crossings of u8/u16
  // boundaries that could expose endianness bugs in the counter increment
  // path). RFC 8439 limits each ChaCha20 stream to 2^32 blocks; SIMD
  // kernels assert counter + batch fits in u32, so we stay well below max.
  for &counter in &[0u32, 1, 7, 64, 1023, 65_536, 0x1000_0000] {
    for &len in &[64usize, 256, 1024] {
      let plain = deterministic_buffer(0x99, len);

      let mut expected = plain.clone();
      diag_chacha20_xor_keystream_portable(&key, counter, &nonce, &mut expected);

      for backend in runnable_backends() {
        let mut actual = plain.clone();
        (backend.xor_keystream)(&key, counter, &nonce, &mut actual);
        assert_eq!(
          actual, expected,
          "ChaCha20 backend {} diverged from portable at len={len}, counter={counter}",
          backend.name
        );
      }
    }
  }
}

/// XOR keystream is its own inverse: applying it twice must restore the
/// plaintext. This test catches a class of bugs where the kernel produces
/// stable but wrong output (consistent across runs but not actually XOR).
#[test]
fn all_chacha20_backends_self_inverse() {
  let key = [0x77u8; 32];
  let nonce = [0x88u8; 12];

  for &len in TEST_SIZES {
    let original = deterministic_buffer(0xDE, len);

    for backend in runnable_backends() {
      let mut buffer = original.clone();
      (backend.xor_keystream)(&key, 0, &nonce, &mut buffer);
      (backend.xor_keystream)(&key, 0, &nonce, &mut buffer);
      assert_eq!(
        buffer, original,
        "ChaCha20 backend {} not self-inverse at len={len}",
        backend.name
      );
    }
  }
}

fn runnable_backends() -> impl Iterator<Item = &'static Backend> {
  let caps = platform::caps();
  BACKENDS
    .iter()
    .filter(move |backend| backend.required == Caps::NONE || caps.has(backend.required))
}
