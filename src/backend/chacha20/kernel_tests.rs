use super::xor_keystream_portable;
#[cfg(target_arch = "aarch64")]
use crate::platform::caps::aarch64;
#[cfg(all(target_arch = "powerpc64", target_endian = "little", not(feature = "portable-only")))]
use crate::platform::caps::power;
#[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
use crate::platform::caps::riscv;
#[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
use crate::platform::caps::s390x;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::platform::caps::wasm;
#[cfg(target_arch = "x86_64")]
use crate::platform::caps::x86;
use crate::platform::{self, Caps};
use alloc::vec::Vec;
use serde_json::json;

/// Function pointer type for ChaCha20 XOR-keystream kernels.
type XorKeystreamFn = unsafe fn(&[u8; 32], u32, &[u8; 12], &mut [u8]);

#[derive(Clone, Copy)]
struct Backend {
  id: &'static str,
  required: Caps,
  xor_keystream: XorKeystreamFn,
}

/// Compiled backends keyed by name and required runtime capability. Backends
/// not compiled on this target are absent from the table; compiled backends
/// whose instructions are not legal on this host are skipped before invocation.
const BACKENDS: &[Backend] = &[
  #[cfg(target_arch = "aarch64")]
  Backend {
    id: "aarch64/neon",
    required: aarch64::NEON,
    xor_keystream: super::aarch64_neon::xor_keystream,
  },
  #[cfg(target_arch = "x86_64")]
  Backend {
    id: "x86_64/avx2",
    required: x86::AVX2,
    xor_keystream: super::x86_avx2::xor_keystream,
  },
  #[cfg(target_arch = "x86_64")]
  Backend {
    id: "x86_64/avx512",
    required: x86::AVX512_READY,
    xor_keystream: super::x86_avx512::xor_keystream,
  },
  #[cfg(all(target_arch = "powerpc64", target_endian = "little", not(feature = "portable-only")))]
  Backend {
    id: "powerpc64/vector",
    required: power::POWER8_VECTOR,
    xor_keystream: super::power_vsx::xor_keystream,
  },
  #[cfg(all(target_arch = "s390x", not(feature = "portable-only")))]
  Backend {
    id: "s390x/vector",
    required: s390x::VECTOR,
    xor_keystream: super::s390x_vector::xor_keystream,
  },
  #[cfg(all(target_arch = "riscv64", not(feature = "portable-only")))]
  Backend {
    id: "riscv64/vector",
    required: riscv::V,
    xor_keystream: super::riscv64_vector::xor_keystream,
  },
  #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
  Backend {
    id: "wasm32/simd128",
    required: wasm::SIMD128,
    xor_keystream: super::wasm_simd128::xor_keystream,
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
  let backends: Vec<_> = runnable_backends().collect();
  let mut cases = 0usize;
  let mut kernel_calls = 0usize;
  let key = [0xA5u8; 32];
  let nonce = [0x5Au8; 12];

  for &len in TEST_SIZES {
    let plain = deterministic_buffer(0x11, len);

    let mut expected = plain.clone();
    xor_keystream_portable(&key, 0, &nonce, &mut expected);

    for backend in &backends {
      let mut actual = plain.clone();
      // SAFETY: `runnable_backends` proves the exact capability set; `TEST_SIZES` uses at most 65 blocks from zero.
      unsafe { (backend.xor_keystream)(&key, 0, &nonce, &mut actual) };
      cases = cases.strict_add(1);
      kernel_calls = kernel_calls.strict_add(1);
      assert_eq!(
        actual, expected,
        "ChaCha20 backend {} diverged from portable at len={len}, counter=0",
        backend.id
      );
    }
  }
  report_backend_execution("counter-zero", &backends, cases, kernel_calls);
}

/// Backends must produce byte-identical output across mid-stream counter
/// values too. ChaCha20 is counter-driven; the rotation-amount bug fixed in
/// 2631aefa surfaced on every counter, but a hypothetical regression that
/// only affected one counter parity (e.g. odd-counter blocks) would slip
/// through `counter=0` testing alone.
#[test]
fn all_chacha20_backends_match_portable_at_arbitrary_counters() {
  let backends: Vec<_> = runnable_backends().collect();
  let mut cases = 0usize;
  let mut kernel_calls = 0usize;
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
      xor_keystream_portable(&key, counter, &nonce, &mut expected);

      for backend in &backends {
        let mut actual = plain.clone();
        // SAFETY: `runnable_backends` proves the exact capability set; at most 16 blocks from the largest tested
        // counter, 0x1000_0000, stays within the `u32` counter range.
        unsafe { (backend.xor_keystream)(&key, counter, &nonce, &mut actual) };
        cases = cases.strict_add(1);
        kernel_calls = kernel_calls.strict_add(1);
        assert_eq!(
          actual, expected,
          "ChaCha20 backend {} diverged from portable at len={len}, counter={counter}",
          backend.id
        );
      }
    }
  }
  report_backend_execution("arbitrary-counters", &backends, cases, kernel_calls);
}

fn runnable_backends() -> impl Iterator<Item = &'static Backend> {
  let caps = platform::caps();
  BACKENDS
    .iter()
    .filter(move |backend| backend.required == Caps::NONE || caps.has(backend.required))
}

fn report_backend_execution(test: &str, backends: &[&Backend], cases: usize, kernel_calls: usize) {
  let caps = platform::caps();
  let dispatch = if cfg!(feature = "portable-only") {
    "portable-only"
  } else {
    "production-auto"
  };
  let status = if cfg!(feature = "portable-only") {
    "not-selected"
  } else if BACKENDS.is_empty() {
    "not-compiled"
  } else if backends.is_empty() {
    "unavailable"
  } else {
    "pass"
  };
  let compiled: Vec<_> = BACKENDS
    .iter()
    .map(|backend| {
      json!({
        "id": backend.id,
        "required_features": backend.required.feature_names().collect::<Vec<_>>(),
        "runtime_available": (!cfg!(feature = "portable-only"))
          .then(|| backend.required == Caps::NONE || caps.has(backend.required)),
      })
    })
    .collect();
  let executed: Vec<_> = backends.iter().map(|backend| backend.id).collect();
  let record = json!({
    "schema": 1,
    "kind": "rscrypto.backend-execution",
    "primitive": "chacha20",
    "test": test,
    "dispatch": dispatch,
    "target_arch": std::env::consts::ARCH,
    "compiled": compiled,
    "executed_backend_ids": executed,
    "executed_case_count": cases,
    "kernel_call_count": kernel_calls,
    "result": status,
  });
  use std::io::Write;
  writeln!(std::io::stderr().lock(), "RSCRYPTO_BACKEND_EVIDENCE={record}").expect("write backend execution record");
}
