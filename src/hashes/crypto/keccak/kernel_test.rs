use alloc::vec::Vec;

use super::kernels::{ALL, permute_fn, required_caps};

#[derive(Clone, Debug)]
pub struct KernelResult {
  pub name: &'static str,
  pub state: [u64; 25],
}

fn state_from_bytes(data: &[u8]) -> [u64; 25] {
  let mut buf = [0u8; 200];
  for (dst, src) in buf.iter_mut().zip(data.iter()) {
    *dst = *src;
  }

  let mut out = [0u64; 25];
  let (chunks, rem) = buf.as_chunks::<8>();
  debug_assert!(rem.is_empty());
  for (lane, chunk) in out.iter_mut().zip(chunks.iter()) {
    *lane = u64::from_le_bytes(*chunk);
  }
  out
}

#[must_use]
pub fn run_all_keccakf1600_kernels(data: &[u8]) -> Vec<KernelResult> {
  let caps = crate::platform::caps();
  let mut out = Vec::with_capacity(ALL.len());
  let init = state_from_bytes(data);

  for &id in ALL {
    if !caps.has(required_caps(id)) {
      continue;
    }
    let mut st = init;
    (permute_fn(id))(&mut st);
    out.push(KernelResult {
      name: id.as_str(),
      state: st,
    });
  }

  out
}

pub fn verify_keccakf1600_kernels(data: &[u8]) -> Result<(), &'static str> {
  let results = run_all_keccakf1600_kernels(data);
  let Some(first) = results.first() else {
    return Ok(());
  };
  for r in &results[1..] {
    if r.state != first.state {
      return Err("keccakf1600 kernel mismatch");
    }
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn keccakf1600_zero_vector_matches_known_state() {
    // Known output of Keccak-f[1600] applied to an all-zero state.
    // (This vector is widely used for self-tests across implementations.)
    let expected: [u64; 25] = [
      0xF1258F7940E1DDE7,
      0x84D5CCF933C0478A,
      0xD598261EA65AA9EE,
      0xBD1547306F80494D,
      0x8B284E056253D057,
      0xFF97A42D7F8E6FD4,
      0x90FEE5A0A44647C4,
      0x8C5BDA0CD6192E76,
      0xAD30A6F71B19059C,
      0x30935AB7D08FFC64,
      0xEB5AA93F2317D635,
      0xA9A6E6260D712103,
      0x81A57C16DBCF555F,
      0x43B831CD0347C826,
      0x01F22F1A11A5569F,
      0x05E5635A21D9AE61,
      0x64BEFEF28CC970F2,
      0x613670957BC46611,
      0xB87C5A554FD00ECB,
      0x8C3EE88A1CCF32C8,
      0x940C7922AE3A2614,
      0x1841F924A2C509E4,
      0x16F53526E70465C2,
      0x75F644E97F30A13B,
      0xEAF1FF7B5CECA249,
    ];

    let results = run_all_keccakf1600_kernels(&[]);
    assert!(!results.is_empty());
    for r in results {
      assert_eq!(r.state, expected, "keccakf1600 mismatch for kernel={}", r.name);
    }
  }

  #[test]
  fn run_all_agree() {
    verify_keccakf1600_kernels(b"abc").expect("kernels should agree");
    verify_keccakf1600_kernels(&[0u8; 200]).expect("kernels should agree");
  }

  /// Every rscrypto keccakf1600 kernel permutes the state identically to the
  /// `keccak` crate's soft backend, for a batch of patterned inputs.
  #[cfg(not(miri))]
  #[test]
  fn all_kernels_match_keccak_crate_oracle() {
    let cases: &[&[u8]] = &[
      &[],
      b"abc",
      b"1600-bit block fills exactly",
      &[0u8; 200],
      &[0xffu8; 200],
    ];

    let k = keccak::Keccak::new();
    for input in cases {
      let expected = {
        let mut state = state_from_bytes(input);
        k.with_f1600(|f1600| f1600(&mut state));
        state
      };

      for r in run_all_keccakf1600_kernels(input) {
        assert_eq!(r.state, expected, "kernel={} mismatches keccak crate oracle", r.name);
      }
    }
  }

  #[cfg(not(miri))]
  proptest::proptest! {
    /// Every rscrypto keccakf1600 kernel matches the `keccak` crate across
    /// arbitrary 1600-bit state inputs.
    #[test]
    fn kernels_match_keccak_crate_oracle(bytes in proptest::collection::vec(proptest::prelude::any::<u8>(), 0..=200)) {
      let k = keccak::Keccak::new();
      let expected = {
        let mut state = state_from_bytes(&bytes);
        k.with_f1600(|f1600| f1600(&mut state));
        state
      };

      for r in run_all_keccakf1600_kernels(&bytes) {
        proptest::prop_assert_eq!(r.state, expected, "kernel {} mismatches keccak oracle", r.name);
      }
    }
  }

  /// Verify the x86_64 AVX-512 two-state kernel matches two scalar permutations.
  #[test]
  #[cfg(all(target_arch = "x86_64", not(miri)))]
  fn keccakf1600_x86_avx512_x2_matches_single_state() {
    let caps = crate::platform::caps();
    let required = crate::platform::caps::x86::AVX512F
      .union(crate::platform::caps::x86::AVX512VL)
      .union(crate::platform::caps::x86::SSE41);
    if !caps.has(required) {
      return;
    }

    for seed in 0u8..16 {
      let mut input_a = [0u8; 200];
      let mut input_b = [0u8; 200];
      for (i, byte) in input_a.iter_mut().enumerate() {
        *byte = seed.wrapping_add((i as u8).wrapping_mul(17));
      }
      for (i, byte) in input_b.iter_mut().enumerate() {
        *byte = seed.wrapping_mul(3).wrapping_add((i as u8).wrapping_mul(29));
      }

      let mut state_a = state_from_bytes(&input_a);
      let mut state_b = state_from_bytes(&input_b);
      let mut expected_a = state_a;
      let mut expected_b = state_b;

      super::super::keccakf_portable(&mut expected_a);
      super::super::keccakf_portable(&mut expected_b);
      // SAFETY: AVX-512 x2 test call because:
      // 1. Runtime caps are checked above for AVX512F, AVX512VL, and SSE4.1.
      // 2. `state_a` and `state_b` are distinct initialized Keccak states.
      unsafe { super::super::x86_64::keccakf_x86_avx512_x2(&mut state_a, &mut state_b) };

      assert_eq!(state_a, expected_a, "state_a mismatch for seed={seed}");
      assert_eq!(state_b, expected_b, "state_b mismatch for seed={seed}");
    }
  }

  /// Verify 2-state interleaved kernel matches two independent single-state runs.
  #[test]
  #[cfg(all(target_arch = "aarch64", not(miri)))]
  fn keccakf1600_x2_matches_single_state() {
    let caps = crate::platform::caps();
    if !caps.has(crate::platform::caps::aarch64::SHA3) {
      return; // SHA3 CE not available on this hardware
    }

    // Use two distinct initial states.
    let mut state_a = state_from_bytes(b"state_a_test_data_for_keccak");
    let mut state_b = state_from_bytes(b"state_b_different_test_input");

    // Single-state reference: permute each independently.
    let mut ref_a = state_a;
    let mut ref_b = state_b;
    super::super::aarch64::keccakf_aarch64_sha3_single(&mut ref_a);
    super::super::aarch64::keccakf_aarch64_sha3_single(&mut ref_b);

    // 2-state interleaved: permute both simultaneously.
    super::super::aarch64::keccakf_aarch64_sha3_x2(&mut state_a, &mut state_b);

    assert_eq!(state_a, ref_a, "x2 state_a mismatch vs single-state");
    assert_eq!(state_b, ref_b, "x2 state_b mismatch vs single-state");
  }

  /// Verify SVE2-SHA3 4-state kernel matches four independent portable runs.
  #[test]
  #[cfg(all(target_arch = "aarch64", target_os = "linux", not(miri)))]
  fn keccakf1600_sve2_sha3_x4_matches_portable() {
    let caps = crate::platform::caps();
    if !caps.has(crate::platform::caps::aarch64::SVE2_SHA3) {
      return; // SVE2-SHA3 not available on this hardware
    }

    let mut state_a = state_from_bytes(b"sve2_sha3_state_a");
    let mut state_b = state_from_bytes(b"sve2_sha3_state_b_different");
    let mut state_c = state_from_bytes(b"sve2_sha3_state_c_third");
    let mut state_d = state_from_bytes(b"sve2_sha3_state_d_fourth");

    let mut expected_a = state_a;
    let mut expected_b = state_b;
    let mut expected_c = state_c;
    let mut expected_d = state_d;
    super::super::keccakf_portable(&mut expected_a);
    super::super::keccakf_portable(&mut expected_b);
    super::super::keccakf_portable(&mut expected_c);
    super::super::keccakf_portable(&mut expected_d);

    if !super::super::aarch64::keccakf_aarch64_sve2_sha3_x4(&mut state_a, &mut state_b, &mut state_c, &mut state_d) {
      return; // Runtime SVE VL is below four u64 lanes; dispatch must fall back.
    }

    assert_eq!(state_a, expected_a, "SVE2-SHA3 x4 state_a mismatch");
    assert_eq!(state_b, expected_b, "SVE2-SHA3 x4 state_b mismatch");
    assert_eq!(state_c, expected_c, "SVE2-SHA3 x4 state_c mismatch");
    assert_eq!(state_d, expected_d, "SVE2-SHA3 x4 state_d mismatch");
  }

  /// Verify batched full-block absorb matches repeated single-block absorb.
  #[test]
  #[cfg(all(target_arch = "aarch64", not(miri)))]
  fn keccakf1600_absorb_blocks_matches_single_absorb() {
    let caps = crate::platform::caps();
    if !caps.has(crate::platform::caps::aarch64::SHA3) {
      return; // SHA3 CE not available on this hardware
    }

    fn assert_rate<const RATE: usize>() {
      let len = RATE.strict_mul(4);
      let mut blocks = Vec::with_capacity(len);
      for i in 0..len {
        blocks.push(((i.strict_mul(37).strict_add(11)) & 0xff) as u8);
      }

      let mut expected = state_from_bytes(b"batch-absorb-reference-state");
      let mut actual = expected;
      let (chunks, rem) = blocks.as_chunks::<RATE>();
      debug_assert!(rem.is_empty());
      for block in chunks {
        super::super::xor_block_into::<RATE>(&mut expected, block);
        super::super::aarch64::keccakf_aarch64_sha3_single(&mut expected);
      }

      super::super::aarch64::keccakf_aarch64_sha3_absorb_blocks::<RATE>(&mut actual, &blocks);

      assert_eq!(actual, expected, "batched absorb mismatch for RATE={RATE}");
    }

    assert_rate::<72>();
    assert_rate::<104>();
    assert_rate::<136>();
    assert_rate::<144>();
    assert_rate::<168>();
  }
}
