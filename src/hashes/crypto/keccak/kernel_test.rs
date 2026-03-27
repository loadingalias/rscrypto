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

  /// Verify 2-state interleaved kernel matches two independent single-state runs.
  #[test]
  #[cfg(target_arch = "aarch64")]
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
    super::super::aarch64::keccakf_aarch64_sha3(&mut ref_a);
    super::super::aarch64::keccakf_aarch64_sha3(&mut ref_b);

    // 2-state interleaved: permute both simultaneously.
    super::super::aarch64::keccakf_aarch64_sha3_x2(&mut state_a, &mut state_b);

    assert_eq!(state_a, ref_a, "x2 state_a mismatch vs single-state");
    assert_eq!(state_b, ref_b, "x2 state_b mismatch vs single-state");
  }
}
