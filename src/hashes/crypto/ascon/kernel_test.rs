extern crate alloc;

use alloc::vec::Vec;

use super::kernels::{ALL, permute_fn, required_caps};

#[derive(Clone, Debug)]
pub struct KernelResult {
  pub name: &'static str,
  pub state: [u64; 5],
}

fn state_from_bytes(data: &[u8]) -> [u64; 5] {
  let mut buf = [0u8; 40];
  for (dst, src) in buf.iter_mut().zip(data.iter()) {
    *dst = *src;
  }

  let mut out = [0u64; 5];
  let (chunks, rem) = buf.as_chunks::<8>();
  debug_assert!(rem.is_empty());
  for (lane, chunk) in out.iter_mut().zip(chunks.iter()) {
    *lane = u64::from_le_bytes(*chunk);
  }
  out
}

#[must_use]
pub fn run_all_ascon_p12_kernels(data: &[u8]) -> Vec<KernelResult> {
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

pub fn verify_ascon_p12_kernels(data: &[u8]) -> Result<(), &'static str> {
  let results = run_all_ascon_p12_kernels(data);
  let Some(first) = results.first() else {
    return Ok(());
  };
  for r in &results[1..] {
    if r.state != first.state {
      return Err("ascon p12 kernel mismatch");
    }
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::hashes::crypto::ascon::kernels::simd_degree;

  #[test]
  fn run_all_agree() {
    verify_ascon_p12_kernels(b"abc").expect("kernels should agree");
    verify_ascon_p12_kernels(&[0u8; 40]).expect("kernels should agree");
  }

  #[test]
  fn digest_matches_portable_reference() {
    use crate::{hashes::crypto::ascon::kernels::AsconPermute12KernelId, traits::Digest as _};

    let data = b"ascon test input";
    let ours = crate::hashes::crypto::AsconHash256::digest(data);
    let expected = crate::hashes::crypto::AsconHash256::digest_with_kernel(AsconPermute12KernelId::Portable, data);
    assert_eq!(ours, expected);

    let mut ours_xof = [0u8; 64];
    crate::hashes::crypto::AsconXof::hash_into(data, &mut ours_xof);
    let mut exp_xof = [0u8; 64];
    crate::hashes::crypto::AsconXof::hash_into_with_kernel(AsconPermute12KernelId::Portable, data, &mut exp_xof);
    assert_eq!(ours_xof, exp_xof);
  }

  #[test]
  fn digest_many_matches_scalar() {
    let caps = crate::platform::caps();
    let inputs_storage = [
      vec![0xA3; 4096],
      vec![0x5C; 4096],
      vec![0x11; 4096],
      vec![0xF0; 4096],
      vec![0x37; 4096],
    ];
    let inputs: Vec<&[u8]> = inputs_storage.iter().map(Vec::as_slice).collect();

    for &id in ALL {
      if simd_degree(id) == 1 || !caps.has(required_caps(id)) {
        continue;
      }

      let mut batch = [[0u8; 32]; 5];
      crate::hashes::crypto::AsconHash256::digest_many_with_kernel(id, &inputs, &mut batch);

      for (input, actual) in inputs.iter().zip(batch.iter()) {
        let expected = crate::hashes::crypto::AsconHash256::digest_with_kernel(id, input);
        assert_eq!(*actual, expected, "digest_many mismatch for {}", id.as_str());
      }
    }
  }

  #[test]
  fn xof_many_matches_scalar() {
    let caps = crate::platform::caps();
    let out_len = 64usize;
    let inputs_storage = [
      vec![0x42; 4096],
      vec![0x24; 4096],
      vec![0x99; 4096],
      vec![0x18; 4096],
      vec![0x7E; 4096],
    ];
    let inputs: Vec<&[u8]> = inputs_storage.iter().map(Vec::as_slice).collect();

    for &id in ALL {
      if simd_degree(id) == 1 || !caps.has(required_caps(id)) {
        continue;
      }

      let mut batch = vec![0u8; inputs.len() * out_len];
      crate::hashes::crypto::AsconXof::hash_many_into_with_kernel(id, &inputs, out_len, &mut batch);

      for (index, input) in inputs.iter().enumerate() {
        let mut expected = vec![0u8; out_len];
        crate::hashes::crypto::AsconXof::hash_into_with_kernel(id, input, &mut expected);
        let base = index * out_len;
        assert_eq!(
          &batch[base..base + out_len],
          expected.as_slice(),
          "xof_many mismatch for {}",
          id.as_str()
        );
      }
    }
  }
}
