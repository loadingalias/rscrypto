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
  let caps = platform::caps();
  let mut out = Vec::new();
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

  #[test]
  fn run_all_agree() {
    verify_ascon_p12_kernels(b"abc").expect("kernels should agree");
    verify_ascon_p12_kernels(&[0u8; 40]).expect("kernels should agree");
  }

  #[test]
  fn digest_matches_oracle() {
    use traits::Digest as _;

    let data = b"ascon test input";
    let ours = crate::crypto::AsconHash256::digest(data);

    use ascon_hash256::Digest as _;
    let expected = ascon_hash256::AsconHash256::digest(data);
    let mut exp = [0u8; 32];
    exp.copy_from_slice(&expected);
    assert_eq!(ours, exp);

    let mut ours_xof = [0u8; 64];
    crate::crypto::AsconXof128::hash_into(data, &mut ours_xof);

    use ascon_hash256::digest::{ExtendableOutput, Update, XofReader};
    let mut hasher = ascon_hash256::AsconXof128::default();
    hasher.update(data);
    let mut reader = hasher.finalize_xof();
    let mut exp_xof = [0u8; 64];
    reader.read(&mut exp_xof);
    assert_eq!(ours_xof, exp_xof);
  }
}
