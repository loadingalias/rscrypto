#![cfg(feature = "sha2")]

use rscrypto::{Digest, Sha224, Sha256, Sha384, Sha512, Sha512_256};

fn exercise_streaming<D: Digest, R: sha2::Digest>() {
  let data: Vec<u8> = (0u8..=255).cycle().take(4112).map(|i| i.wrapping_mul(37)).collect();
  for offset in [0usize, 1, 15] {
    for len in [
      0, 1, 55, 56, 63, 64, 65, 111, 112, 127, 128, 129, 255, 256, 257, 4095, 4096, 4097,
    ] {
      let message = &data[offset..offset.strict_add(len)];
      let expected = R::digest(message);
      assert_eq!(D::digest(message).as_ref(), expected.as_slice());
      for chunk in [1, 63, 64, 65, 127, 128, 129, 256] {
        let mut original = D::new();
        // Include cloning before initialization as well as after a partial/full block.
        let split = if chunk == 1 { 0 } else { len.min(chunk) };
        original.update(&message[..split]);
        original.update(&[]);
        let mut cloned = original.clone();
        for part in message[split..].chunks(chunk) {
          original.update(part);
        }
        cloned.update(&message[split..]);
        assert_eq!(original.finalize().as_ref(), expected.as_slice());
        assert_eq!(original.finalize(), cloned.finalize());
        assert_eq!(original.finalize().as_ref(), expected.as_slice());
        original.reset();
        assert_eq!(original.finalize().as_ref(), R::digest([]).as_slice());
        original.update(message);
        assert_eq!(original.finalize().as_ref(), expected.as_slice());
      }
    }
  }
}

#[test]
fn sha224_streaming_clone_reset_boundaries() {
  exercise_streaming::<Sha224, sha2::Sha224>();
}

#[test]
fn sha256_streaming_clone_reset_boundaries() {
  exercise_streaming::<Sha256, sha2::Sha256>();
}

#[test]
fn sha384_streaming_clone_reset_boundaries() {
  exercise_streaming::<Sha384, sha2::Sha384>();
}

#[test]
fn sha512_streaming_clone_reset_boundaries() {
  exercise_streaming::<Sha512, sha2::Sha512>();
}

#[test]
fn sha512_256_streaming_clone_reset_boundaries() {
  exercise_streaming::<Sha512_256, sha2::Sha512_256>();
}

#[cfg(feature = "diag")]
#[test]
fn diagnostic_backend_is_independent_of_length() {
  use rscrypto::hashes::introspect::{KernelIntrospect, kernel_for};

  fn check<D: KernelIntrospect>() {
    let expected = kernel_for::<D>(0);
    assert!(!expected.is_empty());
    for len in [1, 63, 64, 65, 255, 256, 257, 4095, 4096, 4097, usize::MAX] {
      assert_eq!(kernel_for::<D>(len), expected);
    }
  }
  check::<Sha224>();
  check::<Sha256>();
  check::<Sha384>();
  check::<Sha512>();
  check::<Sha512_256>();
  assert_eq!(kernel_for::<Sha224>(0), kernel_for::<Sha256>(0));
}
