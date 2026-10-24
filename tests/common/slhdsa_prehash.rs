//! Independent message hashing for the fips205 HashSLH-DSA oracle.

use sha2::Digest as _;
use shake::{ExtendableOutput as _, Update as _, XofReader as _};

pub(super) fn sha256(message: &[u8]) -> ([u8; 32], &'static [u8]) {
  (sha2::Sha256::digest(message).into(), &fips205::pre_hash::SHA2_256)
}

pub(super) fn sha512(message: &[u8]) -> ([u8; 64], &'static [u8]) {
  (sha2::Sha512::digest(message).into(), &fips205::pre_hash::SHA2_512)
}

pub(super) fn shake128(message: &[u8]) -> ([u8; 32], &'static [u8]) {
  let mut state = shake::Shake128::default();
  state.update(message);
  let mut digest = [0; 32];
  state.finalize_xof().read(&mut digest);
  (digest, &fips205::pre_hash::SHAKE_128)
}

pub(super) fn shake256(message: &[u8]) -> ([u8; 64], &'static [u8]) {
  let mut state = shake::Shake256::default();
  state.update(message);
  let mut digest = [0; 64];
  state.finalize_xof().read(&mut digest);
  (digest, &fips205::pre_hash::SHAKE_256)
}
