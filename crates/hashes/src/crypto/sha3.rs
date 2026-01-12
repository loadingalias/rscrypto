//! SHA-3 family (FIPS 202): SHA3-256, SHA3-512, SHAKE256.
//!
//! Portable, `no_std`, pure Rust Keccak-f\[1600\] sponge.

use traits::{Digest, Xof};

use super::keccak::{KeccakCore, KeccakXof};

/// SHA3-256.
#[derive(Clone, Default)]
pub struct Sha3_256 {
  core: KeccakCore<136>,
}

/// SHA3-224.
#[derive(Clone, Default)]
pub struct Sha3_224 {
  core: KeccakCore<144>,
}

impl Digest for Sha3_224 {
  const OUTPUT_SIZE: usize = 28;
  type Output = [u8; 28];

  #[inline]
  fn new() -> Self {
    Self::default()
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.core.update(data);
  }

  #[inline]
  fn finalize(&self) -> Self::Output {
    let mut out = [0u8; 28];
    self.core.finalize_into_fixed(0x06, &mut out);
    out
  }

  #[inline]
  fn reset(&mut self) {
    *self = Self::default();
  }
}

impl Digest for Sha3_256 {
  const OUTPUT_SIZE: usize = 32;
  type Output = [u8; 32];

  #[inline]
  fn new() -> Self {
    Self::default()
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.core.update(data);
  }

  #[inline]
  fn finalize(&self) -> Self::Output {
    let mut out = [0u8; 32];
    self.core.finalize_into_fixed(0x06, &mut out);
    out
  }

  #[inline]
  fn reset(&mut self) {
    *self = Self::default();
  }
}

/// SHA3-512.
#[derive(Clone, Default)]
pub struct Sha3_512 {
  core: KeccakCore<72>,
}

/// SHA3-384.
#[derive(Clone, Default)]
pub struct Sha3_384 {
  core: KeccakCore<104>,
}

impl Digest for Sha3_384 {
  const OUTPUT_SIZE: usize = 48;
  type Output = [u8; 48];

  #[inline]
  fn new() -> Self {
    Self::default()
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.core.update(data);
  }

  #[inline]
  fn finalize(&self) -> Self::Output {
    let mut out = [0u8; 48];
    self.core.finalize_into_fixed(0x06, &mut out);
    out
  }

  #[inline]
  fn reset(&mut self) {
    *self = Self::default();
  }
}

impl Digest for Sha3_512 {
  const OUTPUT_SIZE: usize = 64;
  type Output = [u8; 64];

  #[inline]
  fn new() -> Self {
    Self::default()
  }

  #[inline]
  fn update(&mut self, data: &[u8]) {
    self.core.update(data);
  }

  #[inline]
  fn finalize(&self) -> Self::Output {
    let mut out = [0u8; 64];
    self.core.finalize_into_fixed(0x06, &mut out);
    out
  }

  #[inline]
  fn reset(&mut self) {
    *self = Self::default();
  }
}

/// SHAKE256 (XOF).
#[derive(Clone, Default)]
pub struct Shake256 {
  core: KeccakCore<136>,
}

/// SHAKE128 (XOF).
#[derive(Clone, Default)]
pub struct Shake128 {
  core: KeccakCore<168>,
}

impl Shake128 {
  #[inline]
  #[must_use]
  pub fn new() -> Self {
    Self::default()
  }

  #[inline]
  pub fn update(&mut self, data: &[u8]) {
    self.core.update(data);
  }

  #[inline]
  #[must_use]
  pub fn finalize_xof(&self) -> Shake128Xof {
    Shake128Xof {
      inner: self.core.finalize_xof(0x1F),
    }
  }

  #[inline]
  pub fn reset(&mut self) {
    *self = Self::default();
  }

  #[inline]
  pub fn hash_into(data: &[u8], out: &mut [u8]) {
    let mut h = Self::new();
    h.update(data);
    h.core.finalize_xof_into(0x1F, out);
  }
}

#[derive(Clone)]
pub struct Shake128Xof {
  inner: KeccakXof<168>,
}

impl Xof for Shake128Xof {
  #[inline]
  fn squeeze(&mut self, out: &mut [u8]) {
    self.inner.squeeze_into(out);
  }
}

impl Shake256 {
  #[inline]
  #[must_use]
  pub fn new() -> Self {
    Self::default()
  }

  #[inline]
  pub fn update(&mut self, data: &[u8]) {
    self.core.update(data);
  }

  #[inline]
  #[must_use]
  pub fn finalize_xof(&self) -> Shake256Xof {
    Shake256Xof {
      inner: self.core.finalize_xof(0x1F),
    }
  }

  #[inline]
  pub fn reset(&mut self) {
    *self = Self::default();
  }

  #[inline]
  pub fn hash_into(data: &[u8], out: &mut [u8]) {
    let mut h = Self::new();
    h.update(data);
    h.core.finalize_xof_into(0x1F, out);
  }
}

#[derive(Clone)]
pub struct Shake256Xof {
  inner: KeccakXof<136>,
}

impl Xof for Shake256Xof {
  #[inline]
  fn squeeze(&mut self, out: &mut [u8]) {
    self.inner.squeeze_into(out);
  }
}

#[cfg(test)]
mod tests {
  use traits::Digest;

  use super::{Sha3_256, Sha3_512, Shake256};

  fn hex(bytes: &[u8]) -> alloc::string::String {
    use alloc::string::String;
    use core::fmt::Write;
    let mut s = String::new();
    for &b in bytes {
      write!(&mut s, "{:02x}", b).unwrap();
    }
    s
  }

  extern crate alloc;

  #[test]
  fn sha3_256_vectors() {
    assert_eq!(
      hex(&Sha3_256::digest(b"")),
      "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
    );
    assert_eq!(
      hex(&Sha3_256::digest(b"abc")),
      "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"
    );
  }

  #[test]
  fn sha3_512_vectors() {
    assert_eq!(
      hex(&Sha3_512::digest(b"")),
      "a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc1475c80a615b2123af1f5f94c11e3e9402c3ac558f500199d95b6d3e301758586281dcd26"
    );
    assert_eq!(
      hex(&Sha3_512::digest(b"abc")),
      "b751850b1a57168a5693cd924b6b096e08f621827444f70d884f5d0240d2712e10e116e9192af3c91a7ec57647e3934057340b4cf408d5a56592f8274eec53f0"
    );
  }

  #[test]
  fn shake256_vectors() {
    // NIST FIPS 202 test vector: SHAKE256(M="", 512 bits)
    let mut out = [0u8; 64];
    Shake256::hash_into(b"", &mut out);
    assert_eq!(
      hex(&out),
      "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762fd75dc4ddd8c0f200cb05019d67b592f6fc821c49479ab48640292eacb3b7c4be"
    );
  }
}
