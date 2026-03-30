//! SHA-3 family (FIPS 202): SHA3-256, SHA3-512, SHAKE256.
//!
//! Portable, `no_std`, pure Rust Keccak-f\[1600\] sponge.

use super::keccak::{KeccakCore, KeccakXof};
use crate::traits::{Digest, Xof};

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

impl core::fmt::Debug for Sha3_256 {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Sha3_256").finish_non_exhaustive()
  }
}

impl core::fmt::Debug for Sha3_224 {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Sha3_224").finish_non_exhaustive()
  }
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
  fn digest(data: &[u8]) -> Self::Output {
    super::keccak::oneshot_fixed::<144, 28>(0x06, data)
  }

  #[inline]
  fn reset(&mut self) {
    *self = Self::default();
  }
}

impl Sha3_224 {}

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
  fn digest(data: &[u8]) -> Self::Output {
    super::keccak::oneshot_fixed::<136, 32>(0x06, data)
  }

  #[inline]
  fn reset(&mut self) {
    *self = Self::default();
  }
}

impl Sha3_256 {
  /// Hash two independent messages in parallel, returning both 32-byte digests.
  ///
  /// On aarch64 with SHA3 Crypto Extensions, this achieves ~2× the aggregate
  /// throughput of two sequential [`digest`](Digest::digest) calls by using
  /// 2-state NEON interleaving.
  #[inline]
  #[must_use]
  pub fn digest_pair(a: &[u8], b: &[u8]) -> ([u8; 32], [u8; 32]) {
    super::keccak::oneshot_pair::<136, 32>(0x06, a, b)
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

impl core::fmt::Debug for Sha3_512 {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Sha3_512").finish_non_exhaustive()
  }
}

impl core::fmt::Debug for Sha3_384 {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Sha3_384").finish_non_exhaustive()
  }
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
  fn digest(data: &[u8]) -> Self::Output {
    super::keccak::oneshot_fixed::<104, 48>(0x06, data)
  }

  #[inline]
  fn reset(&mut self) {
    *self = Self::default();
  }
}

impl Sha3_384 {}

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
  fn digest(data: &[u8]) -> Self::Output {
    super::keccak::oneshot_fixed::<72, 64>(0x06, data)
  }

  #[inline]
  fn reset(&mut self) {
    *self = Self::default();
  }
}

impl Sha3_512 {}

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

impl core::fmt::Debug for Shake256 {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Shake256").finish_non_exhaustive()
  }
}

impl core::fmt::Debug for Shake128 {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Shake128").finish_non_exhaustive()
  }
}

impl Shake128 {
  #[inline]
  #[must_use]
  pub fn new() -> Self {
    Self::default()
  }

  #[inline]
  #[must_use]
  pub fn xof(data: &[u8]) -> Shake128Xof {
    let mut h = Self::new();
    h.update(data);
    h.finalize_xof()
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
  /// Convenience one-shot XOF output for callers that only need bytes.
  ///
  /// The canonical one-shot API is [`Self::xof`]. Use [`Self::new`],
  /// [`Self::update`], and [`Self::finalize_xof`] for streaming.
  pub fn hash_into(data: &[u8], out: &mut [u8]) {
    Self::xof(data).squeeze(out);
  }
}

#[derive(Clone)]
pub struct Shake128Xof {
  inner: KeccakXof<168>,
}

impl core::fmt::Debug for Shake128Xof {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Shake128Xof").finish_non_exhaustive()
  }
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
  #[must_use]
  pub fn xof(data: &[u8]) -> Shake256Xof {
    let mut h = Self::new();
    h.update(data);
    h.finalize_xof()
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
  /// Convenience one-shot XOF output for callers that only need bytes.
  ///
  /// The canonical one-shot API is [`Self::xof`]. Use [`Self::new`],
  /// [`Self::update`], and [`Self::finalize_xof`] for streaming.
  pub fn hash_into(data: &[u8], out: &mut [u8]) {
    Self::xof(data).squeeze(out);
  }
}

#[derive(Clone)]
pub struct Shake256Xof {
  inner: KeccakXof<136>,
}

impl core::fmt::Debug for Shake256Xof {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Shake256Xof").finish_non_exhaustive()
  }
}

impl Xof for Shake256Xof {
  #[inline]
  fn squeeze(&mut self, out: &mut [u8]) {
    self.inner.squeeze_into(out);
  }
}

#[cfg(test)]
mod tests {
  use super::{Sha3_256, Sha3_512, Shake128, Shake256};
  use crate::traits::{Digest, Xof};

  fn hex(bytes: &[u8]) -> alloc::string::String {
    use alloc::string::String;
    use core::fmt::Write;
    let mut s = String::new();
    for &b in bytes {
      write!(&mut s, "{:02x}", b).unwrap();
    }
    s
  }

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
    Shake256::xof(b"").squeeze(&mut out);
    assert_eq!(
      hex(&out),
      "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762fd75dc4ddd8c0f200cb05019d67b592f6fc821c49479ab48640292eacb3b7c4be"
    );
  }

  #[test]
  fn shake_xof_matches_finalize_xof() {
    let data = b"hello world";

    let mut via_finalize = Shake128::new();
    via_finalize.update(data);
    let mut via_finalize = via_finalize.finalize_xof();

    let mut via_oneshot = Shake128::xof(data);

    let mut finalize_out = [0u8; 96];
    let mut oneshot_out = [0u8; 96];
    via_finalize.squeeze(&mut finalize_out);
    via_oneshot.squeeze(&mut oneshot_out);

    assert_eq!(finalize_out, oneshot_out);
  }

  /// Test inputs of length `RATE - 1` for each SHA-3 variant.
  ///
  /// At `len == RATE - 1`, the domain separator byte and the pad10*1 `0x80`
  /// byte target the same lane (and the same byte within that lane). This
  /// exercises the most subtle edge case in the direct-XOR padding path.
  #[test]
  fn sha3_padding_boundary_rate_minus_one() {
    use super::{Sha3_224, Sha3_384, Sha3_512};

    // SHA3-256: RATE=136, so test len=135
    let input = &[0xAB_u8; 135];
    let oneshot = Sha3_256::digest(input);
    let mut streaming = Sha3_256::new();
    streaming.update(input);
    assert_eq!(oneshot, streaming.finalize(), "SHA3-256 RATE-1 mismatch");

    // SHA3-512: RATE=72, so test len=71
    let input = &[0xCD_u8; 71];
    let oneshot = Sha3_512::digest(input);
    let mut streaming = Sha3_512::new();
    streaming.update(input);
    assert_eq!(oneshot, streaming.finalize(), "SHA3-512 RATE-1 mismatch");

    // SHA3-224: RATE=144, so test len=143
    let input = &[0xEF_u8; 143];
    let oneshot = Sha3_224::digest(input);
    let mut streaming = Sha3_224::new();
    streaming.update(input);
    assert_eq!(oneshot, streaming.finalize(), "SHA3-224 RATE-1 mismatch");

    // SHA3-384: RATE=104, so test len=103
    let input = &[0x12_u8; 103];
    let oneshot = Sha3_384::digest(input);
    let mut streaming = Sha3_384::new();
    streaming.update(input);
    assert_eq!(oneshot, streaming.finalize(), "SHA3-384 RATE-1 mismatch");

    // Also test exact-rate lengths (no remainder — ds byte starts a new block)
    let input = &[0x34_u8; 136];
    let oneshot = Sha3_256::digest(input);
    let mut streaming = Sha3_256::new();
    streaming.update(input);
    assert_eq!(oneshot, streaming.finalize(), "SHA3-256 exact-rate mismatch");
  }

  #[test]
  fn sha3_256_digest_pair_matches_sequential() {
    let msg_a = b"The quick brown fox jumps over the lazy dog";
    let msg_b = b"";
    let msg_c = &[0xABu8; 4096];

    let ref_a = Sha3_256::digest(msg_a);
    let ref_b = Sha3_256::digest(msg_b);
    let ref_c = Sha3_256::digest(msg_c);

    // Equal-length pair
    let (out_a, out_b) = Sha3_256::digest_pair(msg_a, msg_b);
    assert_eq!(out_a, ref_a, "digest_pair output A mismatch");
    assert_eq!(out_b, ref_b, "digest_pair output B mismatch");

    // Mismatched lengths (long vs short — exercises the fallback path)
    let (out_c, out_b2) = Sha3_256::digest_pair(msg_c, msg_b);
    assert_eq!(out_c, ref_c, "digest_pair long output mismatch");
    assert_eq!(out_b2, ref_b, "digest_pair short output mismatch");

    // Same message both sides
    let (out_c1, out_c2) = Sha3_256::digest_pair(msg_c, msg_c);
    assert_eq!(out_c1, ref_c, "digest_pair same-msg output 1 mismatch");
    assert_eq!(out_c2, ref_c, "digest_pair same-msg output 2 mismatch");
  }
}
