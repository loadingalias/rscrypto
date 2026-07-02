//! SHA-3 family (FIPS 202): SHA3-256, SHA3-512, SHAKE256.
//!
//! Portable, `no_std`, pure Rust Keccak-f\[1600\] sponge.

#[cfg(all(test, feature = "ml-kem"))]
use super::keccak::xof_quad;
#[cfg(all(
  feature = "ml-kem",
  any(
    test,
    feature = "diag",
    not(all(
      target_arch = "aarch64",
      target_endian = "little",
      not(miri),
      not(feature = "portable-only")
    ))
  )
))]
use super::keccak::xof_seeded_32_2_triple as keccak_xof_seeded_32_2_triple;
use super::keccak::{PublicKeccakCore, PublicKeccakXof};
#[cfg(all(
  feature = "ml-kem",
  target_arch = "aarch64",
  target_endian = "little",
  not(miri),
  not(feature = "portable-only")
))]
use super::keccak::{PublicKeccakTripleXof, xof_seeded_32_2_triple_cursor as keccak_xof_seeded_32_2_triple_cursor};
#[cfg(feature = "ml-kem")]
use super::keccak::{
  xof_seeded_32_1 as keccak_xof_seeded_32_1, xof_seeded_32_1_pair as keccak_xof_seeded_32_1_pair,
  xof_seeded_32_1_quad as keccak_xof_seeded_32_1_quad, xof_seeded_32_2 as keccak_xof_seeded_32_2,
  xof_seeded_32_2_pair as keccak_xof_seeded_32_2_pair, xof_seeded_32_2_quad as keccak_xof_seeded_32_2_quad,
};
use crate::traits::{Digest, Xof};

/// SHA3-256 digest state.
///
/// Standardized in FIPS 202.
///
/// # Examples
///
/// ```
/// use rscrypto::{Digest, Sha3_256};
///
/// let mut hasher = Sha3_256::new();
/// hasher.update(b"abc");
///
/// assert_eq!(hasher.finalize(), Sha3_256::digest(b"abc"));
/// ```
#[derive(Clone, Default)]
pub struct Sha3_256 {
  core: PublicKeccakCore<136>,
}

/// SHA3-224 digest state.
///
/// Standardized in FIPS 202.
///
/// # Examples
///
/// ```
/// use rscrypto::{Digest, Sha3_224};
///
/// let mut hasher = Sha3_224::new();
/// hasher.update(b"abc");
///
/// assert_eq!(hasher.finalize(), Sha3_224::digest(b"abc"));
/// ```
#[derive(Clone, Default)]
pub struct Sha3_224 {
  core: PublicKeccakCore<144>,
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

impl Sha3_224 {
  /// Hash two independent messages in parallel, returning both 28-byte digests.
  ///
  /// On aarch64 with SHA3 Crypto Extensions, this achieves ~2× the aggregate
  /// throughput of two sequential [`digest`](Digest::digest) calls by using
  /// 2-state NEON interleaving.
  #[inline]
  #[must_use]
  pub fn digest_pair(a: &[u8], b: &[u8]) -> ([u8; 28], [u8; 28]) {
    super::keccak::oneshot_pair::<144, 28>(0x06, a, b)
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

/// SHA3-512 digest state.
///
/// Standardized in FIPS 202.
///
/// # Examples
///
/// ```
/// use rscrypto::{Digest, Sha3_512};
///
/// let mut hasher = Sha3_512::new();
/// hasher.update(b"abc");
///
/// assert_eq!(hasher.finalize(), Sha3_512::digest(b"abc"));
/// ```
#[derive(Clone, Default)]
pub struct Sha3_512 {
  core: PublicKeccakCore<72>,
}

/// SHA3-384 digest state.
///
/// Standardized in FIPS 202.
///
/// # Examples
///
/// ```
/// use rscrypto::{Digest, Sha3_384};
///
/// let mut hasher = Sha3_384::new();
/// hasher.update(b"abc");
///
/// assert_eq!(hasher.finalize(), Sha3_384::digest(b"abc"));
/// ```
#[derive(Clone, Default)]
pub struct Sha3_384 {
  core: PublicKeccakCore<104>,
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

impl Sha3_384 {
  /// Hash two independent messages in parallel, returning both 48-byte digests.
  ///
  /// On aarch64 with SHA3 Crypto Extensions, this achieves ~2× the aggregate
  /// throughput of two sequential [`digest`](Digest::digest) calls by using
  /// 2-state NEON interleaving.
  #[inline]
  #[must_use]
  pub fn digest_pair(a: &[u8], b: &[u8]) -> ([u8; 48], [u8; 48]) {
    super::keccak::oneshot_pair::<104, 48>(0x06, a, b)
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
  fn digest(data: &[u8]) -> Self::Output {
    super::keccak::oneshot_fixed::<72, 64>(0x06, data)
  }

  #[inline]
  fn reset(&mut self) {
    *self = Self::default();
  }
}

impl Sha3_512 {
  /// Hash two independent messages in parallel, returning both 64-byte digests.
  ///
  /// On aarch64 with SHA3 Crypto Extensions, this achieves ~2× the aggregate
  /// throughput of two sequential [`digest`](Digest::digest) calls by using
  /// 2-state NEON interleaving.
  #[inline]
  #[must_use]
  pub fn digest_pair(a: &[u8], b: &[u8]) -> ([u8; 64], [u8; 64]) {
    super::keccak::oneshot_pair::<72, 64>(0x06, a, b)
  }
}

impl_std_io_write_for_digest!(Sha3_224);
impl_std_io_write_for_digest!(Sha3_256);
impl_std_io_write_for_digest!(Sha3_384);
impl_std_io_write_for_digest!(Sha3_512);

/// SHAKE256 extendable-output state.
///
/// Standardized in FIPS 202.
///
/// # Examples
///
/// ```
/// use rscrypto::{Shake256, Xof};
///
/// let mut reader = Shake256::xof(b"abc");
/// let mut out = [0u8; 32];
/// reader.squeeze(&mut out);
///
/// assert_ne!(out, [0u8; 32]);
/// ```
#[derive(Clone, Default)]
pub struct Shake256 {
  core: PublicKeccakCore<136>,
}

/// SHAKE128 extendable-output state.
///
/// Standardized in FIPS 202.
///
/// # Examples
///
/// ```
/// use rscrypto::{Shake128, Xof};
///
/// let mut reader = Shake128::xof(b"abc");
/// let mut out = [0u8; 32];
/// reader.squeeze(&mut out);
///
/// assert_ne!(out, [0u8; 32]);
/// ```
#[derive(Clone, Default)]
pub struct Shake128 {
  core: PublicKeccakCore<168>,
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
  pub fn xof(data: &[u8]) -> Shake128XofReader {
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
  pub fn finalize_xof(self) -> Shake128XofReader {
    Shake128XofReader {
      inner: self.core.into_xof(0x1F),
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

  #[inline]
  #[cfg(feature = "ml-kem")]
  pub(crate) fn xof_seeded_32_2(seed: &[u8; 32], x: u8, y: u8) -> Shake128XofReader {
    Shake128XofReader {
      inner: keccak_xof_seeded_32_2::<168>(0x1F, seed, x, y),
    }
  }

  #[inline]
  #[cfg(feature = "ml-kem")]
  pub(crate) fn xof_seeded_32_2_pair(
    seed: &[u8; 32],
    a: (u8, u8),
    b: (u8, u8),
  ) -> (Shake128XofReader, Shake128XofReader) {
    let (a, b) = keccak_xof_seeded_32_2_pair::<168>(0x1F, seed, a, b);
    (Shake128XofReader { inner: a }, Shake128XofReader { inner: b })
  }

  #[inline]
  #[cfg(all(
    feature = "ml-kem",
    any(
      test,
      feature = "diag",
      not(all(
        target_arch = "aarch64",
        target_endian = "little",
        not(miri),
        not(feature = "portable-only")
      ))
    )
  ))]
  pub(crate) fn xof_seeded_32_2_triple(
    seed: &[u8; 32],
    a: (u8, u8),
    b: (u8, u8),
    c: (u8, u8),
  ) -> (Shake128XofReader, Shake128XofReader, Shake128XofReader) {
    let (a, b, c) = keccak_xof_seeded_32_2_triple::<168>(0x1F, seed, a, b, c);
    (
      Shake128XofReader { inner: a },
      Shake128XofReader { inner: b },
      Shake128XofReader { inner: c },
    )
  }

  #[inline]
  #[cfg(all(
    feature = "ml-kem",
    target_arch = "aarch64",
    target_endian = "little",
    not(miri),
    not(feature = "portable-only")
  ))]
  pub(crate) fn xof_seeded_32_2_triple_cursor(
    seed: &[u8; 32],
    a: (u8, u8),
    b: (u8, u8),
    c: (u8, u8),
  ) -> Shake128TripleXofReader {
    Shake128TripleXofReader {
      inner: keccak_xof_seeded_32_2_triple_cursor::<168>(0x1F, seed, a, b, c),
    }
  }

  #[inline]
  #[cfg(feature = "ml-kem")]
  pub(crate) fn xof_seeded_32_2_quad(
    seed: &[u8; 32],
    a: (u8, u8),
    b: (u8, u8),
    c: (u8, u8),
    d: (u8, u8),
  ) -> (
    Shake128XofReader,
    Shake128XofReader,
    Shake128XofReader,
    Shake128XofReader,
  ) {
    let (a, b, c, d) = keccak_xof_seeded_32_2_quad::<168>(0x1F, seed, a, b, c, d);
    (
      Shake128XofReader { inner: a },
      Shake128XofReader { inner: b },
      Shake128XofReader { inner: c },
      Shake128XofReader { inner: d },
    )
  }

  #[inline]
  #[cfg(all(test, feature = "ml-kem"))]
  pub(crate) fn xof_quad(
    data_a: &[u8],
    data_b: &[u8],
    data_c: &[u8],
    data_d: &[u8],
  ) -> (
    Shake128XofReader,
    Shake128XofReader,
    Shake128XofReader,
    Shake128XofReader,
  ) {
    let (a, b, c, d) = xof_quad::<168>(0x1F, data_a, data_b, data_c, data_d);
    (
      Shake128XofReader { inner: a },
      Shake128XofReader { inner: b },
      Shake128XofReader { inner: c },
      Shake128XofReader { inner: d },
    )
  }
}

#[derive(Clone)]
pub struct Shake128XofReader {
  inner: PublicKeccakXof<168>,
}

#[cfg(all(
  feature = "ml-kem",
  target_arch = "aarch64",
  target_endian = "little",
  not(miri),
  not(feature = "portable-only")
))]
pub(crate) struct Shake128TripleXofReader {
  inner: PublicKeccakTripleXof<168>,
}

impl core::fmt::Debug for Shake128XofReader {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Shake128XofReader").finish_non_exhaustive()
  }
}

impl Xof for Shake128XofReader {
  #[inline]
  fn squeeze(&mut self, out: &mut [u8]) {
    self.inner.squeeze_into(out);
  }
}

impl Shake128XofReader {
  #[inline]
  #[cfg(feature = "ml-kem")]
  pub(crate) fn squeeze_pair(a: &mut Self, b: &mut Self, out_a: &mut [u8], out_b: &mut [u8]) {
    PublicKeccakXof::<168>::squeeze_pair_into(&mut a.inner, &mut b.inner, out_a, out_b);
  }

  #[inline]
  #[cfg(all(
    feature = "ml-kem",
    not(all(
      target_arch = "aarch64",
      target_endian = "little",
      not(miri),
      not(feature = "portable-only")
    ))
  ))]
  pub(crate) fn squeeze_triple(
    a: &mut Self,
    b: &mut Self,
    c: &mut Self,
    out_a: &mut [u8],
    out_b: &mut [u8],
    out_c: &mut [u8],
  ) {
    PublicKeccakXof::<168>::squeeze_triple_into(&mut a.inner, &mut b.inner, &mut c.inner, out_a, out_b, out_c);
  }

  #[inline]
  #[cfg(all(
    feature = "ml-kem",
    any(
      test,
      not(all(
        target_arch = "aarch64",
        target_endian = "little",
        not(miri),
        not(feature = "portable-only")
      ))
    )
  ))]
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn squeeze_quad(
    a: &mut Self,
    b: &mut Self,
    c: &mut Self,
    d: &mut Self,
    out_a: &mut [u8],
    out_b: &mut [u8],
    out_c: &mut [u8],
    out_d: &mut [u8],
  ) {
    PublicKeccakXof::<168>::squeeze_quad_into(
      &mut a.inner,
      &mut b.inner,
      &mut c.inner,
      &mut d.inner,
      out_a,
      out_b,
      out_c,
      out_d,
    );
  }

  #[inline]
  #[cfg(all(
    feature = "ml-kem",
    target_arch = "aarch64",
    target_endian = "little",
    not(miri),
    not(feature = "portable-only")
  ))]
  pub(crate) fn with_quad_rate_block(
    a: &mut Self,
    b: &mut Self,
    c: &mut Self,
    d: &mut Self,
    f: impl FnOnce(&[u64; 25], &[u64; 25], &[u64; 25], &[u64; 25]),
  ) {
    PublicKeccakXof::<168>::with_quad_rate_block(&mut a.inner, &mut b.inner, &mut c.inner, &mut d.inner, f);
  }

  #[inline]
  #[cfg(all(
    feature = "ml-kem",
    target_arch = "aarch64",
    target_endian = "little",
    any(test, feature = "diag"),
    not(miri),
    not(feature = "portable-only")
  ))]
  pub(crate) fn with_triple_rate_block(
    a: &mut Self,
    b: &mut Self,
    c: &mut Self,
    f: impl FnOnce(&[u64; 25], &[u64; 25], &[u64; 25]),
  ) {
    PublicKeccakXof::<168>::with_triple_rate_block(&mut a.inner, &mut b.inner, &mut c.inner, f);
  }
}

#[cfg(all(
  feature = "ml-kem",
  target_arch = "aarch64",
  target_endian = "little",
  not(miri),
  not(feature = "portable-only")
))]
impl Shake128TripleXofReader {
  #[inline(always)]
  pub(crate) fn with_triple_rate_block(&mut self, f: impl FnOnce([&[u64; 25]; 3])) {
    self.inner.with_triple_rate_block(f);
  }

  #[inline(always)]
  pub(crate) fn with_lane_rate_block(&mut self, lane: usize, f: impl FnOnce(&[u64; 25])) {
    self.inner.with_lane_rate_block(lane, f);
  }
}

impl_xof_read!(Shake128XofReader);

impl Shake256 {
  #[inline]
  #[must_use]
  pub fn new() -> Self {
    Self::default()
  }

  #[inline]
  #[must_use]
  pub fn xof(data: &[u8]) -> Shake256XofReader {
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
  pub fn finalize_xof(self) -> Shake256XofReader {
    Shake256XofReader {
      inner: self.core.into_xof(0x1F),
    }
  }

  #[inline]
  pub fn reset(&mut self) {
    *self = Self::default();
  }

  #[inline]
  #[cfg(feature = "ml-kem")]
  pub(crate) fn xof_seeded_32_1(seed: &[u8; 32], x: u8) -> Shake256XofReader {
    Shake256XofReader {
      inner: keccak_xof_seeded_32_1::<136>(0x1F, seed, x),
    }
  }

  #[inline]
  #[cfg(feature = "ml-kem")]
  pub(crate) fn xof_seeded_32_1_pair(seed: &[u8; 32], a: u8, b: u8) -> (Shake256XofReader, Shake256XofReader) {
    let (a, b) = keccak_xof_seeded_32_1_pair::<136>(0x1F, seed, a, b);
    (Shake256XofReader { inner: a }, Shake256XofReader { inner: b })
  }

  #[inline]
  #[cfg(feature = "ml-kem")]
  pub(crate) fn xof_seeded_32_1_quad(
    seed: &[u8; 32],
    a: u8,
    b: u8,
    c: u8,
    d: u8,
  ) -> (
    Shake256XofReader,
    Shake256XofReader,
    Shake256XofReader,
    Shake256XofReader,
  ) {
    let (a, b, c, d) = keccak_xof_seeded_32_1_quad::<136>(0x1F, seed, a, b, c, d);
    (
      Shake256XofReader { inner: a },
      Shake256XofReader { inner: b },
      Shake256XofReader { inner: c },
      Shake256XofReader { inner: d },
    )
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
pub struct Shake256XofReader {
  inner: PublicKeccakXof<136>,
}

impl core::fmt::Debug for Shake256XofReader {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Shake256XofReader").finish_non_exhaustive()
  }
}

impl Xof for Shake256XofReader {
  #[inline]
  fn squeeze(&mut self, out: &mut [u8]) {
    self.inner.squeeze_into(out);
  }
}

impl Shake256XofReader {
  #[inline]
  #[cfg(feature = "ml-kem")]
  pub(crate) fn squeeze_pair(a: &mut Self, b: &mut Self, out_a: &mut [u8], out_b: &mut [u8]) {
    PublicKeccakXof::<136>::squeeze_pair_into(&mut a.inner, &mut b.inner, out_a, out_b);
  }

  #[inline]
  #[cfg(feature = "ml-kem")]
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn squeeze_quad(
    a: &mut Self,
    b: &mut Self,
    c: &mut Self,
    d: &mut Self,
    out_a: &mut [u8],
    out_b: &mut [u8],
    out_c: &mut [u8],
    out_d: &mut [u8],
  ) {
    PublicKeccakXof::<136>::squeeze_quad_into(
      &mut a.inner,
      &mut b.inner,
      &mut c.inner,
      &mut d.inner,
      out_a,
      out_b,
      out_c,
      out_d,
    );
  }
}

impl_xof_read!(Shake256XofReader);

#[cfg(test)]
mod tests {
  use super::{Sha3_224, Sha3_256, Sha3_384, Sha3_512, Shake128, Shake128XofReader, Shake256};
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

  #[cfg(feature = "ml-kem")]
  #[test]
  fn shake128_xof_quad_matches_sequential() {
    let msg_a = b"ml-kem matrix lane 0";
    let msg_b = &[0x42u8; 34];
    let msg_c = &[0x77u8; 255];
    let msg_d = &[0xa5u8; 409];

    let (mut quad_a, mut quad_b, mut quad_c, mut quad_d) = Shake128::xof_quad(msg_a, msg_b, msg_c, msg_d);
    let mut actual_a = [0u8; 320];
    let mut actual_b = [0u8; 320];
    let mut actual_c = [0u8; 320];
    let mut actual_d = [0u8; 320];
    Shake128XofReader::squeeze_quad(
      &mut quad_a,
      &mut quad_b,
      &mut quad_c,
      &mut quad_d,
      &mut actual_a[..168],
      &mut actual_b[..168],
      &mut actual_c[..168],
      &mut actual_d[..168],
    );
    Shake128XofReader::squeeze_quad(
      &mut quad_a,
      &mut quad_b,
      &mut quad_c,
      &mut quad_d,
      &mut actual_a[168..],
      &mut actual_b[168..],
      &mut actual_c[168..],
      &mut actual_d[168..],
    );

    let mut expected_a = [0u8; 320];
    let mut expected_b = [0u8; 320];
    let mut expected_c = [0u8; 320];
    let mut expected_d = [0u8; 320];
    Shake128::xof(msg_a).squeeze(&mut expected_a);
    Shake128::xof(msg_b).squeeze(&mut expected_b);
    Shake128::xof(msg_c).squeeze(&mut expected_c);
    Shake128::xof(msg_d).squeeze(&mut expected_d);

    assert_eq!(actual_a, expected_a, "lane 0");
    assert_eq!(actual_b, expected_b, "lane 1");
    assert_eq!(actual_c, expected_c, "lane 2");
    assert_eq!(actual_d, expected_d, "lane 3");
  }

  #[cfg(all(
    feature = "ml-kem",
    target_arch = "aarch64",
    target_endian = "little",
    not(miri),
    not(feature = "portable-only")
  ))]
  fn copy_shake128_rate_block(state: &[u64; 25]) -> [u8; 168] {
    let mut out = [0u8; 168];
    let mut offset = 0usize;
    for &word in state.iter().take(168 / 8) {
      out[offset..offset + 8].copy_from_slice(&word.to_le_bytes());
      offset += 8;
    }
    out
  }

  #[cfg(all(
    feature = "ml-kem",
    target_arch = "aarch64",
    target_endian = "little",
    not(miri),
    not(feature = "portable-only")
  ))]
  #[test]
  fn shake128_triple_cursor_matches_xof_readers() {
    let mut seed = [0u8; 32];
    for (i, byte) in seed.iter_mut().enumerate() {
      *byte = (i.strict_mul(37).strict_add(11)) as u8;
    }

    let lanes = [(0, 0), (1, 2), (2, 1)];
    let (mut reader0, mut reader1, mut reader2) = Shake128::xof_seeded_32_2_triple(&seed, lanes[0], lanes[1], lanes[2]);
    let mut cursor = Shake128::xof_seeded_32_2_triple_cursor(&seed, lanes[0], lanes[1], lanes[2]);

    for block in 0..4 {
      let mut actual = [[0u8; 168]; 3];
      cursor.with_triple_rate_block(|states| {
        actual[0] = copy_shake128_rate_block(states[0]);
        actual[1] = copy_shake128_rate_block(states[1]);
        actual[2] = copy_shake128_rate_block(states[2]);
      });

      let mut expected = [[0u8; 168]; 3];
      reader0.squeeze(&mut expected[0]);
      reader1.squeeze(&mut expected[1]);
      reader2.squeeze(&mut expected[2]);

      assert_eq!(actual, expected, "triple block {block}");
    }

    for lane in 0..3 {
      let mut actual = [0u8; 168];
      cursor.with_lane_rate_block(lane, |state| {
        actual = copy_shake128_rate_block(state);
      });

      let mut expected = [0u8; 168];
      match lane {
        0 => reader0.squeeze(&mut expected),
        1 => reader1.squeeze(&mut expected),
        2 => reader2.squeeze(&mut expected),
        _ => unreachable!(),
      }

      assert_eq!(actual, expected, "lane tail {lane}");
    }
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

  #[test]
  fn sha3_224_digest_pair_matches_sequential() {
    let msg_a = b"The quick brown fox jumps over the lazy dog";
    let msg_b = b"";
    let msg_c = &[0xABu8; 4096];

    let ref_a = Sha3_224::digest(msg_a);
    let ref_b = Sha3_224::digest(msg_b);
    let ref_c = Sha3_224::digest(msg_c);

    let (out_a, out_b) = Sha3_224::digest_pair(msg_a, msg_b);
    assert_eq!(out_a, ref_a);
    assert_eq!(out_b, ref_b);

    let (out_c, out_b2) = Sha3_224::digest_pair(msg_c, msg_b);
    assert_eq!(out_c, ref_c);
    assert_eq!(out_b2, ref_b);

    let (out_c1, out_c2) = Sha3_224::digest_pair(msg_c, msg_c);
    assert_eq!(out_c1, ref_c);
    assert_eq!(out_c2, ref_c);
  }

  #[test]
  fn sha3_384_digest_pair_matches_sequential() {
    let msg_a = b"The quick brown fox jumps over the lazy dog";
    let msg_b = b"";
    let msg_c = &[0xABu8; 4096];

    let ref_a = Sha3_384::digest(msg_a);
    let ref_b = Sha3_384::digest(msg_b);
    let ref_c = Sha3_384::digest(msg_c);

    let (out_a, out_b) = Sha3_384::digest_pair(msg_a, msg_b);
    assert_eq!(out_a, ref_a);
    assert_eq!(out_b, ref_b);

    let (out_c, out_b2) = Sha3_384::digest_pair(msg_c, msg_b);
    assert_eq!(out_c, ref_c);
    assert_eq!(out_b2, ref_b);

    let (out_c1, out_c2) = Sha3_384::digest_pair(msg_c, msg_c);
    assert_eq!(out_c1, ref_c);
    assert_eq!(out_c2, ref_c);
  }

  #[test]
  fn sha3_512_digest_pair_matches_sequential() {
    let msg_a = b"The quick brown fox jumps over the lazy dog";
    let msg_b = b"";
    let msg_c = &[0xABu8; 4096];

    let ref_a = Sha3_512::digest(msg_a);
    let ref_b = Sha3_512::digest(msg_b);
    let ref_c = Sha3_512::digest(msg_c);

    let (out_a, out_b) = Sha3_512::digest_pair(msg_a, msg_b);
    assert_eq!(out_a, ref_a);
    assert_eq!(out_b, ref_b);

    let (out_c, out_b2) = Sha3_512::digest_pair(msg_c, msg_b);
    assert_eq!(out_c, ref_c);
    assert_eq!(out_b2, ref_b);

    let (out_c1, out_c2) = Sha3_512::digest_pair(msg_c, msg_c);
    assert_eq!(out_c1, ref_c);
    assert_eq!(out_c2, ref_c);
  }
}
