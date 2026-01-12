//! SP800-185 derived Keccak primitives: cSHAKE and KMAC.
//!
//! These are built on top of the internal Keccak sponge core and are usable in
//! `no_std` environments.

#![allow(clippy::indexing_slicing)] // Fixed-size encoding buffers + audited indexing

use traits::Xof;

use super::keccak::{KeccakCore, KeccakXof};

const DS_SHAKE: u8 = 0x1F;
const DS_CSHAKE: u8 = 0x04;

const ZEROES_168: [u8; 168] = [0u8; 168];

#[inline(always)]
fn left_encode(val: u64, b: &mut [u8; 9]) -> &[u8] {
  b[1..].copy_from_slice(&val.to_be_bytes());
  let i = b[1..8].iter().take_while(|&&a| a == 0).count();
  b[i] = (8 - i) as u8;
  &b[i..]
}

#[inline(always)]
fn right_encode(val: u64, b: &mut [u8; 9]) -> &[u8] {
  b[..8].copy_from_slice(&val.to_be_bytes());
  let i = b[..7].iter().take_while(|&&a| a == 0).count();
  b[8] = (8 - i) as u8;
  &b[i..=8]
}

#[inline(always)]
fn absorb_encode_string<const RATE: usize>(core: &mut KeccakCore<RATE>, data: &[u8], bytes: &mut usize) {
  let mut b = [0u8; 9];
  let bit_len = (data.len() as u64) * 8;
  let le = left_encode(bit_len, &mut b);
  core.update(le);
  *bytes += le.len();
  core.update(data);
  *bytes += data.len();
}

#[inline(always)]
fn absorb_bytepad_start<const RATE: usize>(core: &mut KeccakCore<RATE>, bytes: &mut usize) {
  let mut b = [0u8; 9];
  let le = left_encode(RATE as u64, &mut b);
  core.update(le);
  *bytes += le.len();
}

#[inline(always)]
fn absorb_bytepad_finish<const RATE: usize>(core: &mut KeccakCore<RATE>, bytes: &mut usize) {
  let rem = *bytes % RATE;
  if rem == 0 {
    return;
  }
  let pad = RATE - rem;
  core.update(&ZEROES_168[..pad]);
  *bytes += pad;
}

#[inline(always)]
fn absorb_bytepad_encode_string<const RATE: usize>(core: &mut KeccakCore<RATE>, data: &[u8]) {
  let mut bytes = 0usize;
  absorb_bytepad_start(core, &mut bytes);
  absorb_encode_string(core, data, &mut bytes);
  absorb_bytepad_finish(core, &mut bytes);
}

#[inline(always)]
fn absorb_cshake_prefix<const RATE: usize>(
  core: &mut KeccakCore<RATE>,
  function_name: &[u8],
  customization: &[u8],
) -> u8 {
  if function_name.is_empty() && customization.is_empty() {
    return DS_SHAKE;
  }

  let mut bytes = 0usize;
  absorb_bytepad_start(core, &mut bytes);
  absorb_encode_string(core, function_name, &mut bytes);
  absorb_encode_string(core, customization, &mut bytes);
  absorb_bytepad_finish(core, &mut bytes);
  DS_CSHAKE
}

#[derive(Clone)]
pub struct CShake128 {
  initial: KeccakCore<168>,
  core: KeccakCore<168>,
  ds: u8,
}

impl CShake128 {
  #[inline]
  #[must_use]
  pub fn new(customization: &[u8]) -> Self {
    Self::new_with_function_name(&[], customization)
  }

  #[inline]
  #[must_use]
  pub fn new_with_function_name(function_name: &[u8], customization: &[u8]) -> Self {
    let mut core = KeccakCore::<168>::default();
    let ds = absorb_cshake_prefix(&mut core, function_name, customization);
    Self {
      initial: core.clone(),
      core,
      ds,
    }
  }

  #[inline]
  pub fn update(&mut self, data: &[u8]) {
    self.core.update(data);
  }

  #[inline]
  #[must_use]
  pub fn finalize_xof(&self) -> CShake128Xof {
    CShake128Xof {
      inner: self.core.finalize_xof(self.ds),
    }
  }

  #[inline]
  pub fn reset(&mut self) {
    self.core = self.initial.clone();
  }

  #[inline]
  pub fn hash_into(function_name: &[u8], customization: &[u8], data: &[u8], out: &mut [u8]) {
    let mut h = Self::new_with_function_name(function_name, customization);
    h.update(data);
    h.core.finalize_xof_into(h.ds, out);
  }
}

#[derive(Clone)]
pub struct CShake128Xof {
  inner: KeccakXof<168>,
}

impl Xof for CShake128Xof {
  #[inline]
  fn squeeze(&mut self, out: &mut [u8]) {
    self.inner.squeeze_into(out);
  }
}

#[derive(Clone)]
pub struct CShake256 {
  initial: KeccakCore<136>,
  core: KeccakCore<136>,
  ds: u8,
}

impl CShake256 {
  #[inline]
  #[must_use]
  pub fn new(customization: &[u8]) -> Self {
    Self::new_with_function_name(&[], customization)
  }

  #[inline]
  #[must_use]
  pub fn new_with_function_name(function_name: &[u8], customization: &[u8]) -> Self {
    let mut core = KeccakCore::<136>::default();
    let ds = absorb_cshake_prefix(&mut core, function_name, customization);
    Self {
      initial: core.clone(),
      core,
      ds,
    }
  }

  #[inline]
  pub fn update(&mut self, data: &[u8]) {
    self.core.update(data);
  }

  #[inline]
  #[must_use]
  pub fn finalize_xof(&self) -> CShake256Xof {
    CShake256Xof {
      inner: self.core.finalize_xof(self.ds),
    }
  }

  #[inline]
  pub fn reset(&mut self) {
    self.core = self.initial.clone();
  }

  #[inline]
  pub fn hash_into(function_name: &[u8], customization: &[u8], data: &[u8], out: &mut [u8]) {
    let mut h = Self::new_with_function_name(function_name, customization);
    h.update(data);
    h.core.finalize_xof_into(h.ds, out);
  }
}

#[derive(Clone)]
pub struct CShake256Xof {
  inner: KeccakXof<136>,
}

impl Xof for CShake256Xof {
  #[inline]
  fn squeeze(&mut self, out: &mut [u8]) {
    self.inner.squeeze_into(out);
  }
}

#[derive(Clone)]
pub struct Kmac128 {
  initial: KeccakCore<168>,
  core: KeccakCore<168>,
}

impl Kmac128 {
  #[inline]
  #[must_use]
  pub fn new(key: &[u8], customization: &[u8]) -> Self {
    let mut core = KeccakCore::<168>::default();
    let _ds = absorb_cshake_prefix(&mut core, b"KMAC", customization);
    absorb_bytepad_encode_string(&mut core, key);
    Self {
      initial: core.clone(),
      core,
    }
  }

  #[inline]
  pub fn update(&mut self, data: &[u8]) {
    self.core.update(data);
  }

  #[inline]
  pub fn reset(&mut self) {
    self.core = self.initial.clone();
  }

  /// KMAC128 with fixed output length `out.len()`.
  #[inline]
  pub fn finalize_into(&self, out: &mut [u8]) {
    let mut tmp = self.core.clone();
    let mut b = [0u8; 9];
    let re = right_encode((out.len() as u64) * 8, &mut b);
    tmp.update(re);
    tmp.finalize_xof_into(DS_CSHAKE, out);
  }

  /// KMACXOF128 (extendable output).
  #[inline]
  #[must_use]
  pub fn finalize_xof(&self) -> Kmac128Xof {
    let mut tmp = self.core.clone();
    let mut b = [0u8; 9];
    let re = right_encode(0, &mut b);
    tmp.update(re);
    Kmac128Xof {
      inner: tmp.finalize_xof(DS_CSHAKE),
    }
  }
}

#[derive(Clone)]
pub struct Kmac128Xof {
  inner: KeccakXof<168>,
}

impl Xof for Kmac128Xof {
  #[inline]
  fn squeeze(&mut self, out: &mut [u8]) {
    self.inner.squeeze_into(out);
  }
}

#[derive(Clone)]
pub struct Kmac256 {
  initial: KeccakCore<136>,
  core: KeccakCore<136>,
}

impl Kmac256 {
  #[inline]
  #[must_use]
  pub fn new(key: &[u8], customization: &[u8]) -> Self {
    let mut core = KeccakCore::<136>::default();
    let _ds = absorb_cshake_prefix(&mut core, b"KMAC", customization);
    absorb_bytepad_encode_string(&mut core, key);
    Self {
      initial: core.clone(),
      core,
    }
  }

  #[inline]
  pub fn update(&mut self, data: &[u8]) {
    self.core.update(data);
  }

  #[inline]
  pub fn reset(&mut self) {
    self.core = self.initial.clone();
  }

  /// KMAC256 with fixed output length `out.len()`.
  #[inline]
  pub fn finalize_into(&self, out: &mut [u8]) {
    let mut tmp = self.core.clone();
    let mut b = [0u8; 9];
    let re = right_encode((out.len() as u64) * 8, &mut b);
    tmp.update(re);
    tmp.finalize_xof_into(DS_CSHAKE, out);
  }

  /// KMACXOF256 (extendable output).
  #[inline]
  #[must_use]
  pub fn finalize_xof(&self) -> Kmac256Xof {
    let mut tmp = self.core.clone();
    let mut b = [0u8; 9];
    let re = right_encode(0, &mut b);
    tmp.update(re);
    Kmac256Xof {
      inner: tmp.finalize_xof(DS_CSHAKE),
    }
  }
}

#[derive(Clone)]
pub struct Kmac256Xof {
  inner: KeccakXof<136>,
}

impl Xof for Kmac256Xof {
  #[inline]
  fn squeeze(&mut self, out: &mut [u8]) {
    self.inner.squeeze_into(out);
  }
}
