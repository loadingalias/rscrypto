//! SipHash (**NOT CRYPTO**).
//!
//! SipHash is a *keyed* hash designed to defend hash tables against collision
//! attacks on untrusted inputs. It is not a cryptographic MAC.

#![allow(clippy::indexing_slicing)] // Tight block parsing

use traits::FastHash;

#[doc(hidden)]
pub mod dispatch;
#[doc(hidden)]
pub mod dispatch_tables;
pub(crate) mod kernels;

#[derive(Clone, Default)]
pub struct SipHash13;

#[derive(Clone, Default)]
pub struct SipHash24;

const C0: u64 = 0x736f_6d65_7073_6575;
const C1: u64 = 0x646f_7261_6e64_6f6d;
const C2: u64 = 0x6c79_6765_6e65_7261;
const C3: u64 = 0x7465_6462_7974_6573;

#[inline(always)]
fn sip_round(v0: &mut u64, v1: &mut u64, v2: &mut u64, v3: &mut u64) {
  *v0 = v0.wrapping_add(*v1);
  *v1 = v1.rotate_left(13);
  *v1 ^= *v0;
  *v0 = v0.rotate_left(32);

  *v2 = v2.wrapping_add(*v3);
  *v3 = v3.rotate_left(16);
  *v3 ^= *v2;

  *v0 = v0.wrapping_add(*v3);
  *v3 = v3.rotate_left(21);
  *v3 ^= *v0;

  *v2 = v2.wrapping_add(*v1);
  *v1 = v1.rotate_left(17);
  *v1 ^= *v2;
  *v2 = v2.rotate_left(32);
}

#[inline(always)]
fn siphash13(key: [u64; 2], data: &[u8]) -> u64 {
  let k0 = key[0];
  let k1 = key[1];

  let mut v0 = C0 ^ k0;
  let mut v1 = C1 ^ k1;
  let mut v2 = C2 ^ k0;
  let mut v3 = C3 ^ k1;

  let (blocks, tail) = data.as_chunks::<8>();
  for block in blocks {
    let m = u64::from_le_bytes(*block);
    v3 ^= m;
    sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
    v0 ^= m;
  }

  let mut b = (data.len() as u64) << 56;
  match tail.len() {
    7 => {
      b |= (tail[6] as u64) << 48;
      b |= (tail[5] as u64) << 40;
      b |= (tail[4] as u64) << 32;
      b |= (tail[3] as u64) << 24;
      b |= (tail[2] as u64) << 16;
      b |= (tail[1] as u64) << 8;
      b |= tail[0] as u64;
    }
    6 => {
      b |= (tail[5] as u64) << 40;
      b |= (tail[4] as u64) << 32;
      b |= (tail[3] as u64) << 24;
      b |= (tail[2] as u64) << 16;
      b |= (tail[1] as u64) << 8;
      b |= tail[0] as u64;
    }
    5 => {
      b |= (tail[4] as u64) << 32;
      b |= (tail[3] as u64) << 24;
      b |= (tail[2] as u64) << 16;
      b |= (tail[1] as u64) << 8;
      b |= tail[0] as u64;
    }
    4 => {
      b |= (tail[3] as u64) << 24;
      b |= (tail[2] as u64) << 16;
      b |= (tail[1] as u64) << 8;
      b |= tail[0] as u64;
    }
    3 => {
      b |= (tail[2] as u64) << 16;
      b |= (tail[1] as u64) << 8;
      b |= tail[0] as u64;
    }
    2 => {
      b |= (tail[1] as u64) << 8;
      b |= tail[0] as u64;
    }
    1 => {
      b |= tail[0] as u64;
    }
    _ => {}
  }

  v3 ^= b;
  sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
  v0 ^= b;

  v2 ^= 0xff;
  sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
  sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
  sip_round(&mut v0, &mut v1, &mut v2, &mut v3);

  v0 ^ v1 ^ v2 ^ v3
}

#[inline(always)]
fn siphash24(key: [u64; 2], data: &[u8]) -> u64 {
  let k0 = key[0];
  let k1 = key[1];

  let mut v0 = C0 ^ k0;
  let mut v1 = C1 ^ k1;
  let mut v2 = C2 ^ k0;
  let mut v3 = C3 ^ k1;

  let (blocks, tail) = data.as_chunks::<8>();
  for block in blocks {
    let m = u64::from_le_bytes(*block);
    v3 ^= m;
    sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
    sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
    v0 ^= m;
  }

  let mut b = (data.len() as u64) << 56;
  match tail.len() {
    7 => {
      b |= (tail[6] as u64) << 48;
      b |= (tail[5] as u64) << 40;
      b |= (tail[4] as u64) << 32;
      b |= (tail[3] as u64) << 24;
      b |= (tail[2] as u64) << 16;
      b |= (tail[1] as u64) << 8;
      b |= tail[0] as u64;
    }
    6 => {
      b |= (tail[5] as u64) << 40;
      b |= (tail[4] as u64) << 32;
      b |= (tail[3] as u64) << 24;
      b |= (tail[2] as u64) << 16;
      b |= (tail[1] as u64) << 8;
      b |= tail[0] as u64;
    }
    5 => {
      b |= (tail[4] as u64) << 32;
      b |= (tail[3] as u64) << 24;
      b |= (tail[2] as u64) << 16;
      b |= (tail[1] as u64) << 8;
      b |= tail[0] as u64;
    }
    4 => {
      b |= (tail[3] as u64) << 24;
      b |= (tail[2] as u64) << 16;
      b |= (tail[1] as u64) << 8;
      b |= tail[0] as u64;
    }
    3 => {
      b |= (tail[2] as u64) << 16;
      b |= (tail[1] as u64) << 8;
      b |= tail[0] as u64;
    }
    2 => {
      b |= (tail[1] as u64) << 8;
      b |= tail[0] as u64;
    }
    1 => {
      b |= tail[0] as u64;
    }
    _ => {}
  }

  v3 ^= b;
  sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
  sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
  v0 ^= b;

  v2 ^= 0xff;
  sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
  sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
  sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
  sip_round(&mut v0, &mut v1, &mut v2, &mut v3);

  v0 ^ v1 ^ v2 ^ v3
}

impl FastHash for SipHash13 {
  const OUTPUT_SIZE: usize = 8;
  type Output = u64;
  type Seed = [u64; 2];

  #[inline]
  fn hash_with_seed(seed: Self::Seed, data: &[u8]) -> Self::Output {
    dispatch::hash13_with_seed(seed, data)
  }
}

impl FastHash for SipHash24 {
  const OUTPUT_SIZE: usize = 8;
  type Output = u64;
  type Seed = [u64; 2];

  #[inline]
  fn hash_with_seed(seed: Self::Seed, data: &[u8]) -> Self::Output {
    dispatch::hash24_with_seed(seed, data)
  }
}
