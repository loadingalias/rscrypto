//! Carryless-multiply CRC engine for vector units whose multiply instruction folds a whole lane.
//!
//! POWER `VPMSUMD` and IBM Z `VGFMG` both compute `a.lo * b.lo ^ a.hi * b.hi` for 128-bit lanes `a`
//! and `b`, which is one fold. This engine serves both, every CRC width, and both bit orders; see
//! the parent module for the 64-bit domain.
//!
//! A lane's `lo` is its low 64 bits. On a big-endian target that is element 1 of the vector.

use core::simd::{ToBytes, u8x16, u64x2};

use super::{Keys, MASK};

/// The sum of the two 64-bit carryless products of a lane pair, as one 128-bit lane.
pub(in crate::checksum) trait SumOfProducts {
  /// `a.lo * b.lo ^ a.hi * b.hi`.
  ///
  /// # Safety
  ///
  /// The CPU must support the instruction.
  unsafe fn sum(a: u64x2, b: u64x2) -> u64x2;
}

/// Lanes in the main loop: eight 16-byte lanes, 128 bytes per iteration.
const LANES: usize = 8;

const LITTLE: bool = cfg!(target_endian = "little");

#[inline(always)]
fn pair(lo: u64, hi: u64) -> u64x2 {
  if LITTLE {
    u64x2::from_array([lo, hi])
  } else {
    u64x2::from_array([hi, lo])
  }
}

#[inline(always)]
fn lo(v: u64x2) -> u64 {
  v[usize::from(!LITTLE)]
}

#[inline(always)]
fn hi(v: u64x2) -> u64 {
  v[usize::from(LITTLE)]
}

#[inline(always)]
fn key(k: &[u64; 2]) -> u64x2 {
  pair(k[0], k[1])
}

/// Turns 16 input bytes into a lane in the engine's bit order.
///
/// A reflected lane is the little-endian value of the bytes; an MSB-first lane is the big-endian
/// value. A native load needs a full byte reversal exactly when the two orders differ.
#[inline(always)]
fn lane<const REFLECT: bool>(bytes: u8x16) -> u64x2 {
  if REFLECT == LITTLE {
    u64x2::from_ne_bytes(bytes)
  } else {
    u64x2::from_ne_bytes(bytes.reverse())
  }
}

#[inline(always)]
fn load_lane<const REFLECT: bool>(bytes: &[u8; 16]) -> u64x2 {
  lane::<REFLECT>(u8x16::from_array(*bytes))
}

/// The engine, generic over the multiply instruction.
pub(in crate::checksum) struct Engine<M>(core::marker::PhantomData<M>);

impl<M: SumOfProducts> Engine<M> {
  /// `clmul(a, b)` as a 128-bit lane.
  ///
  /// # Safety
  ///
  /// The CPU must support `M`.
  #[inline(always)]
  unsafe fn clmul(a: u64, b: u64) -> u64x2 {
    // SAFETY: the caller guarantees the instruction.
    unsafe { M::sum(pair(a, 0), pair(b, 0)) }
  }

  /// Barrett reduction of a 128-bit product modulo `G'`.
  ///
  /// # Safety
  ///
  /// The CPU must support `M`.
  #[inline(always)]
  pub(in crate::checksum) unsafe fn barrett<const REFLECT: bool>(y: u64x2, keys: &Keys) -> u64 {
    // SAFETY: the caller guarantees the instruction.
    unsafe {
      if REFLECT {
        let q = lo(Self::clmul(lo(y), keys.mu));
        hi(y) ^ hi(Self::clmul(q, keys.poly)) ^ (q & keys.top)
      } else {
        let q = hi(y) ^ hi(Self::clmul(hi(y), keys.mu));
        lo(y) ^ lo(Self::clmul(q, keys.poly))
      }
    }
  }

  /// Multiplies an engine-domain value by `x^(8 * len)`, for `combine`.
  ///
  /// # Safety
  ///
  /// The CPU must support `M`.
  #[inline(always)]
  pub(in crate::checksum) unsafe fn shift<const REFLECT: bool>(
    value: u64,
    len: usize,
    powers: &[u64; 64],
    keys: &Keys,
  ) -> u64 {
    // SAFETY: the caller guarantees the instruction.
    super::shift_by(value, len, powers, |value, power| unsafe {
      Self::barrett::<REFLECT>(Self::clmul(value, power), keys)
    })
  }

  /// Updates a 64-bit-domain CRC state with at least 16 bytes.
  ///
  /// # Safety
  ///
  /// The CPU must support `M`, and `data` must hold at least 16 bytes.
  #[inline(always)]
  pub(in crate::checksum) unsafe fn update<const REFLECT: bool>(state: u64, data: &[u8], keys: &Keys) -> u64 {
    // SAFETY: the caller guarantees the instruction; every load reads a slice element.
    unsafe {
      let fold = |x: u64x2, k: &[u64; 2]| M::sum(x, key(k));
      let (blocks, tail) = data.as_chunks::<16>();
      let inject = if REFLECT { pair(state, 0) } else { pair(0, state) };
      let (mut x, rest) = if let Some((first, rest)) = blocks.split_first_chunk::<LANES>() {
        let mut acc = first.map(|block| load_lane::<REFLECT>(&block));
        acc[0] ^= inject;
        let k = key(&keys.fold[LANES.strict_sub(1)]);
        let (groups, rest) = rest.as_chunks::<LANES>();
        for group in groups {
          for (lane, block) in acc.iter_mut().zip(group) {
            *lane = M::sum(*lane, k) ^ load_lane::<REFLECT>(block);
          }
        }
        // Lane `j` moves by `7 - j` blocks; pairwise sums keep the dependency chains short.
        let a = fold(acc[0], &keys.fold[6]) ^ fold(acc[1], &keys.fold[5]);
        let b = fold(acc[2], &keys.fold[4]) ^ fold(acc[3], &keys.fold[3]);
        let c = fold(acc[4], &keys.fold[2]) ^ fold(acc[5], &keys.fold[1]);
        let d = fold(acc[6], &keys.fold[0]) ^ acc[7];
        ((a ^ b) ^ (c ^ d), rest)
      } else {
        let (first, rest) = blocks.split_at(1);
        (load_lane::<REFLECT>(&first[0]) ^ inject, rest)
      };
      // `x` precedes the remaining blocks; each moves by its distance from the last block.
      if let Some((last, middle)) = rest.split_last() {
        let n = rest.len();
        let mut sum = fold(x, &keys.fold[n.strict_sub(1)]) ^ load_lane::<REFLECT>(last);
        for (j, block) in middle.iter().enumerate() {
          sum ^= fold(load_lane::<REFLECT>(block), &keys.fold[n.strict_sub(2).strict_sub(j)]);
        }
        x = sum;
      }
      if !tail.is_empty() {
        let r = tail.len();
        let last = &data[data.len().strict_sub(16)..].as_chunks::<16>().0[0];
        let mask = u8x16::from_array(MASK[r..].as_chunks::<16>().0[0]);
        x = fold(x, &keys.tail[r]) ^ lane::<REFLECT>(u8x16::from_array(*last) & mask);
      }
      Self::barrett::<REFLECT>(fold(x, &keys.tail[8]), keys)
    }
  }
}
