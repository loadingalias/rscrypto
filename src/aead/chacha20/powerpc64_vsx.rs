use core::simd::i64x2;

use super::{BLOCK_SIZE, KEY_SIZE, NONCE_SIZE, load_u32_le, xor_keystream_portable};

const BLOCKS_PER_BATCH: usize = 4;

#[inline]
pub(super) fn xor_keystream(key: &[u8; KEY_SIZE], initial_counter: u32, nonce: &[u8; NONCE_SIZE], buffer: &mut [u8]) {
  // SAFETY: Backend selection guarantees POWER vector support before this wrapper is chosen.
  unsafe { xor_keystream_impl(key, initial_counter, nonce, buffer) }
}

/// Precomputed `vrlw` shift vectors for the four ChaCha20 rotations.
///
/// ChaCha20's constants (16, 12, 8, 7) are rotate-RIGHT values. `vrlw` is a
/// rotate-LEFT by the shift in each lane, so each stored splat carries the
/// ROL amount `32 - ROR`. Field names encode the ChaCha20 (ROR) semantic;
/// the ROR→ROL conversion is centralized in `splat_ror` below.
struct RotShifts {
  rot16: i64x2,
  rot12: i64x2,
  rot8: i64x2,
  rot7: i64x2,
}

/// Build a `vrlw` shift splat from a ChaCha20 rotate-RIGHT constant.
///
/// Returns `splat(32 - ROR)`; `rol(32 - ROR)` is equivalent to `ror(ROR)`
/// for u32, and `vrlw` rotates LEFT by its shift operand.
#[inline(always)]
fn splat_ror(ror: u32) -> i64x2 {
  debug_assert!(ror > 0 && ror < 32, "ROR must be in 1..32 for u32 rotate-right");
  splat(32 - ror)
}

impl RotShifts {
  fn new() -> Self {
    Self {
      rot16: splat_ror(16),
      rot12: splat_ror(12),
      rot8: splat_ror(8),
      rot7: splat_ror(7),
    }
  }
}

#[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
unsafe fn xor_keystream_impl(key: &[u8; KEY_SIZE], initial_counter: u32, nonce: &[u8; NONCE_SIZE], buffer: &mut [u8]) {
  let rot = RotShifts::new();

  let mut counter = initial_counter;
  let mut batches = buffer.chunks_exact_mut(BLOCK_SIZE * BLOCKS_PER_BATCH);
  for chunk in &mut batches {
    debug_assert!(counter.checked_add((BLOCKS_PER_BATCH - 1) as u32).is_some());

    let mut x0 = splat(0x6170_7865);
    let mut x1 = splat(0x3320_646e);
    let mut x2 = splat(0x7962_2d32);
    let mut x3 = splat(0x6b20_6574);
    let mut x4 = splat(load_u32_le(&key[0..4]));
    let mut x5 = splat(load_u32_le(&key[4..8]));
    let mut x6 = splat(load_u32_le(&key[8..12]));
    let mut x7 = splat(load_u32_le(&key[12..16]));
    let mut x8 = splat(load_u32_le(&key[16..20]));
    let mut x9 = splat(load_u32_le(&key[20..24]));
    let mut x10 = splat(load_u32_le(&key[24..28]));
    let mut x11 = splat(load_u32_le(&key[28..32]));
    let mut x12 = from_u32x4([
      counter,
      counter.wrapping_add(1),
      counter.wrapping_add(2),
      counter.wrapping_add(3),
    ]);
    let mut x13 = splat(load_u32_le(&nonce[0..4]));
    let mut x14 = splat(load_u32_le(&nonce[4..8]));
    let mut x15 = splat(load_u32_le(&nonce[8..12]));

    let o0 = x0;
    let o1 = x1;
    let o2 = x2;
    let o3 = x3;
    let o4 = x4;
    let o5 = x5;
    let o6 = x6;
    let o7 = x7;
    let o8 = x8;
    let o9 = x9;
    let o10 = x10;
    let o11 = x11;
    let o12 = x12;
    let o13 = x13;
    let o14 = x14;
    let o15 = x15;

    let mut round = 0usize;
    while round < 10 {
      // SAFETY: target_feature ensures VSX availability.
      unsafe {
        quarter_round(&mut x0, &mut x4, &mut x8, &mut x12, &rot);
        quarter_round(&mut x1, &mut x5, &mut x9, &mut x13, &rot);
        quarter_round(&mut x2, &mut x6, &mut x10, &mut x14, &rot);
        quarter_round(&mut x3, &mut x7, &mut x11, &mut x15, &rot);

        quarter_round(&mut x0, &mut x5, &mut x10, &mut x15, &rot);
        quarter_round(&mut x1, &mut x6, &mut x11, &mut x12, &rot);
        quarter_round(&mut x2, &mut x7, &mut x8, &mut x13, &rot);
        quarter_round(&mut x3, &mut x4, &mut x9, &mut x14, &rot);
      }

      round = round.strict_add(1);
    }

    // SAFETY: target_feature ensures VSX availability.
    unsafe {
      x0 = vadduwm(x0, o0);
      x1 = vadduwm(x1, o1);
      x2 = vadduwm(x2, o2);
      x3 = vadduwm(x3, o3);
      x4 = vadduwm(x4, o4);
      x5 = vadduwm(x5, o5);
      x6 = vadduwm(x6, o6);
      x7 = vadduwm(x7, o7);
      x8 = vadduwm(x8, o8);
      x9 = vadduwm(x9, o9);
      x10 = vadduwm(x10, o10);
      x11 = vadduwm(x11, o11);
      x12 = vadduwm(x12, o12);
      x13 = vadduwm(x13, o13);
      x14 = vadduwm(x14, o14);
      x15 = vadduwm(x15, o15);
    }

    let words = [
      to_u32x4(x0),
      to_u32x4(x1),
      to_u32x4(x2),
      to_u32x4(x3),
      to_u32x4(x4),
      to_u32x4(x5),
      to_u32x4(x6),
      to_u32x4(x7),
      to_u32x4(x8),
      to_u32x4(x9),
      to_u32x4(x10),
      to_u32x4(x11),
      to_u32x4(x12),
      to_u32x4(x13),
      to_u32x4(x14),
      to_u32x4(x15),
    ];

    let mut block_index = 0usize;
    while block_index < BLOCKS_PER_BATCH {
      let mut word_index = 0usize;
      while word_index < 16 {
        let offset = block_index.strict_mul(BLOCK_SIZE).strict_add(word_index.strict_mul(4));
        let keystream = words[word_index][block_index].to_le_bytes();
        chunk[offset..offset.strict_add(4)]
          .iter_mut()
          .zip(keystream)
          .for_each(|(dst, src)| *dst ^= src);
        word_index = word_index.strict_add(1);
      }
      block_index = block_index.strict_add(1);
    }

    counter = counter.wrapping_add(BLOCKS_PER_BATCH as u32);
  }

  let remainder = batches.into_remainder();
  if !remainder.is_empty() {
    xor_keystream_portable(key, counter, nonce, remainder);
  }
}

#[inline(always)]
fn from_u32x4(words: [u32; 4]) -> i64x2 {
  // SAFETY: [u32; 4] and i64x2 are both 16 bytes with no alignment mismatch.
  unsafe { core::mem::transmute(words) }
}

#[inline(always)]
fn to_u32x4(v: i64x2) -> [u32; 4] {
  // SAFETY: i64x2 and [u32; 4] are both 16 bytes.
  unsafe { core::mem::transmute(v) }
}

#[inline(always)]
fn splat(val: u32) -> i64x2 {
  from_u32x4([val; 4])
}

/// Vector add unsigned word modulo: `vadduwm`.
#[inline(always)]
unsafe fn vadduwm(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: POWER8+ VSX available via enclosing target_feature.
  unsafe {
    core::arch::asm!(
      "vadduwm {out}, {a}, {b}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// Vector XOR: `vxor`.
#[inline(always)]
unsafe fn vxor(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: POWER8+ VSX available via enclosing target_feature.
  unsafe {
    core::arch::asm!(
      "vxor {out}, {a}, {b}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// Vector rotate left word: `vrlw`.
#[inline(always)]
unsafe fn vrlw(value: i64x2, shift: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: POWER8+ VSX available via enclosing target_feature.
  unsafe {
    core::arch::asm!(
      "vrlw {out}, {val}, {shift}",
      out = lateout(vreg) out,
      val = in(vreg) value,
      shift = in(vreg) shift,
      options(nomem, nostack, pure)
    );
  }
  out
}

#[inline(always)]
unsafe fn quarter_round(a: &mut i64x2, b: &mut i64x2, c: &mut i64x2, d: &mut i64x2, rot: &RotShifts) {
  // SAFETY: POWER8+ VSX available via enclosing target_feature.
  unsafe {
    *a = vadduwm(*a, *b);
    *d = vrlw(vxor(*d, *a), rot.rot16);

    *c = vadduwm(*c, *d);
    *b = vrlw(vxor(*b, *c), rot.rot12);

    *a = vadduwm(*a, *b);
    *d = vrlw(vxor(*d, *a), rot.rot8);

    *c = vadduwm(*c, *d);
    *b = vrlw(vxor(*b, *c), rot.rot7);
  }
}

#[cfg(test)]
mod tests {
  use super::{RotShifts, from_u32x4, to_u32x4, vrlw};
  use crate::platform::caps::power;

  const ROT_SEEDS: [u32; 4] = [0x0123_4567, 0x89AB_CDEF, 0xDEAD_BEEF, 0x0000_00FF];

  #[target_feature(enable = "altivec", enable = "vsx", enable = "power8-vector")]
  unsafe fn assert_vrlw_matches_portable() {
    let input = from_u32x4(ROT_SEEDS);
    let shifts = RotShifts::new();

    // SAFETY: target_feature on this function guarantees POWER8+ VSX.
    let (r16, r12, r8, r7) = unsafe {
      (
        vrlw(input, shifts.rot16),
        vrlw(input, shifts.rot12),
        vrlw(input, shifts.rot8),
        vrlw(input, shifts.rot7),
      )
    };

    let expect = |ror: u32| -> [u32; 4] {
      [
        ROT_SEEDS[0].rotate_right(ror),
        ROT_SEEDS[1].rotate_right(ror),
        ROT_SEEDS[2].rotate_right(ror),
        ROT_SEEDS[3].rotate_right(ror),
      ]
    };

    assert_eq!(to_u32x4(r16), expect(16), "ror16 mismatch");
    assert_eq!(to_u32x4(r12), expect(12), "ror12 mismatch");
    assert_eq!(to_u32x4(r8), expect(8), "ror8 mismatch");
    assert_eq!(to_u32x4(r7), expect(7), "ror7 mismatch");
  }

  #[test]
  fn vrlw_with_rotshifts_matches_u32_rotate_right() {
    if !crate::platform::caps().has(power::POWER8_READY) {
      return; // Feature not available on this ppc64le host; nothing to assert.
    }
    // SAFETY: the runtime capability check above guarantees POWER8+ VSX.
    unsafe { assert_vrlw_matches_portable() };
  }
}
