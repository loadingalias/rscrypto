use core::simd::i64x2;

use super::{BLOCK_SIZE, KEY_SIZE, NONCE_SIZE, load_u32_le, xor_keystream_portable};

const BLOCKS_PER_BATCH: usize = 4;

#[inline]
pub(super) fn xor_keystream(key: &[u8; KEY_SIZE], initial_counter: u32, nonce: &[u8; NONCE_SIZE], buffer: &mut [u8]) {
  // SAFETY: Backend selection guarantees the z/Vector facility before this wrapper is chosen.
  unsafe { xor_keystream_impl(key, initial_counter, nonce, buffer) }
}

#[target_feature(enable = "vector")]
unsafe fn xor_keystream_impl(key: &[u8; KEY_SIZE], initial_counter: u32, nonce: &[u8; NONCE_SIZE], buffer: &mut [u8]) {
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
      // SAFETY: target_feature ensures z/Vector availability.
      unsafe {
        quarter_round(&mut x0, &mut x4, &mut x8, &mut x12);
        quarter_round(&mut x1, &mut x5, &mut x9, &mut x13);
        quarter_round(&mut x2, &mut x6, &mut x10, &mut x14);
        quarter_round(&mut x3, &mut x7, &mut x11, &mut x15);

        quarter_round(&mut x0, &mut x5, &mut x10, &mut x15);
        quarter_round(&mut x1, &mut x6, &mut x11, &mut x12);
        quarter_round(&mut x2, &mut x7, &mut x8, &mut x13);
        quarter_round(&mut x3, &mut x4, &mut x9, &mut x14);
      }

      round = round.strict_add(1);
    }

    // SAFETY: target_feature ensures z/Vector availability.
    unsafe {
      x0 = vaf(x0, o0);
      x1 = vaf(x1, o1);
      x2 = vaf(x2, o2);
      x3 = vaf(x3, o3);
      x4 = vaf(x4, o4);
      x5 = vaf(x5, o5);
      x6 = vaf(x6, o6);
      x7 = vaf(x7, o7);
      x8 = vaf(x8, o8);
      x9 = vaf(x9, o9);
      x10 = vaf(x10, o10);
      x11 = vaf(x11, o11);
      x12 = vaf(x12, o12);
      x13 = vaf(x13, o13);
      x14 = vaf(x14, o14);
      x15 = vaf(x15, o15);
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

/// Vector add fullword: `vaf`.
#[inline]
#[target_feature(enable = "vector")]
unsafe fn vaf(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z/Vector facility available via target_feature.
  unsafe {
    core::arch::asm!(
      "vaf {out}, {a}, {b}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// Vector exclusive OR: `vx`.
#[inline]
#[target_feature(enable = "vector")]
unsafe fn vx(a: i64x2, b: i64x2) -> i64x2 {
  let out: i64x2;
  // SAFETY: z/Vector facility available via target_feature.
  unsafe {
    core::arch::asm!(
      "vx {out}, {a}, {b}",
      out = lateout(vreg) out,
      a = in(vreg) a,
      b = in(vreg) b,
      options(nomem, nostack, pure)
    );
  }
  out
}

/// Element rotate-LEFT u32 via z/Arch `verll` (M4=2, fullword elements).
///
/// ChaCha20 quarter rounds rotate LEFT by 16, 12, 8, and 7 bits, and `verll`
/// already rotates LEFT. The immediate maps directly to the ChaCha constant.
#[inline]
#[target_feature(enable = "vector")]
unsafe fn rotl32_via_verll<const BITS: u32>(a: i64x2) -> i64x2 {
  const {
    assert!(BITS > 0 && BITS < 32, "rotate-left amount must be in 1..32 for u32");
  }
  let out: i64x2;
  // SAFETY: z/Vector facility available via target_feature; immediate is
  // compile-time bounded to 1..32 by the const assertion above.
  unsafe {
    core::arch::asm!(
      "verll {out}, {a}, {bits}, 2",
      out = lateout(vreg) out,
      a = in(vreg) a,
      bits = const BITS,
      options(nomem, nostack, pure)
    );
  }
  out
}

#[inline]
#[target_feature(enable = "vector")]
unsafe fn quarter_round(a: &mut i64x2, b: &mut i64x2, c: &mut i64x2, d: &mut i64x2) {
  // SAFETY: z/Vector facility available via enclosing target_feature.
  unsafe {
    *a = vaf(*a, *b);
    *d = rotl32_via_verll::<16>(vx(*d, *a));

    *c = vaf(*c, *d);
    *b = rotl32_via_verll::<12>(vx(*b, *c));

    *a = vaf(*a, *b);
    *d = rotl32_via_verll::<8>(vx(*d, *a));

    *c = vaf(*c, *d);
    *b = rotl32_via_verll::<7>(vx(*b, *c));
  }
}

#[cfg(test)]
mod tests {
  use super::{from_u32x4, rotl32_via_verll, to_u32x4};

  const ROT_SEEDS: [u32; 4] = [0x0123_4567, 0x89AB_CDEF, 0xDEAD_BEEF, 0x0000_00FF];

  #[target_feature(enable = "vector")]
  unsafe fn assert_rotr_matches_portable() {
    let input = from_u32x4(ROT_SEEDS);
    // SAFETY: target_feature guarantees z/Vector on this function.
    let (r16, r12, r8, r7) = unsafe {
      (
        rotl32_via_verll::<16>(input),
        rotl32_via_verll::<12>(input),
        rotl32_via_verll::<8>(input),
        rotl32_via_verll::<7>(input),
      )
    };

    let expect = |bits: u32| -> [u32; 4] {
      [
        ROT_SEEDS[0].rotate_left(bits),
        ROT_SEEDS[1].rotate_left(bits),
        ROT_SEEDS[2].rotate_left(bits),
        ROT_SEEDS[3].rotate_left(bits),
      ]
    };

    assert_eq!(to_u32x4(r16), expect(16), "rotl16 mismatch");
    assert_eq!(to_u32x4(r12), expect(12), "rotl12 mismatch");
    assert_eq!(to_u32x4(r8), expect(8), "rotl8 mismatch");
    assert_eq!(to_u32x4(r7), expect(7), "rotl7 mismatch");
  }

  #[test]
  fn rotl32_via_verll_matches_u32_rotate_left() {
    assert!(
      std::arch::is_s390x_feature_detected!("vector"),
      "s390x vector facility is required for ChaCha20 rotation tests"
    );
    // SAFETY: the runtime check above guarantees z/Vector.
    unsafe { assert_rotr_matches_portable() };
  }
}
