//! Table-free RV64 scalar AES-256 fallback.
//!
//! This is a focused adaptation of the RustCrypto AES 0.8.4 64-bit fixslice
//! backend, reduced to AES-256 encryption over four parallel blocks. The
//! original code is MIT OR Apache-2.0 and derives from Alexandre Adomnicai's
//! fixsliced AES implementation.
//!
//! Reference: Adomnicai et al., "Fixslicing AES-like Ciphers",
//! <https://eprint.iacr.org/2020/1123.pdf>.

#![allow(clippy::unreadable_literal)]

use super::{BLOCK_SIZE, KEY_SIZE};

type State = [u64; 8];

#[derive(Clone)]
pub(super) struct RvFixsliceRoundKeys {
  keys: [u64; 120],
}

impl RvFixsliceRoundKeys {
  #[inline]
  pub(super) fn new(key: &[u8; KEY_SIZE]) -> Self {
    Self {
      keys: aes256_key_schedule(key),
    }
  }

  #[inline]
  #[allow(dead_code)]
  pub(super) fn zeroize(&mut self) {
    // SAFETY: `[u64; 120]` is contiguous and valid to view as bytes for its
    // exact initialized size.
    crate::traits::ct::zeroize(unsafe {
      core::slice::from_raw_parts_mut(self.keys.as_mut_ptr().cast::<u8>(), self.keys.len().strict_mul(8))
    });
  }
}

#[inline]
pub(super) fn encrypt_block(rkeys: &RvFixsliceRoundKeys, block: &mut [u8; BLOCK_SIZE]) {
  let mut blocks = [*block; 4];
  encrypt_4blocks(rkeys, &mut blocks);
  *block = blocks[0];
}

#[inline]
pub(super) fn encrypt_4blocks(rkeys: &RvFixsliceRoundKeys, blocks: &mut [[u8; BLOCK_SIZE]; 4]) {
  let mut state = State::default();
  bitslice(&mut state, &blocks[0], &blocks[1], &blocks[2], &blocks[3]);

  add_round_key(&mut state, &rkeys.keys[..8]);

  let mut rk_off = 8usize;
  loop {
    sub_bytes(&mut state);
    mix_columns_1(&mut state);
    add_round_key(&mut state, &rkeys.keys[rk_off..rk_off.strict_add(8)]);
    rk_off = rk_off.strict_add(8);

    if rk_off == 112 {
      break;
    }

    sub_bytes(&mut state);
    mix_columns_2(&mut state);
    add_round_key(&mut state, &rkeys.keys[rk_off..rk_off.strict_add(8)]);
    rk_off = rk_off.strict_add(8);

    sub_bytes(&mut state);
    mix_columns_3(&mut state);
    add_round_key(&mut state, &rkeys.keys[rk_off..rk_off.strict_add(8)]);
    rk_off = rk_off.strict_add(8);

    sub_bytes(&mut state);
    mix_columns_0(&mut state);
    add_round_key(&mut state, &rkeys.keys[rk_off..rk_off.strict_add(8)]);
    rk_off = rk_off.strict_add(8);
  }

  shift_rows_2(&mut state);
  sub_bytes(&mut state);
  add_round_key(&mut state, &rkeys.keys[112..]);

  *blocks = inv_bitslice(&state);
}

#[inline]
#[allow(dead_code)]
pub(super) fn cipher_round_4(blocks: &mut [[u8; BLOCK_SIZE]; 4], round_keys: &[[u8; BLOCK_SIZE]; 4]) {
  let mut state = State::default();
  bitslice(&mut state, &blocks[0], &blocks[1], &blocks[2], &blocks[3]);
  sub_bytes(&mut state);
  sub_bytes_nots(&mut state);
  shift_rows_1(&mut state);
  mix_columns_0(&mut state);

  let mut out = inv_bitslice(&state);
  let mut lane = 0usize;
  while lane < 4 {
    xor_block(&mut out[lane], &round_keys[lane]);
    lane = lane.strict_add(1);
  }
  *blocks = out;
}

fn aes256_key_schedule(key: &[u8; KEY_SIZE]) -> [u64; 120] {
  let mut rkeys = [0u64; 120];

  bitslice(&mut rkeys[..8], &key[..16], &key[..16], &key[..16], &key[..16]);
  bitslice(&mut rkeys[8..16], &key[16..], &key[16..], &key[16..], &key[16..]);

  let mut rk_off = 8usize;
  let mut rcon = 0usize;
  loop {
    memshift32(&mut rkeys, rk_off);
    rk_off = rk_off.strict_add(8);

    sub_bytes(&mut rkeys[rk_off..rk_off.strict_add(8)]);
    sub_bytes_nots(&mut rkeys[rk_off..rk_off.strict_add(8)]);

    add_round_constant_bit(&mut rkeys[rk_off..rk_off.strict_add(8)], rcon);
    xor_columns(&mut rkeys, rk_off, 16, ror_distance(1, 3));
    rcon = rcon.strict_add(1);

    if rcon == 7 {
      break;
    }

    memshift32(&mut rkeys, rk_off);
    rk_off = rk_off.strict_add(8);

    sub_bytes(&mut rkeys[rk_off..rk_off.strict_add(8)]);
    sub_bytes_nots(&mut rkeys[rk_off..rk_off.strict_add(8)]);

    xor_columns(&mut rkeys, rk_off, 16, ror_distance(0, 3));
  }

  let mut i = 8usize;
  while i < 104 {
    inv_shift_rows_1(&mut rkeys[i..i.strict_add(8)]);
    inv_shift_rows_2(&mut rkeys[i.strict_add(8)..i.strict_add(16)]);
    inv_shift_rows_3(&mut rkeys[i.strict_add(16)..i.strict_add(24)]);
    i = i.strict_add(32);
  }
  inv_shift_rows_1(&mut rkeys[104..112]);

  i = 1;
  while i < 15 {
    sub_bytes_nots(&mut rkeys[i.strict_mul(8)..i.strict_mul(8).strict_add(8)]);
    i = i.strict_add(1);
  }

  rkeys
}

/// Bitsliced implementation of the AES S-box by Boyar, Peralta and Calik.
fn sub_bytes(state: &mut [u64]) {
  debug_assert_eq!(state.len(), 8);

  let u7 = state[0];
  let u6 = state[1];
  let u5 = state[2];
  let u4 = state[3];
  let u3 = state[4];
  let u2 = state[5];
  let u1 = state[6];
  let u0 = state[7];

  let y14 = u3 ^ u5;
  let y13 = u0 ^ u6;
  let y12 = y13 ^ y14;
  let t1 = u4 ^ y12;
  let y15 = t1 ^ u5;
  let t2 = y12 & y15;
  let y6 = y15 ^ u7;
  let y20 = t1 ^ u1;
  let y9 = u0 ^ u3;
  let y11 = y20 ^ y9;
  let t12 = y9 & y11;
  let y7 = u7 ^ y11;
  let y8 = u0 ^ u5;
  let t0 = u1 ^ u2;
  let y10 = y15 ^ t0;
  let y17 = y10 ^ y11;
  let t13 = y14 & y17;
  let t14 = t13 ^ t12;
  let y19 = y10 ^ y8;
  let t15 = y8 & y10;
  let t16 = t15 ^ t12;
  let y16 = t0 ^ y11;
  let y21 = y13 ^ y16;
  let t7 = y13 & y16;
  let y18 = u0 ^ y16;
  let y1 = t0 ^ u7;
  let y4 = y1 ^ u3;
  let t5 = y4 & u7;
  let t6 = t5 ^ t2;
  let t18 = t6 ^ t16;
  let t22 = t18 ^ y19;
  let y2 = y1 ^ u0;
  let t10 = y2 & y7;
  let t11 = t10 ^ t7;
  let t20 = t11 ^ t16;
  let t24 = t20 ^ y18;
  let y5 = y1 ^ u6;
  let t8 = y5 & y1;
  let t9 = t8 ^ t7;
  let t19 = t9 ^ t14;
  let t23 = t19 ^ y21;
  let y3 = y5 ^ y8;
  let t3 = y3 & y6;
  let t4 = t3 ^ t2;
  let t17 = t4 ^ y20;
  let t21 = t17 ^ t14;
  let t26 = t21 & t23;
  let t27 = t24 ^ t26;
  let t31 = t22 ^ t26;
  let t25 = t21 ^ t22;
  let t28 = t25 & t27;
  let t29 = t28 ^ t22;
  let z14 = t29 & y2;
  let z5 = t29 & y7;
  let t30 = t23 ^ t24;
  let t32 = t31 & t30;
  let t33 = t32 ^ t24;
  let t35 = t27 ^ t33;
  let t36 = t24 & t35;
  let t38 = t27 ^ t36;
  let t39 = t29 & t38;
  let t40 = t25 ^ t39;
  let t43 = t29 ^ t40;
  let z3 = t43 & y16;
  let tc12 = z3 ^ z5;
  let z12 = t43 & y13;
  let z13 = t40 & y5;
  let z4 = t40 & y1;
  let tc6 = z3 ^ z4;
  let t34 = t23 ^ t33;
  let t37 = t36 ^ t34;
  let t41 = t40 ^ t37;
  let z8 = t41 & y10;
  let z17 = t41 & y8;
  let t44 = t33 ^ t37;
  let z0 = t44 & y15;
  let z9 = t44 & y12;
  let z10 = t37 & y3;
  let z1 = t37 & y6;
  let tc5 = z1 ^ z0;
  let tc11 = tc6 ^ tc5;
  let z11 = t33 & y4;
  let t42 = t29 ^ t33;
  let t45 = t42 ^ t41;
  let z7 = t45 & y17;
  let tc8 = z7 ^ tc6;
  let z16 = t45 & y14;
  let z6 = t42 & y11;
  let tc16 = z6 ^ tc8;
  let z15 = t42 & y9;
  let tc20 = z15 ^ tc16;
  let tc1 = z15 ^ z16;
  let tc2 = z10 ^ tc1;
  let tc21 = tc2 ^ z11;
  let tc3 = z9 ^ tc2;
  let s0 = tc3 ^ tc16;
  let s3 = tc3 ^ tc11;
  let s1 = s3 ^ tc16;
  let tc13 = z13 ^ tc1;
  let z2 = t33 & u7;
  let tc4 = z0 ^ z2;
  let tc7 = z12 ^ tc4;
  let tc9 = z8 ^ tc7;
  let tc10 = tc8 ^ tc9;
  let tc17 = z14 ^ tc10;
  let s5 = tc21 ^ tc17;
  let tc26 = tc17 ^ tc20;
  let s2 = tc26 ^ z17;
  let tc14 = tc4 ^ tc12;
  let tc18 = tc13 ^ tc14;
  let s6 = tc10 ^ tc18;
  let s7 = z12 ^ tc18;
  let s4 = tc14 ^ s3;

  state[0] = s7;
  state[1] = s6;
  state[2] = s5;
  state[3] = s4;
  state[4] = s3;
  state[5] = s2;
  state[6] = s1;
  state[7] = s0;
}

#[inline]
fn sub_bytes_nots(state: &mut [u64]) {
  debug_assert_eq!(state.len(), 8);
  state[0] ^= 0xffffffffffffffff;
  state[1] ^= 0xffffffffffffffff;
  state[5] ^= 0xffffffffffffffff;
  state[6] ^= 0xffffffffffffffff;
}

macro_rules! define_mix_columns {
  ($name:ident, $first_rotate:path, $second_rotate:path) => {
#[rustfmt::skip]
    fn $name(state: &mut State) {
      let (a0, a1, a2, a3, a4, a5, a6, a7) = (
        state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7]
      );
      let (b0, b1, b2, b3, b4, b5, b6, b7) = (
        $first_rotate(a0),
        $first_rotate(a1),
        $first_rotate(a2),
        $first_rotate(a3),
        $first_rotate(a4),
        $first_rotate(a5),
        $first_rotate(a6),
        $first_rotate(a7),
      );
      let (c0, c1, c2, c3, c4, c5, c6, c7) = (
        a0 ^ b0,
        a1 ^ b1,
        a2 ^ b2,
        a3 ^ b3,
        a4 ^ b4,
        a5 ^ b5,
        a6 ^ b6,
        a7 ^ b7,
      );
      state[0] = b0      ^ c7 ^ $second_rotate(c0);
      state[1] = b1 ^ c0 ^ c7 ^ $second_rotate(c1);
      state[2] = b2 ^ c1      ^ $second_rotate(c2);
      state[3] = b3 ^ c2 ^ c7 ^ $second_rotate(c3);
      state[4] = b4 ^ c3 ^ c7 ^ $second_rotate(c4);
      state[5] = b5 ^ c4      ^ $second_rotate(c5);
      state[6] = b6 ^ c5      ^ $second_rotate(c6);
      state[7] = b7 ^ c6      ^ $second_rotate(c7);
    }
  };
}

define_mix_columns!(mix_columns_0, rotate_rows_1, rotate_rows_2);
define_mix_columns!(mix_columns_1, rotate_rows_and_columns_1_1, rotate_rows_and_columns_2_2);
define_mix_columns!(mix_columns_2, rotate_rows_and_columns_1_2, rotate_rows_2);
define_mix_columns!(mix_columns_3, rotate_rows_and_columns_1_3, rotate_rows_and_columns_2_2);

#[inline]
fn delta_swap_1(a: &mut u64, shift: u32, mask: u64) {
  let t = (*a ^ ((*a) >> shift)) & mask;
  *a ^= t ^ (t << shift);
}

#[inline]
fn delta_swap_2(a: &mut u64, b: &mut u64, shift: u32, mask: u64) {
  let t = (*a ^ ((*b) >> shift)) & mask;
  *a ^= t;
  *b ^= t << shift;
}

#[inline]
fn shift_rows_1(state: &mut [u64]) {
  debug_assert_eq!(state.len(), 8);
  for x in state {
    delta_swap_1(x, 8, 0x00f000ff000f0000);
    delta_swap_1(x, 4, 0x0f0f00000f0f0000);
  }
}

#[inline]
fn shift_rows_2(state: &mut [u64]) {
  debug_assert_eq!(state.len(), 8);
  for x in state {
    delta_swap_1(x, 8, 0x00ff000000ff0000);
  }
}

#[inline]
fn shift_rows_3(state: &mut [u64]) {
  debug_assert_eq!(state.len(), 8);
  for x in state {
    delta_swap_1(x, 8, 0x000f00ff00f00000);
    delta_swap_1(x, 4, 0x0f0f00000f0f0000);
  }
}

#[inline(always)]
fn inv_shift_rows_1(state: &mut [u64]) {
  shift_rows_3(state);
}

#[inline(always)]
fn inv_shift_rows_2(state: &mut [u64]) {
  shift_rows_2(state);
}

#[inline(always)]
fn inv_shift_rows_3(state: &mut [u64]) {
  shift_rows_1(state);
}

fn xor_columns(rkeys: &mut [u64], offset: usize, idx_xor: usize, idx_ror: u32) {
  let mut i = 0usize;
  while i < 8 {
    let off_i = offset.strict_add(i);
    let rk = rkeys[off_i.strict_sub(idx_xor)] ^ (0x000f000f000f000f & ror(rkeys[off_i], idx_ror));
    rkeys[off_i] =
      rk ^ (0xfff0fff0fff0fff0 & (rk << 4)) ^ (0xff00ff00ff00ff00 & (rk << 8)) ^ (0xf000f000f000f000 & (rk << 12));
    i = i.strict_add(1);
  }
}

fn bitslice(output: &mut [u64], input0: &[u8], input1: &[u8], input2: &[u8], input3: &[u8]) {
  debug_assert_eq!(output.len(), 8);
  debug_assert_eq!(input0.len(), 16);
  debug_assert_eq!(input1.len(), 16);
  debug_assert_eq!(input2.len(), 16);
  debug_assert_eq!(input3.len(), 16);

  #[rustfmt::skip]
  fn read_reordered(input: &[u8]) -> u64 {
    (u64::from(input[0x0])        ) |
    (u64::from(input[0x1]) << 0x10) |
    (u64::from(input[0x2]) << 0x20) |
    (u64::from(input[0x3]) << 0x30) |
    (u64::from(input[0x8]) << 0x08) |
    (u64::from(input[0x9]) << 0x18) |
    (u64::from(input[0xa]) << 0x28) |
    (u64::from(input[0xb]) << 0x38)
  }

  let mut t0 = read_reordered(&input0[0x00..0x0c]);
  let mut t4 = read_reordered(&input0[0x04..0x10]);
  let mut t1 = read_reordered(&input1[0x00..0x0c]);
  let mut t5 = read_reordered(&input1[0x04..0x10]);
  let mut t2 = read_reordered(&input2[0x00..0x0c]);
  let mut t6 = read_reordered(&input2[0x04..0x10]);
  let mut t3 = read_reordered(&input3[0x00..0x0c]);
  let mut t7 = read_reordered(&input3[0x04..0x10]);

  let m0 = 0x5555555555555555;
  delta_swap_2(&mut t1, &mut t0, 1, m0);
  delta_swap_2(&mut t3, &mut t2, 1, m0);
  delta_swap_2(&mut t5, &mut t4, 1, m0);
  delta_swap_2(&mut t7, &mut t6, 1, m0);

  let m1 = 0x3333333333333333;
  delta_swap_2(&mut t2, &mut t0, 2, m1);
  delta_swap_2(&mut t3, &mut t1, 2, m1);
  delta_swap_2(&mut t6, &mut t4, 2, m1);
  delta_swap_2(&mut t7, &mut t5, 2, m1);

  let m2 = 0x0f0f0f0f0f0f0f0f;
  delta_swap_2(&mut t4, &mut t0, 4, m2);
  delta_swap_2(&mut t5, &mut t1, 4, m2);
  delta_swap_2(&mut t6, &mut t2, 4, m2);
  delta_swap_2(&mut t7, &mut t3, 4, m2);

  output[0] = t0;
  output[1] = t1;
  output[2] = t2;
  output[3] = t3;
  output[4] = t4;
  output[5] = t5;
  output[6] = t6;
  output[7] = t7;
}

fn inv_bitslice(input: &[u64]) -> [[u8; BLOCK_SIZE]; 4] {
  debug_assert_eq!(input.len(), 8);

  let mut t0 = input[0];
  let mut t1 = input[1];
  let mut t2 = input[2];
  let mut t3 = input[3];
  let mut t4 = input[4];
  let mut t5 = input[5];
  let mut t6 = input[6];
  let mut t7 = input[7];

  let m0 = 0x5555555555555555;
  delta_swap_2(&mut t1, &mut t0, 1, m0);
  delta_swap_2(&mut t3, &mut t2, 1, m0);
  delta_swap_2(&mut t5, &mut t4, 1, m0);
  delta_swap_2(&mut t7, &mut t6, 1, m0);

  let m1 = 0x3333333333333333;
  delta_swap_2(&mut t2, &mut t0, 2, m1);
  delta_swap_2(&mut t3, &mut t1, 2, m1);
  delta_swap_2(&mut t6, &mut t4, 2, m1);
  delta_swap_2(&mut t7, &mut t5, 2, m1);

  let m2 = 0x0f0f0f0f0f0f0f0f;
  delta_swap_2(&mut t4, &mut t0, 4, m2);
  delta_swap_2(&mut t5, &mut t1, 4, m2);
  delta_swap_2(&mut t6, &mut t2, 4, m2);
  delta_swap_2(&mut t7, &mut t3, 4, m2);

  #[rustfmt::skip]
  fn write_reordered(columns: u64, output: &mut [u8]) {
    output[0x0] = (columns        ) as u8;
    output[0x1] = (columns >> 0x10) as u8;
    output[0x2] = (columns >> 0x20) as u8;
    output[0x3] = (columns >> 0x30) as u8;
    output[0x8] = (columns >> 0x08) as u8;
    output[0x9] = (columns >> 0x18) as u8;
    output[0xa] = (columns >> 0x28) as u8;
    output[0xb] = (columns >> 0x38) as u8;
  }

  let mut output = [[0u8; BLOCK_SIZE]; 4];
  write_reordered(t0, &mut output[0][0x00..0x0c]);
  write_reordered(t4, &mut output[0][0x04..0x10]);
  write_reordered(t1, &mut output[1][0x00..0x0c]);
  write_reordered(t5, &mut output[1][0x04..0x10]);
  write_reordered(t2, &mut output[2][0x00..0x0c]);
  write_reordered(t6, &mut output[2][0x04..0x10]);
  write_reordered(t3, &mut output[3][0x00..0x0c]);
  write_reordered(t7, &mut output[3][0x04..0x10]);
  output
}

fn memshift32(buffer: &mut [u64], src_offset: usize) {
  debug_assert_eq!(src_offset % 8, 0);

  let dst_offset = src_offset.strict_add(8);
  debug_assert!(dst_offset.strict_add(8) <= buffer.len());

  let mut i = 8usize;
  while i > 0 {
    i = i.strict_sub(1);
    buffer[dst_offset.strict_add(i)] = buffer[src_offset.strict_add(i)];
  }
}

#[inline]
fn add_round_key(state: &mut State, rkey: &[u64]) {
  debug_assert_eq!(rkey.len(), 8);
  let mut i = 0usize;
  while i < 8 {
    state[i] ^= rkey[i];
    i = i.strict_add(1);
  }
}

#[inline(always)]
fn add_round_constant_bit(state: &mut [u64], bit: usize) {
  state[bit] ^= 0x00000000f0000000;
}

#[inline(always)]
fn xor_block(dst: &mut [u8; BLOCK_SIZE], src: &[u8; BLOCK_SIZE]) {
  let mut i = 0usize;
  while i < BLOCK_SIZE {
    dst[i] ^= src[i];
    i = i.strict_add(1);
  }
}

#[inline(always)]
fn ror(x: u64, y: u32) -> u64 {
  x.rotate_right(y)
}

#[inline(always)]
fn ror_distance(rows: u32, cols: u32) -> u32 {
  (rows << 4) + (cols << 2)
}

#[inline(always)]
fn rotate_rows_1(x: u64) -> u64 {
  ror(x, ror_distance(1, 0))
}

#[inline(always)]
fn rotate_rows_2(x: u64) -> u64 {
  ror(x, ror_distance(2, 0))
}

#[inline(always)]
#[rustfmt::skip]
fn rotate_rows_and_columns_1_1(x: u64) -> u64 {
  (ror(x, ror_distance(1, 1)) & 0x0fff0fff0fff0fff) |
  (ror(x, ror_distance(0, 1)) & 0xf000f000f000f000)
}

#[inline(always)]
#[rustfmt::skip]
fn rotate_rows_and_columns_1_2(x: u64) -> u64 {
  (ror(x, ror_distance(1, 2)) & 0x00ff00ff00ff00ff) |
  (ror(x, ror_distance(0, 2)) & 0xff00ff00ff00ff00)
}

#[inline(always)]
#[rustfmt::skip]
fn rotate_rows_and_columns_1_3(x: u64) -> u64 {
  (ror(x, ror_distance(1, 3)) & 0x000f000f000f000f) |
  (ror(x, ror_distance(0, 3)) & 0xfff0fff0fff0fff0)
}

#[inline(always)]
#[rustfmt::skip]
fn rotate_rows_and_columns_2_2(x: u64) -> u64 {
  (ror(x, ror_distance(2, 2)) & 0x00ff00ff00ff00ff) |
  (ror(x, ror_distance(1, 2)) & 0xff00ff00ff00ff00)
}
