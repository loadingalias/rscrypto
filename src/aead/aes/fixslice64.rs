//! Table-free 64-bit scalar AES fallback.
//!
//! This is a focused adaptation of the RustCrypto AES 0.8.4 64-bit fixslice
//! backend, reduced to AES-128 and AES-256 encryption over four parallel blocks. The
//! original code is MIT OR Apache-2.0 and derives from Alexandre Adomnicai's
//! fixsliced AES implementation. It is the portable authority on RV64 and
//! s390x, where the byte-oriented arithmetic S-box is not a reliable
//! constant-time source representation after target-specific lowering.
//!
//! Reference: Adomnicai et al., "Fixslicing AES-like Ciphers",
//! <https://eprint.iacr.org/2020/1123.pdf>.

use super::{BLOCK_SIZE, KEY_SIZE, KEY_SIZE_128};

use crate::aead::aes_fixslice_round::{
  State, bitslice, define_mix_columns, delta_swap_1, inv_bitslice, mix_columns_0, ror, ror_distance, rotate_rows_2,
  shift_rows_1, sub_bytes, sub_bytes_nots,
};

pub(super) struct FixsliceRoundKeys {
  keys: [u64; 120],
}

impl FixsliceRoundKeys {
  #[inline]
  pub(super) fn new(key: &[u8; KEY_SIZE]) -> Self {
    Self {
      keys: aes256_key_schedule(key),
    }
  }

  #[inline]
  #[cfg(any(target_arch = "riscv64", target_arch = "s390x"))]
  pub(super) fn zeroize(&mut self) {
    // SAFETY: `[u64; 120]` is contiguous and valid to view as bytes for its
    // exact initialized size.
    crate::traits::ct::zeroize(unsafe {
      core::slice::from_raw_parts_mut(self.keys.as_mut_ptr().cast::<u8>(), self.keys.len().strict_mul(8))
    });
  }
}

/// Bitsliced AES-128 round keys (10 rounds → 11 round keys × 8 u64 per group).
pub(super) struct Fixslice128RoundKeys {
  keys: [u64; 88],
}

impl Fixslice128RoundKeys {
  #[inline]
  pub(super) fn new(key: &[u8; KEY_SIZE_128]) -> Self {
    Self {
      keys: aes128_key_schedule(key),
    }
  }

  #[inline]
  #[cfg(any(target_arch = "riscv64", target_arch = "s390x"))]
  pub(super) fn zeroize(&mut self) {
    // SAFETY: `[u64; 88]` is contiguous and valid to view as bytes for its
    // exact initialized size.
    crate::traits::ct::zeroize(unsafe {
      core::slice::from_raw_parts_mut(self.keys.as_mut_ptr().cast::<u8>(), self.keys.len().strict_mul(8))
    });
  }
}

#[inline]
pub(super) fn encrypt_block(rkeys: &FixsliceRoundKeys, block: &mut [u8; BLOCK_SIZE]) {
  let mut blocks = [*block; 4];
  encrypt_4blocks(rkeys, &mut blocks);
  *block = blocks[0];
}

#[inline]
pub(super) fn encrypt_4blocks(rkeys: &FixsliceRoundKeys, blocks: &mut [[u8; BLOCK_SIZE]; 4]) {
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
pub(super) fn encrypt_block_128(rkeys: &Fixslice128RoundKeys, block: &mut [u8; BLOCK_SIZE]) {
  let mut blocks = [*block; 4];
  encrypt_4blocks_128(rkeys, &mut blocks);
  *block = blocks[0];
}

#[inline]
pub(super) fn encrypt_4blocks_128(rkeys: &Fixslice128RoundKeys, blocks: &mut [[u8; BLOCK_SIZE]; 4]) {
  let mut state = State::default();
  bitslice(&mut state, &blocks[0], &blocks[1], &blocks[2], &blocks[3]);

  add_round_key(&mut state, &rkeys.keys[..8]);

  let mut rk_off = 8usize;
  loop {
    sub_bytes(&mut state);
    mix_columns_1(&mut state);
    add_round_key(&mut state, &rkeys.keys[rk_off..rk_off.strict_add(8)]);
    rk_off = rk_off.strict_add(8);

    if rk_off == 80 {
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
  add_round_key(&mut state, &rkeys.keys[80..]);

  *blocks = inv_bitslice(&state);
}

fn aes128_key_schedule(key: &[u8; KEY_SIZE_128]) -> [u64; 88] {
  let mut rkeys = [0u64; 88];

  bitslice(&mut rkeys[..8], key, key, key, key);

  let mut rk_off = 0usize;
  let mut rcon = 0usize;
  while rcon < 10 {
    memshift32(&mut rkeys, rk_off);
    rk_off = rk_off.strict_add(8);

    sub_bytes(&mut rkeys[rk_off..rk_off.strict_add(8)]);
    sub_bytes_nots(&mut rkeys[rk_off..rk_off.strict_add(8)]);

    if rcon < 8 {
      add_round_constant_bit(&mut rkeys[rk_off..rk_off.strict_add(8)], rcon);
    } else if rcon == 8 {
      // RCON byte for round 9 is 0x1b = bits 0, 1, 3, 4.
      add_round_constant_bit(&mut rkeys[rk_off..rk_off.strict_add(8)], 0);
      add_round_constant_bit(&mut rkeys[rk_off..rk_off.strict_add(8)], 1);
      add_round_constant_bit(&mut rkeys[rk_off..rk_off.strict_add(8)], 3);
      add_round_constant_bit(&mut rkeys[rk_off..rk_off.strict_add(8)], 4);
    } else {
      // RCON byte for round 10 is 0x36 = bits 1, 2, 4, 5.
      add_round_constant_bit(&mut rkeys[rk_off..rk_off.strict_add(8)], 1);
      add_round_constant_bit(&mut rkeys[rk_off..rk_off.strict_add(8)], 2);
      add_round_constant_bit(&mut rkeys[rk_off..rk_off.strict_add(8)], 4);
      add_round_constant_bit(&mut rkeys[rk_off..rk_off.strict_add(8)], 5);
    }

    xor_columns(&mut rkeys, rk_off, 8, ror_distance(1, 3));
    rcon = rcon.strict_add(1);
  }

  // Fold the cumulative ShiftRows rotation into the keys for rounds whose
  // mix_columns variant in the encrypt loop is non-zero. Pattern mirrors
  // [`aes256_key_schedule`] adapted to AES-128's 10-round count.
  let mut i = 8usize;
  while i < 72 {
    inv_shift_rows_1(&mut rkeys[i..i.strict_add(8)]);
    inv_shift_rows_2(&mut rkeys[i.strict_add(8)..i.strict_add(16)]);
    inv_shift_rows_3(&mut rkeys[i.strict_add(16)..i.strict_add(24)]);
    i = i.strict_add(32);
  }
  inv_shift_rows_1(&mut rkeys[72..80]);

  // Account for the NOTs absorbed by sub_bytes during the schedule above.
  i = 1;
  while i < 11 {
    sub_bytes_nots(&mut rkeys[i.strict_mul(8)..i.strict_mul(8).strict_add(8)]);
    i = i.strict_add(1);
  }

  rkeys
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

define_mix_columns!(mix_columns_1, rotate_rows_and_columns_1_1, rotate_rows_and_columns_2_2);
define_mix_columns!(mix_columns_2, rotate_rows_and_columns_1_2, rotate_rows_2);
define_mix_columns!(mix_columns_3, rotate_rows_and_columns_1_3, rotate_rows_and_columns_2_2);

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
