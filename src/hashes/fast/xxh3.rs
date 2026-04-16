//! XXH3 (**NOT CRYPTO**).
//!
//! Hardware-accelerated on x86-64 (AVX2, AVX-512), aarch64 (NEON), POWER
//! (VSX), s390x (z/Vector), and WASM (SIMD128), with a portable scalar
//! fallback.

#![allow(clippy::indexing_slicing)] // Tight block parsing + fixed-size arrays

use core::mem;

use crate::traits::FastHash;

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64_neon;
#[doc(hidden)]
pub(crate) mod dispatch;
#[doc(hidden)]
pub(crate) mod dispatch_tables;
pub(crate) mod kernels;
#[cfg(all(target_arch = "powerpc64", target_endian = "little"))]
pub(crate) mod power;
#[cfg(target_arch = "riscv64")]
pub(crate) mod riscv64_v;
#[cfg(target_arch = "s390x")]
pub(crate) mod s390x;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64_avx2;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64_avx512;

#[derive(Clone, Debug, Default)]
pub struct Xxh3_64;

#[derive(Clone, Debug, Default)]
pub struct Xxh3_128;

// xxh32 primes (used in scramble/mix)
const PRIME32_1: u32 = 0x9E37_79B1;
const PRIME32_2: u32 = 0x85EB_CA77;
const PRIME32_3: u32 = 0xC2B2_AE3D;

// xxh64 primes
const PRIME64_1: u64 = 0x9E37_79B1_85EB_CA87;
const PRIME64_2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const PRIME64_3: u64 = 0x1656_67B1_9E37_79F9;
const PRIME64_4: u64 = 0x85EB_CA77_C2B2_AE63;
const PRIME64_5: u64 = 0x27D4_EB2F_1656_67C5;

// XXH3 constants
const STRIPE_LEN: usize = 64;
const SECRET_CONSUME_RATE: usize = 8;
const ACC_NB: usize = STRIPE_LEN / mem::size_of::<u64>();

const SECRET_MERGEACCS_START: usize = 11;
const SECRET_LASTACC_START: usize = 7; // not 8-aligned; last secret differs from acc & scrambler

const MID_SIZE_MAX: usize = 240;
const SECRET_SIZE_MIN: usize = 136;
const DEFAULT_SECRET_SIZE: usize = 192;

const DEFAULT_SECRET: [u8; DEFAULT_SECRET_SIZE] = [
  0xb8, 0xfe, 0x6c, 0x39, 0x23, 0xa4, 0x4b, 0xbe, 0x7c, 0x01, 0x81, 0x2c, 0xf7, 0x21, 0xad, 0x1c, 0xde, 0xd4, 0x6d,
  0xe9, 0x83, 0x90, 0x97, 0xdb, 0x72, 0x40, 0xa4, 0xa4, 0xb7, 0xb3, 0x67, 0x1f, 0xcb, 0x79, 0xe6, 0x4e, 0xcc, 0xc0,
  0xe5, 0x78, 0x82, 0x5a, 0xd0, 0x7d, 0xcc, 0xff, 0x72, 0x21, 0xb8, 0x08, 0x46, 0x74, 0xf7, 0x43, 0x24, 0x8e, 0xe0,
  0x35, 0x90, 0xe6, 0x81, 0x3a, 0x26, 0x4c, 0x3c, 0x28, 0x52, 0xbb, 0x91, 0xc3, 0x00, 0xcb, 0x88, 0xd0, 0x65, 0x8b,
  0x1b, 0x53, 0x2e, 0xa3, 0x71, 0x64, 0x48, 0x97, 0xa2, 0x0d, 0xf9, 0x4e, 0x38, 0x19, 0xef, 0x46, 0xa9, 0xde, 0xac,
  0xd8, 0xa8, 0xfa, 0x76, 0x3f, 0xe3, 0x9c, 0x34, 0x3f, 0xf9, 0xdc, 0xbb, 0xc7, 0xc7, 0x0b, 0x4f, 0x1d, 0x8a, 0x51,
  0xe0, 0x4b, 0xcd, 0xb4, 0x59, 0x31, 0xc8, 0x9f, 0x7e, 0xc9, 0xd9, 0x78, 0x73, 0x64, 0xea, 0xc5, 0xac, 0x83, 0x34,
  0xd3, 0xeb, 0xc3, 0xc5, 0x81, 0xa0, 0xff, 0xfa, 0x13, 0x63, 0xeb, 0x17, 0x0d, 0xdd, 0x51, 0xb7, 0xf0, 0xda, 0x49,
  0xd3, 0x16, 0x55, 0x26, 0x29, 0xd4, 0x68, 0x9e, 0x2b, 0x16, 0xbe, 0x58, 0x7d, 0x47, 0xa1, 0xfc, 0x8f, 0xf8, 0xb8,
  0xd1, 0x7a, 0xd0, 0x31, 0xce, 0x45, 0xcb, 0x3a, 0x8f, 0x95, 0x16, 0x04, 0x28, 0xaf, 0xd7, 0xfb, 0xca, 0xbb, 0x4b,
  0x40, 0x7e,
];

const INITIAL_ACC: [u64; ACC_NB] = [
  PRIME32_3 as u64,
  PRIME64_1,
  PRIME64_2,
  PRIME64_3,
  PRIME64_4,
  PRIME32_2 as u64,
  PRIME64_5,
  PRIME32_1 as u64,
];

/// # Safety
///
/// Caller must ensure `offset + 4 <= input.len()`.
#[inline(always)]
unsafe fn read_u32_le(input: &[u8], offset: usize) -> u32 {
  debug_assert!(offset + 4 <= input.len());
  // SAFETY: caller ensures `offset + 4 <= input.len()`, and `read_unaligned` supports unaligned
  // loads.
  let v = unsafe { core::ptr::read_unaligned(input.as_ptr().add(offset) as *const u32) };
  u32::from_le(v)
}

/// # Safety
///
/// Caller must ensure `offset + 8 <= input.len()`.
#[inline(always)]
unsafe fn read_u64_le(input: &[u8], offset: usize) -> u64 {
  debug_assert!(offset + 8 <= input.len());
  // SAFETY: caller ensures `offset + 8 <= input.len()`, and `read_unaligned` supports unaligned
  // loads.
  let v = unsafe { core::ptr::read_unaligned(input.as_ptr().add(offset) as *const u64) };
  u64::from_le(v)
}

/// Extract a pair of 8-byte arrays at `offset` from a byte slice.
///
/// Matches xxhash-rust's `get_aligned_chunk_ref::<[[u8; 8]; 2]>` pattern
/// for optimal codegen: `from_ne_bytes(chunk[0])` compiles to a single `ldr`.
///
/// # Safety
///
/// Caller must ensure `offset + 16 <= data.len()`.
#[inline(always)]
unsafe fn chunk16(data: &[u8], offset: usize) -> &[[u8; 8]; 2] {
  debug_assert!(offset + 16 <= data.len());
  // SAFETY: caller ensures bounds. The resulting reference is valid for 16 bytes.
  unsafe { &*(data.as_ptr().add(offset) as *const [[u8; 8]; 2]) }
}

/// Extract two consecutive 16-byte chunks (32 bytes) at `offset`.
///
/// Used by `mix32_b` to pass both secret halves as a single reference,
/// matching xxhash-rust's `get_aligned_chunk_ref::<[[[u8; 8]; 2]; 2]>`.
///
/// # Safety
///
/// Caller must ensure `offset + 32 <= data.len()`.
#[inline(always)]
unsafe fn chunk32(data: &[u8], offset: usize) -> &[[[u8; 8]; 2]; 2] {
  debug_assert!(offset + 32 <= data.len());
  // SAFETY: caller ensures bounds. The resulting reference is valid for 32 bytes.
  unsafe { &*(data.as_ptr().add(offset) as *const [[[u8; 8]; 2]; 2]) }
}

#[inline(always)]
const fn mult32_to64(left: u32, right: u32) -> u64 {
  (left as u64).wrapping_mul(right as u64)
}

#[inline(always)]
const fn xorshift64(value: u64, shift: u64) -> u64 {
  value ^ (value >> shift)
}

#[inline(always)]
const fn xxh3_avalanche(mut value: u64) -> u64 {
  value = xorshift64(value, 37);
  value = value.wrapping_mul(0x1656_6791_9E37_79F9);
  xorshift64(value, 32)
}

#[inline(always)]
const fn strong_avalanche(mut value: u64, len: u64) -> u64 {
  value ^= value.rotate_left(49) ^ value.rotate_left(24);
  value = value.wrapping_mul(0x9FB2_1C65_1E98_DF25);
  value ^= (value >> 35).wrapping_add(len);
  value = value.wrapping_mul(0x9FB2_1C65_1E98_DF25);
  xorshift64(value, 28)
}

#[inline(always)]
const fn mul64_to128(left: u64, right: u64) -> (u64, u64) {
  let product = (left as u128).wrapping_mul(right as u128);
  (product as u64, (product >> 64) as u64)
}

#[inline(always)]
const fn mul128_fold64(left: u64, right: u64) -> u64 {
  let (low, high) = mul64_to128(left, right);
  low ^ high
}

#[inline(always)]
const fn xxh64_avalanche(mut input: u64) -> u64 {
  input ^= input >> 33;
  input = input.wrapping_mul(PRIME64_2);
  input ^= input >> 29;
  input = input.wrapping_mul(PRIME64_3);
  input ^= input >> 32;
  input
}

// ---------------------------------------------------------------------------
#[inline(always)]
fn mix16_b(data: &[[u8; 8]; 2], secret: &[[u8; 8]; 2], seed: u64) -> u64 {
  let input_lo = u64::from_ne_bytes(data[0]).to_le();
  let input_hi = u64::from_ne_bytes(data[1]).to_le();
  let secret_lo = u64::from_ne_bytes(secret[0]).to_le();
  let secret_hi = u64::from_ne_bytes(secret[1]).to_le();

  mul128_fold64(
    input_lo ^ secret_lo.wrapping_add(seed),
    input_hi ^ secret_hi.wrapping_sub(seed),
  )
}

#[inline(always)]
fn mix32_b(
  lo: &mut u64,
  hi: &mut u64,
  data_1: &[[u8; 8]; 2],
  data_2: &[[u8; 8]; 2],
  secret: &[[[u8; 8]; 2]; 2],
  seed: u64,
) {
  *lo = lo.wrapping_add(mix16_b(data_1, &secret[0], seed));
  *lo ^= u64::from_ne_bytes(data_2[0])
    .to_le()
    .wrapping_add(u64::from_ne_bytes(data_2[1]).to_le());

  *hi = hi.wrapping_add(mix16_b(data_2, &secret[1], seed));
  *hi ^= u64::from_ne_bytes(data_1[0])
    .to_le()
    .wrapping_add(u64::from_ne_bytes(data_1[1]).to_le());
}

#[inline(always)]
fn xxh3_64_9to16(input: &[u8], seed: u64, secret: &[u8]) -> u64 {
  // SAFETY: input.len() is 9..=16, secret.len() >= SECRET_SIZE_MIN (136).
  unsafe {
    let flip1 = (read_u64_le(secret, 24) ^ read_u64_le(secret, 32)).wrapping_add(seed);
    let flip2 = (read_u64_le(secret, 40) ^ read_u64_le(secret, 48)).wrapping_sub(seed);

    let input_lo = read_u64_le(input, 0) ^ flip1;
    let input_hi = read_u64_le(input, input.len() - 8) ^ flip2;

    let acc = (input.len() as u64)
      .wrapping_add(input_lo.swap_bytes())
      .wrapping_add(input_hi)
      .wrapping_add(mul128_fold64(input_lo, input_hi));

    xxh3_avalanche(acc)
  }
}

#[inline(always)]
fn xxh3_64_4to8(input: &[u8], mut seed: u64, secret: &[u8]) -> u64 {
  seed ^= ((seed as u32).swap_bytes() as u64) << 32;

  // SAFETY: input.len() is 4..=8, secret.len() >= SECRET_SIZE_MIN (136).
  unsafe {
    let input1 = read_u32_le(input, 0);
    let input2 = read_u32_le(input, input.len() - 4);

    let flip = (read_u64_le(secret, 8) ^ read_u64_le(secret, 16)).wrapping_sub(seed);
    let input64 = (input2 as u64).wrapping_add((input1 as u64) << 32);
    let keyed = input64 ^ flip;

    strong_avalanche(keyed, input.len() as u64)
  }
}

#[inline(always)]
fn xxh3_64_1to3(input: &[u8], seed: u64, secret: &[u8]) -> u64 {
  let combo = ((input[0] as u32) << 16)
    | ((input[input.len() >> 1] as u32) << 24)
    | (input[input.len() - 1] as u32)
    | ((input.len() as u32) << 8);

  // SAFETY: secret.len() >= SECRET_SIZE_MIN (136), so offsets 0 and 4 are in bounds.
  let flip = unsafe { ((read_u32_le(secret, 0) ^ read_u32_le(secret, 4)) as u64).wrapping_add(seed) };
  xxh64_avalanche((combo as u64) ^ flip)
}

#[inline(always)]
fn xxh3_64_0to16(input: &[u8], seed: u64, secret: &[u8]) -> u64 {
  if input.len() > 8 {
    xxh3_64_9to16(input, seed, secret)
  } else if input.len() >= 4 {
    xxh3_64_4to8(input, seed, secret)
  } else if !input.is_empty() {
    xxh3_64_1to3(input, seed, secret)
  } else {
    // SAFETY: secret.len() >= SECRET_SIZE_MIN (136), so offsets 56 and 64 are in bounds.
    unsafe { xxh64_avalanche(seed ^ read_u64_le(secret, 56) ^ read_u64_le(secret, 64)) }
  }
}

#[inline(always)]
fn xxh3_64_7to128(input: &[u8], seed: u64, secret: &[u8]) -> u64 {
  // SAFETY: callers ensure input.len() >= 17 and secret.len() >= SECRET_SIZE_MIN (136).
  // All chunk16 offsets are within bounds for those guarantees.
  unsafe {
    let mut acc = (input.len() as u64).wrapping_mul(PRIME64_1);

    if input.len() > 32 {
      if input.len() > 64 {
        if input.len() > 96 {
          acc = acc.wrapping_add(mix16_b(chunk16(input, 48), chunk16(secret, 96), seed));
          acc = acc.wrapping_add(mix16_b(chunk16(input, input.len() - 64), chunk16(secret, 112), seed));
        }

        acc = acc.wrapping_add(mix16_b(chunk16(input, 32), chunk16(secret, 64), seed));
        acc = acc.wrapping_add(mix16_b(chunk16(input, input.len() - 48), chunk16(secret, 80), seed));
      }

      acc = acc.wrapping_add(mix16_b(chunk16(input, 16), chunk16(secret, 32), seed));
      acc = acc.wrapping_add(mix16_b(chunk16(input, input.len() - 32), chunk16(secret, 48), seed));
    }

    acc = acc.wrapping_add(mix16_b(chunk16(input, 0), chunk16(secret, 0), seed));
    acc = acc.wrapping_add(mix16_b(chunk16(input, input.len() - 16), chunk16(secret, 16), seed));

    xxh3_avalanche(acc)
  }
}

#[inline(never)]
fn xxh3_64_129to240(input: &[u8], seed: u64, secret: &[u8]) -> u64 {
  const START_OFFSET: usize = 3;
  const LAST_OFFSET: usize = 17;

  // SAFETY: input.len() is 129..=240 and secret.len() >= SECRET_SIZE_MIN (136).
  // All chunk16 offsets are within bounds.
  unsafe {
    let mut acc = (input.len() as u64).wrapping_mul(PRIME64_1);
    let nb_rounds = input.len() / 16;

    let mut idx = 0usize;
    while idx < 8 {
      acc = acc.wrapping_add(mix16_b(chunk16(input, 16 * idx), chunk16(secret, 16 * idx), seed));
      idx += 1;
    }
    acc = xxh3_avalanche(acc);

    while idx < nb_rounds {
      acc = acc.wrapping_add(mix16_b(
        chunk16(input, 16 * idx),
        chunk16(secret, 16 * (idx - 8) + START_OFFSET),
        seed,
      ));
      idx += 1;
    }

    acc = acc.wrapping_add(mix16_b(
      chunk16(input, input.len() - 16),
      chunk16(secret, SECRET_SIZE_MIN - LAST_OFFSET),
      seed,
    ));

    xxh3_avalanche(acc)
  }
}

#[inline(always)]
fn mix_two_accs(acc: &[u64], acc_offset: usize, secret: &[u8], secret_offset: usize) -> u64 {
  // SAFETY: callers ensure secret_offset+16 <= secret.len().
  unsafe {
    mul128_fold64(
      acc[acc_offset] ^ read_u64_le(secret, secret_offset),
      acc[acc_offset + 1] ^ read_u64_le(secret, secret_offset + 8),
    )
  }
}

#[inline(always)]
fn merge_accs(acc: &[u64], secret: &[u8], secret_offset: usize, mut result: u64) -> u64 {
  let mut idx = 0usize;
  while idx < 4 {
    result = result.wrapping_add(mix_two_accs(acc, idx * 2, secret, secret_offset + idx * 16));
    idx += 1;
  }

  xxh3_avalanche(result)
}

#[inline(always)]
fn scramble_acc(mut acc: [u64; ACC_NB], secret: &[u8], secret_offset: usize) -> [u64; ACC_NB] {
  let secret_stripe = &secret[secret_offset..secret_offset + STRIPE_LEN];
  let (secret_chunks, _) = secret_stripe.as_chunks::<8>();

  let mut idx = 0usize;
  while idx < ACC_NB {
    let key = u64::from_le_bytes(secret_chunks[idx]);
    let mut acc_val = xorshift64(acc[idx], 47);
    acc_val ^= key;
    acc[idx] = acc_val.wrapping_mul(PRIME32_1 as u64);

    idx += 1;
  }

  acc
}

#[inline(always)]
fn accumulate_512(mut acc: [u64; ACC_NB], stripe: &[u8], secret_stripe: &[u8]) -> [u64; ACC_NB] {
  debug_assert_eq!(stripe.len(), STRIPE_LEN);
  debug_assert_eq!(secret_stripe.len(), STRIPE_LEN);
  let (data_chunks, _) = stripe.as_chunks::<8>();
  let (secret_chunks, _) = secret_stripe.as_chunks::<8>();

  let mut idx = 0usize;
  while idx < ACC_NB {
    let data_val = u64::from_le_bytes(data_chunks[idx]);
    let data_key = data_val ^ u64::from_le_bytes(secret_chunks[idx]);

    acc[idx ^ 1] = acc[idx ^ 1].wrapping_add(data_val);
    acc[idx] = acc[idx].wrapping_add(mult32_to64((data_key & 0xFFFF_FFFF) as u32, (data_key >> 32) as u32));

    idx += 1;
  }

  acc
}

#[inline(always)]
fn accumulate_loop(
  mut acc: [u64; ACC_NB],
  input: &[u8],
  input_offset: usize,
  secret: &[u8],
  secret_offset: usize,
  nb_stripes: usize,
) -> [u64; ACC_NB] {
  let mut stripe_offset = input_offset;
  let mut secret_stripe_offset = secret_offset;
  let input_len = input.len();
  let secret_len = secret.len();

  let mut idx = 0usize;
  while idx < nb_stripes {
    debug_assert!(stripe_offset + STRIPE_LEN <= input_len);
    debug_assert!(secret_stripe_offset + STRIPE_LEN <= secret_len);
    acc = accumulate_512(
      acc,
      &input[stripe_offset..stripe_offset + STRIPE_LEN],
      &secret[secret_stripe_offset..secret_stripe_offset + STRIPE_LEN],
    );
    stripe_offset += STRIPE_LEN;
    secret_stripe_offset += SECRET_CONSUME_RATE;
    idx += 1;
  }

  acc
}

#[inline(never)]
fn hash_long_internal_loop(input: &[u8], secret: &[u8]) -> [u64; ACC_NB] {
  let mut acc = INITIAL_ACC;

  let nb_stripes = (secret.len() - STRIPE_LEN) / SECRET_CONSUME_RATE;
  let block_len = STRIPE_LEN * nb_stripes;
  let nb_blocks = (input.len() - 1) / block_len;

  let mut idx = 0usize;
  while idx < nb_blocks {
    acc = accumulate_loop(acc, input, idx * block_len, secret, 0, nb_stripes);
    acc = scramble_acc(acc, secret, secret.len() - STRIPE_LEN);
    idx += 1;
  }

  let nb_stripes = ((input.len() - 1) - (block_len * nb_blocks)) / STRIPE_LEN;
  acc = accumulate_loop(acc, input, nb_blocks * block_len, secret, 0, nb_stripes);

  accumulate_512(
    acc,
    &input[input.len() - STRIPE_LEN..],
    &secret[secret.len() - STRIPE_LEN - SECRET_LASTACC_START..secret.len() - SECRET_LASTACC_START],
  )
}

#[inline(never)]
fn xxh3_64_long_impl(input: &[u8], secret: &[u8]) -> u64 {
  let acc = hash_long_internal_loop(input, secret);
  merge_accs(
    &acc,
    secret,
    SECRET_MERGEACCS_START,
    (input.len() as u64).wrapping_mul(PRIME64_1),
  )
}

#[inline(never)]
fn custom_default_secret(seed: u64) -> [u8; DEFAULT_SECRET_SIZE] {
  if seed == 0 {
    return DEFAULT_SECRET;
  }

  let mut result = [0u8; DEFAULT_SECRET_SIZE];
  const NB_ROUNDS: usize = DEFAULT_SECRET_SIZE / 16;

  let mut idx = 0usize;
  while idx < NB_ROUNDS {
    // SAFETY: idx < NB_ROUNDS = DEFAULT_SECRET_SIZE/16, so idx*16+16 <= DEFAULT_SECRET_SIZE.
    let lo = unsafe { read_u64_le(&DEFAULT_SECRET, idx * 16).wrapping_add(seed).to_le_bytes() };
    // SAFETY: idx < NB_ROUNDS = DEFAULT_SECRET_SIZE/16, so idx*16+8+8 <= DEFAULT_SECRET_SIZE.
    let hi = unsafe {
      read_u64_le(&DEFAULT_SECRET, idx * 16 + 8)
        .wrapping_sub(seed)
        .to_le_bytes()
    };

    result[idx * 16..idx * 16 + 8].copy_from_slice(&lo);
    result[idx * 16 + 8..idx * 16 + 16].copy_from_slice(&hi);

    idx += 1;
  }

  result
}

/// Long-path entry point (>240B) — no ≤240B branches.
///
/// Called from compile-time dispatch when the caller already knows `input.len() > MID_SIZE_MAX`.
pub(crate) fn xxh3_64_long(input: &[u8], seed: u64) -> u64 {
  if seed == 0 {
    xxh3_64_long_impl(input, &DEFAULT_SECRET)
  } else {
    let secret = custom_default_secret(seed);
    xxh3_64_long_impl(input, &secret)
  }
}

#[cfg(any(test, feature = "diag"))]
#[inline(always)]
fn xxh3_64_with_seed(input: &[u8], seed: u64) -> u64 {
  if input.len() <= 16 {
    xxh3_64_0to16(input, seed, &DEFAULT_SECRET)
  } else if input.len() <= 128 {
    xxh3_64_7to128(input, seed, &DEFAULT_SECRET)
  } else if input.len() <= MID_SIZE_MAX {
    xxh3_64_129to240(input, seed, &DEFAULT_SECRET)
  } else {
    xxh3_64_long(input, seed)
  }
}

#[inline(always)]
fn xxh3_128_1to3(input: &[u8], seed: u64, secret: &[u8]) -> u128 {
  let c1 = input[0];
  let c2 = input[input.len() >> 1];
  let c3 = input[input.len() - 1];
  let input_lo = (c1 as u32) << 16 | (c2 as u32) << 24 | c3 as u32 | (input.len() as u32) << 8;
  let input_hi = input_lo.swap_bytes().rotate_left(13);

  // SAFETY: secret.len() >= SECRET_SIZE_MIN (136), so offsets 0..16 are in bounds.
  unsafe {
    let flip_lo = (read_u32_le(secret, 0) as u64 ^ read_u32_le(secret, 4) as u64).wrapping_add(seed);
    let flip_hi = (read_u32_le(secret, 8) as u64 ^ read_u32_le(secret, 12) as u64).wrapping_sub(seed);
    let keyed_lo = input_lo as u64 ^ flip_lo;
    let keyed_hi = input_hi as u64 ^ flip_hi;

    (xxh64_avalanche(keyed_lo) as u128) | ((xxh64_avalanche(keyed_hi) as u128) << 64)
  }
}

#[inline(always)]
fn xxh3_128_4to8(input: &[u8], mut seed: u64, secret: &[u8]) -> u128 {
  seed ^= ((seed as u32).swap_bytes() as u64) << 32;

  // SAFETY: input.len() is 4..=8, secret.len() >= SECRET_SIZE_MIN (136).
  unsafe {
    let lo = read_u32_le(input, 0);
    let hi = read_u32_le(input, input.len() - 4);
    let input_64 = (lo as u64).wrapping_add((hi as u64) << 32);

    let flip = (read_u64_le(secret, 16) ^ read_u64_le(secret, 24)).wrapping_add(seed);
    let keyed = input_64 ^ flip;

    let (mut lo, mut hi) = mul64_to128(keyed, PRIME64_1.wrapping_add((input.len() as u64) << 2));

    hi = hi.wrapping_add(lo << 1);
    lo ^= hi >> 3;

    lo = xorshift64(lo, 35).wrapping_mul(0x9FB2_1C65_1E98_DF25);
    lo = xorshift64(lo, 28);
    hi = xxh3_avalanche(hi);

    (lo as u128) | ((hi as u128) << 64)
  }
}

#[inline(always)]
fn xxh3_128_9to16(input: &[u8], seed: u64, secret: &[u8]) -> u128 {
  // SAFETY: input.len() is 9..=16, secret.len() >= SECRET_SIZE_MIN (136).
  unsafe {
    let flip_lo = (read_u64_le(secret, 32) ^ read_u64_le(secret, 40)).wrapping_sub(seed);
    let flip_hi = (read_u64_le(secret, 48) ^ read_u64_le(secret, 56)).wrapping_add(seed);
    let input_lo = read_u64_le(input, 0);
    let mut input_hi = read_u64_le(input, input.len() - 8);

    let (mut mul_low, mut mul_high) = mul64_to128(input_lo ^ input_hi ^ flip_lo, PRIME64_1);

    mul_low = mul_low.wrapping_add(((input.len() as u64) - 1) << 54);
    input_hi ^= flip_hi;
    mul_high = mul_high.wrapping_add(input_hi.wrapping_add(mult32_to64(input_hi as u32, PRIME32_2 - 1)));

    mul_low ^= mul_high.swap_bytes();

    let (result_low, mut result_hi) = mul64_to128(mul_low, PRIME64_2);
    result_hi = result_hi.wrapping_add(mul_high.wrapping_mul(PRIME64_2));

    (xxh3_avalanche(result_low) as u128) | ((xxh3_avalanche(result_hi) as u128) << 64)
  }
}

#[inline(always)]
fn xxh3_128_0to16(input: &[u8], seed: u64, secret: &[u8]) -> u128 {
  if input.len() > 8 {
    xxh3_128_9to16(input, seed, secret)
  } else if input.len() >= 4 {
    xxh3_128_4to8(input, seed, secret)
  } else if !input.is_empty() {
    xxh3_128_1to3(input, seed, secret)
  } else {
    // SAFETY: secret.len() >= SECRET_SIZE_MIN (136), so offsets 64..96 are in bounds.
    unsafe {
      let flip_lo = read_u64_le(secret, 64) ^ read_u64_le(secret, 72);
      let flip_hi = read_u64_le(secret, 80) ^ read_u64_le(secret, 88);
      (xxh64_avalanche(seed ^ flip_lo) as u128) | ((xxh64_avalanche(seed ^ flip_hi) as u128) << 64)
    }
  }
}

#[inline(always)]
fn xxh3_128_7to128(input: &[u8], seed: u64, secret: &[u8]) -> u128 {
  // SAFETY: callers ensure input.len() >= 17 and secret.len() >= SECRET_SIZE_MIN (136).
  // All chunk16/chunk32 offsets are within bounds for those guarantees.
  unsafe {
    let mut lo = (input.len() as u64).wrapping_mul(PRIME64_1);
    let mut hi = 0u64;

    if input.len() > 32 {
      if input.len() > 64 {
        if input.len() > 96 {
          mix32_b(
            &mut lo,
            &mut hi,
            chunk16(input, 48),
            chunk16(input, input.len() - 64),
            chunk32(secret, 96),
            seed,
          );
        }

        mix32_b(
          &mut lo,
          &mut hi,
          chunk16(input, 32),
          chunk16(input, input.len() - 48),
          chunk32(secret, 64),
          seed,
        );
      }

      mix32_b(
        &mut lo,
        &mut hi,
        chunk16(input, 16),
        chunk16(input, input.len() - 32),
        chunk32(secret, 32),
        seed,
      );
    }

    mix32_b(
      &mut lo,
      &mut hi,
      chunk16(input, 0),
      chunk16(input, input.len() - 16),
      chunk32(secret, 0),
      seed,
    );

    let result_lo = lo.wrapping_add(hi);
    let result_hi = lo
      .wrapping_mul(PRIME64_1)
      .wrapping_add(hi.wrapping_mul(PRIME64_4))
      .wrapping_add(((input.len() as u64).wrapping_sub(seed)).wrapping_mul(PRIME64_2));

    (xxh3_avalanche(result_lo) as u128) | (((0u64.wrapping_sub(xxh3_avalanche(result_hi))) as u128) << 64)
  }
}

#[inline(never)]
fn xxh3_128_129to240(input: &[u8], seed: u64, secret: &[u8]) -> u128 {
  const START_OFFSET: usize = 3;
  const LAST_OFFSET: usize = 17;
  let nb_rounds = input.len() / 32;

  // SAFETY: input.len() is 129..=240 and secret.len() >= SECRET_SIZE_MIN (136).
  // All chunk16/chunk32 offsets are within bounds for those guarantees.
  unsafe {
    let mut lo = (input.len() as u64).wrapping_mul(PRIME64_1);
    let mut hi = 0u64;

    let mut idx = 0usize;
    while idx < 4 {
      mix32_b(
        &mut lo,
        &mut hi,
        chunk16(input, 32 * idx),
        chunk16(input, (32 * idx) + 16),
        chunk32(secret, 32 * idx),
        seed,
      );
      idx += 1;
    }

    lo = xxh3_avalanche(lo);
    hi = xxh3_avalanche(hi);

    while idx < nb_rounds {
      let sec_off = START_OFFSET.wrapping_add(32 * (idx - 4));
      mix32_b(
        &mut lo,
        &mut hi,
        chunk16(input, 32 * idx),
        chunk16(input, (32 * idx) + 16),
        chunk32(secret, sec_off),
        seed,
      );
      idx += 1;
    }

    mix32_b(
      &mut lo,
      &mut hi,
      chunk16(input, input.len() - 16),
      chunk16(input, input.len() - 32),
      chunk32(secret, SECRET_SIZE_MIN - LAST_OFFSET - 16),
      0u64.wrapping_sub(seed),
    );

    let result_lo = lo.wrapping_add(hi);
    let result_hi = lo
      .wrapping_mul(PRIME64_1)
      .wrapping_add(hi.wrapping_mul(PRIME64_4))
      .wrapping_add(((input.len() as u64).wrapping_sub(seed)).wrapping_mul(PRIME64_2));

    (xxh3_avalanche(result_lo) as u128) | (0u128.wrapping_sub(xxh3_avalanche(result_hi) as u128) << 64)
  }
}

#[inline(never)]
fn xxh3_128_long_impl(input: &[u8], secret: &[u8]) -> u128 {
  let acc = hash_long_internal_loop(input, secret);

  let lo = merge_accs(
    &acc,
    secret,
    SECRET_MERGEACCS_START,
    (input.len() as u64).wrapping_mul(PRIME64_1),
  );
  let hi = merge_accs(
    &acc,
    secret,
    secret.len() - (ACC_NB * mem::size_of::<u64>()) - SECRET_MERGEACCS_START,
    !(input.len() as u64).wrapping_mul(PRIME64_2),
  );

  (lo as u128) | ((hi as u128) << 64)
}

/// Long-path entry point (>240B) — no ≤240B branches.
pub(crate) fn xxh3_128_long(input: &[u8], seed: u64) -> u128 {
  if seed == 0 {
    xxh3_128_long_impl(input, &DEFAULT_SECRET)
  } else {
    let secret = custom_default_secret(seed);
    xxh3_128_long_impl(input, &secret)
  }
}

impl FastHash for Xxh3_64 {
  const OUTPUT_SIZE: usize = 8;
  type Output = u64;
  type Seed = u64;

  #[inline(always)]
  fn hash_with_seed(seed: Self::Seed, data: &[u8]) -> Self::Output {
    dispatch::hash64_with_seed(seed, data)
  }
}

impl FastHash for Xxh3_128 {
  const OUTPUT_SIZE: usize = 16;
  type Output = u128;
  type Seed = u64;

  #[inline(always)]
  fn hash_with_seed(seed: Self::Seed, data: &[u8]) -> Self::Output {
    dispatch::hash128_with_seed(seed, data)
  }
}

// ─── BuildHasher support ──────────────────────────────────────────────────

/// Streaming [`core::hash::Hasher`] backed by XXH3-64.
///
/// Created by [`Xxh3BuildHasher`]. Buffers input and computes the hash
/// on [`finish`](core::hash::Hasher::finish).
#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub struct Xxh3Hasher {
  buf: alloc::vec::Vec<u8>,
  seed: u64,
}

#[cfg(feature = "alloc")]
impl core::fmt::Debug for Xxh3Hasher {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("Xxh3Hasher")
      .field("seed", &self.seed)
      .field("buffered", &self.buf.len())
      .finish()
  }
}

#[cfg(feature = "alloc")]
impl core::hash::Hasher for Xxh3Hasher {
  #[inline]
  fn write(&mut self, bytes: &[u8]) {
    self.buf.extend_from_slice(bytes);
  }

  #[inline]
  fn finish(&self) -> u64 {
    Xxh3_64::hash_with_seed(self.seed, &self.buf)
  }
}

/// [`BuildHasher`](core::hash::BuildHasher) producing [`Xxh3Hasher`] instances.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
///
/// use rscrypto::hashes::fast::xxh3::Xxh3BuildHasher;
///
/// let mut map: HashMap<&str, i32, Xxh3BuildHasher> = HashMap::with_hasher(Xxh3BuildHasher::new());
/// map.insert("hello", 42);
/// assert_eq!(map["hello"], 42);
/// ```
#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[derive(Clone, Debug)]
pub struct Xxh3BuildHasher {
  seed: u64,
}

#[cfg(feature = "alloc")]
impl Xxh3BuildHasher {
  /// Create a builder with the default seed (0).
  #[inline]
  #[must_use]
  pub const fn new() -> Self {
    Self { seed: 0 }
  }

  /// Create a builder with a custom seed.
  #[inline]
  #[must_use]
  pub const fn with_seed(seed: u64) -> Self {
    Self { seed }
  }
}

#[cfg(feature = "alloc")]
impl Default for Xxh3BuildHasher {
  #[inline]
  fn default() -> Self {
    Self::new()
  }
}

#[cfg(feature = "alloc")]
impl core::hash::BuildHasher for Xxh3BuildHasher {
  type Hasher = Xxh3Hasher;

  #[inline]
  fn build_hasher(&self) -> Self::Hasher {
    Xxh3Hasher {
      buf: alloc::vec::Vec::new(),
      seed: self.seed,
    }
  }
}

#[cfg(test)]
mod tests {

  use alloc::vec::Vec;

  use proptest::prelude::*;

  use super::{Xxh3_64, Xxh3_128};

  #[test]
  fn smoke_empty_matches_oracle() {
    assert_eq!(
      <Xxh3_64 as crate::traits::FastHash>::hash_with_seed(0, b""),
      xxhash_rust::xxh3::xxh3_64_with_seed(b"", 0)
    );
    assert_eq!(
      <Xxh3_128 as crate::traits::FastHash>::hash_with_seed(0, b""),
      xxhash_rust::xxh3::xxh3_128_with_seed(b"", 0)
    );
  }

  proptest! {
    #[test]
    fn xxh3_64_matches_oracle(seed in any::<u64>(), data in proptest::collection::vec(any::<u8>(), 0..2048)) {
      let ours = <Xxh3_64 as crate::traits::FastHash>::hash_with_seed(seed, &data);
      let theirs = xxhash_rust::xxh3::xxh3_64_with_seed(&data, seed);
      prop_assert_eq!(ours, theirs);
    }

    #[test]
    fn xxh3_128_matches_oracle(seed in any::<u64>(), data in proptest::collection::vec(any::<u8>(), 0..2048)) {
      let ours = <Xxh3_128 as crate::traits::FastHash>::hash_with_seed(seed, &data);
      let theirs = xxhash_rust::xxh3::xxh3_128_with_seed(&data, seed);
      prop_assert_eq!(ours, theirs);
    }
  }

  fn deterministic_bytes(len: usize) -> Vec<u8> {
    let mut out = alloc::vec![0u8; len];
    let mut x = 0x243f_6a88_85a3_08d3u64;
    for b in &mut out {
      x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
      *b = (x >> 56) as u8;
    }
    out
  }

  #[test]
  fn xxh3_long_paths_match_oracle() {
    let sizes = [0usize, 1, 2, 3, 4, 8, 16, 17, 128, 129, 240, 241, 1024, 4096, 65536];
    let seeds = [0u64, 1u64, 0x0123_4567_89ab_cdef];

    for &seed in &seeds {
      for &len in &sizes {
        let data = deterministic_bytes(len);
        assert_eq!(
          <Xxh3_64 as crate::traits::FastHash>::hash_with_seed(seed, &data),
          xxhash_rust::xxh3::xxh3_64_with_seed(&data, seed),
          "xxh3_64 mismatch (seed={seed}, len={len})"
        );
        assert_eq!(
          <Xxh3_128 as crate::traits::FastHash>::hash_with_seed(seed, &data),
          xxhash_rust::xxh3::xxh3_128_with_seed(&data, seed),
          "xxh3_128 mismatch (seed={seed}, len={len})"
        );
      }
    }
  }
}
