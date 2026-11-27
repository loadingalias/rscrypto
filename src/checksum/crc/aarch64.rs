//! AArch64 carryless-multiply CRC engine (PMULL, and PMULL with EOR3).
//!
//! One engine serves every CRC width and both bit orders; see the parent module for the 64-bit
//! domain. The EOR3 tier merges each fold's three-way XOR into one `EOR3`.

use core::arch::aarch64::{
  uint8x16_t, uint64x2_t, vandq_u8, vcombine_u64, vcreate_u64, veor3q_u64, veorq_u64, vgetq_lane_u64, vld1q_u8,
  vmull_high_p64, vmull_p64, vqtbl1q_u8, vreinterpretq_p64_u64, vreinterpretq_u64_p128, vreinterpretq_u64_u8,
};

use super::{
  Algo, Hw, Keys, MASK, Word,
  dispatch::{Kernel, Tier, fold_with},
  shift_by,
};

/// Lanes in the main loop: eight 16-byte lanes, 128 bytes per iteration.
const LANES: usize = 8;

/// Indices that reverse the bytes of a vector.
const REVERSE: [u8; 16] = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];

/// # Safety
///
/// The CPU must support NEON.
#[inline]
#[target_feature(enable = "neon")]
fn load(bytes: &[u8; 16]) -> uint8x16_t {
  // SAFETY: `bytes` is 16 readable bytes, and `vld1q_u8` needs no alignment.
  unsafe { vld1q_u8(bytes.as_ptr()) }
}

/// Puts a loaded block in the engine's bit order: unchanged when reflected, byte-reversed when
/// MSB-first.
///
/// # Safety
///
/// The CPU must support NEON.
#[inline]
#[target_feature(enable = "neon")]
fn lane<const REFLECT: bool>(v: uint8x16_t) -> uint64x2_t {
  if REFLECT {
    vreinterpretq_u64_u8(v)
  } else {
    vreinterpretq_u64_u8(vqtbl1q_u8(v, load(&REVERSE)))
  }
}

/// # Safety
///
/// The CPU must support NEON.
#[inline]
#[target_feature(enable = "neon")]
fn load_lane<const REFLECT: bool>(bytes: &[u8; 16]) -> uint64x2_t {
  lane::<REFLECT>(load(bytes))
}

/// # Safety
///
/// The CPU must support NEON.
#[inline]
#[target_feature(enable = "neon")]
fn pair(lo: u64, hi: u64) -> uint64x2_t {
  vcombine_u64(vcreate_u64(lo), vcreate_u64(hi))
}

/// # Safety
///
/// The CPU must support NEON.
#[inline]
#[target_feature(enable = "neon")]
fn key(k: &[u64; 2]) -> uint64x2_t {
  pair(k[0], k[1])
}

/// # Safety
///
/// The CPU must support NEON.
#[inline]
#[target_feature(enable = "neon")]
fn lo(v: uint64x2_t) -> u64 {
  vgetq_lane_u64::<0>(v)
}

/// # Safety
///
/// The CPU must support NEON.
#[inline]
#[target_feature(enable = "neon")]
fn hi(v: uint64x2_t) -> u64 {
  vgetq_lane_u64::<1>(v)
}

/// `clmul(a, b)` as a 128-bit lane.
///
/// # Safety
///
/// The CPU must support NEON, AES (PMULL).
#[inline]
#[target_feature(enable = "neon,aes")]
fn clmul(a: u64, b: u64) -> uint64x2_t {
  vreinterpretq_u64_p128(vmull_p64(a, b))
}

/// The two products of a fold: `clmul(x.lo, k.lo)` and `clmul(x.hi, k.hi)`.
///
/// # Safety
///
/// The CPU must support NEON, AES (PMULL).
#[inline]
#[target_feature(enable = "neon,aes")]
fn products(x: uint64x2_t, k: uint64x2_t) -> (uint64x2_t, uint64x2_t) {
  let high = vreinterpretq_u64_p128(vmull_high_p64(vreinterpretq_p64_u64(x), vreinterpretq_p64_u64(k)));
  (clmul(lo(x), lo(k)), high)
}

/// Folds `x` by the distance of `k`.
///
/// # Safety
///
/// The CPU must support NEON, AES (PMULL).
#[inline]
#[target_feature(enable = "neon,aes")]
fn fold(x: uint64x2_t, k: uint64x2_t) -> uint64x2_t {
  let (low, high) = products(x, k);
  veorq_u64(low, high)
}

/// # Safety
///
/// The CPU must support NEON, SHA3.
#[inline]
#[target_feature(enable = "neon,sha3")]
fn eor3(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
  veor3q_u64(a, b, c)
}

/// Folds `x` by the distance of `k` and adds `data`.
///
/// # Safety
///
/// The CPU must support NEON and AES, and SHA3 when `EOR3`.
#[inline(always)]
unsafe fn fold_add<const EOR3: bool>(x: uint64x2_t, k: uint64x2_t, data: uint64x2_t) -> uint64x2_t {
  // SAFETY: the caller guarantees the target features.
  unsafe {
    let (low, high) = products(x, k);
    if EOR3 {
      eor3(low, high, data)
    } else {
      veorq_u64(veorq_u64(low, high), data)
    }
  }
}

/// Barrett reduction of a 128-bit product modulo `G'`.
///
/// # Safety
///
/// The CPU must support NEON, AES (PMULL).
#[inline]
#[target_feature(enable = "neon,aes")]
fn barrett<const REFLECT: bool>(y: uint64x2_t, keys: &Keys) -> u64 {
  if REFLECT {
    let q = lo(clmul(lo(y), keys.mu));
    hi(y) ^ hi(clmul(q, keys.poly)) ^ (q & keys.top)
  } else {
    let q = hi(y) ^ hi(clmul(hi(y), keys.mu));
    lo(y) ^ lo(clmul(q, keys.poly))
  }
}

/// Folds eight consecutive lanes into one: lane `j` moves by `7 - j` blocks.
///
/// # Safety
///
/// The CPU must support NEON and AES, and SHA3 when `EOR3`.
#[inline(always)]
unsafe fn merge8<const EOR3: bool>(acc: &[uint64x2_t; LANES], keys: &Keys) -> uint64x2_t {
  // SAFETY: the caller guarantees the target features.
  unsafe {
    // Pairwise sums keep the dependency chains short.
    let a = fold_add::<EOR3>(acc[0], key(&keys.fold[6]), fold(acc[1], key(&keys.fold[5])));
    let b = fold_add::<EOR3>(acc[2], key(&keys.fold[4]), fold(acc[3], key(&keys.fold[3])));
    let c = fold_add::<EOR3>(acc[4], key(&keys.fold[2]), fold(acc[5], key(&keys.fold[1])));
    let d = fold_add::<EOR3>(acc[6], key(&keys.fold[0]), acc[7]);
    veorq_u64(veorq_u64(a, b), veorq_u64(c, d))
  }
}

/// Updates a 64-bit-domain CRC state with at least 16 bytes.
///
/// # Safety
///
/// The CPU must support NEON and AES, and SHA3 when `EOR3`. `data` must hold at least 16 bytes.
#[inline(always)]
unsafe fn engine<const REFLECT: bool, const EOR3: bool>(state: u64, data: &[u8], keys: &Keys) -> u64 {
  // SAFETY: the caller guarantees the target features; every load reads a slice element.
  unsafe {
    let (blocks, tail) = data.as_chunks::<16>();
    let inject = if REFLECT { pair(state, 0) } else { pair(0, state) };
    let (mut x, rest) = if let Some((first, rest)) = blocks.split_first_chunk::<LANES>() {
      let mut acc = first.map(|block| load_lane::<REFLECT>(&block));
      acc[0] = veorq_u64(acc[0], inject);
      let k = key(&keys.fold[LANES.strict_sub(1)]);
      let (groups, rest) = rest.as_chunks::<LANES>();
      for group in groups {
        for (lane, block) in acc.iter_mut().zip(group) {
          *lane = fold_add::<EOR3>(*lane, k, load_lane::<REFLECT>(block));
        }
      }
      (merge8::<EOR3>(&acc, keys), rest)
    } else {
      let (first, rest) = blocks.split_at(1);
      (veorq_u64(load_lane::<REFLECT>(&first[0]), inject), rest)
    };
    // `x` precedes the remaining blocks; each moves by its distance from the last block.
    if let Some((last, middle)) = rest.split_last() {
      let n = rest.len();
      let mut sum = fold_add::<EOR3>(x, key(&keys.fold[n.strict_sub(1)]), load_lane::<REFLECT>(last));
      for (j, block) in middle.iter().enumerate() {
        sum = fold_add::<EOR3>(
          load_lane::<REFLECT>(block),
          key(&keys.fold[n.strict_sub(2).strict_sub(j)]),
          sum,
        );
      }
      x = sum;
    }
    if !tail.is_empty() {
      let r = tail.len();
      let last = &data[data.len().strict_sub(16)..].as_chunks::<16>().0[0];
      let masked = vandq_u8(load(last), load(&MASK[r..].as_chunks::<16>().0[0]));
      x = fold_add::<EOR3>(x, key(&keys.tail[r]), lane::<REFLECT>(masked));
    }
    barrett::<REFLECT>(fold(x, key(&keys.tail[8])), keys)
  }
}

/// PMULL tier.
///
/// # Safety
///
/// The CPU must support NEON and AES (PMULL), and `data` must hold at least 16 bytes.
#[target_feature(enable = "neon,aes")]
pub(in crate::checksum) unsafe fn pmull<const REFLECT: bool>(state: u64, data: &[u8], keys: &Keys) -> u64 {
  // SAFETY: this function enables the features, and the caller guarantees the length.
  unsafe { engine::<REFLECT, false>(state, data, keys) }
}

/// PMULL tier with EOR3.
///
/// # Safety
///
/// The CPU must support NEON, AES (PMULL), and SHA3 (EOR3), and `data` must hold at least 16
/// bytes.
#[target_feature(enable = "neon,aes,sha3")]
pub(in crate::checksum) unsafe fn pmull_eor3<const REFLECT: bool>(state: u64, data: &[u8], keys: &Keys) -> u64 {
  // SAFETY: this function enables the features, and the caller guarantees the length.
  unsafe { engine::<REFLECT, true>(state, data, keys) }
}

/// CRC-32 or CRC-32C of eight bytes with the CRC extension.
///
/// # Safety
///
/// The CPU must support the CRC extension.
#[inline]
#[target_feature(enable = "crc")]
fn crc32_word<const CASTAGNOLI: bool>(crc: u32, word: &[u8; 8]) -> u32 {
  use core::arch::aarch64::{__crc32cd, __crc32d};
  let word = u64::from_le_bytes(*word);
  if CASTAGNOLI {
    __crc32cd(crc, word)
  } else {
    __crc32d(crc, word)
  }
}

/// CRC-32 or CRC-32C with the CRC extension, eight bytes per instruction.
///
/// # Safety
///
/// The CPU must support the CRC extension.
#[inline]
#[target_feature(enable = "crc")]
fn crc32_hw<const CASTAGNOLI: bool>(mut crc: u32, data: &[u8]) -> u32 {
  use core::arch::aarch64::{__crc32b, __crc32cb};
  let (words, tail) = data.as_chunks::<8>();
  for word in words {
    crc = crc32_word::<CASTAGNOLI>(crc, word);
  }
  for &byte in tail {
    crc = if CASTAGNOLI {
      __crc32cb(crc, byte)
    } else {
      __crc32b(crc, byte)
    };
  }
  crc
}

/// Bytes per fused step: 128 bytes of PMULL lanes and 16 bytes on each of three CRC streams.
const FUSED_STEP: usize = 176;

/// Fused steps in the shortest fused round.
const FUSED_MIN_STEPS: usize = 8;

/// Inputs below this length run the CRC instructions alone.
const HW_ONLY_MAX: usize = 127;

/// One fused round over `n * FUSED_STEP` bytes, `n` a power of two.
///
/// The PMULL lanes take the first `128 * n` bytes, and three CRC-instruction streams take the next
/// three `16 * n`-byte parts from a zero register. Linearity joins them:
/// `crc = s_v * x^(48n) ^ s_1 * x^(32n) ^ s_2 * x^(16n) ^ s_3`, where each multiplier is a combine
/// power because `16n` is a power of two.
///
/// # Safety
///
/// The CPU must support NEON, AES, and CRC, and SHA3 when `EOR3`.
#[inline(always)]
unsafe fn fused_round<const CASTAGNOLI: bool, const EOR3: bool>(
  state: u64,
  round: &[u8],
  n: usize,
  keys: &Keys,
  powers: &[u64; 64],
) -> u64 {
  // SAFETY: the caller guarantees the target features; every load reads a slice element.
  unsafe {
    let (vector, scalar) = round.split_at(n.strict_mul(128));
    let (groups, _) = vector.as_chunks::<16>().0.as_chunks::<LANES>();
    let (words, _) = scalar.as_chunks::<8>();
    let (one, rest) = words.split_at(n.strict_mul(2));
    let (two, three) = rest.split_at(n.strict_mul(2));
    assert!(groups.len() == n && three.len() == n.strict_mul(2));
    let mut acc = groups[0].map(|block| load_lane::<true>(&block));
    acc[0] = veorq_u64(acc[0], pair(state, 0));
    let mut c1 = crc32_word::<CASTAGNOLI>(crc32_word::<CASTAGNOLI>(0, &one[0]), &one[1]);
    let mut c2 = crc32_word::<CASTAGNOLI>(crc32_word::<CASTAGNOLI>(0, &two[0]), &two[1]);
    let mut c3 = crc32_word::<CASTAGNOLI>(crc32_word::<CASTAGNOLI>(0, &three[0]), &three[1]);
    let k = key(&keys.fold[LANES.strict_sub(1)]);
    let mut i = 1usize;
    while i < n {
      let group = &groups[i];
      for (lane, block) in acc.iter_mut().zip(group) {
        *lane = fold_add::<EOR3>(*lane, k, load_lane::<true>(block));
      }
      let w = i.strict_mul(2);
      c1 = crc32_word::<CASTAGNOLI>(crc32_word::<CASTAGNOLI>(c1, &one[w]), &one[w.strict_add(1)]);
      c2 = crc32_word::<CASTAGNOLI>(crc32_word::<CASTAGNOLI>(c2, &two[w]), &two[w.strict_add(1)]);
      c3 = crc32_word::<CASTAGNOLI>(crc32_word::<CASTAGNOLI>(c3, &three[w]), &three[w.strict_add(1)]);
      i = i.strict_add(1);
    }
    // The CRC of the 16 bytes of a lane from a zero register is the lane times x^32 mod P, which
    // is the register after the vector part.
    let x = merge8::<EOR3>(&acc, keys);
    let vector = crc32_word::<CASTAGNOLI>(crc32_word::<CASTAGNOLI>(0, &lo(x).to_le_bytes()), &hi(x).to_le_bytes());
    let j = n.trailing_zeros() as usize;
    let (p16, p32) = (powers[j.strict_add(4)], powers[j.strict_add(5)]);
    let multiply = |value: u64, power: u64| barrett::<true>(clmul(value, power), keys);
    multiply(multiply(u64::from(vector), p32), p16)
      ^ multiply(u64::from(c1), p32)
      ^ multiply(u64::from(c2), p16)
      ^ u64::from(c3)
  }
}

/// CRC-32 or CRC-32C: CRC instructions for short inputs, fused rounds for long ones, and PMULL
/// for the rest.
///
/// # Safety
///
/// The CPU must support NEON, AES, and CRC, and SHA3 when `EOR3`.
#[inline(always)]
unsafe fn fused<const CASTAGNOLI: bool, const EOR3: bool>(
  mut state: u64,
  mut data: &[u8],
  keys: &Keys,
  powers: &[u64; 64],
) -> u64 {
  // SAFETY: the caller guarantees the target features.
  unsafe {
    if data.len() <= HW_ONLY_MAX {
      return u64::from(crc32_hw::<CASTAGNOLI>(super::low32(state), data));
    }
    loop {
      let steps = data.len().strict_div(FUSED_STEP);
      if steps < FUSED_MIN_STEPS {
        break;
      }
      let n = 1usize << steps.ilog2();
      let (round, rest) = data.split_at(n.strict_mul(FUSED_STEP));
      state = fused_round::<CASTAGNOLI, EOR3>(state, round, n, keys, powers);
      data = rest;
    }
    if data.len() < 16 {
      return u64::from(crc32_hw::<CASTAGNOLI>(super::low32(state), data));
    }
    engine::<true, EOR3>(state, data, keys)
  }
}

/// PMULL tier for CRC-32 and CRC-32C, fused with the CRC extension.
///
/// # Safety
///
/// The CPU must support NEON, AES (PMULL), and CRC.
#[target_feature(enable = "neon,aes,crc")]
pub(in crate::checksum) unsafe fn pmull_crc<const CASTAGNOLI: bool>(
  state: u64,
  data: &[u8],
  keys: &Keys,
  powers: &[u64; 64],
) -> u64 {
  // SAFETY: this function enables the features.
  unsafe { fused::<CASTAGNOLI, false>(state, data, keys, powers) }
}

/// PMULL with EOR3 tier for CRC-32 and CRC-32C, fused with the CRC extension.
///
/// # Safety
///
/// The CPU must support NEON, AES (PMULL), SHA3 (EOR3), and CRC.
#[target_feature(enable = "neon,aes,sha3,crc")]
pub(in crate::checksum) unsafe fn pmull_eor3_crc<const CASTAGNOLI: bool>(
  state: u64,
  data: &[u8],
  keys: &Keys,
  powers: &[u64; 64],
) -> u64 {
  // SAFETY: this function enables the features.
  unsafe { fused::<CASTAGNOLI, true>(state, data, keys, powers) }
}

/// The 32-bit reflected register of a CRC-32 or CRC-32C word.
#[inline(always)]
fn crc32_register<A: Algo>(register: A::W) -> u32 {
  super::low32(register.widen())
}

/// Hwcrc tier. The selector runs it only on a CPU with the CRC extension, for CRC-32 and CRC-32C.
fn hwcrc_kernel<A: Algo>(register: A::W, data: &[u8]) -> A::W {
  let crc = crc32_register::<A>(register);
  // SAFETY: the selector checks the CRC extension and the CRC's instruction support.
  let crc = unsafe {
    if A::HW == Hw::Crc32c {
      crc32_hw::<true>(crc, data)
    } else {
      crc32_hw::<false>(crc, data)
    }
  };
  A::W::narrow(u64::from(crc))
}

/// A fused tier for CRC-32 and CRC-32C. The selector runs it only on a CPU with PMULL and CRC,
/// and SHA3 when `EOR3`.
fn fused_kernel<A: Algo, const EOR3: bool>(register: A::W, data: &[u8]) -> A::W {
  let crc = A::CRC;
  let state = u64::from(crc32_register::<A>(register));
  // SAFETY: the selector checks the features and the CRC's instruction support.
  let state = unsafe {
    match (A::HW == Hw::Crc32c, EOR3) {
      (true, true) => pmull_eor3_crc::<true>(state, data, &crc.keys, &crc.powers),
      (false, true) => pmull_eor3_crc::<false>(state, data, &crc.keys, &crc.powers),
      (true, false) => pmull_crc::<true>(state, data, &crc.keys, &crc.powers),
      (false, false) => pmull_crc::<false>(state, data, &crc.keys, &crc.powers),
    }
  };
  A::W::narrow(state)
}

/// PMULL tier.
fn pmull_kernel<A: Algo>(register: A::W, data: &[u8]) -> A::W {
  // SAFETY: the selector checks PMULL.
  unsafe { fold_with::<A>(register, data, pmull::<true>, pmull::<false>) }
}

/// PMULL with EOR3 tier.
fn pmull_eor3_kernel<A: Algo>(register: A::W, data: &[u8]) -> A::W {
  // SAFETY: the selector checks PMULL and SHA3.
  unsafe { fold_with::<A>(register, data, pmull_eor3::<true>, pmull_eor3::<false>) }
}

/// The kernel of an AArch64 tier for CRC `A`. The caller has checked that the CPU runs `tier`.
pub(in crate::checksum) fn kernel<A: Algo>(tier: Tier) -> Kernel<A::W> {
  let fuse = A::HW != Hw::None && crate::platform::caps().has(crate::platform::caps::aarch64::CRC_READY);
  match (tier, fuse) {
    (Tier::Hwcrc, _) => hwcrc_kernel::<A>,
    (Tier::PmullEor3, true) => fused_kernel::<A, true>,
    (Tier::PmullEor3, false) => pmull_eor3_kernel::<A>,
    (_, true) => fused_kernel::<A, false>,
    (_, false) => pmull_kernel::<A>,
  }
}

/// Multiplies an engine-domain value by `x^(8 * len)`, for `combine`.
///
/// # Safety
///
/// The CPU must support NEON and AES (PMULL).
#[target_feature(enable = "neon,aes")]
pub(in crate::checksum) unsafe fn shift<const REFLECT: bool>(
  value: u64,
  len: usize,
  powers: &[u64; 64],
  keys: &Keys,
) -> u64 {
  shift_by(value, len, powers, |value, power| {
    barrett::<REFLECT>(clmul(value, power), keys)
  })
}

/// `combine` multiplier of the PMULL tiers. The selector runs it only on a CPU with PMULL.
pub(in crate::checksum) fn shift_kernel<A: Algo>(value: u64, len: usize) -> u64 {
  let crc = A::CRC;
  // SAFETY: the selector checks PMULL.
  unsafe {
    if crc.spec.reflected {
      shift::<true>(value, len, &crc.powers, &crc.keys)
    } else {
      shift::<false>(value, len, &crc.powers, &crc.keys)
    }
  }
}
