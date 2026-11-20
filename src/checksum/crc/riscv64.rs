//! RISC-V CRC engines: Zbc scalar carryless multiply, and Zvbc vector carryless multiply.
//!
//! One engine serves every CRC width and both bit orders; see the parent module for the 64-bit
//! domain. A lane is a pair of 64-bit registers. The Zvbc tier runs the main loop on eight lanes at
//! once, with the low halves of the lanes in one vector register group and the high halves in
//! another, and finishes with the scalar lane code.

use core::arch::asm;

use super::{Algo, Keys, MASK, dispatch::fold_with, shift_by};

/// A 128-bit lane.
#[derive(Clone, Copy)]
struct Lane {
  lo: u64,
  hi: u64,
}

impl core::ops::BitXor for Lane {
  type Output = Self;

  #[inline(always)]
  fn bitxor(self, other: Self) -> Self {
    Self {
      lo: self.lo ^ other.lo,
      hi: self.hi ^ other.hi,
    }
  }
}

/// A 64-bit carryless multiply with a 128-bit result.
trait Clmul {
  /// `a * b` as a lane.
  ///
  /// # Safety
  ///
  /// The CPU must support the instructions.
  unsafe fn clmul(a: u64, b: u64) -> Lane;
}

/// Zbc `CLMUL` and `CLMULH`.
struct Zbc;

impl Clmul for Zbc {
  #[inline]
  #[target_feature(enable = "zbc")]
  unsafe fn clmul(a: u64, b: u64) -> Lane {
    let (lo, hi): (u64, u64);
    // SAFETY: the caller guarantees Zbc; the instructions read and write registers only. `lo` is
    // written before `clmulh` reads `a` and `b`, so it must not share their register (`out`, not
    // `lateout`); `hi` is written last.
    unsafe {
      asm!(
        "clmul {lo}, {a}, {b}",
        "clmulh {hi}, {a}, {b}",
        lo = out(reg) lo,
        hi = lateout(reg) hi,
        a = in(reg) a,
        b = in(reg) b,
        options(nomem, nostack, pure)
      );
    }
    Lane { lo, hi }
  }
}

/// Zvbc `VCLMUL` and `VCLMULH` on one element, for CPUs without Zbc.
struct ZvbcScalar;

impl Clmul for ZvbcScalar {
  #[inline]
  #[target_feature(enable = "v,zvbc")]
  unsafe fn clmul(a: u64, b: u64) -> Lane {
    let (lo, hi): (u64, u64);
    // SAFETY: the caller guarantees V and Zvbc; the instructions use v1..v3 and no memory.
    unsafe {
      asm!(
        "vsetivli zero, 1, e64, m1, ta, ma",
        "vmv.s.x v1, {a}",
        "vclmul.vx v2, v1, {b}",
        "vclmulh.vx v3, v1, {b}",
        "vmv.x.s {lo}, v2",
        "vmv.x.s {hi}, v3",
        lo = lateout(reg) lo,
        hi = lateout(reg) hi,
        a = in(reg) a,
        b = in(reg) b,
        out("v1") _,
        out("v2") _,
        out("v3") _,
        options(nomem, nostack, pure)
      );
    }
    Lane { lo, hi }
  }
}

/// Lanes in the scalar main loop: four lanes, 64 bytes per iteration.
const LANES: usize = 4;

/// Lanes in the Zvbc main loop: eight lanes, 128 bytes per iteration.
const VECTOR_LANES: usize = 8;

/// A block as a lane in the engine's bit order: little-endian halves when reflected, the
/// big-endian value when MSB-first.
#[inline(always)]
fn lane<const REFLECT: bool>(bytes: &[u8; 16]) -> Lane {
  let (first, second) = bytes.split_at(8);
  let first: [u8; 8] = first.as_chunks::<8>().0[0];
  let second: [u8; 8] = second.as_chunks::<8>().0[0];
  if REFLECT {
    Lane {
      lo: u64::from_le_bytes(first),
      hi: u64::from_le_bytes(second),
    }
  } else {
    Lane {
      lo: u64::from_be_bytes(second),
      hi: u64::from_be_bytes(first),
    }
  }
}

/// Folds `x` by the distance of `k`: `clmul(x.lo, k[0]) ^ clmul(x.hi, k[1])`.
///
/// # Safety
///
/// The CPU must support `C`.
#[inline(always)]
unsafe fn fold<C: Clmul>(x: Lane, k: &[u64; 2]) -> Lane {
  // SAFETY: the caller guarantees the instructions.
  unsafe { C::clmul(x.lo, k[0]) ^ C::clmul(x.hi, k[1]) }
}

/// Barrett reduction of a 128-bit product modulo `G'`.
///
/// # Safety
///
/// The CPU must support `C`.
#[inline(always)]
unsafe fn barrett<C: Clmul, const REFLECT: bool>(y: Lane, keys: &Keys) -> u64 {
  // SAFETY: the caller guarantees the instructions.
  unsafe {
    if REFLECT {
      let q = C::clmul(y.lo, keys.mu).lo;
      y.hi ^ C::clmul(q, keys.poly).hi ^ (q & keys.top)
    } else {
      let q = y.hi ^ C::clmul(y.hi, keys.mu).hi;
      y.lo ^ C::clmul(q, keys.poly).lo
    }
  }
}

/// Folds consecutive lanes into one, then the remaining whole blocks, then the partial block, and
/// reduces to the 64-bit CRC state.
///
/// # Safety
///
/// The CPU must support `C`, `lanes` must not be empty, and `data` must hold at least 16 bytes.
#[inline(always)]
unsafe fn finish<C: Clmul, const REFLECT: bool>(lanes: &[Lane], rest: &[[u8; 16]], data: &[u8], keys: &Keys) -> u64 {
  // SAFETY: the caller guarantees the instructions.
  unsafe {
    // Every lane and block moves by its distance from the last block.
    let n = lanes.len().strict_add(rest.len());
    let mut x = Lane { lo: 0, hi: 0 };
    let blocks = lanes.iter().copied().chain(rest.iter().map(lane::<REFLECT>));
    for (j, block) in blocks.enumerate() {
      let distance = n.strict_sub(1).strict_sub(j);
      x = if distance == 0 {
        x ^ block
      } else {
        x ^ fold::<C>(block, &keys.fold[distance.strict_sub(1)])
      };
    }
    let r = data.len().strict_rem(16);
    if r != 0 {
      let mut last = data[data.len().strict_sub(16)..].as_chunks::<16>().0[0];
      for (byte, mask) in last.iter_mut().zip(&MASK[r..]) {
        *byte &= mask;
      }
      x = fold::<C>(x, &keys.tail[r]) ^ lane::<REFLECT>(&last);
    }
    barrett::<C, REFLECT>(fold::<C>(x, &keys.tail[8]), keys)
  }
}

/// Updates a 64-bit-domain CRC state with at least 16 bytes, on scalar lanes.
///
/// # Safety
///
/// The CPU must support `C`, and `data` must hold at least 16 bytes.
#[inline(always)]
unsafe fn engine<C: Clmul, const REFLECT: bool>(state: u64, data: &[u8], keys: &Keys) -> u64 {
  // SAFETY: the caller guarantees the instructions and the length.
  unsafe {
    let (blocks, _) = data.as_chunks::<16>();
    let inject = if REFLECT {
      Lane { lo: state, hi: 0 }
    } else {
      Lane { lo: 0, hi: state }
    };
    if let Some((first, rest)) = blocks.split_first_chunk::<LANES>() {
      let mut acc = first.map(|block| lane::<REFLECT>(&block));
      acc[0] = acc[0] ^ inject;
      let k = &keys.fold[LANES.strict_sub(1)];
      let (groups, rest) = rest.as_chunks::<LANES>();
      for group in groups {
        for (lane_acc, block) in acc.iter_mut().zip(group) {
          *lane_acc = fold::<C>(*lane_acc, k) ^ lane::<REFLECT>(block);
        }
      }
      finish::<C, REFLECT>(&acc, rest, data, keys)
    } else {
      let (first, rest) = blocks.split_at(1);
      finish::<C, REFLECT>(&[lane::<REFLECT>(&first[0]) ^ inject], rest, data, keys)
    }
  }
}

/// Zbc tier.
///
/// # Safety
///
/// The CPU must support Zbc, and `data` must hold at least 16 bytes.
#[target_feature(enable = "zbc")]
pub(in crate::checksum) unsafe fn zbc<const REFLECT: bool>(state: u64, data: &[u8], keys: &Keys) -> u64 {
  // SAFETY: this function enables Zbc, and the caller guarantees the length.
  unsafe { engine::<Zbc, REFLECT>(state, data, keys) }
}

/// Folds 128-byte groups of reflected blocks into eight lanes with Zvbc.
///
/// `lo` and `hi` hold the low and high halves of the eight lanes, which precede `groups`.
///
/// # Safety
///
/// The CPU must support V and Zvbc.
#[inline]
#[target_feature(enable = "v,zvbc")]
unsafe fn fold_groups(
  lo: &mut [u64; VECTOR_LANES],
  hi: &mut [u64; VECTOR_LANES],
  groups: &[[[u8; 16]; 8]],
  k: &[u64; 2],
) {
  if groups.is_empty() {
    return;
  }
  // SAFETY: the caller guarantees V and Zvbc. `vsetivli` selects eight 64-bit elements in groups
  // of four registers, which VLEN >= 128 provides. The strided loads read the low and high halves
  // of the eight blocks of each 128-byte group, all inside `groups`; the unit-stride accesses read
  // and write the eight elements of `lo` and `hi`. The asm touches no other memory.
  unsafe {
    asm!(
      "vsetivli zero, 8, e64, m4, ta, ma",
      "vle64.v v0, ({lo})",
      "vle64.v v4, ({hi})",
      "2:",
      "vlse64.v v8, ({data}), {stride}",
      "addi {tmp}, {data}, 8",
      "vlse64.v v12, ({tmp}), {stride}",
      "vclmul.vx v16, v0, {k0}",
      "vclmulh.vx v20, v0, {k0}",
      "vclmul.vx v24, v4, {k1}",
      "vclmulh.vx v28, v4, {k1}",
      "vxor.vv v0, v16, v24",
      "vxor.vv v0, v0, v8",
      "vxor.vv v4, v20, v28",
      "vxor.vv v4, v4, v12",
      "addi {data}, {data}, 128",
      "addi {count}, {count}, -1",
      "bnez {count}, 2b",
      "vse64.v v0, ({lo})",
      "vse64.v v4, ({hi})",
      lo = in(reg) lo.as_mut_ptr(),
      hi = in(reg) hi.as_mut_ptr(),
      data = inout(reg) groups.as_ptr() => _,
      count = inout(reg) groups.len() => _,
      stride = in(reg) 16usize,
      k0 = in(reg) k[0],
      k1 = in(reg) k[1],
      tmp = out(reg) _,
      out("v0") _, out("v1") _, out("v2") _, out("v3") _,
      out("v4") _, out("v5") _, out("v6") _, out("v7") _,
      out("v8") _, out("v9") _, out("v10") _, out("v11") _,
      out("v12") _, out("v13") _, out("v14") _, out("v15") _,
      out("v16") _, out("v17") _, out("v18") _, out("v19") _,
      out("v20") _, out("v21") _, out("v22") _, out("v23") _,
      out("v24") _, out("v25") _, out("v26") _, out("v27") _,
      out("v28") _, out("v29") _, out("v30") _, out("v31") _,
      options(nostack)
    );
  }
}

/// Zvbc tier: the eight-lane vector main loop for reflected CRCs, and scalar lanes otherwise.
///
/// # Safety
///
/// The CPU must support V and Zvbc, and `data` must hold at least 16 bytes.
#[target_feature(enable = "v,zvbc")]
pub(in crate::checksum) unsafe fn zvbc<const REFLECT: bool>(state: u64, data: &[u8], keys: &Keys) -> u64 {
  // SAFETY: this function enables V and Zvbc, and the caller guarantees the length.
  unsafe {
    let (blocks, _) = data.as_chunks::<16>();
    let Some((first, rest)) = blocks.split_first_chunk::<VECTOR_LANES>() else {
      return engine::<ZvbcScalar, REFLECT>(state, data, keys);
    };
    if !REFLECT {
      return engine::<ZvbcScalar, REFLECT>(state, data, keys);
    }
    let mut lo = [0u64; VECTOR_LANES];
    let mut hi = [0u64; VECTOR_LANES];
    for ((lo, hi), block) in lo.iter_mut().zip(hi.iter_mut()).zip(first) {
      let block = lane::<REFLECT>(block);
      *lo = block.lo;
      *hi = block.hi;
    }
    lo[0] ^= state;
    let (groups, rest) = rest.as_chunks::<VECTOR_LANES>();
    fold_groups(&mut lo, &mut hi, groups, &keys.fold[VECTOR_LANES.strict_sub(1)]);
    let lanes: [Lane; VECTOR_LANES] = core::array::from_fn(|i| Lane { lo: lo[i], hi: hi[i] });
    finish::<ZvbcScalar, REFLECT>(&lanes, rest, data, keys)
  }
}

/// Multiplies an engine-domain value by `x^(8 * len)`, for `combine`, with Zbc.
///
/// # Safety
///
/// The CPU must support Zbc.
#[target_feature(enable = "zbc")]
pub(in crate::checksum) unsafe fn shift_zbc<const REFLECT: bool>(
  value: u64,
  len: usize,
  powers: &[u64; 64],
  keys: &Keys,
) -> u64 {
  // SAFETY: this function enables Zbc.
  shift_by(value, len, powers, |value, power| unsafe {
    barrett::<Zbc, REFLECT>(Zbc::clmul(value, power), keys)
  })
}

/// Multiplies an engine-domain value by `x^(8 * len)`, for `combine`, with Zvbc.
///
/// # Safety
///
/// The CPU must support V and Zvbc.
#[target_feature(enable = "v,zvbc")]
pub(in crate::checksum) unsafe fn shift_zvbc<const REFLECT: bool>(
  value: u64,
  len: usize,
  powers: &[u64; 64],
  keys: &Keys,
) -> u64 {
  // SAFETY: this function enables V and Zvbc.
  shift_by(value, len, powers, |value, power| unsafe {
    barrett::<ZvbcScalar, REFLECT>(ZvbcScalar::clmul(value, power), keys)
  })
}

/// Zbc tier. The selector runs it only on a CPU with Zbc.
pub(in crate::checksum) fn zbc_kernel<A: Algo>(register: A::W, data: &[u8]) -> A::W {
  // SAFETY: the selector checks Zbc.
  unsafe { fold_with::<A>(register, data, zbc::<true>, zbc::<false>) }
}

/// Zvbc tier. The selector runs it only on a CPU with V and Zvbc.
pub(in crate::checksum) fn zvbc_kernel<A: Algo>(register: A::W, data: &[u8]) -> A::W {
  // SAFETY: the selector checks V and Zvbc.
  unsafe { fold_with::<A>(register, data, zvbc::<true>, zvbc::<false>) }
}

/// `combine` multiplier of the Zbc tier.
pub(in crate::checksum) fn shift_zbc_kernel<A: Algo>(value: u64, len: usize) -> u64 {
  let crc = A::CRC;
  // SAFETY: the selector checks Zbc.
  unsafe {
    if crc.spec.reflected {
      shift_zbc::<true>(value, len, &crc.powers, &crc.keys)
    } else {
      shift_zbc::<false>(value, len, &crc.powers, &crc.keys)
    }
  }
}

/// `combine` multiplier of the Zvbc tier.
pub(in crate::checksum) fn shift_zvbc_kernel<A: Algo>(value: u64, len: usize) -> u64 {
  let crc = A::CRC;
  // SAFETY: the selector checks V and Zvbc.
  unsafe {
    if crc.spec.reflected {
      shift_zvbc::<true>(value, len, &crc.powers, &crc.keys)
    } else {
      shift_zvbc::<false>(value, len, &crc.powers, &crc.keys)
    }
  }
}
