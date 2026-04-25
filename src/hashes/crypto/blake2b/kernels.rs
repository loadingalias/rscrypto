//! Blake2b portable compression function and kernel dispatch (RFC 7693).

use crate::platform::Caps;
#[cfg(target_arch = "aarch64")]
use crate::platform::caps::aarch64;
#[cfg(target_arch = "powerpc64")]
use crate::platform::caps::power;
#[cfg(target_arch = "riscv64")]
use crate::platform::caps::riscv;
#[cfg(target_arch = "s390x")]
use crate::platform::caps::s390x;
#[cfg(target_arch = "wasm32")]
use crate::platform::caps::wasm;
#[cfg(target_arch = "x86_64")]
use crate::platform::caps::x86;

/// Blake2b compress function pointer type.
///
/// Parameters: mutable state (8×u64), 128-byte message block, byte counter, finalization flag.
pub(crate) type CompressFn = fn(&mut [u64; 8], &[u8; 128], u128, bool);

/// Blake2b kernel identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
#[cfg_attr(target_os = "macos", allow(dead_code))]
pub enum Blake2bKernelId {
  Portable = 0,
  #[cfg(target_arch = "x86_64")]
  X86Avx2 = 1,
  #[cfg(target_arch = "x86_64")]
  X86Avx512vl = 2,
  #[cfg(target_arch = "aarch64")]
  Aarch64Neon = 3,
  #[cfg(target_arch = "s390x")]
  S390xVector = 4,
  #[cfg(target_arch = "powerpc64")]
  PowerVsx = 5,
  #[cfg(target_arch = "riscv64")]
  Riscv64V = 6,
  #[cfg(target_arch = "wasm32")]
  WasmSimd128 = 7,
}

impl Blake2bKernelId {
  #[cfg(any(test, feature = "diag"))]
  #[inline]
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Portable => "portable",
      #[cfg(target_arch = "x86_64")]
      Self::X86Avx2 => "x86/avx2",
      #[cfg(target_arch = "x86_64")]
      Self::X86Avx512vl => "x86/avx512vl",
      #[cfg(target_arch = "aarch64")]
      Self::Aarch64Neon => "aarch64/neon",
      #[cfg(target_arch = "s390x")]
      Self::S390xVector => "s390x/vector",
      #[cfg(target_arch = "powerpc64")]
      Self::PowerVsx => "power/vsx",
      #[cfg(target_arch = "riscv64")]
      Self::Riscv64V => "riscv64/v",
      #[cfg(target_arch = "wasm32")]
      Self::WasmSimd128 => "wasm/simd128",
    }
  }
}

// ─── Safe wrappers — dispatch validates caps before calling ──────────────────

#[cfg(target_arch = "x86_64")]
fn compress_x86_avx2(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
  // SAFETY: Only called when dispatch has verified AVX2 is available.
  unsafe { super::x86_64::compress_avx2(h, block, t, last) }
}

#[cfg(target_arch = "x86_64")]
fn compress_x86_avx512vl(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
  // SAFETY: Only called when dispatch has verified AVX-512F+VL is available.
  unsafe { super::x86_64::compress_avx512vl(h, block, t, last) }
}

#[cfg(target_arch = "aarch64")]
fn compress_aarch64_neon(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
  // SAFETY: NEON is baseline on AArch64.
  unsafe { super::aarch64::compress_neon(h, block, t, last) }
}

#[cfg(target_arch = "s390x")]
fn compress_s390x_vector(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
  // SAFETY: Only called when dispatch has verified vector facility is available.
  unsafe { super::s390x::compress_vector(h, block, t, last) }
}

#[cfg(target_arch = "powerpc64")]
fn compress_power_vsx(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
  // SAFETY: Only called when dispatch has verified VSX is available.
  unsafe { super::power::compress_vsx(h, block, t, last) }
}

#[cfg(target_arch = "riscv64")]
fn compress_riscv64_v(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
  // SAFETY: Only called when dispatch has verified V extension is available.
  unsafe { super::riscv64::compress_rvv(h, block, t, last) }
}

#[cfg(target_arch = "wasm32")]
fn compress_wasm_simd128(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
  // SAFETY: Only called when dispatch has verified SIMD128 is available.
  unsafe { super::wasm::compress_simd128(h, block, t, last) }
}

/// Return the compress function for the given kernel.
#[must_use]
pub(crate) fn compress_fn(id: Blake2bKernelId) -> CompressFn {
  match id {
    Blake2bKernelId::Portable => compress,
    #[cfg(target_arch = "x86_64")]
    Blake2bKernelId::X86Avx2 => compress_x86_avx2,
    #[cfg(target_arch = "x86_64")]
    Blake2bKernelId::X86Avx512vl => compress_x86_avx512vl,
    #[cfg(target_arch = "aarch64")]
    Blake2bKernelId::Aarch64Neon => compress_aarch64_neon,
    #[cfg(target_arch = "s390x")]
    Blake2bKernelId::S390xVector => compress_s390x_vector,
    #[cfg(target_arch = "powerpc64")]
    Blake2bKernelId::PowerVsx => compress_power_vsx,
    #[cfg(target_arch = "riscv64")]
    Blake2bKernelId::Riscv64V => compress_riscv64_v,
    #[cfg(target_arch = "wasm32")]
    Blake2bKernelId::WasmSimd128 => compress_wasm_simd128,
  }
}

/// Capabilities required to run the given kernel.
#[inline]
#[must_use]
#[allow(dead_code)] // Used by runtime dispatch on targets that don't bypass to a fixed kernel.
pub const fn required_caps(id: Blake2bKernelId) -> Caps {
  match id {
    Blake2bKernelId::Portable => Caps::NONE,
    #[cfg(target_arch = "x86_64")]
    Blake2bKernelId::X86Avx2 => x86::AVX2,
    #[cfg(target_arch = "x86_64")]
    Blake2bKernelId::X86Avx512vl => x86::AVX512F.union(x86::AVX512VL),
    #[cfg(target_arch = "aarch64")]
    Blake2bKernelId::Aarch64Neon => aarch64::NEON,
    #[cfg(target_arch = "s390x")]
    Blake2bKernelId::S390xVector => s390x::VECTOR,
    #[cfg(target_arch = "powerpc64")]
    Blake2bKernelId::PowerVsx => power::VSX,
    #[cfg(target_arch = "riscv64")]
    Blake2bKernelId::Riscv64V => riscv::V,
    #[cfg(target_arch = "wasm32")]
    Blake2bKernelId::WasmSimd128 => wasm::SIMD128,
  }
}

/// All kernel IDs for agreement testing.
#[cfg(test)]
pub const ALL: &[Blake2bKernelId] = &[
  Blake2bKernelId::Portable,
  #[cfg(target_arch = "x86_64")]
  Blake2bKernelId::X86Avx2,
  #[cfg(target_arch = "x86_64")]
  Blake2bKernelId::X86Avx512vl,
  #[cfg(target_arch = "aarch64")]
  Blake2bKernelId::Aarch64Neon,
  #[cfg(target_arch = "s390x")]
  Blake2bKernelId::S390xVector,
  #[cfg(target_arch = "powerpc64")]
  Blake2bKernelId::PowerVsx,
  #[cfg(target_arch = "riscv64")]
  Blake2bKernelId::Riscv64V,
  #[cfg(target_arch = "wasm32")]
  Blake2bKernelId::WasmSimd128,
];

// ─── Compile-time dispatch bypass ────────────────────────────────────────────

/// Whether the best Blake2b kernel is known at compile time.
pub(crate) const COMPILE_TIME_HW: bool = cfg!(any(
  all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl"
  ),
  all(target_arch = "x86_64", target_feature = "avx2"),
  all(
    target_arch = "aarch64",
    target_feature = "neon",
    not(target_os = "macos")
  ),
));

/// Returns the compile-time-best compress function.
#[inline(always)]
pub(crate) fn compile_time_best() -> CompressFn {
  #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vl"))]
  {
    return compress_x86_avx512vl;
  }
  #[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(all(target_feature = "avx512f", target_feature = "avx512vl"))
  ))]
  {
    return compress_x86_avx2;
  }
  #[cfg(all(target_arch = "aarch64", target_feature = "neon", not(target_os = "macos")))]
  {
    return compress_aarch64_neon;
  }
  #[allow(unreachable_code)]
  compress
}

// ─── Portable implementation ─────────────────────────────────────────────────

/// Blake2b initialization vectors (same as SHA-512 fractional parts).
pub(crate) const IV: [u64; 8] = [
  0x6a09_e667_f3bc_c908,
  0xbb67_ae85_84ca_a73b,
  0x3c6e_f372_fe94_f82b,
  0xa54f_f53a_5f1d_36f1,
  0x510e_527f_ade6_82d1,
  0x9b05_688c_2b3e_6c1f,
  0x1f83_d9ab_fb41_bd6b,
  0x5be0_cd19_137e_2179,
];

/// Message-word permutation schedule (10 rows, reused cyclically for 12 rounds).
#[allow(dead_code)] // Used by target-specific SIMD backends that are not compiled on every host.
pub(crate) const SIGMA: [[u8; 16]; 10] = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
  [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
  [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
  [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
  [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
  [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
  [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
  [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
  [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

#[derive(Clone, Copy)]
struct U64x4(u64, u64, u64, u64);

impl U64x4 {
  #[inline(always)]
  const fn new(a: u64, b: u64, c: u64, d: u64) -> Self {
    Self(a, b, c, d)
  }

  #[inline(always)]
  fn gather(src: &[u64; 16], i0: usize, i1: usize, i2: usize, i3: usize) -> Self {
    debug_assert!(i0 < src.len() && i1 < src.len() && i2 < src.len() && i3 < src.len());
    // SAFETY: all gather indices are derived from the fixed Blake2 sigma schedule,
    // which only contains values in 0..16. The debug assertion above checks that
    // contract in debug builds.
    unsafe {
      Self(
        *src.get_unchecked(i0),
        *src.get_unchecked(i1),
        *src.get_unchecked(i2),
        *src.get_unchecked(i3),
      )
    }
  }

  #[inline(always)]
  fn wrapping_add(self, rhs: Self) -> Self {
    Self(
      self.0.wrapping_add(rhs.0),
      self.1.wrapping_add(rhs.1),
      self.2.wrapping_add(rhs.2),
      self.3.wrapping_add(rhs.3),
    )
  }

  #[inline(always)]
  fn rotate_right_const(self, n: u32) -> Self {
    Self(
      self.0.rotate_right(n),
      self.1.rotate_right(n),
      self.2.rotate_right(n),
      self.3.rotate_right(n),
    )
  }

  #[inline(always)]
  const fn shuffle_left_1(self) -> Self {
    Self(self.1, self.2, self.3, self.0)
  }

  #[inline(always)]
  const fn shuffle_left_2(self) -> Self {
    Self(self.2, self.3, self.0, self.1)
  }

  #[inline(always)]
  const fn shuffle_left_3(self) -> Self {
    Self(self.3, self.0, self.1, self.2)
  }

  #[inline(always)]
  const fn shuffle_right_1(self) -> Self {
    self.shuffle_left_3()
  }

  #[inline(always)]
  const fn shuffle_right_2(self) -> Self {
    self.shuffle_left_2()
  }

  #[inline(always)]
  const fn shuffle_right_3(self) -> Self {
    self.shuffle_left_1()
  }
}

impl core::ops::BitXor for U64x4 {
  type Output = Self;

  #[inline(always)]
  fn bitxor(self, rhs: Self) -> Self::Output {
    Self(self.0 ^ rhs.0, self.1 ^ rhs.1, self.2 ^ rhs.2, self.3 ^ rhs.3)
  }
}

#[inline(always)]
fn quarter_round(v: &mut [U64x4; 4], rd: u32, rb: u32, m: U64x4) {
  v[0] = v[0].wrapping_add(v[1]).wrapping_add(m);
  v[3] = (v[3] ^ v[0]).rotate_right_const(rd);
  v[2] = v[2].wrapping_add(v[3]);
  v[1] = (v[1] ^ v[2]).rotate_right_const(rb);
}

#[inline(always)]
fn shuffle(v: &mut [U64x4; 4]) {
  v[1] = v[1].shuffle_left_1();
  v[2] = v[2].shuffle_left_2();
  v[3] = v[3].shuffle_left_3();
}

#[inline(always)]
fn unshuffle(v: &mut [U64x4; 4]) {
  v[1] = v[1].shuffle_right_1();
  v[2] = v[2].shuffle_right_2();
  v[3] = v[3].shuffle_right_3();
}

#[inline(always)]
fn round(v: &mut [U64x4; 4], m: &[u64; 16], s: &[u8; 16]) {
  quarter_round(
    v,
    32,
    24,
    U64x4::gather(m, s[0] as usize, s[2] as usize, s[4] as usize, s[6] as usize),
  );
  quarter_round(
    v,
    16,
    63,
    U64x4::gather(m, s[1] as usize, s[3] as usize, s[5] as usize, s[7] as usize),
  );

  shuffle(v);
  quarter_round(
    v,
    32,
    24,
    U64x4::gather(m, s[8] as usize, s[10] as usize, s[12] as usize, s[14] as usize),
  );
  quarter_round(
    v,
    16,
    63,
    U64x4::gather(m, s[9] as usize, s[11] as usize, s[13] as usize, s[15] as usize),
  );
  unshuffle(v);
}

/// Load 16 little-endian u64 message words from a 128-byte block.
#[inline(always)]
#[allow(clippy::indexing_slicing)] // i is always in 0..16 (loop bound).
pub(crate) fn load_msg(block: &[u8; 128]) -> [u64; 16] {
  let mut m = [0u64; 16];
  let src = block.as_ptr();
  for (i, word) in m.iter_mut().enumerate() {
    // SAFETY: block is 128 bytes, i < 16, so src + i*8 is within bounds.
    // read_unaligned handles the 1-byte alignment of [u8; 128].
    let raw = unsafe { core::ptr::read_unaligned(src.add(i.strict_mul(8)).cast::<u64>()) };
    *word = u64::from_le(raw);
  }
  m
}

/// Initialize the 16-word working vector from state, IV, counter, and finalization flag.
#[cfg(any(
  target_arch = "x86_64",
  target_arch = "aarch64",
  target_arch = "wasm32",
  target_arch = "riscv64",
  target_arch = "s390x",
  target_arch = "powerpc64"
))]
#[inline(always)]
pub(crate) fn init_v(h: &[u64; 8], t: u128, last: bool) -> [u64; 16] {
  let mut v = [0u64; 16];
  v[..8].copy_from_slice(h);
  v[8] = IV[0];
  v[9] = IV[1];
  v[10] = IV[2];
  v[11] = IV[3];
  v[12] = IV[4] ^ (t as u64);
  v[13] = IV[5] ^ ((t >> 64) as u64);
  v[14] = if last { IV[6] ^ u64::MAX } else { IV[6] };
  v[15] = IV[7];
  v
}

/// Compress a single 128-byte block into the Blake2b state.
///
/// `t` is the total number of input bytes after this block (inclusive).
/// `last` is `true` for the final block (sets the finalization flag).
#[allow(clippy::indexing_slicing)]
pub(crate) fn compress(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
  let m = load_msg(block);
  let t0 = t as u64;
  let t1 = (t >> 64) as u64;
  let f0 = if last { u64::MAX } else { 0 };

  let mut v = [
    U64x4::new(h[0], h[1], h[2], h[3]),
    U64x4::new(h[4], h[5], h[6], h[7]),
    U64x4::new(IV[0], IV[1], IV[2], IV[3]),
    U64x4::new(IV[4] ^ t0, IV[5] ^ t1, IV[6] ^ f0, IV[7]),
  ];

  round(&mut v, &m, &SIGMA[0]);
  round(&mut v, &m, &SIGMA[1]);
  round(&mut v, &m, &SIGMA[2]);
  round(&mut v, &m, &SIGMA[3]);
  round(&mut v, &m, &SIGMA[4]);
  round(&mut v, &m, &SIGMA[5]);
  round(&mut v, &m, &SIGMA[6]);
  round(&mut v, &m, &SIGMA[7]);
  round(&mut v, &m, &SIGMA[8]);
  round(&mut v, &m, &SIGMA[9]);
  round(&mut v, &m, &SIGMA[0]);
  round(&mut v, &m, &SIGMA[1]);

  let a = v[0] ^ v[2];
  let b = v[1] ^ v[3];
  h[0] ^= a.0;
  h[1] ^= a.1;
  h[2] ^= a.2;
  h[3] ^= a.3;
  h[4] ^= b.0;
  h[5] ^= b.1;
  h[6] ^= b.2;
  h[7] ^= b.3;
}
