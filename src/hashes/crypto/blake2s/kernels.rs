//! Blake2s portable compression function and kernel dispatch (RFC 7693).

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

/// Blake2s compress function pointer type.
///
/// Parameters: mutable state (8×u32), 64-byte message block, byte counter, finalization flag.
pub(crate) type CompressFn = fn(&mut [u32; 8], &[u8; 64], u64, bool);

/// Blake2s kernel identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
#[non_exhaustive]
pub enum Blake2sKernelId {
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

impl Blake2sKernelId {
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

// ─── Safe wrappers ───────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
fn compress_aarch64_neon(h: &mut [u32; 8], block: &[u8; 64], t: u64, last: bool) {
  unsafe { super::aarch64::compress_neon(h, block, t, last) }
}

/// Return the compress function for the given kernel.
#[must_use]
pub(crate) fn compress_fn(id: Blake2sKernelId) -> CompressFn {
  match id {
    Blake2sKernelId::Portable => compress,
    #[cfg(target_arch = "x86_64")]
    Blake2sKernelId::X86Avx2 => compress, // TODO: SIMD kernel
    #[cfg(target_arch = "x86_64")]
    Blake2sKernelId::X86Avx512vl => compress, // TODO: SIMD kernel
    #[cfg(target_arch = "aarch64")]
    Blake2sKernelId::Aarch64Neon => compress_aarch64_neon,
    #[cfg(target_arch = "s390x")]
    Blake2sKernelId::S390xVector => compress, // TODO: SIMD kernel
    #[cfg(target_arch = "powerpc64")]
    Blake2sKernelId::PowerVsx => compress, // TODO: SIMD kernel
    #[cfg(target_arch = "riscv64")]
    Blake2sKernelId::Riscv64V => compress, // TODO: SIMD kernel
    #[cfg(target_arch = "wasm32")]
    Blake2sKernelId::WasmSimd128 => compress, // TODO: SIMD kernel
  }
}

/// Capabilities required to run the given kernel.
#[inline]
#[must_use]
pub const fn required_caps(id: Blake2sKernelId) -> Caps {
  match id {
    Blake2sKernelId::Portable => Caps::NONE,
    #[cfg(target_arch = "x86_64")]
    Blake2sKernelId::X86Avx2 => x86::AVX2,
    #[cfg(target_arch = "x86_64")]
    Blake2sKernelId::X86Avx512vl => x86::AVX512F.union(x86::AVX512VL),
    #[cfg(target_arch = "aarch64")]
    Blake2sKernelId::Aarch64Neon => aarch64::NEON,
    #[cfg(target_arch = "s390x")]
    Blake2sKernelId::S390xVector => s390x::VECTOR,
    #[cfg(target_arch = "powerpc64")]
    Blake2sKernelId::PowerVsx => power::VSX,
    #[cfg(target_arch = "riscv64")]
    Blake2sKernelId::Riscv64V => riscv::V,
    #[cfg(target_arch = "wasm32")]
    Blake2sKernelId::WasmSimd128 => wasm::SIMD128,
  }
}

/// All kernel IDs for agreement testing.
#[cfg(test)]
pub const ALL: &[Blake2sKernelId] = &[
  Blake2sKernelId::Portable,
  #[cfg(target_arch = "aarch64")]
  Blake2sKernelId::Aarch64Neon,
];

// ─── Compile-time dispatch bypass ────────────────────────────────────────────

pub(crate) const COMPILE_TIME_HW: bool = cfg!(any(
  all(target_arch = "aarch64", target_feature = "neon"),
));

#[inline(always)]
pub(crate) fn compile_time_best() -> CompressFn {
  #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
  {
    return compress_aarch64_neon;
  }
  #[allow(unreachable_code)]
  compress
}

// ─── Portable implementation ─────────────────────────────────────────────────

/// Blake2s initialization vectors (same as SHA-256 fractional parts).
pub(crate) const IV: [u32; 8] = [
  0x6a09_e667, 0xbb67_ae85, 0x3c6e_f372, 0xa54f_f53a,
  0x510e_527f, 0x9b05_688c, 0x1f83_d9ab, 0x5be0_cd19,
];

/// Message-word permutation schedule (10 rows, reused cyclically for 10 rounds).
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

/// Blake2s quarter-round G mixing function.
///
/// Rotation constants: 16, 12, 8, 7 (same as ChaCha20).
#[inline(always)]
#[allow(clippy::indexing_slicing)]
fn g(v: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, x: u32, y: u32) {
  v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
  v[d] = (v[d] ^ v[a]).rotate_right(16);
  v[c] = v[c].wrapping_add(v[d]);
  v[b] = (v[b] ^ v[c]).rotate_right(12);
  v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
  v[d] = (v[d] ^ v[a]).rotate_right(8);
  v[c] = v[c].wrapping_add(v[d]);
  v[b] = (v[b] ^ v[c]).rotate_right(7);
}

/// Load 16 little-endian u32 message words from a 64-byte block.
#[inline(always)]
#[allow(clippy::indexing_slicing)]
pub(crate) fn load_msg(block: &[u8; 64]) -> [u32; 16] {
  let mut m = [0u32; 16];
  let src = block.as_ptr();
  for i in 0..16usize {
    // SAFETY: block is 64 bytes, i < 16, so src + i*4 is within bounds.
    let raw = unsafe { core::ptr::read_unaligned(src.add(i.strict_mul(4)).cast::<u32>()) };
    m[i] = u32::from_le(raw);
  }
  m
}

/// Initialize the 16-word working vector.
#[inline(always)]
pub(crate) fn init_v(h: &[u32; 8], t: u64, last: bool) -> [u32; 16] {
  let mut v = [0u32; 16];
  v[..8].copy_from_slice(h);
  v[8] = IV[0];
  v[9] = IV[1];
  v[10] = IV[2];
  v[11] = IV[3];
  v[12] = IV[4] ^ (t as u32);
  v[13] = IV[5] ^ ((t >> 32) as u32);
  v[14] = if last { IV[6] ^ u32::MAX } else { IV[6] };
  v[15] = IV[7];
  v
}

/// Compress a single 64-byte block into the Blake2s state.
///
/// `t` is the total number of input bytes after this block (inclusive).
/// `last` is `true` for the final block (sets the finalization flag).
#[allow(clippy::indexing_slicing)]
pub(crate) fn compress(h: &mut [u32; 8], block: &[u8; 64], t: u64, last: bool) {
  let m = load_msg(block);
  let mut v = init_v(h, t, last);

  // 10 rounds of column + diagonal mixing
  for round in 0..10u8 {
    let s = &SIGMA[round as usize];

    // Column step
    g(&mut v, 0, 4, 8, 12, m[s[0] as usize], m[s[1] as usize]);
    g(&mut v, 1, 5, 9, 13, m[s[2] as usize], m[s[3] as usize]);
    g(&mut v, 2, 6, 10, 14, m[s[4] as usize], m[s[5] as usize]);
    g(&mut v, 3, 7, 11, 15, m[s[6] as usize], m[s[7] as usize]);

    // Diagonal step
    g(&mut v, 0, 5, 10, 15, m[s[8] as usize], m[s[9] as usize]);
    g(&mut v, 1, 6, 11, 12, m[s[10] as usize], m[s[11] as usize]);
    g(&mut v, 2, 7, 8, 13, m[s[12] as usize], m[s[13] as usize]);
    g(&mut v, 3, 4, 9, 14, m[s[14] as usize], m[s[15] as usize]);
  }

  // Finalize: h[i] ^= v[i] ^ v[i+8]
  for i in 0..8 {
    h[i] ^= v[i] ^ v[i.strict_add(8)];
  }
}
