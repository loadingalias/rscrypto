//! Keccak-f[1600] x86_64 AVX-512 two-state kernel.
//!
//! This path packs two independent Keccak states lane-wise in 128-bit vectors:
//! state A in lane 0, state B in lane 1. AVX-512VL gives us VPROLQ for the
//! rotation-heavy rho step and VPTERNLOGQ for chi.
//!
//! # Safety
//!
//! All functions require `avx512f,avx512vl,sse4.1`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
macro_rules! rol {
  ($value:expr, $left:literal) => {{ _mm_rol_epi64::<$left>($value) }};
}

#[cfg(target_arch = "x86_64")]
macro_rules! chi {
  ($a:expr, $b:expr, $c:expr) => {{ _mm_ternarylogic_epi64($a, $b, $c, 0xD2) }};
}

#[cfg(target_arch = "x86_64")]
macro_rules! rol4 {
  ($value:expr, $left:literal) => {{ _mm256_rol_epi64::<$left>($value) }};
}

#[cfg(target_arch = "x86_64")]
macro_rules! chi4 {
  ($a:expr, $b:expr, $c:expr) => {{ _mm256_ternarylogic_epi64($a, $b, $c, 0xD2) }};
}

/// Apply Keccak-f[1600] to two independent states using AVX-512VL.
///
/// # Safety
///
/// Caller must ensure `avx512f`, `avx512vl`, and `sse4.1` CPU features are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,sse4.1")]
#[inline]
unsafe fn keccakf_x86_avx512_x2_impl(state_a: &mut [u64; 25], state_b: &mut [u64; 25]) {
  macro_rules! load {
    ($i:literal) => {
      _mm_set_epi64x(state_b[$i] as i64, state_a[$i] as i64)
    };
  }

  macro_rules! store {
    ($i:literal, $value:expr) => {{
      state_a[$i] = _mm_extract_epi64::<0>($value) as u64;
      state_b[$i] = _mm_extract_epi64::<1>($value) as u64;
    }};
  }

  macro_rules! xor {
    ($a:expr, $b:expr) => {
      _mm_xor_si128($a, $b)
    };
    ($a:expr, $b:expr, $c:expr) => {
      _mm_xor_si128(_mm_xor_si128($a, $b), $c)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr) => {
      _mm_xor_si128(_mm_xor_si128($a, $b), _mm_xor_si128(_mm_xor_si128($c, $d), $e))
    };
  }

  let mut a0 = load!(0);
  let mut a1 = load!(1);
  let mut a2 = load!(2);
  let mut a3 = load!(3);
  let mut a4 = load!(4);
  let mut a5 = load!(5);
  let mut a6 = load!(6);
  let mut a7 = load!(7);
  let mut a8 = load!(8);
  let mut a9 = load!(9);
  let mut a10 = load!(10);
  let mut a11 = load!(11);
  let mut a12 = load!(12);
  let mut a13 = load!(13);
  let mut a14 = load!(14);
  let mut a15 = load!(15);
  let mut a16 = load!(16);
  let mut a17 = load!(17);
  let mut a18 = load!(18);
  let mut a19 = load!(19);
  let mut a20 = load!(20);
  let mut a21 = load!(21);
  let mut a22 = load!(22);
  let mut a23 = load!(23);
  let mut a24 = load!(24);

  for &rc in &super::RC {
    let c0 = xor!(a0, a5, a10, a15, a20);
    let c1 = xor!(a1, a6, a11, a16, a21);
    let c2 = xor!(a2, a7, a12, a17, a22);
    let c3 = xor!(a3, a8, a13, a18, a23);
    let c4 = xor!(a4, a9, a14, a19, a24);

    let d0 = xor!(c4, rol!(c1, 1));
    let d1 = xor!(c0, rol!(c2, 1));
    let d2 = xor!(c1, rol!(c3, 1));
    let d3 = xor!(c2, rol!(c4, 1));
    let d4 = xor!(c3, rol!(c0, 1));

    a0 = xor!(a0, d0);
    a5 = xor!(a5, d0);
    a10 = xor!(a10, d0);
    a15 = xor!(a15, d0);
    a20 = xor!(a20, d0);
    a1 = xor!(a1, d1);
    a6 = xor!(a6, d1);
    a11 = xor!(a11, d1);
    a16 = xor!(a16, d1);
    a21 = xor!(a21, d1);
    a2 = xor!(a2, d2);
    a7 = xor!(a7, d2);
    a12 = xor!(a12, d2);
    a17 = xor!(a17, d2);
    a22 = xor!(a22, d2);
    a3 = xor!(a3, d3);
    a8 = xor!(a8, d3);
    a13 = xor!(a13, d3);
    a18 = xor!(a18, d3);
    a23 = xor!(a23, d3);
    a4 = xor!(a4, d4);
    a9 = xor!(a9, d4);
    a14 = xor!(a14, d4);
    a19 = xor!(a19, d4);
    a24 = xor!(a24, d4);

    let b0 = a0;
    let b10 = rol!(a1, 1);
    let b20 = rol!(a2, 62);
    let b5 = rol!(a3, 28);
    let b15 = rol!(a4, 27);
    let b16 = rol!(a5, 36);
    let b1 = rol!(a6, 44);
    let b11 = rol!(a7, 6);
    let b21 = rol!(a8, 55);
    let b6 = rol!(a9, 20);
    let b7 = rol!(a10, 3);
    let b17 = rol!(a11, 10);
    let b2 = rol!(a12, 43);
    let b12 = rol!(a13, 25);
    let b22 = rol!(a14, 39);
    let b23 = rol!(a15, 41);
    let b8 = rol!(a16, 45);
    let b18 = rol!(a17, 15);
    let b3 = rol!(a18, 21);
    let b13 = rol!(a19, 8);
    let b14 = rol!(a20, 18);
    let b24 = rol!(a21, 2);
    let b9 = rol!(a22, 61);
    let b19 = rol!(a23, 56);
    let b4 = rol!(a24, 14);

    a0 = chi!(b0, b1, b2);
    a1 = chi!(b1, b2, b3);
    a2 = chi!(b2, b3, b4);
    a3 = chi!(b3, b4, b0);
    a4 = chi!(b4, b0, b1);

    a5 = chi!(b5, b6, b7);
    a6 = chi!(b6, b7, b8);
    a7 = chi!(b7, b8, b9);
    a8 = chi!(b8, b9, b5);
    a9 = chi!(b9, b5, b6);

    a10 = chi!(b10, b11, b12);
    a11 = chi!(b11, b12, b13);
    a12 = chi!(b12, b13, b14);
    a13 = chi!(b13, b14, b10);
    a14 = chi!(b14, b10, b11);

    a15 = chi!(b15, b16, b17);
    a16 = chi!(b16, b17, b18);
    a17 = chi!(b17, b18, b19);
    a18 = chi!(b18, b19, b15);
    a19 = chi!(b19, b15, b16);

    a20 = chi!(b20, b21, b22);
    a21 = chi!(b21, b22, b23);
    a22 = chi!(b22, b23, b24);
    a23 = chi!(b23, b24, b20);
    a24 = chi!(b24, b20, b21);

    a0 = xor!(a0, _mm_set1_epi64x(rc as i64));
  }

  store!(0, a0);
  store!(1, a1);
  store!(2, a2);
  store!(3, a3);
  store!(4, a4);
  store!(5, a5);
  store!(6, a6);
  store!(7, a7);
  store!(8, a8);
  store!(9, a9);
  store!(10, a10);
  store!(11, a11);
  store!(12, a12);
  store!(13, a13);
  store!(14, a14);
  store!(15, a15);
  store!(16, a16);
  store!(17, a17);
  store!(18, a18);
  store!(19, a19);
  store!(20, a20);
  store!(21, a21);
  store!(22, a22);
  store!(23, a23);
  store!(24, a24);
}

/// Apply Keccak-f[1600] to two independent states using x86_64 AVX-512VL.
///
/// # Safety
///
/// Caller must ensure `avx512f`, `avx512vl`, and `sse4.1` CPU features are available.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) unsafe fn keccakf_x86_avx512_x2(state_a: &mut [u64; 25], state_b: &mut [u64; 25]) {
  // SAFETY: AVX-512 x2 implementation call because:
  // 1. The caller guarantees AVX512F, AVX512VL, and SSE4.1 are available.
  // 2. The mutable state references are initialized 25-lane Keccak states.
  unsafe { keccakf_x86_avx512_x2_impl(state_a, state_b) }
}

/// Apply Keccak-f[1600] to four independent states using AVX-512VL.
///
/// # Safety
///
/// Caller must ensure `avx512f`, `avx512vl`, and `sse4.1` CPU features are available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,sse4.1")]
#[inline]
unsafe fn keccakf_x86_avx512_x4_impl(
  state_a: &mut [u64; 25],
  state_b: &mut [u64; 25],
  state_c: &mut [u64; 25],
  state_d: &mut [u64; 25],
) {
  macro_rules! load {
    ($i:literal) => {
      _mm256_set_epi64x(
        state_d[$i] as i64,
        state_c[$i] as i64,
        state_b[$i] as i64,
        state_a[$i] as i64,
      )
    };
  }

  macro_rules! store {
    ($i:literal, $value:expr) => {{
      state_a[$i] = _mm256_extract_epi64::<0>($value) as u64;
      state_b[$i] = _mm256_extract_epi64::<1>($value) as u64;
      state_c[$i] = _mm256_extract_epi64::<2>($value) as u64;
      state_d[$i] = _mm256_extract_epi64::<3>($value) as u64;
    }};
  }

  macro_rules! xor {
    ($a:expr, $b:expr) => {
      _mm256_xor_si256($a, $b)
    };
    ($a:expr, $b:expr, $c:expr) => {
      _mm256_xor_si256(_mm256_xor_si256($a, $b), $c)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr) => {
      _mm256_xor_si256(_mm256_xor_si256($a, $b), _mm256_xor_si256(_mm256_xor_si256($c, $d), $e))
    };
  }

  let mut a0 = load!(0);
  let mut a1 = load!(1);
  let mut a2 = load!(2);
  let mut a3 = load!(3);
  let mut a4 = load!(4);
  let mut a5 = load!(5);
  let mut a6 = load!(6);
  let mut a7 = load!(7);
  let mut a8 = load!(8);
  let mut a9 = load!(9);
  let mut a10 = load!(10);
  let mut a11 = load!(11);
  let mut a12 = load!(12);
  let mut a13 = load!(13);
  let mut a14 = load!(14);
  let mut a15 = load!(15);
  let mut a16 = load!(16);
  let mut a17 = load!(17);
  let mut a18 = load!(18);
  let mut a19 = load!(19);
  let mut a20 = load!(20);
  let mut a21 = load!(21);
  let mut a22 = load!(22);
  let mut a23 = load!(23);
  let mut a24 = load!(24);

  for &rc in &super::RC {
    let c0 = xor!(a0, a5, a10, a15, a20);
    let c1 = xor!(a1, a6, a11, a16, a21);
    let c2 = xor!(a2, a7, a12, a17, a22);
    let c3 = xor!(a3, a8, a13, a18, a23);
    let c4 = xor!(a4, a9, a14, a19, a24);

    let d0 = xor!(c4, rol4!(c1, 1));
    let d1 = xor!(c0, rol4!(c2, 1));
    let d2 = xor!(c1, rol4!(c3, 1));
    let d3 = xor!(c2, rol4!(c4, 1));
    let d4 = xor!(c3, rol4!(c0, 1));

    a0 = xor!(a0, d0);
    a5 = xor!(a5, d0);
    a10 = xor!(a10, d0);
    a15 = xor!(a15, d0);
    a20 = xor!(a20, d0);
    a1 = xor!(a1, d1);
    a6 = xor!(a6, d1);
    a11 = xor!(a11, d1);
    a16 = xor!(a16, d1);
    a21 = xor!(a21, d1);
    a2 = xor!(a2, d2);
    a7 = xor!(a7, d2);
    a12 = xor!(a12, d2);
    a17 = xor!(a17, d2);
    a22 = xor!(a22, d2);
    a3 = xor!(a3, d3);
    a8 = xor!(a8, d3);
    a13 = xor!(a13, d3);
    a18 = xor!(a18, d3);
    a23 = xor!(a23, d3);
    a4 = xor!(a4, d4);
    a9 = xor!(a9, d4);
    a14 = xor!(a14, d4);
    a19 = xor!(a19, d4);
    a24 = xor!(a24, d4);

    let b0 = a0;
    let b10 = rol4!(a1, 1);
    let b20 = rol4!(a2, 62);
    let b5 = rol4!(a3, 28);
    let b15 = rol4!(a4, 27);
    let b16 = rol4!(a5, 36);
    let b1 = rol4!(a6, 44);
    let b11 = rol4!(a7, 6);
    let b21 = rol4!(a8, 55);
    let b6 = rol4!(a9, 20);
    let b7 = rol4!(a10, 3);
    let b17 = rol4!(a11, 10);
    let b2 = rol4!(a12, 43);
    let b12 = rol4!(a13, 25);
    let b22 = rol4!(a14, 39);
    let b23 = rol4!(a15, 41);
    let b8 = rol4!(a16, 45);
    let b18 = rol4!(a17, 15);
    let b3 = rol4!(a18, 21);
    let b13 = rol4!(a19, 8);
    let b14 = rol4!(a20, 18);
    let b24 = rol4!(a21, 2);
    let b9 = rol4!(a22, 61);
    let b19 = rol4!(a23, 56);
    let b4 = rol4!(a24, 14);

    a0 = chi4!(b0, b1, b2);
    a1 = chi4!(b1, b2, b3);
    a2 = chi4!(b2, b3, b4);
    a3 = chi4!(b3, b4, b0);
    a4 = chi4!(b4, b0, b1);

    a5 = chi4!(b5, b6, b7);
    a6 = chi4!(b6, b7, b8);
    a7 = chi4!(b7, b8, b9);
    a8 = chi4!(b8, b9, b5);
    a9 = chi4!(b9, b5, b6);

    a10 = chi4!(b10, b11, b12);
    a11 = chi4!(b11, b12, b13);
    a12 = chi4!(b12, b13, b14);
    a13 = chi4!(b13, b14, b10);
    a14 = chi4!(b14, b10, b11);

    a15 = chi4!(b15, b16, b17);
    a16 = chi4!(b16, b17, b18);
    a17 = chi4!(b17, b18, b19);
    a18 = chi4!(b18, b19, b15);
    a19 = chi4!(b19, b15, b16);

    a20 = chi4!(b20, b21, b22);
    a21 = chi4!(b21, b22, b23);
    a22 = chi4!(b22, b23, b24);
    a23 = chi4!(b23, b24, b20);
    a24 = chi4!(b24, b20, b21);

    a0 = xor!(a0, _mm256_set1_epi64x(rc as i64));
  }

  store!(0, a0);
  store!(1, a1);
  store!(2, a2);
  store!(3, a3);
  store!(4, a4);
  store!(5, a5);
  store!(6, a6);
  store!(7, a7);
  store!(8, a8);
  store!(9, a9);
  store!(10, a10);
  store!(11, a11);
  store!(12, a12);
  store!(13, a13);
  store!(14, a14);
  store!(15, a15);
  store!(16, a16);
  store!(17, a17);
  store!(18, a18);
  store!(19, a19);
  store!(20, a20);
  store!(21, a21);
  store!(22, a22);
  store!(23, a23);
  store!(24, a24);
}

/// Apply Keccak-f[1600] to four independent states using x86_64 AVX-512VL.
///
/// # Safety
///
/// Caller must ensure `avx512f`, `avx512vl`, and `sse4.1` CPU features are available.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) unsafe fn keccakf_x86_avx512_x4(
  state_a: &mut [u64; 25],
  state_b: &mut [u64; 25],
  state_c: &mut [u64; 25],
  state_d: &mut [u64; 25],
) {
  // SAFETY: AVX-512 x4 implementation call because:
  // 1. The caller guarantees AVX512F, AVX512VL, and SSE4.1 are available.
  // 2. The mutable state references are initialized 25-lane Keccak states.
  unsafe { keccakf_x86_avx512_x4_impl(state_a, state_b, state_c, state_d) }
}
