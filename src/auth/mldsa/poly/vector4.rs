//! Shared four-lane transforms for the original IBM Z and POWER arithmetic kernels.
//!
//! The canonical coefficients, fixed stage schedule, and register permutations
//! are common. Architecture modules own reduction and multiplication. Entry
//! points require the CPU/OS capabilities checked by `available()` below.

use core::simd::{simd_swizzle, u32x4};

use super::{INV_N, N, Poly, Q, R2, ROOTS};

#[cfg(target_arch = "s390x")]
#[path = "s390x.rs"]
mod arithmetic;
#[cfg(target_arch = "powerpc64")]
#[path = "powerpc64.rs"]
mod arithmetic;
use arithmetic::{add_lanes, multiply, reduce, sub_lanes};

pub(super) fn available() -> bool {
  #[cfg(target_arch = "s390x")]
  let required = crate::platform::caps::s390x::VECTOR;
  #[cfg(target_arch = "powerpc64")]
  let required = crate::platform::caps::power::ALTIVEC
    | crate::platform::caps::power::VSX
    | crate::platform::caps::power::POWER8_VECTOR;
  crate::platform::caps().has(required)
}

/// # Safety
/// The caller must establish the CPU/OS capabilities checked by `available()`.
#[inline]
#[cfg_attr(target_arch = "s390x", target_feature(enable = "vector"))]
#[cfg_attr(target_arch = "powerpc64", target_feature(enable = "altivec,vsx,power8-vector"))]
unsafe fn forward(a: u32x4, b: u32x4, zeta: u32x4) -> (u32x4, u32x4) {
  // SAFETY: The caller establishes the vector ISA and canonical input lanes. The
  // product is canonical, so both unreduced outputs are below 2q.
  unsafe {
    let t = multiply(b, zeta);
    (
      reduce(add_lanes(a, t)),
      reduce(sub_lanes(add_lanes(a, u32x4::splat(Q)), t)),
    )
  }
}

/// # Safety
/// The caller must establish the CPU/OS capabilities checked by `available()`.
#[inline]
#[cfg_attr(target_arch = "s390x", target_feature(enable = "vector"))]
#[cfg_attr(target_arch = "powerpc64", target_feature(enable = "altivec,vsx,power8-vector"))]
unsafe fn inverse(a: u32x4, b: u32x4, zeta: u32x4) -> (u32x4, u32x4) {
  // SAFETY: The caller establishes the vector ISA and canonical lanes. Both sums
  // passed to the arithmetic helpers are below 2q.
  unsafe {
    (
      reduce(add_lanes(a, b)),
      multiply(sub_lanes(add_lanes(a, u32x4::splat(Q)), b), zeta),
    )
  }
}

/// # Safety
/// The caller must establish the CPU/OS capabilities checked by `available()`.
#[cfg_attr(target_arch = "s390x", target_feature(enable = "vector"))]
#[cfg_attr(target_arch = "powerpc64", target_feature(enable = "altivec,vsx,power8-vector"))]
pub(super) unsafe fn ntt(poly: &mut Poly) {
  for chunk in poly.0.as_chunks_mut::<4>().0 {
    // SAFETY: the vector ISA is established; all four input coefficients are canonical.
    *chunk = unsafe { multiply(u32x4::from_array(*chunk), u32x4::splat(R2)) }.to_array();
  }
  let mut root = 1usize;
  let mut width = N / 2;
  while width >= 4 {
    for block in poly.0.chunks_exact_mut(width.strict_mul(2)) {
      let zeta = u32x4::splat(ROOTS[root]);
      root = root.strict_add(1);
      let (left, right) = block.split_at_mut(width);
      for (a, b) in left.as_chunks_mut::<4>().0.iter_mut().zip(right.as_chunks_mut::<4>().0) {
        // SAFETY: The public stage schedule provides disjoint four-coefficient
        // arrays. No vector alignment is assumed; inputs are canonical.
        let (sum, difference) = unsafe { forward(u32x4::from_array(*a), u32x4::from_array(*b), zeta) };
        *a = sum.to_array();
        *b = difference.to_array();
      }
    }
    width >>= 1;
  }
  for (index, block) in poly.0.as_chunks_mut::<8>().0.iter_mut().enumerate() {
    let halves = block.as_chunks_mut::<4>().0;
    let lo = u32x4::from_array(halves[0]);
    let hi = u32x4::from_array(halves[1]);
    let root2 = (N / 4).strict_add(index.strict_mul(2));
    let root1 = (N / 2).strict_add(index.strict_mul(4));
    let zeta2 = u32x4::from_array([
      ROOTS[root2],
      ROOTS[root2],
      ROOTS[root2.strict_add(1)],
      ROOTS[root2.strict_add(1)],
    ]);
    let zeta1 = u32x4::from_slice(&ROOTS[root1..root1.strict_add(4)]);
    // SAFETY: The caller establishes the vector ISA. Permutations pair exactly the
    // width-two and width-one butterflies; roots and lane indices are public.
    let (sum, difference) = unsafe {
      let (sum, difference) = forward(
        simd_swizzle!(lo, hi, [0, 1, 4, 5]),
        simd_swizzle!(lo, hi, [2, 3, 6, 7]),
        zeta2,
      );
      forward(
        simd_swizzle!(sum, difference, [0, 4, 2, 6]),
        simd_swizzle!(sum, difference, [1, 5, 3, 7]),
        zeta1,
      )
    };
    halves[0] = simd_swizzle!(sum, difference, [0, 4, 1, 5]).to_array();
    halves[1] = simd_swizzle!(sum, difference, [2, 6, 3, 7]).to_array();
  }
}

/// # Safety
/// The caller must establish the CPU/OS capabilities checked by `available()`.
#[cfg_attr(target_arch = "s390x", target_feature(enable = "vector"))]
#[cfg_attr(target_arch = "powerpc64", target_feature(enable = "altivec,vsx,power8-vector"))]
pub(super) unsafe fn inverse_ntt(poly: &mut Poly) {
  for (index, block) in poly.0.as_chunks_mut::<8>().0.iter_mut().enumerate() {
    let halves = block.as_chunks_mut::<4>().0;
    let lo = u32x4::from_array(halves[0]);
    let hi = u32x4::from_array(halves[1]);
    let root1 = N.strict_sub(index.strict_mul(4)).strict_sub(1);
    let root2 = (N / 2).strict_sub(index.strict_mul(2)).strict_sub(1);
    let zeta1 = u32x4::from_array(core::array::from_fn(|i| Q.strict_sub(ROOTS[root1.strict_sub(i)])));
    let zeta2 = u32x4::from_array([
      Q.strict_sub(ROOTS[root2]),
      Q.strict_sub(ROOTS[root2]),
      Q.strict_sub(ROOTS[root2.strict_sub(1)]),
      Q.strict_sub(ROOTS[root2.strict_sub(1)]),
    ]);
    // SAFETY: the vector ISA and canonical inputs are established. Fixed permutations
    // reverse the forward narrow-stage lane schedule, using reversed roots.
    let (sum, difference) = unsafe {
      let (sum, difference) = inverse(
        simd_swizzle!(lo, hi, [0, 2, 4, 6]),
        simd_swizzle!(lo, hi, [1, 3, 5, 7]),
        zeta1,
      );
      inverse(
        simd_swizzle!(sum, difference, [0, 4, 2, 6]),
        simd_swizzle!(sum, difference, [1, 5, 3, 7]),
        zeta2,
      )
    };
    halves[0] = simd_swizzle!(sum, difference, [0, 1, 4, 5]).to_array();
    halves[1] = simd_swizzle!(sum, difference, [2, 3, 6, 7]).to_array();
  }
  let mut root = N / 4;
  let mut width = 4usize;
  while width < N {
    for block in poly.0.chunks_exact_mut(width.strict_mul(2)) {
      root = root.strict_sub(1);
      let zeta = u32x4::splat(Q.strict_sub(ROOTS[root]));
      let (left, right) = block.split_at_mut(width);
      for (a, b) in left.as_chunks_mut::<4>().0.iter_mut().zip(right.as_chunks_mut::<4>().0) {
        // SAFETY: the vector ISA is established. Disjoint arrays contain canonical
        // coefficients, and the fixed public stage schedule covers each once.
        let (sum, difference) = unsafe { inverse(u32x4::from_array(*a), u32x4::from_array(*b), zeta) };
        *a = sum.to_array();
        *b = difference.to_array();
      }
    }
    width = width.strict_mul(2);
  }
  for chunk in poly.0.as_chunks_mut::<4>().0 {
    // SAFETY: the vector ISA is established; canonical inputs and INV_N are below q.
    *chunk = unsafe { multiply(u32x4::from_array(*chunk), u32x4::splat(INV_N)) }.to_array();
  }
}

/// # Safety
/// The caller must establish the CPU/OS capabilities checked by `available()`.
#[cfg_attr(target_arch = "s390x", target_feature(enable = "vector"))]
#[cfg_attr(target_arch = "powerpc64", target_feature(enable = "altivec,vsx,power8-vector"))]
pub(super) unsafe fn accumulate_product(out: &mut [u32; N], a: &[u32; N], b: &[u32; N]) {
  for ((out, a), b) in out
    .as_chunks_mut::<4>()
    .0
    .iter_mut()
    .zip(a.as_chunks::<4>().0)
    .zip(b.as_chunks::<4>().0)
  {
    // SAFETY: the vector ISA is established; input and output coefficients are
    // canonical. Fixed-size references keep the destination disjoint from inputs.
    *out = unsafe {
      reduce(add_lanes(
        u32x4::from_array(*out),
        multiply(u32x4::from_array(*a), u32x4::from_array(*b)),
      ))
    }
    .to_array();
  }
}

/// # Safety
/// The caller must establish the CPU/OS capabilities checked by `available()`.
#[cfg_attr(target_arch = "s390x", target_feature(enable = "vector"))]
#[cfg_attr(target_arch = "powerpc64", target_feature(enable = "altivec,vsx,power8-vector"))]
pub(super) unsafe fn product(out: &mut [u32; N], a: &[u32; N], b: &[u32; N]) {
  for ((out, a), b) in out
    .as_chunks_mut::<4>()
    .0
    .iter_mut()
    .zip(a.as_chunks::<4>().0)
    .zip(b.as_chunks::<4>().0)
  {
    // SAFETY: the vector ISA is established and all input coefficients are canonical.
    *out = unsafe { multiply(u32x4::from_array(*a), u32x4::from_array(*b)) }.to_array();
  }
}

#[cfg(test)]
mod tests {
  use super::{
    Q,
    arithmetic::{multiply, reduce},
    u32x4,
  };

  #[test]
  fn vector_arithmetic_matches_integer_modulo_at_boundaries() {
    if !super::available() {
      return;
    }
    let edges = [0, 1, Q - 1, Q, Q + 1, 2 * Q - 1];
    for i in 0..edges.len() {
      for j in 0..edges.len() {
        // Distinct lanes catch endian-sensitive even/odd-product reconstruction bugs.
        let a = core::array::from_fn(|lane| edges[(i + lane) % edges.len()]);
        let b = core::array::from_fn(|lane| edges[(j + 2 * lane) % edges.len()]);
        // SAFETY: The capability guard proves the selected vector ISA. Every lane is below 2q.
        let (product, reduced) = unsafe {
          (
            multiply(u32x4::from_array(a), u32x4::from_array(b)).to_array(),
            reduce(u32x4::from_array(a)).to_array(),
          )
        };
        for lane in 0..4 {
          let expected = ((u64::from(a[lane]) * u64::from(b[lane])) % u64::from(Q)) * 8_265_825 % u64::from(Q);
          assert_eq!(u64::from(product[lane]), expected, "product {i}, {j}, lane {lane}");
          assert_eq!(reduced[lane], a[lane] % Q, "reduction {i}, lane {lane}");
        }
      }
    }
  }
}
