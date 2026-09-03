//! Target-specific P-256 acceleration shared by ECDSA and ECDH.
//!
//! ECDH reaches the Apple and Linux AArch64 routines and the Linux System V or
//! Windows Microsoft x64 BMI2/ADX and baseline routines. Their deterministic
//! provenance transforms clear secret-derived stack frames, saved-register
//! spill slots, and volatile integer registers. Other ECDH targets retain the
//! portable authority until equivalent target-native evidence exists. The
//! caller owns scalar validation, point validation, result typing, and
//! Rust-side cleanup.

use core::arch::global_asm;

#[cfg(feature = "p256-ecdh")]
use super::p256_portable::PublicPoint;

#[path = "ecdsa_aarch64_tables.rs"]
mod fixed_base_tables;

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
global_asm!(include_str!(
  "asm/rscrypto_p256_scalarmulbase_alt_aarch64_apple_darwin.s"
));
#[cfg(all(target_arch = "aarch64", target_os = "macos", feature = "p256-ecdh"))]
global_asm!(include_str!("asm/rscrypto_bignum_tomont_p256_aarch64_apple_darwin.s"));
#[cfg(all(target_arch = "aarch64", target_os = "macos", feature = "p256-ecdh"))]
global_asm!(include_str!("asm/rscrypto_bignum_montsqr_p256_aarch64_apple_darwin.s"));
#[cfg(all(target_arch = "aarch64", target_os = "macos", feature = "p256-ecdh"))]
global_asm!(include_str!("asm/rscrypto_bignum_montmul_p256_aarch64_apple_darwin.s"));
#[cfg(all(target_arch = "aarch64", target_os = "macos", feature = "p256-ecdh"))]
global_asm!(include_str!("asm/rscrypto_p256_scalarmul_alt_aarch64_apple_darwin.s"));
#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "p256-ecdh"))]
global_asm!(include_str!("asm/rscrypto_bignum_tomont_p256_aarch64_unknown_linux.s"));
#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "p256-ecdh"))]
global_asm!(include_str!("asm/rscrypto_bignum_montsqr_p256_aarch64_unknown_linux.s"));
#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "p256-ecdh"))]
global_asm!(include_str!("asm/rscrypto_bignum_montmul_p256_aarch64_unknown_linux.s"));
#[cfg(all(target_arch = "aarch64", target_os = "linux", feature = "p256-ecdh"))]
global_asm!(include_str!("asm/rscrypto_p256_scalarmul_alt_aarch64_unknown_linux.s"));
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
global_asm!(include_str!(
  "asm/rscrypto_p256_scalarmulbase_alt_aarch64_unknown_linux.s"
));
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
global_asm!(
  include_str!("asm/rscrypto_p256_scalarmulbase_x86_64_unknown_linux.S"),
  options(att_syntax)
);
#[cfg(all(target_arch = "x86_64", target_os = "linux", feature = "p256-ecdh"))]
global_asm!(
  include_str!("asm/rscrypto_p256_scalarmul_x86_64_unknown_linux.S"),
  options(att_syntax)
);
#[cfg(all(target_arch = "x86_64", target_os = "linux", feature = "p256-ecdh"))]
global_asm!(
  include_str!("asm/rscrypto_p256_scalarmul_alt_x86_64_unknown_linux.S"),
  options(att_syntax)
);
#[cfg(all(target_arch = "x86_64", target_os = "linux", feature = "p256-ecdh"))]
global_asm!(
  include_str!("asm/rscrypto_bignum_tomont_p256_x86_64_unknown_linux.S"),
  options(att_syntax)
);
#[cfg(all(target_arch = "x86_64", target_os = "linux", feature = "p256-ecdh"))]
global_asm!(
  include_str!("asm/rscrypto_bignum_montsqr_p256_x86_64_unknown_linux.S"),
  options(att_syntax)
);
#[cfg(all(target_arch = "x86_64", target_os = "linux", feature = "p256-ecdh"))]
global_asm!(
  include_str!("asm/rscrypto_bignum_montmul_p256_x86_64_unknown_linux.S"),
  options(att_syntax)
);
#[cfg(all(target_arch = "x86_64", target_os = "linux", feature = "p256-ecdh"))]
global_asm!(
  include_str!("asm/rscrypto_bignum_tomont_p256_alt_x86_64_unknown_linux.S"),
  options(att_syntax)
);
#[cfg(all(target_arch = "x86_64", target_os = "linux", feature = "p256-ecdh"))]
global_asm!(
  include_str!("asm/rscrypto_bignum_montsqr_p256_alt_x86_64_unknown_linux.S"),
  options(att_syntax)
);
#[cfg(all(target_arch = "x86_64", target_os = "linux", feature = "p256-ecdh"))]
global_asm!(
  include_str!("asm/rscrypto_bignum_montmul_p256_alt_x86_64_unknown_linux.S"),
  options(att_syntax)
);
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
global_asm!(
  include_str!("asm/rscrypto_p256_scalarmulbase_alt_x86_64_unknown_linux.S"),
  options(att_syntax)
);
#[cfg(all(target_arch = "x86_64", target_os = "windows", feature = "p256-ecdh"))]
global_asm!(
  include_str!("asm/rscrypto_p256_scalarmulbase_x86_64_pc_windows_msvc.S"),
  options(att_syntax)
);
#[cfg(all(target_arch = "x86_64", target_os = "windows", feature = "p256-ecdh"))]
global_asm!(
  include_str!("asm/rscrypto_p256_scalarmulbase_alt_x86_64_pc_windows_msvc.S"),
  options(att_syntax)
);
#[cfg(all(target_arch = "x86_64", target_os = "windows", feature = "p256-ecdh"))]
global_asm!(
  include_str!("asm/rscrypto_p256_scalarmul_x86_64_pc_windows_msvc.S"),
  options(att_syntax)
);
#[cfg(all(target_arch = "x86_64", target_os = "windows", feature = "p256-ecdh"))]
global_asm!(
  include_str!("asm/rscrypto_p256_scalarmul_alt_x86_64_pc_windows_msvc.S"),
  options(att_syntax)
);
#[cfg(all(target_arch = "x86_64", target_os = "windows", feature = "p256-ecdh"))]
global_asm!(
  include_str!("asm/rscrypto_p256_curve_terms_x86_64_pc_windows_msvc.S"),
  options(att_syntax)
);
#[cfg(all(target_arch = "x86_64", target_os = "windows", feature = "p256-ecdh"))]
global_asm!(
  include_str!("asm/rscrypto_p256_curve_terms_alt_x86_64_pc_windows_msvc.S"),
  options(att_syntax)
);

unsafe extern "C" {
  fn rscrypto_p256_scalarmulbase_alt(out: *mut u64, scalar: *const u64, blocksize: u64, table: *const u64);
}

#[cfg(all(
  feature = "p256-ecdh",
  any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", target_os = "linux")
  )
))]
unsafe extern "C" {
  fn rscrypto_bignum_tomont_p256(out: *mut u64, input: *const u64);
  fn rscrypto_bignum_montsqr_p256(out: *mut u64, input: *const u64);
  fn rscrypto_bignum_montmul_p256(out: *mut u64, left: *const u64, right: *const u64);
}

#[cfg(all(
  feature = "p256-ecdh",
  any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  )
))]
unsafe extern "C" {
  fn rscrypto_p256_scalarmul_alt(out: *mut u64, scalar: *const u64, point: *const u64);
}

#[cfg(all(
  target_arch = "x86_64",
  any(target_os = "linux", target_os = "windows"),
  feature = "p256-ecdh"
))]
unsafe extern "C" {
  fn rscrypto_p256_scalarmul(out: *mut u64, scalar: *const u64, point: *const u64);
}

#[cfg(all(target_arch = "x86_64", target_os = "linux", feature = "p256-ecdh"))]
unsafe extern "C" {
  fn rscrypto_bignum_tomont_p256_alt(out: *mut u64, input: *const u64);
  fn rscrypto_bignum_montsqr_p256_alt(out: *mut u64, input: *const u64);
  fn rscrypto_bignum_montmul_p256_alt(out: *mut u64, left: *const u64, right: *const u64);
}

#[cfg(all(target_arch = "x86_64", target_os = "windows", feature = "p256-ecdh"))]
unsafe extern "C" {
  fn rscrypto_p256_curve_terms(out: *mut P256CurveTerms, words: *const u64);
  fn rscrypto_p256_curve_terms_alt(out: *mut P256CurveTerms, words: *const u64);
}

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows")))]
unsafe extern "C" {
  fn rscrypto_p256_scalarmulbase(out: *mut u64, scalar: *const u64, blocksize: u64, table: *const u64);
}

#[cfg(feature = "p256-ecdh")]
#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
#[repr(C)]
struct P256CurveTerms {
  x: [u64; 4],
  y: [u64; 4],
  y_squared: [u64; 4],
  x_cubed: [u64; 4],
}

#[cfg(all(feature = "p256-ecdh", target_arch = "aarch64"))]
#[inline(always)]
fn p256_curve_terms(words: &[u64; 8]) -> P256CurveTerms {
  let mut terms = P256CurveTerms {
    x: [0; 4],
    y: [0; 4],
    y_squared: [0; 4],
    x_cubed: [0; 4],
  };
  let (x, y) = words.split_at(4);
  let mut x_squared = [0u64; 4];

  // SAFETY: the P-256 AArch64 public-field ABI is sound because:
  // 1. This function is compiled only for the matching Apple or Linux ABI.
  // 2. Every pointer names a live, properly aligned four-limb array; output
  //    arrays are disjoint from inputs and from one another for every call.
  // 3. SEC1 parsing has already established canonical field inputs. Each
  //    routine returns a reduced Montgomery residue with the same semantics
  //    as the portable authority.
  // 4. The routines operate exclusively on attacker-visible public-point
  //    coordinates. No secret owner or cleanup boundary is introduced here.
  // 5. Provenance, arithmetic, ABI, and target-native evidence remain scoped
  //    to the exact embedded snapshots and targets recorded by the project.
  unsafe {
    rscrypto_bignum_tomont_p256(terms.x.as_mut_ptr(), x.as_ptr());
    rscrypto_bignum_tomont_p256(terms.y.as_mut_ptr(), y.as_ptr());
    rscrypto_bignum_montsqr_p256(terms.y_squared.as_mut_ptr(), terms.y.as_ptr());
    rscrypto_bignum_montsqr_p256(x_squared.as_mut_ptr(), terms.x.as_ptr());
    rscrypto_bignum_montmul_p256(terms.x_cubed.as_mut_ptr(), x_squared.as_ptr(), terms.x.as_ptr());
  }
  terms
}

#[cfg(all(
  feature = "p256-ecdh",
  target_arch = "x86_64",
  any(target_os = "linux", target_os = "windows")
))]
#[inline(always)]
fn p256_curve_terms_bmi2_adx(words: &[u64; 8]) -> P256CurveTerms {
  let mut terms = P256CurveTerms {
    x: [0; 4],
    y: [0; 4],
    y_squared: [0; 4],
    x_cubed: [0; 4],
  };
  #[cfg(target_os = "linux")]
  let (x, y) = words.split_at(4);
  #[cfg(target_os = "linux")]
  let mut x_squared = [0u64; 4];

  // SAFETY: the P-256 x86-64 public-field ABI is sound because:
  // 1. The caller has proved BMI2 and ADX support before entering this helper.
  //    Linux calls each System V body directly. Windows calls one generated
  //    Microsoft x64 batch wrapper around the same three upstream bodies.
  // 2. Every pointer names a live, aligned four-limb array, and every output
  //    is disjoint from its inputs. `P256CurveTerms` has the exact declared C
  //    field layout consumed by the Windows wrapper.
  // 3. SEC1 parsing has established canonical public field inputs; the exact
  //    arithmetic and evidence scope match the AArch64 boundary above.
  unsafe {
    #[cfg(target_os = "linux")]
    {
      rscrypto_bignum_tomont_p256(terms.x.as_mut_ptr(), x.as_ptr());
      rscrypto_bignum_tomont_p256(terms.y.as_mut_ptr(), y.as_ptr());
      rscrypto_bignum_montsqr_p256(terms.y_squared.as_mut_ptr(), terms.y.as_ptr());
      rscrypto_bignum_montsqr_p256(x_squared.as_mut_ptr(), terms.x.as_ptr());
      rscrypto_bignum_montmul_p256(terms.x_cubed.as_mut_ptr(), x_squared.as_ptr(), terms.x.as_ptr());
    }
    #[cfg(target_os = "windows")]
    rscrypto_p256_curve_terms(&mut terms, words.as_ptr());
  }
  terms
}

#[cfg(all(
  feature = "p256-ecdh",
  target_arch = "x86_64",
  any(target_os = "linux", target_os = "windows")
))]
#[inline(always)]
fn p256_curve_terms_baseline(words: &[u64; 8]) -> P256CurveTerms {
  let mut terms = P256CurveTerms {
    x: [0; 4],
    y: [0; 4],
    y_squared: [0; 4],
    x_cubed: [0; 4],
  };
  #[cfg(target_os = "linux")]
  let (x, y) = words.split_at(4);
  #[cfg(target_os = "linux")]
  let mut x_squared = [0u64; 4];

  // SAFETY: the P-256 baseline x86-64 public-field ABI is sound because:
  // 1. The `_alt` routines require no optional CPU feature.
  //    Linux calls each System V body directly. Windows calls one generated
  //    Microsoft x64 batch wrapper around the same three upstream bodies.
  // 2. Pointer validity, non-aliasing, canonical public inputs, arithmetic
  //    semantics, declared C output layout, and evidence scope match the
  //    boundary documented above.
  unsafe {
    #[cfg(target_os = "linux")]
    {
      rscrypto_bignum_tomont_p256_alt(terms.x.as_mut_ptr(), x.as_ptr());
      rscrypto_bignum_tomont_p256_alt(terms.y.as_mut_ptr(), y.as_ptr());
      rscrypto_bignum_montsqr_p256_alt(terms.y_squared.as_mut_ptr(), terms.y.as_ptr());
      rscrypto_bignum_montsqr_p256_alt(x_squared.as_mut_ptr(), terms.x.as_ptr());
      rscrypto_bignum_montmul_p256_alt(terms.x_cubed.as_mut_ptr(), x_squared.as_ptr(), terms.x.as_ptr());
    }
    #[cfg(target_os = "windows")]
    rscrypto_p256_curve_terms_alt(&mut terms, words.as_ptr());
  }
  terms
}

/// Validate a canonical affine P-256 point using target-native public-field
/// arithmetic, then reconstruct the portable authority's point representation.
#[cfg(feature = "p256-ecdh")]
#[inline]
pub(super) fn p256_public_point(words: &[u64; 8]) -> Option<PublicPoint> {
  #[cfg(target_arch = "aarch64")]
  let terms = p256_curve_terms(words);

  #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows")))]
  let terms = {
    use crate::platform::caps::x86;

    if crate::platform::caps().has(x86::BMI2.union(x86::ADX)) {
      p256_curve_terms_bmi2_adx(words)
    } else {
      p256_curve_terms_baseline(words)
    }
  };

  PublicPoint::from_montgomery_curve_terms(terms.x, terms.y, terms.y_squared, terms.x_cubed)
}

/// Multiply the P-256 generator by a validated nonzero scalar.
///
/// The result contains little-endian canonical affine `x || y` limbs.
#[inline]
pub(super) fn p256_scalarmulbase_generator(scalar: &[u64; 4]) -> [u64; 8] {
  let mut out = [0u64; 8];

  #[cfg(target_arch = "aarch64")]
  {
    // SAFETY: P-256 AArch64 fixed-base scalar multiplication because:
    // 1. This module is compiled only for the Apple or Linux AArch64 ABI
    //    matching the embedded object source.
    // 2. `out` and `scalar` have the exact eight- and four-limb layouts the
    //    routine requires, and their borrows remain live for the call.
    // 3. The generated table has the exact public block size and complete
    //    entry count consumed by the routine.
    // 4. The routine accepts all 256 scalar bits; callers establish nonzero,
    //    below-order scalar semantics before crossing this boundary.
    // 5. Generated-code timing and register-residue claims remain scoped by
    //    `ct.toml` and target-native evidence.
    unsafe {
      rscrypto_p256_scalarmulbase_alt(
        out.as_mut_ptr(),
        scalar.as_ptr(),
        fixed_base_tables::P256_AARCH64_BASEPOINT_BLOCKSIZE,
        fixed_base_tables::P256_AARCH64_BASEPOINT_TABLE.as_ptr(),
      )
    };
  }

  #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows")))]
  {
    use crate::platform::caps::x86;

    let has_bmi2_adx = crate::platform::caps().has(x86::BMI2.union(x86::ADX));
    if has_bmi2_adx {
      // SAFETY: P-256 x86-64 fixed-base scalar multiplication because:
      // 1. Linux calls the System V body directly; Windows calls the upstream
      //    Microsoft x64 wrapper around that same body.
      // 2. Runtime capability detection proves BMI2 and ADX before the
      //    accelerated symbol is called.
      // 3. Array layouts, table extent, scalar contract, and evidence scope
      //    are identical to the AArch64 boundary documented above.
      unsafe {
        rscrypto_p256_scalarmulbase(
          out.as_mut_ptr(),
          scalar.as_ptr(),
          fixed_base_tables::P256_AARCH64_BASEPOINT_BLOCKSIZE,
          fixed_base_tables::P256_AARCH64_BASEPOINT_TABLE.as_ptr(),
        )
      };
    } else {
      // SAFETY: Baseline P-256 x86-64 fixed-base multiplication because:
      // 1. Linux calls the System V body directly; Windows calls the upstream
      //    Microsoft x64 wrapper around that same body.
      // 2. The `_alt` implementation requires no optional CPU feature.
      // 3. Array layouts, table extent, scalar contract, and evidence scope
      //    are identical to the AArch64 boundary documented above.
      unsafe {
        rscrypto_p256_scalarmulbase_alt(
          out.as_mut_ptr(),
          scalar.as_ptr(),
          fixed_base_tables::P256_AARCH64_BASEPOINT_BLOCKSIZE,
          fixed_base_tables::P256_AARCH64_BASEPOINT_TABLE.as_ptr(),
        )
      };
    }
  }

  out
}

/// Multiply a validated affine P-256 point by a validated nonzero scalar.
///
/// Inputs and output contain little-endian canonical affine limbs.
#[cfg(all(
  feature = "p256-ecdh",
  any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows"))
  )
))]
#[inline]
pub(super) fn p256_scalarmul(scalar: &[u64; 4], point: &[u64; 8]) -> [u64; 8] {
  let mut out = [0u64; 8];

  #[cfg(target_arch = "aarch64")]
  {
    // SAFETY: the P-256 AArch64 scalar-multiplication ABI is sound because:
    // 1. This function is compiled only for the matching Apple or Linux ABI.
    // 2. `out`, `scalar`, and `point` have the exact extents and limb layouts
    //    consumed by the embedded routine, and their borrows remain live.
    // 3. The caller validates the nonzero below-order scalar and the canonical
    //    on-curve affine point before crossing this boundary.
    // 4. The deterministic provenance transform clears every secret-bearing
    //    frame, saved-register spill slot, and volatile integer register.
    // 5. Mathematical, timing, cleanup, and ABI claims remain scoped by the
    //    target-native evidence recorded for this exact transformed snapshot.
    unsafe { rscrypto_p256_scalarmul_alt(out.as_mut_ptr(), scalar.as_ptr(), point.as_ptr()) };
  }

  #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows")))]
  {
    use crate::platform::caps::x86;

    // SAFETY: the P-256 x86-64 scalar-multiplication ABI is sound because:
    // 1. Linux calls the System V body directly; Windows calls the upstream
    //    Microsoft x64 wrapper around that same body.
    // 2. Runtime capability detection proves BMI2 and ADX before selecting
    //    the accelerated routine; the `_alt` routine is baseline x86-64.
    // 3. Array layouts, validated scalar/point semantics, deterministic
    //    cleanup, and evidence scope are the same boundary documented above.
    unsafe {
      if crate::platform::caps().has(x86::BMI2.union(x86::ADX)) {
        rscrypto_p256_scalarmul(out.as_mut_ptr(), scalar.as_ptr(), point.as_ptr());
      } else {
        rscrypto_p256_scalarmul_alt(out.as_mut_ptr(), scalar.as_ptr(), point.as_ptr());
      }
    }
  }

  out
}

#[cfg(all(test, feature = "p256-ecdh"))]
mod tests {
  const FIELD_MODULUS_MINUS_ONE: [u64; 4] = [
    0xffff_ffff_ffff_fffe,
    0x0000_0000_ffff_ffff,
    0x0000_0000_0000_0000,
    0xffff_ffff_0000_0001,
  ];
  const GENERATOR: [u64; 8] = [
    0xf4a1_3945_d898_c296,
    0x7703_7d81_2deb_33a0,
    0xf8bc_e6e5_63a4_40f2,
    0x6b17_d1f2_e12c_4247,
    0xcbb6_4068_37bf_51f5,
    0x2bce_3357_6b31_5ece,
    0x8ee7_eb4a_7c0f_9e16,
    0x4fe3_42e2_fe1a_7f9b,
  ];

  fn samples() -> [[u64; 8]; 5] {
    let mut modulus_edges = [0u64; 8];
    modulus_edges[..4].copy_from_slice(&FIELD_MODULUS_MINUS_ONE);
    modulus_edges[4..].copy_from_slice(&FIELD_MODULUS_MINUS_ONE);
    [
      [0; 8],
      [1, 0, 0, 0, 1, 0, 0, 0],
      GENERATOR,
      modulus_edges,
      [
        0x0123_4567_89ab_cdef,
        0xfedc_ba98_7654_3210,
        0x0f0f_f0f0_55aa_aa55,
        0x7fff_ffff_0000_0000,
        0xdead_beef_cafe_babe,
        0x1111_2222_3333_4444,
        0xabcd_ef01_2345_6789,
        0x7abc_def0_1234_5678,
      ],
    ]
  }

  fn portable_terms(words: &[u64; 8]) -> super::P256CurveTerms {
    let (x, y, y_squared, x_cubed) = super::super::p256_portable::curve_terms_for_test(words);
    super::P256CurveTerms {
      x,
      y,
      y_squared,
      x_cubed,
    }
  }

  #[cfg(target_arch = "aarch64")]
  #[test]
  fn aarch64_public_field_assembly_matches_portable_authority() {
    for words in samples() {
      assert_eq!(super::p256_curve_terms(&words), portable_terms(&words));
    }
  }

  #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows")))]
  #[test]
  fn x86_64_public_field_assembly_matches_portable_authority() {
    use crate::platform::caps::x86;

    for words in samples() {
      let portable = portable_terms(&words);
      assert_eq!(super::p256_curve_terms_baseline(&words), portable);
      if crate::platform::caps().has(x86::BMI2.union(x86::ADX)) {
        assert_eq!(super::p256_curve_terms_bmi2_adx(&words), portable);
      }
    }
  }

  #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows")))]
  fn x86_64_scalarmulbase_backend(scalar: &[u64; 4], accelerated: bool) -> [u64; 8] {
    let mut out = [0u64; 8];

    // SAFETY: the production boundary proves these exact ABIs. The test uses
    // disjoint, live arrays and the production table extent. It calls the
    // optional-feature implementation only after runtime BMI2+ADX detection.
    unsafe {
      if accelerated {
        super::rscrypto_p256_scalarmulbase(
          out.as_mut_ptr(),
          scalar.as_ptr(),
          super::fixed_base_tables::P256_AARCH64_BASEPOINT_BLOCKSIZE,
          super::fixed_base_tables::P256_AARCH64_BASEPOINT_TABLE.as_ptr(),
        );
      } else {
        super::rscrypto_p256_scalarmulbase_alt(
          out.as_mut_ptr(),
          scalar.as_ptr(),
          super::fixed_base_tables::P256_AARCH64_BASEPOINT_BLOCKSIZE,
          super::fixed_base_tables::P256_AARCH64_BASEPOINT_TABLE.as_ptr(),
        );
      }
    }
    out
  }

  #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows")))]
  fn x86_64_scalarmul_backend(scalar: &[u64; 4], point: &[u64; 8], accelerated: bool) -> [u64; 8] {
    let mut out = [0u64; 8];

    // SAFETY: the production boundary proves these exact ABIs. The test uses
    // disjoint, live arrays and a validated affine point, and it calls the
    // optional-feature implementation only after runtime BMI2+ADX detection.
    unsafe {
      if accelerated {
        super::rscrypto_p256_scalarmul(out.as_mut_ptr(), scalar.as_ptr(), point.as_ptr());
      } else {
        super::rscrypto_p256_scalarmul_alt(out.as_mut_ptr(), scalar.as_ptr(), point.as_ptr());
      }
    }
    out
  }

  #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows")))]
  #[test]
  fn x86_64_scalar_assembly_backends_match_production_dispatch() {
    use crate::platform::caps::x86;

    let scalars = [
      [1, 0, 0, 0],
      [0x1111_1111_1111_1111; 4],
      [0x4242_4242_4242_4242; 4],
      [
        0xf3b9_cac2_fc63_2550,
        0xbce6_faad_a717_9e84,
        0xffff_ffff_ffff_ffff,
        0xffff_ffff_0000_0000,
      ],
    ];
    let peer = super::p256_scalarmulbase_generator(&[0x2424_2424_2424_2424; 4]);
    let has_bmi2_adx = crate::platform::caps().has(x86::BMI2.union(x86::ADX));

    for scalar in scalars {
      let selected_fixed = super::p256_scalarmulbase_generator(&scalar);
      assert_eq!(x86_64_scalarmulbase_backend(&scalar, false), selected_fixed);

      let selected_arbitrary = super::p256_scalarmul(&scalar, &peer);
      assert_eq!(x86_64_scalarmul_backend(&scalar, &peer, false), selected_arbitrary);

      if has_bmi2_adx {
        assert_eq!(x86_64_scalarmulbase_backend(&scalar, true), selected_fixed);
        assert_eq!(x86_64_scalarmul_backend(&scalar, &peer, true), selected_arbitrary);
      }
    }
  }
}
