//! Internal evidence adapters over production ML-DSA operations.
//!
//! Callers supply synthetic secrets. Polynomial inputs are canonical; encoded
//! inputs have the fixed public parameter-set width. Owners retain their normal
//! Drop cleanup. These adapters do not implement an alternate algorithm.

use super::{P44, P65, P87, poly::Poly, portable::SigningState, sampling};

/// Execute the production forward transform on canonical coefficients.
#[must_use]
pub(crate) fn diag_mldsa_ntt(input: &[u32; 256]) -> u32 {
  crate::traits::ct::with_data_independent_timing(|| {
    let mut value = Poly::zero();
    value.0.copy_from_slice(input);
    value.ntt();
    core::hint::black_box(&value.0).iter().fold(0, |digest, x| digest ^ x)
  })
}

/// Execute the production product on two canonical polynomials.
#[must_use]
pub(crate) fn diag_mldsa_product(left: &[u32; 256], right: &[u32; 256]) -> u32 {
  crate::traits::ct::with_data_independent_timing(|| {
    let mut left_owner = Poly::zero();
    left_owner.0.copy_from_slice(left);
    let mut out = Poly::zero();
    out.product(&left_owner, right);
    core::hint::black_box(&out.0).iter().fold(0, |digest, x| digest ^ x)
  })
}

/// Execute production accumulation; all three inputs contain canonical residues.
#[must_use]
pub(crate) fn diag_mldsa_accumulate(accumulator: &[u32; 256], left: &[u32; 256], right: &[u32; 256]) -> u32 {
  crate::traits::ct::with_data_independent_timing(|| {
    let mut out = Poly::zero();
    out.0.copy_from_slice(accumulator);
    let mut right_owner = Poly::zero();
    right_owner.0.copy_from_slice(right);
    out.accumulate_product(left, &right_owner);
    core::hint::black_box(&out.0).iter().fold(0, |digest, x| digest ^ x)
  })
}

/// Execute the production fused row product over five columns, as ML-DSA-65 signing computes
/// `Â·ŷ`; every input contains canonical residues.
#[must_use]
pub(crate) fn diag_mldsa_row_product(entries: &[[u32; 256]; 5], inputs: &[[u32; 256]; 5]) -> u32 {
  crate::traits::ct::with_data_independent_timing(|| {
    let input_owners: [Poly; 5] = core::array::from_fn(|i| {
      let mut owner = Poly::zero();
      owner.0.copy_from_slice(&inputs[i]);
      owner
    });
    let mut out = Poly::zero();
    out.row_product(entries, &input_owners);
    core::hint::black_box(&out.0).iter().fold(0, |digest, x| digest ^ x)
  })
}

/// Execute the production complete norm scan at a public bound.
#[must_use]
pub(crate) fn diag_mldsa_norm(input: &[u32; 256], bound: u32) -> u32 {
  crate::traits::ct::with_data_independent_timing(|| {
    let mut value = Poly::zero();
    value.0.copy_from_slice(input);
    value.exceeds_bound(bound)
  })
}

/// Execute production rounding and decomposition at a standard public gamma2.
#[must_use]
pub(crate) fn diag_mldsa_rounding(input: &[u32; 256], gamma2: u32) -> u32 {
  crate::traits::ct::with_data_independent_timing(|| {
    let mut digest = 0;
    for &coefficient in input {
      let (high, low) = core::hint::black_box(super::poly::power2_round(coefficient));
      let (high2, low2) = core::hint::black_box(super::poly::decompose(coefficient, gamma2));
      digest ^= high ^ low ^ high2 ^ low2;
    }
    digest
  })
}

/// Execute production mask sampling; `large` selects the public gamma1 profile.
#[must_use]
pub(crate) fn diag_mldsa_mask(seed: &[u8; 64], large: bool) -> u32 {
  crate::traits::ct::with_data_independent_timing(|| {
    let mut out = Poly::zero();
    sampling::mask(seed, 0, if large { P65 } else { P44 }, &mut out);
    core::hint::black_box(&out.0).iter().fold(0, |digest, x| digest ^ x)
  })
}

macro_rules! prepare_adapter {
  ($name:ident, $size:literal, $k:literal, $l:literal, $parameters:ident) => {
    /// Decode and transform secret key components through production SigningState.
    ///
    /// This fixed-work preparation boundary excludes public matrix expansion and
    /// redundant-field validation. Inputs have the standard encoded secret width.
    #[must_use]
    pub(crate) fn $name(secret: &[u8; $size]) -> bool {
      let mut state = SigningState::<$k, $l>::zero();
      let valid = state.decode(secret, $parameters);
      core::hint::black_box(&state);
      valid
    }
  };
}
prepare_adapter!(diag_mldsa_prepare44, 2560, 4, 4, P44);
prepare_adapter!(diag_mldsa_prepare65, 4032, 6, 5, P65);
prepare_adapter!(diag_mldsa_prepare87, 4896, 8, 7, P87);

/// Execute production portable Montgomery products without polynomial copies.
///
/// Adjacent canonical coefficients form independent operand pairs. This narrow
/// adapter separates arithmetic timing from transform scheduling and cleanup.
#[must_use]
pub(crate) fn diag_mldsa_montgomery_batch(input: &[u32; 256]) -> u32 {
  crate::traits::ct::with_data_independent_timing(|| {
    let mut digest = 0;
    for pair in input.as_chunks::<2>().0 {
      digest ^= core::hint::black_box(super::poly::montgomery(pair[0], pair[1]));
    }
    digest
  })
}

use super::*;

/// Diagnostic execution of the production inverse NTT on canonical coefficients.
///
/// Available only to internal evidence builds. Inputs must be below 8,380,417.
#[must_use]
pub(crate) fn diag_mldsa_inverse_ntt(input: &[u32; 256]) -> u32 {
  crate::traits::ct::with_data_independent_timing(|| {
    let mut value = poly::Poly::zero();
    value.0.copy_from_slice(input);
    value.inverse_ntt();
    core::hint::black_box(&value.0).iter().fold(0, |digest, x| digest ^ x)
  })
}

/// Diagnostic execution of the production scalar inverse NTT, bypassing dispatch.
///
/// Available only to internal evidence builds. Inputs must be below 8,380,417.
#[must_use]
pub(crate) fn diag_mldsa_inverse_ntt_portable(input: &[u32; 256]) -> u32 {
  crate::traits::ct::with_data_independent_timing(|| {
    let mut value = poly::Poly::zero();
    value.0.copy_from_slice(input);
    value.inverse_ntt_scalar::<false>();
    core::hint::black_box(&value.0).iter().fold(0, |digest, x| digest ^ x)
  })
}

/// Diagnostic execution of the production secret-noise sampler.
///
/// Available only to internal evidence builds. `eta` must be two or four.
#[must_use]
pub(crate) fn diag_mldsa_noise(seed: &[u8; 64], eta: u32) -> (bool, u32) {
  crate::traits::ct::with_data_independent_timing(|| {
    let mut value = poly::Poly::zero();
    let valid = sampling::noise(seed, 0, eta, &mut value).is_ok();
    (
      valid,
      core::hint::black_box(&value.0).iter().fold(0, |digest, x| digest ^ x),
    )
  })
}

/// Diagnostic execution of the production unpublished-challenge sampler.
///
/// Available only to internal evidence builds. Use standard seed lengths and tau.
#[must_use]
pub(crate) fn diag_mldsa_challenge(seed: &[u8], tau: usize) -> (bool, u32) {
  crate::traits::ct::with_data_independent_timing(|| {
    let mut value = poly::Poly::zero();
    let valid = sampling::challenge(seed, tau, &mut value).is_ok();
    (
      valid,
      core::hint::black_box(&value.0).iter().fold(0, |digest, x| digest ^ x),
    )
  })
}
