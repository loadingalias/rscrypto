//! Retained ML-DSA evidence roots over production operations.
//!
//! Inputs belong to the caller. Roots borrow them and let the production
//! diagnostic adapters own and clear any temporary polynomial/key state.

use rscrypto::auth;

macro_rules! polynomial_entry {
  ($name:ident, $operation:expr) => {
    /// Execute a production ML-DSA polynomial operation. Null is rejected.
    ///
    /// # Safety
    /// A non-null input must be aligned, readable for 256 initialized u32s,
    /// immutable during the call, and contain only canonical residues below q.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn $name(input: *const [u32; 256]) -> u32 {
      // SAFETY: The caller supplies a live, aligned, initialized array with no
      // concurrent writes. The borrow stays within this call; null is rejected.
      let input = unsafe { input.as_ref() };
      let Some(input) = input else {
        return 0;
      };
      ($operation)(input)
    }
  };
}
polynomial_entry!(ct_entry_mldsa_ntt, auth::diag_mldsa_ntt);
polynomial_entry!(ct_entry_mldsa_montgomery, auth::diag_mldsa_montgomery_batch);
polynomial_entry!(ct_entry_mldsa_inverse_ntt, auth::diag_mldsa_inverse_ntt);
polynomial_entry!(
  ct_entry_mldsa_inverse_ntt_portable,
  auth::diag_mldsa_inverse_ntt_portable
);
polynomial_entry!(ct_entry_mldsa_norm, |input| auth::diag_mldsa_norm(input, 95_232));
polynomial_entry!(ct_entry_mldsa_rounding44, |input| auth::diag_mldsa_rounding(
  input, 95_232
));
polynomial_entry!(ct_entry_mldsa_rounding65, |input| auth::diag_mldsa_rounding(
  input, 261_888
));

/// Execute a production ML-DSA product. Null inputs are rejected.
///
/// # Safety
/// Both inputs must be aligned, readable for 256 initialized u32s, immutable
/// during the call, and contain canonical residues below q. They may overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ct_entry_mldsa_product(left: *const [u32; 256], right: *const [u32; 256]) -> u32 {
  // SAFETY: Both array borrows satisfy the caller contract and stay in this call.
  let (Some(left), Some(right)) = (unsafe { (left.as_ref(), right.as_ref()) }) else {
    return 0;
  };
  auth::diag_mldsa_product(left, right)
}

/// Execute production ML-DSA accumulation. Null inputs are rejected.
///
/// # Safety
/// All inputs must be aligned, readable for 256 initialized u32s, immutable
/// during the call, and contain canonical residues below q. They may overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ct_entry_mldsa_accumulate(
  accumulator: *const [u32; 256],
  left: *const [u32; 256],
  right: *const [u32; 256],
) -> u32 {
  // SAFETY: All array borrows satisfy the caller contract and stay in this call.
  let (Some(accumulator), Some(left), Some(right)) = (unsafe { (accumulator.as_ref(), left.as_ref(), right.as_ref()) })
  else {
    return 0;
  };
  auth::diag_mldsa_accumulate(accumulator, left, right)
}

macro_rules! seed_entry {
  ($name:ident, $operation:expr) => {
    /// Execute a production ML-DSA sampler. Null is rejected.
    ///
    /// # Safety
    /// A non-null input must be readable for 64 initialized bytes and immutable
    /// for the duration of this call. The borrow never escapes.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn $name(seed: *const [u8; 64]) -> u32 {
      // SAFETY: The caller supplies a live, initialized, immutable byte array.
      let seed = unsafe { seed.as_ref() };
      let Some(seed) = seed else {
        return 0;
      };
      ($operation)(seed)
    }
  };
}
seed_entry!(ct_entry_mldsa_noise_eta2, |seed| {
  let (valid, digest) = auth::diag_mldsa_noise(seed, 2);
  core::hint::black_box(valid);
  digest
});
seed_entry!(ct_entry_mldsa_noise_eta4, |seed| {
  let (valid, digest) = auth::diag_mldsa_noise(seed, 4);
  core::hint::black_box(valid);
  digest
});
seed_entry!(ct_entry_mldsa_challenge44, |seed: &[u8; 64]| {
  let (valid, digest) = auth::diag_mldsa_challenge(&seed[..32], 39);
  core::hint::black_box(valid);
  digest
});
seed_entry!(ct_entry_mldsa_challenge65, |seed: &[u8; 64]| {
  let (valid, digest) = auth::diag_mldsa_challenge(&seed[..48], 49);
  core::hint::black_box(valid);
  digest
});
seed_entry!(ct_entry_mldsa_challenge87, |seed: &[u8; 64]| {
  let (valid, digest) = auth::diag_mldsa_challenge(seed, 60);
  core::hint::black_box(valid);
  digest
});
seed_entry!(ct_entry_mldsa_mask17, |seed| auth::diag_mldsa_mask(seed, false));
seed_entry!(ct_entry_mldsa_mask19, |seed| auth::diag_mldsa_mask(seed, true));

macro_rules! preparation_entry {
  ($name:ident, $size:literal, $operation:path) => {
    /// Decode and transform production secret-key components. Null is rejected.
    ///
    /// # Safety
    /// A non-null pointer must be readable for the parameter set's complete
    /// initialized encoding and immutable throughout the call.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn $name(secret: *const [u8; $size]) -> u8 {
      // SAFETY: The caller supplies the complete immutable encoding. The borrow
      // ends before this function returns; production decoding validates noise.
      let secret = unsafe { secret.as_ref() };
      let Some(secret) = secret else {
        return 0;
      };
      u8::from($operation(secret))
    }
  };
}
preparation_entry!(ct_entry_mldsa_prepare44, 2560, auth::diag_mldsa_prepare44);
preparation_entry!(ct_entry_mldsa_prepare65, 4032, auth::diag_mldsa_prepare65);
preparation_entry!(ct_entry_mldsa_prepare87, 4896, auth::diag_mldsa_prepare87);
