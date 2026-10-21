//! Original portable implementation of FIPS 204 Algorithms 6–8.
//!
//! Matrix rows are expanded on demand to bound stack use independently of k*l.
//! Secret polynomial owners are cleared on every return path.

use super::{
  MlDsaError, MlDsaKeyError, Parameters, encoding,
  poly::{NttBlinding, Poly, power2_round},
  sampling,
};
use crate::secret::ZeroizingBytes;

fn index_byte(index: usize) -> u8 {
  u8::try_from(index).expect("ML-DSA matrix dimensions are at most eight")
}

/// Expand matrix entries two streams at a time where the platform permuter runs two Keccak
/// states together: SHA3 on AArch64 macOS and Linux, and AVX-512VL on Linux x86-64 when the CPU
/// has it (without it the paired call runs the two permutations in turn).
const PAIRED_MATRIX: bool = cfg!(all(
  any(
    all(target_arch = "aarch64", any(target_os = "macos", target_os = "linux")),
    all(target_arch = "x86_64", target_os = "linux")
  ),
  not(feature = "portable-only")
));

fn nonce(index: usize) -> u16 {
  u16::try_from(index).expect("bounded ML-DSA nonce fits two bytes")
}

fn vector<const LEN: usize>() -> [Poly; LEN] {
  core::array::from_fn(|_| Poly::zero())
}

// Matrix coefficients depend only on the public rho seed. They need no secret
// Drop owner. Expansion writes into caller-owned prepared storage to avoid a
// matrix-sized temporary frame.
pub(super) struct Matrix<const K: usize, const L: usize>([[[u32; 256]; L]; K]);

impl<const K: usize, const L: usize> Matrix<K, L> {
  pub(super) const fn zero() -> Self {
    Self([[[0; 256]; L]; K])
  }

  pub(super) fn expand_into(&mut self, rho: &[u8]) -> Result<(), MlDsaError> {
    if !PAIRED_MATRIX {
      for (i, row) in self.0.iter_mut().enumerate() {
        for (j, entry) in row.iter_mut().enumerate() {
          sampling::matrix(rho, index_byte(i), index_byte(j), entry)?;
        }
      }
      return Ok(());
    }
    // Pair entries in row-major order, across row boundaries, so every stream shares a
    // two-state permutation; K·L is even for every parameter set.
    const { assert!((K * L).is_multiple_of(2)) };
    let (pairs, _) = self.0.as_flattened_mut().as_chunks_mut::<2>();
    for (pair, [a, b]) in pairs.iter_mut().enumerate() {
      let first = pair.strict_mul(2);
      let second = first.strict_add(1);
      sampling::matrix_pair(rho, [position::<L>(first), position::<L>(second)], a, b)?;
    }
    Ok(())
  }

  /// Row `i` of `Â·input` for NTT-domain `input`, left in the NTT domain.
  // Out of line for signing's code generation; see `sign_with_state`.
  #[inline(never)]
  fn row_ntt(&self, i: usize, input: &[Poly; L], out: &mut Poly) {
    out.row_product(&self.0[i], input);
  }
}

/// Row `row` of `Â·input`, expanding `Â` from `rho`; NTT-domain input and output.
// Out of line for signing's code generation; see `sign_with_state`.
#[inline(never)]
fn matrix_row_ntt<const L: usize>(rho: &[u8], row: usize, input: &[Poly; L], out: &mut Poly) -> Result<(), MlDsaError> {
  out.0.fill(0);
  if !PAIRED_MATRIX {
    let mut entry = Poly::zero();
    for (column, input) in input.iter().enumerate() {
      sampling::matrix(rho, index_byte(row), index_byte(column), &mut entry.0)?;
      out.accumulate_product(&entry.0, input);
    }
    return Ok(());
  }
  // Matrix entries are public; only the accumulated result owns secret data.
  let mut entry_a = [0u32; 256];
  let mut entry_b = [0u32; 256];
  let (pairs, tail) = input.as_chunks::<2>();
  for (pair, [a, b]) in pairs.iter().enumerate() {
    let column = pair.strict_mul(2);
    sampling::matrix_pair(
      rho,
      [
        (index_byte(row), index_byte(column)),
        (index_byte(row), index_byte(column.strict_add(1))),
      ],
      &mut entry_a,
      &mut entry_b,
    )?;
    out.accumulate_product(&entry_a, a);
    out.accumulate_product(&entry_b, b);
  }
  if let [last] = tail {
    sampling::matrix(rho, index_byte(row), index_byte(L.strict_sub(1)), &mut entry_a)?;
    out.accumulate_product(&entry_a, last);
  }
  Ok(())
}

/// Rows `first` and `first + 1` of `Â·input`, expanding `Â` from `rho`; NTT-domain input and
/// output. Each column's two entries share one two-state stream, so an odd L leaves no
/// single-state stream.
#[inline(never)]
fn matrix_rows_ntt<const L: usize>(
  rho: &[u8],
  first: usize,
  input: &[Poly; L],
  [out_a, out_b]: [&mut Poly; 2],
) -> Result<(), MlDsaError> {
  if !PAIRED_MATRIX {
    matrix_row_ntt(rho, first, input, out_a)?;
    return matrix_row_ntt(rho, first.strict_add(1), input, out_b);
  }
  out_a.0.fill(0);
  out_b.0.fill(0);
  let rows = [index_byte(first), index_byte(first.strict_add(1))];
  let mut entry_a = [0u32; 256];
  let mut entry_b = [0u32; 256];
  for (column, input) in input.iter().enumerate() {
    let column = index_byte(column);
    sampling::matrix_pair(rho, [(rows[0], column), (rows[1], column)], &mut entry_a, &mut entry_b)?;
    out_a.accumulate_product(&entry_a, input);
    out_b.accumulate_product(&entry_b, input);
  }
  Ok(())
}

/// The `(row, column)` bytes of row-major matrix entry `index`.
fn position<const L: usize>(index: usize) -> (u8, u8) {
  (index_byte(index.strict_div(L)), index_byte(index.strict_rem(L)))
}

pub(super) fn keygen<const K: usize, const L: usize>(
  seed: &[u8; 32],
  p: Parameters,
  public: &mut [u8],
  secret: &mut [u8],
) -> Result<(), MlDsaError> {
  let _dit = crate::traits::ct::DataIndependentTiming::enter();
  let mut expanded = ZeroizingBytes::<128>::zeroed();
  sampling::hash(&[seed, &[index_byte(K), index_byte(L)]], expanded.as_mut_array());
  let bytes = expanded.as_array();
  let rho = &bytes[..32];
  public[..32].copy_from_slice(rho);
  secret[..32].copy_from_slice(rho);
  secret[32..64].copy_from_slice(&bytes[96..]);
  let noise_bytes = p.eta_bits.strict_mul(32);
  let s2_start = 128usize.strict_add(L.strict_mul(noise_bytes));
  let t0_start = s2_start.strict_add(K.strict_mul(noise_bytes));
  let blinding = NttBlinding::new(&bytes[96..]);
  let mut s1 = vector::<L>();
  for (i, poly) in s1.iter_mut().enumerate() {
    sampling::noise(&bytes[32..96], nonce(i), p.eta, poly)?;
    let start = 128usize.strict_add(i.strict_mul(noise_bytes));
    encoding::pack(
      poly,
      p.eta_bits,
      Some(p.eta),
      &mut secret[start..start.strict_add(noise_bytes)],
    );
    blinding.ntt(poly);
  }
  let mut row = Poly::zero();
  let mut s2 = Poly::zero();
  let mut high = Poly::zero();
  let mut low = Poly::zero();
  for i in 0..K {
    sampling::noise(&bytes[32..96], nonce(L.strict_add(i)), p.eta, &mut s2)?;
    let start = s2_start.strict_add(i.strict_mul(noise_bytes));
    encoding::pack(
      &s2,
      p.eta_bits,
      Some(p.eta),
      &mut secret[start..start.strict_add(noise_bytes)],
    );
    matrix_row_ntt(rho, i, &s1, &mut row)?;
    row.inverse_ntt();
    row.add_assign(&s2);
    for ((&coefficient, high), low) in row.0.iter().zip(&mut high.0).zip(&mut low.0) {
      (*high, *low) = power2_round(coefficient);
    }
    let pk_start = 32usize.strict_add(i.strict_mul(320));
    encoding::pack(&high, 10, None, &mut public[pk_start..pk_start.strict_add(320)]);
    let sk_start = t0_start.strict_add(i.strict_mul(416));
    encoding::pack(&low, 13, Some(4096), &mut secret[sk_start..sk_start.strict_add(416)]);
  }
  sampling::hash(&[public], &mut secret[64..128]);
  Ok(())
}

pub(super) struct SigningState<const K: usize, const L: usize> {
  s1: [Poly; L],
  s2: [Poly; K],
  t0: [Poly; K],
}

impl<const K: usize, const L: usize> SigningState<K, L> {
  pub(super) const fn zero() -> Self {
    Self {
      s1: [const { Poly::zero() }; L],
      s2: [const { Poly::zero() }; K],
      t0: [const { Poly::zero() }; K],
    }
  }

  /// Clear every transformed secret polynomial while the owner stays in place.
  pub(super) fn clear(&mut self) {
    for poly in self.s1.iter_mut().chain(&mut self.s2).chain(&mut self.t0) {
      poly.clear();
    }
  }

  #[cfg(test)]
  pub(super) fn is_clear(&self) -> bool {
    self
      .s1
      .iter()
      .chain(&self.s2)
      .chain(&self.t0)
      .all(|poly| poly.0.iter().all(|&x| x == 0))
  }

  /// Decode and transform the secret polynomials. Returns whether every noise
  /// coefficient was in range; the result is computed without early exit.
  #[must_use]
  pub(super) fn decode(&mut self, secret: &[u8], p: Parameters) -> bool {
    let _dit = crate::traits::ct::DataIndependentTiming::enter();
    let mut offset = 128usize;
    let size = p.eta_bits.strict_mul(32);
    let mut valid = true;
    let blinding = NttBlinding::new(&secret[32..64]);
    for poly in self.s1.iter_mut().chain(self.s2.iter_mut()) {
      valid &= encoding::decode_noise(&secret[offset..offset.strict_add(size)], p, poly);
      offset = offset.strict_add(size);
      blinding.ntt(poly);
    }
    for poly in &mut self.t0 {
      encoding::unpack(&secret[offset..offset.strict_add(416)], 13, 4096, poly);
      offset = offset.strict_add(416);
      blinding.ntt(poly);
    }
    valid
  }
}

/// Reconstruct the public key and validate the redundant t0 and tr fields.
pub(super) fn validate_secret<const K: usize, const L: usize>(
  secret: &[u8],
  p: Parameters,
  public: &mut [u8],
) -> Result<(), MlDsaKeyError> {
  let _dit = crate::traits::ct::DataIndependentTiming::enter();
  let mut state = SigningState::<K, L>::zero();
  if !state.decode(secret, p) {
    return Err(MlDsaKeyError::InvalidSecretKey);
  }
  public[..32].copy_from_slice(&secret[..32]);
  let mut difference = 0u32;
  let mut row = Poly::zero();
  let mut low = Poly::zero();
  let mut high = Poly::zero();
  for i in 0..K {
    // Matrix expansion fails only by exhausting its FIPS 204 sampling bound.
    matrix_row_ntt(&secret[..32], i, &state.s1, &mut row).map_err(|_| MlDsaKeyError::RejectionLimit)?;
    // Both terms are NTT-domain Montgomery residues; one inverse transform covers their sum.
    row.add_assign(&state.s2[i]);
    row.inverse_ntt();
    low.copy_from(&state.t0[i]);
    low.inverse_ntt();
    for ((&x, &expected_low), high) in row.0.iter().zip(&low.0).zip(&mut high.0) {
      let (h, l) = power2_round(x);
      *high = h;
      difference |= l ^ expected_low;
    }
    let start = 32usize.strict_add(i.strict_mul(320));
    encoding::pack(&high, 10, None, &mut public[start..start.strict_add(320)]);
  }
  let mut tr = [0u8; 64];
  sampling::hash(&[public], &mut tr);
  for (&a, &b) in tr.iter().zip(&secret[64..128]) {
    difference |= u32::from(a ^ b);
  }
  if difference != 0 {
    return Err(MlDsaKeyError::InvalidSecretKey);
  }
  Ok(())
}

/// Sign an already domain-separated message representative (Algorithm 7).
/// The caller owns the output in a zeroizing guard until this function succeeds.
pub(super) fn sign<const K: usize, const L: usize>(
  secret: &[u8],
  mu: &[u8; 64],
  random: &[u8; 32],
  p: Parameters,
  signature: &mut [u8],
) -> Result<(), MlDsaError> {
  let _dit = crate::traits::ct::DataIndependentTiming::enter();
  let mut state = SigningState::<K, L>::zero();
  if !state.decode(secret, p) {
    return Err(MlDsaError::InvalidSecretKey);
  }
  sign_with_state(secret, mu, random, p, signature, &state, None)
}

/// Signing-attempt counter for the signing-tail baseline (`tools/mldsa-tail`).
///
/// Only builds with `--cfg rscrypto_mldsa_attempts` contain it; production,
/// test, and constant-time evidence builds do not. It counts attempts across
/// every signature in the process and is meant for one measuring thread.
#[cfg(rscrypto_mldsa_attempts)]
pub(super) mod attempts {
  use core::sync::atomic::{AtomicUsize, Ordering};

  static ATTEMPTS: AtomicUsize = AtomicUsize::new(0);

  pub(super) fn record() {
    ATTEMPTS.store(ATTEMPTS.load(Ordering::Relaxed).strict_add(1), Ordering::Relaxed);
  }

  /// Return the attempts counted since the last call and reset the count.
  pub(crate) fn take() -> usize {
    let attempts = ATTEMPTS.load(Ordering::Relaxed);
    ATTEMPTS.store(0, Ordering::Relaxed);
    attempts
  }
}

// Kept out of line with both row products: when LLVM inlined this loop into
// the generic prepared-key callers, prepared signing became measurably slower.
#[inline(never)]
pub(super) fn sign_with_state<const K: usize, const L: usize>(
  secret: &[u8],
  mu: &[u8; 64],
  random: &[u8; 32],
  p: Parameters,
  signature: &mut [u8],
  state: &SigningState<K, L>,
  matrix: Option<&Matrix<K, L>>,
) -> Result<(), MlDsaError> {
  let _dit = crate::traits::ct::DataIndependentTiming::enter();
  let mut rho_prime = ZeroizingBytes::<64>::zeroed();
  sampling::hash(&[&secret[32..64], random, mu], rho_prime.as_mut_array());
  let mut y = vector::<L>();
  let mut z = vector::<L>();
  let mut w = vector::<K>();
  let mut challenge = Poly::zero();
  let mut work = Poly::zero();
  let mut high = Poly::zero();
  let mut packed_high = ZeroizingBytes::<192>::zeroed();
  let w1_len = p.w1_bits.strict_mul(32);
  let z_len = p.z_bits.strict_mul(32);
  let hints_start = p.challenge_bytes.strict_add(L.strict_mul(z_len));
  let mut commitment = ZeroizingBytes::<64>::zeroed();

  // FIPS 204 potential corrections (2026-07-31): the minimum cap is 821.
  // Each attempt consumes L nonces; 821*7 remains below 2^16.
  for attempt in 0usize..821 {
    #[cfg(rscrypto_mldsa_attempts)]
    attempts::record();
    signature.fill(0);
    // Nonces stay attempt·L + i; consecutive pairs share one two-state Keccak schedule.
    let first_nonce = attempt.strict_mul(L);
    let (pairs, last) = y.as_chunks_mut::<2>();
    for (pair_index, [first, second]) in pairs.iter_mut().enumerate() {
      let i = first_nonce.strict_add(pair_index.strict_mul(2));
      sampling::mask_pair(
        rho_prime.as_array(),
        [nonce(i), nonce(i.strict_add(1))],
        p,
        [first, second],
      );
    }
    if let Some(y) = last.first_mut() {
      sampling::mask(
        rho_prime.as_array(),
        nonce(first_nonce.strict_add(L).strict_sub(1)),
        p,
        y,
      );
    }
    for (z, y) in z.iter_mut().zip(&y) {
      z.copy_from(y);
      z.ntt();
    }
    let mut hash = crate::hashes::crypto::keccak::KeccakCore::<136>::default();
    hash.update(mu);
    for (i, row) in w.iter_mut().enumerate() {
      if let Some(matrix) = matrix {
        matrix.row_ntt(i, &z, row);
      } else {
        matrix_row_ntt(&secret[..32], i, &z, row)?;
      }
      row.inverse_ntt();
      row.high_bits_into(p.gamma2, &mut high);
      encoding::pack(&high, p.w1_bits, None, &mut packed_high.as_mut_array()[..w1_len]);
      hash.update(&packed_high.as_array()[..w1_len]);
    }
    hash
      .finalize_xof(0x1f)
      .squeeze_into(&mut commitment.as_mut_array()[..p.challenge_bytes]);
    sampling::challenge(&commitment.as_array()[..p.challenge_bytes], p.tau, &mut challenge)?;
    challenge.ntt();
    let mut invalid = 0u32;
    for ((z, y), s1) in z.iter_mut().zip(&y).zip(&state.s1) {
      z.product(&challenge, &s1.0);
      z.inverse_ntt();
      z.add_assign(y);
      invalid |= z.exceeds_bound(p.gamma1.strict_sub(p.beta));
    }
    for (row, s2) in w.iter_mut().zip(&state.s2) {
      work.product(&challenge, &s2.0);
      work.inverse_ntt();
      row.sub_assign(&work);
      row.low_bits_into(p.gamma2, &mut high);
      invalid |= high.exceeds_bound(p.gamma2.strict_sub(p.beta));
    }
    // This is the scheme's aggregate rejection decision, never an error oracle.
    if invalid != 0 {
      continue;
    }
    let mut count = 0usize;
    for (row, t0) in w.iter_mut().zip(&state.t0) {
      work.product(&challenge, &t0.0);
      work.inverse_ntt();
      invalid |= work.exceeds_bound(p.gamma2);
      // Candidate hints remain in the zeroizing polynomial owner until every
      // rejection check passes. The candidate controls no output addresses.
      count = count.strict_add(row.make_hints(&work, p.gamma2));
    }
    if invalid != 0 || count > p.omega {
      continue;
    }
    // The accepted hint positions are part of the public signature.
    count = 0;
    for (i, row) in w.iter().enumerate() {
      for (j, &hint) in row.0.iter().enumerate() {
        if hint != 0 {
          signature[hints_start.strict_add(count)] = index_byte(j);
          count = count.strict_add(1);
        }
      }
      signature[hints_start.strict_add(p.omega).strict_add(i)] = index_byte(count);
    }
    signature[..p.challenge_bytes].copy_from_slice(&commitment.as_array()[..p.challenge_bytes]);
    for (i, z) in z.iter().enumerate() {
      let start = p.challenge_bytes.strict_add(i.strict_mul(z_len));
      encoding::pack(
        z,
        p.z_bits,
        Some(p.gamma1),
        &mut signature[start..start.strict_add(z_len)],
      );
    }
    return Ok(());
  }
  Err(MlDsaError::RejectionLimit)
}

pub(super) fn verify<const K: usize, const L: usize>(
  public: &[u8],
  mu: &[u8; 64],
  signature: &[u8],
  p: Parameters,
) -> Result<(), MlDsaError> {
  verify_with_state::<K, L>(VerifyingKey::Encoded(public), mu, signature, p)
}

/// The verification key: an encoded key whose rows are expanded on demand, or
/// a prepared state.
#[derive(Clone, Copy)]
pub(super) enum VerifyingKey<'a, const K: usize, const L: usize> {
  Encoded(&'a [u8]),
  Prepared(&'a VerifyingState<K, L>),
}

pub(super) struct VerifyingState<const K: usize, const L: usize> {
  pub(super) tr: [u8; 64],
  matrix: Matrix<K, L>,
  t1: [[u32; 256]; K],
}

impl<const K: usize, const L: usize> VerifyingState<K, L> {
  // Copying one zero constant lets LLVM write caller storage in place. The
  // field-wise literal built a 64 KiB stack temporary for ML-DSA-87.
  const ZERO: Self = Self {
    tr: [0; 64],
    matrix: Matrix::zero(),
    t1: [[0; 256]; K],
  };

  pub(super) const fn zero() -> Self {
    Self::ZERO
  }

  /// Overwrite every field from `public` in place. All inputs are public.
  pub(super) fn prepare(&mut self, public: &[u8]) -> Result<(), MlDsaError> {
    self.matrix.expand_into(&public[..32])?;
    sampling::hash(&[public], &mut self.tr);
    let mut poly = Poly::zero();
    for (i, t1) in self.t1.iter_mut().enumerate() {
      decode_public_poly(public, i, &mut poly);
      t1.copy_from_slice(&poly.0);
    }
    Ok(())
  }
}

fn decode_public_poly(public: &[u8], i: usize, t1: &mut Poly) {
  let start = 32usize.strict_add(i.strict_mul(320));
  encoding::unpack_raw(&public[start..start.strict_add(320)], 10, t1);
  for x in &mut t1.0 {
    *x <<= 13;
  }
  t1.ntt();
}

pub(super) fn verify_with_state<const K: usize, const L: usize>(
  key: VerifyingKey<'_, K, L>,
  mu: &[u8; 64],
  signature: &[u8],
  p: Parameters,
) -> Result<(), MlDsaError> {
  if !encoding::valid_signature_structure(signature, p) {
    return Err(MlDsaError::InvalidSignature);
  }
  let mut z = vector::<L>();
  let z_len = p.z_bits.strict_mul(32);
  for (i, z) in z.iter_mut().enumerate() {
    let start = p.challenge_bytes.strict_add(i.strict_mul(z_len));
    encoding::unpack(&signature[start..start.strict_add(z_len)], p.z_bits, p.gamma1, z);
    // The response norm completes `encoding::valid_signature` on this single decoding.
    if z.exceeds_bound(p.gamma1.strict_sub(p.beta)) != 0 {
      return Err(MlDsaError::InvalidSignature);
    }
    z.ntt();
  }
  let hints = &signature[p.challenge_bytes.strict_add(L.strict_mul(z_len))..];
  let mut challenge = Poly::zero();
  // Verification challenges are public; only unpublished signing candidates
  // need the fixed-address sampler.
  sampling::challenge_public(&signature[..p.challenge_bytes], p.tau, &mut challenge)?;
  challenge.ntt();
  // Rows are taken in pairs, so an encoded key expands both rows' entries of each column
  // with one two-state stream; K is even for every parameter set.
  const { assert!(K.is_multiple_of(2)) };
  let mut rows = [Poly::zero(), Poly::zero()];
  let mut t1 = Poly::zero();
  let mut work = Poly::zero();
  let mut packed = [0u8; 192];
  let w1_len = p.w1_bits.strict_mul(32);
  let mut hash = crate::hashes::crypto::keccak::KeccakCore::<136>::default();
  hash.update(mu);
  let mut hint_start = 0usize;
  for first in (0..K).step_by(2) {
    let [row_a, row_b] = &mut rows;
    match key {
      VerifyingKey::Prepared(state) => {
        state.matrix.row_ntt(first, &z, row_a);
        state.matrix.row_ntt(first.strict_add(1), &z, row_b);
      }
      VerifyingKey::Encoded(public) => matrix_rows_ntt(&public[..32], first, &z, [row_a, row_b])?,
    }
    for (i, row) in (first..).zip(&mut rows) {
      // w'ᵢ = NTT⁻¹(Âᵢ·ẑ − ĉ·t̂1ᵢ·2^d): both terms are NTT-domain Montgomery
      // residues, so one inverse transform covers their difference.
      match key {
        VerifyingKey::Prepared(state) => work.product(&challenge, &state.t1[i]),
        VerifyingKey::Encoded(public) => {
          decode_public_poly(public, i, &mut t1);
          work.product(&challenge, &t1.0);
        }
      }
      row.sub_assign(&work);
      row.inverse_ntt();
      // The validated hint positions of row i are strictly increasing and below N.
      let hint_end = usize::from(hints[p.omega.strict_add(i)]);
      let mut row_hints = [0u32; 256];
      for &position in &hints[hint_start..hint_end] {
        row_hints[usize::from(position)] = 1;
      }
      row.use_hints(&row_hints, p.gamma2);
      hint_start = hint_end;
      encoding::pack(row, p.w1_bits, None, &mut packed[..w1_len]);
      hash.update(&packed[..w1_len]);
    }
  }
  let mut expected = [0u8; 64];
  hash.finalize_xof(0x1f).squeeze_into(&mut expected[..p.challenge_bytes]);
  if expected[..p.challenge_bytes] != signature[..p.challenge_bytes] {
    return Err(MlDsaError::InvalidSignature);
  }
  Ok(())
}
