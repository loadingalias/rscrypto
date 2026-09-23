//! Original portable implementation of FIPS 204 Algorithms 6–8.
//!
//! Matrix rows are expanded on demand to bound stack use independently of k*l.
//! Secret polynomial owners are cleared on every return path.

use super::{
  MlDsaError, Parameters, encoding,
  poly::{Poly, add, decompose, high_bits, power2_round, use_hint},
  sampling,
};
use crate::secret::ZeroizingBytes;

fn index_byte(index: usize) -> u8 {
  u8::try_from(index).expect("ML-DSA matrix dimensions are at most eight")
}

fn nonce(index: usize) -> u16 {
  u16::try_from(index).expect("bounded ML-DSA nonce fits two bytes")
}

fn vector<const LEN: usize>() -> [Poly; LEN] {
  core::array::from_fn(|_| Poly::zero())
}

// Matrix coefficients depend only on the public rho seed. They need no secret
// Drop owner. Expansion writes into the prepared owner to avoid a matrix-sized
// temporary frame.
pub(super) struct Matrix<const K: usize, const L: usize>([[[u32; 256]; L]; K]);

impl<const K: usize, const L: usize> Matrix<K, L> {
  pub(super) const fn zero() -> Self {
    Self([[[0; 256]; L]; K])
  }

  pub(super) fn expand_into(&mut self, rho: &[u8]) -> Result<(), MlDsaError> {
    if !cfg!(all(
      target_os = "macos",
      target_arch = "aarch64",
      not(feature = "portable-only")
    )) {
      for (i, row) in self.0.iter_mut().enumerate() {
        for (j, entry) in row.iter_mut().enumerate() {
          sampling::matrix(rho, index_byte(i), index_byte(j), entry)?;
        }
      }
      return Ok(());
    }
    for (i, row) in self.0.iter_mut().enumerate() {
      let (pairs, tail) = row.as_chunks_mut::<2>();
      for (j, [a, b]) in pairs.iter_mut().enumerate() {
        let column = j.strict_mul(2);
        sampling::matrix_pair(
          rho,
          index_byte(i),
          [index_byte(column), index_byte(column.strict_add(1))],
          a,
          b,
        )?;
      }
      if let [entry] = tail {
        sampling::matrix(rho, index_byte(i), index_byte(L.strict_sub(1)), entry)?;
      }
    }
    Ok(())
  }

  fn row(&self, i: usize, input: &[Poly; L], out: &mut Poly) {
    out.0.fill(0);
    for (entry, input) in self.0[i].iter().zip(input) {
      out.accumulate_product(entry, input);
    }
    out.inverse_ntt();
  }
}

fn matrix_row<const L: usize>(rho: &[u8], row: usize, input: &[Poly; L], out: &mut Poly) -> Result<(), MlDsaError> {
  out.0.fill(0);
  if !cfg!(all(
    target_os = "macos",
    target_arch = "aarch64",
    not(feature = "portable-only")
  )) {
    let mut entry = Poly::zero();
    for (column, input) in input.iter().enumerate() {
      sampling::matrix(rho, index_byte(row), index_byte(column), &mut entry.0)?;
      out.accumulate_product(&entry.0, input);
    }
    out.inverse_ntt();
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
      index_byte(row),
      [index_byte(column), index_byte(column.strict_add(1))],
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
  out.inverse_ntt();
  Ok(())
}

pub(super) fn keygen<const K: usize, const L: usize>(
  seed: &[u8; 32],
  p: Parameters,
  public: &mut [u8],
  secret: &mut [u8],
) -> Result<(), MlDsaError> {
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
    poly.ntt();
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
    matrix_row(rho, i, &s1, &mut row)?;
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
  pub(super) fn zero() -> Self {
    Self {
      s1: vector(),
      s2: vector(),
      t0: vector(),
    }
  }

  pub(super) fn decode(&mut self, secret: &[u8], p: Parameters) -> Result<(), MlDsaError> {
    let mut offset = 128usize;
    let size = p.eta_bits.strict_mul(32);
    let mut valid = true;
    for poly in self.s1.iter_mut().chain(self.s2.iter_mut()) {
      valid &= encoding::decode_noise(&secret[offset..offset.strict_add(size)], p, poly);
      offset = offset.strict_add(size);
      poly.ntt();
    }
    for poly in &mut self.t0 {
      encoding::unpack(&secret[offset..offset.strict_add(416)], 13, 4096, poly);
      offset = offset.strict_add(416);
      poly.ntt();
    }
    if !valid {
      return Err(MlDsaError::InvalidSecretKey);
    }
    Ok(())
  }
}

/// Reconstruct the public key and validate the redundant t0 and tr fields.
pub(super) fn validate_secret<const K: usize, const L: usize>(
  secret: &[u8],
  p: Parameters,
  public: &mut [u8],
) -> Result<(), MlDsaError> {
  let mut state = SigningState::<K, L>::zero();
  state.decode(secret, p)?;
  public[..32].copy_from_slice(&secret[..32]);
  let mut difference = 0u32;
  let mut row = Poly::zero();
  let mut low = Poly::zero();
  let mut s2 = Poly::zero();
  let mut high = Poly::zero();
  for i in 0..K {
    matrix_row(&secret[..32], i, &state.s1, &mut row)?;
    s2.copy_from(&state.s2[i]);
    s2.inverse_ntt();
    row.add_assign(&s2);
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
    return Err(MlDsaError::InvalidSecretKey);
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
  let mut state = SigningState::<K, L>::zero();
  state.decode(secret, p)?;
  sign_with_state(secret, mu, random, p, signature, &state, None)
}

pub(super) fn sign_with_state<const K: usize, const L: usize>(
  secret: &[u8],
  mu: &[u8; 64],
  random: &[u8; 32],
  p: Parameters,
  signature: &mut [u8],
  state: &SigningState<K, L>,
  matrix: Option<&Matrix<K, L>>,
) -> Result<(), MlDsaError> {
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
    signature.fill(0);
    for (i, (y, z)) in y.iter_mut().zip(&mut z).enumerate() {
      sampling::mask(rho_prime.as_array(), nonce(attempt.strict_mul(L).strict_add(i)), p, y);
      z.copy_from(y);
      z.ntt();
    }
    let mut hash = crate::hashes::crypto::keccak::KeccakCore::<136>::default();
    hash.update(mu);
    for (i, row) in w.iter_mut().enumerate() {
      if let Some(matrix) = matrix {
        matrix.row(i, &z, row);
      } else {
        matrix_row(&secret[..32], i, &z, row)?;
      }
      for (&x, high) in row.0.iter().zip(&mut high.0) {
        *high = high_bits(x, p.gamma2);
      }
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
      for (&x, low) in row.0.iter().zip(&mut high.0) {
        *low = decompose(x, p.gamma2).1;
      }
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
      for (r, &t) in row.0.iter_mut().zip(&work.0) {
        let hint = high_bits(*r, p.gamma2) != high_bits(add(*r, t), p.gamma2);
        // Candidate hints remain in the zeroizing polynomial owner until every
        // rejection check passes. The candidate controls no output addresses.
        *r = u32::from(hint);
        count = count.strict_add(usize::from(hint));
      }
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
  verify_with_state::<K, L>(public, mu, signature, p, None)
}

pub(super) struct VerifyingState<const K: usize, const L: usize> {
  pub(super) tr: [u8; 64],
  matrix: Matrix<K, L>,
  t1: [[u32; 256]; K],
}

impl<const K: usize, const L: usize> VerifyingState<K, L> {
  pub(super) fn prepare(public: &[u8]) -> Result<Self, MlDsaError> {
    let mut state = Self {
      tr: [0; 64],
      matrix: Matrix::zero(),
      t1: [[0; 256]; K],
    };
    state.matrix.expand_into(&public[..32])?;
    sampling::hash(&[public], &mut state.tr);
    let mut poly = Poly::zero();
    for (i, t1) in state.t1.iter_mut().enumerate() {
      decode_public_poly(public, i, &mut poly);
      t1.copy_from_slice(&poly.0);
    }
    Ok(state)
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
  public: &[u8],
  mu: &[u8; 64],
  signature: &[u8],
  p: Parameters,
  prepared: Option<&VerifyingState<K, L>>,
) -> Result<(), MlDsaError> {
  if !encoding::valid_signature(signature, p) {
    return Err(MlDsaError::InvalidSignature);
  }
  let mut z = vector::<L>();
  let z_len = p.z_bits.strict_mul(32);
  for (i, z) in z.iter_mut().enumerate() {
    let start = p.challenge_bytes.strict_add(i.strict_mul(z_len));
    encoding::unpack(&signature[start..start.strict_add(z_len)], p.z_bits, p.gamma1, z);
    z.ntt();
  }
  let hints = &signature[p.challenge_bytes.strict_add(L.strict_mul(z_len))..];
  let mut challenge = Poly::zero();
  // Verification challenges are public; only unpublished signing candidates
  // need the fixed-address sampler.
  sampling::challenge_public(&signature[..p.challenge_bytes], p.tau, &mut challenge)?;
  challenge.ntt();
  let mut row = Poly::zero();
  let mut t1 = Poly::zero();
  let mut work = Poly::zero();
  let mut packed = [0u8; 192];
  let w1_len = p.w1_bits.strict_mul(32);
  let mut hash = crate::hashes::crypto::keccak::KeccakCore::<136>::default();
  hash.update(mu);
  let mut hint_start = 0usize;
  for i in 0..K {
    if let Some(state) = prepared {
      state.matrix.row(i, &z, &mut row);
      work.product(&challenge, &state.t1[i]);
    } else {
      matrix_row(&public[..32], i, &z, &mut row)?;
      decode_public_poly(public, i, &mut t1);
      work.product(&challenge, &t1.0);
    }
    work.inverse_ntt();
    row.sub_assign(&work);
    let hint_end = usize::from(hints[p.omega.strict_add(i)]);
    let mut next = hint_start;
    for (j, x) in row.0.iter_mut().enumerate() {
      let hint = next < hint_end && usize::from(hints[next]) == j;
      if hint {
        next = next.strict_add(1);
      }
      *x = use_hint(*x, hint, p.gamma2);
    }
    hint_start = hint_end;
    encoding::pack(&row, p.w1_bits, None, &mut packed[..w1_len]);
    hash.update(&packed[..w1_len]);
  }
  let mut expected = [0u8; 64];
  hash.finalize_xof(0x1f).squeeze_into(&mut expected[..p.challenge_bytes]);
  if expected[..p.challenge_bytes] != signature[..p.challenge_bytes] {
    return Err(MlDsaError::InvalidSignature);
  }
  Ok(())
}
