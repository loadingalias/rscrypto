//! FIPS 204 Algorithms 16–28. Wire polynomials use little-endian bit packing.

use super::{
  Parameters,
  poly::{Poly, sub},
};

/// Encode b-x (mod q) in `bits` bits, or encode x when b is absent.
pub(super) fn pack(poly: &Poly, bits: usize, upper: Option<u32>, out: &mut [u8]) {
  let mut buffer = 0u64;
  let mut available = 0usize;
  let mut offset = 0usize;
  for &x in &poly.0 {
    let value = upper.map_or(x, |b| sub(b, x));
    buffer |= u64::from(value) << available;
    available = available.strict_add(bits);
    while available >= 8 {
      out[offset] = buffer.to_le_bytes()[0];
      buffer >>= 8;
      available = available.strict_sub(8);
      offset = offset.strict_add(1);
    }
  }
}

pub(super) fn unpack(bytes: &[u8], bits: usize, upper: u32, out: &mut Poly) {
  unpack_raw(bytes, bits, out);
  for x in &mut out.0 {
    *x = sub(upper, *x);
  }
}

pub(super) fn unpack_raw(bytes: &[u8], bits: usize, out: &mut Poly) {
  let mut buffer = 0u64;
  let mut available = 0usize;
  let mut offset = 0usize;
  let mask = (1u64 << bits).strict_sub(1);
  for x in &mut out.0 {
    while available < bits {
      buffer |= u64::from(bytes[offset]) << available;
      offset = offset.strict_add(1);
      available = available.strict_add(8);
    }
    let b = (buffer & mask).to_le_bytes();
    *x = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    buffer >>= bits;
    available = available.strict_sub(bits);
  }
}

pub(super) fn decode_noise(bytes: &[u8], p: Parameters, out: &mut Poly) -> bool {
  unpack_raw(bytes, p.eta_bits, out);
  let mut invalid = 0u32;
  for x in &mut out.0 {
    invalid |= u32::from(*x > p.eta.strict_mul(2));
    *x = sub(p.eta, *x);
  }
  invalid == 0
}

/// Parse hint positions with strict order, cumulative counts, and zero padding.
pub(super) fn valid_hints(hints: &[u8], p: Parameters) -> bool {
  let mut start = 0usize;
  for &end in &hints[p.omega..] {
    let end = usize::from(end);
    if end < start || end > p.omega {
      return false;
    }
    if hints[start..end].windows(2).any(|pair| pair[0] >= pair[1]) {
      return false;
    }
    start = end;
  }
  hints[start..p.omega].iter().all(|&x| x == 0)
}

pub(super) fn valid_signature(signature: &[u8], p: Parameters) -> bool {
  if signature.len() != p.signature_len() {
    return false;
  }
  let z_len = p.z_bits.strict_mul(32);
  let hints_start = p.challenge_bytes.strict_add(p.l.strict_mul(z_len));
  if !valid_hints(&signature[hints_start..], p) {
    return false;
  }
  let mut z = Poly::zero();
  for bytes in signature[p.challenge_bytes..hints_start].chunks_exact(z_len) {
    unpack(bytes, p.z_bits, p.gamma1, &mut z);
    if z.exceeds_bound(p.gamma1.strict_sub(p.beta)) != 0 {
      return false;
    }
  }
  true
}
