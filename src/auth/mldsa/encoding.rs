//! FIPS 204 Algorithms 16–28. Wire polynomials use little-endian bit packing.

use super::{
  Parameters,
  poly::{N, Poly, sub},
};

// Each FIPS 204 width packs a fixed group of coefficients into whole bytes:
// eight per `BITS` bytes, or four per `BITS / 2` bytes for the 18- and 20-bit
// responses. A group fits one `u128`, so every field moves by constant shifts
// and masks: control flow and addresses depend only on the public width.
macro_rules! by_width {
  ($bits:expr, $fixed:ident($($argument:expr),*), $any:ident) => {
    match $bits {
      3 => $fixed::<3, 8, 3>($($argument),*),
      4 => $fixed::<4, 8, 4>($($argument),*),
      6 => $fixed::<6, 8, 6>($($argument),*),
      10 => $fixed::<10, 8, 10>($($argument),*),
      13 => $fixed::<13, 8, 13>($($argument),*),
      18 => $fixed::<18, 4, 9>($($argument),*),
      20 => $fixed::<20, 4, 10>($($argument),*),
      bits => $any($($argument,)* bits),
    }
  };
}

/// Encode b-x (mod q) in `bits` bits, or encode x when b is absent.
pub(super) fn pack(poly: &Poly, bits: usize, upper: Option<u32>, out: &mut [u8]) {
  by_width!(bits, pack_fixed(poly, upper, out), pack_any);
}

#[inline(always)]
fn pack_fixed<const BITS: usize, const GROUP: usize, const BYTES: usize>(
  poly: &Poly,
  upper: Option<u32>,
  out: &mut [u8],
) {
  const { assert!(GROUP * BITS == BYTES * 8 && BYTES <= 16) };
  let (chunks, _) = out[..const { BYTES * (N / GROUP) }].as_chunks_mut::<BYTES>();
  for (chunk, coefficients) in chunks.iter_mut().zip(poly.0.as_chunks::<GROUP>().0) {
    let mut group = 0u128;
    for (i, &x) in coefficients.iter().enumerate() {
      let value = upper.map_or(x, |b| sub(b, x));
      group |= u128::from(value) << i.strict_mul(BITS);
    }
    chunk.copy_from_slice(&group.to_le_bytes()[..BYTES]);
  }
}

/// Bit-serial packing for any width; the reference for [`pack_fixed`].
fn pack_any(poly: &Poly, upper: Option<u32>, out: &mut [u8], bits: usize) {
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
  by_width!(bits, unpack_fixed(bytes, out), unpack_any);
}

#[inline(always)]
fn unpack_fixed<const BITS: usize, const GROUP: usize, const BYTES: usize>(bytes: &[u8], out: &mut Poly) {
  const { assert!(GROUP * BITS == BYTES * 8 && BYTES <= 16) };
  let mask = (1u128 << BITS).strict_sub(1);
  let (chunks, _) = bytes[..const { BYTES * (N / GROUP) }].as_chunks::<BYTES>();
  for (chunk, coefficients) in chunks.iter().zip(out.0.as_chunks_mut::<GROUP>().0) {
    let mut wide = [0u8; 16];
    wide[..BYTES].copy_from_slice(chunk);
    let group = u128::from_le_bytes(wide);
    for (i, x) in coefficients.iter_mut().enumerate() {
      #[expect(clippy::cast_possible_truncation, reason = "the mask keeps at most 20 bits")]
      let field = ((group >> i.strict_mul(BITS)) & mask) as u32;
      *x = field;
    }
  }
}

/// Bit-serial unpacking for any width; the reference for [`unpack_fixed`].
fn unpack_any(bytes: &[u8], out: &mut Poly, bits: usize) {
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

/// Check the length, hint order and padding: every rule except the response norm.
pub(super) fn valid_signature_structure(signature: &[u8], p: Parameters) -> bool {
  if signature.len() != p.signature_len() {
    return false;
  }
  let hints_start = p.challenge_bytes.strict_add(p.l.strict_mul(p.z_bits.strict_mul(32)));
  valid_hints(&signature[hints_start..], p)
}

pub(super) fn valid_signature(signature: &[u8], p: Parameters) -> bool {
  if !valid_signature_structure(signature, p) {
    return false;
  }
  let z_len = p.z_bits.strict_mul(32);
  let hints_start = p.challenge_bytes.strict_add(p.l.strict_mul(z_len));
  let mut z = Poly::zero();
  for bytes in signature[p.challenge_bytes..hints_start].chunks_exact(z_len) {
    unpack(bytes, p.z_bits, p.gamma1, &mut z);
    if z.exceeds_bound(p.gamma1.strict_sub(p.beta)) != 0 {
      return false;
    }
  }
  true
}

#[cfg(test)]
mod tests {
  use super::*;

  const WIDTHS: [usize; 7] = [3, 4, 6, 10, 13, 18, 20];

  #[test]
  fn fixed_width_codec_matches_bit_serial_reference() {
    // Random streams plus all-zero and all-one fields exercise every field
    // position and both group boundaries; `upper` covers the subtracted form.
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      state
    };
    for bits in WIDTHS {
      let len = bits.strict_mul(32);
      let mut streams = vec![vec![0u8; len], vec![0xff; len]];
      for _ in 0..8 {
        streams.push((0..len).map(|_| next().to_le_bytes()[0]).collect());
      }
      for bytes in &streams {
        let (mut fixed, mut reference) = (Poly::zero(), Poly::zero());
        unpack_raw(bytes, bits, &mut fixed);
        unpack_any(bytes, &mut reference, bits);
        assert_eq!(fixed.0, reference.0, "{bits}-bit unpack");

        let (mut fixed_out, mut reference_out) = (vec![0u8; len], vec![0u8; len]);
        pack(&fixed, bits, None, &mut fixed_out);
        pack_any(&fixed, None, &mut reference_out, bits);
        assert_eq!(fixed_out, reference_out, "{bits}-bit pack");
        assert_eq!(&fixed_out, bytes, "{bits}-bit round trip");

        // b - x for x in [0, 2^bits) with b = 2^(bits-1), as z, t0 and noise use it.
        let upper = 1u32 << bits.strict_sub(1);
        unpack(bytes, bits, upper, &mut fixed);
        pack(&fixed, bits, Some(upper), &mut fixed_out);
        pack_any(&fixed, Some(upper), &mut reference_out, bits);
        assert_eq!(fixed_out, reference_out, "{bits}-bit subtracted pack");
        assert_eq!(&fixed_out, bytes, "{bits}-bit subtracted round trip");
      }
    }
  }
}
