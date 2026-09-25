//! FIPS 204 Algorithms 29–34, with Appendix C output bounds.

use super::{
  MlDsaError, Parameters,
  poly::{N, Poly, Q, select, to_montgomery},
};
use crate::{
  hashes::crypto::keccak::{KeccakCore, KeccakXof, PublicKeccakXof, xof_seeded_32_2_pair},
  secret::ZeroizingBytes,
};

const NOISE_BYTES: usize = 481;
const NOISE_CANDIDATES: usize = NOISE_BYTES * 2;
const NOISE_INPUT_PLANE_BYTES: usize = 4 * 8;
const NOISE_PLANE_BYTES: usize = 4 * 4 * 8;
const CHALLENGE_BYTES: usize = 221;

/// Return one when `value` is non-zero and zero otherwise.
#[inline]
const fn nonzero_bit(value: u32) -> u32 {
  (value | value.wrapping_neg()) >> 31
}

/// Compare values known to be below 2^31 without data-dependent control flow.
#[inline]
const fn less_than_bit(left: u32, right: u32) -> u32 {
  left.wrapping_sub(right) >> 31
}

#[inline]
const fn equal_bit(left: u32, right: u32) -> u32 {
  nonzero_bit(left ^ right) ^ 1
}

#[inline]
fn mask_u64(choice: u32) -> u64 {
  // Each caller supplies one bit. Hide that range before widening so LLVM
  // cannot replace word selections with branches on secret sampler state.
  0u64.wrapping_sub(u64::from(super::poly::opaque_mask(choice)))
}

#[inline]
fn select_u64(left: u64, right: u64, choice: u32) -> u64 {
  let mask = mask_u64(choice);
  (left & !mask) | (right & mask)
}

#[inline]
fn fixed_shl(mut value: u64, shift: u32) -> u64 {
  let mut bit = 0u32;
  while bit < 6 {
    let distance = 1 << bit;
    value = select_u64(value, value << distance, (shift >> bit) & 1);
    bit = bit.strict_add(1);
  }
  value
}

#[inline]
fn fixed_shr(mut value: u64, shift: u32) -> u64 {
  let mut bit = 0u32;
  while bit < 6 {
    let distance = 1 << bit;
    value = select_u64(value, value >> distance, (shift >> bit) & 1);
    bit = bit.strict_add(1);
  }
  value
}

#[inline]
fn popcount_u64(mut value: u64) -> u32 {
  // These word-parallel additions and subtraction are intentionally modular.
  value = value.wrapping_sub((value >> 1) & 0x5555_5555_5555_5555);
  value = (value & 0x3333_3333_3333_3333).wrapping_add((value >> 2) & 0x3333_3333_3333_3333);
  value = value.wrapping_add(value >> 4) & 0x0f0f_0f0f_0f0f_0f0f;
  value = value.wrapping_add(value >> 8);
  value = value.wrapping_add(value >> 16);
  value = value.wrapping_add(value >> 32);
  u32::try_from(value & 0x7f).expect("a 64-bit population count fits u32")
}

#[inline]
fn compress_bits(mut value: u64, mut mask: u64) -> u64 {
  // Hacker's Delight compress-right network: fixed shifts and register masks,
  // equivalent to extracting the bits selected by `mask` in original order.
  value &= mask;
  let mut movable = !mask << 1;
  for bit in 0..6 {
    let mut prefix = movable ^ (movable << 1);
    prefix ^= prefix << 2;
    prefix ^= prefix << 4;
    prefix ^= prefix << 8;
    prefix ^= prefix << 16;
    prefix ^= prefix << 32;
    let selected = prefix & mask;
    mask = (mask ^ selected) | (selected >> (1 << bit));
    let moved = value & selected;
    value = (value ^ moved) | (moved >> (1 << bit));
    movable &= !prefix;
  }
  value
}

#[inline]
fn read_u64_word<const BYTES: usize>(bytes: &[u8; BYTES], word: usize) -> u64 {
  let offset = word.strict_mul(8);
  u64::from_le_bytes(
    bytes[offset..offset.strict_add(8)]
      .try_into()
      .expect("fixed-width word"),
  )
}

#[inline]
fn write_u64_word<const BYTES: usize>(bytes: &mut [u8; BYTES], word: usize, value: u64) {
  let offset = word.strict_mul(8);
  bytes[offset..offset.strict_add(8)].copy_from_slice(&value.to_le_bytes());
}

fn append_noise_plane(planes: &mut [u8; NOISE_PLANE_BYTES], plane: usize, packed: u64, accepted: u32) {
  let word_index = accepted >> 6;
  let bit_index = accepted & 63;
  let low = fixed_shl(packed, bit_index);
  let high_shift = bit_index.wrapping_neg() & 63;
  let high = fixed_shr(packed, high_shift) & mask_u64(nonzero_bit(bit_index));
  for word in 0..4 {
    let word = u32::try_from(word).expect("noise output word index fits u32");
    let low_mask = mask_u64(equal_bit(word_index, word));
    let high_mask = mask_u64(equal_bit(word_index.strict_add(1), word));
    let output_word = plane
      .strict_mul(4)
      .strict_add(usize::try_from(word).expect("word index fits usize"));
    let output = read_u64_word(planes, output_word);
    write_u64_word(planes, output_word, output | (low & low_mask) | (high & high_mask));
  }
}

pub(super) fn xof<const RATE: usize>(parts: &[&[u8]]) -> KeccakXof<RATE> {
  let mut state = KeccakCore::<RATE>::default();
  for part in parts {
    state.update(part);
  }
  state.finalize_xof(0x1f)
}

pub(super) fn hash(parts: &[&[u8]], out: &mut [u8]) {
  xof::<136>(parts).squeeze_into(out);
}

pub(super) fn matrix(rho: &[u8], row: u8, column: u8, out: &mut [u32; N]) -> Result<(), MlDsaError> {
  let mut reader = xof::<168>(&[rho, &[column, row]]);
  let mut block = [0u8; 168];
  let mut count = 0usize;
  let mut remaining = 298usize;
  // At least 894 bytes are required by FIPS 204 Appendix C.
  while remaining != 0 {
    let candidates = remaining.min(56);
    let bytes = &mut block[..candidates.strict_mul(3)];
    reader.squeeze_into(bytes);
    for bytes in bytes.as_chunks::<3>().0 {
      let value = u32::from_le_bytes([bytes[0], bytes[1], bytes[2] & 0x7f, 0]);
      if value < Q {
        out[count] = to_montgomery(value);
        count = count.strict_add(1);
        if count == N {
          return Ok(());
        }
      }
    }
    remaining = remaining.strict_sub(candidates);
  }
  Err(MlDsaError::RejectionLimit)
}

// Both streams and their rejection decisions depend only on the public rho seed.
// Keep their squeeze positions equal so the shared Keccak backend can permute
// two independent states together. Each stream retains the 298-candidate bound.
pub(super) fn matrix_pair(
  rho: &[u8],
  row: u8,
  columns: [u8; 2],
  out_a: &mut [u32; N],
  out_b: &mut [u32; N],
) -> Result<(), MlDsaError> {
  let seed = rho.try_into().expect("ML-DSA rho is 32 bytes");
  let (mut a, mut b) = xof_seeded_32_2_pair::<168>(0x1f, seed, (columns[0], row), (columns[1], row));
  let mut block_a = [0u8; 168];
  let mut block_b = [0u8; 168];
  let mut counts = [0usize; 2];
  let mut remaining = 298usize;
  while remaining != 0 {
    let candidates = remaining.min(56);
    let bytes = candidates.strict_mul(3);
    PublicKeccakXof::squeeze_pair_into(&mut a, &mut b, &mut block_a[..bytes], &mut block_b[..bytes]);
    for ((block, out), count) in [(&block_a, &mut *out_a), (&block_b, &mut *out_b)]
      .into_iter()
      .zip(&mut counts)
    {
      if *count == N {
        continue;
      }
      for candidate in block[..bytes].as_chunks::<3>().0 {
        let value = u32::from_le_bytes([candidate[0], candidate[1], candidate[2] & 0x7f, 0]);
        if value < Q {
          out[*count] = to_montgomery(value);
          *count = count.strict_add(1);
          if *count == N {
            break;
          }
        }
      }
    }
    if counts == [N; 2] {
      return Ok(());
    }
    remaining = remaining.strict_sub(candidates);
  }
  Err(MlDsaError::RejectionLimit)
}

pub(super) fn noise(seed: &[u8], nonce: u16, eta: u32, out: &mut Poly) -> Result<(), MlDsaError> {
  let mut bytes = ZeroizingBytes::<NOISE_BYTES>::zeroed();
  xof::<136>(&[seed, &nonce.to_le_bytes()]).squeeze_into(bytes.as_mut_array());
  noise_from_bytes(bytes.as_array(), eta, out)
}

fn noise_from_bytes(bytes: &[u8; NOISE_BYTES], eta: u32, out: &mut Poly) -> Result<(), MlDsaError> {
  let limit = if eta == 2 { 15 } else { 9 };
  let mut output_planes = ZeroizingBytes::<NOISE_PLANE_BYTES>::zeroed();
  let mut accepted = 0u32;
  for block in 0usize..16 {
    let mut acceptance = 0u64;
    let mut input_planes = ZeroizingBytes::<NOISE_INPUT_PLANE_BYTES>::zeroed();
    for lane in 0usize..64 {
      let index = block.strict_mul(64).strict_add(lane);
      let (nibble, accept) = if index < NOISE_CANDIDATES {
        let byte = bytes[index / 2];
        let nibble = if index & 1 == 0 { byte & 15 } else { byte >> 4 };
        (nibble, less_than_bit(u32::from(nibble), limit))
      } else {
        (0, 0)
      };
      acceptance |= u64::from(accept) << lane;
      for bit in 0..4 {
        let plane = read_u64_word(input_planes.as_array(), bit);
        write_u64_word(
          input_planes.as_mut_array(),
          bit,
          plane | (u64::from((nibble >> bit) & 1) << lane),
        );
      }
    }
    for plane in 0usize..4 {
      append_noise_plane(
        output_planes.as_mut_array(),
        plane,
        compress_bits(read_u64_word(input_planes.as_array(), plane), acceptance),
        accepted,
      );
    }
    // There are only 962 candidates. This addition is intentionally modular
    // so overflow checks cannot branch on the secret-derived acceptance count.
    accepted = accepted.wrapping_add(popcount_u64(acceptance));
  }

  for (index, output) in out.0.iter_mut().enumerate() {
    let word = index / 64;
    let bit = index & 63;
    let mut n = 0u32;
    for plane in 0usize..4 {
      let output_word = plane.strict_mul(4).strict_add(word);
      n |= u32::try_from((read_u64_word(output_planes.as_array(), output_word) >> bit) & 1).expect("one bit fits u32")
        << plane;
    }
    // For eta=2, reduce the accepted range 0..15 modulo five with masked
    // subtraction. eta is a public parameter-set constant.
    let value = if eta == 2 {
      n.wrapping_sub(5u32.strict_mul(less_than_bit(n, 5) ^ 1))
        .wrapping_sub(5u32.strict_mul(less_than_bit(n, 10) ^ 1))
    } else {
      n
    };
    // `value` is in 0..=2*eta. Reduce eta-value modulo q using explicitly
    // modular arithmetic so debug overflow checks cannot introduce a branch
    // on this secret coefficient.
    let difference = eta.wrapping_add(Q).wrapping_sub(value);
    let reduced = difference.wrapping_sub(Q);
    *output = reduced.wrapping_add(0u32.wrapping_sub(reduced >> 31) & Q);
  }

  if less_than_bit(accepted, 256) == 0 {
    Ok(())
  } else {
    Err(MlDsaError::RejectionLimit)
  }
}

pub(super) fn mask(seed: &[u8], nonce: u16, p: Parameters, out: &mut Poly) {
  let mut bytes = ZeroizingBytes::<640>::zeroed();
  let len = p.z_bits.strict_mul(32);
  xof::<136>(&[seed, &nonce.to_le_bytes()]).squeeze_into(&mut bytes.as_mut_array()[..len]);
  super::encoding::unpack(&bytes.as_array()[..len], p.z_bits, p.gamma1, out);
}

pub(super) fn challenge(seed: &[u8], tau: usize, out: &mut Poly) -> Result<(), MlDsaError> {
  let mut bytes = ZeroizingBytes::<CHALLENGE_BYTES>::zeroed();
  xof::<136>(&[seed]).squeeze_into(bytes.as_mut_array());
  challenge_from_bytes(bytes.as_array(), tau, out)
}

pub(super) fn challenge_public(seed: &[u8], tau: usize, out: &mut Poly) -> Result<(), MlDsaError> {
  let mut bytes = ZeroizingBytes::<CHALLENGE_BYTES>::zeroed();
  xof::<136>(&[seed]).squeeze_into(bytes.as_mut_array());
  let b = bytes.as_array();
  let mut signs = u64::from_le_bytes(b[..8].try_into().expect("fixed challenge sign prefix"));
  let mut offset = 8usize;
  out.0.fill(0);
  for i in N.strict_sub(tau)..N {
    let j = loop {
      let Some(&candidate) = b.get(offset) else {
        return Err(MlDsaError::RejectionLimit);
      };
      offset = offset.strict_add(1);
      if usize::from(candidate) <= i {
        break usize::from(candidate);
      }
    };
    let sign = select(1, Q.strict_sub(1), u32::from(signs & 1 != 0));
    out.0[i] = out.0[j];
    out.0[j] = sign;
    signs >>= 1;
  }
  Ok(())
}

fn challenge_from_bytes(bytes: &[u8; CHALLENGE_BYTES], tau: usize, out: &mut Poly) -> Result<(), MlDsaError> {
  if challenge_from_bytes_valid(bytes, tau, out) == 1 {
    Ok(())
  } else {
    Err(MlDsaError::RejectionLimit)
  }
}

// Keep one fixed scan out of the outer tau loop. Inlining lets LLVM hoist
// expanded secret bytes into many spill slots and greatly enlarges the frame.
#[inline(never)]
fn first_challenge_candidate(bytes: &[u8; CHALLENGE_BYTES], offset: u32, limit: u32) -> u32 {
  // Pack position above value so an unsigned minimum chooses the first
  // eligible byte. Independent candidates let the compiler parallelize the
  // fixed scan without a secret-dependent address or early exit.
  let mut first = u32::MAX;
  for (index, &candidate) in bytes[8..].iter().enumerate() {
    let position = u32::try_from(index)
      .expect("challenge buffer indices fit u32")
      .strict_add(8);
    let candidate = u32::from(candidate);
    let after_offset = less_than_bit(position, offset) ^ 1;
    let in_range = less_than_bit(limit, candidate) ^ 1;
    let packed = (position << 8) | candidate;
    let candidate = select(u32::MAX, packed, after_offset & in_range);
    #[cfg(any(target_arch = "s390x", target_arch = "riscv64", target_arch = "riscv32"))]
    {
      // Keep the minimum behind the same register barrier as selection. On
      // these targets an unprotected integer minimum can become a secret branch.
      first = select(first, candidate, u32::from(candidate < first));
    }
    #[cfg(not(any(target_arch = "s390x", target_arch = "riscv64", target_arch = "riscv32")))]
    {
      first = first.min(candidate);
    }
  }
  first
}

fn challenge_from_bytes_valid(bytes: &[u8; CHALLENGE_BYTES], tau: usize, out: &mut Poly) -> u32 {
  let b = bytes;
  let mut signs = u64::from_le_bytes(b[..8].try_into().expect("fixed challenge sign prefix"));
  let mut offset = 8u32;
  let mut valid = 1u32;
  out.0.fill(0);
  for i in N.strict_sub(tau)..N {
    let i_u32 = u32::try_from(i).expect("ML-DSA polynomial indices fit u32");
    let first = first_challenge_candidate(b, offset, i_u32);
    let found = nonzero_bit(first ^ u32::MAX);
    let j = select(0, first & 255, found);
    offset = select(offset, (first >> 8).strict_add(1), found);
    valid &= found;
    // A signing candidate's challenge is not public until that candidate is
    // accepted. Scan both the XOF buffer and polynomial instead of indexing
    // either one with secret-derived rejection state.
    let sign = select(1, Q.strict_sub(1), u32::from(signs & 1 != 0));
    let mut previous = 0;
    for (index, coefficient) in out.0.iter_mut().enumerate() {
      let index = u32::try_from(index).expect("ML-DSA polynomial indices fit u32");
      let hit = equal_bit(index, j);
      previous |= select(0, *coefficient, hit);
      *coefficient = select(*coefficient, sign, hit);
    }
    out.0[i] = select(previous, sign, equal_bit(i_u32, j));
    signs >>= 1;
  }
  valid
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn paired_matrix_preserves_stream_order_and_coefficients() {
    // Compare distinct production paths: paired public SHAKE against the
    // original scalar stream, including reversed and repeated coordinates.
    for seed_byte in [0u8, 1, 127, 255] {
      for row in 0..8 {
        for columns in [[0, 1], [4, 5], [6, 0], [2, 2]] {
          let seed = [seed_byte; 32];
          let mut a = [0; N];
          let mut b = [0; N];
          let mut expected_a = [0; N];
          let mut expected_b = [0; N];
          matrix(&seed, row, columns[0], &mut expected_a).expect("scalar matrix");
          matrix(&seed, row, columns[1], &mut expected_b).expect("scalar matrix");
          matrix_pair(&seed, row, columns, &mut a, &mut b).expect("paired matrix");
          assert_eq!(a, expected_a);
          assert_eq!(b, expected_b);
        }
      }
    }
  }

  fn set_nibble(bytes: &mut [u8; NOISE_BYTES], index: usize, nibble: u8) {
    if index & 1 == 0 {
      bytes[index / 2] = (bytes[index / 2] & 0xf0) | nibble;
    } else {
      bytes[index / 2] = (bytes[index / 2] & 0x0f) | (nibble << 4);
    }
  }

  #[test]
  fn bounded_noise_compaction_preserves_order_across_blocks() {
    for (eta, modulus) in [(2, 5u8), (4, 9u8)] {
      let mut bytes = [0xff; NOISE_BYTES];
      let mut expected = [0u32; N];
      for (coefficient, output) in expected.iter_mut().enumerate() {
        let nibble = u8::try_from(coefficient % usize::from(modulus)).expect("test nibble fits u8");
        set_nibble(&mut bytes, coefficient.strict_mul(3), nibble);
        let value = u32::from(nibble);
        *output = if eta >= value { eta - value } else { Q - (value - eta) };
      }

      let mut output = Poly::zero();
      noise_from_bytes(&bytes, eta, &mut output).expect("exactly 256 candidates are admissible");
      assert_eq!(output.0, expected);
    }
  }

  #[test]
  fn bounded_noise_compaction_accepts_first_coefficients_and_reports_exhaustion() {
    for eta in [2, 4] {
      let mut output = Poly::zero();
      noise_from_bytes(&[0; NOISE_BYTES], eta, &mut output).expect("zero nibbles are admissible");
      assert!(output.0.iter().all(|&coefficient| coefficient == eta));

      let mut bytes = [0xff; NOISE_BYTES];
      for index in 0..N.strict_sub(1) {
        set_nibble(&mut bytes, index, 0);
      }
      assert_eq!(
        noise_from_bytes(&bytes, eta, &mut output),
        Err(MlDsaError::RejectionLimit)
      );
      // Only the final available nibble supplies coefficient 256. This also
      // checks that padding in the last 64-candidate block is never accepted.
      set_nibble(&mut bytes, NOISE_CANDIDATES.strict_sub(1), 0);
      noise_from_bytes(&bytes, eta, &mut output).expect("the final candidate completes the polynomial");
      assert!(output.0.iter().all(|&coefficient| coefficient == eta));
    }
  }

  #[test]
  fn bounded_challenge_scan_reports_exhaustion() {
    for tau in [super::super::P44.tau, super::super::P65.tau, super::super::P87.tau] {
      let mut output = Poly::zero();
      assert_eq!(
        challenge_from_bytes(&[0xff; CHALLENGE_BYTES], tau, &mut output),
        Err(MlDsaError::RejectionLimit)
      );
      let mut bytes = [0xff; CHALLENGE_BYTES];
      bytes[..8].fill(0);
      let start = CHALLENGE_BYTES.strict_sub(tau);
      for (candidate, index) in bytes[start..].iter_mut().zip(N.strict_sub(tau)..N) {
        *candidate = u8::try_from(index).expect("polynomial index fits a byte");
      }
      // Reject the prefix, then select j=i for each remaining coefficient.
      // The last selection consumes byte 221; zero sign bits select +1.
      challenge_from_bytes(&bytes, tau, &mut output).expect("the final byte completes the challenge");
      assert!(
        output.0[..N.strict_sub(tau)]
          .iter()
          .all(|&coefficient| coefficient == 0)
      );
      assert!(
        output.0[N.strict_sub(tau)..]
          .iter()
          .all(|&coefficient| coefficient == 1)
      );
      bytes[start] = 0xff;
      assert_eq!(
        challenge_from_bytes(&bytes, tau, &mut output),
        Err(MlDsaError::RejectionLimit)
      );
    }
  }
}
