//! MGF1 mask generation (RFC 8017 Appendix B.2.1) over a fixed-output digest.
//!
//! RSA OAEP and PSS masks and the SLH-DSA SHA-2 message hash (FIPS 205
//! Section 11.2) use it.

use crate::traits::Digest;

/// Fill `out` with MGF1(`seed`, `out.len()`) using digest `D`.
pub(crate) fn mgf1<D>(seed: &[u8], out: &mut [u8])
where
  D: Digest,
{
  let mut counter = 0u32;
  let mut offset = 0usize;
  while offset < out.len() {
    let digest = D::digest_vectored(&[seed, &counter.to_be_bytes()]);
    let chunk_len = core::cmp::min(D::OUTPUT_SIZE, out.len().strict_sub(offset));
    if let Some(dst) = out.get_mut(offset..offset.strict_add(chunk_len)) {
      let src = digest.as_ref().get(..chunk_len).unwrap_or_default();
      dst.copy_from_slice(src);
    }
    offset = offset.strict_add(chunk_len);
    counter = counter.strict_add(1);
  }
}
