//! Shared fuzz testing infrastructure for rscrypto.
//!
//! Provides deterministic input parsing and generic property-test harnesses
//! for AEAD, digest, MAC, and checksum primitives.

#[cfg(feature = "aead")]
use rscrypto::traits::Aead;
use rscrypto::traits::{Checksum, ChecksumCombine, Digest, Mac};

/// Early-return from a `fuzz_target!` closure when the fuzzer hasn't provided
/// enough input bytes to construct a meaningful test case.
#[macro_export]
macro_rules! some_or_return {
    ($e:expr) => {
        match $e {
            Some(v) => v,
            None => return,
        }
    };
}

/// Deterministic cursor over fuzzer-provided bytes.
///
/// Splits raw fuzz input into structured components (keys, nonces, data,
/// split ratios) without the overhead of the `Arbitrary` trait.
pub struct FuzzInput<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> FuzzInput<'a> {
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Extract exactly `N` bytes as a fixed-size array.
    #[inline]
    pub fn bytes<const N: usize>(&mut self) -> Option<[u8; N]> {
        if self.pos.strict_add(N) > self.data.len() {
            return None;
        }
        let arr = self.data[self.pos..self.pos.strict_add(N)]
            .try_into()
            .ok()?;
        self.pos = self.pos.strict_add(N);
        Some(arr)
    }

    /// Extract a single byte.
    #[inline]
    pub fn byte(&mut self) -> Option<u8> {
        let [b] = self.bytes::<1>()?;
        Some(b)
    }

    /// All remaining bytes.
    #[inline]
    pub fn rest(&self) -> &'a [u8] {
        &self.data[self.pos..]
    }

    /// Split remaining bytes into two parts using the next byte as a ratio.
    /// Consumes one control byte, then returns `(first, second)`.
    #[inline]
    pub fn split_rest(&mut self) -> Option<(&'a [u8], &'a [u8])> {
        let ratio = self.byte()?;
        let rest = self.rest();
        Some(rest.split_at(split_point(rest.len(), ratio)))
    }
}

/// Map a `ratio` byte (0–255) to a split position within `[0, len]`.
#[inline]
fn split_point(len: usize, ratio: u8) -> usize {
    if len == 0 {
        return 0;
    }
    // Map both endpoints: 0 -> 0 and 255 -> len.
    // len * ratio fits in usize for any practical fuzz input size.
    (len.wrapping_mul(ratio as usize)) / 255
}

/// Split `data` into two slices at a position derived from `ratio`.
#[inline]
pub fn split_at_ratio(data: &[u8], ratio: u8) -> (&[u8], &[u8]) {
    data.split_at(split_point(data.len(), ratio))
}

#[cfg(feature = "aead")]
/// Assert that `encrypt_in_place → decrypt_in_place` recovers the original
/// plaintext for arbitrary inputs.
pub fn assert_aead_roundtrip<A: Aead>(
    cipher: &A,
    nonce: &A::Nonce,
    aad: &[u8],
    plaintext: &[u8],
) {
    let original = plaintext.to_vec();
    let mut buf = original.clone();

    let tag = cipher
        .encrypt_in_place(nonce, aad, &mut buf)
        .expect("roundtrip: encrypt must succeed");
    cipher
        .decrypt_in_place(nonce, aad, &mut buf, &tag)
        .expect("roundtrip: decrypt must succeed");

    assert_eq!(buf, original, "roundtrip: plaintext mismatch after decrypt");

    let ct_len = plaintext.len().strict_add(A::TAG_SIZE);
    let mut sealed = vec![0u8; ct_len];
    cipher
        .encrypt(nonce, aad, plaintext, &mut sealed)
        .expect("combined encrypt must succeed");
    let mut recovered = vec![0u8; plaintext.len()];
    cipher
        .decrypt(nonce, aad, &sealed, &mut recovered)
        .expect("combined decrypt must succeed");
    assert_eq!(
        recovered, original,
        "roundtrip: combined encrypt/decrypt mismatch"
    );
}

#[cfg(feature = "aead")]
/// Assert that flipping a single bit in ciphertext, tag, or AAD causes
/// `decrypt_in_place` to reject.
pub fn assert_aead_forgery<A: Aead>(
    cipher: &A,
    nonce: &A::Nonce,
    aad: &[u8],
    plaintext: &[u8],
    control: u8,
) {
    let mut ct = plaintext.to_vec();
    let tag = cipher
        .encrypt_in_place(nonce, aad, &mut ct)
        .expect("forgery: encrypt must succeed");

    let target = control % 3;
    let seed = control / 3;

    match target {
        0 if !ct.is_empty() => {
            let mut forged = ct.clone();
            let idx = seed as usize % forged.len();
            forged[idx] ^= 1u8 << (seed as u32 & 7);
            assert!(
                cipher
                    .decrypt_in_place(nonce, aad, &mut forged, &tag)
                    .is_err(),
                "forgery: accepted tampered ciphertext"
            );
        }
        2 if !aad.is_empty() => {
            let mut forged_aad = aad.to_vec();
            let idx = seed as usize % forged_aad.len();
            forged_aad[idx] ^= 1u8 << (seed as u32 & 7);
            let mut ct_copy = ct.clone();
            assert!(
                cipher
                    .decrypt_in_place(nonce, &forged_aad, &mut ct_copy, &tag)
                    .is_err(),
                "forgery: accepted tampered aad"
            );
        }
        _ => {
            let tag_ref = tag.as_ref();
            let mut tag_bytes = tag_ref.to_vec();
            let idx = seed as usize % tag_bytes.len();
            tag_bytes[idx] ^= 1u8 << (seed as u32 & 7);
            let forged_tag = A::tag_from_slice(&tag_bytes).unwrap();
            let mut ct_copy = ct.clone();
            assert!(
                cipher
                    .decrypt_in_place(nonce, aad, &mut ct_copy, &forged_tag)
                    .is_err(),
                "forgery: accepted tampered tag"
            );
        }
    }
}

/// Assert that splitting input at an arbitrary boundary and streaming via
/// `update` produces the same digest as one-shot `digest()`.
pub fn assert_digest_chunked<D: Digest>(data: &[u8], split_byte: u8) {
    let expected = D::digest(data);

    let (a, b) = split_at_ratio(data, split_byte);
    let mut h = D::new();
    h.update(a);
    h.update(b);
    h.update(&[]);
    let got = h.finalize();

    assert_eq!(expected, got, "chunk-split: digest mismatch");
}

/// Assert that `reset()` restores initial state: hashing the same data after
/// reset must produce the same result.
pub fn assert_digest_reset<D: Digest>(data: &[u8]) {
    let mut h = D::new();
    h.update(data);
    let first = h.finalize();

    h.reset();
    h.update(data);
    let second = h.finalize();

    assert_eq!(first, second, "reset: digest changed after reset + re-hash");
}

/// Assert that streaming MAC with an arbitrary split matches one-shot MAC.
pub fn assert_mac_streaming<M: Mac>(key: &[u8], data: &[u8], split_byte: u8) {
    let expected = M::mac(key, data);

    let (a, b) = split_at_ratio(data, split_byte);
    let mut m = M::new(key);
    m.update(a);
    m.update(b);
    m.update(&[]);
    let got = m.finalize();

    assert_eq!(expected, got, "mac: streaming mismatch");
}

/// Assert that squeezing a larger output is a prefix-extension of a smaller one.
#[macro_export]
macro_rules! assert_xof_prefix {
    ($make_reader:expr, $small_len:expr, $large_len:expr, $msg:expr) => {{
        let mut r1 = $make_reader;
        let mut small = vec![0u8; $small_len];
        rscrypto::Xof::squeeze(&mut r1, &mut small);

        let mut r2 = $make_reader;
        let mut large = vec![0u8; $large_len];
        rscrypto::Xof::squeeze(&mut r2, &mut large);

        assert_eq!(&small[..], &large[..$small_len], $msg);
    }};
}

/// Assert that `checksum(a || b) == combine(checksum(a), checksum(b), len(b))`
/// for an arbitrary split point.
pub fn assert_checksum_combine<C: ChecksumCombine>(data: &[u8], split_byte: u8) {
    let expected = C::checksum(data);

    let (a, b) = split_at_ratio(data, split_byte);
    let crc_a = C::checksum(a);
    let crc_b = C::checksum(b);
    let got = C::combine(crc_a, crc_b, b.len());

    assert_eq!(expected, got, "combine: checksum mismatch");
}

/// Assert that streaming checksum with an arbitrary split matches one-shot.
pub fn assert_checksum_chunked<C: Checksum>(data: &[u8], split_byte: u8) {
    let expected = C::checksum(data);

    let (a, b) = split_at_ratio(data, split_byte);
    let mut c = C::new();
    c.update(a);
    c.update(b);
    c.update(&[]);
    let got = c.finalize();

    assert_eq!(expected, got, "checksum: streaming mismatch");
}
