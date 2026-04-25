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

/// Build a fixed-size salt-shaped buffer from a fuzzer-provided slice.
///
/// Password-hash fuzz targets (Argon2, scrypt, PHC) need a salt that
/// satisfies the algorithm's minimum length (RFC 9106 §3.1: `MIN_SALT_LEN
/// = 8`). When the fuzzer's slice is empty, the leading half of the
/// returned array is filled with `filler` so the salt is not all-zero;
/// otherwise the slice is repeated cyclically into the buffer. This shape
/// is uniform across every password-hash target.
#[inline]
#[must_use]
pub fn pad_salt_to<const N: usize>(material: &[u8], filler: u8) -> [u8; N] {
    let mut out = [0u8; N];
    if material.is_empty() {
        out[..N / 2].fill(filler);
    } else {
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = material[i % material.len()];
        }
    }
    out
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
/// Assert byte-identical round-trip against a RustCrypto-style AEAD oracle
/// that returns ciphertext and tag as a single combined buffer.
///
/// Captures the four-step pattern shared by every AEAD whose oracle
/// implements the `aead` crate's `Aead`/`AeadInPlace` traits:
///
/// 1. rscrypto encrypts; oracle decrypts the combined output.
/// 2. Oracle encrypts; rscrypto decrypts after splitting the `A::TAG_SIZE`
///    suffix from the body.
///
/// `oracle_encrypt` and `oracle_decrypt` adapt the per-oracle nonce and
/// `Payload` plumbing — they receive `(plaintext, aad)` / `(combined, aad)`
/// and return the corresponding `Vec<u8>`. Closures should panic with a
/// descriptive message on oracle errors; libFuzzer surfaces them as a crash.
///
/// AEAD constructions whose oracle returns the tag separately (e.g. `aegis`)
/// cannot use this helper and stay bespoke in their target.
pub fn assert_aead_against_oracle<A, EncFn, DecFn>(
    cipher: &A,
    nonce: &A::Nonce,
    aad: &[u8],
    plaintext: &[u8],
    oracle_encrypt: EncFn,
    oracle_decrypt: DecFn,
) where
    A: Aead,
    EncFn: FnOnce(&[u8], &[u8]) -> Vec<u8>,
    DecFn: FnOnce(&[u8], &[u8]) -> Vec<u8>,
{
    // ── ours encrypt → oracle decrypt ─────────────────────────────────────
    let mut ct = plaintext.to_vec();
    let tag = cipher
        .encrypt_in_place(nonce, aad, &mut ct)
        .expect("differential: rscrypto encrypt must succeed");
    let mut combined = ct.clone();
    combined.extend_from_slice(tag.as_ref());
    let pt = oracle_decrypt(&combined, aad);
    assert_eq!(pt, plaintext, "differential: oracle failed to decrypt our ciphertext");

    // ── oracle encrypt → ours decrypt ─────────────────────────────────────
    let oct = oracle_encrypt(plaintext, aad);
    let split = oct.len().strict_sub(A::TAG_SIZE);
    let (body, otag) = oct.split_at(split);
    let mut buf = body.to_vec();
    cipher
        .decrypt_in_place(
            nonce,
            aad,
            &mut buf,
            &A::tag_from_slice(otag).expect("oracle tag length mismatch"),
        )
        .expect("differential: rscrypto failed to decrypt oracle ciphertext");
    assert_eq!(buf, plaintext, "differential: decrypt mismatch on oracle ciphertext");
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

/// Assert that `reset()` restores the keyed initial state for a MAC.
///
/// Mirrors [`assert_digest_reset`] for `Mac` implementors so that every
/// keyed primitive is exercised on the same property surface.
pub fn assert_mac_reset<M: Mac>(key: &[u8], data: &[u8]) {
    let mut m = M::new(key);
    m.update(data);
    let first = m.finalize();
    m.reset();
    m.update(data);
    let second = m.finalize();
    assert_eq!(first, second, "mac: changed after reset");
}

/// Compare a freshly-computed MAC tag against an oracle-produced tag.
///
/// `oracle` receives `(key, data)` and returns the canonical reference
/// tag bytes. The helper asserts byte equality with the rscrypto tag —
/// both are compared as raw `[u8]` slices to absorb fixed-size vs. vec
/// representations.
pub fn assert_mac_against_oracle<M: Mac>(
    key: &[u8],
    data: &[u8],
    tag: &M::Tag,
    oracle: impl FnOnce(&[u8], &[u8]) -> Vec<u8>,
) {
    let oracle_tag = oracle(key, data);
    assert_eq!(
        tag.as_ref(),
        oracle_tag.as_slice(),
        "mac: oracle tag mismatch"
    );
}

/// Compare an HKDF expand result against an oracle expand result.
///
/// `ours_ok` / `oracle_ok` are `Result::is_ok()` for the expand calls;
/// `ours_okm` / `oracle_okm` are the (possibly-undefined-on-Err) output
/// buffers. The helper asserts both sides agree on success and, on
/// success, agree on bytes.
pub fn assert_hkdf_against_oracle(
    ours_ok: bool,
    ours_okm: &[u8],
    oracle_ok: bool,
    oracle_okm: &[u8],
) {
    assert_eq!(ours_ok, oracle_ok, "hkdf: oracle result mismatch");
    if ours_ok {
        assert_eq!(ours_okm, oracle_okm, "hkdf: oracle bytes mismatch");
    }
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

// ── Partial-IO scaffolding (feature: `traits_io`) ───────────────────────────
//
// Reusable adversarial `Read` / `Write` adapters for fuzzing rscrypto's
// streaming traits. They model real-world short reads/writes by capping the
// per-call byte count and tracking flush events. Used by `traits_io.rs` and
// available for any future streaming-IO target.

#[cfg(feature = "traits_io")]
pub use partial_io::{PartialReader, PartialWriter};

#[cfg(feature = "traits_io")]
mod partial_io {
    use std::io::{self, IoSlice, IoSliceMut, Read, Write};

    /// `Read` adapter that returns at most `max_per_call` bytes per call,
    /// regardless of the destination buffer size. Models adversarial short
    /// reads.
    #[derive(Clone, Debug)]
    pub struct PartialReader {
        pub data: Vec<u8>,
        pub pos: usize,
        pub max_per_call: usize,
    }

    impl PartialReader {
        #[inline]
        #[must_use]
        pub fn new(data: &[u8], max_per_call: usize) -> Self {
            Self {
                data: data.to_vec(),
                pos: 0,
                max_per_call,
            }
        }
    }

    impl Read for PartialReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let remaining = &self.data[self.pos..];
            let n = remaining.len().min(buf.len()).min(self.max_per_call);
            buf[..n].copy_from_slice(&remaining[..n]);
            self.pos = self.pos.strict_add(n);
            Ok(n)
        }

        fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
            let remaining = &self.data[self.pos..];
            let capacity = bufs
                .iter()
                .fold(0usize, |sum, buf| sum.strict_add(buf.len()));
            let n = remaining.len().min(capacity).min(self.max_per_call);

            let mut copied = 0usize;
            for buf in bufs {
                if copied == n {
                    break;
                }
                let take = buf.len().min(n.strict_sub(copied));
                if take == 0 {
                    continue;
                }
                buf[..take].copy_from_slice(&remaining[copied..copied.strict_add(take)]);
                copied = copied.strict_add(take);
            }

            self.pos = self.pos.strict_add(n);
            Ok(n)
        }
    }

    /// `Write` adapter that accepts at most `max_per_call` bytes per call
    /// and tracks the flush count. Models adversarial backpressure.
    #[derive(Clone, Debug, Default)]
    pub struct PartialWriter {
        pub inner: Vec<u8>,
        pub flushes: usize,
        pub max_per_call: usize,
    }

    impl PartialWriter {
        #[inline]
        #[must_use]
        pub fn new(max_per_call: usize) -> Self {
            Self {
                inner: Vec::new(),
                flushes: 0,
                max_per_call,
            }
        }
    }

    impl Write for PartialWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let n = buf.len().min(self.max_per_call);
            self.inner.extend_from_slice(&buf[..n]);
            Ok(n)
        }

        fn flush(&mut self) -> io::Result<()> {
            self.flushes = self.flushes.strict_add(1);
            Ok(())
        }

        fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
            let mut remaining = self.max_per_call;
            let mut written = 0usize;

            for buf in bufs {
                let take = remaining.min(buf.len());
                if remaining == 0 {
                    break;
                }
                if take == 0 {
                    continue;
                }
                self.inner.extend_from_slice(&buf[..take]);
                written = written.strict_add(take);
                remaining = remaining.strict_sub(take);
            }

            Ok(written)
        }
    }
}
