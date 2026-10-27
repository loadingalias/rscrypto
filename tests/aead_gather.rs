//! Gather encryption preserves one message across arbitrary slice boundaries.

#![cfg(any(feature = "aes-gcm", feature = "chacha20poly1305"))]

use aes_gcm::aead::{KeyInit, Payload, array::Array};
use proptest::prelude::*;
use rscrypto::{
  AeadGather,
  aead::{Nonce96, SealError, expert::AeadWithNonce},
};

fn check<C: AeadGather<Nonce = Nonce96>>(
  cipher: &C,
  nonce: [u8; 12],
  aad: &[u8],
  plaintext: &[u8],
  cuts: &[usize],
  expected: &[u8],
) {
  let mut offsets: Vec<_> = cuts
    .iter()
    .map(|cut| cut.rem_euclid(plaintext.len().strict_add(1)))
    .collect();
  offsets.extend([0, plaintext.len()]);
  offsets.sort_unstable();
  // Guards and independent allocations prevent a native gather kernel from
  // accidentally treating adjacent logical slices as one physical input range.
  let storage: Vec<_> = offsets
    .windows(2)
    .map(|ends| {
      let mut bytes = vec![0xa5];
      bytes.extend_from_slice(&plaintext[ends[0]..ends[1]]);
      bytes.push(0x5a);
      bytes
    })
    .collect();
  let chunks: Vec<_> = storage
    .iter()
    .map(|bytes| &bytes[1..bytes.len().strict_sub(1)])
    .collect();
  check_chunks(cipher, nonce, aad, plaintext, &chunks, expected);
}

fn check_chunks<C: AeadGather<Nonce = Nonce96>>(
  cipher: &C,
  nonce: [u8; 12],
  aad: &[u8],
  plaintext: &[u8],
  chunks: &[&[u8]],
  expected: &[u8],
) {
  let nonce = Nonce96::from_bytes(nonce);
  let (expected_text, expected_tag) = expected.split_at(plaintext.len());

  let mut inplace = plaintext.to_vec();
  let tag = cipher
    .encrypt_in_place(&nonce, aad, &mut inplace)
    .expect("valid in-place fixture");
  assert_eq!(inplace, expected_text);
  assert_eq!(tag.as_ref(), expected_tag);

  let mut initialized = vec![0xa5; plaintext.len().strict_add(2)];
  let end = plaintext.len().strict_add(1);
  let tag = cipher
    .encrypt_gather(&nonce, aad, chunks.iter().copied(), &mut initialized[1..end])
    .expect("valid gather message");
  assert_eq!(&initialized[1..end], expected_text);
  assert_eq!(tag.as_ref(), expected_tag);
  assert_eq!((initialized[0], initialized[end]), (0xa5, 0xa5));

  let mut output = Vec::with_capacity(plaintext.len().strict_add(2));
  output.extend_from_slice(&[0xa5, 0x5a]);
  let (tag, ciphertext) = cipher
    .encrypt_gather_uninit(
      &nonce,
      aad,
      chunks.iter().copied(),
      &mut output.spare_capacity_mut()[..plaintext.len()],
    )
    .expect("valid uninitialized gather output");
  assert_eq!(ciphertext, expected_text);
  assert_eq!(tag.as_ref(), expected_tag);
  assert_eq!(output, [0xa5, 0x5a]);

  #[cfg(feature = "alloc")]
  {
    let mut appended = vec![0xa5, 0x5a];
    let tag = cipher
      .encrypt_gather_append(&nonce, aad, chunks.iter().copied(), plaintext.len(), &mut appended)
      .expect("append gathered ciphertext");
    assert_eq!(&appended[..2], &[0xa5, 0x5a]);
    assert_eq!(&appended[2..], expected_text);
    assert_eq!(tag.as_ref(), expected_tag);
  }

  if plaintext.is_empty() {
    let tag = cipher
      .encrypt_gather(&nonce, aad, core::iter::empty(), &mut [])
      .expect("empty iterator is an empty message");
    assert_eq!(tag.as_ref(), expected_tag);
  }
}

fn oracle<O>(oracle: &O, nonce: &[u8; 12], aad: &[u8], plaintext: &[u8]) -> Vec<u8>
where
  O: aes_gcm::aead::Aead<NonceSize = aes_gcm::aead::consts::U12>,
{
  oracle
    .encrypt(&Array(*nonce), Payload { msg: plaintext, aad })
    .expect("independent valid fixture")
}

proptest! {
  #![proptest_config(ProptestConfig {
    cases: 64,
    rng_seed: proptest::test_runner::RngSeed::Fixed(0x54a9_3c2e_10b7_8f61),
    ..ProptestConfig::default()
  })]

  #[test]
  #[cfg(feature = "aes-gcm")]
  fn aes128_partitions_match_rustcrypto(
    key in any::<[u8; 16]>(), nonce in any::<[u8; 12]>(),
    aad in prop::collection::vec(any::<u8>(), 0..80),
    plaintext in prop::collection::vec(any::<u8>(), 0..2049),
    cuts in prop::collection::vec(0usize..4097, 0..65),
  ) {
    let cipher = rscrypto::Aes128Gcm::new(&rscrypto::Aes128GcmKey::from_bytes(key));
    let expected = oracle(&aes_gcm::Aes128Gcm::new(&Array(key)), &nonce, &aad, &plaintext);
    check(&cipher, nonce, &aad, &plaintext, &cuts, &expected);
  }

  #[test]
  #[cfg(feature = "aes-gcm")]
  fn aes256_partitions_match_rustcrypto(
    key in any::<[u8; 32]>(), nonce in any::<[u8; 12]>(),
    aad in prop::collection::vec(any::<u8>(), 0..80),
    plaintext in prop::collection::vec(any::<u8>(), 0..2049),
    cuts in prop::collection::vec(0usize..4097, 0..65),
  ) {
    let cipher = rscrypto::Aes256Gcm::new(&rscrypto::Aes256GcmKey::from_bytes(key));
    let expected = oracle(&aes_gcm::Aes256Gcm::new(&Array(key)), &nonce, &aad, &plaintext);
    check(&cipher, nonce, &aad, &plaintext, &cuts, &expected);
  }

  #[test]
  #[cfg(feature = "chacha20poly1305")]
  fn chacha_partitions_match_rustcrypto(
    key in any::<[u8; 32]>(), nonce in any::<[u8; 12]>(),
    aad in prop::collection::vec(any::<u8>(), 0..80),
    plaintext in prop::collection::vec(any::<u8>(), 0..2049),
    cuts in prop::collection::vec(0usize..4097, 0..65),
  ) {
    let cipher = rscrypto::ChaCha20Poly1305::new(&rscrypto::ChaCha20Poly1305Key::from_bytes(key));
    let expected = oracle(&chacha20poly1305::ChaCha20Poly1305::new(&Array(key)), &nonce, &aad, &plaintext);
    check(&cipher, nonce, &aad, &plaintext, &cuts, &expected);
  }
}

fn boundaries<C: AeadGather<Nonce = Nonce96>, O>(cipher: &C, oracle_cipher: &O)
where
  O: aes_gcm::aead::Aead + aes_gcm::aead::AeadCore<NonceSize = aes_gcm::aead::consts::U12>,
{
  for len in (0usize..=129).chain([255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 16_384]) {
    let plaintext: Vec<_> = (0..len).map(|i| i.to_le_bytes()[0]).collect();
    let cuts: Vec<_> = (0..=len).collect();
    let nonce = [0x3a; 12];
    let expected = oracle(oracle_cipher, &nonce, b"record header", &plaintext);
    check(cipher, nonce, b"record header", &plaintext, &cuts, &expected);
  }
  // Contiguous prefixes enter native whole-block kernels. Seams immediately
  // around their boundaries must preserve one counter stream and one GHASH or
  // Poly1305 message, including the TLS content-type byte after a full record.
  for len in [
    255usize, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2048, 4095, 4096, 4097, 16_384, 16_385, 32_768, 32_769, 65_537,
  ] {
    let plaintext: Vec<_> = (0..len).map(|i| i.to_le_bytes()[0]).collect();
    let nonce = [0x3a; 12];
    let expected = oracle(oracle_cipher, &nonce, b"record header", &plaintext);
    for cuts in [
      vec![],
      vec![1, 15, 16, 17, 63, 64, 65, 255, 256, 257],
      (1usize..64).map(|i| len.strict_mul(i).strict_div(64)).collect(),
      vec![len.strict_sub(1)],
    ] {
      check(cipher, nonce, b"record header", &plaintext, &cuts, &expected);
    }
  }
}

#[test]
fn byte_chunks_cross_cipher_and_batch_boundaries() {
  #[cfg(feature = "aes-gcm")]
  {
    boundaries(
      &rscrypto::Aes128Gcm::new(&rscrypto::Aes128GcmKey::from_bytes([0x42; 16])),
      &aes_gcm::Aes128Gcm::new(&Array([0x42; 16])),
    );
    boundaries(
      &rscrypto::Aes256Gcm::new(&rscrypto::Aes256GcmKey::from_bytes([0x42; 32])),
      &aes_gcm::Aes256Gcm::new(&Array([0x42; 32])),
    );
  }
  #[cfg(feature = "chacha20poly1305")]
  boundaries(
    &rscrypto::ChaCha20Poly1305::new(&rscrypto::ChaCha20Poly1305Key::from_bytes([0x42; 32])),
    &chacha20poly1305::ChaCha20Poly1305::new(&Array([0x42; 32])),
  );
}

fn descriptors<C: AeadGather<Nonce = Nonce96>, O>(cipher: &C, oracle_cipher: &O)
where
  O: aes_gcm::aead::Aead + aes_gcm::aead::AeadCore<NonceSize = aes_gcm::aead::consts::U12>,
{
  let storage: Vec<Vec<u8>> = (0u8..131)
    .map(|i| (0u16..256).map(|j| i ^ j.to_le_bytes()[0]).collect())
    .collect();
  let mut chunks: Vec<_> = storage.iter().rev().map(Vec::as_slice).collect();
  chunks.insert(1, &[]);
  chunks.push(&storage[130]);
  chunks.push(&[]);
  chunks.push(&storage[0][..1]);
  let plaintext = chunks.concat();
  let nonce = [0x3a; 12];
  let expected = oracle(oracle_cipher, &nonce, b"record header", &plaintext);
  check_chunks(cipher, nonce, b"record header", &plaintext, &chunks, &expected);

  // Input and output can share an allocation when their exclusive ranges do
  // not overlap; native pointer-table traversal must preserve the input range.
  let mut allocation = plaintext.clone();
  allocation.resize(plaintext.len().strict_mul(2), 0xa5);
  let (input, output) = allocation.split_at_mut(plaintext.len());
  let tag = cipher
    .encrypt_gather(&Nonce96::from_bytes(nonce), b"record header", [&*input], output)
    .expect("disjoint ranges in the same allocation");
  assert_eq!(input, plaintext);
  assert_eq!(output, &expected[..plaintext.len()]);
  assert_eq!(tag.as_ref(), &expected[plaintext.len()..]);
}

#[test]
fn disjoint_reordered_and_repeated_chunks_cross_native_descriptor_batches() {
  #[cfg(feature = "aes-gcm")]
  {
    descriptors(
      &rscrypto::Aes128Gcm::new(&rscrypto::Aes128GcmKey::from_bytes([0x42; 16])),
      &aes_gcm::Aes128Gcm::new(&Array([0x42; 16])),
    );
    descriptors(
      &rscrypto::Aes256Gcm::new(&rscrypto::Aes256GcmKey::from_bytes([0x42; 32])),
      &aes_gcm::Aes256Gcm::new(&Array([0x42; 32])),
    );
  }
  #[cfg(feature = "chacha20poly1305")]
  descriptors(
    &rscrypto::ChaCha20Poly1305::new(&rscrypto::ChaCha20Poly1305Key::from_bytes([0x42; 32])),
    &chacha20poly1305::ChaCha20Poly1305::new(&Array([0x42; 32])),
  );
}

#[expect(clippy::panic, reason = "panicking input iterators must exercise output cleanup")]
fn failure_cleanup<C: AeadGather<Nonce = Nonce96>>(cipher: &C) {
  let nonce = Nonce96::from_bytes([0x44; 12]);
  #[cfg(feature = "alloc")]
  {
    let mut out = Vec::new();
    let error = cipher.encrypt_gather_append(&nonce, b"", core::iter::empty(), usize::MAX, &mut out);
    assert_eq!(
      error.expect_err("unrepresentable allocation rejects before allocation"),
      SealError::buffer()
    );
    assert_eq!((out.len(), out.capacity()), (0, 0));
    #[cfg(target_pointer_width = "64")]
    {
      let too_large = usize::try_from(1u64 << 40).expect("length fits this target's usize");
      let error = cipher.encrypt_gather_append(&nonce, b"", core::iter::empty(), too_large, &mut out);
      assert_eq!(
        error.expect_err("algorithm length rejects before allocation"),
        SealError::too_large()
      );
      assert_eq!((out.len(), out.capacity()), (0, 0));
    }
    out.extend_from_slice(&[0xa5, 0x5a]);
    let error = cipher.encrypt_gather_append(&nonce, b"", core::iter::empty(), usize::MAX, &mut out);
    assert_eq!(
      error.expect_err("append length overflows before allocation"),
      SealError::buffer()
    );
    assert_eq!(out, [0xa5, 0x5a]);
  }
  for length in [3usize, 5] {
    let mut out = vec![0xa5; length.strict_add(2)];
    let end = length.strict_add(1);
    let error = cipher.encrypt_gather(&nonce, b"", [b"ab".as_slice(), b"cd".as_slice()], &mut out[1..end]);
    assert_eq!(
      error.expect_err("chunk sum differs from output length"),
      SealError::buffer()
    );
    assert_eq!(&out[1..end], vec![0; length]);
    assert_eq!((out[0], out[end]), (0xa5, 0xa5));
    #[cfg(feature = "alloc")]
    {
      let mut appended = vec![0xa5, 0x5a];
      let error =
        cipher.encrypt_gather_append(&nonce, b"", [b"ab".as_slice(), b"cd".as_slice()], length, &mut appended);
      assert_eq!(
        error.expect_err("declared length differs from slice sum"),
        SealError::buffer()
      );
      assert_eq!(appended, [0xa5, 0x5a]);
    }
  }
  let mut out = [0xa5; 4];
  let chunks =
    core::iter::once(b"ab".as_slice()).chain(core::iter::from_fn(|| -> Option<&[u8]> { panic!("iterator failure") }));
  let result = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| {
    let _result = cipher.encrypt_gather(&nonce, b"", chunks, &mut out);
  }));
  assert!(result.is_err());
  assert_eq!(out, [0; 4]);

  let first = [0x31; 1024];
  let mut out = [0xa5; 1025];
  let chunks = core::iter::once(first.as_slice()).chain(core::iter::from_fn(|| -> Option<&[u8]> {
    panic!("iterator failure after native prefix")
  }));
  let result = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| {
    let _result = cipher.encrypt_gather(&nonce, b"", chunks, &mut out);
  }));
  assert!(result.is_err());
  assert_eq!(out, [0; 1025]);

  let error = cipher.encrypt_gather(&nonce, b"", [first.as_slice()], &mut out);
  assert_eq!(
    error.expect_err("missing byte after native prefix"),
    SealError::buffer()
  );
  assert_eq!(out, [0; 1025]);
}

#[test]
fn mismatched_lengths_and_iterator_unwind_clear_output() {
  #[cfg(feature = "aes-gcm")]
  {
    failure_cleanup(&rscrypto::Aes128Gcm::new(&rscrypto::Aes128GcmKey::from_bytes(
      [0x42; 16],
    )));
    failure_cleanup(&rscrypto::Aes256Gcm::new(&rscrypto::Aes256GcmKey::from_bytes(
      [0x42; 32],
    )));
  }
  #[cfg(feature = "chacha20poly1305")]
  failure_cleanup(&rscrypto::ChaCha20Poly1305::new(
    &rscrypto::ChaCha20Poly1305Key::from_bytes([0x42; 32]),
  ));
}

#[cfg(feature = "aes-gcm")]
#[test]
fn gather_nonce_counters_consume_a_nonce_when_slice_lengths_are_wrong() {
  macro_rules! check_counter {
    ($cipher:ty, $key:ty) => {{
      let cipher = <$cipher>::new(&<$key>::from_bytes([0x11; <$cipher>::KEY_SIZE]));
      let mut counter = rscrypto::aead::NonceCounter::<$cipher>::new(*b"test");
      let mut out = [0xa5; 2];
      let error = counter.encrypt_gather(&cipher, b"aad", [b"x".as_slice()], &mut out);
      assert_eq!(
        error.expect_err("wrong slice sum"),
        rscrypto::aead::NonceCounterSealError::Seal(SealError::buffer())
      );
      assert_eq!(counter.next_counter(), 1);
      assert_eq!(out, [0; 2]);
      let (nonce, tag) = counter
        .encrypt_gather(&cipher, b"aad", [b"ok".as_slice()], &mut out)
        .expect("next nonce seals the next message");
      assert_eq!(nonce.as_bytes(), b"test\0\0\0\0\0\0\0\x01");
      assert_eq!(counter.next_counter(), 2);
      cipher
        .decrypt_in_place(&nonce, b"aad", &mut out, &tag)
        .expect("next message authenticates");
      assert_eq!(&out, b"ok");
    }};
  }
  check_counter!(rscrypto::Aes128Gcm, rscrypto::Aes128GcmKey);
  check_counter!(rscrypto::Aes256Gcm, rscrypto::Aes256GcmKey);
}
