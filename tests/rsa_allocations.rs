#![cfg(feature = "rsa")]

use core::{
  alloc::{GlobalAlloc, Layout},
  sync::atomic::{AtomicUsize, Ordering},
};
use std::alloc::System;

use rsa::{BigUint, RsaPrivateKey as RustCryptoRsaPrivateKey, pkcs1::EncodeRsaPrivateKey};
use rscrypto::{
  RsaEncryptionError, RsaOaepProfile, RsaPkcs1v15Profile, RsaPrivateKey, RsaPssProfile, RsaPublicKey,
  RsaPublicKeyPolicy, RsaSignatureProfile, RsaX509PublicKey,
};

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
const ID_RSASSA_PSS_OID: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0a];

struct CountingAlloc;

// SAFETY: CountingAlloc preserves the global allocator contract because:
// 1. Every allocation operation delegates to `System` with the original `Layout`.
// 2. Returned pointers and deallocations are exactly those produced/accepted by `System`.
// 3. The only added behavior is an atomic counter update independent of allocation memory.
unsafe impl GlobalAlloc for CountingAlloc {
  unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: Delegating allocation to `System` because:
    // 1. `layout` is forwarded unchanged from the caller.
    // 2. `System` is the platform allocator and defines the allocation contract.
    unsafe { System.alloc(layout) }
  }

  unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
    // SAFETY: Delegating deallocation to `System` because:
    // 1. `ptr` was returned by this allocator's delegated `System` allocation path.
    // 2. `layout` is forwarded unchanged from the caller.
    unsafe { System.dealloc(ptr, layout) };
  }

  unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: Delegating reallocation to `System` because:
    // 1. `ptr` and `layout` identify an allocation owned by this allocator.
    // 2. `new_size` is forwarded unchanged from the caller.
    unsafe { System.realloc(ptr, layout, new_size) }
  }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

const MESSAGE_PSS: &[u8] = b"rscrypto RSA-PSS verification fixture";
const MESSAGE_PKCS1V15: &[u8] = b"rscrypto RSA-PKCS1-v1_5 verification fixture";
const X509_PSS_SHA256_ALGORITHM: &[u8] = &[
  0x30, 0x41, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0a, 0x30, 0x34, 0xa0, 0x0f, 0x30, 0x0d,
  0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x05, 0x00, 0xa1, 0x1c, 0x30, 0x1a, 0x06, 0x09,
  0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x08, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03,
  0x04, 0x02, 0x01, 0x05, 0x00, 0xa2, 0x03, 0x02, 0x01, 0x20,
];
const X509_PSS_SHA256_OVERSIZED_SALT_ALGORITHM: &[u8] = &[
  0x30, 0x42, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0a, 0x30, 0x35, 0xa0, 0x0f, 0x30, 0x0d,
  0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x05, 0x00, 0xa1, 0x1c, 0x30, 0x1a, 0x06, 0x09,
  0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x08, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03,
  0x04, 0x02, 0x01, 0x05, 0x00, 0xa2, 0x04, 0x02, 0x02, 0x10, 0x00,
];
const X509_SHA256_WITH_RSA_ENCRYPTION: &[u8] = &[
  0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0b, 0x05, 0x00,
];
const X509_SHA1_WITH_RSA_ENCRYPTION: &[u8] = &[
  0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x05, 0x05, 0x00,
];
const X509_PSS_DEFAULT_SHA1_ALGORITHM: &[u8] = &[
  0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0a, 0x30, 0x00,
];

fn hex_to_vec(hex: &str) -> Vec<u8> {
  assert_eq!(hex.len() % 2, 0);
  let mut out = Vec::with_capacity(hex.len() / 2);
  for chunk in hex.as_bytes().chunks_exact(2) {
    out.push((hex_value(chunk[0]) << 4) | hex_value(chunk[1]));
  }
  out
}

fn hex_value(byte: u8) -> u8 {
  match byte {
    b'0'..=b'9' => byte - b'0',
    b'a'..=b'f' => byte - b'a' + 10,
    b'A'..=b'F' => byte - b'A' + 10,
    _ => panic!("invalid hex digit"),
  }
}

fn der_len(len: usize) -> Vec<u8> {
  if len < 128 {
    return vec![len as u8];
  }

  let bytes = len.to_be_bytes();
  let first_nonzero = bytes.iter().position(|&byte| byte != 0).unwrap();
  let len_bytes = &bytes[first_nonzero..];
  let mut out = Vec::with_capacity(1 + len_bytes.len());
  out.push(0x80 | len_bytes.len() as u8);
  out.extend_from_slice(len_bytes);
  out
}

fn tlv(tag: u8, value: &[u8]) -> Vec<u8> {
  let mut out = Vec::with_capacity(1 + der_len(value.len()).len() + value.len());
  out.push(tag);
  out.extend_from_slice(&der_len(value.len()));
  out.extend_from_slice(value);
  out
}

fn sequence(value: &[u8]) -> Vec<u8> {
  tlv(0x30, value)
}

fn fill_rsa_random_with(byte: u8) -> impl FnMut(&mut [u8]) -> Result<(), RsaEncryptionError> {
  move |out| {
    out.fill(byte);
    Ok(())
  }
}

fn fill_rsa_random_from(bytes: &[u8]) -> impl FnMut(&mut [u8]) -> Result<(), RsaEncryptionError> + '_ {
  let mut offset = 0usize;
  move |out| {
    let end = offset.checked_add(out.len()).ok_or(RsaEncryptionError::InvalidLength)?;
    let Some(random) = bytes.get(offset..end) else {
      return Err(RsaEncryptionError::InvalidLength);
    };
    out.copy_from_slice(random);
    offset = end;
    Ok(())
  }
}

fn oid(value: &[u8]) -> Vec<u8> {
  tlv(0x06, value)
}

fn bit_string(value: &[u8]) -> Vec<u8> {
  tlv(0x03, value)
}

fn integer_unsigned(value: &[u8]) -> Vec<u8> {
  let first_nonzero = value.iter().position(|&byte| byte != 0);
  let value = first_nonzero.map_or(&[0u8][..], |index| &value[index..]);
  let mut encoded = Vec::with_capacity(value.len() + usize::from(value[0] & 0x80 != 0));
  if value[0] & 0x80 != 0 {
    encoded.push(0);
  }
  encoded.extend_from_slice(value);
  tlv(0x02, &encoded)
}

fn algorithm_identifier(algorithm_oid: &[u8], params: Option<&[u8]>) -> Vec<u8> {
  let mut body = Vec::new();
  body.extend_from_slice(&oid(algorithm_oid));
  if let Some(params) = params {
    body.extend_from_slice(params);
  }
  sequence(&body)
}

fn exponent_bytes(value: u64) -> Vec<u8> {
  let bytes = value.to_be_bytes();
  let first_nonzero = bytes.iter().position(|&byte| byte != 0).unwrap_or(bytes.len() - 1);
  bytes[first_nonzero..].to_vec()
}

fn pkcs1_from_public_key(key: &RsaPublicKey) -> Vec<u8> {
  let mut body = Vec::new();
  body.extend_from_slice(&integer_unsigned(key.modulus()));
  body.extend_from_slice(&integer_unsigned(&exponent_bytes(key.public_exponent().as_u64())));
  sequence(&body)
}

fn spki_for_pkcs1_with_algorithm(pkcs1: &[u8], algorithm: &[u8]) -> Vec<u8> {
  let mut subject_public_key = Vec::with_capacity(1 + pkcs1.len());
  subject_public_key.push(0);
  subject_public_key.extend_from_slice(pkcs1);

  let mut spki = Vec::new();
  spki.extend_from_slice(algorithm);
  spki.extend_from_slice(&bit_string(&subject_public_key));
  sequence(&spki)
}

fn pss_algorithm_spki_from_rsa_encryption_spki(spki: &[u8]) -> Vec<u8> {
  let public_key = legacy_public_key_from_spki(spki);
  let pkcs1 = pkcs1_from_public_key(&public_key);
  spki_for_pkcs1_with_algorithm(&pkcs1, &algorithm_identifier(ID_RSASSA_PSS_OID, None))
}

fn legacy_public_key_from_spki(spki: &[u8]) -> RsaPublicKey {
  RsaPublicKey::from_spki_der_with_policy(spki, &RsaPublicKeyPolicy::legacy_verification()).unwrap()
}

fn legacy_x509_public_key_from_spki(spki: &[u8]) -> RsaX509PublicKey {
  RsaX509PublicKey::from_spki_der_with_policy(spki, &RsaPublicKeyPolicy::legacy_verification()).unwrap()
}

fn minimal_tbs_certificate(signature_algorithm_der: &[u8]) -> Vec<u8> {
  let mut tbs = Vec::new();
  tbs.extend_from_slice(&integer_unsigned(&[1]));
  tbs.extend_from_slice(signature_algorithm_der);
  sequence(&tbs)
}

fn x509_certificate(tbs_certificate_der: &[u8], signature_algorithm_der: &[u8], signature: &[u8]) -> Vec<u8> {
  let mut signature_value = Vec::with_capacity(signature.len() + 1);
  signature_value.push(0);
  signature_value.extend_from_slice(signature);

  let mut certificate = Vec::new();
  certificate.extend_from_slice(tbs_certificate_der);
  certificate.extend_from_slice(signature_algorithm_der);
  certificate.extend_from_slice(&bit_string(&signature_value));
  sequence(&certificate)
}

fn pss_spki() -> Vec<u8> {
  hex_to_vec(
    "\
30820122300d06092a864886f70d01010105000382010f003082010a0282010100ee0768fd95d4\
b01ba98b31bae03ce5a63f4dbffa7ea7b43d876f3e2c49847f695248cebb213b867a3c219276d\
fd7d1187a9981f8d42131d67464b60658fdcbdb3dc036d38c4024b7c1dc8ac096d19d6db048d\
9db7c72d56675e923fbe0169e30e5dcb75b4170def5dc655d4a05ab317e0f19c2a79b8882b7\
ef6fdbbc66de73485bc59fd6957cf76972d5869fcf7ab75e84b2e4a665a29e2e0cf6a135a93\
0e0b5366e23bcad7f5e708effa078cd1387259c0c7730ba55343c568fa9b0da9c13d0c8ca03\
81a6833554b387ba45e9088064d6ec627ebb5513a183f375e59130b5e6e77081dc1cf807f2bf\
ab77fb0a09194a482e05eb931ea7b37deeb11d6d7f18110203010001",
  )
}

fn pss_signature_sha256() -> Vec<u8> {
  hex_to_vec(
    "\
2641e0207f279b526767343e03007a293a0523db41828c0e335c10dea7ab7dec988ad09cd220d\
1754f5f89f7cd3b2a9ab1f315709b019989ee96e9060d3158d0240b3f8dbed179c55c1b0fa78\
f31249d706256748c325ded4835224e80b3daa066489d1cf28a4062fa4129b21723f6336f8c\
55d6785cd2d284437748b78b47e0162da7cddd61df0536b378a1cc5c327cba76db99c253795c\
19b49007146e44dbe47f3ee9c2da2248710cd264661815bd5508f604d6ee4a663a46c472f6d\
323739fe6b142ddc3b006a9c113d3b81da524e0fe358f9cb141f686dc459b66b1150e5418e8\
b6fcf2590d0706da27017429d91fe9f521f9fbb2ae2044f2eecfe87c7d",
  )
}

fn pkcs1v15_spki() -> Vec<u8> {
  hex_to_vec(
    "\
30820122300d06092a864886f70d01010105000382010f003082010a0282010100bf882d756861\
2271d36ac41584c3434145d13af573b889b353af0d1b257dcdefabb18e3adb717507fa981c70e\
870c73359fe878feaa1b7cec819accaffa6646232d08650b8e1aefd6626cb8ab032cde1bb2cf\
d378a9275d4cf828313fcc108184cf05727595701c44d7009590c747a82d24e7b651c7e7a96\
b5e6141b880e3be517fea22ea5d73415de297f4c6ea66019689eaf6fb4732288355974d84438\
003aa9fb72b18a10e909062816e79e8ee15e2d4066c5000d3251f251c44e0486f85644f00a9\
6ae40d5462e07e1bd02fe3eae2c5a793716b81ab690edb9cb6d8532baab08c080b8b08bdf50\
d1d3dc07b4d3f4afbb0f86ea48971a16d04132ec2432870203010001",
  )
}

fn pkcs1v15_signature_sha256() -> Vec<u8> {
  hex_to_vec(
    "\
94781246d705f79659d01ac8894b6f41076abe165e28711ec8fa41c1c8767b175a9c63e5118b\
d30de86da0d7b8934e963ef69c438ace976e4453dfce6b9b84a7d37a27ee61512656333dfda1\
ac40197fe4f9396bb016b25054f98f149d126c0248fc007cddc3d75d178eb34ecda0e0df822\
825ca133c062d3cdcb19e20a3e377541d8253af795a9b49a41ddb5795592502b9efbb153afc\
dd4fcc492a891d8536ef91cc228a3dbf66f0c70596f9cd101fe95d127550e7a4a9864430bd3\
4a88d8df93f4df7b54e8a4b8643891481e4bdcf87be3f98a1fdc475a819e3dc3a114aff86e\
48929a430fc39333f81064701be7d5501a3a7b4ec6c68f6feda6190d66b16",
  )
}

fn rustcrypto_fixture_private_key() -> RustCryptoRsaPrivateKey {
  RustCryptoRsaPrivateKey::from_components(
    BigUint::parse_bytes(
      b"00d397b84d98a4c26138ed1b695a8106ead91d553bf06041b62d3fdc50a041e222b8f4529689c1b82c5e71554f5dd69fa2f4b6158cf0dbeb57811a0fc327e1f28e74fe74d3bc166c1eabdc1b8b57b934ca8be5b00b4f29975bcc99acaf415b59bb28a6782bb41a2c3c2976b3c18dbadef62f00c6bb226640095096c0cc60d22fe7ef987d75c6a81b10d96bf292028af110dc7cc1bbc43d22adab379a0cd5d8078cc780ff5cd6209dea34c922cf784f7717e428d75b5aec8ff30e5f0141510766e2e0ab8d473c84e8710b2b98227c3db095337ad3452f19e2b9bfbccdd8148abf6776fa552775e6e75956e45229ae5a9c46949bab1e622f0e48f56524a84ed3483b",
      16,
    )
    .unwrap(),
    BigUint::parse_bytes(b"010001", 16).unwrap(),
    BigUint::parse_bytes(
      b"00c4e70c689162c94c660828191b52b4d8392115df486a9adbe831e458d73958320dc1b755456e93701e9702d76fb0b92f90e01d1fe248153281fe79aa9763a92fae69d8d7ecd144de29fa135bd14f9573e349e45031e3b76982f583003826c552e89a397c1a06bd2163488630d92e8c2bb643d7abef700da95d685c941489a46f54b5316f62b5d2c3a7f1bbd134cb37353a44683fdc9d95d36458de22f6c44057fe74a0a436c4308f73f4da42f35c47ac16a7138d483afc91e41dc3a1127382e0c0f5119b0221b4fc639d6b9c38177a6de9b526ebd88c38d7982c07f98a0efd877d508aae275b946915c02e2e1106d175d74ec6777f5e80d12c053d9c7be1e341",
      16,
    )
    .unwrap(),
    vec![
      BigUint::parse_bytes(
        b"00f827bbf3a41877c7cc59aebf42ed4b29c32defcb8ed96863d5b090a05a8930dd624a21c9dcf9838568fdfa0df65b8462a5f2ac913d6c56f975532bd8e78fb07bd405ca99a484bcf59f019bbddcb3933f2bce706300b4f7b110120c5df9018159067c35da3061a56c8635a52b54273b31271b4311f0795df6021e6355e1a42e61",
        16,
      )
      .unwrap(),
      BigUint::parse_bytes(
        b"00da4817ce0089dd36f2ade6a3ff410c73ec34bf1b4f6bda38431bfede11cef1f7f6efa70e5f8063a3b1f6e17296ffb15feefa0912a0325b8d1fd65a559e717b5b961ec345072e0ec5203d03441d29af4d64054a04507410cf1da78e7b6119d909ec66e6ad625bf995b279a4b3c5be7d895cd7c5b9c4c497fde730916fcdb4e41b",
        16,
      )
      .unwrap(),
    ],
  )
  .unwrap()
}

fn private_key() -> RsaPrivateKey {
  let der = rustcrypto_fixture_private_key().to_pkcs1_der().unwrap();
  RsaPrivateKey::from_pkcs1_der_with_policy(der.as_bytes(), &RsaPublicKeyPolicy::legacy_verification()).unwrap()
}

fn factor_two_and_inverse(modulus: &[u8]) -> (Vec<u8>, Vec<u8>) {
  let mut factor = vec![0u8; modulus.len()];
  if let Some(last) = factor.last_mut() {
    *last = 2;
  }

  let mut plus_one = modulus.to_vec();
  let mut carry = 1u16;
  for byte in plus_one.iter_mut().rev() {
    let sum = u16::from(*byte) + carry;
    *byte = sum as u8;
    carry = sum >> 8;
    if carry == 0 {
      break;
    }
  }
  if carry != 0 {
    plus_one.insert(0, carry as u8);
  }

  let mut quotient = Vec::with_capacity(plus_one.len());
  let mut remainder = 0u16;
  for byte in plus_one {
    let value = (remainder << 8) | u16::from(byte);
    quotient.push((value / 2) as u8);
    remainder = value % 2;
  }
  let first_nonzero = quotient
    .iter()
    .position(|&byte| byte != 0)
    .unwrap_or(quotient.len() - 1);
  let inverse = quotient[first_nonzero..].to_vec();

  let mut inverse_fixed = vec![0u8; modulus.len()];
  inverse_fixed[modulus.len() - inverse.len()..].copy_from_slice(&inverse);
  (factor, inverse_fixed)
}

fn public_operation_input() -> Vec<u8> {
  hex_to_vec(
    "\
3450869c4ccbee98815e55cb42f2dd85a3427d3f65e33d29352293e18cde9582a9fbc54b440984\
1ba8d931a9a9411192516a9fbd3a7b886e7f8b8f3f7bb5403309eee9d7234df0b5934e18a1dc\
9e3b568a3fab6947cefe50500abcbda19fd9ab7b7e90a95801e36a020ba79bdc94346198d98131\
6864a06a43448b62acb7a8472661323175f04c5e447d0017e4073efc55f59f79f34aaa3be8ae7\
0d26db78b25e9dfb23856d1b1e024aedfcfd649d209412c0c80832ca3466965eeff539afe791f\
451b554e212cff4d92466438062c5202169b0adf0c95b7d3d31414602cf9d185252b550cc2e8f\
5be08b7fc71f51210ff88363badadfaf5c2915c3a10b2389e",
  )
}

fn reset_allocations() {
  ALLOCATIONS.store(0, Ordering::SeqCst);
}

fn allocation_count() -> usize {
  ALLOCATIONS.load(Ordering::SeqCst)
}

#[test]
fn reused_scratch_rsa_operations_do_not_allocate() {
  let key = legacy_public_key_from_spki(&pss_spki());
  let input = public_operation_input();
  let mut out = vec![0u8; key.modulus().len()];

  let mut scratch = key.public_scratch();

  key
    .public_operation_with_scratch(&input, &mut out, &mut scratch)
    .unwrap();

  reset_allocations();
  key
    .public_operation_with_scratch(&input, &mut out, &mut scratch)
    .unwrap();
  assert_eq!(allocation_count(), 0);

  let sig = pss_signature_sha256();
  let pss_sha256 = RsaSignatureProfile::pss(RsaPssProfile::Sha256);
  reset_allocations();
  key
    .verify_pss_with_scratch(RsaPssProfile::Sha256, MESSAGE_PSS, &sig, &mut scratch)
    .unwrap();
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  key
    .verify_signature_with_scratch(
      RsaSignatureProfile::pss(RsaPssProfile::Sha256),
      MESSAGE_PSS,
      &sig,
      &mut scratch,
    )
    .unwrap();
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  key
    .verify_expected_jwt_alg_with_scratch("PS256", "PS256", pss_sha256, MESSAGE_PSS, &sig, &mut scratch)
    .unwrap();
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  key
    .verify_expected_cose_algorithm_id_with_scratch(-37, -37, pss_sha256, MESSAGE_PSS, &sig, &mut scratch)
    .unwrap();
  assert_eq!(allocation_count(), 0);

  let seed = [0x42; 32];
  reset_allocations();
  key
    .encrypt_oaep_with_random_fill_and_scratch(
      RsaOaepProfile::Sha256,
      b"allocation-oracle",
      b"scratch-backed OAEP encryption",
      &mut out,
      &mut scratch,
      fill_rsa_random_from(&seed),
    )
    .unwrap();
  assert_eq!(allocation_count(), 0);

  let x509_key = legacy_x509_public_key_from_spki(&pss_spki());
  let mut scratch = x509_key.public_key().public_scratch();
  reset_allocations();
  x509_key
    .verify_signature_from_x509_algorithm_der_with_scratch(X509_PSS_SHA256_ALGORITHM, MESSAGE_PSS, &sig, &mut scratch)
    .unwrap();
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  x509_key
    .verify_expected_tls13_signature_scheme_with_scratch(0x0804, 0x0804, pss_sha256, MESSAGE_PSS, &sig, &mut scratch)
    .unwrap();
  assert_eq!(allocation_count(), 0);

  let key = legacy_public_key_from_spki(&pkcs1v15_spki());
  let sig = pkcs1v15_signature_sha256();
  let pkcs1v15_sha256 = RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256);
  let mut scratch = key.public_scratch();
  reset_allocations();
  key
    .verify_pkcs1v15_with_scratch(RsaPkcs1v15Profile::Sha256, MESSAGE_PKCS1V15, &sig, &mut scratch)
    .unwrap();
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  key
    .verify_signature_with_scratch(
      RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256),
      MESSAGE_PKCS1V15,
      &sig,
      &mut scratch,
    )
    .unwrap();
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  key
    .verify_expected_jwt_alg_with_scratch("RS256", "RS256", pkcs1v15_sha256, MESSAGE_PKCS1V15, &sig, &mut scratch)
    .unwrap();
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  key
    .verify_expected_cose_algorithm_id_with_scratch(-257, -257, pkcs1v15_sha256, MESSAGE_PKCS1V15, &sig, &mut scratch)
    .unwrap();
  assert_eq!(allocation_count(), 0);

  let message = b"scratch-backed RSAES-PKCS1-v1_5 encryption";
  let mut ciphertext = vec![0u8; key.modulus().len()];
  reset_allocations();
  key
    .encrypt_pkcs1v15_with_random_fill_and_scratch(message, &mut ciphertext, &mut scratch, fill_rsa_random_with(0x5d))
    .unwrap();
  assert_eq!(allocation_count(), 0);

  let x509_key = legacy_x509_public_key_from_spki(&pkcs1v15_spki());
  let mut scratch = x509_key.public_key().public_scratch();
  reset_allocations();
  x509_key
    .verify_signature_from_x509_algorithm_der_with_scratch(
      X509_SHA256_WITH_RSA_ENCRYPTION,
      MESSAGE_PKCS1V15,
      &sig,
      &mut scratch,
    )
    .unwrap();
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  x509_key
    .verify_expected_tls_certificate_signature_scheme_with_scratch(
      0x0401,
      0x0401,
      pkcs1v15_sha256,
      MESSAGE_PKCS1V15,
      &sig,
      &mut scratch,
    )
    .unwrap();
  assert_eq!(allocation_count(), 0);

  assert_one_shot_protocol_rejects_fail_before_scratch_allocation();
  #[cfg(feature = "getrandom")]
  assert_private_protocol_signing_rejects_fail_before_entropy_allocation();
  assert_private_scratch_operations_do_not_allocate();
  #[cfg(feature = "getrandom")]
  assert_rng_private_scratch_operations_do_not_allocate();
}

#[cfg(feature = "getrandom")]
fn assert_private_protocol_signing_rejects_fail_before_entropy_allocation() {
  let key = private_key();
  let mut scratch = key.private_scratch();
  let mut signature = vec![0xa5; key.signature_len()];
  let message = b"private protocol signing allocation reject";

  reset_allocations();
  assert!(
    key
      .sign_tls13_signature_scheme(0x0401, message, &mut signature)
      .is_err()
  );
  assert_eq!(allocation_count(), 0);
  assert!(signature.iter().all(|&byte| byte == 0));

  signature.fill(0xa5);
  reset_allocations();
  assert!(
    key
      .sign_tls13_signature_scheme_with_scratch(0x0401, message, &mut signature, &mut scratch)
      .is_err()
  );
  assert_eq!(allocation_count(), 0);
  assert!(signature.iter().all(|&byte| byte == 0));

  signature.fill(0xa5);
  reset_allocations();
  assert!(
    key
      .sign_tls_certificate_signature_scheme(0x0201, message, &mut signature)
      .is_err()
  );
  assert_eq!(allocation_count(), 0);
  assert!(signature.iter().all(|&byte| byte == 0));

  signature.fill(0xa5);
  reset_allocations();
  assert!(
    key
      .sign_tls_certificate_signature_scheme_with_scratch(0x0201, message, &mut signature, &mut scratch)
      .is_err()
  );
  assert_eq!(allocation_count(), 0);
  assert!(signature.iter().all(|&byte| byte == 0));

  signature.fill(0xa5);
  reset_allocations();
  assert!(key.sign_jwt_alg("HS256", message, &mut signature).is_err());
  assert_eq!(allocation_count(), 0);
  assert!(signature.iter().all(|&byte| byte == 0));

  signature.fill(0xa5);
  reset_allocations();
  assert!(
    key
      .sign_jwt_alg_with_scratch("HS256", message, &mut signature, &mut scratch)
      .is_err()
  );
  assert_eq!(allocation_count(), 0);
  assert!(signature.iter().all(|&byte| byte == 0));

  signature.fill(0xa5);
  reset_allocations();
  assert!(key.sign_cose_algorithm_id(-7, message, &mut signature).is_err());
  assert_eq!(allocation_count(), 0);
  assert!(signature.iter().all(|&byte| byte == 0));

  signature.fill(0xa5);
  reset_allocations();
  assert!(
    key
      .sign_cose_algorithm_id_with_scratch(-7, message, &mut signature, &mut scratch)
      .is_err()
  );
  assert_eq!(allocation_count(), 0);
  assert!(signature.iter().all(|&byte| byte == 0));

  signature.fill(0xa5);
  reset_allocations();
  assert!(
    key
      .sign_x509_signature_algorithm_der(X509_SHA1_WITH_RSA_ENCRYPTION, message, &mut signature)
      .is_err()
  );
  assert_eq!(allocation_count(), 0);
  assert!(signature.iter().all(|&byte| byte == 0));

  signature.fill(0xa5);
  reset_allocations();
  assert!(
    key
      .sign_x509_signature_algorithm_der_with_scratch(
        X509_SHA1_WITH_RSA_ENCRYPTION,
        message,
        &mut signature,
        &mut scratch,
      )
      .is_err()
  );
  assert_eq!(allocation_count(), 0);
  assert!(signature.iter().all(|&byte| byte == 0));

  signature.fill(0xa5);
  reset_allocations();
  assert!(
    key
      .sign_signature(
        RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha256, usize::MAX),
        message,
        &mut signature
      )
      .is_err()
  );
  assert_eq!(allocation_count(), 0);
  assert!(signature.iter().all(|&byte| byte == 0));

  signature.fill(0xa5);
  reset_allocations();
  assert!(
    key
      .sign_signature_with_scratch(
        RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha256, usize::MAX),
        message,
        &mut signature,
        &mut scratch,
      )
      .is_err()
  );
  assert_eq!(allocation_count(), 0);
  assert!(signature.iter().all(|&byte| byte == 0));
}

fn assert_private_scratch_operations_do_not_allocate() {
  let key = private_key();
  let (blinding_factor, blinding_factor_inverse) = factor_two_and_inverse(key.public_key().modulus());
  let mut scratch = key.private_scratch();
  let mut signature = vec![0u8; key.signature_len()];
  let mut decrypted = vec![0u8; key.signature_len()];

  reset_allocations();
  key
    .sign_pkcs1v15_with_blinding_factor_and_scratch(
      RsaPkcs1v15Profile::Sha256,
      b"private scratch allocation PKCS1v15",
      &blinding_factor,
      &blinding_factor_inverse,
      &mut signature,
      &mut scratch,
    )
    .unwrap();
  assert_eq!(allocation_count(), 0);

  let salt = [0x7a; 32];
  reset_allocations();
  key
    .sign_pss_with_salt_and_blinding_factor_and_scratch(
      RsaPssProfile::Sha256,
      b"private scratch allocation PSS",
      &salt,
      &blinding_factor,
      &blinding_factor_inverse,
      &mut signature,
      &mut scratch,
    )
    .unwrap();
  assert_eq!(allocation_count(), 0);

  let label = b"private-scratch-allocation";
  let plaintext = b"private scratch allocation OAEP";
  let seed = [0x52; 32];
  let mut ciphertext = vec![0u8; key.signature_len()];
  key
    .public_key()
    .encrypt_oaep_with_random_fill(
      RsaOaepProfile::Sha256,
      label,
      plaintext,
      &mut ciphertext,
      fill_rsa_random_from(&seed),
    )
    .unwrap();

  reset_allocations();
  let len = key
    .decrypt_oaep_with_blinding_factor_and_scratch(
      RsaOaepProfile::Sha256,
      label,
      &ciphertext,
      &blinding_factor,
      &blinding_factor_inverse,
      &mut decrypted,
      &mut scratch,
    )
    .unwrap();
  assert_eq!(&decrypted[..len], plaintext);
  assert_eq!(allocation_count(), 0);

  let pkcs1v15_plaintext = b"private scratch allocation RSAES-PKCS1-v1_5";
  key
    .public_key()
    .encrypt_pkcs1v15_with_random_fill(pkcs1v15_plaintext, &mut ciphertext, fill_rsa_random_with(0x5d))
    .unwrap();

  reset_allocations();
  let len = key
    .decrypt_pkcs1v15_with_blinding_factor_and_scratch(
      &ciphertext,
      &blinding_factor,
      &blinding_factor_inverse,
      &mut decrypted,
      &mut scratch,
    )
    .unwrap();
  assert_eq!(&decrypted[..len], pkcs1v15_plaintext);
  assert_eq!(allocation_count(), 0);
}

#[cfg(feature = "getrandom")]
fn assert_rng_private_scratch_operations_do_not_allocate() {
  let key = private_key();
  let mut scratch = key.private_scratch();
  let mut signature = vec![0u8; key.signature_len()];
  let mut decrypted = vec![0u8; key.signature_len()];

  reset_allocations();
  key
    .sign_pkcs1v15_with_scratch(
      RsaPkcs1v15Profile::Sha256,
      b"rng private scratch allocation PKCS1v15",
      &mut signature,
      &mut scratch,
    )
    .unwrap();
  assert_eq!(allocation_count(), 0);
  key
    .public_key()
    .verify_pkcs1v15(
      RsaPkcs1v15Profile::Sha256,
      b"rng private scratch allocation PKCS1v15",
      &signature,
    )
    .unwrap();

  reset_allocations();
  key
    .sign_pss_with_scratch(
      RsaPssProfile::Sha256,
      b"rng private scratch allocation PSS",
      &mut signature,
      &mut scratch,
    )
    .unwrap();
  assert_eq!(allocation_count(), 0);
  key
    .public_key()
    .verify_pss(RsaPssProfile::Sha256, b"rng private scratch allocation PSS", &signature)
    .unwrap();

  let label = b"rng-private-scratch-allocation";
  let plaintext = b"rng private scratch allocation OAEP";
  let seed = [0x52; 32];
  let mut ciphertext = vec![0u8; key.signature_len()];
  key
    .public_key()
    .encrypt_oaep_with_random_fill(
      RsaOaepProfile::Sha256,
      label,
      plaintext,
      &mut ciphertext,
      fill_rsa_random_from(&seed),
    )
    .unwrap();

  reset_allocations();
  let len = key
    .decrypt_oaep_with_scratch(RsaOaepProfile::Sha256, label, &ciphertext, &mut decrypted, &mut scratch)
    .unwrap();
  assert_eq!(&decrypted[..len], plaintext);
  assert_eq!(allocation_count(), 0);

  let pkcs1v15_plaintext = b"rng private scratch allocation RSAES-PKCS1-v1_5";
  key
    .public_key()
    .encrypt_pkcs1v15_with_random_fill(pkcs1v15_plaintext, &mut ciphertext, fill_rsa_random_with(0x5d))
    .unwrap();

  reset_allocations();
  let len = key
    .decrypt_pkcs1v15_with_scratch(&ciphertext, &mut decrypted, &mut scratch)
    .unwrap();
  assert_eq!(&decrypted[..len], pkcs1v15_plaintext);
  assert_eq!(allocation_count(), 0);
}

fn assert_one_shot_protocol_rejects_fail_before_scratch_allocation() {
  let key = legacy_public_key_from_spki(&pss_spki());
  let sig = pss_signature_sha256();
  let pkcs1_sig = pkcs1v15_signature_sha256();
  let x509_key = legacy_x509_public_key_from_spki(&pss_spki());
  let pss_algorithm_x509_key =
    legacy_x509_public_key_from_spki(&pss_algorithm_spki_from_rsa_encryption_spki(&pss_spki()));

  reset_allocations();
  assert!(key.verify_jwt_alg("none", MESSAGE_PSS, &sig).is_err());
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  assert!(key.verify_jwt_alg("PS1", MESSAGE_PSS, &sig).is_err());
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  assert!(key.verify_cose_algorithm_id(1, MESSAGE_PSS, &sig).is_err());
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  assert!(key.verify_cose_algorithm_id(-65535, MESSAGE_PSS, &sig).is_err());
  assert_eq!(allocation_count(), 0);

  for impossible_salt_len in [4096, usize::MAX] {
    reset_allocations();
    assert!(
      key
        .verify_pss_with_salt_len(RsaPssProfile::Sha256, impossible_salt_len, MESSAGE_PSS, &sig)
        .is_err()
    );
    assert_eq!(allocation_count(), 0);

    reset_allocations();
    assert!(
      key
        .verify_signature(
          RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha256, impossible_salt_len),
          MESSAGE_PSS,
          &sig,
        )
        .is_err()
    );
    assert_eq!(allocation_count(), 0);

    let mut scratch = key.public_scratch();
    reset_allocations();
    assert!(
      key
        .verify_signature_with_scratch(
          RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha256, impossible_salt_len),
          MESSAGE_PSS,
          &sig,
          &mut scratch,
        )
        .is_err()
    );
    assert_eq!(allocation_count(), 0);
  }

  reset_allocations();
  assert!(
    x509_key
      .verify_signature_from_x509_algorithm_der(&[0x30, 0x00], MESSAGE_PSS, &sig)
      .is_err()
  );
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  assert!(
    x509_key
      .verify_signature_from_x509_algorithm_der(X509_SHA1_WITH_RSA_ENCRYPTION, MESSAGE_PSS, &sig)
      .is_err()
  );
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  assert!(
    x509_key
      .verify_signature_from_x509_algorithm_der(X509_PSS_DEFAULT_SHA1_ALGORITHM, MESSAGE_PSS, &sig)
      .is_err()
  );
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  assert!(
    x509_key
      .verify_signature_from_x509_algorithm_der(X509_PSS_SHA256_OVERSIZED_SALT_ALGORITHM, MESSAGE_PSS, &sig)
      .is_err()
  );
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  assert!(
    x509_key
      .verify_tls13_signature_scheme(0x0401, MESSAGE_PSS, &sig)
      .is_err()
  );
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  assert!(
    x509_key
      .verify_tls13_signature_scheme(0x0809, MESSAGE_PSS, &sig)
      .is_err()
  );
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  assert!(
    x509_key
      .verify_tls13_signature_scheme(0x0420, MESSAGE_PSS, &sig)
      .is_err()
  );
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  assert!(
    x509_key
      .verify_tls_certificate_signature_scheme(0x0201, MESSAGE_PSS, &sig)
      .is_err()
  );
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  assert!(
    pss_algorithm_x509_key
      .verify_tls13_signature_scheme(0x0804, MESSAGE_PSS, &sig)
      .is_err()
  );
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  assert!(
    pss_algorithm_x509_key
      .verify_tls_certificate_signature_scheme(0x0401, MESSAGE_PKCS1V15, &pkcs1_sig)
      .is_err()
  );
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  assert!(
    pss_algorithm_x509_key
      .verify_signature_from_x509_algorithm_der(X509_SHA256_WITH_RSA_ENCRYPTION, MESSAGE_PKCS1V15, &pkcs1_sig,)
      .is_err()
  );
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  assert!(x509_key.verify_x509_certificate_signature_der(&[0x30, 0x00]).is_err());
  assert_eq!(allocation_count(), 0);

  let sha1_certificate = x509_certificate(
    &minimal_tbs_certificate(X509_SHA1_WITH_RSA_ENCRYPTION),
    X509_SHA1_WITH_RSA_ENCRYPTION,
    &sig,
  );
  reset_allocations();
  assert!(
    x509_key
      .verify_x509_certificate_signature_der(&sha1_certificate)
      .is_err()
  );
  assert_eq!(allocation_count(), 0);

  let pss_default_sha1_certificate = x509_certificate(
    &minimal_tbs_certificate(X509_PSS_DEFAULT_SHA1_ALGORITHM),
    X509_PSS_DEFAULT_SHA1_ALGORITHM,
    &sig,
  );
  reset_allocations();
  assert!(
    x509_key
      .verify_x509_certificate_signature_der(&pss_default_sha1_certificate)
      .is_err()
  );
  assert_eq!(allocation_count(), 0);

  let pss_oversized_salt_certificate = x509_certificate(
    &minimal_tbs_certificate(X509_PSS_SHA256_OVERSIZED_SALT_ALGORITHM),
    X509_PSS_SHA256_OVERSIZED_SALT_ALGORITHM,
    &sig,
  );
  reset_allocations();
  assert!(
    x509_key
      .verify_x509_certificate_signature_der(&pss_oversized_salt_certificate)
      .is_err()
  );
  assert_eq!(allocation_count(), 0);
}
