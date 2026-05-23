#![cfg(feature = "rsa")]

use core::{
  alloc::{GlobalAlloc, Layout},
  sync::atomic::{AtomicUsize, Ordering},
};
use std::alloc::System;

use rscrypto::{RsaPkcs1v15Profile, RsaPssProfile, RsaPublicKey, RsaSignatureProfile, RsaX509PublicKey};

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

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
const X509_SHA256_WITH_RSA_ENCRYPTION: &[u8] = &[
  0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0b, 0x05, 0x00,
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
  let key = RsaPublicKey::from_spki_der(&pss_spki()).unwrap();
  let input = public_operation_input();
  let mut out = vec![0u8; key.modulus().len()];

  reset_allocations();
  let mut scratch = key.public_scratch();
  assert_eq!(allocation_count(), 2);

  reset_allocations();
  key
    .public_operation_with_scratch(&input, &mut out, &mut scratch)
    .unwrap();
  assert_eq!(allocation_count(), 0);

  let sig = pss_signature_sha256();
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
    .verify_jwt_alg_with_scratch("PS256", MESSAGE_PSS, &sig, &mut scratch)
    .unwrap();
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  key
    .verify_cose_algorithm_id_with_scratch(-37, MESSAGE_PSS, &sig, &mut scratch)
    .unwrap();
  assert_eq!(allocation_count(), 0);

  let x509_key = RsaX509PublicKey::from_spki_der(&pss_spki()).unwrap();
  let mut scratch = x509_key.public_key().public_scratch();
  reset_allocations();
  x509_key
    .verify_signature_from_x509_algorithm_der_with_scratch(X509_PSS_SHA256_ALGORITHM, MESSAGE_PSS, &sig, &mut scratch)
    .unwrap();
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  x509_key
    .verify_tls13_signature_scheme_with_scratch(0x0804, MESSAGE_PSS, &sig, &mut scratch)
    .unwrap();
  assert_eq!(allocation_count(), 0);

  let key = RsaPublicKey::from_spki_der(&pkcs1v15_spki()).unwrap();
  let sig = pkcs1v15_signature_sha256();
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
    .verify_jwt_alg_with_scratch("RS256", MESSAGE_PKCS1V15, &sig, &mut scratch)
    .unwrap();
  assert_eq!(allocation_count(), 0);

  reset_allocations();
  key
    .verify_cose_algorithm_id_with_scratch(-257, MESSAGE_PKCS1V15, &sig, &mut scratch)
    .unwrap();
  assert_eq!(allocation_count(), 0);

  let x509_key = RsaX509PublicKey::from_spki_der(&pkcs1v15_spki()).unwrap();
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
    .verify_tls_certificate_signature_scheme_with_scratch(0x0401, MESSAGE_PKCS1V15, &sig, &mut scratch)
    .unwrap();
  assert_eq!(allocation_count(), 0);
}
