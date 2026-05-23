#![cfg(feature = "rsa")]

#[cfg(not(any(target_arch = "s390x", target_arch = "powerpc64")))]
use core::{
  ffi::c_void,
  mem::MaybeUninit,
  ptr::{self, NonNull},
};
use std::{
  fs,
  path::PathBuf,
  process::{self, Command},
  sync::OnceLock,
  time::{SystemTime, UNIX_EPOCH},
};

#[cfg(not(any(target_arch = "s390x", target_arch = "powerpc64")))]
use aws_lc_rs::signature as aws_signature;
#[cfg(not(any(target_arch = "s390x", target_arch = "powerpc64")))]
use aws_lc_sys as aws_lc;
use proptest::prelude::*;
use ring::signature as ring_signature;
use rsa::{
  RsaPublicKey as RustCryptoRsaPublicKey,
  pkcs1::DecodeRsaPublicKey,
  pkcs1v15::{Signature as RustCryptoPkcs1v15Signature, VerifyingKey as RustCryptoPkcs1v15VerifyingKey},
  pkcs8::DecodePublicKey,
  pss::{Signature as RustCryptoPssSignature, VerifyingKey as RustCryptoPssVerifyingKey},
  signature::Verifier as _,
};
#[cfg(feature = "diag")]
use rscrypto::auth::rsa::{
  diag_rsa_public_operation_bitserial, diag_rsa_public_operation_cios, diag_rsa_public_operation_product,
  diag_rsa_verify_pkcs1v15_encoded, diag_rsa_verify_pss_encoded, diag_rsa_verify_pss_encoded_with_scratch,
};
use rscrypto::{
  RsaKeyError, RsaPkcs1v15Profile, RsaProtocolAlgorithmError, RsaPssProfile, RsaPublicKey, RsaPublicKeyPolicy,
  RsaPublicOpError, RsaSignatureProfile, RsaTlsSignatureSchemes, RsaX509PublicKey, RsaX509PublicKeyAlgorithm,
  VerificationError,
};

const RSA_ENCRYPTION_OID: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01];
const SHA1_WITH_RSA_ENCRYPTION_OID: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x05];
const ID_RSASSA_PSS_OID: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0a];
const SHA256_WITH_RSA_ENCRYPTION_OID: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0b];
const SHA384_WITH_RSA_ENCRYPTION_OID: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0c];
const SHA512_WITH_RSA_ENCRYPTION_OID: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0d];
const ID_MGF1_OID: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x08];
const ID_SHA1_OID: &[u8] = &[0x2b, 0x0e, 0x03, 0x02, 0x1a];
const ID_SHA256_OID: &[u8] = &[0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01];
const ID_SHA384_OID: &[u8] = &[0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x02];
const ID_SHA512_OID: &[u8] = &[0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x03];
const CAVP_SIGVER_186_3: &str = include_str!("../testdata/rsa/nist_cavp/rsa_sigver_186_3_subset.json");
const RSA3072_SPKI: &[u8] = include_bytes!("../benches/rsa_fixtures/rsa3072_spki.der");
const RSA3072_PSS_SHA256: &[u8] = include_bytes!("../benches/rsa_fixtures/rsa3072_pss_sha256.sig");
const RSA3072_PKCS1V15_SHA256: &[u8] = include_bytes!("../benches/rsa_fixtures/rsa3072_pkcs1v15_sha256.sig");
const RSA4096_SPKI: &[u8] = include_bytes!("../benches/rsa_fixtures/rsa4096_spki.der");
const RSA4096_PSS_SHA256: &[u8] = include_bytes!("../benches/rsa_fixtures/rsa4096_pss_sha256.sig");
const RSA4096_PKCS1V15_SHA256: &[u8] = include_bytes!("../benches/rsa_fixtures/rsa4096_pkcs1v15_sha256.sig");
const RSA8192_SPKI: &[u8] = include_bytes!("../benches/rsa_fixtures/rsa8192_spki.der");
const RSA8192_PSS_SHA256: &[u8] = include_bytes!("../benches/rsa_fixtures/rsa8192_pss_sha256.sig");
const RSA8192_PKCS1V15_SHA256: &[u8] = include_bytes!("../benches/rsa_fixtures/rsa8192_pkcs1v15_sha256.sig");

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

fn oid(value: &[u8]) -> Vec<u8> {
  tlv(0x06, value)
}

fn null() -> Vec<u8> {
  tlv(0x05, &[])
}

fn bit_string(value: &[u8]) -> Vec<u8> {
  tlv(0x03, value)
}

fn context_constructed(index: u8, value: &[u8]) -> Vec<u8> {
  tlv(0xa0 | index, value)
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

fn hex_to_vec(hex: &str) -> Vec<u8> {
  assert_eq!(hex.len() % 2, 0);
  let mut out = Vec::with_capacity(hex.len() / 2);
  for chunk in hex.as_bytes().chunks_exact(2) {
    let hi = hex_value(chunk[0]);
    let lo = hex_value(chunk[1]);
    out.push((hi << 4) | lo);
  }
  out
}

fn cavp_hex_to_vec(hex: &str) -> Vec<u8> {
  let mut padded;
  let hex = if hex.len().is_multiple_of(2) {
    hex
  } else {
    padded = String::with_capacity(hex.len().strict_add(1));
    padded.push('0');
    padded.push_str(hex);
    &padded
  };
  hex_to_vec(hex)
}

fn hex_value(byte: u8) -> u8 {
  match byte {
    b'0'..=b'9' => byte - b'0',
    b'a'..=b'f' => byte - b'a' + 10,
    b'A'..=b'F' => byte - b'A' + 10,
    _ => panic!("invalid hex digit"),
  }
}

fn rsa2048_modulus() -> Vec<u8> {
  let mut modulus = vec![0u8; 256];
  modulus[0] = 0x80;
  modulus[255] = 0x01;
  modulus
}

fn pkcs1_with_parts(modulus_integer: Vec<u8>, exponent_integer: Vec<u8>) -> Vec<u8> {
  let mut body = Vec::new();
  body.extend_from_slice(&modulus_integer);
  body.extend_from_slice(&exponent_integer);
  sequence(&body)
}

fn valid_pkcs1() -> Vec<u8> {
  pkcs1_with_parts(
    integer_unsigned(&rsa2048_modulus()),
    integer_unsigned(&[0x01, 0x00, 0x01]),
  )
}

fn valid_pkcs1_with_modulus_and_exponent(modulus: &[u8], exponent: &[u8]) -> Vec<u8> {
  pkcs1_with_parts(integer_unsigned(modulus), integer_unsigned(exponent))
}

fn exponent_bytes(exponent: u64) -> Vec<u8> {
  let bytes = exponent.to_be_bytes();
  let first_nonzero = bytes.iter().position(|&byte| byte != 0).unwrap_or(bytes.len() - 1);
  bytes[first_nonzero..].to_vec()
}

fn x509_hash_algorithm(profile: RsaPssProfile) -> Vec<u8> {
  let oid = match profile {
    RsaPssProfile::Sha256 => ID_SHA256_OID,
    RsaPssProfile::Sha384 => ID_SHA384_OID,
    RsaPssProfile::Sha512 => ID_SHA512_OID,
  };
  algorithm_identifier(oid, Some(&null()))
}

fn x509_mgf1_algorithm(profile: RsaPssProfile) -> Vec<u8> {
  let hash = x509_hash_algorithm(profile);
  algorithm_identifier(ID_MGF1_OID, Some(&hash))
}

fn x509_pss_algorithm(profile: RsaPssProfile, salt_len: usize, trailer: Option<usize>) -> Vec<u8> {
  let mut params = Vec::new();
  params.extend_from_slice(&context_constructed(0, &x509_hash_algorithm(profile)));
  params.extend_from_slice(&context_constructed(1, &x509_mgf1_algorithm(profile)));
  params.extend_from_slice(&context_constructed(
    2,
    &integer_unsigned(&exponent_bytes(u64::try_from(salt_len).unwrap())),
  ));
  if let Some(trailer) = trailer {
    params.extend_from_slice(&context_constructed(
      3,
      &integer_unsigned(&exponent_bytes(u64::try_from(trailer).unwrap())),
    ));
  }
  algorithm_identifier(ID_RSASSA_PSS_OID, Some(&sequence(&params)))
}

fn x509_pss_algorithm_without_salt_len(profile: RsaPssProfile) -> Vec<u8> {
  let mut params = Vec::new();
  params.extend_from_slice(&context_constructed(0, &x509_hash_algorithm(profile)));
  params.extend_from_slice(&context_constructed(1, &x509_mgf1_algorithm(profile)));
  algorithm_identifier(ID_RSASSA_PSS_OID, Some(&sequence(&params)))
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

fn minimal_tbs_certificate(signature_algorithm_der: &[u8]) -> Vec<u8> {
  let mut tbs = Vec::new();
  tbs.extend_from_slice(&integer_unsigned(&[1]));
  tbs.extend_from_slice(signature_algorithm_der);
  sequence(&tbs)
}

fn pss_fixture_public_key() -> Vec<u8> {
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

fn pss_fixture_message() -> &'static [u8] {
  b"rscrypto RSA-PSS verification fixture"
}

fn pss_fixture_signature_sha256() -> Vec<u8> {
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

fn pkcs1v15_fixture_public_key() -> Vec<u8> {
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

fn pkcs1v15_fixture_message() -> &'static [u8] {
  b"rscrypto RSA-PKCS1-v1_5 verification fixture"
}

fn pkcs1v15_fixture_signature_sha256() -> Vec<u8> {
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

fn x509_certificate_fixture_public_key() -> Vec<u8> {
  hex_to_vec(
    "\
30820122300d06092a864886f70d01010105000382010f003082010a0282010100a246ccf6bd59720287837151de9fa5\
5d4a811e456643f7fd0ced5a9ffa8fe52a89d52a8f6bd96246c9f0d23cd4f215609bfd0fd09dfcf13305440cae6e1b9a\
3c48e8e360438ca9993c1cd8ec03363cc3d79edbc4df7764c7f8ddb75f1148037847b356d2697f7d0158072a2e4f38f9\
40c8db08b70305dedb6fe97aeb530dccc009274f7864442f6f02cf6191b5a32268234bcbd7827bf3e570206c0cddf147\
df5169ceda6883b2169768878fd5b107a092ab7482d8ba7f46364b566aaa72153068b6a0174f2f5e0e5f9bcd0213dd4e\
8689d56ffa0be918a16fffcbe4830157eb8535c1a2a50636f8fc8a57f9ae0488b91159456ca94d7e64a1286babad3e92\
f70203010001",
  )
}

fn x509_pkcs1v15_certificate_fixture() -> Vec<u8> {
  hex_to_vec(
    "\
3082031b30820203a00302010202144abda3eea77ef52d888f7ab507cd9016cacc900f300d06092a864886f70d01010b\
0500301d311b301906035504030c12727363727970746f2d7273612d706b637331301e170d3236303532323232333332\
305a170d3236303632313232333332305a301d311b301906035504030c12727363727970746f2d7273612d706b637331\
30820122300d06092a864886f70d01010105000382010f003082010a0282010100a246ccf6bd59720287837151de9fa5\
5d4a811e456643f7fd0ced5a9ffa8fe52a89d52a8f6bd96246c9f0d23cd4f215609bfd0fd09dfcf13305440cae6e1b9a\
3c48e8e360438ca9993c1cd8ec03363cc3d79edbc4df7764c7f8ddb75f1148037847b356d2697f7d0158072a2e4f38f9\
40c8db08b70305dedb6fe97aeb530dccc009274f7864442f6f02cf6191b5a32268234bcbd7827bf3e570206c0cddf147\
df5169ceda6883b2169768878fd5b107a092ab7482d8ba7f46364b566aaa72153068b6a0174f2f5e0e5f9bcd0213dd4e\
8689d56ffa0be918a16fffcbe4830157eb8535c1a2a50636f8fc8a57f9ae0488b91159456ca94d7e64a1286babad3e92\
f70203010001a3533051301d0603551d0e04160414fd0e576ce3f05b08884ad67ef3e8b4d39039c65d301f0603551d23\
041830168014fd0e576ce3f05b08884ad67ef3e8b4d39039c65d300f0603551d130101ff040530030101ff300d06092a\
864886f70d01010b050003820101008ed399c2f78e0325f9ec4ae7cd0b978b5cd03d30af8fb61a91925213b6388adb00\
0a59f657dc6a3e983706ff8053b0a4049d5f532c00640e2b67aac3ce30518ad7c4b762078c3816c6f325e9a841920b80\
6d0c5ac16450e8b385c6e50434bf1dc575816a263696b9de661a8bf2ae143853951745d25fa1bbc49d66270197b572aa\
052b0d23d35243cd9087fdc2d79bd2b1d27ad2c67fc7c1960b370b77edc038aee1ee653bec34782bbca87c5aefb3dc92\
5eba1d3c83019a37696d52ea1f14366a13ec6c3f74bda1941c745771bd33ea117f81e6c0968b9692d4dd349743acc149\
73eb5c2a1fcd85691f6dcbd6937b03fe525cbe51610a1f5be86a189de2d2d3",
  )
}

fn x509_pss_certificate_fixture() -> Vec<u8> {
  hex_to_vec(
    "\
3082037f30820233a0030201020214663f635a01b30adf27a6e6eab0482e8aaeda37b3304106092a864886f70d01010a\
3034a00f300d06096086480165030402010500a11c301a06092a864886f70d010108300d060960864801650304020105\
00a203020120301b3119301706035504030c10727363727970746f2d7273612d707373301e170d323630353232323233\
3332305a170d3236303632313232333332305a301b3119301706035504030c10727363727970746f2d7273612d707373\
30820122300d06092a864886f70d01010105000382010f003082010a0282010100a246ccf6bd59720287837151de9fa5\
5d4a811e456643f7fd0ced5a9ffa8fe52a89d52a8f6bd96246c9f0d23cd4f215609bfd0fd09dfcf13305440cae6e1b9a\
3c48e8e360438ca9993c1cd8ec03363cc3d79edbc4df7764c7f8ddb75f1148037847b356d2697f7d0158072a2e4f38f9\
40c8db08b70305dedb6fe97aeb530dccc009274f7864442f6f02cf6191b5a32268234bcbd7827bf3e570206c0cddf147\
df5169ceda6883b2169768878fd5b107a092ab7482d8ba7f46364b566aaa72153068b6a0174f2f5e0e5f9bcd0213dd4e\
8689d56ffa0be918a16fffcbe4830157eb8535c1a2a50636f8fc8a57f9ae0488b91159456ca94d7e64a1286babad3e92\
f70203010001a3533051301d0603551d0e04160414fd0e576ce3f05b08884ad67ef3e8b4d39039c65d301f0603551d23\
041830168014fd0e576ce3f05b08884ad67ef3e8b4d39039c65d300f0603551d130101ff040530030101ff304106092a\
864886f70d01010a3034a00f300d06096086480165030402010500a11c301a06092a864886f70d010108300d06096086\
480165030402010500a203020120038201010045256ddda4a30dbb8ffc987c52382f7f391853c4c9438a3ef185294090\
fd8b533e8b261ae21ffde98f7a44eb4fc84f7f5a857374c6d12e0ff1e0d54fd4d0f2ddd0686863e358ede2a629fdcc96\
aa2be4bdf07fa306c13709aef649d2cc6508311760418470aa70388c9d4205c15ee580349eac6624517d622961e0d588\
e069547141e2dd64f1868861bea549b8c179d7a40128a6ede9141ca170f35044fca9bca38439e38433ff1df2f5d51542\
03f6a158e6cd74b9f21ad1ff36366c65ec9ed67314b39c69e78251788aa4e524345b63063ef3f8eb61513f51bb8efb27\
eb10e16719b6b54d1768dc5278e6bcebc67d45226ab0164ede685b74a3d53eb14bcdfb",
  )
}

fn spki_for_pkcs1_with_algorithm(pkcs1: &[u8], algorithm: &[u8]) -> Vec<u8> {
  let mut subject_public_key = Vec::with_capacity(pkcs1.len() + 1);
  subject_public_key.push(0);
  subject_public_key.extend_from_slice(pkcs1);

  let mut spki = Vec::new();
  spki.extend_from_slice(algorithm);
  spki.extend_from_slice(&bit_string(&subject_public_key));
  sequence(&spki)
}

fn spki_for_pkcs1(pkcs1: &[u8]) -> Vec<u8> {
  spki_for_pkcs1_with_algorithm(pkcs1, &algorithm_identifier(RSA_ENCRYPTION_OID, Some(&null())))
}

fn assert_ring_pss_sha256(spki: &[u8], message: &[u8], signature: &[u8], expected: bool) {
  let key = ring_signature::UnparsedPublicKey::new(&ring_signature::RSA_PSS_2048_8192_SHA256, spki);
  assert_eq!(key.verify(message, signature).is_ok(), expected);
}

fn assert_ring_pkcs1v15_sha256(spki: &[u8], message: &[u8], signature: &[u8], expected: bool) {
  let key = ring_signature::UnparsedPublicKey::new(&ring_signature::RSA_PKCS1_2048_8192_SHA256, spki);
  assert_eq!(key.verify(message, signature).is_ok(), expected);
}

#[cfg(not(any(target_arch = "s390x", target_arch = "powerpc64")))]
fn assert_aws_lc_rs_pss_sha256(spki: &[u8], message: &[u8], signature: &[u8], expected: bool) {
  let key = aws_signature::UnparsedPublicKey::new(&aws_signature::RSA_PSS_2048_8192_SHA256, spki);
  assert_eq!(key.verify(message, signature).is_ok(), expected);
}

#[cfg(any(target_arch = "s390x", target_arch = "powerpc64"))]
fn assert_aws_lc_rs_pss_sha256(spki: &[u8], message: &[u8], signature: &[u8], expected: bool) {
  let _ = (spki, message, signature, expected);
}

#[cfg(not(any(target_arch = "s390x", target_arch = "powerpc64")))]
fn assert_aws_lc_rs_pkcs1v15_sha256(spki: &[u8], message: &[u8], signature: &[u8], expected: bool) {
  let key = aws_signature::UnparsedPublicKey::new(&aws_signature::RSA_PKCS1_2048_8192_SHA256, spki);
  assert_eq!(key.verify(message, signature).is_ok(), expected);
}

#[cfg(any(target_arch = "s390x", target_arch = "powerpc64"))]
fn assert_aws_lc_rs_pkcs1v15_sha256(spki: &[u8], message: &[u8], signature: &[u8], expected: bool) {
  let _ = (spki, message, signature, expected);
}

fn assert_ring_cavp(scheme: &str, sha: &str, pkcs1: &[u8], message: &[u8], signature: &[u8], expected: bool) {
  let actual = match (scheme, sha) {
    ("pss", "SHA256") => ring_signature::UnparsedPublicKey::new(&ring_signature::RSA_PSS_2048_8192_SHA256, pkcs1)
      .verify(message, signature)
      .is_ok(),
    ("pss", "SHA384") => ring_signature::UnparsedPublicKey::new(&ring_signature::RSA_PSS_2048_8192_SHA384, pkcs1)
      .verify(message, signature)
      .is_ok(),
    ("pss", "SHA512") => ring_signature::UnparsedPublicKey::new(&ring_signature::RSA_PSS_2048_8192_SHA512, pkcs1)
      .verify(message, signature)
      .is_ok(),
    ("pkcs1v15", "SHA256") => {
      ring_signature::UnparsedPublicKey::new(&ring_signature::RSA_PKCS1_2048_8192_SHA256, pkcs1)
        .verify(message, signature)
        .is_ok()
    }
    ("pkcs1v15", "SHA384") => {
      ring_signature::UnparsedPublicKey::new(&ring_signature::RSA_PKCS1_2048_8192_SHA384, pkcs1)
        .verify(message, signature)
        .is_ok()
    }
    ("pkcs1v15", "SHA512") => {
      ring_signature::UnparsedPublicKey::new(&ring_signature::RSA_PKCS1_2048_8192_SHA512, pkcs1)
        .verify(message, signature)
        .is_ok()
    }
    other => panic!("unsupported ring CAVP profile {other:?}"),
  };
  assert_eq!(actual, expected, "ring mismatch for {scheme}/{sha}");
}

#[cfg(not(any(target_arch = "s390x", target_arch = "powerpc64")))]
fn assert_aws_lc_rs_cavp(scheme: &str, sha: &str, pkcs1: &[u8], message: &[u8], signature: &[u8], expected: bool) {
  let actual = match (scheme, sha) {
    ("pss", "SHA256") => aws_signature::UnparsedPublicKey::new(&aws_signature::RSA_PSS_2048_8192_SHA256, pkcs1)
      .verify(message, signature)
      .is_ok(),
    ("pss", "SHA384") => aws_signature::UnparsedPublicKey::new(&aws_signature::RSA_PSS_2048_8192_SHA384, pkcs1)
      .verify(message, signature)
      .is_ok(),
    ("pss", "SHA512") => aws_signature::UnparsedPublicKey::new(&aws_signature::RSA_PSS_2048_8192_SHA512, pkcs1)
      .verify(message, signature)
      .is_ok(),
    ("pkcs1v15", "SHA256") => aws_signature::UnparsedPublicKey::new(&aws_signature::RSA_PKCS1_2048_8192_SHA256, pkcs1)
      .verify(message, signature)
      .is_ok(),
    ("pkcs1v15", "SHA384") => aws_signature::UnparsedPublicKey::new(&aws_signature::RSA_PKCS1_2048_8192_SHA384, pkcs1)
      .verify(message, signature)
      .is_ok(),
    ("pkcs1v15", "SHA512") => aws_signature::UnparsedPublicKey::new(&aws_signature::RSA_PKCS1_2048_8192_SHA512, pkcs1)
      .verify(message, signature)
      .is_ok(),
    other => panic!("unsupported aws-lc-rs CAVP profile {other:?}"),
  };
  assert_eq!(actual, expected, "aws-lc-rs mismatch for {scheme}/{sha}");
}

#[cfg(any(target_arch = "s390x", target_arch = "powerpc64"))]
fn assert_aws_lc_rs_cavp(scheme: &str, sha: &str, pkcs1: &[u8], message: &[u8], signature: &[u8], expected: bool) {
  let _ = (scheme, sha, pkcs1, message, signature, expected);
}

#[cfg(not(any(target_arch = "s390x", target_arch = "powerpc64")))]
struct AwsLcPublicKey(NonNull<aws_lc::EVP_PKEY>);

#[cfg(not(any(target_arch = "s390x", target_arch = "powerpc64")))]
impl Drop for AwsLcPublicKey {
  fn drop(&mut self) {
    // SAFETY: Releases the owned AWS-LC EVP_PKEY because:
    // 1. `self.0` was returned by `EVP_parse_public_key` and checked for null before wrapping.
    // 2. `AwsLcPublicKey` owns exactly one reference and does not implement `Clone`.
    // 3. AWS-LC permits freeing EVP_PKEY values through `EVP_PKEY_free`.
    unsafe { aws_lc::EVP_PKEY_free(self.0.as_ptr()) }
  }
}

#[cfg(not(any(target_arch = "s390x", target_arch = "powerpc64")))]
struct AwsLcMdCtx(NonNull<aws_lc::EVP_MD_CTX>);

#[cfg(not(any(target_arch = "s390x", target_arch = "powerpc64")))]
impl Drop for AwsLcMdCtx {
  fn drop(&mut self) {
    // SAFETY: Releases the owned AWS-LC EVP_MD_CTX because:
    // 1. `self.0` was returned by `EVP_MD_CTX_new` and checked for null before wrapping.
    // 2. `AwsLcMdCtx` owns exactly one context and does not implement `Clone`.
    // 3. AWS-LC permits freeing EVP_MD_CTX values through `EVP_MD_CTX_free`.
    unsafe { aws_lc::EVP_MD_CTX_free(self.0.as_ptr()) }
  }
}

#[cfg(not(any(target_arch = "s390x", target_arch = "powerpc64")))]
fn aws_lc_md(sha: &str) -> *const aws_lc::EVP_MD {
  match sha {
    "SHA256" => {
      // SAFETY: Obtains AWS-LC's static SHA-256 EVP_MD descriptor because:
      // 1. The function takes no caller-owned pointers.
      // 2. AWS-LC returns a process-static descriptor that must not be freed.
      // 3. The descriptor is only passed back into AWS-LC verification APIs.
      unsafe { aws_lc::EVP_sha256() }
    }
    "SHA384" => {
      // SAFETY: Obtains AWS-LC's static SHA-384 EVP_MD descriptor because:
      // 1. The function takes no caller-owned pointers.
      // 2. AWS-LC returns a process-static descriptor that must not be freed.
      // 3. The descriptor is only passed back into AWS-LC verification APIs.
      unsafe { aws_lc::EVP_sha384() }
    }
    "SHA512" => {
      // SAFETY: Obtains AWS-LC's static SHA-512 EVP_MD descriptor because:
      // 1. The function takes no caller-owned pointers.
      // 2. AWS-LC returns a process-static descriptor that must not be freed.
      // 3. The descriptor is only passed back into AWS-LC verification APIs.
      unsafe { aws_lc::EVP_sha512() }
    }
    other => panic!("unsupported AWS-LC sys hash `{other}`"),
  }
}

#[cfg(not(any(target_arch = "s390x", target_arch = "powerpc64")))]
fn aws_lc_parse_public_key(spki: &[u8]) -> Option<AwsLcPublicKey> {
  let mut cbs = MaybeUninit::<aws_lc::CBS>::uninit();
  // SAFETY: Initializes and parses an AWS-LC CBS over `spki` because:
  // 1. `spki.as_ptr()` is valid for `spki.len()` bytes for the duration of both calls.
  // 2. `CBS_init` fully initializes `cbs` before `EVP_parse_public_key` receives it.
  // 3. The returned EVP_PKEY pointer is checked for null before ownership is wrapped.
  let key = unsafe {
    aws_lc::CBS_init(cbs.as_mut_ptr(), spki.as_ptr(), spki.len());
    aws_lc::EVP_parse_public_key(cbs.as_mut_ptr())
  };
  NonNull::new(key).map(AwsLcPublicKey)
}

#[cfg(not(any(target_arch = "s390x", target_arch = "powerpc64")))]
fn aws_lc_new_md_ctx() -> Option<AwsLcMdCtx> {
  // SAFETY: Allocates a fresh AWS-LC EVP_MD_CTX because:
  // 1. The function takes no caller-owned pointers.
  // 2. The returned pointer is checked for null before wrapping.
  // 3. The wrapper frees the context exactly once with `EVP_MD_CTX_free`.
  let ctx = unsafe { aws_lc::EVP_MD_CTX_new() };
  NonNull::new(ctx).map(AwsLcMdCtx)
}

#[cfg(not(any(target_arch = "s390x", target_arch = "powerpc64")))]
fn aws_lc_sys_verify(
  scheme: &str,
  sha: &str,
  salt_len: Option<usize>,
  spki: &[u8],
  message: &[u8],
  signature: &[u8],
) -> Option<bool> {
  let key = aws_lc_parse_public_key(spki)?;
  let ctx = aws_lc_new_md_ctx()?;
  let md = aws_lc_md(sha);
  let mut pctx = ptr::null_mut();

  // SAFETY: Initializes an AWS-LC digest verification context because:
  // 1. `ctx` and `key` are live, non-null AWS-LC objects owned by RAII wrappers.
  // 2. `md` is a live process-static descriptor returned by AWS-LC for the selected SHA-2 digest.
  // 3. `pctx` points to stack storage where AWS-LC writes a context borrowed by `ctx`; it is not
  //    freed separately.
  let init_rc = unsafe { aws_lc::EVP_DigestVerifyInit(ctx.0.as_ptr(), &mut pctx, md, ptr::null_mut(), key.0.as_ptr()) };
  if init_rc != 1 || pctx.is_null() {
    return None;
  }

  let padding = match scheme {
    "pss" => aws_lc::RSA_PKCS1_PSS_PADDING,
    "pkcs1v15" => aws_lc::RSA_PKCS1_PADDING,
    other => panic!("unsupported AWS-LC sys RSA scheme `{other}`"),
  };

  // SAFETY: Configures RSA verification padding on AWS-LC's borrowed PKEY context because:
  // 1. `pctx` was returned non-null by `EVP_DigestVerifyInit` and remains owned by `ctx`.
  // 2. `padding` is one of AWS-LC's RSA padding constants selected from a closed match.
  // 3. No Rust references alias or inspect the opaque AWS-LC context while it is mutated.
  if unsafe { aws_lc::EVP_PKEY_CTX_set_rsa_padding(pctx, padding) } != 1 {
    return None;
  }

  if scheme == "pss" {
    let salt_len = i32::try_from(salt_len?).ok()?;
    // SAFETY: Configures PSS parameters on AWS-LC's borrowed PKEY context because:
    // 1. `pctx` was returned non-null by `EVP_DigestVerifyInit` and remains owned by `ctx`.
    // 2. `md` is the same live AWS-LC SHA-2 descriptor used for the message digest.
    // 3. `salt_len` is converted to AWS-LC's c_int range and comes from the CAVP profile under test.
    let pss_rc = unsafe {
      aws_lc::EVP_PKEY_CTX_set_rsa_mgf1_md(pctx, md) & aws_lc::EVP_PKEY_CTX_set_rsa_pss_saltlen(pctx, salt_len)
    };
    if pss_rc != 1 {
      return None;
    }
  }

  // SAFETY: Feeds message bytes to AWS-LC's digest verification context because:
  // 1. `ctx` is live and initialized for verification.
  // 2. `message.as_ptr()` is valid for `message.len()` bytes for the duration of the call.
  // 3. AWS-LC treats the pointer as read-only input and does not retain it after the call.
  if unsafe { aws_lc::EVP_DigestVerifyUpdate(ctx.0.as_ptr(), message.as_ptr().cast::<c_void>(), message.len()) } != 1 {
    return None;
  }

  // SAFETY: Finalizes AWS-LC signature verification because:
  // 1. `ctx` is live and has received all message bytes.
  // 2. `signature.as_ptr()` is valid for `signature.len()` bytes for the duration of the call.
  // 3. AWS-LC treats the signature pointer as read-only input and does not retain it after the call.
  let final_rc = unsafe { aws_lc::EVP_DigestVerifyFinal(ctx.0.as_ptr(), signature.as_ptr(), signature.len()) };
  match final_rc {
    0 => Some(false),
    1 => Some(true),
    _ => None,
  }
}

#[cfg(not(any(target_arch = "s390x", target_arch = "powerpc64")))]
fn assert_aws_lc_sys_cavp(
  scheme: &str,
  sha: &str,
  salt_len: Option<usize>,
  pkcs1: &[u8],
  message: &[u8],
  signature: &[u8],
  expected: bool,
) {
  let spki = spki_for_pkcs1(pkcs1);
  let actual = aws_lc_sys_verify(scheme, sha, salt_len, &spki, message, signature)
    .unwrap_or_else(|| panic!("AWS-LC sys setup failed for {scheme}/{sha} salt_len={salt_len:?}"));
  assert_eq!(actual, expected, "AWS-LC sys mismatch for {scheme}/{sha}");
}

#[cfg(any(target_arch = "s390x", target_arch = "powerpc64"))]
fn assert_aws_lc_sys_cavp(
  scheme: &str,
  sha: &str,
  salt_len: Option<usize>,
  pkcs1: &[u8],
  message: &[u8],
  signature: &[u8],
  expected: bool,
) {
  let _ = (scheme, sha, salt_len, pkcs1, message, signature, expected);
}

fn assert_rustcrypto_cavp(
  scheme: &str,
  sha: &str,
  salt_len: Option<usize>,
  pkcs1: &[u8],
  message: &[u8],
  signature: &[u8],
  expected: bool,
) {
  let key = RustCryptoRsaPublicKey::from_pkcs1_der(pkcs1).unwrap();
  let actual = match (scheme, sha) {
    ("pss", "SHA256") => {
      let key = RustCryptoPssVerifyingKey::<sha2_010::Sha256>::new_with_salt_len(key, salt_len.unwrap());
      let signature = RustCryptoPssSignature::try_from(signature).unwrap();
      key.verify(message, &signature).is_ok()
    }
    ("pss", "SHA384") => {
      let key = RustCryptoPssVerifyingKey::<sha2_010::Sha384>::new_with_salt_len(key, salt_len.unwrap());
      let signature = RustCryptoPssSignature::try_from(signature).unwrap();
      key.verify(message, &signature).is_ok()
    }
    ("pss", "SHA512") => {
      let key = RustCryptoPssVerifyingKey::<sha2_010::Sha512>::new_with_salt_len(key, salt_len.unwrap());
      let signature = RustCryptoPssSignature::try_from(signature).unwrap();
      key.verify(message, &signature).is_ok()
    }
    ("pkcs1v15", "SHA256") => {
      let key = RustCryptoPkcs1v15VerifyingKey::<sha2_010::Sha256>::new(key);
      let signature = RustCryptoPkcs1v15Signature::try_from(signature).unwrap();
      key.verify(message, &signature).is_ok()
    }
    ("pkcs1v15", "SHA384") => {
      let key = RustCryptoPkcs1v15VerifyingKey::<sha2_010::Sha384>::new(key);
      let signature = RustCryptoPkcs1v15Signature::try_from(signature).unwrap();
      key.verify(message, &signature).is_ok()
    }
    ("pkcs1v15", "SHA512") => {
      let key = RustCryptoPkcs1v15VerifyingKey::<sha2_010::Sha512>::new(key);
      let signature = RustCryptoPkcs1v15Signature::try_from(signature).unwrap();
      key.verify(message, &signature).is_ok()
    }
    other => panic!("unsupported RustCrypto CAVP profile {other:?}"),
  };
  assert_eq!(actual, expected, "RustCrypto mismatch for {scheme}/{sha}");
}

fn cavp_field<'a>(value: &'a serde_json::Value, name: &'static str) -> &'a str {
  value[name]
    .as_str()
    .unwrap_or_else(|| panic!("missing CAVP string field `{name}`"))
}

fn cavp_rscrypto_result(
  key: &RsaPublicKey,
  scheme: &str,
  sha: &str,
  salt_len: Option<usize>,
  message: &[u8],
  signature: &[u8],
) -> Result<(), rscrypto::VerificationError> {
  let profile = match (scheme, sha) {
    ("pss", "SHA256") => RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha256, salt_len.unwrap()),
    ("pss", "SHA384") => RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha384, salt_len.unwrap()),
    ("pss", "SHA512") => RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha512, salt_len.unwrap()),
    ("pkcs1v15", "SHA256") => RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256),
    ("pkcs1v15", "SHA384") => RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha384),
    ("pkcs1v15", "SHA512") => RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha512),
    other => panic!("unsupported rscrypto CAVP profile {other:?}"),
  };
  key.verify_signature(profile, message, signature)
}

fn assert_rustcrypto_pss_sha256(pkcs1: &[u8], message: &[u8], signature: &[u8], expected: bool) {
  let key = RustCryptoRsaPublicKey::from_pkcs1_der(pkcs1).unwrap();
  let key = RustCryptoPssVerifyingKey::<sha2_010::Sha256>::new(key);
  let signature = RustCryptoPssSignature::try_from(signature).unwrap();

  assert_eq!(key.verify(message, &signature).is_ok(), expected);
}

fn assert_rustcrypto_pkcs1v15_sha256(pkcs1: &[u8], message: &[u8], signature: &[u8], expected: bool) {
  let key = RustCryptoRsaPublicKey::from_pkcs1_der(pkcs1).unwrap();
  let key = RustCryptoPkcs1v15VerifyingKey::<sha2_010::Sha256>::new(key);
  let signature = RustCryptoPkcs1v15Signature::try_from(signature).unwrap();

  assert_eq!(key.verify(message, &signature).is_ok(), expected);
}

fn assert_rustcrypto_spki_pss_sha256(spki: &[u8], message: &[u8], signature: &[u8], expected: bool) {
  let key = RustCryptoRsaPublicKey::from_public_key_der(spki).unwrap();
  let key = RustCryptoPssVerifyingKey::<sha2_010::Sha256>::new(key);
  let signature = RustCryptoPssSignature::try_from(signature).unwrap();

  assert_eq!(key.verify(message, &signature).is_ok(), expected);
}

fn assert_rustcrypto_spki_pkcs1v15_sha256(spki: &[u8], message: &[u8], signature: &[u8], expected: bool) {
  let key = RustCryptoRsaPublicKey::from_public_key_der(spki).unwrap();
  let key = RustCryptoPkcs1v15VerifyingKey::<sha2_010::Sha256>::new(key);
  let signature = RustCryptoPkcs1v15Signature::try_from(signature).unwrap();

  assert_eq!(key.verify(message, &signature).is_ok(), expected);
}

fn assert_openssl_sha256(spki: &[u8], message: &[u8], signature: &[u8], sigopts: &[&str], expected: bool) {
  let Some(actual) = openssl_verify("-sha256", spki, message, signature, sigopts) else {
    eprintln!("skipping OpenSSL RSA differential check because `openssl` is not available");
    return;
  };

  assert_eq!(actual, expected);
}

#[allow(clippy::std_instead_of_core)]
fn openssl_verify(
  digest_arg: &'static str,
  spki: &[u8],
  message: &[u8],
  signature: &[u8],
  sigopts: &[&str],
) -> Option<bool> {
  let id = openssl_temp_id();
  let key_path = openssl_temp_path(&id, "spki.der");
  let msg_path = openssl_temp_path(&id, "msg.bin");
  let sig_path = openssl_temp_path(&id, "sig.bin");

  fs::write(&key_path, spki).unwrap();
  fs::write(&msg_path, message).unwrap();
  fs::write(&sig_path, signature).unwrap();

  let mut command = Command::new("openssl");
  command
    .args(["dgst", digest_arg, "-keyform", "DER", "-verify"])
    .arg(&key_path)
    .arg("-signature")
    .arg(&sig_path);

  for sigopt in sigopts {
    command.args(["-sigopt", sigopt]);
  }
  let output = command.arg(&msg_path).output();

  let _ = fs::remove_file(&key_path);
  let _ = fs::remove_file(&msg_path);
  let _ = fs::remove_file(&sig_path);

  let output = match output {
    Ok(output) => output,
    Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
    Err(error) => panic!("failed to run openssl: {error}"),
  };

  if output.status.success() {
    return Some(true);
  }

  let stdout = String::from_utf8_lossy(&output.stdout);
  let stderr = String::from_utf8_lossy(&output.stderr);
  if stdout.contains("Verification failure") || stderr.contains("bad signature") {
    return Some(false);
  }

  panic!(
    "openssl RSA verify failed unexpectedly: status={:?} stdout={stdout:?} stderr={stderr:?}",
    output.status.code()
  );
}

#[allow(clippy::std_instead_of_core)]
fn boringssl_verify(
  digest: &str,
  scheme: &str,
  salt_len: Option<usize>,
  spki: &[u8],
  message: &[u8],
  signature: &[u8],
) -> Option<bool> {
  let helper = std::env::var_os("BORINGSSL_RSA_VERIFY_HELPER")?;
  let id = openssl_temp_id();
  let key_path = openssl_temp_path(&id, "boringssl-spki.der");
  let msg_path = openssl_temp_path(&id, "boringssl-msg.bin");
  let sig_path = openssl_temp_path(&id, "boringssl-sig.bin");

  fs::write(&key_path, spki).unwrap();
  fs::write(&msg_path, message).unwrap();
  fs::write(&sig_path, signature).unwrap();

  let salt_len = salt_len.map_or_else(|| "-".to_owned(), |salt_len| salt_len.to_string());
  let output = Command::new(helper)
    .arg(digest)
    .arg(scheme)
    .arg(&salt_len)
    .arg(&key_path)
    .arg(&msg_path)
    .arg(&sig_path)
    .output();

  let _ = fs::remove_file(&key_path);
  let _ = fs::remove_file(&msg_path);
  let _ = fs::remove_file(&sig_path);

  let output = output.unwrap_or_else(|error| panic!("failed to run BoringSSL RSA helper: {error}"));
  let stdout = String::from_utf8_lossy(&output.stdout);
  let stderr = String::from_utf8_lossy(&output.stderr);

  if !output.status.success() {
    panic!(
      "BoringSSL RSA helper failed unexpectedly: status={:?} stdout={stdout:?} stderr={stderr:?}",
      output.status.code()
    );
  }

  match stdout.trim() {
    "valid" => Some(true),
    "invalid" => Some(false),
    other => panic!("BoringSSL RSA helper returned unexpected output {other:?}; stderr={stderr:?}"),
  }
}

fn digest_salt_len(sha: &str) -> usize {
  match sha {
    "SHA256" => 32,
    "SHA384" => 48,
    "SHA512" => 64,
    other => panic!("unsupported SHA-2 profile `{other}`"),
  }
}

fn openssl_pss_sigopts(sha: &str, salt_len: usize) -> [&'static str; 3] {
  let salt_len = match salt_len {
    0 => "rsa_pss_saltlen:0",
    1 => "rsa_pss_saltlen:1",
    24 => "rsa_pss_saltlen:24",
    32 => "rsa_pss_saltlen:32",
    48 => "rsa_pss_saltlen:48",
    64 => "rsa_pss_saltlen:64",
    other => panic!("unsupported CAVP PSS salt length `{other}`"),
  };
  let mgf1 = match sha {
    "SHA256" => "rsa_mgf1_md:sha256",
    "SHA384" => "rsa_mgf1_md:sha384",
    "SHA512" => "rsa_mgf1_md:sha512",
    other => panic!("unsupported OpenSSL PSS hash `{other}`"),
  };
  ["rsa_padding_mode:pss", salt_len, mgf1]
}

fn openssl_temp_id() -> String {
  let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
  format!("rscrypto-rsa-{}-{nanos}", process::id())
}

fn openssl_temp_path(id: &str, suffix: &str) -> PathBuf {
  std::env::temp_dir().join(format!("{id}-{suffix}"))
}

fn modulus_minus_one(key: &RsaPublicKey) -> Vec<u8> {
  let mut value = key.modulus().to_vec();
  for byte in value.iter_mut().rev() {
    if *byte != 0 {
      *byte = byte.strict_sub(1);
      break;
    }
    *byte = 0xff;
  }
  value
}

fn arbitrary_verify_key() -> &'static RsaPublicKey {
  static KEY: OnceLock<RsaPublicKey> = OnceLock::new();
  KEY.get_or_init(|| RsaPublicKey::from_spki_der(&pss_fixture_public_key()).unwrap())
}

fn fixed_width_signature_candidate(material: &[u8], len: usize) -> Vec<u8> {
  let mut out = vec![0u8; len];
  if material.is_empty() {
    return out;
  }
  for (index, byte) in out.iter_mut().enumerate() {
    *byte = material[index % material.len()];
  }
  out
}

fn assert_opaque_verification_failure(result: Result<(), rscrypto::VerificationError>) {
  assert_eq!(result, Err(rscrypto::VerificationError::new()));
}

fn advertised_schemes(schemes: RsaTlsSignatureSchemes) -> Vec<u16> {
  schemes.iter().collect()
}

#[test]
fn pkcs1_public_key_accepts_canonical_rsa2048_65537() {
  let key = RsaPublicKey::from_pkcs1_der(&valid_pkcs1()).unwrap();

  assert_eq!(key.modulus_bits(), 2048);
  assert_eq!(key.modulus().len(), 256);
  assert_eq!(key.public_exponent().as_u64(), 65_537);
}

#[test]
fn spki_public_key_accepts_rsa_encryption_with_null_parameters() {
  let pkcs1 = valid_pkcs1();
  let spki = spki_for_pkcs1(&pkcs1);

  let key = RsaPublicKey::from_spki_der(&spki).unwrap();

  assert_eq!(key.modulus(), &rsa2048_modulus());
  assert_eq!(key.public_exponent().as_u64(), 65_537);
}

#[test]
fn x509_spki_public_key_preserves_pss_key_algorithm_constraints() {
  let pkcs1 = valid_pkcs1();

  let rsa_spki = spki_for_pkcs1(&pkcs1);
  let rsa_key = RsaX509PublicKey::from_spki_der(&rsa_spki).unwrap();
  assert_eq!(rsa_key.key_algorithm(), RsaX509PublicKeyAlgorithm::RsaEncryption);
  assert_eq!(rsa_key.public_key().modulus(), &rsa2048_modulus());

  let pss_only_spki = spki_for_pkcs1_with_algorithm(&pkcs1, &algorithm_identifier(ID_RSASSA_PSS_OID, None));
  assert_eq!(
    RsaPublicKey::from_spki_der(&pss_only_spki),
    Err(RsaKeyError::UnsupportedAlgorithm)
  );
  let pss_only_key = RsaX509PublicKey::from_spki_der(&pss_only_spki).unwrap();
  assert_eq!(pss_only_key.key_algorithm(), RsaX509PublicKeyAlgorithm::RsaPss);
  assert!(
    pss_only_key
      .key_algorithm()
      .permits_signature_profile(RsaSignatureProfile::pss(RsaPssProfile::Sha256))
      .is_ok()
  );
  assert_eq!(
    pss_only_key
      .key_algorithm()
      .permits_signature_profile(RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256)),
    Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
  );

  let restricted_spki = spki_for_pkcs1_with_algorithm(&pkcs1, &x509_pss_algorithm(RsaPssProfile::Sha256, 32, None));
  let restricted_key = RsaX509PublicKey::from_spki_der(&restricted_spki).unwrap();
  assert_eq!(
    restricted_key.key_algorithm(),
    RsaX509PublicKeyAlgorithm::RsaPssRestricted {
      profile: RsaPssProfile::Sha256,
      minimum_salt_len: 32,
    }
  );
  assert!(
    restricted_key
      .key_algorithm()
      .permits_signature_profile(RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha256, 64))
      .is_ok()
  );
  assert_eq!(
    restricted_key
      .key_algorithm()
      .permits_signature_profile(RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha256, 20)),
    Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
  );
  assert_eq!(
    restricted_key
      .key_algorithm()
      .permits_signature_profile(RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha384, 48)),
    Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
  );
}

#[test]
fn x509_spki_constraints_are_enforced_during_signature_verification() {
  let unconstrained_key = RsaPublicKey::from_spki_der(&pss_fixture_public_key()).unwrap();
  let pkcs1 = valid_pkcs1_with_modulus_and_exponent(
    unconstrained_key.modulus(),
    &exponent_bytes(unconstrained_key.public_exponent().as_u64()),
  );
  let pss32_spki = spki_for_pkcs1_with_algorithm(&pkcs1, &x509_pss_algorithm(RsaPssProfile::Sha256, 32, None));
  let pss32_key = RsaX509PublicKey::from_spki_der(&pss32_spki).unwrap();
  let pss_algorithm = x509_pss_algorithm(RsaPssProfile::Sha256, 32, None);
  let pss_sig = pss_fixture_signature_sha256();

  assert!(
    pss32_key
      .verify_signature_from_x509_algorithm_der(&pss_algorithm, pss_fixture_message(), &pss_sig)
      .is_ok()
  );
  assert_eq!(
    pss32_key.verify_signature_from_x509_algorithm_der(
      &algorithm_identifier(SHA256_WITH_RSA_ENCRYPTION_OID, Some(&null())),
      pss_fixture_message(),
      &pss_sig,
    ),
    Err(VerificationError::new())
  );

  let pss64_spki = spki_for_pkcs1_with_algorithm(&pkcs1, &x509_pss_algorithm(RsaPssProfile::Sha256, 64, None));
  let pss64_key = RsaX509PublicKey::from_spki_der(&pss64_spki).unwrap();
  assert_eq!(
    pss64_key.verify_signature_from_x509_algorithm_der(&pss_algorithm, pss_fixture_message(), &pss_sig),
    Err(VerificationError::new())
  );
}

#[test]
fn x509_signature_verification_adapter_failures_are_opaque() {
  let unconstrained_key = RsaPublicKey::from_spki_der(&pss_fixture_public_key()).unwrap();
  let pkcs1 = valid_pkcs1_with_modulus_and_exponent(
    unconstrained_key.modulus(),
    &exponent_bytes(unconstrained_key.public_exponent().as_u64()),
  );
  let pss32_spki = spki_for_pkcs1_with_algorithm(&pkcs1, &x509_pss_algorithm(RsaPssProfile::Sha256, 32, None));
  let pss32_key = RsaX509PublicKey::from_spki_der(&pss32_spki).unwrap();
  let pss64_spki = spki_for_pkcs1_with_algorithm(&pkcs1, &x509_pss_algorithm(RsaPssProfile::Sha256, 64, None));
  let pss64_key = RsaX509PublicKey::from_spki_der(&pss64_spki).unwrap();
  let pss_algorithm = x509_pss_algorithm(RsaPssProfile::Sha256, 32, None);
  let pkcs1_algorithm = algorithm_identifier(SHA256_WITH_RSA_ENCRYPTION_OID, Some(&null()));
  let sha1_algorithm = algorithm_identifier(SHA1_WITH_RSA_ENCRYPTION_OID, Some(&null()));
  let pss_sig = pss_fixture_signature_sha256();
  let mut tampered_sig = pss_sig.clone();
  tampered_sig[0] ^= 0x01;

  for result in [
    pss32_key.verify_signature_from_x509_algorithm_der(&[0x30, 0x00], pss_fixture_message(), &pss_sig),
    pss32_key.verify_signature_from_x509_algorithm_der(&sha1_algorithm, pss_fixture_message(), &pss_sig),
    pss32_key.verify_signature_from_x509_algorithm_der(&pkcs1_algorithm, pss_fixture_message(), &pss_sig),
    pss64_key.verify_signature_from_x509_algorithm_der(&pss_algorithm, pss_fixture_message(), &pss_sig),
    pss32_key.verify_signature_from_x509_algorithm_der(&pss_algorithm, pss_fixture_message(), &tampered_sig),
  ] {
    assert_eq!(result, Err(VerificationError::new()));
  }
}

#[test]
fn x509_certificate_signature_verification_accepts_real_rsa_certificates() {
  let issuer = RsaX509PublicKey::from_spki_der(&x509_certificate_fixture_public_key()).unwrap();

  assert!(
    issuer
      .verify_x509_certificate_signature_der(&x509_pkcs1v15_certificate_fixture())
      .is_ok()
  );
  assert!(
    issuer
      .verify_x509_certificate_signature_der(&x509_pss_certificate_fixture())
      .is_ok()
  );

  let public_key = RsaPublicKey::from_spki_der(&x509_certificate_fixture_public_key()).unwrap();
  let pkcs1 = valid_pkcs1_with_modulus_and_exponent(
    public_key.modulus(),
    &exponent_bytes(public_key.public_exponent().as_u64()),
  );
  let pss_issuer = RsaX509PublicKey::from_spki_der(&spki_for_pkcs1_with_algorithm(
    &pkcs1,
    &algorithm_identifier(ID_RSASSA_PSS_OID, None),
  ))
  .unwrap();

  assert!(
    pss_issuer
      .verify_x509_certificate_signature_der(&x509_pss_certificate_fixture())
      .is_ok()
  );
  assert_eq!(
    pss_issuer.verify_x509_certificate_signature_der(&x509_pkcs1v15_certificate_fixture()),
    Err(VerificationError::new())
  );
}

#[test]
fn x509_certificate_signature_verification_rejects_malformed_and_confused_certificates() {
  let issuer = RsaX509PublicKey::from_spki_der(&x509_certificate_fixture_public_key()).unwrap();

  let mut tampered = x509_pkcs1v15_certificate_fixture();
  *tampered.last_mut().unwrap() ^= 0x01;
  assert_eq!(
    issuer.verify_x509_certificate_signature_der(&tampered),
    Err(VerificationError::new())
  );

  let pkcs1_algorithm = algorithm_identifier(SHA256_WITH_RSA_ENCRYPTION_OID, Some(&null()));
  let pss_algorithm = x509_pss_algorithm(RsaPssProfile::Sha256, 32, None);
  let mismatched_algorithm_certificate =
    x509_certificate(&minimal_tbs_certificate(&pkcs1_algorithm), &pss_algorithm, &[1, 2, 3]);
  assert_eq!(
    issuer.verify_x509_certificate_signature_der(&mismatched_algorithm_certificate),
    Err(VerificationError::new())
  );

  let sha1_algorithm = algorithm_identifier(SHA1_WITH_RSA_ENCRYPTION_OID, Some(&null()));
  let sha1_certificate = x509_certificate(&minimal_tbs_certificate(&sha1_algorithm), &sha1_algorithm, &[1, 2, 3]);
  assert_eq!(
    issuer.verify_x509_certificate_signature_der(&sha1_certificate),
    Err(VerificationError::new())
  );

  let mut bad_signature_value_certificate = Vec::new();
  bad_signature_value_certificate.extend_from_slice(&minimal_tbs_certificate(&pkcs1_algorithm));
  bad_signature_value_certificate.extend_from_slice(&pkcs1_algorithm);
  bad_signature_value_certificate.extend_from_slice(&bit_string(&[1, 1, 2, 3]));
  assert_eq!(
    issuer.verify_x509_certificate_signature_der(&sequence(&bad_signature_value_certificate)),
    Err(VerificationError::new())
  );
}

#[test]
fn tls_signature_scheme_mapping_enforces_rsae_vs_pss_key_algorithms() {
  let pkcs1 = valid_pkcs1();
  let rsae_key = RsaX509PublicKey::from_spki_der(&spki_for_pkcs1(&pkcs1)).unwrap();
  let pss_key = RsaX509PublicKey::from_spki_der(&spki_for_pkcs1_with_algorithm(
    &pkcs1,
    &algorithm_identifier(ID_RSASSA_PSS_OID, None),
  ))
  .unwrap();
  let restricted_key = RsaX509PublicKey::from_spki_der(&spki_for_pkcs1_with_algorithm(
    &pkcs1,
    &x509_pss_algorithm(RsaPssProfile::Sha256, 64, None),
  ))
  .unwrap();

  assert_eq!(
    rsae_key.signature_profile_from_tls13_signature_scheme(0x0804),
    Ok(RsaSignatureProfile::pss(RsaPssProfile::Sha256))
  );
  assert_eq!(
    pss_key.signature_profile_from_tls13_signature_scheme(0x0809),
    Ok(RsaSignatureProfile::pss(RsaPssProfile::Sha256))
  );
  assert_eq!(
    rsae_key.signature_profile_from_tls13_signature_scheme(0x0809),
    Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
  );
  assert_eq!(
    pss_key.signature_profile_from_tls13_signature_scheme(0x0804),
    Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
  );
  assert_eq!(
    restricted_key.signature_profile_from_tls13_signature_scheme(0x0809),
    Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
  );
  assert_eq!(
    rsae_key.signature_profile_from_tls13_signature_scheme(0x0401),
    Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
  );
  assert_eq!(
    rsae_key.signature_profile_from_tls13_signature_scheme(0x0201),
    Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
  );

  assert_eq!(
    rsae_key.signature_profile_from_tls_certificate_signature_scheme(0x0401),
    Ok(RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256))
  );
  assert_eq!(
    rsae_key.signature_profile_from_tls_certificate_signature_scheme(0x0804),
    Ok(RsaSignatureProfile::pss(RsaPssProfile::Sha256))
  );
  assert_eq!(
    pss_key.signature_profile_from_tls_certificate_signature_scheme(0x0809),
    Ok(RsaSignatureProfile::pss(RsaPssProfile::Sha256))
  );
  assert_eq!(
    pss_key.signature_profile_from_tls_certificate_signature_scheme(0x0401),
    Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
  );
}

#[test]
fn tls_signature_scheme_advertisement_matches_executable_key_constraints() {
  let pkcs1 = valid_pkcs1();
  let rsae = RsaX509PublicKeyAlgorithm::RsaEncryption;
  let pss = RsaX509PublicKeyAlgorithm::RsaPss;
  let pss_sha256_32 = RsaX509PublicKey::from_spki_der(&spki_for_pkcs1_with_algorithm(
    &pkcs1,
    &x509_pss_algorithm(RsaPssProfile::Sha256, 32, None),
  ))
  .unwrap()
  .key_algorithm();
  let pss_sha256_64 = RsaX509PublicKey::from_spki_der(&spki_for_pkcs1_with_algorithm(
    &pkcs1,
    &x509_pss_algorithm(RsaPssProfile::Sha256, 64, None),
  ))
  .unwrap()
  .key_algorithm();

  assert_eq!(
    advertised_schemes(rsae.advertised_tls13_signature_schemes()),
    vec![0x0804, 0x0805, 0x0806]
  );
  assert_eq!(
    advertised_schemes(rsae.advertised_tls_certificate_signature_schemes()),
    vec![0x0804, 0x0805, 0x0806, 0x0401, 0x0501, 0x0601]
  );
  assert_eq!(
    advertised_schemes(pss.advertised_tls13_signature_schemes()),
    vec![0x0809, 0x080a, 0x080b]
  );
  assert_eq!(
    advertised_schemes(pss.advertised_tls_certificate_signature_schemes()),
    vec![0x0809, 0x080a, 0x080b]
  );
  assert_eq!(
    advertised_schemes(pss_sha256_32.advertised_tls13_signature_schemes()),
    vec![0x0809]
  );
  assert!(pss_sha256_64.advertised_tls13_signature_schemes().is_empty());
  assert!(pss_sha256_64.advertised_tls_certificate_signature_schemes().is_empty());

  for algorithm in [rsae, pss, pss_sha256_32, pss_sha256_64] {
    for scheme in algorithm.advertised_tls13_signature_schemes().iter() {
      assert!(
        algorithm.signature_profile_from_tls13_signature_scheme(scheme).is_ok(),
        "advertised TLS 1.3 scheme {scheme:#06x} must be executable by {algorithm:?}"
      );
    }
    for scheme in algorithm.advertised_tls_certificate_signature_schemes().iter() {
      assert!(
        algorithm
          .signature_profile_from_tls_certificate_signature_scheme(scheme)
          .is_ok(),
        "advertised TLS certificate scheme {scheme:#06x} must be executable by {algorithm:?}"
      );
    }

    for unsupported_legacy_scheme in [0x0101, 0x0201, 0x0301] {
      assert!(
        !algorithm
          .advertised_tls13_signature_schemes()
          .contains(unsupported_legacy_scheme)
      );
      assert!(
        !algorithm
          .advertised_tls_certificate_signature_schemes()
          .contains(unsupported_legacy_scheme)
      );
    }
  }
}

#[test]
fn tls_signature_scheme_verification_rejects_key_algorithm_confusion() {
  let rsae_key = RsaX509PublicKey::from_spki_der(&pss_fixture_public_key()).unwrap();
  assert_eq!(rsae_key.key_algorithm(), RsaX509PublicKeyAlgorithm::RsaEncryption);
  let pkcs1 = valid_pkcs1_with_modulus_and_exponent(
    rsae_key.public_key().modulus(),
    &exponent_bytes(rsae_key.public_key().public_exponent().as_u64()),
  );
  let pss_spki = spki_for_pkcs1_with_algorithm(&pkcs1, &algorithm_identifier(ID_RSASSA_PSS_OID, None));
  let pss_key = RsaX509PublicKey::from_spki_der(&pss_spki).unwrap();
  let pss_signature = pss_fixture_signature_sha256();

  assert!(
    rsae_key
      .verify_tls13_signature_scheme(0x0804, pss_fixture_message(), &pss_signature)
      .is_ok()
  );
  assert!(
    pss_key
      .verify_tls13_signature_scheme(0x0809, pss_fixture_message(), &pss_signature)
      .is_ok()
  );
  assert_eq!(
    rsae_key.verify_tls13_signature_scheme(0x0809, pss_fixture_message(), &pss_signature),
    Err(VerificationError::new())
  );
  assert_eq!(
    pss_key.verify_tls13_signature_scheme(0x0804, pss_fixture_message(), &pss_signature),
    Err(VerificationError::new())
  );
  assert_eq!(
    rsae_key.verify_tls13_signature_scheme(0x0401, pss_fixture_message(), &pss_signature),
    Err(VerificationError::new())
  );

  let rsae_pkcs1_key = RsaX509PublicKey::from_spki_der(&pkcs1v15_fixture_public_key()).unwrap();
  let pkcs1_signature = pkcs1v15_fixture_signature_sha256();
  assert!(
    rsae_pkcs1_key
      .verify_tls_certificate_signature_scheme(0x0401, pkcs1v15_fixture_message(), &pkcs1_signature)
      .is_ok()
  );
  assert_eq!(
    pss_key.verify_tls_certificate_signature_scheme(0x0401, pkcs1v15_fixture_message(), &pkcs1_signature),
    Err(VerificationError::new())
  );
}

#[test]
fn protocol_verification_helpers_collapse_adapter_failures_to_opaque_error() {
  let rsae_pss_key = RsaX509PublicKey::from_spki_der(&pss_fixture_public_key()).unwrap();
  let pss_signature = pss_fixture_signature_sha256();
  let mut tampered_pss_signature = pss_signature.clone();
  tampered_pss_signature[0] ^= 0x01;

  let public_key = rsae_pss_key.public_key();
  let short_pss_signature = &pss_signature[..pss_signature.len().strict_sub(1)];
  let out_of_range_pss_signature = public_key.modulus();
  let mut public_scratch = public_key.public_scratch();
  assert_opaque_verification_failure(public_key.verify_jwt_alg("none", pss_fixture_message(), &pss_signature));
  assert_opaque_verification_failure(public_key.verify_jwt_alg("RS256", pss_fixture_message(), &pss_signature));
  assert_opaque_verification_failure(public_key.verify_jwt_alg("PS256", pss_fixture_message(), short_pss_signature));
  assert_opaque_verification_failure(public_key.verify_jwt_alg(
    "PS256",
    pss_fixture_message(),
    out_of_range_pss_signature,
  ));
  assert_opaque_verification_failure(public_key.verify_jwt_alg_with_scratch(
    "PS256",
    pss_fixture_message(),
    &tampered_pss_signature,
    &mut public_scratch,
  ));
  assert_opaque_verification_failure(public_key.verify_cose_algorithm_id(1, pss_fixture_message(), &pss_signature));
  assert_opaque_verification_failure(public_key.verify_cose_algorithm_id(-257, pss_fixture_message(), &pss_signature));
  assert_opaque_verification_failure(public_key.verify_cose_algorithm_id(
    -37,
    pss_fixture_message(),
    short_pss_signature,
  ));
  assert_opaque_verification_failure(public_key.verify_cose_algorithm_id(
    -37,
    pss_fixture_message(),
    out_of_range_pss_signature,
  ));
  assert_opaque_verification_failure(public_key.verify_cose_algorithm_id_with_scratch(
    -37,
    pss_fixture_message(),
    &tampered_pss_signature,
    &mut public_scratch,
  ));

  let pss_algorithm = x509_pss_algorithm(RsaPssProfile::Sha256, 32, None);
  let pkcs1_algorithm = algorithm_identifier(SHA256_WITH_RSA_ENCRYPTION_OID, Some(&null()));
  let sha1_algorithm = algorithm_identifier(SHA1_WITH_RSA_ENCRYPTION_OID, Some(&null()));
  let mut x509_scratch = public_key.public_scratch();
  assert_opaque_verification_failure(rsae_pss_key.verify_signature_from_x509_algorithm_der(
    &[0x30, 0x00],
    pss_fixture_message(),
    &pss_signature,
  ));
  assert_opaque_verification_failure(rsae_pss_key.verify_signature_from_x509_algorithm_der(
    &sha1_algorithm,
    pss_fixture_message(),
    &pss_signature,
  ));
  assert_opaque_verification_failure(rsae_pss_key.verify_signature_from_x509_algorithm_der(
    &pkcs1_algorithm,
    pss_fixture_message(),
    &pss_signature,
  ));
  assert_opaque_verification_failure(rsae_pss_key.verify_signature_from_x509_algorithm_der(
    &pss_algorithm,
    pss_fixture_message(),
    short_pss_signature,
  ));
  assert_opaque_verification_failure(rsae_pss_key.verify_signature_from_x509_algorithm_der(
    &pss_algorithm,
    pss_fixture_message(),
    out_of_range_pss_signature,
  ));
  assert_opaque_verification_failure(rsae_pss_key.verify_signature_from_x509_algorithm_der_with_scratch(
    &pss_algorithm,
    pss_fixture_message(),
    &tampered_pss_signature,
    &mut x509_scratch,
  ));

  let pkcs1 = valid_pkcs1_with_modulus_and_exponent(
    public_key.modulus(),
    &exponent_bytes(public_key.public_exponent().as_u64()),
  );
  let pss_spki = spki_for_pkcs1_with_algorithm(&pkcs1, &algorithm_identifier(ID_RSASSA_PSS_OID, None));
  let pss_key = RsaX509PublicKey::from_spki_der(&pss_spki).unwrap();
  let mut tls_scratch = public_key.public_scratch();
  assert_opaque_verification_failure(rsae_pss_key.verify_tls13_signature_scheme(
    0x0809,
    pss_fixture_message(),
    &pss_signature,
  ));
  assert_opaque_verification_failure(pss_key.verify_tls13_signature_scheme(
    0x0804,
    pss_fixture_message(),
    &pss_signature,
  ));
  assert_opaque_verification_failure(rsae_pss_key.verify_tls13_signature_scheme(
    0x0401,
    pss_fixture_message(),
    &pss_signature,
  ));
  assert_opaque_verification_failure(rsae_pss_key.verify_tls13_signature_scheme(
    0x0804,
    pss_fixture_message(),
    short_pss_signature,
  ));
  assert_opaque_verification_failure(rsae_pss_key.verify_tls13_signature_scheme(
    0x0804,
    pss_fixture_message(),
    out_of_range_pss_signature,
  ));
  assert_opaque_verification_failure(rsae_pss_key.verify_tls13_signature_scheme_with_scratch(
    0x0804,
    pss_fixture_message(),
    &tampered_pss_signature,
    &mut tls_scratch,
  ));

  let pkcs1_key = RsaX509PublicKey::from_spki_der(&pkcs1v15_fixture_public_key()).unwrap();
  let pkcs1_signature = pkcs1v15_fixture_signature_sha256();
  let mut tampered_pkcs1_signature = pkcs1_signature.clone();
  tampered_pkcs1_signature[0] ^= 0x01;
  let short_pkcs1_signature = &pkcs1_signature[..pkcs1_signature.len().strict_sub(1)];
  let out_of_range_pkcs1_signature = pkcs1_key.public_key().modulus();
  let mut tls_certificate_scratch = pkcs1_key.public_key().public_scratch();
  assert_opaque_verification_failure(pkcs1_key.verify_tls_certificate_signature_scheme(
    0x0201,
    pkcs1v15_fixture_message(),
    &pkcs1_signature,
  ));
  assert_opaque_verification_failure(pkcs1_key.verify_tls_certificate_signature_scheme(
    0x0804,
    pkcs1v15_fixture_message(),
    &pkcs1_signature,
  ));
  assert_opaque_verification_failure(pkcs1_key.verify_tls_certificate_signature_scheme(
    0x0401,
    pkcs1v15_fixture_message(),
    short_pkcs1_signature,
  ));
  assert_opaque_verification_failure(pkcs1_key.verify_tls_certificate_signature_scheme(
    0x0401,
    pkcs1v15_fixture_message(),
    out_of_range_pkcs1_signature,
  ));
  assert_opaque_verification_failure(pkcs1_key.verify_tls_certificate_signature_scheme_with_scratch(
    0x0401,
    pkcs1v15_fixture_message(),
    &tampered_pkcs1_signature,
    &mut tls_certificate_scratch,
  ));

  let issuer = RsaX509PublicKey::from_spki_der(&x509_certificate_fixture_public_key()).unwrap();
  let mut certificate_scratch = issuer.public_key().public_scratch();
  let mut tampered_certificate = x509_pkcs1v15_certificate_fixture();
  *tampered_certificate.last_mut().unwrap() ^= 0x01;
  let certificate_with_short_signature = x509_certificate(
    &minimal_tbs_certificate(&pkcs1_algorithm),
    &pkcs1_algorithm,
    short_pkcs1_signature,
  );
  let certificate_with_out_of_range_signature = x509_certificate(
    &minimal_tbs_certificate(&pkcs1_algorithm),
    &pkcs1_algorithm,
    issuer.public_key().modulus(),
  );
  assert_opaque_verification_failure(issuer.verify_x509_certificate_signature_der(&[0x30, 0x00]));
  assert_opaque_verification_failure(issuer.verify_x509_certificate_signature_der(&certificate_with_short_signature));
  assert_opaque_verification_failure(
    issuer.verify_x509_certificate_signature_der(&certificate_with_out_of_range_signature),
  );
  assert_opaque_verification_failure(
    issuer.verify_x509_certificate_signature_der_with_scratch(&tampered_certificate, &mut certificate_scratch),
  );
}

#[test]
fn rsa_policy_boundary_separates_legacy_rsa2048_from_modern_rsa3072_verification() {
  let legacy_policy = RsaPublicKeyPolicy::legacy_verification();
  let modern_policy = RsaPublicKeyPolicy::modern_verification();

  let legacy_pss_key = RsaPublicKey::from_spki_der_with_policy(&pss_fixture_public_key(), &legacy_policy).unwrap();
  assert_eq!(legacy_pss_key.modulus_bits(), 2048);
  assert!(
    legacy_pss_key
      .verify_pss(
        RsaPssProfile::Sha256,
        pss_fixture_message(),
        &pss_fixture_signature_sha256()
      )
      .is_ok()
  );
  assert_eq!(
    RsaPublicKey::from_spki_der_with_policy(&pss_fixture_public_key(), &modern_policy),
    Err(RsaKeyError::InvalidModulus)
  );

  let legacy_pkcs1v15_key =
    RsaPublicKey::from_spki_der_with_policy(&pkcs1v15_fixture_public_key(), &legacy_policy).unwrap();
  assert_eq!(legacy_pkcs1v15_key.modulus_bits(), 2048);
  assert!(
    legacy_pkcs1v15_key
      .verify_pkcs1v15(
        RsaPkcs1v15Profile::Sha256,
        pkcs1v15_fixture_message(),
        &pkcs1v15_fixture_signature_sha256(),
      )
      .is_ok()
  );
  assert_eq!(
    RsaPublicKey::from_spki_der_with_policy(&pkcs1v15_fixture_public_key(), &modern_policy),
    Err(RsaKeyError::InvalidModulus)
  );

  let modern_key = RsaPublicKey::from_spki_der_with_policy(RSA3072_SPKI, &modern_policy).unwrap();
  assert_eq!(modern_key.modulus_bits(), 3072);
  assert!(
    modern_key
      .verify_pss(RsaPssProfile::Sha256, pss_fixture_message(), RSA3072_PSS_SHA256)
      .is_ok()
  );
  assert!(
    modern_key
      .verify_pkcs1v15(
        RsaPkcs1v15Profile::Sha256,
        pkcs1v15_fixture_message(),
        RSA3072_PKCS1V15_SHA256
      )
      .is_ok()
  );
}

#[test]
fn public_key_der_round_trips_through_canonical_test_encoder() {
  let key = RsaPublicKey::from_spki_der(&pss_fixture_public_key()).unwrap();
  let pkcs1 = valid_pkcs1_with_modulus_and_exponent(key.modulus(), &exponent_bytes(key.public_exponent().as_u64()));
  let spki = spki_for_pkcs1(&pkcs1);

  let pkcs1_key = RsaPublicKey::from_pkcs1_der(&pkcs1).unwrap();
  let spki_key = RsaPublicKey::from_spki_der(&spki).unwrap();

  assert_eq!(pkcs1_key, key);
  assert_eq!(spki_key, key);
}

#[test]
fn spki_public_key_rejects_wrong_algorithm_oid() {
  let pkcs1 = valid_pkcs1();
  let mut algorithm = Vec::new();
  algorithm.extend_from_slice(&oid(&[0x2a, 0x03, 0x04]));
  algorithm.extend_from_slice(&null());

  let mut subject_public_key = vec![0];
  subject_public_key.extend_from_slice(&pkcs1);

  let mut spki = Vec::new();
  spki.extend_from_slice(&sequence(&algorithm));
  spki.extend_from_slice(&bit_string(&subject_public_key));

  assert_eq!(
    RsaPublicKey::from_spki_der(&sequence(&spki)),
    Err(RsaKeyError::UnsupportedAlgorithm)
  );
}

#[test]
fn spki_public_key_rejects_missing_null_parameters() {
  let pkcs1 = valid_pkcs1();
  let mut subject_public_key = vec![0];
  subject_public_key.extend_from_slice(&pkcs1);

  let mut spki = Vec::new();
  spki.extend_from_slice(&sequence(&oid(RSA_ENCRYPTION_OID)));
  spki.extend_from_slice(&bit_string(&subject_public_key));

  assert_eq!(
    RsaPublicKey::from_spki_der(&sequence(&spki)),
    Err(RsaKeyError::MalformedDer)
  );
}

#[test]
fn spki_public_key_rejects_nonzero_bit_string_unused_bits() {
  let pkcs1 = valid_pkcs1();
  let mut subject_public_key = vec![1];
  subject_public_key.extend_from_slice(&pkcs1);

  let mut spki = Vec::new();
  let mut algorithm = Vec::new();
  algorithm.extend_from_slice(&oid(RSA_ENCRYPTION_OID));
  algorithm.extend_from_slice(&null());
  spki.extend_from_slice(&sequence(&algorithm));
  spki.extend_from_slice(&bit_string(&subject_public_key));

  assert_eq!(
    RsaPublicKey::from_spki_der(&sequence(&spki)),
    Err(RsaKeyError::MalformedDer)
  );
}

#[test]
fn pkcs1_public_key_rejects_trailing_data() {
  let mut der = valid_pkcs1();
  der.push(0);

  assert_eq!(RsaPublicKey::from_pkcs1_der(&der), Err(RsaKeyError::MalformedDer));
}

#[test]
fn pkcs1_public_key_rejects_indefinite_and_overlong_lengths() {
  assert_eq!(
    RsaPublicKey::from_pkcs1_der(&[0x30, 0x80, 0x00, 0x00]),
    Err(RsaKeyError::MalformedDer)
  );
  assert_eq!(
    RsaPublicKey::from_pkcs1_der(&[0x30, 0x81, 0x01, 0x00]),
    Err(RsaKeyError::MalformedDer)
  );
}

#[test]
fn public_key_der_rejects_malformed_corpus() {
  let mut nested_extra = Vec::new();
  nested_extra.extend_from_slice(&integer_unsigned(&rsa2048_modulus()));
  nested_extra.extend_from_slice(&integer_unsigned(&[0x01, 0x00, 0x01]));
  nested_extra.extend_from_slice(&null());

  let long_exponent = pkcs1_with_parts(
    integer_unsigned(&rsa2048_modulus()),
    integer_unsigned(&[1, 2, 3, 4, 5, 6, 7, 8, 9]),
  );
  let noncanonical_exponent = pkcs1_with_parts(
    integer_unsigned(&rsa2048_modulus()),
    tlv(0x02, &[0x00, 0x01, 0x00, 0x01]),
  );

  let mut algorithm = Vec::new();
  algorithm.extend_from_slice(&oid(RSA_ENCRYPTION_OID));
  algorithm.extend_from_slice(&null());
  let empty_bit_string_spki = {
    let mut spki = Vec::new();
    spki.extend_from_slice(&sequence(&algorithm));
    spki.extend_from_slice(&bit_string(&[]));
    sequence(&spki)
  };
  let nested_garbage_spki = {
    let mut spki = Vec::new();
    spki.extend_from_slice(&sequence(&algorithm));
    spki.extend_from_slice(&bit_string(&[0, 0x30, 0x03, 0x05, 0x00, 0x00]));
    sequence(&spki)
  };

  for (der, expected) in [
    (Vec::new(), RsaKeyError::MalformedDer),
    (vec![0x30, 0x80, 0x00, 0x00], RsaKeyError::MalformedDer),
    (vec![0x30, 0x89], RsaKeyError::MalformedDer),
    (vec![0x30, 0x81, 0x7f], RsaKeyError::MalformedDer),
    (vec![0x30, 0x82, 0x01, 0x00, 0x02], RsaKeyError::MalformedDer),
    (sequence(&nested_extra), RsaKeyError::MalformedDer),
    (noncanonical_exponent, RsaKeyError::MalformedDer),
    (long_exponent, RsaKeyError::InvalidPublicExponent),
  ] {
    assert_eq!(RsaPublicKey::from_pkcs1_der(&der), Err(expected));
  }

  for der in [empty_bit_string_spki, nested_garbage_spki] {
    assert_eq!(RsaPublicKey::from_spki_der(&der), Err(RsaKeyError::MalformedDer));
  }
}

#[test]
fn pkcs1_public_key_rejects_unnecessary_integer_sign_padding() {
  let mut padded_modulus = rsa2048_modulus();
  padded_modulus[0] = 0x7f;
  let mut modulus_value = vec![0];
  modulus_value.extend_from_slice(&padded_modulus);

  let der = pkcs1_with_parts(tlv(0x02, &modulus_value), integer_unsigned(&[0x01, 0x00, 0x01]));

  assert_eq!(RsaPublicKey::from_pkcs1_der(&der), Err(RsaKeyError::MalformedDer));
}

#[test]
fn pkcs1_public_key_rejects_negative_integer_encoding() {
  let der = pkcs1_with_parts(tlv(0x02, &rsa2048_modulus()), integer_unsigned(&[0x01, 0x00, 0x01]));

  assert_eq!(RsaPublicKey::from_pkcs1_der(&der), Err(RsaKeyError::MalformedDer));
}

#[test]
fn pkcs1_public_key_rejects_zero_even_and_tiny_moduli() {
  let exponent = integer_unsigned(&[0x01, 0x00, 0x01]);

  let zero = pkcs1_with_parts(integer_unsigned(&[0]), exponent.clone());
  assert_eq!(RsaPublicKey::from_pkcs1_der(&zero), Err(RsaKeyError::InvalidModulus));

  let mut even_modulus = rsa2048_modulus();
  even_modulus[255] = 0x02;
  let even = pkcs1_with_parts(integer_unsigned(&even_modulus), exponent.clone());
  assert_eq!(RsaPublicKey::from_pkcs1_der(&even), Err(RsaKeyError::InvalidModulus));

  let tiny = pkcs1_with_parts(integer_unsigned(&[0x01, 0x01]), exponent);
  assert_eq!(RsaPublicKey::from_pkcs1_der(&tiny), Err(RsaKeyError::InvalidModulus));
}

#[test]
fn pkcs1_public_key_enforces_modern_minimum_when_requested() {
  let policy = RsaPublicKeyPolicy::modern_verification();

  assert_eq!(
    RsaPublicKey::from_pkcs1_der_with_policy(&valid_pkcs1(), &policy),
    Err(RsaKeyError::InvalidModulus)
  );
}

#[test]
fn pkcs1_public_key_rejects_even_and_policy_disallowed_exponents() {
  let modulus = integer_unsigned(&rsa2048_modulus());

  let even = pkcs1_with_parts(modulus.clone(), integer_unsigned(&[0x02]));
  assert_eq!(
    RsaPublicKey::from_pkcs1_der(&even),
    Err(RsaKeyError::InvalidPublicExponent)
  );

  let exponent_three = pkcs1_with_parts(modulus, integer_unsigned(&[0x03]));
  assert_eq!(
    RsaPublicKey::from_pkcs1_der(&exponent_three),
    Err(RsaKeyError::InvalidPublicExponent)
  );
}

#[test]
fn pkcs1_public_key_can_accept_legacy_small_fermat_exponents_by_policy() {
  let policy = RsaPublicKeyPolicy::legacy_verification().allow_legacy_small_exponents();
  let der = pkcs1_with_parts(integer_unsigned(&rsa2048_modulus()), integer_unsigned(&[0x03]));

  let key = RsaPublicKey::from_pkcs1_der_with_policy(&der, &policy).unwrap();

  assert_eq!(key.public_exponent().as_u64(), 3);
}

#[test]
fn public_operation_rejects_wrong_length_and_out_of_range_representatives() {
  let key = RsaPublicKey::from_pkcs1_der(&valid_pkcs1()).unwrap();
  let mut out = vec![0u8; key.modulus().len()];

  assert_eq!(
    key.public_operation(&[1, 2, 3], &mut out),
    Err(RsaPublicOpError::InvalidLength)
  );

  let modulus = key.modulus().to_vec();
  assert_eq!(
    key.public_operation(&modulus, &mut out),
    Err(RsaPublicOpError::RepresentativeOutOfRange)
  );
}

#[test]
fn public_operation_matches_independent_65537_vector() {
  let modulus = hex_to_vec(
    "\
ef8bb02b8e4aec1abc6fac7a0d6fb1f2649bb86a1567423fee4a194a250461a9db702558e92e52cc\
907963d84731a7adaf4c609e1b7c7d7c187099a43857f7628f5d20416fcb48987c9d6f12cfc6bc\
260c9b5506be3fe3cd218ddb37ef5b30feb16172a9832312726ed135c0540ef9d3229b87b5566f\
3355c90f301b856aa822878269806079ab7267cdc6c7403d7be3fa652065b2d39f2dbf9fb61ed9\
71fee37432ebe31d9aa465dbae96b0edd5ffddf1b49e03346a02290fed1e4e31f6b3b6e1f839f\
d5add90a8a212c10dd997b0a4efcb3df990808509dcb28c504e0649827a83ffd864395d1f62f2\
9a004f44423a44b07de943a60fba844a9da3603ce5c5",
  );
  let input = hex_to_vec(
    "\
3450869c4ccbee98815e55cb42f2dd85a3427d3f65e33d29352293e18cde9582a9fbc54b440984\
1ba8d931a9a9411192516a9fbd3a7b886e7f8b8f3f7bb5403309eee9d7234df0b5934e18a1dc\
9e3b568a3fab6947cefe50500abcbda19fd9ab7b7e90a95801e36a020ba79bdc94346198d98131\
6864a06a43448b62acb7a8472661323175f04c5e447d0017e4073efc55f59f79f34aaa3be8ae7\
0d26db78b25e9dfb23856d1b1e024aedfcfd649d209412c0c80832ca3466965eeff539afe791f\
451b554e212cff4d92466438062c5202169b0adf0c95b7d3d31414602cf9d185252b550cc2e8f\
5be08b7fc71f51210ff88363badadfaf5c2915c3a10b2389e",
  );
  let expected = hex_to_vec(
    "\
020d016f2b1394b5ec00d4ddd1725435747a31fd4b2489fad76060b68b2259089d304b1d3c98\
e0a343c4b313d15b6022b0400dde22538f30e8474d483189c04ece5acc8aaf1481b362c2d2fe7\
fa853e856a0aba66cc47cf9e59052fdd4c5f4155bcc2a3f3330e2c48b7f45d1e66d8cd04829c\
0ba2e598569b4eeb8538c3cdf8e02c838d04bdc661b5d8c5291b0feebf284eb9deea03dd0226\
bdb322e180a6ab522ee40a02a0daf41094a2938d39698ab16381ed4d3ddd01bd05a8aa9113d8\
ec34e8c72cc58fd5324fbe1ddd9714909caedfaa38706cfa66d9bc1026ba3ec1188092392a54a\
3e94bf239ee74517b71ec2464551f8174dbd0f3952ffb41070c754",
  );

  let key =
    RsaPublicKey::from_pkcs1_der(&valid_pkcs1_with_modulus_and_exponent(&modulus, &[0x01, 0x00, 0x01])).unwrap();
  let mut out = vec![0u8; key.modulus().len()];
  let mut scratch = key.public_scratch();

  key
    .public_operation_with_scratch(&input, &mut out, &mut scratch)
    .unwrap();

  assert_eq!(out, expected);
}

#[test]
fn public_operation_matches_independent_legacy_exponent_vectors() {
  let modulus = hex_to_vec(
    "\
ef8bb02b8e4aec1abc6fac7a0d6fb1f2649bb86a1567423fee4a194a250461a9db702558e92e52cc\
907963d84731a7adaf4c609e1b7c7d7c187099a43857f7628f5d20416fcb48987c9d6f12cfc6bc\
260c9b5506be3fe3cd218ddb37ef5b30feb16172a9832312726ed135c0540ef9d3229b87b5566f\
3355c90f301b856aa822878269806079ab7267cdc6c7403d7be3fa652065b2d39f2dbf9fb61ed9\
71fee37432ebe31d9aa465dbae96b0edd5ffddf1b49e03346a02290fed1e4e31f6b3b6e1f839f\
d5add90a8a212c10dd997b0a4efcb3df990808509dcb28c504e0649827a83ffd864395d1f62f2\
9a004f44423a44b07de943a60fba844a9da3603ce5c5",
  );
  let input = hex_to_vec(
    "\
3450869c4ccbee98815e55cb42f2dd85a3427d3f65e33d29352293e18cde9582a9fbc54b440984\
1ba8d931a9a9411192516a9fbd3a7b886e7f8b8f3f7bb5403309eee9d7234df0b5934e18a1dc\
9e3b568a3fab6947cefe50500abcbda19fd9ab7b7e90a95801e36a020ba79bdc94346198d98131\
6864a06a43448b62acb7a8472661323175f04c5e447d0017e4073efc55f59f79f34aaa3be8ae7\
0d26db78b25e9dfb23856d1b1e024aedfcfd649d209412c0c80832ca3466965eeff539afe791f\
451b554e212cff4d92466438062c5202169b0adf0c95b7d3d31414602cf9d185252b550cc2e8f\
5be08b7fc71f51210ff88363badadfaf5c2915c3a10b2389e",
  );
  let expected_e3 = hex_to_vec(
    "\
c47adb0961fe847a1520b2e3a9f65ce6536dbf7599508d852dfc1b9fb958f06f5917aed756d7c\
7b21f48d8b36ab70d030d735b402d247de55c2c50d80a4f48ebcc5e7219c4964e58305461f31\
a3babb3d8d16ad3f74551fc6b99bbe06497893f9db71ad82fffa0c1830711cc5bb3c3980209b\
2b515848b8e2f3a1d53ab483e7046ce9e0c570c96f3325349bc6d58ed1e1d3be2ae699905ce6\
e52eed21d22185f8510c402d7dbe737957e4628b17668e4fa1016c779a20014d9bef37798698c\
d10d641d758aec54de1511cb5f7085353500c7c49e4ecb975aa4f2616529d84559dc8ded7380\
112eaddf2de6f28a6856fead34ca2149560e75e41598ea59684e77",
  );
  let expected_e17 = hex_to_vec(
    "\
c0f626ce19c684b15078eef9ccce75d142051cc382529056a83babb680795e24f599379beade07\
a2e2963eef379848cc9e8d6138b3e155f02f0a25582658befd513c3d375a2e2de7d431b2019cd\
beb3e1254594b0431c85f333c4a6d280825dbd88993e2bf8bbf360229a673d93bd5b54f3f15e7\
0776f6bb7b61579e93fc16ff5a27c6c76db3b0a77421c748553048ec4392e8e0a79e2ac62c0\
3822f29525afb3e409fdff2f119403da562f923fa962d18caa133495463f220fea53806a01637\
f40f8866e140a78f5bf55491d354087ad3c938d0b9f7b77eebcdcd16b7b628c26f18ae434c15\
fc87eb5a2ea7a141091f3d2fa37a581ef39e6e496e5e476d43b6",
  );

  let policy = RsaPublicKeyPolicy::legacy_verification().allow_legacy_small_exponents();

  let key_e3 =
    RsaPublicKey::from_pkcs1_der_with_policy(&valid_pkcs1_with_modulus_and_exponent(&modulus, &[0x03]), &policy)
      .unwrap();
  let key_e17 =
    RsaPublicKey::from_pkcs1_der_with_policy(&valid_pkcs1_with_modulus_and_exponent(&modulus, &[0x11]), &policy)
      .unwrap();

  let mut out = vec![0u8; modulus.len()];
  key_e3.public_operation(&input, &mut out).unwrap();
  assert_eq!(out, expected_e3);

  key_e17.public_operation(&input, &mut out).unwrap();
  assert_eq!(out, expected_e17);
}

#[test]
fn public_scratch_reuses_after_modulus_minus_one_operation() {
  let key = RsaPublicKey::from_spki_der(&pss_fixture_public_key()).unwrap();
  let representative = modulus_minus_one(&key);
  let mut out = vec![0u8; key.modulus().len()];
  let mut scratch = key.public_scratch();

  for _ in 0..1024 {
    key
      .public_operation_with_scratch(
        core::hint::black_box(&representative),
        core::hint::black_box(&mut out),
        &mut scratch,
      )
      .unwrap();
  }
  key
    .verify_pss_with_scratch(
      RsaPssProfile::Sha256,
      pss_fixture_message(),
      &pss_fixture_signature_sha256(),
      &mut scratch,
    )
    .unwrap();
}

#[cfg(feature = "diag")]
#[test]
fn public_operation_bitserial_baseline_matches_montgomery_path() {
  let modulus = hex_to_vec(
    "\
ef8bb02b8e4aec1abc6fac7a0d6fb1f2649bb86a1567423fee4a194a250461a9db702558e92e52cc\
907963d84731a7adaf4c609e1b7c7d7c187099a43857f7628f5d20416fcb48987c9d6f12cfc6bc\
260c9b5506be3fe3cd218ddb37ef5b30feb16172a9832312726ed135c0540ef9d3229b87b5566f\
3355c90f301b856aa822878269806079ab7267cdc6c7403d7be3fa652065b2d39f2dbf9fb61ed9\
71fee37432ebe31d9aa465dbae96b0edd5ffddf1b49e03346a02290fed1e4e31f6b3b6e1f839f\
d5add90a8a212c10dd997b0a4efcb3df990808509dcb28c504e0649827a83ffd864395d1f62f2\
9a004f44423a44b07de943a60fba844a9da3603ce5c5",
  );
  let input = hex_to_vec(
    "\
3450869c4ccbee98815e55cb42f2dd85a3427d3f65e33d29352293e18cde9582a9fbc54b440984\
1ba8d931a9a9411192516a9fbd3a7b886e7f8b8f3f7bb5403309eee9d7234df0b5934e18a1dc\
9e3b568a3fab6947cefe50500abcbda19fd9ab7b7e90a95801e36a020ba79bdc94346198d98131\
6864a06a43448b62acb7a8472661323175f04c5e447d0017e4073efc55f59f79f34aaa3be8ae7\
0d26db78b25e9dfb23856d1b1e024aedfcfd649d209412c0c80832ca3466965eeff539afe791f\
451b554e212cff4d92466438062c5202169b0adf0c95b7d3d31414602cf9d185252b550cc2e8f\
5be08b7fc71f51210ff88363badadfaf5c2915c3a10b2389e",
  );
  let small_policy = RsaPublicKeyPolicy::legacy_verification().allow_legacy_small_exponents();
  let odd_policy = small_policy.allow_legacy_odd_exponents();

  for (exponent, policy) in [
    (&[0x03][..], small_policy),
    (&[0x11][..], small_policy),
    (&[0x01, 0x00, 0x01][..], RsaPublicKeyPolicy::legacy_verification()),
    (&[0x49, 0xd2, 0xa1][..], odd_policy),
  ] {
    let key =
      RsaPublicKey::from_pkcs1_der_with_policy(&valid_pkcs1_with_modulus_and_exponent(&modulus, exponent), &policy)
        .unwrap();
    let mut current = vec![0u8; modulus.len()];
    let mut product = vec![0u8; modulus.len()];
    let mut bitserial = vec![0u8; modulus.len()];
    let mut scratch = key.public_scratch();
    let mut product_scratch = key.public_scratch();

    key
      .public_operation_with_scratch(&input, &mut current, &mut scratch)
      .unwrap();
    diag_rsa_public_operation_product(&key, &input, &mut product, &mut product_scratch).unwrap();
    diag_rsa_public_operation_bitserial(&key, &input, &mut bitserial).unwrap();

    assert_eq!(
      product, current,
      "product Montgomery mismatch for exponent {exponent:02x?}"
    );
    assert_eq!(
      bitserial, current,
      "bit-serial baseline mismatch for exponent {exponent:02x?}"
    );
  }
}

#[cfg(feature = "diag")]
#[test]
fn public_operation_montgomery_candidates_match_current_path() {
  for (spki, signature) in [
    (&pss_fixture_public_key()[..], &pss_fixture_signature_sha256()[..]),
    (RSA3072_SPKI, RSA3072_PSS_SHA256),
    (RSA4096_SPKI, RSA4096_PSS_SHA256),
    (RSA8192_SPKI, RSA8192_PSS_SHA256),
  ] {
    let key = RsaPublicKey::from_spki_der(spki).unwrap();
    let representative = modulus_minus_one(&key);
    let mut current = vec![0u8; key.modulus().len()];
    let mut cios = vec![0u8; key.modulus().len()];
    let mut product = vec![0u8; key.modulus().len()];
    let mut scratch = key.public_scratch();
    let mut cios_scratch = key.public_scratch();
    let mut product_scratch = key.public_scratch();

    key
      .public_operation_with_scratch(&representative, &mut current, &mut scratch)
      .unwrap();
    diag_rsa_public_operation_cios(&key, &representative, &mut cios, &mut cios_scratch).unwrap();
    diag_rsa_public_operation_product(&key, &representative, &mut product, &mut product_scratch).unwrap();
    assert_eq!(cios, current, "CIOS mismatch for modulus-minus-one representative");
    assert_eq!(
      product, current,
      "product Montgomery mismatch for modulus-minus-one representative"
    );

    key
      .public_operation_with_scratch(signature, &mut current, &mut scratch)
      .unwrap();
    diag_rsa_public_operation_cios(&key, signature, &mut cios, &mut cios_scratch).unwrap();
    diag_rsa_public_operation_product(&key, signature, &mut product, &mut product_scratch).unwrap();
    assert_eq!(cios, current, "CIOS mismatch for fixture signature representative");
    assert_eq!(
      product, current,
      "product Montgomery mismatch for fixture signature representative"
    );
  }
}

#[test]
fn pss_verify_accepts_openssl_sha256_sha384_and_sha512_vectors() {
  let key = RsaPublicKey::from_spki_der(&pss_fixture_public_key()).unwrap();
  let message = pss_fixture_message();

  let sig256 = hex_to_vec(
    "\
2641e0207f279b526767343e03007a293a0523db41828c0e335c10dea7ab7dec988ad09cd220d\
1754f5f89f7cd3b2a9ab1f315709b019989ee96e9060d3158d0240b3f8dbed179c55c1b0fa78\
f31249d706256748c325ded4835224e80b3daa066489d1cf28a4062fa4129b21723f6336f8c\
55d6785cd2d284437748b78b47e0162da7cddd61df0536b378a1cc5c327cba76db99c253795c\
19b49007146e44dbe47f3ee9c2da2248710cd264661815bd5508f604d6ee4a663a46c472f6d\
323739fe6b142ddc3b006a9c113d3b81da524e0fe358f9cb141f686dc459b66b1150e5418e8\
b6fcf2590d0706da27017429d91fe9f521f9fbb2ae2044f2eecfe87c7d",
  );
  let sig384 = hex_to_vec(
    "\
008d755ce5e0516de98e9ea2f638183c9dfe8b7d1946abbf5621a102b7939f9dc75aecf1bde\
af35180dc5215139c2d9bed55513f955da77b3f6308e68755a0af1acde93a5b90173b4705cb1\
042fecae543f89fc2a52ef4f23b0c4435f8fead0feeee5d74a8f51224b57feb3777bea075e4a\
313302a6abb60eb8ab9356c37e4d4bb0525bb0c210e5a72d7b52729c76ef87217888e0780c97\
5447b03f3d6489b5091d47644778037c6b75d8617f3668f61265c60faa5893741d8e32db751d\
7b7dc14e39da782dcad8283745fb484f2dff8b122271c5e80b523eab04d52dfd9f1d06ecb04\
6a3d8d7b124cb14d3b5797787104daa27c2e0ce6ccb9f0c4d8afc5a05e",
  );
  let sig512 = hex_to_vec(
    "\
95989be018de687a9383e0880b1618e16158b3defd0870aab1eba35d33e381836b1f6086ec78\
aec5205b05ff989176db1199e8d34341380f501c34973526d024ef9fd87108e041c16625937a\
7407a977947843f9159cf83c305c07484b5a338e471a9e07492f95b3f1c7eafa140551cbac08\
0f49c57fbfc420b82243aae9a81c32f15cdde15b4d13ea0419d1f0df59de01a8ff97380e2eb\
04722703f458b9d2233ac58b909aa72343b5ce0de367ea1990c6e59ebb596f38f8d44a13e63a\
07b3dd8fde0aa8e582ad5f74fe9fc1c309fca89c65ce5dd5bb132e3d0cb6a350dda9dde14\
c2b691954f9bd86140e31acf6a8a2b9d28cba358e509dfc234c1e33e223c",
  );

  assert!(key.verify_pss(RsaPssProfile::Sha256, message, &sig256).is_ok());
  assert!(key.verify_pss(RsaPssProfile::Sha384, message, &sig384).is_ok());
  assert!(key.verify_pss(RsaPssProfile::Sha512, message, &sig512).is_ok());
}

#[test]
fn pss_verify_rejects_tampered_signature_message_and_profile() {
  let key = RsaPublicKey::from_spki_der(&pss_fixture_public_key()).unwrap();
  let message = pss_fixture_message();
  let mut sig = hex_to_vec(
    "\
2641e0207f279b526767343e03007a293a0523db41828c0e335c10dea7ab7dec988ad09cd220d\
1754f5f89f7cd3b2a9ab1f315709b019989ee96e9060d3158d0240b3f8dbed179c55c1b0fa78\
f31249d706256748c325ded4835224e80b3daa066489d1cf28a4062fa4129b21723f6336f8c\
55d6785cd2d284437748b78b47e0162da7cddd61df0536b378a1cc5c327cba76db99c253795c\
19b49007146e44dbe47f3ee9c2da2248710cd264661815bd5508f604d6ee4a663a46c472f6d\
323739fe6b142ddc3b006a9c113d3b81da524e0fe358f9cb141f686dc459b66b1150e5418e8\
b6fcf2590d0706da27017429d91fe9f521f9fbb2ae2044f2eecfe87c7d",
  );

  assert!(key.verify_pss(RsaPssProfile::Sha384, message, &sig).is_err());
  assert!(key.verify_pss(RsaPssProfile::Sha256, b"wrong message", &sig).is_err());

  sig[17] ^= 0x80;
  assert!(key.verify_pss(RsaPssProfile::Sha256, message, &sig).is_err());
}

#[test]
fn typed_signature_profile_dispatches_and_rejects_algorithm_confusion() {
  let pss_key = RsaPublicKey::from_spki_der(&pss_fixture_public_key()).unwrap();
  let pss_sig = pss_fixture_signature_sha256();
  let mut pss_scratch = pss_key.public_scratch();
  assert!(
    pss_key
      .verify_signature(
        RsaSignatureProfile::pss(RsaPssProfile::Sha256),
        pss_fixture_message(),
        &pss_sig,
      )
      .is_ok()
  );
  assert!(
    pss_key
      .verify_signature_with_scratch(
        RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha256, 32),
        pss_fixture_message(),
        &pss_sig,
        &mut pss_scratch,
      )
      .is_ok()
  );
  assert_eq!(
    pss_key.verify_signature(
      RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256),
      pss_fixture_message(),
      &pss_sig,
    ),
    Err(VerificationError::new())
  );
  assert_eq!(
    pss_key.verify_signature(
      RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha256, 0),
      pss_fixture_message(),
      &pss_sig,
    ),
    Err(VerificationError::new())
  );

  let pkcs1_key = RsaPublicKey::from_spki_der(&pkcs1v15_fixture_public_key()).unwrap();
  let pkcs1_sig = pkcs1v15_fixture_signature_sha256();
  let mut pkcs1_scratch = pkcs1_key.public_scratch();
  assert!(
    pkcs1_key
      .verify_signature(
        RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256),
        pkcs1v15_fixture_message(),
        &pkcs1_sig,
      )
      .is_ok()
  );
  assert!(
    pkcs1_key
      .verify_signature_with_scratch(
        RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256),
        pkcs1v15_fixture_message(),
        &pkcs1_sig,
        &mut pkcs1_scratch,
      )
      .is_ok()
  );
  assert_eq!(
    pkcs1_key.verify_signature(
      RsaSignatureProfile::pss(RsaPssProfile::Sha256),
      pkcs1v15_fixture_message(),
      &pkcs1_sig,
    ),
    Err(VerificationError::new())
  );
}

#[test]
fn protocol_algorithm_mappers_are_explicit_and_fail_closed() {
  assert_eq!(
    RsaSignatureProfile::from_tls13_signature_scheme(0x0804),
    Ok(RsaSignatureProfile::pss(RsaPssProfile::Sha256))
  );
  assert_eq!(
    RsaSignatureProfile::from_tls13_signature_scheme(0x080a),
    Ok(RsaSignatureProfile::pss(RsaPssProfile::Sha384))
  );
  assert_eq!(
    RsaSignatureProfile::from_tls_certificate_signature_scheme(0x0401),
    Ok(RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256))
  );
  assert_eq!(
    RsaSignatureProfile::from_tls_certificate_signature_scheme(0x0601),
    Ok(RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha512))
  );
  assert_eq!(
    RsaSignatureProfile::from_jwt_alg("PS256"),
    Ok(RsaSignatureProfile::pss(RsaPssProfile::Sha256))
  );
  assert_eq!(
    RsaSignatureProfile::from_jwt_alg("RS512"),
    Ok(RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha512))
  );
  assert_eq!(
    RsaSignatureProfile::from_cose_algorithm_id(-37),
    Ok(RsaSignatureProfile::pss(RsaPssProfile::Sha256))
  );
  assert_eq!(
    RsaSignatureProfile::from_cose_algorithm_id(-259),
    Ok(RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha512))
  );

  for tls_scheme in [0x0201, 0x0203, 0x0403, 0x0401, 0xffff] {
    assert_eq!(
      RsaSignatureProfile::from_tls13_signature_scheme(tls_scheme),
      Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
    );
  }
  for tls_scheme in [0x0201, 0x0203, 0x0403, 0xffff] {
    assert_eq!(
      RsaSignatureProfile::from_tls_certificate_signature_scheme(tls_scheme),
      Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
    );
  }
  for jwt_alg in ["none", "HS256", "ES256", "EdDSA", "RS1", "PS1", "rs256"] {
    assert_eq!(
      RsaSignatureProfile::from_jwt_alg(jwt_alg),
      Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
    );
  }
  for cose_algorithm in [0, 1, -7, -65535, 0x7fff_ffff] {
    assert_eq!(
      RsaSignatureProfile::from_cose_algorithm_id(cose_algorithm),
      Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
    );
  }
}

#[test]
fn protocol_mapped_profiles_verify_and_reject_algorithm_confusion() {
  let pss_key = RsaPublicKey::from_spki_der(&pss_fixture_public_key()).unwrap();
  let pss_sig = pss_fixture_signature_sha256();
  assert!(
    pss_key
      .verify_signature(
        RsaSignatureProfile::from_tls13_signature_scheme(0x0804).unwrap(),
        pss_fixture_message(),
        &pss_sig,
      )
      .is_ok()
  );
  assert!(
    pss_key
      .verify_signature(
        RsaSignatureProfile::from_jwt_alg("PS256").unwrap(),
        pss_fixture_message(),
        &pss_sig,
      )
      .is_ok()
  );
  assert_eq!(
    pss_key.verify_signature(
      RsaSignatureProfile::from_cose_algorithm_id(-257).unwrap(),
      pss_fixture_message(),
      &pss_sig,
    ),
    Err(VerificationError::new())
  );

  let pkcs1_key = RsaPublicKey::from_spki_der(&pkcs1v15_fixture_public_key()).unwrap();
  let pkcs1_sig = pkcs1v15_fixture_signature_sha256();
  assert!(
    pkcs1_key
      .verify_signature(
        RsaSignatureProfile::from_tls_certificate_signature_scheme(0x0401).unwrap(),
        pkcs1v15_fixture_message(),
        &pkcs1_sig,
      )
      .is_ok()
  );
  assert!(
    pkcs1_key
      .verify_signature(
        RsaSignatureProfile::from_jwt_alg("RS256").unwrap(),
        pkcs1v15_fixture_message(),
        &pkcs1_sig,
      )
      .is_ok()
  );
  assert_eq!(
    pkcs1_key.verify_signature(
      RsaSignatureProfile::from_cose_algorithm_id(-37).unwrap(),
      pkcs1v15_fixture_message(),
      &pkcs1_sig,
    ),
    Err(VerificationError::new())
  );
}

#[test]
fn jwt_and_cose_verification_helpers_reject_algorithm_confusion() {
  let pss_key = RsaPublicKey::from_spki_der(&pss_fixture_public_key()).unwrap();
  let pss_sig = pss_fixture_signature_sha256();
  let mut scratch = pss_key.public_scratch();

  assert!(pss_key.verify_jwt_alg("PS256", pss_fixture_message(), &pss_sig).is_ok());
  assert!(
    pss_key
      .verify_cose_algorithm_id_with_scratch(-37, pss_fixture_message(), &pss_sig, &mut scratch)
      .is_ok()
  );
  for result in [
    pss_key.verify_jwt_alg("RS256", pss_fixture_message(), &pss_sig),
    pss_key.verify_jwt_alg("none", pss_fixture_message(), &pss_sig),
    pss_key.verify_jwt_alg("HS256", pss_fixture_message(), &pss_sig),
    pss_key.verify_jwt_alg("rs256", pss_fixture_message(), &pss_sig),
    pss_key.verify_cose_algorithm_id(-257, pss_fixture_message(), &pss_sig),
    pss_key.verify_cose_algorithm_id(1, pss_fixture_message(), &pss_sig),
  ] {
    assert_eq!(result, Err(VerificationError::new()));
  }

  let pkcs1_key = RsaPublicKey::from_spki_der(&pkcs1v15_fixture_public_key()).unwrap();
  let pkcs1_sig = pkcs1v15_fixture_signature_sha256();
  let mut scratch = pkcs1_key.public_scratch();

  assert!(
    pkcs1_key
      .verify_jwt_alg_with_scratch("RS256", pkcs1v15_fixture_message(), &pkcs1_sig, &mut scratch)
      .is_ok()
  );
  assert!(
    pkcs1_key
      .verify_cose_algorithm_id(-257, pkcs1v15_fixture_message(), &pkcs1_sig)
      .is_ok()
  );
  for result in [
    pkcs1_key.verify_jwt_alg("PS256", pkcs1v15_fixture_message(), &pkcs1_sig),
    pkcs1_key.verify_jwt_alg("RS1", pkcs1v15_fixture_message(), &pkcs1_sig),
    pkcs1_key.verify_jwt_alg("ES256", pkcs1v15_fixture_message(), &pkcs1_sig),
    pkcs1_key.verify_cose_algorithm_id(-37, pkcs1v15_fixture_message(), &pkcs1_sig),
    pkcs1_key.verify_cose_algorithm_id(-65535, pkcs1v15_fixture_message(), &pkcs1_sig),
  ] {
    assert_eq!(result, Err(VerificationError::new()));
  }
}

#[test]
fn x509_signature_algorithm_mapping_is_strict_and_rejects_sha1_defaults() {
  assert_eq!(
    RsaSignatureProfile::from_x509_signature_algorithm_der(&algorithm_identifier(
      SHA256_WITH_RSA_ENCRYPTION_OID,
      Some(&null())
    )),
    Ok(RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256))
  );
  assert_eq!(
    RsaSignatureProfile::from_x509_signature_algorithm_der(&algorithm_identifier(
      SHA384_WITH_RSA_ENCRYPTION_OID,
      Some(&null())
    )),
    Ok(RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha384))
  );
  assert_eq!(
    RsaSignatureProfile::from_x509_signature_algorithm_der(&algorithm_identifier(
      SHA512_WITH_RSA_ENCRYPTION_OID,
      Some(&null())
    )),
    Ok(RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha512))
  );
  assert_eq!(
    RsaSignatureProfile::from_x509_signature_algorithm_der(&x509_pss_algorithm(RsaPssProfile::Sha256, 32, None)),
    Ok(RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha256, 32))
  );
  assert_eq!(
    RsaSignatureProfile::from_x509_signature_algorithm_der(&x509_pss_algorithm(RsaPssProfile::Sha384, 48, Some(1))),
    Ok(RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha384, 48))
  );
  assert_eq!(
    RsaSignatureProfile::from_x509_signature_algorithm_der(&x509_pss_algorithm_without_salt_len(RsaPssProfile::Sha256)),
    Ok(RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha256, 20))
  );

  let sha256_without_params = algorithm_identifier(ID_SHA256_OID, None);
  let mgf1_sha256_without_params = algorithm_identifier(ID_MGF1_OID, Some(&sha256_without_params));
  let mut absent_hash_params = Vec::new();
  absent_hash_params.extend_from_slice(&context_constructed(0, &sha256_without_params));
  absent_hash_params.extend_from_slice(&context_constructed(1, &mgf1_sha256_without_params));
  absent_hash_params.extend_from_slice(&context_constructed(2, &integer_unsigned(&[32])));
  assert_eq!(
    RsaSignatureProfile::from_x509_signature_algorithm_der(&algorithm_identifier(
      ID_RSASSA_PSS_OID,
      Some(&sequence(&absent_hash_params))
    )),
    Ok(RsaSignatureProfile::pss_with_salt_len(RsaPssProfile::Sha256, 32))
  );

  assert_eq!(
    RsaSignatureProfile::from_x509_signature_algorithm_der(&algorithm_identifier(
      SHA1_WITH_RSA_ENCRYPTION_OID,
      Some(&null())
    )),
    Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
  );
  assert_eq!(
    RsaSignatureProfile::from_x509_signature_algorithm_der(&algorithm_identifier(SHA256_WITH_RSA_ENCRYPTION_OID, None)),
    Err(RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier)
  );
  assert_eq!(
    RsaSignatureProfile::from_x509_signature_algorithm_der(&algorithm_identifier(
      ID_RSASSA_PSS_OID,
      Some(&sequence(&[]))
    )),
    Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
  );

  let mut pss_params = Vec::new();
  pss_params.extend_from_slice(&context_constructed(
    0,
    &algorithm_identifier(ID_SHA1_OID, Some(&null())),
  ));
  pss_params.extend_from_slice(&context_constructed(1, &x509_mgf1_algorithm(RsaPssProfile::Sha256)));
  pss_params.extend_from_slice(&context_constructed(2, &integer_unsigned(&[32])));
  assert_eq!(
    RsaSignatureProfile::from_x509_signature_algorithm_der(&algorithm_identifier(
      ID_RSASSA_PSS_OID,
      Some(&sequence(&pss_params))
    )),
    Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
  );

  let mut mismatched_mgf = Vec::new();
  mismatched_mgf.extend_from_slice(&context_constructed(0, &x509_hash_algorithm(RsaPssProfile::Sha256)));
  mismatched_mgf.extend_from_slice(&context_constructed(1, &x509_mgf1_algorithm(RsaPssProfile::Sha384)));
  mismatched_mgf.extend_from_slice(&context_constructed(2, &integer_unsigned(&[32])));
  assert_eq!(
    RsaSignatureProfile::from_x509_signature_algorithm_der(&algorithm_identifier(
      ID_RSASSA_PSS_OID,
      Some(&sequence(&mismatched_mgf))
    )),
    Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
  );

  let mut bad_trailer = Vec::new();
  bad_trailer.extend_from_slice(&context_constructed(0, &x509_hash_algorithm(RsaPssProfile::Sha256)));
  bad_trailer.extend_from_slice(&context_constructed(1, &x509_mgf1_algorithm(RsaPssProfile::Sha256)));
  bad_trailer.extend_from_slice(&context_constructed(2, &integer_unsigned(&[32])));
  bad_trailer.extend_from_slice(&context_constructed(3, &integer_unsigned(&[2])));
  assert_eq!(
    RsaSignatureProfile::from_x509_signature_algorithm_der(&algorithm_identifier(
      ID_RSASSA_PSS_OID,
      Some(&sequence(&bad_trailer))
    )),
    Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
  );

  let sha256_with_bad_params = algorithm_identifier(ID_SHA256_OID, Some(&integer_unsigned(&[1])));
  let mut malformed_hash_params = Vec::new();
  malformed_hash_params.extend_from_slice(&context_constructed(0, &sha256_with_bad_params));
  malformed_hash_params.extend_from_slice(&context_constructed(1, &x509_mgf1_algorithm(RsaPssProfile::Sha256)));
  malformed_hash_params.extend_from_slice(&context_constructed(2, &integer_unsigned(&[32])));
  assert_eq!(
    RsaSignatureProfile::from_x509_signature_algorithm_der(&algorithm_identifier(
      ID_RSASSA_PSS_OID,
      Some(&sequence(&malformed_hash_params))
    )),
    Err(RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier)
  );
}

#[test]
fn x509_mapped_profiles_verify_real_fixtures_and_reject_padding_mismatch() {
  let pss_key = RsaPublicKey::from_spki_der(&pss_fixture_public_key()).unwrap();
  let pss_sig = pss_fixture_signature_sha256();
  assert!(
    pss_key
      .verify_signature(
        RsaSignatureProfile::from_x509_signature_algorithm_der(&x509_pss_algorithm(RsaPssProfile::Sha256, 32, None))
          .unwrap(),
        pss_fixture_message(),
        &pss_sig,
      )
      .is_ok()
  );

  let pkcs1_key = RsaPublicKey::from_spki_der(&pkcs1v15_fixture_public_key()).unwrap();
  let pkcs1_sig = pkcs1v15_fixture_signature_sha256();
  let pkcs1_profile = RsaSignatureProfile::from_x509_signature_algorithm_der(&algorithm_identifier(
    SHA256_WITH_RSA_ENCRYPTION_OID,
    Some(&null()),
  ))
  .unwrap();
  assert!(
    pkcs1_key
      .verify_signature(pkcs1_profile, pkcs1v15_fixture_message(), &pkcs1_sig)
      .is_ok()
  );
  assert_eq!(
    pss_key.verify_signature(pkcs1_profile, pss_fixture_message(), &pss_sig),
    Err(VerificationError::new())
  );
}

#[test]
fn pkcs1v15_verify_accepts_openssl_sha256_sha384_and_sha512_vectors() {
  let key = RsaPublicKey::from_spki_der(&pkcs1v15_fixture_public_key()).unwrap();
  let message = pkcs1v15_fixture_message();

  let sig256 = hex_to_vec(
    "\
94781246d705f79659d01ac8894b6f41076abe165e28711ec8fa41c1c8767b175a9c63e5118b\
d30de86da0d7b8934e963ef69c438ace976e4453dfce6b9b84a7d37a27ee61512656333dfda1\
ac40197fe4f9396bb016b25054f98f149d126c0248fc007cddc3d75d178eb34ecda0e0df822\
825ca133c062d3cdcb19e20a3e377541d8253af795a9b49a41ddb5795592502b9efbb153afc\
dd4fcc492a891d8536ef91cc228a3dbf66f0c70596f9cd101fe95d127550e7a4a9864430bd3\
4a88d8df93f4df7b54e8a4b8643891481e4bdcf87be3f98a1fdc475a819e3dc3a114aff86e\
48929a430fc39333f81064701be7d5501a3a7b4ec6c68f6feda6190d66b16",
  );
  let sig384 = hex_to_vec(
    "\
3597d240885f8562e6d590649a2bf41b61f4555d1e27de5500d0f226b13e4031de7c1cb70e3\
f9a49ef2c5964c812b70f1fd8d867179f015ff460bb21c3539a9b395a0ef7df9218325d993\
623a0cb82f17fc9c187242bea26d133065df3b1c2c4c3352ec082f5c645002ea08a71d0fee\
60640717e54337d9f98aca78dfca5d8a34ba74f6100322e758a83802c29b767a8e44e3b54d\
f2ab2b51146f9f319287c90188ab48f74513e4696990004c6cfef5a0405f4a8286b8f26304\
367799720e1fb618a6eddfe29b9dc761c2be9ab2f90be744824fcfd2f96e74ac64d57b5d50\
3447d84047f74ab708e66384826fa3da6c0f2043f21a4c7291b32095b25f42921aa",
  );
  let sig512 = hex_to_vec(
    "\
8466ba3e31a0c388460c43a6d9ffff6bb8c6d5c7a1a1f8de79ea9046a9de667200fb2b2943\
331f4d6f0d25dcddc0e43112d43193af610751ffa8531395703d7002020a8f152937f327e76\
54df0c0643ae3ad8779578872b773daecfc78c93bd0ce93482d4b39f2900b58cb812b191141\
fe3db202c44bc37688facc82c84d3272ba2c5227dbf24227bb92c0ac487a09f37f3226981c\
b4f2a9d6ca3c7a6f522611e77f679e59daf62d65d4c03def1001f8cd4e617efa1bc43ebe6\
addc98f3b49437df55aad5e96a6b7db196f9d30de7173dda79944f51fc7a339655cd2727d47\
4630a322103c344bc4c65add2214f60155b3819869210f19730544989fed6921bf",
  );

  assert!(
    key
      .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, message, &sig256)
      .is_ok()
  );
  assert!(
    key
      .verify_pkcs1v15(RsaPkcs1v15Profile::Sha384, message, &sig384)
      .is_ok()
  );
  assert!(
    key
      .verify_pkcs1v15(RsaPkcs1v15Profile::Sha512, message, &sig512)
      .is_ok()
  );
}

#[test]
fn pkcs1v15_verify_rejects_tampered_signature_message_and_profile() {
  let key = RsaPublicKey::from_spki_der(&pkcs1v15_fixture_public_key()).unwrap();
  let message = pkcs1v15_fixture_message();
  let mut sig = hex_to_vec(
    "\
94781246d705f79659d01ac8894b6f41076abe165e28711ec8fa41c1c8767b175a9c63e5118b\
d30de86da0d7b8934e963ef69c438ace976e4453dfce6b9b84a7d37a27ee61512656333dfda1\
ac40197fe4f9396bb016b25054f98f149d126c0248fc007cddc3d75d178eb34ecda0e0df822\
825ca133c062d3cdcb19e20a3e377541d8253af795a9b49a41ddb5795592502b9efbb153afc\
dd4fcc492a891d8536ef91cc228a3dbf66f0c70596f9cd101fe95d127550e7a4a9864430bd3\
4a88d8df93f4df7b54e8a4b8643891481e4bdcf87be3f98a1fdc475a819e3dc3a114aff86e\
48929a430fc39333f81064701be7d5501a3a7b4ec6c68f6feda6190d66b16",
  );

  assert!(key.verify_pkcs1v15(RsaPkcs1v15Profile::Sha384, message, &sig).is_err());
  assert!(
    key
      .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, b"wrong message", &sig)
      .is_err()
  );

  sig[31] ^= 0x40;
  assert!(key.verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, message, &sig).is_err());
}

#[cfg(feature = "diag")]
#[test]
fn pss_encoded_message_oracle_failures_are_opaque() {
  let key = RsaPublicKey::from_spki_der(&pss_fixture_public_key()).unwrap();
  let mut encoded = vec![0u8; key.modulus().len()];
  key
    .public_operation(&pss_fixture_signature_sha256(), &mut encoded)
    .unwrap();
  let em_bits = key.modulus_bits().strict_sub(1);

  assert!(diag_rsa_verify_pss_encoded(RsaPssProfile::Sha256, pss_fixture_message(), &encoded, em_bits).is_ok());

  let mut bad_trailer = encoded.clone();
  *bad_trailer.last_mut().unwrap() ^= 0x01;
  assert_opaque_verification_failure(diag_rsa_verify_pss_encoded(
    RsaPssProfile::Sha256,
    pss_fixture_message(),
    &bad_trailer,
    em_bits,
  ));

  let mut bad_unused_bit = encoded.clone();
  bad_unused_bit[0] |= 0x80;
  assert_opaque_verification_failure(diag_rsa_verify_pss_encoded(
    RsaPssProfile::Sha256,
    pss_fixture_message(),
    &bad_unused_bit,
    em_bits,
  ));

  let mut bad_masked_db = encoded.clone();
  bad_masked_db[17] ^= 0x40;
  assert_opaque_verification_failure(diag_rsa_verify_pss_encoded(
    RsaPssProfile::Sha256,
    pss_fixture_message(),
    &bad_masked_db,
    em_bits,
  ));

  let h_offset = encoded
    .len()
    .strict_sub(RsaPssProfile::Sha256.digest_len())
    .strict_sub(1);
  let mut bad_hash = encoded.clone();
  bad_hash[h_offset] ^= 0x20;
  assert_opaque_verification_failure(diag_rsa_verify_pss_encoded(
    RsaPssProfile::Sha256,
    pss_fixture_message(),
    &bad_hash,
    em_bits,
  ));

  let mut short_db = vec![0u8; 8];
  let mut short_mask = vec![0u8; 8];
  assert_opaque_verification_failure(diag_rsa_verify_pss_encoded_with_scratch(
    RsaPssProfile::Sha256,
    pss_fixture_message(),
    &encoded,
    em_bits,
    &mut short_db,
    &mut short_mask,
  ));

  assert_opaque_verification_failure(diag_rsa_verify_pss_encoded(
    RsaPssProfile::Sha256,
    b"wrong message",
    &encoded,
    em_bits,
  ));
  assert_opaque_verification_failure(diag_rsa_verify_pss_encoded(
    RsaPssProfile::Sha256,
    pss_fixture_message(),
    &[0xbc],
    8,
  ));
}

#[cfg(feature = "diag")]
#[test]
fn pkcs1v15_encoded_message_oracle_failures_are_opaque() {
  let key = RsaPublicKey::from_spki_der(&pkcs1v15_fixture_public_key()).unwrap();
  let mut encoded = vec![0u8; key.modulus().len()];
  key
    .public_operation(&pkcs1v15_fixture_signature_sha256(), &mut encoded)
    .unwrap();

  assert!(diag_rsa_verify_pkcs1v15_encoded(RsaPkcs1v15Profile::Sha256, pkcs1v15_fixture_message(), &encoded).is_ok());

  let mut bad_prefix = encoded.clone();
  bad_prefix[0] = 0x01;
  assert_opaque_verification_failure(diag_rsa_verify_pkcs1v15_encoded(
    RsaPkcs1v15Profile::Sha256,
    pkcs1v15_fixture_message(),
    &bad_prefix,
  ));

  let mut bad_block_type = encoded.clone();
  bad_block_type[1] = 0x02;
  assert_opaque_verification_failure(diag_rsa_verify_pkcs1v15_encoded(
    RsaPkcs1v15Profile::Sha256,
    pkcs1v15_fixture_message(),
    &bad_block_type,
  ));

  let mut bad_padding = encoded.clone();
  bad_padding[2] = 0x00;
  assert_opaque_verification_failure(diag_rsa_verify_pkcs1v15_encoded(
    RsaPkcs1v15Profile::Sha256,
    pkcs1v15_fixture_message(),
    &bad_padding,
  ));

  let digest_info_len = 19usize.strict_add(32);
  let separator_index = encoded.len().strict_sub(digest_info_len).strict_sub(1);
  let mut bad_separator = encoded.clone();
  bad_separator[separator_index] = 0xff;
  assert_opaque_verification_failure(diag_rsa_verify_pkcs1v15_encoded(
    RsaPkcs1v15Profile::Sha256,
    pkcs1v15_fixture_message(),
    &bad_separator,
  ));

  let mut bad_digest_info = encoded.clone();
  bad_digest_info[separator_index.strict_add(1)] ^= 0x01;
  assert_opaque_verification_failure(diag_rsa_verify_pkcs1v15_encoded(
    RsaPkcs1v15Profile::Sha256,
    pkcs1v15_fixture_message(),
    &bad_digest_info,
  ));

  let mut bad_digest = encoded.clone();
  *bad_digest.last_mut().unwrap() ^= 0x80;
  assert_opaque_verification_failure(diag_rsa_verify_pkcs1v15_encoded(
    RsaPkcs1v15Profile::Sha256,
    pkcs1v15_fixture_message(),
    &bad_digest,
  ));

  assert_opaque_verification_failure(diag_rsa_verify_pkcs1v15_encoded(
    RsaPkcs1v15Profile::Sha256,
    b"wrong message",
    &encoded,
  ));
  assert_opaque_verification_failure(diag_rsa_verify_pkcs1v15_encoded(
    RsaPkcs1v15Profile::Sha256,
    pkcs1v15_fixture_message(),
    &[0, 1, 0xff],
  ));
}

#[test]
fn sha256_signature_verification_matches_external_oracles_for_valid_and_tampered_vectors() {
  let pss_spki = pss_fixture_public_key();
  let pss_sig = pss_fixture_signature_sha256();
  let pss_key = RsaPublicKey::from_spki_der(&pss_spki).unwrap();
  let pss_pkcs1 =
    valid_pkcs1_with_modulus_and_exponent(pss_key.modulus(), &exponent_bytes(pss_key.public_exponent().as_u64()));

  assert!(
    pss_key
      .verify_pss(RsaPssProfile::Sha256, pss_fixture_message(), &pss_sig)
      .is_ok()
  );
  assert_ring_pss_sha256(&pss_pkcs1, pss_fixture_message(), &pss_sig, true);
  assert_aws_lc_rs_pss_sha256(&pss_pkcs1, pss_fixture_message(), &pss_sig, true);
  assert_rustcrypto_pss_sha256(&pss_pkcs1, pss_fixture_message(), &pss_sig, true);
  assert_openssl_sha256(
    &pss_spki,
    pss_fixture_message(),
    &pss_sig,
    &["rsa_padding_mode:pss", "rsa_pss_saltlen:32", "rsa_mgf1_md:sha256"],
    true,
  );

  let mut tampered = pss_sig.clone();
  tampered[0] ^= 0x80;
  assert!(
    pss_key
      .verify_pss(RsaPssProfile::Sha256, pss_fixture_message(), &tampered)
      .is_err()
  );
  assert_ring_pss_sha256(&pss_pkcs1, pss_fixture_message(), &tampered, false);
  assert_aws_lc_rs_pss_sha256(&pss_pkcs1, pss_fixture_message(), &tampered, false);
  assert_rustcrypto_pss_sha256(&pss_pkcs1, pss_fixture_message(), &tampered, false);
  assert_openssl_sha256(
    &pss_spki,
    pss_fixture_message(),
    &tampered,
    &["rsa_padding_mode:pss", "rsa_pss_saltlen:32", "rsa_mgf1_md:sha256"],
    false,
  );

  let pkcs1_spki = pkcs1v15_fixture_public_key();
  let pkcs1_sig = pkcs1v15_fixture_signature_sha256();
  let pkcs1_key = RsaPublicKey::from_spki_der(&pkcs1_spki).unwrap();
  let pkcs1_der = valid_pkcs1_with_modulus_and_exponent(
    pkcs1_key.modulus(),
    &exponent_bytes(pkcs1_key.public_exponent().as_u64()),
  );

  assert!(
    pkcs1_key
      .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, pkcs1v15_fixture_message(), &pkcs1_sig)
      .is_ok()
  );
  assert_ring_pkcs1v15_sha256(&pkcs1_der, pkcs1v15_fixture_message(), &pkcs1_sig, true);
  assert_aws_lc_rs_pkcs1v15_sha256(&pkcs1_der, pkcs1v15_fixture_message(), &pkcs1_sig, true);
  assert_rustcrypto_pkcs1v15_sha256(&pkcs1_der, pkcs1v15_fixture_message(), &pkcs1_sig, true);
  assert_openssl_sha256(&pkcs1_spki, pkcs1v15_fixture_message(), &pkcs1_sig, &[], true);

  let mut tampered = pkcs1_sig.clone();
  tampered[0] ^= 0x80;
  assert!(
    pkcs1_key
      .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, pkcs1v15_fixture_message(), &tampered)
      .is_err()
  );
  assert_ring_pkcs1v15_sha256(&pkcs1_der, pkcs1v15_fixture_message(), &tampered, false);
  assert_aws_lc_rs_pkcs1v15_sha256(&pkcs1_der, pkcs1v15_fixture_message(), &tampered, false);
  assert_rustcrypto_pkcs1v15_sha256(&pkcs1_der, pkcs1v15_fixture_message(), &tampered, false);
  assert_openssl_sha256(&pkcs1_spki, pkcs1v15_fixture_message(), &tampered, &[], false);
}

#[test]
fn nist_cavp_sha2_signatures_match_rust_external_oracles() {
  let suite: serde_json::Value = serde_json::from_str(CAVP_SIGVER_186_3).expect("CAVP JSON must parse");
  let tests = suite["tests"].as_array().expect("CAVP tests must be an array");
  let policy = RsaPublicKeyPolicy::legacy_verification().allow_legacy_odd_exponents();

  for test in tests {
    let scheme = cavp_field(test, "scheme");
    let sha = cavp_field(test, "sha");
    let salt_len =
      (scheme == "pss").then(|| test["salt_len"].as_u64().expect("CAVP PSS salt length must be numeric") as usize);
    let expected = match cavp_field(test, "result") {
      "P" => true,
      "F" => false,
      other => panic!("unsupported CAVP result `{other}`"),
    };

    let pkcs1 = valid_pkcs1_with_modulus_and_exponent(
      &cavp_hex_to_vec(cavp_field(test, "n")),
      &cavp_hex_to_vec(cavp_field(test, "e")),
    );
    let key = RsaPublicKey::from_pkcs1_der_with_policy(&pkcs1, &policy).expect("CAVP RSA key must parse");
    let message = cavp_hex_to_vec(cavp_field(test, "msg"));
    let signature = cavp_hex_to_vec(cavp_field(test, "sig"));

    assert_eq!(
      cavp_rscrypto_result(&key, scheme, sha, salt_len, &message, &signature).is_ok(),
      expected,
      "rscrypto mismatch for CAVP tcId {}",
      test["tc_id"]
    );
    if scheme == "pkcs1v15" || salt_len == Some(digest_salt_len(sha)) {
      assert_ring_cavp(scheme, sha, &pkcs1, &message, &signature, expected);
      assert_aws_lc_rs_cavp(scheme, sha, &pkcs1, &message, &signature, expected);
    }
    assert_aws_lc_sys_cavp(scheme, sha, salt_len, &pkcs1, &message, &signature, expected);
    assert_rustcrypto_cavp(scheme, sha, salt_len, &pkcs1, &message, &signature, expected);
  }

  assert_eq!(tests.len(), 216);
}

#[test]
fn nist_cavp_sha2_signatures_match_openssl_cli_when_available() {
  let suite: serde_json::Value = serde_json::from_str(CAVP_SIGVER_186_3).expect("CAVP JSON must parse");
  let tests = suite["tests"].as_array().expect("CAVP tests must be an array");
  let mut checked = 0usize;

  for test in tests {
    let scheme = cavp_field(test, "scheme");
    let sha = cavp_field(test, "sha");
    let salt_len =
      (scheme == "pss").then(|| test["salt_len"].as_u64().expect("CAVP PSS salt length must be numeric") as usize);
    let expected = match cavp_field(test, "result") {
      "P" => true,
      "F" => false,
      other => panic!("unsupported CAVP result `{other}`"),
    };
    let pkcs1 = valid_pkcs1_with_modulus_and_exponent(
      &cavp_hex_to_vec(cavp_field(test, "n")),
      &cavp_hex_to_vec(cavp_field(test, "e")),
    );
    let spki = spki_for_pkcs1(&pkcs1);
    let message = cavp_hex_to_vec(cavp_field(test, "msg"));
    let signature = cavp_hex_to_vec(cavp_field(test, "sig"));
    let digest_arg = match sha {
      "SHA256" => "-sha256",
      "SHA384" => "-sha384",
      "SHA512" => "-sha512",
      other => panic!("unsupported OpenSSL CAVP hash `{other}`"),
    };
    let pss_sigopts;
    let sigopts = if scheme == "pss" {
      pss_sigopts = openssl_pss_sigopts(sha, salt_len.unwrap());
      &pss_sigopts[..]
    } else {
      &[]
    };
    let Some(actual) = openssl_verify(digest_arg, &spki, &message, &signature, sigopts) else {
      eprintln!("skipping OpenSSL CAVP RSA differential check because `openssl` is not available");
      return;
    };

    assert_eq!(
      actual, expected,
      "OpenSSL mismatch for CAVP tcId {} {scheme}/{sha}",
      test["tc_id"]
    );
    checked = checked.strict_add(1);
  }

  assert_eq!(checked, 216);
}

#[test]
fn nist_cavp_sha2_signatures_match_boringssl_helper_when_available() {
  let suite: serde_json::Value = serde_json::from_str(CAVP_SIGVER_186_3).expect("CAVP JSON must parse");
  let tests = suite["tests"].as_array().expect("CAVP tests must be an array");
  let mut checked = 0usize;

  for test in tests {
    let scheme = cavp_field(test, "scheme");
    let sha = cavp_field(test, "sha");
    let salt_len =
      (scheme == "pss").then(|| test["salt_len"].as_u64().expect("CAVP PSS salt length must be numeric") as usize);
    let expected = match cavp_field(test, "result") {
      "P" => true,
      "F" => false,
      other => panic!("unsupported CAVP result `{other}`"),
    };
    let pkcs1 = valid_pkcs1_with_modulus_and_exponent(
      &cavp_hex_to_vec(cavp_field(test, "n")),
      &cavp_hex_to_vec(cavp_field(test, "e")),
    );
    let spki = spki_for_pkcs1(&pkcs1);
    let message = cavp_hex_to_vec(cavp_field(test, "msg"));
    let signature = cavp_hex_to_vec(cavp_field(test, "sig"));
    let Some(actual) = boringssl_verify(sha, scheme, salt_len, &spki, &message, &signature) else {
      eprintln!("skipping BoringSSL CAVP RSA differential check because BORINGSSL_RSA_VERIFY_HELPER is not set");
      return;
    };

    assert_eq!(
      actual, expected,
      "BoringSSL mismatch for CAVP tcId {} {scheme}/{sha}",
      test["tc_id"]
    );
    checked = checked.strict_add(1);
  }

  assert_eq!(checked, 216);
}

#[test]
fn generated_rsa_size_fixtures_verify_for_benchmark_matrix() {
  for (bits, spki, pss_sig, pkcs1_sig) in [
    (3072, RSA3072_SPKI, RSA3072_PSS_SHA256, RSA3072_PKCS1V15_SHA256),
    (4096, RSA4096_SPKI, RSA4096_PSS_SHA256, RSA4096_PKCS1V15_SHA256),
    (8192, RSA8192_SPKI, RSA8192_PSS_SHA256, RSA8192_PKCS1V15_SHA256),
  ] {
    let key = RsaPublicKey::from_spki_der(spki).unwrap();

    assert_eq!(key.modulus_bits(), bits);
    assert!(
      key
        .verify_pss(RsaPssProfile::Sha256, pss_fixture_message(), pss_sig)
        .is_ok()
    );
    assert!(
      key
        .verify_pkcs1v15(RsaPkcs1v15Profile::Sha256, pkcs1v15_fixture_message(), pkcs1_sig)
        .is_ok()
    );

    let pkcs1_der =
      valid_pkcs1_with_modulus_and_exponent(key.modulus(), &exponent_bytes(key.public_exponent().as_u64()));
    assert_ring_pss_sha256(&pkcs1_der, pss_fixture_message(), pss_sig, true);
    assert_aws_lc_rs_pss_sha256(&pkcs1_der, pss_fixture_message(), pss_sig, true);
    if bits <= 4096 {
      assert_rustcrypto_spki_pss_sha256(spki, pss_fixture_message(), pss_sig, true);
    }
    assert_openssl_sha256(
      spki,
      pss_fixture_message(),
      pss_sig,
      &["rsa_padding_mode:pss", "rsa_pss_saltlen:32", "rsa_mgf1_md:sha256"],
      true,
    );

    assert_ring_pkcs1v15_sha256(&pkcs1_der, pkcs1v15_fixture_message(), pkcs1_sig, true);
    assert_aws_lc_rs_pkcs1v15_sha256(&pkcs1_der, pkcs1v15_fixture_message(), pkcs1_sig, true);
    if bits <= 4096 {
      assert_rustcrypto_spki_pkcs1v15_sha256(spki, pkcs1v15_fixture_message(), pkcs1_sig, true);
    }
    assert_openssl_sha256(spki, pkcs1v15_fixture_message(), pkcs1_sig, &[], true);
  }
}

proptest! {
  #[test]
  fn arbitrary_der_inputs_do_not_panic(input in proptest::collection::vec(any::<u8>(), 0..4096)) {
    let _ = RsaPublicKey::from_pkcs1_der(&input);
    let _ = RsaPublicKey::from_spki_der(&input);
  }

  #[test]
  fn arbitrary_signature_lengths_do_not_panic(
    signature in proptest::collection::vec(any::<u8>(), 0..520),
    message in proptest::collection::vec(any::<u8>(), 0..128),
  ) {
    let key = arbitrary_verify_key();
    let mut scratch = key.public_scratch();

    let _ = key.verify_pss_with_scratch(RsaPssProfile::Sha256, &message, &signature, &mut scratch);
    let _ = key.verify_pss_with_scratch(RsaPssProfile::Sha384, &message, &signature, &mut scratch);
    let _ = key.verify_pss_with_scratch(RsaPssProfile::Sha512, &message, &signature, &mut scratch);
    let _ = key.verify_pkcs1v15_with_scratch(RsaPkcs1v15Profile::Sha256, &message, &signature, &mut scratch);
    let _ = key.verify_pkcs1v15_with_scratch(RsaPkcs1v15Profile::Sha384, &message, &signature, &mut scratch);
    let _ = key.verify_pkcs1v15_with_scratch(RsaPkcs1v15Profile::Sha512, &message, &signature, &mut scratch);
  }

  #[test]
  fn arbitrary_fixed_width_signatures_do_not_panic(
    material in proptest::collection::vec(any::<u8>(), 0..128),
    message in proptest::collection::vec(any::<u8>(), 0..128),
  ) {
    let key = arbitrary_verify_key();
    let signature = fixed_width_signature_candidate(&material, key.modulus().len());
    let mut scratch = key.public_scratch();

    let _ = key.verify_pss_with_scratch(RsaPssProfile::Sha256, &message, &signature, &mut scratch);
    let _ = key.verify_pss_with_scratch(RsaPssProfile::Sha384, &message, &signature, &mut scratch);
    let _ = key.verify_pss_with_scratch(RsaPssProfile::Sha512, &message, &signature, &mut scratch);
    let _ = key.verify_pkcs1v15_with_scratch(RsaPkcs1v15Profile::Sha256, &message, &signature, &mut scratch);
    let _ = key.verify_pkcs1v15_with_scratch(RsaPkcs1v15Profile::Sha384, &message, &signature, &mut scratch);
    let _ = key.verify_pkcs1v15_with_scratch(RsaPkcs1v15Profile::Sha512, &message, &signature, &mut scratch);
  }
}
