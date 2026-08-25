//! RSA verification benchmarks for rscrypto public APIs.

use core::hint::black_box;

#[cfg(all(
  any(unix, windows),
  not(target_arch = "wasm32"),
  not(any(target_arch = "s390x", target_arch = "powerpc64"))
))]
use aws_lc_rs::signature as aws_signature;
use criterion::{Criterion, criterion_group, criterion_main};
use ring::signature as ring_signature;
#[cfg(feature = "getrandom")]
use rsa::{
  RsaPrivateKey as RustCryptoRsaPrivateKey,
  pkcs1::DecodeRsaPrivateKey,
  pss::BlindedSigningKey as RustCryptoBlindedPssSigningKey,
  rand_core::{CryptoRng as RustCryptoCryptoRng, Error as RustCryptoRngError, RngCore as RustCryptoRngCore},
  signature::RandomizedSigner as _,
};
use rsa::{
  RsaPublicKey as RustCryptoRsaPublicKey,
  pkcs1v15::{Signature as RustCryptoPkcs1v15Signature, VerifyingKey as RustCryptoPkcs1v15VerifyingKey},
  pkcs8::DecodePublicKey,
  pss::{Signature as RustCryptoPssSignature, VerifyingKey as RustCryptoPssVerifyingKey},
  signature::Verifier as _,
};
#[cfg(feature = "diag")]
use rscrypto::auth::rsa::{
  diag_rsa_blinding_factor_inverse_with_scratch, diag_rsa_public_operation_bitserial, diag_rsa_public_operation_cios,
  diag_rsa_public_operation_cios_portable, diag_rsa_public_operation_generic_exponent,
  diag_rsa_public_operation_product, diag_rsa_verify_pkcs1v15_encoded, diag_rsa_verify_pss_encoded_with_scratch,
};
use rscrypto::{
  Digest, RsaBlindingPair, RsaPkcs1v15Profile, RsaPrivateKey, RsaPrivateKeyParts, RsaPssProfile, RsaPublicKey,
  RsaPublicKeyPolicy, Sha256, Sha384, Sha512,
};

#[cfg(all(
  any(unix, windows),
  not(target_arch = "wasm32"),
  not(any(target_arch = "s390x", target_arch = "powerpc64"))
))]
macro_rules! aws_lc_bench {
  ($($tokens:tt)*) => {
    $($tokens)*
  };
}

#[cfg(not(all(
  any(unix, windows),
  not(target_arch = "wasm32"),
  not(any(target_arch = "s390x", target_arch = "powerpc64"))
)))]
macro_rules! aws_lc_bench {
  ($($tokens:tt)*) => {};
}

const MESSAGE_PSS: &[u8] = b"rscrypto RSA-PSS verification fixture";
const MESSAGE_PKCS1V15: &[u8] = b"rscrypto RSA-PKCS1-v1_5 verification fixture";

const RSA3072_SPKI: &[u8] = include_bytes!("rsa_fixtures/rsa3072_spki.der");
const RSA3072_PSS_SHA256: &[u8] = include_bytes!("rsa_fixtures/rsa3072_pss_sha256.sig");
const RSA3072_PKCS1V15_SHA256: &[u8] = include_bytes!("rsa_fixtures/rsa3072_pkcs1v15_sha256.sig");
const RSA4096_SPKI: &[u8] = include_bytes!("rsa_fixtures/rsa4096_spki.der");
const RSA4096_PSS_SHA256: &[u8] = include_bytes!("rsa_fixtures/rsa4096_pss_sha256.sig");
const RSA4096_PKCS1V15_SHA256: &[u8] = include_bytes!("rsa_fixtures/rsa4096_pkcs1v15_sha256.sig");
const RSA8192_SPKI: &[u8] = include_bytes!("rsa_fixtures/rsa8192_spki.der");
const RSA8192_PSS_SHA256: &[u8] = include_bytes!("rsa_fixtures/rsa8192_pss_sha256.sig");
const RSA8192_PKCS1V15_SHA256: &[u8] = include_bytes!("rsa_fixtures/rsa8192_pkcs1v15_sha256.sig");

const PRIVATE_SIGNING_MESSAGE: &[u8] = b"rscrypto TLS-shaped RSA private signing benchmark";

#[cfg(feature = "getrandom")]
struct GetrandomRng;

#[cfg(feature = "getrandom")]
impl RustCryptoRngCore for GetrandomRng {
  fn next_u32(&mut self) -> u32 {
    rsa::rand_core::impls::next_u32_via_fill(self)
  }

  fn next_u64(&mut self) -> u64 {
    rsa::rand_core::impls::next_u64_via_fill(self)
  }

  fn fill_bytes(&mut self, dest: &mut [u8]) {
    self
      .try_fill_bytes(dest)
      .expect("OS entropy must remain available during the RSA benchmark")
  }

  fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), RustCryptoRngError> {
    getrandom::fill(dest).map_err(RustCryptoRngError::new)
  }
}

#[cfg(feature = "getrandom")]
impl RustCryptoCryptoRng for GetrandomRng {}

fn hex_to_vec(hex: &str) -> Vec<u8> {
  let mut out = Vec::with_capacity(hex.len() / 2);
  for chunk in hex.as_bytes().as_chunks::<2>().0 {
    let high = hex_value(chunk[0]).expect("RSA benchmark fixture must contain hexadecimal digits");
    let low = hex_value(chunk[1]).expect("RSA benchmark fixture must contain hexadecimal digits");
    out.push((high << 4) | low);
  }
  out
}

fn hex_value(byte: u8) -> Option<u8> {
  match byte {
    b'0'..=b'9' => Some(byte.strict_sub(b'0')),
    b'a'..=b'f' => Some(byte.strict_sub(b'a').strict_add(10)),
    b'A'..=b'F' => Some(byte.strict_sub(b'A').strict_add(10)),
    _ => None,
  }
}

fn der_len(len: usize) -> Vec<u8> {
  if len < 128 {
    return vec![u8::try_from(len).expect("short DER length must fit in one byte")];
  }

  let bytes = len.to_be_bytes();
  let first_nonzero = bytes
    .iter()
    .position(|&byte| byte != 0)
    .expect("long DER length must contain a non-zero byte");
  let len_bytes = &bytes[first_nonzero..];
  let mut out = Vec::with_capacity(1usize.strict_add(len_bytes.len()));
  out.push(0x80 | u8::try_from(len_bytes.len()).expect("DER length-of-length must fit in one byte"));
  out.extend_from_slice(len_bytes);
  out
}

fn tlv(tag: u8, value: &[u8]) -> Vec<u8> {
  let encoded_len = der_len(value.len());
  let capacity = 1usize.strict_add(encoded_len.len()).strict_add(value.len());
  let mut out = Vec::with_capacity(capacity);
  out.push(tag);
  out.extend_from_slice(&encoded_len);
  out.extend_from_slice(value);
  out
}

fn sequence(value: &[u8]) -> Vec<u8> {
  tlv(0x30, value)
}

fn integer_unsigned(value: &[u8]) -> Vec<u8> {
  let first_nonzero = value.iter().position(|&byte| byte != 0);
  let value = first_nonzero.map_or(&[0u8][..], |index| &value[index..]);
  let mut encoded = Vec::with_capacity(value.len().strict_add(usize::from(value[0] & 0x80 != 0)));
  if value[0] & 0x80 != 0 {
    encoded.push(0);
  }
  encoded.extend_from_slice(value);
  tlv(0x02, &encoded)
}

fn exponent_bytes(exponent: u64) -> Vec<u8> {
  let bytes = exponent.to_be_bytes();
  let first_nonzero = bytes
    .iter()
    .position(|&byte| byte != 0)
    .unwrap_or_else(|| bytes.len().strict_sub(1));
  bytes[first_nonzero..].to_vec()
}

fn pkcs1_der_from_key(key: &RsaPublicKey) -> Vec<u8> {
  let mut body = Vec::new();
  body.extend_from_slice(&integer_unsigned(key.modulus()));
  body.extend_from_slice(&integer_unsigned(&exponent_bytes(key.public_exponent().as_u64())));
  sequence(&body)
}

fn pkcs1_der_from_modulus_exponent(modulus: &[u8], exponent: &[u8]) -> Vec<u8> {
  let mut body = Vec::new();
  body.extend_from_slice(&integer_unsigned(modulus));
  body.extend_from_slice(&integer_unsigned(exponent));
  sequence(&body)
}

#[cfg(feature = "diag")]
fn synthetic_pkcs1_der(modulus_len: usize) -> Vec<u8> {
  let mut modulus = vec![0xff; modulus_len];
  modulus[0] = 0x80;
  pkcs1_der_from_modulus_exponent(&modulus, &[0x01, 0x00, 0x01])
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

fn legacy_exponent_modulus() -> Vec<u8> {
  hex_to_vec(
    "\
ef8bb02b8e4aec1abc6fac7a0d6fb1f2649bb86a1567423fee4a194a250461a9db702558e92e52cc\
907963d84731a7adaf4c609e1b7c7d7c187099a43857f7628f5d20416fcb48987c9d6f12cfc6bc\
260c9b5506be3fe3cd218ddb37ef5b30feb16172a9832312726ed135c0540ef9d3229b87b5566f\
3355c90f301b856aa822878269806079ab7267cdc6c7403d7be3fa652065b2d39f2dbf9fb61ed9\
71fee37432ebe31d9aa465dbae96b0edd5ffddf1b49e03346a02290fed1e4e31f6b3b6e1f839f\
d5add90a8a212c10dd997b0a4efcb3df990808509dcb28c504e0649827a83ffd864395d1f62f2\
9a004f44423a44b07de943a60fba844a9da3603ce5c5",
  )
}

fn legacy_exponent_input() -> Vec<u8> {
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

fn rsa2048_private_key() -> RsaPrivateKey {
  let modulus = hex_to_vec(
    "d397b84d98a4c26138ed1b695a8106ead91d553bf06041b62d3fdc50a041e222b8f4529689c1b82c5e71554f5dd69fa2f4b6158cf0dbeb57811a0fc327e1f28e74fe74d3bc166c1eabdc1b8b57b934ca8be5b00b4f29975bcc99acaf415b59bb28a6782bb41a2c3c2976b3c18dbadef62f00c6bb226640095096c0cc60d22fe7ef987d75c6a81b10d96bf292028af110dc7cc1bbc43d22adab379a0cd5d8078cc780ff5cd6209dea34c922cf784f7717e428d75b5aec8ff30e5f0141510766e2e0ab8d473c84e8710b2b98227c3db095337ad3452f19e2b9bfbccdd8148abf6776fa552775e6e75956e45229ae5a9c46949bab1e622f0e48f56524a84ed3483b",
  );
  let private_exponent = hex_to_vec(
    "c4e70c689162c94c660828191b52b4d8392115df486a9adbe831e458d73958320dc1b755456e93701e9702d76fb0b92f90e01d1fe248153281fe79aa9763a92fae69d8d7ecd144de29fa135bd14f9573e349e45031e3b76982f583003826c552e89a397c1a06bd2163488630d92e8c2bb643d7abef700da95d685c941489a46f54b5316f62b5d2c3a7f1bbd134cb37353a44683fdc9d95d36458de22f6c44057fe74a0a436c4308f73f4da42f35c47ac16a7138d483afc91e41dc3a1127382e0c0f5119b0221b4fc639d6b9c38177a6de9b526ebd88c38d7982c07f98a0efd877d508aae275b946915c02e2e1106d175d74ec6777f5e80d12c053d9c7be1e341",
  );
  let prime_p = hex_to_vec(
    "f827bbf3a41877c7cc59aebf42ed4b29c32defcb8ed96863d5b090a05a8930dd624a21c9dcf9838568fdfa0df65b8462a5f2ac913d6c56f975532bd8e78fb07bd405ca99a484bcf59f019bbddcb3933f2bce706300b4f7b110120c5df9018159067c35da3061a56c8635a52b54273b31271b4311f0795df6021e6355e1a42e61",
  );
  let prime_q = hex_to_vec(
    "da4817ce0089dd36f2ade6a3ff410c73ec34bf1b4f6bda38431bfede11cef1f7f6efa70e5f8063a3b1f6e17296ffb15feefa0912a0325b8d1fd65a559e717b5b961ec345072e0ec5203d03441d29af4d64054a04507410cf1da78e7b6119d909ec66e6ad625bf995b279a4b3c5be7d895cd7c5b9c4c497fde730916fcdb4e41b",
  );
  let exponent_p = hex_to_vec(
    "1da6e9cf80212856e87522eb59bcef094b7836ba1514a7639e8a1d8dfba37f0245176498315e6337d2c6de5542c5c6b8dee973735b6a91adf735fbfc4c1720587b8a419e40495826e55c14d70803312a103af7b4ecc5b2ff265371c4dcd730348a10d7827ddb7d1fcd9da561db09610a4b88f767b25b5e3de21ced73baa59aa1",
  );
  let exponent_q = hex_to_vec(
    "d737a7c8e43d0a10c85bf0011886a16996a6371b0d46b0c5325de3003f9cc47491539f6a0b7d82407f12851cbf86e1f34da3d7d8367d104967efa7e7ad2e04cbbb8b1f4aeb165d57bd3e8afed8a62602ef304bd74f1ff106d51d44dd9f52a5ed23da1d6d2c82b4e6052fecd5978e0726ad94cd8e295510eb35cc6c49491026ab",
  );
  let coefficient = hex_to_vec(
    "5268d7cf073479aebb2d2ed4dd66b8c89915b52d141e0c4932f56b0c0ed0936141894ec4d27d53bc86453cd8ca5b455045218c7e196209c1c651702ece090a15e3cbcc265971300023a86fe9d34ad527e9ef03b7adfe736e0680747abfd49839b82f2ffdec43bd0343ca30e13961b32af6cdeddd195672c76b53b76fc3ea76f8",
  );

  RsaPrivateKey::from_components_with_policy(
    RsaPrivateKeyParts {
      modulus: &modulus,
      public_exponent: 65_537,
      private_exponent: &private_exponent,
      prime_p: &prime_p,
      prime_q: &prime_q,
      exponent_p: &exponent_p,
      exponent_q: &exponent_q,
      coefficient: &coefficient,
    },
    &RsaPublicKeyPolicy::legacy_verification(),
  )
  .expect("valid RSA private benchmark fixture must succeed")
}

fn factor_two_and_inverse(modulus: &[u8]) -> (Vec<u8>, Vec<u8>) {
  let mut factor = vec![0u8; modulus.len()];
  factor[modulus.len().strict_sub(1)] = 2;

  let mut inverse = vec![0u8; modulus.len()];
  let mut carry = 0u8;
  for (dst, &byte) in inverse.iter_mut().zip(modulus) {
    *dst = (byte >> 1) | carry;
    carry = (byte & 1) << 7;
  }
  for byte in inverse.iter_mut().rev() {
    let (sum, overflow) = byte.overflowing_add(1);
    *byte = sum;
    if !overflow {
      break;
    }
  }

  (factor, inverse)
}

fn rsa_private_signing(c: &mut Criterion) {
  let key = rsa2048_private_key();
  let mut scratch = key.private_scratch();
  let mut signature = vec![0u8; key.signature_len()];
  let salt = [0x5au8; Sha256::OUTPUT_SIZE];
  let (factor, inverse) = factor_two_and_inverse(key.public_key().modulus());
  #[cfg(feature = "getrandom")]
  let rustcrypto_signing_key = {
    let private_key_der = key.to_pkcs1_der();
    let mut private_key = RustCryptoRsaPrivateKey::from_pkcs1_der(&private_key_der)
      .expect("RustCrypto must import the RSA private benchmark fixture");
    private_key
      .precompute()
      .expect("RustCrypto must precompute the RSA private benchmark fixture");
    RustCryptoBlindedPssSigningKey::<rsa::sha2::Sha256>::new(private_key)
  };

  let mut group = c.benchmark_group("rsa-2048-private-signing");
  #[cfg(feature = "diag")]
  {
    let mut inverse_scratch = key.private_scratch();
    let mut computed_inverse = vec![0u8; key.signature_len()];
    group.bench_function("blinding-inverse-scratch-rscrypto", |b| {
      b.iter(|| {
        diag_rsa_blinding_factor_inverse_with_scratch(
          black_box(&key),
          black_box(&factor),
          black_box(&mut computed_inverse),
          black_box(&mut inverse_scratch),
        )
        .expect("valid scratch-backed RSA blinding-factor inversion must succeed")
      })
    });
  }
  group.bench_function("scratch-setup-rscrypto", |b| {
    b.iter(|| black_box(key.private_scratch()))
  });
  group.bench_function("sign-pss-sha256-fixed-entropy-scratch-rscrypto", |b| {
    b.iter(|| {
      key
        .sign_pss_with_salt_and_blinding_factor_and_scratch(
          RsaPssProfile::Sha256,
          black_box(PRIVATE_SIGNING_MESSAGE),
          black_box(&salt),
          RsaBlindingPair::new(black_box(&factor), black_box(&inverse)),
          black_box(&mut signature),
          black_box(&mut scratch),
        )
        .expect("valid scratch-backed RSA-PSS benchmark signing must succeed")
    })
  });
  group.bench_function("sign-pss-sha256-fixed-entropy-oneshot-rscrypto", |b| {
    b.iter(|| {
      key
        .sign_pss_with_salt_and_blinding_factor(
          RsaPssProfile::Sha256,
          black_box(PRIVATE_SIGNING_MESSAGE),
          black_box(&salt),
          RsaBlindingPair::new(black_box(&factor), black_box(&inverse)),
          black_box(&mut signature),
        )
        .expect("valid one-shot RSA-PSS benchmark signing must succeed")
    })
  });
  group.bench_function("sign-pss-sha256-caller-entropy-scratch-rscrypto", |b| {
    b.iter(|| {
      key
        .sign_pss_with_random_fill_and_scratch(
          RsaPssProfile::Sha256,
          black_box(PRIVATE_SIGNING_MESSAGE),
          black_box(&mut signature),
          black_box(&mut scratch),
          |out| {
            if out.len() == salt.len() {
              out.copy_from_slice(black_box(&salt));
            } else {
              out.copy_from_slice(black_box(&factor));
            }
            Ok::<(), ()>(())
          },
        )
        .expect("valid caller-random scratch-backed RSA-PSS benchmark signing must succeed")
    })
  });
  group.bench_function("sign-pss-sha256-caller-entropy-oneshot-rscrypto", |b| {
    b.iter(|| {
      key
        .sign_pss_with_random_fill(
          RsaPssProfile::Sha256,
          black_box(PRIVATE_SIGNING_MESSAGE),
          black_box(&mut signature),
          |out| {
            if out.len() == salt.len() {
              out.copy_from_slice(black_box(&salt));
            } else {
              out.copy_from_slice(black_box(&factor));
            }
            Ok::<(), ()>(())
          },
        )
        .expect("valid caller-random one-shot RSA-PSS benchmark signing must succeed")
    })
  });
  group.bench_function("sign-pkcs1v15-sha256-fixed-entropy-scratch-rscrypto", |b| {
    b.iter(|| {
      key
        .sign_pkcs1v15_with_blinding_factor_and_scratch(
          RsaPkcs1v15Profile::Sha256,
          black_box(PRIVATE_SIGNING_MESSAGE),
          RsaBlindingPair::new(black_box(&factor), black_box(&inverse)),
          black_box(&mut signature),
          black_box(&mut scratch),
        )
        .expect("valid scratch-backed RSA-PKCS1-v1_5 benchmark signing must succeed")
    })
  });
  #[cfg(feature = "getrandom")]
  group.bench_function("sign-pss-sha256-os-entropy-scratch-rscrypto", |b| {
    b.iter(|| {
      key
        .sign_pss_with_scratch(
          RsaPssProfile::Sha256,
          black_box(PRIVATE_SIGNING_MESSAGE),
          black_box(&mut signature),
          black_box(&mut scratch),
        )
        .expect("valid OS-random scratch-backed RSA-PSS benchmark signing must succeed")
    })
  });
  #[cfg(feature = "getrandom")]
  group.bench_function("sign-pss-sha256-os-entropy-oneshot-rscrypto", |b| {
    b.iter(|| {
      key
        .sign_pss(
          RsaPssProfile::Sha256,
          black_box(PRIVATE_SIGNING_MESSAGE),
          black_box(&mut signature),
        )
        .expect("valid OS-random one-shot RSA-PSS benchmark signing must succeed")
    })
  });
  #[cfg(feature = "getrandom")]
  group.bench_function("sign-pss-sha256-os-entropy-oneshot-blinded-rustcrypto", |b| {
    let mut rng = GetrandomRng;
    b.iter(|| {
      black_box(&rustcrypto_signing_key)
        .try_sign_with_rng(&mut rng, black_box(PRIVATE_SIGNING_MESSAGE))
        .expect("valid RustCrypto RSA-PSS benchmark signing must succeed")
    })
  });
  group.finish();
}

fn rsa_components_for_size(
  c: &mut Criterion,
  name: &str,
  pss_spki: &[u8],
  pss_sig: &[u8],
  pkcs1_spki: &[u8],
  pkcs1_sig: &[u8],
  import_policy: &RsaPublicKeyPolicy,
) {
  let pss_key =
    RsaPublicKey::from_spki_der_with_policy(pss_spki, import_policy).expect("valid RSA benchmark fixture must succeed");
  let mut pss_scratch = pss_key.public_scratch();
  let pkcs1_key = RsaPublicKey::from_spki_der_with_policy(pkcs1_spki, import_policy)
    .expect("valid RSA benchmark fixture must succeed");
  let mut pkcs1_scratch = pkcs1_key.public_scratch();
  let pss_pkcs1 = pkcs1_der_from_key(&pss_key);
  let pkcs1_pkcs1 = pkcs1_der_from_key(&pkcs1_key);
  let representative = modulus_minus_one(&pss_key);
  let mut out = vec![0u8; pss_key.modulus().len()];

  #[cfg(feature = "diag")]
  let (pss_encoded, pss_em_bits, mut pss_db, mut pss_db_mask, pkcs1_encoded) = {
    let mut pss_encoded = vec![0u8; pss_key.modulus().len()];
    pss_key
      .public_operation_with_scratch(pss_sig, &mut pss_encoded, &mut pss_scratch)
      .expect("valid RSA benchmark fixture must succeed");
    let pss_em_bits = pss_key.modulus_bits().strict_sub(1);
    let pss_em_len = pss_em_bits.strict_add(7) / 8;
    let leading = pss_encoded.len().strict_sub(pss_em_len);
    pss_encoded.drain(..leading);

    let mut pkcs1_encoded = vec![0u8; pkcs1_key.modulus().len()];
    pkcs1_key
      .public_operation_with_scratch(pkcs1_sig, &mut pkcs1_encoded, &mut pkcs1_scratch)
      .expect("valid RSA benchmark fixture must succeed");

    (
      pss_encoded,
      pss_em_bits,
      vec![0u8; pss_em_len],
      vec![0u8; pss_em_len],
      pkcs1_encoded,
    )
  };

  let ring_pss_key = ring_signature::UnparsedPublicKey::new(&ring_signature::RSA_PSS_2048_8192_SHA256, &pss_pkcs1);
  let ring_pkcs1_key =
    ring_signature::UnparsedPublicKey::new(&ring_signature::RSA_PKCS1_2048_8192_SHA256, &pkcs1_pkcs1);
  aws_lc_bench! {
    let aws_pss_key = aws_signature::UnparsedPublicKey::new(&aws_signature::RSA_PSS_2048_8192_SHA256, &pss_pkcs1);
    let aws_pkcs1_key =
      aws_signature::UnparsedPublicKey::new(&aws_signature::RSA_PKCS1_2048_8192_SHA256, &pkcs1_pkcs1);
  }

  let rustcrypto_pss_key = RustCryptoRsaPublicKey::from_public_key_der(pss_spki)
    .ok()
    .map(RustCryptoPssVerifyingKey::<sha2_010::Sha256>::new);
  let rustcrypto_pss_sig = RustCryptoPssSignature::try_from(pss_sig).expect("valid RSA benchmark fixture must succeed");
  let rustcrypto_pkcs1_key = RustCryptoRsaPublicKey::from_public_key_der(pkcs1_spki)
    .ok()
    .map(RustCryptoPkcs1v15VerifyingKey::<sha2_010::Sha256>::new);
  let rustcrypto_pkcs1_sig =
    RustCryptoPkcs1v15Signature::try_from(pkcs1_sig).expect("valid RSA benchmark fixture must succeed");

  let mut group = c.benchmark_group(name);

  group.bench_function("parse-spki-rscrypto", |b| {
    b.iter(|| {
      black_box(
        RsaPublicKey::from_spki_der_with_policy(black_box(pss_spki), import_policy)
          .expect("valid RSA benchmark fixture must succeed"),
      )
    })
  });
  if rustcrypto_pss_key.is_some() {
    group.bench_function("parse-spki-rustcrypto-rsa", |b| {
      b.iter(|| {
        black_box(
          RustCryptoRsaPublicKey::from_public_key_der(black_box(pss_spki))
            .expect("valid RSA benchmark fixture must succeed"),
        )
      })
    });
  }
  group.bench_function("scratch-setup-rscrypto", |b| {
    b.iter(|| black_box(pss_key.public_scratch()))
  });
  group.bench_function("public-op-e65537", |b| {
    b.iter(|| {
      pss_key
        .public_operation_with_scratch(
          black_box(&representative),
          black_box(&mut out),
          black_box(&mut pss_scratch),
        )
        .expect("valid RSA benchmark fixture must succeed")
    })
  });
  #[cfg(feature = "diag")]
  {
    let mut cios_scratch = pss_key.public_scratch();
    let mut product_scratch = pss_key.public_scratch();
    group.bench_function("public-op-e65537-product-montgomery", |b| {
      b.iter(|| {
        diag_rsa_public_operation_product(
          black_box(&pss_key),
          black_box(&representative),
          black_box(&mut out),
          black_box(&mut product_scratch),
        )
        .expect("valid RSA benchmark fixture must succeed")
      })
    });
    group.bench_function("public-op-e65537-cios-candidate", |b| {
      b.iter(|| {
        diag_rsa_public_operation_cios(
          black_box(&pss_key),
          black_box(&representative),
          black_box(&mut out),
          black_box(&mut cios_scratch),
        )
        .expect("valid RSA benchmark fixture must succeed")
      })
    });
  }
  #[cfg(feature = "diag")]
  group.bench_function("padding-pss-sha256-rscrypto", |b| {
    b.iter(|| {
      diag_rsa_verify_pss_encoded_with_scratch(
        RsaPssProfile::Sha256,
        black_box(MESSAGE_PSS),
        black_box(&pss_encoded),
        black_box(pss_em_bits),
        black_box(&mut pss_db),
        black_box(&mut pss_db_mask),
      )
      .expect("valid RSA benchmark fixture must succeed")
    })
  });
  #[cfg(feature = "diag")]
  group.bench_function("padding-pkcs1v15-sha256-rscrypto", |b| {
    b.iter(|| {
      diag_rsa_verify_pkcs1v15_encoded(
        RsaPkcs1v15Profile::Sha256,
        black_box(MESSAGE_PKCS1V15),
        black_box(&pkcs1_encoded),
      )
      .expect("valid RSA benchmark fixture must succeed")
    })
  });
  group.bench_function("verify-pss-sha256-rscrypto", |b| {
    b.iter(|| {
      pss_key
        .verify_pss_with_scratch(
          RsaPssProfile::Sha256,
          black_box(MESSAGE_PSS),
          black_box(pss_sig),
          black_box(&mut pss_scratch),
        )
        .expect("valid RSA benchmark fixture must succeed")
    })
  });
  group.bench_function("verify-pss-sha256-rscrypto-oneshot", |b| {
    b.iter(|| {
      pss_key
        .verify_pss(RsaPssProfile::Sha256, black_box(MESSAGE_PSS), black_box(pss_sig))
        .expect("valid RSA benchmark fixture must succeed")
    })
  });
  group.bench_function("verify-pss-sha256-rscrypto-cold", |b| {
    b.iter(|| {
      RsaPublicKey::from_spki_der_with_policy(black_box(pss_spki), import_policy)
        .expect("valid RSA benchmark fixture must succeed")
        .verify_pss(RsaPssProfile::Sha256, black_box(MESSAGE_PSS), black_box(pss_sig))
        .expect("valid RSA benchmark fixture must succeed")
    })
  });
  if let Some(rustcrypto_pss_key) = rustcrypto_pss_key {
    group.bench_function("verify-pss-sha256-rustcrypto-rsa", |b| {
      b.iter(|| {
        rustcrypto_pss_key
          .verify(black_box(MESSAGE_PSS), black_box(&rustcrypto_pss_sig))
          .expect("valid RSA benchmark fixture must succeed")
      })
    });
  }
  group.bench_function("verify-pss-sha256-ring", |b| {
    b.iter(|| {
      ring_pss_key
        .verify(black_box(MESSAGE_PSS), black_box(pss_sig))
        .expect("valid RSA benchmark fixture must succeed")
    })
  });
  aws_lc_bench! {
    group.bench_function("verify-pss-sha256-aws-lc-rs", |b| {
      b.iter(|| aws_pss_key.verify(black_box(MESSAGE_PSS), black_box(pss_sig)).expect("valid RSA benchmark fixture must succeed"))
    });
  }
  group.bench_function("verify-pkcs1v15-sha256-rscrypto", |b| {
    b.iter(|| {
      pkcs1_key
        .verify_pkcs1v15_with_scratch(
          RsaPkcs1v15Profile::Sha256,
          black_box(MESSAGE_PKCS1V15),
          black_box(pkcs1_sig),
          black_box(&mut pkcs1_scratch),
        )
        .expect("valid RSA benchmark fixture must succeed")
    })
  });
  group.bench_function("verify-pkcs1v15-sha256-rscrypto-oneshot", |b| {
    b.iter(|| {
      pkcs1_key
        .verify_pkcs1v15(
          RsaPkcs1v15Profile::Sha256,
          black_box(MESSAGE_PKCS1V15),
          black_box(pkcs1_sig),
        )
        .expect("valid RSA benchmark fixture must succeed")
    })
  });
  group.bench_function("verify-pkcs1v15-sha256-rscrypto-cold", |b| {
    b.iter(|| {
      RsaPublicKey::from_spki_der_with_policy(black_box(pkcs1_spki), import_policy)
        .expect("valid RSA benchmark fixture must succeed")
        .verify_pkcs1v15(
          RsaPkcs1v15Profile::Sha256,
          black_box(MESSAGE_PKCS1V15),
          black_box(pkcs1_sig),
        )
        .expect("valid RSA benchmark fixture must succeed")
    })
  });
  if let Some(rustcrypto_pkcs1_key) = rustcrypto_pkcs1_key {
    group.bench_function("verify-pkcs1v15-sha256-rustcrypto-rsa", |b| {
      b.iter(|| {
        rustcrypto_pkcs1_key
          .verify(black_box(MESSAGE_PKCS1V15), black_box(&rustcrypto_pkcs1_sig))
          .expect("valid RSA benchmark fixture must succeed")
      })
    });
  }
  group.bench_function("verify-pkcs1v15-sha256-ring", |b| {
    b.iter(|| {
      ring_pkcs1_key
        .verify(black_box(MESSAGE_PKCS1V15), black_box(pkcs1_sig))
        .expect("valid RSA benchmark fixture must succeed")
    })
  });
  aws_lc_bench! {
    group.bench_function("verify-pkcs1v15-sha256-aws-lc-rs", |b| {
      b.iter(|| {
        aws_pkcs1_key
          .verify(black_box(MESSAGE_PKCS1V15), black_box(pkcs1_sig))
          .expect("valid RSA benchmark fixture must succeed")
      })
    });
  }

  group.finish();
}

fn rsa_public_exponents(c: &mut Criterion) {
  let modulus = legacy_exponent_modulus();
  let input = legacy_exponent_input();
  let policy = RsaPublicKeyPolicy::legacy_verification().allow_legacy_small_exponents();

  let key_e3 = RsaPublicKey::from_pkcs1_der_with_policy(&pkcs1_der_from_modulus_exponent(&modulus, &[0x03]), &policy)
    .expect("valid RSA benchmark fixture must succeed");
  let key_e17 = RsaPublicKey::from_pkcs1_der_with_policy(&pkcs1_der_from_modulus_exponent(&modulus, &[0x11]), &policy)
    .expect("valid RSA benchmark fixture must succeed");
  let key_e65537 =
    RsaPublicKey::from_pkcs1_der_with_policy(&pkcs1_der_from_modulus_exponent(&modulus, &[0x01, 0x00, 0x01]), &policy)
      .expect("valid RSA benchmark fixture must succeed");
  let key_generic = RsaPublicKey::from_pkcs1_der_with_policy(
    &pkcs1_der_from_modulus_exponent(&modulus, &[0x49, 0xd2, 0xa1]),
    &policy.allow_legacy_odd_exponents(),
  )
  .expect("valid RSA benchmark fixture must succeed");

  let mut scratch_e3 = key_e3.public_scratch();
  let mut scratch_e17 = key_e17.public_scratch();
  let mut scratch_e65537 = key_e65537.public_scratch();
  let mut scratch_generic = key_generic.public_scratch();
  #[cfg(feature = "diag")]
  let mut cios_scratch_e3 = key_e3.public_scratch();
  #[cfg(feature = "diag")]
  let mut product_scratch_e3 = key_e3.public_scratch();
  #[cfg(feature = "diag")]
  let mut generic_scratch_e3 = key_e3.public_scratch();
  #[cfg(feature = "diag")]
  let mut cios_scratch_e17 = key_e17.public_scratch();
  #[cfg(feature = "diag")]
  let mut product_scratch_e17 = key_e17.public_scratch();
  #[cfg(feature = "diag")]
  let mut generic_scratch_e17 = key_e17.public_scratch();
  #[cfg(feature = "diag")]
  let mut cios_scratch_e65537 = key_e65537.public_scratch();
  #[cfg(feature = "diag")]
  let mut product_scratch_e65537 = key_e65537.public_scratch();
  #[cfg(feature = "diag")]
  let mut generic_scratch_e65537 = key_e65537.public_scratch();
  #[cfg(feature = "diag")]
  let mut cios_scratch_generic = key_generic.public_scratch();
  #[cfg(feature = "diag")]
  let mut product_scratch_generic = key_generic.public_scratch();
  let mut out = vec![0u8; modulus.len()];

  let mut group = c.benchmark_group("rsa-2048-public-exponents");
  group.bench_function("public-op-e3", |b| {
    b.iter(|| {
      key_e3
        .public_operation_with_scratch(black_box(&input), black_box(&mut out), black_box(&mut scratch_e3))
        .expect("valid RSA benchmark fixture must succeed")
    })
  });
  #[cfg(feature = "diag")]
  group.bench_function("public-op-e3-bitserial-baseline", |b| {
    b.iter(|| {
      diag_rsa_public_operation_bitserial(black_box(&key_e3), black_box(&input), black_box(&mut out))
        .expect("valid RSA benchmark fixture must succeed")
    })
  });
  #[cfg(feature = "diag")]
  group.bench_function("public-op-e3-product-montgomery", |b| {
    b.iter(|| {
      diag_rsa_public_operation_product(
        black_box(&key_e3),
        black_box(&input),
        black_box(&mut out),
        black_box(&mut product_scratch_e3),
      )
      .expect("valid RSA benchmark fixture must succeed")
    })
  });
  #[cfg(feature = "diag")]
  group.bench_function("public-op-e3-generic-exponent", |b| {
    b.iter(|| {
      diag_rsa_public_operation_generic_exponent(
        black_box(&key_e3),
        black_box(&input),
        black_box(&mut out),
        black_box(&mut generic_scratch_e3),
      )
      .expect("valid RSA benchmark fixture must succeed")
    })
  });
  #[cfg(feature = "diag")]
  group.bench_function("public-op-e3-cios-candidate", |b| {
    b.iter(|| {
      diag_rsa_public_operation_cios(
        black_box(&key_e3),
        black_box(&input),
        black_box(&mut out),
        black_box(&mut cios_scratch_e3),
      )
      .expect("valid RSA benchmark fixture must succeed")
    })
  });
  group.bench_function("public-op-e17", |b| {
    b.iter(|| {
      key_e17
        .public_operation_with_scratch(black_box(&input), black_box(&mut out), black_box(&mut scratch_e17))
        .expect("valid RSA benchmark fixture must succeed")
    })
  });
  #[cfg(feature = "diag")]
  group.bench_function("public-op-e17-bitserial-baseline", |b| {
    b.iter(|| {
      diag_rsa_public_operation_bitserial(black_box(&key_e17), black_box(&input), black_box(&mut out))
        .expect("valid RSA benchmark fixture must succeed")
    })
  });
  #[cfg(feature = "diag")]
  group.bench_function("public-op-e17-product-montgomery", |b| {
    b.iter(|| {
      diag_rsa_public_operation_product(
        black_box(&key_e17),
        black_box(&input),
        black_box(&mut out),
        black_box(&mut product_scratch_e17),
      )
      .expect("valid RSA benchmark fixture must succeed")
    })
  });
  #[cfg(feature = "diag")]
  group.bench_function("public-op-e17-generic-exponent", |b| {
    b.iter(|| {
      diag_rsa_public_operation_generic_exponent(
        black_box(&key_e17),
        black_box(&input),
        black_box(&mut out),
        black_box(&mut generic_scratch_e17),
      )
      .expect("valid RSA benchmark fixture must succeed")
    })
  });
  #[cfg(feature = "diag")]
  group.bench_function("public-op-e17-cios-candidate", |b| {
    b.iter(|| {
      diag_rsa_public_operation_cios(
        black_box(&key_e17),
        black_box(&input),
        black_box(&mut out),
        black_box(&mut cios_scratch_e17),
      )
      .expect("valid RSA benchmark fixture must succeed")
    })
  });
  group.bench_function("public-op-e65537", |b| {
    b.iter(|| {
      key_e65537
        .public_operation_with_scratch(black_box(&input), black_box(&mut out), black_box(&mut scratch_e65537))
        .expect("valid RSA benchmark fixture must succeed")
    })
  });
  #[cfg(feature = "diag")]
  group.bench_function("public-op-e65537-bitserial-baseline", |b| {
    b.iter(|| {
      diag_rsa_public_operation_bitserial(black_box(&key_e65537), black_box(&input), black_box(&mut out))
        .expect("valid RSA benchmark fixture must succeed")
    })
  });
  #[cfg(feature = "diag")]
  group.bench_function("public-op-e65537-product-montgomery", |b| {
    b.iter(|| {
      diag_rsa_public_operation_product(
        black_box(&key_e65537),
        black_box(&input),
        black_box(&mut out),
        black_box(&mut product_scratch_e65537),
      )
      .expect("valid RSA benchmark fixture must succeed")
    })
  });
  #[cfg(feature = "diag")]
  group.bench_function("public-op-e65537-generic-exponent", |b| {
    b.iter(|| {
      diag_rsa_public_operation_generic_exponent(
        black_box(&key_e65537),
        black_box(&input),
        black_box(&mut out),
        black_box(&mut generic_scratch_e65537),
      )
      .expect("valid RSA benchmark fixture must succeed")
    })
  });
  #[cfg(feature = "diag")]
  group.bench_function("public-op-e65537-cios-candidate", |b| {
    b.iter(|| {
      diag_rsa_public_operation_cios(
        black_box(&key_e65537),
        black_box(&input),
        black_box(&mut out),
        black_box(&mut cios_scratch_e65537),
      )
      .expect("valid RSA benchmark fixture must succeed")
    })
  });
  group.bench_function("public-op-e0x49d2a1-generic", |b| {
    b.iter(|| {
      key_generic
        .public_operation_with_scratch(black_box(&input), black_box(&mut out), black_box(&mut scratch_generic))
        .expect("valid RSA benchmark fixture must succeed")
    })
  });
  #[cfg(feature = "diag")]
  group.bench_function("public-op-e0x49d2a1-bitserial-baseline", |b| {
    b.iter(|| {
      diag_rsa_public_operation_bitserial(black_box(&key_generic), black_box(&input), black_box(&mut out))
        .expect("valid RSA benchmark fixture must succeed")
    })
  });
  #[cfg(feature = "diag")]
  group.bench_function("public-op-e0x49d2a1-product-montgomery", |b| {
    b.iter(|| {
      diag_rsa_public_operation_product(
        black_box(&key_generic),
        black_box(&input),
        black_box(&mut out),
        black_box(&mut product_scratch_generic),
      )
      .expect("valid RSA benchmark fixture must succeed")
    })
  });
  #[cfg(feature = "diag")]
  group.bench_function("public-op-e0x49d2a1-cios-candidate", |b| {
    b.iter(|| {
      diag_rsa_public_operation_cios(
        black_box(&key_generic),
        black_box(&input),
        black_box(&mut out),
        black_box(&mut cios_scratch_generic),
      )
      .expect("valid RSA benchmark fixture must succeed")
    })
  });
  group.finish();
}

fn rsa_hash_components(c: &mut Criterion) {
  let mut group = c.benchmark_group("rsa-hash-components");

  group.bench_function("sha256-message-pss", |b| {
    b.iter(|| black_box(Sha256::digest(black_box(MESSAGE_PSS))))
  });
  group.bench_function("sha384-message-pss", |b| {
    b.iter(|| black_box(Sha384::digest(black_box(MESSAGE_PSS))))
  });
  group.bench_function("sha512-message-pss", |b| {
    b.iter(|| black_box(Sha512::digest(black_box(MESSAGE_PSS))))
  });
  group.bench_function("sha256-message-pkcs1v15", |b| {
    b.iter(|| black_box(Sha256::digest(black_box(MESSAGE_PKCS1V15))))
  });
  group.bench_function("sha384-message-pkcs1v15", |b| {
    b.iter(|| black_box(Sha384::digest(black_box(MESSAGE_PKCS1V15))))
  });
  group.bench_function("sha512-message-pkcs1v15", |b| {
    b.iter(|| black_box(Sha512::digest(black_box(MESSAGE_PKCS1V15))))
  });

  group.finish();
}

#[cfg(feature = "diag")]
fn rsa_montgomery_thresholds(c: &mut Criterion) {
  let mut group = c.benchmark_group("rsa-montgomery-thresholds");

  for (name, modulus_len) in [
    ("rsa-2048-32-limbs", 256usize),
    ("rsa-3072-48-limbs", 384usize),
    ("rsa-4096-64-limbs", 512usize),
    ("rsa-4160-65-limbs", 520usize),
    ("rsa-6144-96-limbs", 768usize),
    ("rsa-8128-127-limbs", 1016usize),
    ("rsa-8192-128-limbs", 1024usize),
  ] {
    let key = RsaPublicKey::from_pkcs1_der_with_policy(
      &synthetic_pkcs1_der(modulus_len),
      &RsaPublicKeyPolicy::legacy_verification(),
    )
    .expect("valid RSA benchmark fixture must succeed");
    let input = modulus_minus_one(&key);
    let mut out_auto = vec![0u8; key.modulus().len()];
    let mut out_product = vec![0u8; key.modulus().len()];
    let mut out_cios = vec![0u8; key.modulus().len()];
    let mut out_cios_portable = vec![0u8; key.modulus().len()];
    let mut scratch_auto = key.public_scratch();
    let mut scratch_product = key.public_scratch();
    let mut scratch_cios = key.public_scratch();
    let mut scratch_cios_portable = key.public_scratch();

    group.bench_function(format!("{name}/auto"), |b| {
      b.iter(|| {
        key
          .public_operation_with_scratch(
            black_box(&input),
            black_box(&mut out_auto),
            black_box(&mut scratch_auto),
          )
          .expect("valid RSA benchmark fixture must succeed")
      })
    });
    group.bench_function(format!("{name}/product-montgomery"), |b| {
      b.iter(|| {
        diag_rsa_public_operation_product(
          black_box(&key),
          black_box(&input),
          black_box(&mut out_product),
          black_box(&mut scratch_product),
        )
        .expect("valid RSA benchmark fixture must succeed")
      })
    });
    group.bench_function(format!("{name}/cios-candidate"), |b| {
      b.iter(|| {
        diag_rsa_public_operation_cios(
          black_box(&key),
          black_box(&input),
          black_box(&mut out_cios),
          black_box(&mut scratch_cios),
        )
        .expect("valid RSA benchmark fixture must succeed")
      })
    });
    group.bench_function(format!("{name}/cios-portable"), |b| {
      b.iter(|| {
        diag_rsa_public_operation_cios_portable(
          black_box(&key),
          black_box(&input),
          black_box(&mut out_cios_portable),
          black_box(&mut scratch_cios_portable),
        )
        .expect("valid RSA benchmark fixture must succeed")
      })
    });
  }

  group.finish();
}

fn rsa_components(c: &mut Criterion) {
  let pss2048_spki = pss_spki();
  let pss2048_sig = pss_signature_sha256();
  let pkcs1v15_2048_spki = pkcs1v15_spki();
  let pkcs1v15_2048_sig = pkcs1v15_signature_sha256();
  let legacy_policy = RsaPublicKeyPolicy::legacy_verification();
  let modern_policy = RsaPublicKeyPolicy::modern_verification();

  rsa_components_for_size(
    c,
    "rsa-2048",
    &pss2048_spki,
    &pss2048_sig,
    &pkcs1v15_2048_spki,
    &pkcs1v15_2048_sig,
    &legacy_policy,
  );
  rsa_components_for_size(
    c,
    "rsa-3072",
    RSA3072_SPKI,
    RSA3072_PSS_SHA256,
    RSA3072_SPKI,
    RSA3072_PKCS1V15_SHA256,
    &modern_policy,
  );
  rsa_components_for_size(
    c,
    "rsa-4096",
    RSA4096_SPKI,
    RSA4096_PSS_SHA256,
    RSA4096_SPKI,
    RSA4096_PKCS1V15_SHA256,
    &modern_policy,
  );
  rsa_components_for_size(
    c,
    "rsa-8192",
    RSA8192_SPKI,
    RSA8192_PSS_SHA256,
    RSA8192_SPKI,
    RSA8192_PKCS1V15_SHA256,
    &modern_policy,
  );
}

#[cfg(not(feature = "diag"))]
criterion_group!(
  benches,
  rsa_components,
  rsa_private_signing,
  rsa_public_exponents,
  rsa_hash_components
);
#[cfg(feature = "diag")]
criterion_group!(
  benches,
  rsa_components,
  rsa_private_signing,
  rsa_public_exponents,
  rsa_hash_components,
  rsa_montgomery_thresholds
);
criterion_main!(benches);
