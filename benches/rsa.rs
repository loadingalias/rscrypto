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
use rsa::{
  RsaPublicKey as RustCryptoRsaPublicKey,
  pkcs1v15::{Signature as RustCryptoPkcs1v15Signature, VerifyingKey as RustCryptoPkcs1v15VerifyingKey},
  pkcs8::DecodePublicKey,
  pss::{Signature as RustCryptoPssSignature, VerifyingKey as RustCryptoPssVerifyingKey},
  signature::Verifier as _,
};
#[cfg(feature = "diag")]
use rscrypto::auth::rsa::{
  diag_rsa_public_operation_bitserial, diag_rsa_public_operation_cios, diag_rsa_public_operation_generic_exponent,
  diag_rsa_public_operation_product, diag_rsa_verify_pkcs1v15_encoded, diag_rsa_verify_pss_encoded_with_scratch,
};
use rscrypto::{RsaPkcs1v15Profile, RsaPssProfile, RsaPublicKey, RsaPublicKeyPolicy, Sha256, Sha384, Sha512};

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

fn hex_to_vec(hex: &str) -> Vec<u8> {
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

fn exponent_bytes(exponent: u64) -> Vec<u8> {
  let bytes = exponent.to_be_bytes();
  let first_nonzero = bytes.iter().position(|&byte| byte != 0).unwrap_or(bytes.len() - 1);
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

fn rsa_components_for_size(
  c: &mut Criterion,
  name: &str,
  pss_spki: &[u8],
  pss_sig: &[u8],
  pkcs1_spki: &[u8],
  pkcs1_sig: &[u8],
  import_policy: &RsaPublicKeyPolicy,
) {
  let pss_key = RsaPublicKey::from_spki_der_with_policy(pss_spki, import_policy).unwrap();
  let mut pss_scratch = pss_key.public_scratch();
  let pkcs1_key = RsaPublicKey::from_spki_der_with_policy(pkcs1_spki, import_policy).unwrap();
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
      .unwrap();
    let pss_em_bits = pss_key.modulus_bits().strict_sub(1);
    let pss_em_len = pss_em_bits.strict_add(7) / 8;
    let leading = pss_encoded.len().strict_sub(pss_em_len);
    pss_encoded.drain(..leading);

    let mut pkcs1_encoded = vec![0u8; pkcs1_key.modulus().len()];
    pkcs1_key
      .public_operation_with_scratch(pkcs1_sig, &mut pkcs1_encoded, &mut pkcs1_scratch)
      .unwrap();

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
  let rustcrypto_pss_sig = RustCryptoPssSignature::try_from(pss_sig).unwrap();
  let rustcrypto_pkcs1_key = RustCryptoRsaPublicKey::from_public_key_der(pkcs1_spki)
    .ok()
    .map(RustCryptoPkcs1v15VerifyingKey::<sha2_010::Sha256>::new);
  let rustcrypto_pkcs1_sig = RustCryptoPkcs1v15Signature::try_from(pkcs1_sig).unwrap();

  let mut group = c.benchmark_group(name);

  group.bench_function("parse-spki-rscrypto", |b| {
    b.iter(|| black_box(RsaPublicKey::from_spki_der_with_policy(black_box(pss_spki), import_policy).unwrap()))
  });
  if rustcrypto_pss_key.is_some() {
    group.bench_function("parse-spki-rustcrypto-rsa", |b| {
      b.iter(|| black_box(RustCryptoRsaPublicKey::from_public_key_der(black_box(pss_spki)).unwrap()))
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
        .unwrap()
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
        .unwrap()
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
        .unwrap()
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
      .unwrap()
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
      .unwrap()
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
        .unwrap()
    })
  });
  group.bench_function("verify-pss-sha256-rscrypto-oneshot", |b| {
    b.iter(|| {
      pss_key
        .verify_pss(RsaPssProfile::Sha256, black_box(MESSAGE_PSS), black_box(pss_sig))
        .unwrap()
    })
  });
  group.bench_function("verify-pss-sha256-rscrypto-cold", |b| {
    b.iter(|| {
      RsaPublicKey::from_spki_der_with_policy(black_box(pss_spki), import_policy)
        .unwrap()
        .verify_pss(RsaPssProfile::Sha256, black_box(MESSAGE_PSS), black_box(pss_sig))
        .unwrap()
    })
  });
  if let Some(rustcrypto_pss_key) = rustcrypto_pss_key {
    group.bench_function("verify-pss-sha256-rustcrypto-rsa", |b| {
      b.iter(|| {
        rustcrypto_pss_key
          .verify(black_box(MESSAGE_PSS), black_box(&rustcrypto_pss_sig))
          .unwrap()
      })
    });
  }
  group.bench_function("verify-pss-sha256-ring", |b| {
    b.iter(|| ring_pss_key.verify(black_box(MESSAGE_PSS), black_box(pss_sig)).unwrap())
  });
  aws_lc_bench! {
    group.bench_function("verify-pss-sha256-aws-lc-rs", |b| {
      b.iter(|| aws_pss_key.verify(black_box(MESSAGE_PSS), black_box(pss_sig)).unwrap())
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
        .unwrap()
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
        .unwrap()
    })
  });
  group.bench_function("verify-pkcs1v15-sha256-rscrypto-cold", |b| {
    b.iter(|| {
      RsaPublicKey::from_spki_der_with_policy(black_box(pkcs1_spki), import_policy)
        .unwrap()
        .verify_pkcs1v15(
          RsaPkcs1v15Profile::Sha256,
          black_box(MESSAGE_PKCS1V15),
          black_box(pkcs1_sig),
        )
        .unwrap()
    })
  });
  if let Some(rustcrypto_pkcs1_key) = rustcrypto_pkcs1_key {
    group.bench_function("verify-pkcs1v15-sha256-rustcrypto-rsa", |b| {
      b.iter(|| {
        rustcrypto_pkcs1_key
          .verify(black_box(MESSAGE_PKCS1V15), black_box(&rustcrypto_pkcs1_sig))
          .unwrap()
      })
    });
  }
  group.bench_function("verify-pkcs1v15-sha256-ring", |b| {
    b.iter(|| {
      ring_pkcs1_key
        .verify(black_box(MESSAGE_PKCS1V15), black_box(pkcs1_sig))
        .unwrap()
    })
  });
  aws_lc_bench! {
    group.bench_function("verify-pkcs1v15-sha256-aws-lc-rs", |b| {
      b.iter(|| {
        aws_pkcs1_key
          .verify(black_box(MESSAGE_PKCS1V15), black_box(pkcs1_sig))
          .unwrap()
      })
    });
  }

  group.finish();
}

fn rsa_public_exponents(c: &mut Criterion) {
  let modulus = legacy_exponent_modulus();
  let input = legacy_exponent_input();
  let policy = RsaPublicKeyPolicy::legacy_verification().allow_legacy_small_exponents();

  let key_e3 =
    RsaPublicKey::from_pkcs1_der_with_policy(&pkcs1_der_from_modulus_exponent(&modulus, &[0x03]), &policy).unwrap();
  let key_e17 =
    RsaPublicKey::from_pkcs1_der_with_policy(&pkcs1_der_from_modulus_exponent(&modulus, &[0x11]), &policy).unwrap();
  let key_e65537 =
    RsaPublicKey::from_pkcs1_der_with_policy(&pkcs1_der_from_modulus_exponent(&modulus, &[0x01, 0x00, 0x01]), &policy)
      .unwrap();
  let key_generic = RsaPublicKey::from_pkcs1_der_with_policy(
    &pkcs1_der_from_modulus_exponent(&modulus, &[0x49, 0xd2, 0xa1]),
    &policy.allow_legacy_odd_exponents(),
  )
  .unwrap();

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
        .unwrap()
    })
  });
  #[cfg(feature = "diag")]
  group.bench_function("public-op-e3-bitserial-baseline", |b| {
    b.iter(|| diag_rsa_public_operation_bitserial(black_box(&key_e3), black_box(&input), black_box(&mut out)).unwrap())
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
      .unwrap()
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
      .unwrap()
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
      .unwrap()
    })
  });
  group.bench_function("public-op-e17", |b| {
    b.iter(|| {
      key_e17
        .public_operation_with_scratch(black_box(&input), black_box(&mut out), black_box(&mut scratch_e17))
        .unwrap()
    })
  });
  #[cfg(feature = "diag")]
  group.bench_function("public-op-e17-bitserial-baseline", |b| {
    b.iter(|| diag_rsa_public_operation_bitserial(black_box(&key_e17), black_box(&input), black_box(&mut out)).unwrap())
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
      .unwrap()
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
      .unwrap()
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
      .unwrap()
    })
  });
  group.bench_function("public-op-e65537", |b| {
    b.iter(|| {
      key_e65537
        .public_operation_with_scratch(black_box(&input), black_box(&mut out), black_box(&mut scratch_e65537))
        .unwrap()
    })
  });
  #[cfg(feature = "diag")]
  group.bench_function("public-op-e65537-bitserial-baseline", |b| {
    b.iter(|| {
      diag_rsa_public_operation_bitserial(black_box(&key_e65537), black_box(&input), black_box(&mut out)).unwrap()
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
      .unwrap()
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
      .unwrap()
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
      .unwrap()
    })
  });
  group.bench_function("public-op-e0x49d2a1-generic", |b| {
    b.iter(|| {
      key_generic
        .public_operation_with_scratch(black_box(&input), black_box(&mut out), black_box(&mut scratch_generic))
        .unwrap()
    })
  });
  #[cfg(feature = "diag")]
  group.bench_function("public-op-e0x49d2a1-bitserial-baseline", |b| {
    b.iter(|| {
      diag_rsa_public_operation_bitserial(black_box(&key_generic), black_box(&input), black_box(&mut out)).unwrap()
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
      .unwrap()
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
      .unwrap()
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
    ("rsa-4096-64-limbs", 512usize),
    ("rsa-4160-65-limbs", 520usize),
    ("rsa-8192-128-limbs", 1024usize),
  ] {
    let key = RsaPublicKey::from_pkcs1_der(&synthetic_pkcs1_der(modulus_len)).unwrap();
    let input = modulus_minus_one(&key);
    let mut out_auto = vec![0u8; key.modulus().len()];
    let mut out_product = vec![0u8; key.modulus().len()];
    let mut out_cios = vec![0u8; key.modulus().len()];
    let mut scratch_auto = key.public_scratch();
    let mut scratch_product = key.public_scratch();
    let mut scratch_cios = key.public_scratch();

    group.bench_function(format!("{name}/auto"), |b| {
      b.iter(|| {
        key
          .public_operation_with_scratch(
            black_box(&input),
            black_box(&mut out_auto),
            black_box(&mut scratch_auto),
          )
          .unwrap()
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
        .unwrap()
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
        .unwrap()
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
criterion_group!(benches, rsa_components, rsa_public_exponents, rsa_hash_components);
#[cfg(feature = "diag")]
criterion_group!(
  benches,
  rsa_components,
  rsa_public_exponents,
  rsa_hash_components,
  rsa_montgomery_thresholds
);
criterion_main!(benches);
