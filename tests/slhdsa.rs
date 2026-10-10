//! Differential tests of the public SLH-DSA API against two independent
//! FIPS 205 implementations: RustCrypto `slh-dsa` 0.2.0-rc.5 (pure signing,
//! key generation) and `fips205` 0.4.1 (pure and HashSLH-DSA, both variants).
#![cfg(feature = "slh-dsa")]

use fips205::traits::{SerDes as _, Signer as _, Verifier as _};
use rscrypto::*;

/// Deterministic test bytes that vary with `seed`.
fn bytes(len: usize, seed: u8) -> Vec<u8> {
  (0..len)
    .map(|index| u8::try_from(index % 251).expect("small index") ^ seed.wrapping_mul(31))
    .collect()
}

macro_rules! differential {
  (
    $test:ident,
    $profile:ident,
    $public:ident,
    $secret:ident,
    $hash_public:ident,
    $hash_secret:ident,
    $rustcrypto:ty,
    $fips:ident,
    $ph:expr,
    $n:literal,
    $cases:literal
  ) => {
    #[test]
    fn $test() {
      use fips205::$fips as oracle;
      const SIGNATURE: usize = $profile::SIGNATURE_LENGTH;
      for case in 0..$cases {
        let case = u8::try_from(case).expect("case index");
        let seeds = bytes(3 * $n, case);
        let (public, secret) = $profile::generate_keypair(|out| {
          out.copy_from_slice(&seeds);
          Ok(())
        })
        .expect("key generation");
        let encoded = secret.expose_secret();

        // Key generation from the same seeds.
        let rustcrypto = rustcrypto_slh_dsa::SigningKey::<$rustcrypto>::slh_keygen_internal(
          &seeds[..$n],
          &seeds[$n..2 * $n],
          &seeds[2 * $n..],
        );
        assert_eq!(rustcrypto.to_bytes().as_slice(), encoded.as_bytes());
        let fips_secret = oracle::PrivateKey::try_from_bytes(encoded.as_bytes()).expect("fips205 secret key");
        let fips_public = oracle::PublicKey::try_from_bytes(&public.to_bytes()).expect("fips205 public key");

        let message = bytes(usize::from(case).strict_mul(97).strict_add(1), case ^ 0x5a);
        let context = bytes(usize::from(case).strict_mul(61), case ^ 0xc3);
        let addrnd = bytes($n, case ^ 0x99);

        // Pure, deterministic: three byte-identical signatures.
        let mut ours = [0; SIGNATURE];
        secret
          .sign_deterministic(&message, &context, &mut ours)
          .expect("deterministic signing");
        let theirs = rustcrypto
          .try_sign_with_context(&message, &context, None)
          .expect("RustCrypto signing");
        assert_eq!(theirs.to_bytes().as_slice(), ours.as_slice());
        assert_eq!(
          fips_secret
            .try_sign(&message, &context, false)
            .expect("fips205 signing"),
          ours
        );

        // Pure, hedged: the same addrnd gives the same signature, and an
        // oracle's random hedged signature verifies here.
        secret
          .sign_with(
            &message,
            &context,
            |random| {
              random.copy_from_slice(&addrnd);
              Ok(())
            },
            &mut ours,
          )
          .expect("hedged signing");
        let theirs = rustcrypto
          .try_sign_with_context(&message, &context, Some(&addrnd))
          .expect("RustCrypto hedged signing");
        assert_eq!(theirs.to_bytes().as_slice(), ours.as_slice());
        assert!(fips_public.verify(&message, &ours, &context));
        let random = fips_secret
          .try_sign(&message, &context, true)
          .expect("fips205 hedged signing");
        public
          .verify_with_context(&message, &context, &random)
          .expect("fips205 hedged signature");

        // HashSLH-DSA with the RFC 9909 pairing, against fips205.
        let hash_secret = $hash_secret::try_from_slice(encoded.as_bytes()).expect("HashSLH-DSA secret key");
        let hash_public = $hash_public::from_bytes(public.to_bytes());
        hash_secret
          .sign_deterministic(&message, &context, &mut ours)
          .expect("deterministic HashSLH-DSA");
        assert_eq!(
          fips_secret
            .try_hash_sign(&message, &context, &$ph, false)
            .expect("fips205 HashSLH-DSA"),
          ours
        );
        assert!(fips_public.hash_verify(&message, &ours, &context, &$ph));
        assert!(!fips_public.verify(&message, &ours, &context));
        let random = fips_secret
          .try_hash_sign(&message, &context, &$ph, true)
          .expect("fips205 hedged HashSLH-DSA");
        hash_public
          .verify_with_context(&message, &context, &random)
          .expect("fips205 hedged HashSLH-DSA signature");
        assert!(public.verify_with_context(&message, &context, &random).is_err());
      }

      // An independently generated key passes the import root check and
      // signs identically.
      let (fips_public, fips_secret) = oracle::try_keygen().expect("fips205 key generation");
      let secret = $secret::try_from_slice(&fips_secret.clone().into_bytes()).expect("fips205 key imports");
      assert_eq!(secret.public_key().as_bytes(), &fips_public.into_bytes());
      let mut ours = [0; SIGNATURE];
      secret
        .sign_deterministic(b"independent key", b"", &mut ours)
        .expect("signing");
      assert_eq!(
        fips_secret
          .try_sign(b"independent key", b"", false)
          .expect("fips205 signing"),
        ours
      );
    }
  };
}

differential!(
  sha2_128s_matches_independent_implementations,
  SlhDsaSha2_128s,
  SlhDsaSha2_128sPublicKey,
  SlhDsaSha2_128sSecretKey,
  HashSlhDsaSha2_128sWithSha256PublicKey,
  HashSlhDsaSha2_128sWithSha256SecretKey,
  rustcrypto_slh_dsa::Sha2_128s,
  slh_dsa_sha2_128s,
  fips205::Ph::SHA256,
  16,
  1
);
differential!(
  sha2_128f_matches_independent_implementations,
  SlhDsaSha2_128f,
  SlhDsaSha2_128fPublicKey,
  SlhDsaSha2_128fSecretKey,
  HashSlhDsaSha2_128fWithSha256PublicKey,
  HashSlhDsaSha2_128fWithSha256SecretKey,
  rustcrypto_slh_dsa::Sha2_128f,
  slh_dsa_sha2_128f,
  fips205::Ph::SHA256,
  16,
  3
);
differential!(
  sha2_192s_matches_independent_implementations,
  SlhDsaSha2_192s,
  SlhDsaSha2_192sPublicKey,
  SlhDsaSha2_192sSecretKey,
  HashSlhDsaSha2_192sWithSha512PublicKey,
  HashSlhDsaSha2_192sWithSha512SecretKey,
  rustcrypto_slh_dsa::Sha2_192s,
  slh_dsa_sha2_192s,
  fips205::Ph::SHA512,
  24,
  1
);
differential!(
  sha2_192f_matches_independent_implementations,
  SlhDsaSha2_192f,
  SlhDsaSha2_192fPublicKey,
  SlhDsaSha2_192fSecretKey,
  HashSlhDsaSha2_192fWithSha512PublicKey,
  HashSlhDsaSha2_192fWithSha512SecretKey,
  rustcrypto_slh_dsa::Sha2_192f,
  slh_dsa_sha2_192f,
  fips205::Ph::SHA512,
  24,
  3
);
differential!(
  sha2_256s_matches_independent_implementations,
  SlhDsaSha2_256s,
  SlhDsaSha2_256sPublicKey,
  SlhDsaSha2_256sSecretKey,
  HashSlhDsaSha2_256sWithSha512PublicKey,
  HashSlhDsaSha2_256sWithSha512SecretKey,
  rustcrypto_slh_dsa::Sha2_256s,
  slh_dsa_sha2_256s,
  fips205::Ph::SHA512,
  32,
  1
);
differential!(
  sha2_256f_matches_independent_implementations,
  SlhDsaSha2_256f,
  SlhDsaSha2_256fPublicKey,
  SlhDsaSha2_256fSecretKey,
  HashSlhDsaSha2_256fWithSha512PublicKey,
  HashSlhDsaSha2_256fWithSha512SecretKey,
  rustcrypto_slh_dsa::Sha2_256f,
  slh_dsa_sha2_256f,
  fips205::Ph::SHA512,
  32,
  3
);
differential!(
  shake_128s_matches_independent_implementations,
  SlhDsaShake128s,
  SlhDsaShake128sPublicKey,
  SlhDsaShake128sSecretKey,
  HashSlhDsaShake128sWithShake128PublicKey,
  HashSlhDsaShake128sWithShake128SecretKey,
  rustcrypto_slh_dsa::Shake128s,
  slh_dsa_shake_128s,
  fips205::Ph::SHAKE128,
  16,
  1
);
differential!(
  shake_128f_matches_independent_implementations,
  SlhDsaShake128f,
  SlhDsaShake128fPublicKey,
  SlhDsaShake128fSecretKey,
  HashSlhDsaShake128fWithShake128PublicKey,
  HashSlhDsaShake128fWithShake128SecretKey,
  rustcrypto_slh_dsa::Shake128f,
  slh_dsa_shake_128f,
  fips205::Ph::SHAKE128,
  16,
  3
);
differential!(
  shake_192s_matches_independent_implementations,
  SlhDsaShake192s,
  SlhDsaShake192sPublicKey,
  SlhDsaShake192sSecretKey,
  HashSlhDsaShake192sWithShake256PublicKey,
  HashSlhDsaShake192sWithShake256SecretKey,
  rustcrypto_slh_dsa::Shake192s,
  slh_dsa_shake_192s,
  fips205::Ph::SHAKE256,
  24,
  1
);
differential!(
  shake_192f_matches_independent_implementations,
  SlhDsaShake192f,
  SlhDsaShake192fPublicKey,
  SlhDsaShake192fSecretKey,
  HashSlhDsaShake192fWithShake256PublicKey,
  HashSlhDsaShake192fWithShake256SecretKey,
  rustcrypto_slh_dsa::Shake192f,
  slh_dsa_shake_192f,
  fips205::Ph::SHAKE256,
  24,
  3
);
differential!(
  shake_256s_matches_independent_implementations,
  SlhDsaShake256s,
  SlhDsaShake256sPublicKey,
  SlhDsaShake256sSecretKey,
  HashSlhDsaShake256sWithShake256PublicKey,
  HashSlhDsaShake256sWithShake256SecretKey,
  rustcrypto_slh_dsa::Shake256s,
  slh_dsa_shake_256s,
  fips205::Ph::SHAKE256,
  32,
  1
);
differential!(
  shake_256f_matches_independent_implementations,
  SlhDsaShake256f,
  SlhDsaShake256fPublicKey,
  SlhDsaShake256fSecretKey,
  HashSlhDsaShake256fWithShake256PublicKey,
  HashSlhDsaShake256fWithShake256SecretKey,
  rustcrypto_slh_dsa::Shake256f,
  slh_dsa_shake_256f,
  fips205::Ph::SHAKE256,
  32,
  3
);
