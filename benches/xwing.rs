//! X-Wing production workloads (draft-connolly-cfrg-xwing-kem-11).
//! Fixtures use a public, deterministic seed and deterministic encapsulation
//! randomness.
//!
//! Comparators: `libcrux-kem` (`XWingKemDraft06` and its array `XWing`) on
//! libcrux ML-KEM and HACL X25519, and RustCrypto `x-wing` on RustCrypto
//! ML-KEM and x25519-dalek. AWS-LC has no X-Wing. Each row uses its library's
//! fastest form for the task. Before timing, every row checks its output
//! against another implementation.

#[path = "common/criterion.rs"]
mod bench_config;

use core::hint::black_box;

use bench_config::checked;
use criterion::Criterion;
use libcrux_kem::{Algorithm, Ct, PrivateKey, PublicKey, XWing as LibcruxXWing};
use libcrux_traits::kem::arrayref::Kem as _;
use rscrypto::{Kem, XWing, XWingCiphertext, XWingDecapsulationKey, XWingEncapsulationKey, XWingError};
use x_wing::{Decapsulator as _, KeyExport as _, TryDecapsulate as _};

const SEED: [u8; XWing::DECAPSULATION_KEY_SIZE] = [0x42; XWing::DECAPSULATION_KEY_SIZE];
const RANDOMNESS: [u8; XWing::ENCAPSULATION_RANDOM_SIZE] = [0x5a; XWing::ENCAPSULATION_RANDOM_SIZE];
/// libcrux's expanded decapsulation key: the ML-KEM-768 key, then the X25519 key.
const LIBCRUX_EXPANDED_KEY_SIZE: usize = 2400 + 32;

type Encapsulated = ([u8; XWing::CIPHERTEXT_SIZE], [u8; XWing::SHARED_SECRET_SIZE]);

fn filled(bytes: &[u8]) -> impl FnMut(&mut [u8]) -> Result<(), XWingError> + '_ {
  |out| {
    out.copy_from_slice(bytes);
    Ok(())
  }
}

fn array<const N: usize>(bytes: &[u8]) -> [u8; N] {
  bytes.try_into().expect("fixed benchmark length")
}

fn libcrux_keygen(
  seed: &[u8; XWing::DECAPSULATION_KEY_SIZE],
) -> ([u8; XWing::ENCAPSULATION_KEY_SIZE], [u8; LIBCRUX_EXPANDED_KEY_SIZE]) {
  let mut encapsulation_key = [0u8; XWing::ENCAPSULATION_KEY_SIZE];
  let mut decapsulation_key = [0u8; LIBCRUX_EXPANDED_KEY_SIZE];
  LibcruxXWing::keygen(&mut encapsulation_key, &mut decapsulation_key, seed).expect("libcrux key generation");
  (encapsulation_key, decapsulation_key)
}

fn libcrux_encapsulate(encapsulation_key: &[u8; XWing::ENCAPSULATION_KEY_SIZE]) -> Encapsulated {
  let mut ciphertext = [0u8; XWing::CIPHERTEXT_SIZE];
  let mut shared_secret = [0u8; XWing::SHARED_SECRET_SIZE];
  LibcruxXWing::encaps(&mut ciphertext, &mut shared_secret, encapsulation_key, &RANDOMNESS)
    .expect("libcrux encapsulation");
  (ciphertext, shared_secret)
}

fn xwing(c: &mut Criterion) {
  if !bench_config::selected("xwing/") {
    return;
  }
  let oracle = x_wing::DecapsulationKeyRejectNonContrib::from(SEED);
  let oracle_key = oracle.encapsulation_key().clone();
  let expected_key: [u8; XWing::ENCAPSULATION_KEY_SIZE] = array(&oracle_key.to_bytes());
  let (oracle_ciphertext, oracle_secret) = oracle_key.encapsulate_deterministic(&RANDOMNESS.into());
  let expected: Encapsulated = (array(&oracle_ciphertext), array(&oracle_secret));
  let (libcrux_key, libcrux_expanded) = libcrux_keygen(&SEED);
  assert_eq!(libcrux_key, expected_key, "libcrux key generation must match x-wing");
  // Coefficient 0 becomes 4095, at least q = 3329, so the FIPS 203 section 7.2
  // modulus check must reject this key.
  let mut malformed_key = expected_key;
  malformed_key[0] = 0xff;
  malformed_key[1] |= 0x0f;

  // Expand the seed and encode the encapsulation key. libcrux also writes its
  // expanded decapsulation key, and x-wing keeps its expanded encapsulation key.
  let mut group = c.benchmark_group("xwing/keygen/derand-encoded");
  group.bench_function(
    "rscrypto",
    checked(
      || {
        XWing::generate_keypair(filled(black_box(&SEED)))
          .expect("key generation")
          .0
          .to_bytes()
      },
      |key| assert_eq!(key, expected_key),
    ),
  );
  group.bench_function(
    "libcrux",
    checked(
      || libcrux_keygen(black_box(&SEED)).0,
      |key| assert_eq!(key, expected_key),
    ),
  );
  group.bench_function(
    "rustcrypto",
    checked(
      || {
        let key = x_wing::DecapsulationKey::from(*black_box(&SEED));
        array::<{ XWing::ENCAPSULATION_KEY_SIZE }>(&key.encapsulation_key().to_bytes())
      },
      |key| assert_eq!(key, libcrux_key),
    ),
  );
  group.finish();

  // Each library's best reusable key, built and validated before timing.
  // rscrypto's prepared key holds the decoded ML-KEM key, its expanded matrix,
  // and H(ek). x-wing's key holds the decoded ML-KEM key and H(ek). libcrux's
  // array form takes the encoded key, so decoding and the matrix stay timed.
  let prepared_key = XWingEncapsulationKey::from_bytes(expected_key)
    .prepare()
    .expect("key preparation");
  let mut group = c.benchmark_group("xwing/encapsulate/derand-reuse-key");
  group.bench_function(
    "rscrypto",
    checked(
      || {
        let (ciphertext, shared_secret) = black_box(&prepared_key)
          .encapsulate(filled(black_box(&RANDOMNESS)))
          .expect("encapsulation");
        (ciphertext.to_bytes(), *shared_secret.as_bytes())
      },
      |output| assert_eq!(output, expected),
    ),
  );
  group.bench_function(
    "libcrux",
    checked(
      || libcrux_encapsulate(black_box(&libcrux_key)),
      |output| assert_eq!(output, expected),
    ),
  );
  group.bench_function(
    "rustcrypto",
    checked(
      || {
        let (ciphertext, shared_secret) =
          black_box(&oracle_key).encapsulate_deterministic(&(*black_box(&RANDOMNESS)).into());
        (array(&ciphertext), array(&shared_secret))
      },
      |output| assert_eq!(output, libcrux_encapsulate(&libcrux_key)),
    ),
  );
  group.finish();

  // Import and validate identical key bytes, then encapsulate. Each path must
  // reject the key with an unreduced coefficient.
  let rscrypto_import = |bytes: &[u8; XWing::ENCAPSULATION_KEY_SIZE]| -> Option<Encapsulated> {
    let key = XWingEncapsulationKey::from_bytes(*bytes).prepare().ok()?;
    let (ciphertext, shared_secret) = key.encapsulate(filled(black_box(&RANDOMNESS))).ok()?;
    Some((ciphertext.to_bytes(), *shared_secret.as_bytes()))
  };
  let libcrux_import = |bytes: &[u8; XWing::ENCAPSULATION_KEY_SIZE]| -> Option<Encapsulated> {
    let key = PublicKey::decode(Algorithm::XWingKemDraft06, bytes).ok()?;
    let (shared_secret, ciphertext) = key.encapsulate_derand(black_box(&RANDOMNESS)).ok()?;
    Some((array(&ciphertext.encode()), array(&shared_secret.encode())))
  };
  let rustcrypto_import = |bytes: &[u8; XWing::ENCAPSULATION_KEY_SIZE]| -> Option<Encapsulated> {
    let key = x_wing::EncapsulationKey::try_from(bytes.as_slice()).ok()?;
    let (ciphertext, shared_secret) = key.encapsulate_deterministic(&(*black_box(&RANDOMNESS)).into());
    Some((array(&ciphertext), array(&shared_secret)))
  };
  let mut group = c.benchmark_group("xwing/encapsulate/derand-import-encoded");
  group.bench_function(
    "rscrypto",
    checked(
      || rscrypto_import(black_box(&expected_key)),
      |output| {
        assert_eq!(output, Some(expected));
        assert_eq!(rscrypto_import(&malformed_key), None);
      },
    ),
  );
  group.bench_function(
    "libcrux",
    checked(
      || libcrux_import(black_box(&expected_key)),
      |output| {
        assert_eq!(output, Some(expected));
        assert_eq!(libcrux_import(&malformed_key), None);
      },
    ),
  );
  group.bench_function(
    "rustcrypto",
    checked(
      || rustcrypto_import(black_box(&expected_key)),
      |output| {
        assert_eq!(output, libcrux_import(&expected_key));
        assert_eq!(rustcrypto_import(&malformed_key), None);
      },
    ),
  );
  group.finish();

  // Decapsulate one ciphertext with each library's best reusable key, built
  // before timing. rscrypto's prepared key holds the expanded ML-KEM and X25519
  // keys and the ML-KEM matrix. libcrux's array form takes its expanded
  // decapsulation key. x-wing's key expands the seed in each call.
  let ciphertext = XWingCiphertext::from_bytes(expected.0);
  let prepared = XWingDecapsulationKey::from_bytes(SEED).prepare();
  let mut group = c.benchmark_group("xwing/decapsulate/reuse-key");
  group.bench_function(
    "rscrypto",
    checked(
      || {
        *black_box(&prepared)
          .decapsulate(black_box(&ciphertext))
          .expect("decapsulation")
          .as_bytes()
      },
      |secret| assert_eq!(secret, expected.1),
    ),
  );
  group.bench_function(
    "libcrux",
    checked(
      || {
        let mut shared_secret = [0u8; XWing::SHARED_SECRET_SIZE];
        LibcruxXWing::decaps(&mut shared_secret, black_box(&expected.0), black_box(&libcrux_expanded))
          .expect("libcrux decapsulation");
        shared_secret
      },
      |secret| assert_eq!(secret, expected.1),
    ),
  );
  group.bench_function(
    "rustcrypto",
    checked(
      || {
        array::<{ XWing::SHARED_SECRET_SIZE }>(
          &black_box(&oracle)
            .try_decapsulate(&(*black_box(&expected.0)).into())
            .expect("x-wing decapsulation"),
        )
      },
      |secret| assert_eq!(secret, expected.1),
    ),
  );
  group.finish();

  // Import identical 32-byte decapsulation-key bytes and the ciphertext, then
  // decapsulate: every row expands the seed inside timing.
  let mut group = c.benchmark_group("xwing/decapsulate/import-encoded");
  group.bench_function(
    "rscrypto",
    checked(
      || {
        let key = XWingDecapsulationKey::try_from_slice(black_box(&SEED)).expect("key import");
        let ciphertext = XWingCiphertext::try_from_slice(black_box(&expected.0)).expect("ciphertext import");
        *XWing::decapsulate(&key, &ciphertext).expect("decapsulation").as_bytes()
      },
      |secret| assert_eq!(secret, expected.1),
    ),
  );
  group.bench_function(
    "libcrux",
    checked(
      || {
        let key = PrivateKey::decode(Algorithm::XWingKemDraft06, black_box(&SEED)).expect("libcrux key import");
        let ciphertext =
          Ct::decode(Algorithm::XWingKemDraft06, black_box(&expected.0)).expect("libcrux ciphertext import");
        array::<{ XWing::SHARED_SECRET_SIZE }>(&ciphertext.decapsulate(&key).expect("libcrux decapsulation").encode())
      },
      |secret| assert_eq!(secret, expected.1),
    ),
  );
  group.bench_function(
    "rustcrypto",
    checked(
      || {
        let key = x_wing::DecapsulationKeyRejectNonContrib::from(*black_box(&SEED));
        array::<{ XWing::SHARED_SECRET_SIZE }>(
          &key
            .try_decapsulate(&(*black_box(&expected.0)).into())
            .expect("x-wing decapsulation"),
        )
      },
      |secret| assert_eq!(secret, expected.1),
    ),
  );
  group.finish();
}

fn main() {
  // X-Wing key expansion holds the prepared ML-KEM-768 key on the stack; keep
  // the comparators' stacks within an explicit bound on every host.
  std::thread::Builder::new()
    .stack_size(8 << 20)
    .spawn(|| bench_config::run(&[xwing]))
    .expect("spawn benchmark thread")
    .join()
    .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
}
