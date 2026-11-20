//! HPKE production workloads (RFC 9180 Base mode).
//!
//! Comparators: rust-hpke (`hpke` 0.15.1) on RustCrypto x25519-dalek,
//! aes-gcm, chacha20poly1305, hkdf, and x-wing, for both suites; and AWS-LC's
//! EVP_HPKE through `aws-lc-sys` for the X25519 suite. AWS-LC has no X-Wing.
//! Each row's output is checked before timing: a ciphertext opens with
//! another implementation, and setups and exports agree.

#[path = "common/aws_lc.rs"]
mod aws_lc;
#[path = "common/criterion.rs"]
mod bench_config;

use core::hint::black_box;

use aws_lc::aws_lc_bench;
use bench_config::{checked, checked_routine, checked_with};
use criterion::{Bencher, Criterion};
use hpke::{Deserializable as _, Kem as _, OpModeR, OpModeS, Serializable as _};
use rscrypto::{
  ChaCha20Rng, HpkeError, HpkeReceiver, HpkeSender, HpkeSuite, HpkeX25519HkdfSha256Aes128Gcm as X25519Suite,
  HpkeXWingHkdfSha256ChaCha20Poly1305 as XWingSuite,
};

const INFO: &[u8] = b"rscrypto HPKE benchmark info";
const AAD: &[u8] = b"rscrypto HPKE benchmark aad";
const LENGTHS: [usize; 2] = [64, 16384];
const SEED: [u8; 64] = [0x5a; 64];

type OracleKdf = hpke::kdf::HkdfSha256;

fn filled(bytes: &[u8]) -> impl FnMut(&mut [u8]) -> Result<(), HpkeError> + '_ {
  |out| {
    out.copy_from_slice(&bytes[..out.len()]);
    Ok(())
  }
}

fn rng() -> ChaCha20Rng {
  ChaCha20Rng::from_seed([0x5a; 32])
}

/// Times `seal` of one `length`-byte message with an established sender context into a reused
/// `ciphertext || tag` buffer. Each seal advances the sequence number, so before timing, `open`
/// must accept the first two messages in order.
fn context_seal<S>(
  sender: S,
  length: usize,
  seal: impl FnMut(&mut S, &mut [u8]),
  mut open: impl FnMut(&[u8]),
) -> impl FnMut(&mut Bencher<'_>) {
  checked_routine(
    (sender, vec![0u8; length.strict_add(16)], seal),
    move |(sender, out, seal)| {
      for _ in 0..2 {
        seal(sender, out);
        open(out);
      }
    },
    |b, (sender, out, seal)| b.iter(|| seal(sender, out)),
  )
}

aws_lc_bench! {
  /// Safe owners of AWS-LC's EVP_HPKE objects for DHKEM(X25519, HKDF-SHA256),
  /// HKDF-SHA256, and AES-128-GCM.
  mod aws {
    use aws_lc_sys as ffi;

    pub(crate) struct Key(*mut ffi::EVP_HPKE_KEY);

    impl Key {
      pub(crate) fn new(secret: &[u8; 32]) -> Self {
        // SAFETY: EVP_HPKE_KEY_new returns an owned key or null, checked below.
        let key = unsafe { ffi::EVP_HPKE_KEY_new() };
        assert!(!key.is_null(), "AWS-LC key allocation");
        // SAFETY: `key` is a live key from EVP_HPKE_KEY_new, and `secret` holds
        // the 32 bytes that the length argument names.
        let ok = unsafe { ffi::EVP_HPKE_KEY_init(key, ffi::EVP_hpke_x25519_hkdf_sha256(), secret.as_ptr(), secret.len()) };
        assert_eq!(ok, 1, "AWS-LC key import");
        Self(key)
      }
    }

    impl Drop for Key {
      fn drop(&mut self) {
        // SAFETY: `self.0` came from EVP_HPKE_KEY_new and is freed one time.
        unsafe { ffi::EVP_HPKE_KEY_free(self.0) };
      }
    }

    pub(crate) struct Context(*mut ffi::EVP_HPKE_CTX);

    impl Context {
      fn new() -> Self {
        // SAFETY: EVP_HPKE_CTX_new returns an owned context or null, checked below.
        let context = unsafe { ffi::EVP_HPKE_CTX_new() };
        assert!(!context.is_null(), "AWS-LC context allocation");
        Self(context)
      }

      /// A sender context whose ephemeral key comes from `seed`.
      pub(crate) fn sender(public_key: &[u8; 32], info: &[u8], seed: &[u8; 32]) -> ([u8; 32], Self) {
        let context = Self::new();
        let mut encapsulation = [0u8; 32];
        let mut length = 0usize;
        // SAFETY: the context is live; every pointer names a buffer of the length
        // passed with it, and `encapsulation` has room for the 32-byte X25519 enc.
        let ok = unsafe {
          ffi::EVP_HPKE_CTX_setup_sender_with_seed_for_testing(
            context.0,
            encapsulation.as_mut_ptr(),
            &raw mut length,
            encapsulation.len(),
            ffi::EVP_hpke_x25519_hkdf_sha256(),
            ffi::EVP_hpke_hkdf_sha256(),
            ffi::EVP_hpke_aes_128_gcm(),
            public_key.as_ptr(),
            public_key.len(),
            info.as_ptr(),
            info.len(),
            seed.as_ptr(),
            seed.len(),
          )
        };
        assert!(ok == 1 && length == 32, "AWS-LC sender setup");
        (encapsulation, context)
      }

      pub(crate) fn recipient(key: &Key, encapsulation: &[u8; 32], info: &[u8]) -> Self {
        let context = Self::new();
        // SAFETY: the context and key are live, and each pointer names a buffer
        // of the length passed with it.
        let ok = unsafe {
          ffi::EVP_HPKE_CTX_setup_recipient(
            context.0,
            key.0,
            ffi::EVP_hpke_hkdf_sha256(),
            ffi::EVP_hpke_aes_128_gcm(),
            encapsulation.as_ptr(),
            encapsulation.len(),
            info.as_ptr(),
            info.len(),
          )
        };
        assert_eq!(ok, 1, "AWS-LC recipient setup");
        context
      }

      pub(crate) fn seal(&mut self, aad: &[u8], plaintext: &[u8], out: &mut [u8]) {
        let mut length = 0usize;
        // SAFETY: the context is live; `out` has room for the plaintext and the
        // 16-byte tag, and each input pointer names a buffer of its length.
        let ok = unsafe {
          ffi::EVP_HPKE_CTX_seal(
            self.0,
            out.as_mut_ptr(),
            &raw mut length,
            out.len(),
            plaintext.as_ptr(),
            plaintext.len(),
            aad.as_ptr(),
            aad.len(),
          )
        };
        assert!(ok == 1 && length == out.len(), "AWS-LC seal");
      }

      pub(crate) fn open(&mut self, aad: &[u8], ciphertext: &[u8], out: &mut [u8]) -> bool {
        let mut length = 0usize;
        // SAFETY: the context is live; `out` has room for the plaintext, and each
        // input pointer names a buffer of its length.
        let ok = unsafe {
          ffi::EVP_HPKE_CTX_open(
            self.0,
            out.as_mut_ptr(),
            &raw mut length,
            out.len(),
            ciphertext.as_ptr(),
            ciphertext.len(),
            aad.as_ptr(),
            aad.len(),
          )
        };
        ok == 1 && length == out.len()
      }

      pub(crate) fn export(&self, context: &[u8], out: &mut [u8]) {
        // SAFETY: the context is live, and each pointer names a buffer of its length.
        let ok = unsafe { ffi::EVP_HPKE_CTX_export(self.0, out.as_mut_ptr(), out.len(), context.as_ptr(), context.len()) };
        assert_eq!(ok, 1, "AWS-LC export");
      }
    }

    impl Drop for Context {
      fn drop(&mut self) {
        // SAFETY: `self.0` came from EVP_HPKE_CTX_new and is freed one time.
        unsafe { ffi::EVP_HPKE_CTX_free(self.0) };
      }
    }
  }
}

fn x25519(c: &mut Criterion) {
  if !bench_config::selected("hpke-x25519-aes128gcm/") {
    return;
  }
  let recipient = X25519Suite::derive_keypair(b"rscrypto HPKE benchmark recipient");
  let public_key = recipient.public_key();
  let (oracle_secret, oracle_public) =
    hpke::kem::X25519HkdfSha256::derive_keypair(b"rscrypto HPKE benchmark recipient");
  let (encapsulation, _) = HpkeSender::<X25519Suite>::setup(&public_key, INFO, filled(&SEED)).expect("setup");
  let recipient_encapsulation = encapsulation;
  let oracle_encapsulation =
    <hpke::kem::X25519HkdfSha256 as hpke::Kem>::EncappedKey::from_bytes(encapsulation.as_bytes()).expect("import");
  // The exporter of each library's receiver must agree with rscrypto's sender.
  let expected_export = {
    let (_, sender) = HpkeSender::<X25519Suite>::setup(&public_key, INFO, filled(&SEED)).expect("setup");
    let mut out = [0u8; 32];
    sender.export(b"export", &mut out).expect("export");
    out
  };

  // Encapsulate to the recipient and run the key schedule.
  let mut group = c.benchmark_group("hpke-x25519-aes128gcm/setup-sender/derand");
  group.bench_function(
    "rscrypto",
    checked(
      || HpkeSender::<X25519Suite>::setup(black_box(&public_key), INFO, filled(black_box(&SEED))).expect("setup"),
      |(encapsulation, sender)| {
        let mut out = [0u8; 32];
        sender.export(b"export", &mut out).expect("export");
        assert_eq!(out, expected_export);
        assert_eq!(encapsulation.as_bytes(), recipient_encapsulation.as_bytes());
      },
    ),
  );
  group.bench_function(
    "rust-hpke",
    checked(
      || {
        hpke::setup_sender_with_rng::<hpke::aead::AesGcm128, OracleKdf, hpke::kem::X25519HkdfSha256>(
          &OpModeS::Base,
          black_box(&oracle_public),
          INFO,
          &mut rng(),
        )
        .expect("setup")
      },
      |(encapsulation, mut sender)| {
        let ciphertext = sender.seal(b"message", AAD).expect("seal");
        let encapsulation = rscrypto::X25519PublicKey::from_bytes(encapsulation.to_bytes().into());
        let mut receiver = HpkeReceiver::<X25519Suite>::setup(&encapsulation, &recipient, INFO).expect("setup");
        assert_eq!(receiver.open_to_vec(AAD, &ciphertext).expect("open"), b"message");
      },
    ),
  );
  aws_lc_bench! {
    group.bench_function(
      "aws-lc",
      checked(
        || aws::Context::sender(black_box(public_key.as_bytes()), INFO, black_box(&[0x5a; 32])),
        |(encapsulation, mut sender)| {
          let mut ciphertext = [0u8; 7 + 16];
          sender.seal(AAD, b"message", &mut ciphertext);
          let encapsulation = rscrypto::X25519PublicKey::from_bytes(encapsulation);
          let mut receiver = HpkeReceiver::<X25519Suite>::setup(&encapsulation, &recipient, INFO).expect("setup");
          assert_eq!(receiver.open_to_vec(AAD, &ciphertext).expect("open"), b"message");
        },
      ),
    );
  }
  group.finish();

  // Decapsulate with a key imported before timing and run the key schedule.
  let mut group = c.benchmark_group("hpke-x25519-aes128gcm/setup-receiver/reuse-key");
  group.bench_function(
    "rscrypto",
    checked(
      || HpkeReceiver::<X25519Suite>::setup(black_box(&encapsulation), black_box(&recipient), INFO).expect("setup"),
      |receiver| {
        let mut out = [0u8; 32];
        receiver.export(b"export", &mut out).expect("export");
        assert_eq!(out, expected_export);
      },
    ),
  );
  group.bench_function(
    "rust-hpke",
    checked(
      || {
        hpke::setup_receiver::<hpke::aead::AesGcm128, OracleKdf, hpke::kem::X25519HkdfSha256>(
          &OpModeR::Base,
          black_box(&oracle_secret),
          black_box(&oracle_encapsulation),
          INFO,
        )
        .expect("setup")
      },
      |receiver| {
        let mut out = [0u8; 32];
        receiver.export(b"export", &mut out).expect("export");
        assert_eq!(out, expected_export);
      },
    ),
  );
  aws_lc_bench! {
    let aws_key = aws::Key::new(recipient.secret_key().as_bytes());
    group.bench_function(
      "aws-lc",
      checked(
        || aws::Context::recipient(black_box(&aws_key), black_box(encapsulation.as_bytes()), INFO),
        |receiver| {
          let mut out = [0u8; 32];
          receiver.export(b"export", &mut out);
          assert_eq!(out, expected_export);
        },
      ),
    );
  }
  group.finish();

  // Export 32 bytes for a short exporter context from an established context.
  let mut group = c.benchmark_group("hpke-x25519-aes128gcm/export/32");
  let receiver = HpkeReceiver::<X25519Suite>::setup(&encapsulation, &recipient, INFO).expect("setup");
  group.bench_function(
    "rscrypto",
    checked(
      || {
        let mut out = [0u8; 32];
        receiver.export(black_box(b"export"), &mut out).expect("export");
        out
      },
      |out| assert_eq!(out, expected_export),
    ),
  );
  let oracle_receiver = hpke::setup_receiver::<hpke::aead::AesGcm128, OracleKdf, hpke::kem::X25519HkdfSha256>(
    &OpModeR::Base,
    &oracle_secret,
    &oracle_encapsulation,
    INFO,
  )
  .expect("setup");
  group.bench_function(
    "rust-hpke",
    checked(
      || {
        let mut out = [0u8; 32];
        oracle_receiver.export(black_box(b"export"), &mut out).expect("export");
        out
      },
      |out| assert_eq!(out, expected_export),
    ),
  );
  aws_lc_bench! {
    let aws_key = aws::Key::new(recipient.secret_key().as_bytes());
    let aws_receiver = aws::Context::recipient(&aws_key, encapsulation.as_bytes(), INFO);
    group.bench_function(
      "aws-lc",
      checked(
        || {
          let mut out = [0u8; 32];
          aws_receiver.export(black_box(b"export"), &mut out);
          out
        },
        |out| assert_eq!(out, expected_export),
      ),
    );
  }
  group.finish();

  for length in LENGTHS {
    let plaintext = vec![0xa5u8; length];
    let (_, mut reference) = HpkeSender::<X25519Suite>::setup(&public_key, INFO, filled(&SEED)).expect("setup");
    let ciphertext = reference.seal_to_vec(AAD, &plaintext).expect("seal");

    // Single-shot Seal: set up a sender and seal one message.
    let mut group = c.benchmark_group(format!("hpke-x25519-aes128gcm/single-shot-seal/{length}"));
    group.bench_function(
      "rscrypto",
      checked_with(
        vec![0u8; length.strict_add(16)],
        |out| {
          X25519Suite::seal(
            black_box(&public_key),
            INFO,
            AAD,
            black_box(&plaintext),
            out,
            filled(&SEED),
          )
          .expect("seal")
        },
        |_, sealed| assert_eq!(sealed, &ciphertext),
      ),
    );
    group.bench_function(
      "rust-hpke",
      checked(
        || {
          hpke::single_shot_seal_with_rng::<hpke::aead::AesGcm128, OracleKdf, hpke::kem::X25519HkdfSha256>(
            &OpModeS::Base,
            black_box(&oracle_public),
            INFO,
            black_box(&plaintext),
            AAD,
            &mut rng(),
          )
          .expect("seal")
        },
        |(encapsulation, sealed)| {
          let encapsulation = rscrypto::X25519PublicKey::from_bytes(encapsulation.to_bytes().into());
          let mut opened = vec![0u8; length];
          X25519Suite::open(&encapsulation, &recipient, INFO, AAD, &sealed, &mut opened).expect("open");
          assert_eq!(opened, plaintext);
        },
      ),
    );
    aws_lc_bench! {
      group.bench_function(
        "aws-lc",
        checked_with(
          vec![0u8; length.strict_add(16)],
          |out| {
            let (encapsulation, mut sender) = aws::Context::sender(black_box(public_key.as_bytes()), INFO, &[0x5a; 32]);
            sender.seal(AAD, black_box(&plaintext), out);
            encapsulation
          },
          |encapsulation, sealed| {
            let encapsulation = rscrypto::X25519PublicKey::from_bytes(encapsulation);
            let mut opened = vec![0u8; length];
            X25519Suite::open(&encapsulation, &recipient, INFO, AAD, sealed, &mut opened).expect("open");
            assert_eq!(opened, plaintext);
          },
        ),
      );
    }
    group.finish();

    // Single-shot Open: set up a receiver and open one message.
    let mut group = c.benchmark_group(format!("hpke-x25519-aes128gcm/single-shot-open/{length}"));
    group.bench_function(
      "rscrypto",
      checked_with(
        vec![0u8; length],
        |out| {
          X25519Suite::open(
            black_box(&encapsulation),
            black_box(&recipient),
            INFO,
            AAD,
            black_box(&ciphertext),
            out,
          )
          .expect("open")
        },
        |(), opened| assert_eq!(opened, &plaintext),
      ),
    );
    group.bench_function(
      "rust-hpke",
      checked(
        || {
          hpke::single_shot_open::<hpke::aead::AesGcm128, OracleKdf, hpke::kem::X25519HkdfSha256>(
            &OpModeR::Base,
            black_box(&oracle_secret),
            black_box(&oracle_encapsulation),
            INFO,
            black_box(&ciphertext),
            AAD,
          )
          .expect("open")
        },
        |opened| assert_eq!(opened, plaintext),
      ),
    );
    aws_lc_bench! {
      let aws_key = aws::Key::new(recipient.secret_key().as_bytes());
      group.bench_function(
        "aws-lc",
        checked_with(
          vec![0u8; length],
          |out| aws::Context::recipient(black_box(&aws_key), encapsulation.as_bytes(), INFO).open(AAD, black_box(&ciphertext), out),
          |opened, out| {
            assert!(opened);
            assert_eq!(out, &plaintext);
          },
        ),
      );
    }
    group.finish();
  }

  // Context Seal: seal one short message of a stream with an established sender context.
  // Each row's messages open in order with another implementation's receiver.
  for length in [16, 64] {
    let plaintext = vec![0xa5u8; length];
    let opens_in_rscrypto = |encapsulation: [u8; 32]| {
      let mut receiver =
        HpkeReceiver::<X25519Suite>::setup(&rscrypto::X25519PublicKey::from_bytes(encapsulation), &recipient, INFO)
          .expect("setup");
      let plaintext = &plaintext;
      move |sealed: &[u8]| assert_eq!(&receiver.open_to_vec(AAD, sealed).expect("open"), plaintext)
    };
    let mut group = c.benchmark_group(format!("hpke-x25519-aes128gcm/context-seal/{length}"));
    let (_, sender) = HpkeSender::<X25519Suite>::setup(&public_key, INFO, filled(&SEED)).expect("setup");
    let mut oracle = hpke::setup_receiver::<hpke::aead::AesGcm128, OracleKdf, hpke::kem::X25519HkdfSha256>(
      &OpModeR::Base,
      &oracle_secret,
      &oracle_encapsulation,
      INFO,
    )
    .expect("setup");
    group.bench_function(
      "rscrypto",
      context_seal(
        sender,
        length,
        |sender, out| sender.seal(AAD, black_box(&plaintext), out).expect("seal"),
        |sealed| assert_eq!(oracle.open(sealed, AAD).expect("open"), plaintext),
      ),
    );
    let (encapsulation, sender) = hpke::setup_sender_with_rng::<
      hpke::aead::AesGcm128,
      OracleKdf,
      hpke::kem::X25519HkdfSha256,
    >(&OpModeS::Base, &oracle_public, INFO, &mut rng())
    .expect("setup");
    group.bench_function(
      "rust-hpke",
      context_seal(
        sender,
        length,
        |sender, out| {
          let (ciphertext, tag) = out.split_at_mut(length);
          ciphertext.copy_from_slice(black_box(&plaintext));
          tag.copy_from_slice(
            &sender
              .seal_inout_detached(ciphertext.into(), AAD)
              .expect("seal")
              .to_bytes(),
          );
        },
        opens_in_rscrypto(encapsulation.to_bytes().into()),
      ),
    );
    aws_lc_bench! {
      let (encapsulation, sender) = aws::Context::sender(public_key.as_bytes(), INFO, &[0x5a; 32]);
      group.bench_function(
        "aws-lc",
        context_seal(
          sender,
          length,
          |sender, out| sender.seal(AAD, black_box(&plaintext), out),
          opens_in_rscrypto(encapsulation),
        ),
      );
    }
    group.finish();
  }
}

fn xwing(c: &mut Criterion) {
  if !bench_config::selected("hpke-xwing-chacha20poly1305/") {
    return;
  }
  let recipient = XWingSuite::derive_keypair(b"rscrypto HPKE benchmark recipient");
  let public_key = recipient.encapsulation_key().prepare();
  let (oracle_secret, oracle_public) = hpke::kem::XWing::derive_keypair(b"rscrypto HPKE benchmark recipient");
  let (encapsulation, _) = HpkeSender::<XWingSuite>::setup(&public_key, INFO, filled(&SEED)).expect("setup");
  let oracle_encapsulation =
    <hpke::kem::XWing as hpke::Kem>::EncappedKey::from_bytes(encapsulation.as_bytes()).expect("import");
  let expected_export = {
    let (_, sender) = HpkeSender::<XWingSuite>::setup(&public_key, INFO, filled(&SEED)).expect("setup");
    let mut out = [0u8; 32];
    sender.export(b"export", &mut out).expect("export");
    out
  };

  let mut group = c.benchmark_group("hpke-xwing-chacha20poly1305/setup-sender/derand");
  group.bench_function(
    "rscrypto",
    checked(
      || HpkeSender::<XWingSuite>::setup(black_box(&public_key), INFO, filled(black_box(&SEED))).expect("setup"),
      |(_, sender)| {
        let mut out = [0u8; 32];
        sender.export(b"export", &mut out).expect("export");
        assert_eq!(out, expected_export);
      },
    ),
  );
  group.bench_function(
    "rust-hpke",
    checked(
      || {
        hpke::setup_sender_with_rng::<hpke::aead::ChaCha20Poly1305, OracleKdf, hpke::kem::XWing>(
          &OpModeS::Base,
          black_box(&oracle_public),
          INFO,
          &mut rng(),
        )
        .expect("setup")
      },
      |(encapsulation, mut sender)| {
        let ciphertext = sender.seal(b"message", AAD).expect("seal");
        let encapsulation = rscrypto::XWingCiphertext::try_from_slice(&encapsulation.to_bytes()).expect("import");
        let mut receiver = HpkeReceiver::<XWingSuite>::setup(&encapsulation, &recipient, INFO).expect("setup");
        assert_eq!(receiver.open_to_vec(AAD, &ciphertext).expect("open"), b"message");
      },
    ),
  );
  group.finish();

  let mut group = c.benchmark_group("hpke-xwing-chacha20poly1305/setup-receiver/reuse-key");
  group.bench_function(
    "rscrypto",
    checked(
      || HpkeReceiver::<XWingSuite>::setup(black_box(&encapsulation), black_box(&recipient), INFO).expect("setup"),
      |receiver| {
        let mut out = [0u8; 32];
        receiver.export(b"export", &mut out).expect("export");
        assert_eq!(out, expected_export);
      },
    ),
  );
  group.bench_function(
    "rust-hpke",
    checked(
      || {
        hpke::setup_receiver::<hpke::aead::ChaCha20Poly1305, OracleKdf, hpke::kem::XWing>(
          &OpModeR::Base,
          black_box(&oracle_secret),
          black_box(&oracle_encapsulation),
          INFO,
        )
        .expect("setup")
      },
      |receiver| {
        let mut out = [0u8; 32];
        receiver.export(b"export", &mut out).expect("export");
        assert_eq!(out, expected_export);
      },
    ),
  );
  group.finish();

  // Export 32 bytes for a short exporter context from an established context.
  let mut group = c.benchmark_group("hpke-xwing-chacha20poly1305/export/32");
  let receiver = HpkeReceiver::<XWingSuite>::setup(&encapsulation, &recipient, INFO).expect("setup");
  group.bench_function(
    "rscrypto",
    checked(
      || {
        let mut out = [0u8; 32];
        receiver.export(black_box(b"export"), &mut out).expect("export");
        out
      },
      |out| assert_eq!(out, expected_export),
    ),
  );
  let oracle_receiver = hpke::setup_receiver::<hpke::aead::ChaCha20Poly1305, OracleKdf, hpke::kem::XWing>(
    &OpModeR::Base,
    &oracle_secret,
    &oracle_encapsulation,
    INFO,
  )
  .expect("setup");
  group.bench_function(
    "rust-hpke",
    checked(
      || {
        let mut out = [0u8; 32];
        oracle_receiver.export(black_box(b"export"), &mut out).expect("export");
        out
      },
      |out| assert_eq!(out, expected_export),
    ),
  );
  group.finish();

  for length in LENGTHS {
    let plaintext = vec![0xa5u8; length];
    let (_, mut reference) = HpkeSender::<XWingSuite>::setup(&public_key, INFO, filled(&SEED)).expect("setup");
    let ciphertext = reference.seal_to_vec(AAD, &plaintext).expect("seal");

    let mut group = c.benchmark_group(format!("hpke-xwing-chacha20poly1305/single-shot-seal/{length}"));
    group.bench_function(
      "rscrypto",
      checked_with(
        vec![0u8; length.strict_add(16)],
        |out| {
          XWingSuite::seal(
            black_box(&public_key),
            INFO,
            AAD,
            black_box(&plaintext),
            out,
            filled(&SEED),
          )
          .expect("seal")
        },
        |_, sealed| assert_eq!(sealed, &ciphertext),
      ),
    );
    group.bench_function(
      "rust-hpke",
      checked(
        || {
          hpke::single_shot_seal_with_rng::<hpke::aead::ChaCha20Poly1305, OracleKdf, hpke::kem::XWing>(
            &OpModeS::Base,
            black_box(&oracle_public),
            INFO,
            black_box(&plaintext),
            AAD,
            &mut rng(),
          )
          .expect("seal")
        },
        |(encapsulation, sealed)| {
          let encapsulation = rscrypto::XWingCiphertext::try_from_slice(&encapsulation.to_bytes()).expect("import");
          let mut opened = vec![0u8; length];
          XWingSuite::open(&encapsulation, &recipient, INFO, AAD, &sealed, &mut opened).expect("open");
          assert_eq!(opened, plaintext);
        },
      ),
    );
    group.finish();

    let mut group = c.benchmark_group(format!("hpke-xwing-chacha20poly1305/single-shot-open/{length}"));
    group.bench_function(
      "rscrypto",
      checked_with(
        vec![0u8; length],
        |out| {
          XWingSuite::open(
            black_box(&encapsulation),
            black_box(&recipient),
            INFO,
            AAD,
            black_box(&ciphertext),
            out,
          )
          .expect("open")
        },
        |(), opened| assert_eq!(opened, &plaintext),
      ),
    );
    group.bench_function(
      "rust-hpke",
      checked(
        || {
          hpke::single_shot_open::<hpke::aead::ChaCha20Poly1305, OracleKdf, hpke::kem::XWing>(
            &OpModeR::Base,
            black_box(&oracle_secret),
            black_box(&oracle_encapsulation),
            INFO,
            black_box(&ciphertext),
            AAD,
          )
          .expect("open")
        },
        |opened| assert_eq!(opened, plaintext),
      ),
    );
    group.finish();
  }
}

fn main() {
  std::thread::Builder::new()
    .stack_size(8 << 20)
    .spawn(|| bench_config::run(&[x25519, xwing]))
    .expect("spawn benchmark thread")
    .join()
    .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
}
