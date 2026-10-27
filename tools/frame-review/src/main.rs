//! Linked release artifact for secret-worker frame review.
//!
//! `scripts/stack/frames.py` reads the call graph and the compiler's
//! `.stack_sizes` records from this binary. Each public operation runs once so
//! every secret-hashing worker it reaches is linked exactly as an ordinary
//! caller would link it. Inputs pass through `black_box` so constant folding
//! cannot remove or specialize those paths.

extern crate alloc;

use alloc::alloc::Global;
use core::hint::black_box;

use rscrypto::aead::{Nonce96, expert::AeadWithNonce};
use rscrypto::{
  Aes128Gcm, Aes128GcmKey, Aes256Gcm, Aes256GcmKey, Kem, MlDsa44, MlDsa44PreparedSecretKey,
  MlDsa44PreparedSecretKeyStorage, MlDsa44SecretKey, MlDsa65, MlDsa65PreparedSecretKey,
  MlDsa65PreparedSecretKeyStorage, MlDsa65SecretKey, MlDsa87, MlDsa87PreparedSecretKey,
  MlDsa87PreparedSecretKeyStorage, MlDsa87SecretKey, MlDsaError, MlKem512, MlKem768, MlKem1024,
};

// These wrappers preserve the actual preparation return type and keep an
// inspectable call boundary after LTO, including borrowed preparation.
macro_rules! preparation_boundaries {
  ($owned:ident, $borrowed:ident, $secret:ty, $storage:ty, $prepared:ident) => {
    #[inline(never)]
    fn $owned(key: &$secret) -> Result<$prepared<Box<$storage>>, MlDsaError> {
      key.prepare_in(Global)
    }

    #[inline(never)]
    fn $borrowed<'a>(key: &$secret, storage: &'a mut $storage) -> Result<$prepared<&'a mut $storage>, MlDsaError> {
      key.prepare(storage)
    }
  };
}

preparation_boundaries!(
  prepare_44_in,
  prepare_44_borrowed,
  MlDsa44SecretKey,
  MlDsa44PreparedSecretKeyStorage,
  MlDsa44PreparedSecretKey
);
preparation_boundaries!(
  prepare_65_in,
  prepare_65_borrowed,
  MlDsa65SecretKey,
  MlDsa65PreparedSecretKeyStorage,
  MlDsa65PreparedSecretKey
);
preparation_boundaries!(
  prepare_87_in,
  prepare_87_borrowed,
  MlDsa87SecretKey,
  MlDsa87PreparedSecretKeyStorage,
  MlDsa87PreparedSecretKey
);

macro_rules! mldsa {
  ($profile:ty, $storage:ty, $owned:ident, $borrowed:ident, $seed:expr) => {{
    let (public, secret) = <$profile>::keypair_from_seed(black_box($seed)).expect("ML-DSA key generation");
    let signature = secret
      .sign_deterministic(black_box(b"frame review"), b"")
      .expect("ML-DSA signing");
    public
      .verify_with_context(b"frame review", b"", &signature)
      .expect("ML-DSA verification");
    let mut storage = Box::new(<$storage>::new());
    let prepared = $borrowed(&secret, &mut storage).expect("ML-DSA preparation");
    let hedged = prepared
      .sign_with(b"frame review", b"", |out| {
        out.fill(black_box(3));
        Ok(())
      })
      .expect("ML-DSA prepared signing");
    public
      .verify_with_context(b"frame review", b"", &hedged)
      .expect("ML-DSA verification");
    let allocated = $owned(&secret).expect("ML-DSA allocated preparation");
    let hedged = allocated
      .sign_with(b"frame review", b"", |out| {
        out.fill(black_box(4));
        Ok(())
      })
      .expect("ML-DSA allocated prepared signing");
    public
      .prepare_in(Global)
      .expect("ML-DSA allocated public preparation")
      .verify_with_context(b"frame review", b"", &hedged)
      .expect("ML-DSA prepared verification");
  }};
}

macro_rules! mlkem {
  ($profile:ty, $byte:expr) => {{
    let fill = |out: &mut [u8]| {
      out.fill(black_box($byte));
      Ok(())
    };
    let (encapsulation_key, decapsulation_key) = <$profile>::generate_keypair(fill).expect("ML-KEM key generation");
    let (ciphertext, shared) = <$profile>::encapsulate(&encapsulation_key, fill).expect("ML-KEM encapsulation");
    let decapsulated = <$profile>::decapsulate(&decapsulation_key, &ciphertext).expect("ML-KEM decapsulation");
    assert!(shared.ct_eq(&decapsulated).declassify());
    let prepared = <$profile>::prepare_decapsulation_key(&decapsulation_key).expect("ML-KEM preparation");
    let again = prepared
      .decapsulate(&ciphertext)
      .expect("ML-KEM prepared decapsulation");
    assert!(shared.ct_eq(&again).declassify());
  }};
}

macro_rules! gcm {
  ($cipher:ty, $key:ty, $byte:expr) => {{
    let key = <$key>::generate(|out| out.fill(black_box($byte)));
    let cipher = <$cipher>::new(&key);
    let nonce = Nonce96::from_bytes(black_box([$byte; 12]));
    let mut data = black_box([37u8; 129]);
    let tag = cipher
      .encrypt_in_place(&nonce, black_box(b"frame review"), &mut data)
      .expect("AES-GCM encryption");
    black_box(&data);
    cipher
      .decrypt_in_place(&nonce, black_box(b"frame review"), &mut data, &tag)
      .expect("AES-GCM decryption");
    black_box(data);
  }};
}

fn main() {
  gcm!(Aes128Gcm, Aes128GcmKey, 7);
  gcm!(Aes256Gcm, Aes256GcmKey, 8);
  mldsa!(
    MlDsa44,
    MlDsa44PreparedSecretKeyStorage,
    prepare_44_in,
    prepare_44_borrowed,
    &[1; 32]
  );
  mldsa!(
    MlDsa65,
    MlDsa65PreparedSecretKeyStorage,
    prepare_65_in,
    prepare_65_borrowed,
    &[2; 32]
  );
  mldsa!(
    MlDsa87,
    MlDsa87PreparedSecretKeyStorage,
    prepare_87_in,
    prepare_87_borrowed,
    &[3; 32]
  );
  mlkem!(MlKem512, 4);
  mlkem!(MlKem768, 5);
  mlkem!(MlKem1024, 6);
}
