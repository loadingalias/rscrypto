//! Linked release artifact for secret-worker frame review.
//!
//! `scripts/stack/frames.py` reads the call graph and the compiler's
//! `.stack_sizes` records from this binary. Each public operation runs once so
//! every secret worker it reaches is linked exactly as an ordinary
//! caller would link it. Inputs pass through `black_box` so constant folding
//! cannot remove or specialize those paths.

extern crate alloc;

use alloc::alloc::Global;
use core::hint::black_box;

use rscrypto::aead::{Nonce96, OpenError, SealError, expert::AeadWithNonce};
use rscrypto::{
  Aead, Aes128Gcm, Aes128GcmKey, Aes256Gcm, Aes256GcmKey, ChaCha20Poly1305, ChaCha20Poly1305Key, Kem, MlDsa44,
  MlDsa44PreparedSecretKey, MlDsa44PreparedSecretKeyStorage, MlDsa44PublicKey, MlDsa44SecretKey, MlDsa44Signature,
  MlDsa65, MlDsa65PreparedSecretKey, MlDsa65PreparedSecretKeyStorage, MlDsa65PublicKey, MlDsa65SecretKey,
  MlDsa65Signature, MlDsa87, MlDsa87PreparedSecretKey, MlDsa87PreparedSecretKeyStorage, MlDsa87PublicKey,
  MlDsa87SecretKey, MlDsa87Signature, MlDsaError, MlKem512, MlKem512PreparedDecapsulationKey, MlKem768,
  MlKem768PreparedDecapsulationKey, MlKem1024, MlKem1024PreparedDecapsulationKey, MlKemError, VerificationError,
};

// These wrappers preserve the actual preparation return type and keep an
// inspectable call boundary after LTO, including borrowed preparation.
macro_rules! preparation_boundaries {
  ($owned:ident, $borrowed:ident, $owned_lifecycle:ident, $borrowed_lifecycle:ident, $secret:ty, $storage:ty, $prepared:ident) => {
    #[inline(never)]
    fn $owned(key: &$secret) -> Result<$prepared<Box<$storage>>, MlDsaError> {
      key.prepare_in(Global)
    }

    #[inline(never)]
    fn $borrowed<'a>(key: &$secret, storage: &'a mut $storage) -> Result<$prepared<&'a mut $storage>, MlDsaError> {
      key.prepare(storage)
    }

    // Include both successful destruction and preparation's recoverable error
    // cleanup in the reviewed call graph. Returning the prepared key alone
    // cannot establish the depth of its later allocator deallocation callback.
    #[inline(never)]
    fn $owned_lifecycle(key: &$secret) -> Result<(), MlDsaError> {
      let prepared = $owned(key)?;
      black_box(&prepared);
      drop(prepared);
      Ok(())
    }

    #[inline(never)]
    fn $borrowed_lifecycle(key: &$secret, storage: &mut $storage) -> Result<(), MlDsaError> {
      let prepared = $borrowed(key, storage)?;
      black_box(&prepared);
      drop(prepared);
      Ok(())
    }
  };
}

preparation_boundaries!(
  prepare_44_in,
  prepare_44_borrowed,
  prepare_44_owned_lifecycle,
  prepare_44_borrowed_lifecycle,
  MlDsa44SecretKey,
  MlDsa44PreparedSecretKeyStorage,
  MlDsa44PreparedSecretKey
);
preparation_boundaries!(
  prepare_65_in,
  prepare_65_borrowed,
  prepare_65_owned_lifecycle,
  prepare_65_borrowed_lifecycle,
  MlDsa65SecretKey,
  MlDsa65PreparedSecretKeyStorage,
  MlDsa65PreparedSecretKey
);
preparation_boundaries!(
  prepare_87_in,
  prepare_87_borrowed,
  prepare_87_owned_lifecycle,
  prepare_87_borrowed_lifecycle,
  MlDsa87SecretKey,
  MlDsa87PreparedSecretKeyStorage,
  MlDsa87PreparedSecretKey
);

// Signing, verification, ML-KEM, and AEAD record boundaries for whole-call
// review. Each wraps one public operation with its production arguments.
macro_rules! signing_boundaries {
  (
    $deterministic:ident, $prepared_sign:ident, $verify:ident,
    $secret:ty, $public:ty, $signature:ty, $storage:ty, $prepared:ident
  ) => {
    #[inline(never)]
    fn $deterministic(key: &$secret, message: &[u8]) -> Result<$signature, MlDsaError> {
      key.sign_deterministic(message, b"")
    }

    #[inline(never)]
    fn $prepared_sign(key: &$prepared<&mut $storage>, message: &[u8]) -> Result<$signature, MlDsaError> {
      key.sign_with(message, b"", |out| {
        out.fill(black_box(3));
        Ok(())
      })
    }

    #[inline(never)]
    fn $verify(key: &$public, message: &[u8], signature: &$signature) -> Result<(), VerificationError> {
      key.verify_with_context(message, b"", signature)
    }
  };
}

signing_boundaries!(
  sign_44_deterministic,
  sign_44_prepared,
  verify_44,
  MlDsa44SecretKey,
  MlDsa44PublicKey,
  MlDsa44Signature,
  MlDsa44PreparedSecretKeyStorage,
  MlDsa44PreparedSecretKey
);
signing_boundaries!(
  sign_65_deterministic,
  sign_65_prepared,
  verify_65,
  MlDsa65SecretKey,
  MlDsa65PublicKey,
  MlDsa65Signature,
  MlDsa65PreparedSecretKeyStorage,
  MlDsa65PreparedSecretKey
);
signing_boundaries!(
  sign_87_deterministic,
  sign_87_prepared,
  verify_87,
  MlDsa87SecretKey,
  MlDsa87PublicKey,
  MlDsa87Signature,
  MlDsa87PreparedSecretKeyStorage,
  MlDsa87PreparedSecretKey
);

macro_rules! kem_boundaries {
  ($profile:ty, $prepared_key:ty, $generate:ident, $encapsulate:ident, $decapsulate:ident, $prepared_decapsulate:ident) => {
    #[inline(never)]
    fn $generate() -> Result<(<$profile as Kem>::EncapsulationKey, <$profile as Kem>::DecapsulationKey), MlKemError> {
      <$profile>::generate_keypair(|out: &mut [u8]| {
        out.fill(black_box(4));
        Ok(())
      })
    }

    #[inline(never)]
    fn $encapsulate(
      key: &<$profile as Kem>::EncapsulationKey,
    ) -> Result<(<$profile as Kem>::Ciphertext, <$profile as Kem>::SharedSecret), MlKemError> {
      <$profile>::encapsulate(key, |out: &mut [u8]| {
        out.fill(black_box(5));
        Ok(())
      })
    }

    #[inline(never)]
    fn $decapsulate(
      key: &<$profile as Kem>::DecapsulationKey,
      ciphertext: &<$profile as Kem>::Ciphertext,
    ) -> Result<<$profile as Kem>::SharedSecret, MlKemError> {
      <$profile>::decapsulate(key, ciphertext)
    }

    #[inline(never)]
    fn $prepared_decapsulate(
      key: &$prepared_key,
      ciphertext: &<$profile as Kem>::Ciphertext,
    ) -> Result<<$profile as Kem>::SharedSecret, MlKemError> {
      key.decapsulate(ciphertext)
    }
  };
}

kem_boundaries!(
  MlKem512,
  MlKem512PreparedDecapsulationKey,
  mlkem_512_generate,
  mlkem_512_encapsulate,
  mlkem_512_decapsulate,
  mlkem_512_prepared_decapsulate
);
kem_boundaries!(
  MlKem768,
  MlKem768PreparedDecapsulationKey,
  mlkem_768_generate,
  mlkem_768_encapsulate,
  mlkem_768_decapsulate,
  mlkem_768_prepared_decapsulate
);
kem_boundaries!(
  MlKem1024,
  MlKem1024PreparedDecapsulationKey,
  mlkem_1024_generate,
  mlkem_1024_encapsulate,
  mlkem_1024_decapsulate,
  mlkem_1024_prepared_decapsulate
);

macro_rules! record_boundaries {
  ($cipher:ty, $seal:ident, $open:ident) => {
    #[inline(never)]
    fn $seal(
      cipher: &$cipher,
      nonce: &<$cipher as Aead>::Nonce,
      aad: &[u8],
      record: &mut [u8],
    ) -> Result<<$cipher as Aead>::Tag, SealError> {
      cipher.encrypt_in_place(nonce, aad, record)
    }

    #[inline(never)]
    fn $open(
      cipher: &$cipher,
      nonce: &<$cipher as Aead>::Nonce,
      aad: &[u8],
      record: &mut [u8],
      tag: &<$cipher as Aead>::Tag,
    ) -> Result<(), OpenError> {
      cipher.decrypt_in_place(nonce, aad, record, tag)
    }
  };
}

record_boundaries!(Aes128Gcm, seal_aes128gcm, open_aes128gcm);
record_boundaries!(Aes256Gcm, seal_aes256gcm, open_aes256gcm);
record_boundaries!(ChaCha20Poly1305, seal_chacha20poly1305, open_chacha20poly1305);

macro_rules! mldsa {
  (
    $profile:ty, $storage:ty, $owned:ident, $borrowed:ident, $owned_lifecycle:ident, $borrowed_lifecycle:ident,
    $deterministic:ident, $prepared_sign:ident, $verify:ident, $seed:expr
  ) => {{
    let (public, secret) = <$profile>::keypair_from_seed(black_box($seed)).expect("ML-DSA key generation");
    let message = black_box(&b"frame review"[..]);
    let signature = $deterministic(&secret, message).expect("ML-DSA signing");
    $verify(&public, message, &signature).expect("ML-DSA verification");
    let mut storage = Box::new(<$storage>::new());
    let prepared = $borrowed(&secret, &mut storage).expect("ML-DSA preparation");
    let hedged = $prepared_sign(&prepared, message).expect("ML-DSA prepared signing");
    $verify(&public, message, &hedged).expect("ML-DSA verification");
    drop(prepared);
    $borrowed_lifecycle(&secret, &mut storage).expect("ML-DSA borrowed preparation lifecycle");
    $owned_lifecycle(&secret).expect("ML-DSA owned preparation lifecycle");
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
  ($profile:ty, $generate:ident, $encapsulate:ident, $decapsulate:ident, $prepared_decapsulate:ident) => {{
    let (encapsulation_key, decapsulation_key) = $generate().expect("ML-KEM key generation");
    let (ciphertext, shared) = $encapsulate(&encapsulation_key).expect("ML-KEM encapsulation");
    let decapsulated = $decapsulate(&decapsulation_key, &ciphertext).expect("ML-KEM decapsulation");
    assert!(shared.ct_eq(&decapsulated).declassify());
    let prepared = <$profile>::prepare_decapsulation_key(&decapsulation_key).expect("ML-KEM preparation");
    let again = $prepared_decapsulate(&prepared, &ciphertext).expect("ML-KEM prepared decapsulation");
    assert!(shared.ct_eq(&again).declassify());
  }};
}

// Lengths pass through `black_box` so every length-selected path stays linked:
// the VAES message tails, the AVX-512 one-pass open, the row pass, and the column batches.
macro_rules! records {
  ($cipher:ty, $key:expr, $seal:ident, $open:ident, $byte:expr) => {{
    let cipher = <$cipher>::new(&$key);
    let nonce = Nonce96::from_bytes(black_box([$byte; 12]));
    for len in [black_box(129), black_box(1029)] {
      let mut data = vec![37u8; len];
      let tag = $seal(&cipher, &nonce, black_box(b"frame review"), &mut data).expect("AEAD seal");
      black_box(&data);
      $open(&cipher, &nonce, black_box(b"frame review"), &mut data, &tag).expect("AEAD open");
      black_box(data);
    }
  }};
}

fn main() {
  records!(
    Aes128Gcm,
    Aes128GcmKey::generate(|out| out.fill(black_box(7))),
    seal_aes128gcm,
    open_aes128gcm,
    7
  );
  records!(
    Aes256Gcm,
    Aes256GcmKey::generate(|out| out.fill(black_box(8))),
    seal_aes256gcm,
    open_aes256gcm,
    8
  );
  records!(
    ChaCha20Poly1305,
    ChaCha20Poly1305Key::from_bytes(black_box([9; 32])),
    seal_chacha20poly1305,
    open_chacha20poly1305,
    9
  );
  mldsa!(
    MlDsa44,
    MlDsa44PreparedSecretKeyStorage,
    prepare_44_in,
    prepare_44_borrowed,
    prepare_44_owned_lifecycle,
    prepare_44_borrowed_lifecycle,
    sign_44_deterministic,
    sign_44_prepared,
    verify_44,
    &[1; 32]
  );
  mldsa!(
    MlDsa65,
    MlDsa65PreparedSecretKeyStorage,
    prepare_65_in,
    prepare_65_borrowed,
    prepare_65_owned_lifecycle,
    prepare_65_borrowed_lifecycle,
    sign_65_deterministic,
    sign_65_prepared,
    verify_65,
    &[2; 32]
  );
  mldsa!(
    MlDsa87,
    MlDsa87PreparedSecretKeyStorage,
    prepare_87_in,
    prepare_87_borrowed,
    prepare_87_owned_lifecycle,
    prepare_87_borrowed_lifecycle,
    sign_87_deterministic,
    sign_87_prepared,
    verify_87,
    &[3; 32]
  );
  mlkem!(
    MlKem512,
    mlkem_512_generate,
    mlkem_512_encapsulate,
    mlkem_512_decapsulate,
    mlkem_512_prepared_decapsulate
  );
  mlkem!(
    MlKem768,
    mlkem_768_generate,
    mlkem_768_encapsulate,
    mlkem_768_decapsulate,
    mlkem_768_prepared_decapsulate
  );
  mlkem!(
    MlKem1024,
    mlkem_1024_generate,
    mlkem_1024_encapsulate,
    mlkem_1024_decapsulate,
    mlkem_1024_prepared_decapsulate
  );
}
