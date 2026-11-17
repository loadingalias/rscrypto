//! Linked release artifact for secret-worker frame review.
//!
//! `scripts/stack/frames.py` reads the call graph and the compiler's
//! `.stack_sizes` and `.llvm_jump_table_sizes` records from this binary. Each
//! public operation runs once so every secret worker it reaches is linked
//! exactly as an ordinary caller would link it. Inputs pass through
//! `black_box` so constant folding cannot remove or specialize those paths.

extern crate alloc;

use alloc::alloc::Global;
use alloc::boxed::Box;
use core::alloc::{AllocError, Allocator, Layout};
use core::cell::UnsafeCell;
use core::hint::black_box;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, Ordering};

use rscrypto::aead::{Nonce96, OpenError, SealError, expert::AeadWithNonce};
use rscrypto::{
  Aead, Aes128Gcm, Aes128GcmKey, Aes256Gcm, Aes256GcmKey, ChaCha20Poly1305, ChaCha20Poly1305Key, CompositeMlDsaError,
  HpkeError, HpkeReceiver, HpkeSender, HpkeSuite, HpkeX25519HkdfSha256Aes128Gcm, HpkeXWingHkdfSha256ChaCha20Poly1305,
  Kem, MlDsa44, MlDsa44PreparedSecretKey, MlDsa44PreparedSecretKeyStorage, MlDsa44PublicKey, MlDsa44SecretKey,
  MlDsa44Signature, MlDsa65, MlDsa65EcdsaP256, MlDsa65EcdsaP256PreparedSecretKey, MlDsa65EcdsaP256PublicKey,
  MlDsa65EcdsaP256SecretKey, MlDsa65EcdsaP256Signature, MlDsa65Ed25519, MlDsa65Ed25519PreparedSecretKey,
  MlDsa65Ed25519PublicKey, MlDsa65Ed25519SecretKey, MlDsa65Ed25519Signature, MlDsa65PreparedSecretKey,
  MlDsa65PreparedSecretKeyStorage, MlDsa65PublicKey, MlDsa65SecretKey, MlDsa65Signature, MlDsa87,
  MlDsa87PreparedSecretKey, MlDsa87PreparedSecretKeyStorage, MlDsa87PublicKey, MlDsa87SecretKey, MlDsa87Signature,
  MlDsaError, MlKem512, MlKem512PreparedDecapsulationKey, MlKem768, MlKem768PreparedDecapsulationKey, MlKem1024,
  MlKem1024PreparedDecapsulationKey, MlKemError, VerificationError, XWing, XWingError, XWingPreparedDecapsulationKey,
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
  (
    $profile:ty,
    $prepared_key:ty,
    $error:ty,
    $generate:ident,
    $encapsulate:ident,
    $decapsulate:ident,
    $prepared_decapsulate:ident
  ) => {
    #[inline(never)]
    fn $generate() -> Result<(<$profile as Kem>::EncapsulationKey, <$profile as Kem>::DecapsulationKey), $error> {
      <$profile>::generate_keypair(|out: &mut [u8]| {
        out.fill(black_box(4));
        Ok(())
      })
    }

    #[inline(never)]
    fn $encapsulate(
      key: &<$profile as Kem>::EncapsulationKey,
    ) -> Result<(<$profile as Kem>::Ciphertext, <$profile as Kem>::SharedSecret), $error> {
      <$profile>::encapsulate(key, |out: &mut [u8]| {
        out.fill(black_box(5));
        Ok(())
      })
    }

    #[inline(never)]
    fn $decapsulate(
      key: &<$profile as Kem>::DecapsulationKey,
      ciphertext: &<$profile as Kem>::Ciphertext,
    ) -> Result<<$profile as Kem>::SharedSecret, $error> {
      <$profile>::decapsulate(key, ciphertext)
    }

    #[inline(never)]
    fn $prepared_decapsulate(
      key: &$prepared_key,
      ciphertext: &<$profile as Kem>::Ciphertext,
    ) -> Result<<$profile as Kem>::SharedSecret, $error> {
      key.decapsulate(ciphertext)
    }
  };
}

kem_boundaries!(
  MlKem512,
  MlKem512PreparedDecapsulationKey,
  MlKemError,
  mlkem_512_generate,
  mlkem_512_encapsulate,
  mlkem_512_decapsulate,
  mlkem_512_prepared_decapsulate
);
kem_boundaries!(
  MlKem768,
  MlKem768PreparedDecapsulationKey,
  MlKemError,
  mlkem_768_generate,
  mlkem_768_encapsulate,
  mlkem_768_decapsulate,
  mlkem_768_prepared_decapsulate
);
kem_boundaries!(
  MlKem1024,
  MlKem1024PreparedDecapsulationKey,
  MlKemError,
  mlkem_1024_generate,
  mlkem_1024_encapsulate,
  mlkem_1024_decapsulate,
  mlkem_1024_prepared_decapsulate
);
kem_boundaries!(
  XWing,
  XWingPreparedDecapsulationKey,
  XWingError,
  xwing_generate,
  xwing_encapsulate,
  xwing_decapsulate,
  xwing_prepared_decapsulate
);

macro_rules! hpke_boundaries {
  ($suite:ty, $derive:ident, $seal:ident, $open:ident) => {
    #[inline(never)]
    fn $derive(ikm: &[u8]) -> <$suite as HpkeSuite>::PrivateKey {
      <$suite>::derive_keypair(ikm)
    }

    #[inline(never)]
    fn $seal(
      public_key: &<$suite as HpkeSuite>::PublicKey,
      plaintext: &[u8],
      out: &mut [u8],
    ) -> Result<<$suite as HpkeSuite>::Encapsulation, HpkeError> {
      <$suite>::seal(public_key, b"info", b"aad", plaintext, out, |out: &mut [u8]| {
        out.fill(black_box(6));
        Ok(())
      })
    }

    #[inline(never)]
    fn $open(
      encapsulation: &<$suite as HpkeSuite>::Encapsulation,
      private_key: &<$suite as HpkeSuite>::PrivateKey,
      ciphertext: &[u8],
      out: &mut [u8],
    ) -> Result<(), HpkeError> {
      <$suite>::open(encapsulation, private_key, b"info", b"aad", ciphertext, out)
    }
  };
}

hpke_boundaries!(
  HpkeX25519HkdfSha256Aes128Gcm,
  hpke_x25519_derive_keypair,
  hpke_x25519_seal,
  hpke_x25519_open
);
hpke_boundaries!(
  HpkeXWingHkdfSha256ChaCha20Poly1305,
  hpke_xwing_derive_keypair,
  hpke_xwing_seal,
  hpke_xwing_open
);

const ARENA_BYTES: usize = 256 * 1024;

/// The allocator of the HPKE `_in` roots: a bump arena over a static buffer.
///
/// With a caller's allocator, an `_in` root reaches no libc allocator edge,
/// so its bound covers rscrypto's frames and this arena's. `allocate` reserves
/// the next block that fits the layout's size and alignment, and fails when
/// the buffer is full. `deallocate` keeps the block reserved.
struct ReviewArena {
  buffer: UnsafeCell<[u8; ARENA_BYTES]>,
  used: AtomicUsize,
}

// SAFETY: `ReviewArena` writes no byte of `buffer`. Each byte belongs to at
// most one block: `allocate` reserves `[start, end)` with one atomic update of
// `used`, and `used` never decreases. So threads that share the arena reach
// each byte only through the single block that holds it.
unsafe impl Sync for ReviewArena {}

static ARENA: ReviewArena = ReviewArena {
  buffer: UnsafeCell::new([0; ARENA_BYTES]),
  used: AtomicUsize::new(0),
};

// SAFETY: A block is a reserved range of the static `buffer`, so it stays
// valid for the whole program, and `deallocate` never makes it available
// again. `&ReviewArena` is `Copy`; every copy refers to the one `ARENA` and
// shares its reservations, so a block from any copy is valid for all of them.
unsafe impl Allocator for &ReviewArena {
  fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
    let base = self.buffer.get().cast::<u8>();
    let mut start = 0;
    self
      .used
      .try_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
        let aligned = base
          .addr()
          .checked_add(used)?
          .checked_next_multiple_of(layout.align())?;
        start = aligned.checked_sub(base.addr())?;
        let end = start.checked_add(layout.size())?;
        (end <= ARENA_BYTES).then_some(end)
      })
      .map_err(|_| AllocError)?;
    // `start + size <= ARENA_BYTES`, so the block lies inside `buffer` and keeps its provenance.
    let block = NonNull::new(base.wrapping_add(start)).ok_or(AllocError)?;
    Ok(NonNull::slice_from_raw_parts(block, layout.size()))
  }

  unsafe fn deallocate(&self, _block: NonNull<u8>, _layout: Layout) {}
}

// The HPKE `_in` operations with the review arena. Each root also drops what
// it creates, so the drop and the arena's deallocation are in its call graph.
macro_rules! hpke_in_boundaries {
  ($suite:ty, $derive:ident, $sender:ident, $receiver:ident) => {
    #[inline(never)]
    fn $derive(ikm: &[u8]) {
      let private_key = <$suite>::derive_keypair_in(ikm, &ARENA);
      black_box(&private_key);
      drop(private_key);
    }

    #[inline(never)]
    fn $sender(
      public_key: &<$suite as HpkeSuite>::PublicKey,
    ) -> Result<<$suite as HpkeSuite>::Encapsulation, HpkeError> {
      let (encapsulation, sender) = HpkeSender::<$suite>::setup_in(
        public_key,
        b"info",
        |out: &mut [u8]| {
          out.fill(black_box(7));
          Ok(())
        },
        &ARENA,
      )?;
      black_box(&sender);
      drop(sender);
      Ok(encapsulation)
    }

    #[inline(never)]
    fn $receiver(
      encapsulation: &<$suite as HpkeSuite>::Encapsulation,
      private_key: &<$suite as HpkeSuite>::PrivateKey,
    ) -> Result<(), HpkeError> {
      let receiver = HpkeReceiver::<$suite>::setup_in(encapsulation, private_key, b"info", &ARENA)?;
      black_box(&receiver);
      drop(receiver);
      Ok(())
    }
  };
}

hpke_in_boundaries!(
  HpkeX25519HkdfSha256Aes128Gcm,
  hpke_x25519_derive_keypair_in,
  hpke_x25519_sender_in,
  hpke_x25519_receiver_in
);
hpke_in_boundaries!(
  HpkeXWingHkdfSha256ChaCha20Poly1305,
  hpke_xwing_derive_keypair_in,
  hpke_xwing_sender_in,
  hpke_xwing_receiver_in
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

// Composite ML-DSA signing and verification, each behind an inspectable call
// boundary: deterministic and hedged signing from a secret key, signing from a
// prepared key, and verification.
macro_rules! composite_boundaries {
  (
    $secret:ty, $prepared:ident, $public:ty, $signature:ty, $sign:ident, $sign_hedged:ident, $sign_prepared:ident,
    $verify:ident
  ) => {
    #[inline(never)]
    fn $sign(key: &$secret, message: &[u8]) -> Result<$signature, CompositeMlDsaError> {
      key.sign_deterministic(message, b"frame review")
    }

    #[inline(never)]
    fn $sign_hedged(key: &$secret, message: &[u8]) -> Result<$signature, CompositeMlDsaError> {
      key.sign_with(message, b"frame review", |out| {
        out.fill(black_box(5));
        Ok(())
      })
    }

    #[inline(never)]
    fn $sign_prepared(
      key: &$prepared<Box<MlDsa65PreparedSecretKeyStorage>>,
      message: &[u8],
    ) -> Result<$signature, CompositeMlDsaError> {
      key.sign_deterministic(message, b"frame review")
    }

    #[inline(never)]
    fn $verify(key: &$public, message: &[u8], signature: &$signature) -> Result<(), VerificationError> {
      key.verify_with_context(message, b"frame review", signature)
    }
  };
}

composite_boundaries!(
  MlDsa65Ed25519SecretKey,
  MlDsa65Ed25519PreparedSecretKey,
  MlDsa65Ed25519PublicKey,
  MlDsa65Ed25519Signature,
  composite_ed25519_sign,
  composite_ed25519_sign_hedged,
  composite_ed25519_sign_prepared,
  composite_ed25519_verify
);
composite_boundaries!(
  MlDsa65EcdsaP256SecretKey,
  MlDsa65EcdsaP256PreparedSecretKey,
  MlDsa65EcdsaP256PublicKey,
  MlDsa65EcdsaP256Signature,
  composite_p256_sign,
  composite_p256_sign_hedged,
  composite_p256_sign_prepared,
  composite_p256_verify
);

macro_rules! composite {
  ($profile:ty, $sign:ident, $sign_hedged:ident, $sign_prepared:ident, $verify:ident) => {{
    let (public, secret) = <$profile>::generate_keypair(|out| {
      out.fill(black_box(6));
      Ok(())
    })
    .expect("composite key generation");
    let message = black_box(&b"frame review"[..]);
    let signature = $sign(&secret, message).expect("composite signing");
    $verify(&public, message, &signature).expect("composite verification");
    let hedged = $sign_hedged(&secret, message).expect("composite hedged signing");
    $verify(&public, message, &hedged).expect("composite verification");
    let prepared = secret.prepare_in(Global).expect("composite preparation");
    let signature = $sign_prepared(&prepared, message).expect("composite prepared signing");
    $verify(&public, message, &signature).expect("composite verification");
  }};
}

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
  // QUIC-LB: a four-pass length links the scrubbed fallback worker on every target.
  {
    use rscrypto::aead::expert::quic_lb::{QuicLbCipher, QuicLbKey};
    let key = QuicLbKey::from_bytes(black_box([10; 16]));
    let cipher = QuicLbCipher::new(&key, black_box(3), black_box(4)).expect("QUIC-LB lengths");
    let mut block = black_box([11u8; 19]);
    cipher.encrypt(&mut block);
    cipher.decrypt_server_id(&mut block);
    black_box(block);
  }
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
  {
    let (encapsulation_key, decapsulation_key) = xwing_generate().expect("X-Wing key generation");
    let (ciphertext, shared) = xwing_encapsulate(&encapsulation_key).expect("X-Wing encapsulation");
    let decapsulated = xwing_decapsulate(&decapsulation_key, &ciphertext).expect("X-Wing decapsulation");
    assert!(shared.ct_eq(&decapsulated).declassify());
    let again =
      xwing_prepared_decapsulate(&decapsulation_key.prepare(), &ciphertext).expect("X-Wing prepared decapsulation");
    assert!(shared.ct_eq(&again).declassify());
  }
  composite!(
    MlDsa65Ed25519,
    composite_ed25519_sign,
    composite_ed25519_sign_hedged,
    composite_ed25519_sign_prepared,
    composite_ed25519_verify
  );
  composite!(
    MlDsa65EcdsaP256,
    composite_p256_sign,
    composite_p256_sign_hedged,
    composite_p256_sign_prepared,
    composite_p256_verify
  );
  {
    let plaintext = black_box(*b"frame review");
    let mut ciphertext = [0u8; 12 + 16];
    let mut opened = [0u8; 12];
    let recipient = hpke_x25519_derive_keypair(black_box(&[8; 32]));
    let encapsulation =
      hpke_x25519_seal(&recipient.public_key(), &plaintext, &mut ciphertext).expect("HPKE X25519 seal");
    hpke_x25519_open(&encapsulation, &recipient, &ciphertext, &mut opened).expect("HPKE X25519 open");
    assert_eq!(opened, plaintext);
    hpke_x25519_derive_keypair_in(black_box(&[8; 32]));
    let encapsulation = hpke_x25519_sender_in(&recipient.public_key()).expect("HPKE X25519 sender setup");
    hpke_x25519_receiver_in(&encapsulation, &recipient).expect("HPKE X25519 receiver setup");
    let recipient = hpke_xwing_derive_keypair(black_box(&[9; 32]));
    let public_key = recipient.encapsulation_key().prepare().expect("derived X-Wing key");
    let encapsulation = hpke_xwing_seal(&public_key, &plaintext, &mut ciphertext).expect("HPKE X-Wing seal");
    hpke_xwing_open(&encapsulation, &recipient, &ciphertext, &mut opened).expect("HPKE X-Wing open");
    assert_eq!(opened, plaintext);
    hpke_xwing_derive_keypair_in(black_box(&[9; 32]));
    let encapsulation = hpke_xwing_sender_in(&public_key).expect("HPKE X-Wing sender setup");
    hpke_xwing_receiver_in(&encapsulation, &recipient).expect("HPKE X-Wing receiver setup");
  }
}
