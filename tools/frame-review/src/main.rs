//! Linked release artifact for secret-worker frame review.
//!
//! `scripts/frames/frames.py` reads the call graph and the compiler's
//! `.stack_sizes` records from this binary. Each public operation runs once so
//! every secret-hashing worker it reaches is linked exactly as an ordinary
//! caller would link it. Inputs pass through `black_box` so constant folding
//! cannot remove or specialize those paths.

use core::hint::black_box;

use rscrypto::{
  Kem, MlDsa44, MlDsa44PreparedSecretKeyStorage, MlDsa65, MlDsa65PreparedSecretKeyStorage, MlDsa87,
  MlDsa87PreparedSecretKeyStorage, MlKem512, MlKem768, MlKem1024,
};

macro_rules! mldsa {
  ($profile:ty, $storage:ty, $seed:expr) => {{
    let (public, secret) = <$profile>::keypair_from_seed(black_box($seed)).expect("ML-DSA key generation");
    let signature = secret
      .sign_deterministic(black_box(b"frame review"), b"")
      .expect("ML-DSA signing");
    public
      .verify_with_context(b"frame review", b"", &signature)
      .expect("ML-DSA verification");
    let mut storage = Box::new(<$storage>::new());
    let prepared = secret.prepare(&mut storage).expect("ML-DSA preparation");
    let hedged = prepared
      .sign_with(b"frame review", b"", |out| {
        out.fill(black_box(3));
        Ok(())
      })
      .expect("ML-DSA prepared signing");
    public
      .verify_with_context(b"frame review", b"", &hedged)
      .expect("ML-DSA verification");
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

fn main() {
  mldsa!(MlDsa44, MlDsa44PreparedSecretKeyStorage, &[1; 32]);
  mldsa!(MlDsa65, MlDsa65PreparedSecretKeyStorage, &[2; 32]);
  mldsa!(MlDsa87, MlDsa87PreparedSecretKeyStorage, &[3; 32]);
  mlkem!(MlKem512, 4);
  mlkem!(MlKem768, 5);
  mlkem!(MlKem1024, 6);
}
