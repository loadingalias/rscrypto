//! Run with: `cargo run --example rsa_pss_verify --features rsa`

use rscrypto::{RsaPssProfile, RsaPublicKey, RsaSignatureProfile};

fn main() -> Result<(), Box<dyn core::error::Error>> {
  let key = RsaPublicKey::from_spki_der(include_bytes!("../benches/rsa_fixtures/rsa3072_spki.der"))?;
  let message = b"rscrypto RSA-PSS verification fixture";
  let signature = include_bytes!("../benches/rsa_fixtures/rsa3072_pss_sha256.sig");
  let profile = RsaSignatureProfile::pss(RsaPssProfile::Sha256);

  key.verify_signature(profile, message, signature)?;

  println!("RSA-PSS/SHA-256 fixture verified");
  Ok(())
}
