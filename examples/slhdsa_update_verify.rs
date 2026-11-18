//! Pin an SLH-DSA-SHA2-128s root key as an RFC 9909 SubjectPublicKeyInfo, then verify a signed
//! update manifest against it.

use rscrypto::{SlhDsaSha2_128s, SlhDsaSha2_128sPublicKey};

fn main() -> Result<(), Box<dyn core::error::Error>> {
  // Release side: the offline root key signs each update manifest under a protocol context.
  let (public, secret) = SlhDsaSha2_128s::try_generate_keypair()?;
  let pinned_root = public.to_spki_der();
  let manifest = b"firmware 1.4.2 sha256=9f2c...";
  let context = b"example update v1";
  let mut signature = [0u8; SlhDsaSha2_128s::SIGNATURE_LENGTH];
  secret.try_sign(manifest, context, &mut signature)?;

  // Device side: the image carries only the DER root. It accepts the update only if it verifies.
  let root = SlhDsaSha2_128sPublicKey::from_spki_der(&pinned_root)?;
  root.verify_with_context(manifest, context, &signature)?;
  Ok(())
}
