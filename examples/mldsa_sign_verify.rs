//! Sign a manifest with a protocol context and verify the encoded signature.

use rscrypto::{MlDsa65, MlDsa65Signature};

fn main() -> Result<(), Box<dyn core::error::Error>> {
  let (public, secret) = MlDsa65::try_generate_keypair()?;
  let message = b"release manifest";
  let context = b"example manifest v1";
  let signature = secret.try_sign(message, context)?;
  let received = MlDsa65Signature::try_from_slice(signature.as_bytes())?;
  public.verify_with_context(message, context, &received)?;
  Ok(())
}
