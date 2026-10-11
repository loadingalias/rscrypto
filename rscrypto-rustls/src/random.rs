//! Operating-system entropy through getrandom.

use rustls::crypto::{GetRandomFailed, SecureRandom};

/// Operating-system entropy. A failed fill clears any partial output.
pub static SECURE_RANDOM: &dyn SecureRandom = &OsRandom;

#[derive(Debug)]
struct OsRandom;

impl SecureRandom for OsRandom {
  fn fill(&self, output: &mut [u8]) -> Result<(), GetRandomFailed> {
    fill_with(output, getrandom::fill)
  }
}

fn fill_with<E>(output: &mut [u8], fill: impl FnOnce(&mut [u8]) -> Result<(), E>) -> Result<(), GetRandomFailed> {
  if fill(output).is_err() {
    rscrypto::traits::ct::zeroize(output);
    return Err(GetRandomFailed);
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::fill_with;

  #[test]
  fn entropy_failure_clears_partial_output_and_propagates() {
    let mut output = [0xa5; 32];
    let result = fill_with(&mut output, |out| {
      out[..16].fill(0x42);
      Err(())
    });
    assert!(matches!(result, Err(rustls::crypto::GetRandomFailed)));
    assert_eq!(output, [0; 32]);
  }
}
