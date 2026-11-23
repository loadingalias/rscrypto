use core::fmt;

use super::{Nonce96, SealError};
use crate::platform::ProcessOrigin;

define_unit_error! {
  /// No fresh nonce remains for this counter in this process.
  ///
  /// The counter issued its deterministic invocation budget, or a process
  /// fork copied it from the process that created it.
  pub struct NonceCounterExhausted;
  "nonce counter exhausted"
}

/// A nonce sequence ran out or authenticated encryption failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NonceCounterSealError {
  /// No fresh nonce remains for the counter in this process.
  Exhausted(NonceCounterExhausted),
  /// Authenticated encryption failed.
  Seal(SealError),
}

impl NonceCounterSealError {
  /// Construct a nonce-exhaustion error.
  #[inline]
  #[must_use]
  pub const fn exhausted() -> Self {
    Self::Exhausted(NonceCounterExhausted::new())
  }

  /// Construct a sealing error.
  #[inline]
  #[must_use]
  pub const fn seal(err: SealError) -> Self {
    Self::Seal(err)
  }
}

impl fmt::Display for NonceCounterSealError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Exhausted(err) => err.fmt(f),
      Self::Seal(err) => err.fmt(f),
    }
  }
}

impl core::error::Error for NonceCounterSealError {
  fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
    match self {
      Self::Exhausted(err) => Some(err),
      Self::Seal(err) => Some(err),
    }
  }
}

impl From<NonceCounterExhausted> for NonceCounterSealError {
  #[inline]
  fn from(value: NonceCounterExhausted) -> Self {
    Self::Exhausted(value)
  }
}

impl From<SealError> for NonceCounterSealError {
  #[inline]
  fn from(value: SealError) -> Self {
    Self::Seal(value)
  }
}

/// A process-bound sequence of 64-bit invocation counters below `LIMIT`.
pub(super) struct NonceSequence<const LIMIT: u64> {
  next: u64,
  origin: ProcessOrigin,
}

impl<const LIMIT: u64> NonceSequence<LIMIT> {
  pub(super) fn new(next: u64) -> Self {
    Self {
      next,
      origin: ProcessOrigin::current(),
    }
  }

  pub(super) const fn next_counter(&self) -> u64 {
    self.next
  }

  pub(super) fn advance(&mut self, prefix: [u8; 4]) -> Result<Nonce96, NonceCounterExhausted> {
    let counter = self.next_counter();
    if counter >= LIMIT || !self.origin.is_current() {
      return Err(NonceCounterExhausted::new());
    }

    let mut nonce = [0; Nonce96::LENGTH];
    nonce[..4].copy_from_slice(&prefix);
    nonce[4..].copy_from_slice(&counter.to_be_bytes());
    self.next = counter.strict_add(1);
    Ok(Nonce96::from_bytes(nonce))
  }
}
