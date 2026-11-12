//! Replayable TLS entropy from caller-owned generators.
//!
//! rustls draws entropy from `&'static` sources and passes no connection
//! identity to them. [`ScopedRandom`] is a stateless source for a `static`. It
//! draws from the [`Stream`] of the innermost [`scope`] on the calling thread.
//! Give each connection its own stream, such as
//! `Stream::new(endpoint.child(b"tls-con\0", n))` for the `n`th connection.
//! Run each rustls call of the connection inside a scope of that stream: the
//! call that builds the connection, and each call that processes peer data. A
//! replay that rebuilds the same generators then gets the same handshake bytes.
//!
//! Outside a scope, [`ScopedRandom`] returns `GetRandomFailed`. A missed scope
//! therefore fails the handshake and never draws from the stream of another
//! connection. Values that rustls shares between connections, such as signing
//! keys and session-ticket keys, draw from the scope in which they are built or
//! rotated.
//!
//! [`scope`] takes a shared `&Stream`, not a `&mut ChaCha20Rng`. A stream keeps
//! its generator in one heap allocation for its whole life. So a scope needs no
//! unsafe code, copies no generator state, and lets a connection move between
//! threads.
//!
//! ```
//! # #[cfg(all(feature = "hkdf", feature = "aes-gcm", feature = "x25519", feature = "ecdsa-p256"))]
//! # {
//! use std::sync::Arc;
//!
//! use rscrypto::ChaCha20Rng;
//! use rscrypto_rustls::replay::{self, ScopedRandom, Stream};
//!
//! static RANDOM: ScopedRandom = ScopedRandom;
//! static CLIENT: rscrypto_rustls::ClientProvider = rscrypto_rustls::ClientProvider::new(&RANDOM);
//!
//! let config = Arc::new(
//!   rustls::ClientConfig::builder(Arc::new(CLIENT.provider()))
//!     .with_root_certificates(rustls::RootCertStore::empty())
//!     .with_no_client_auth()?,
//! );
//! // Use the recorded root seed of the run; this placeholder is not secret.
//! let root = ChaCha20Rng::from_seed([0x5a; ChaCha20Rng::SEED_SIZE]);
//! let stream = Stream::new(root.child(b"tls-con\0", 0));
//! let name = rustls::pki_types::ServerName::try_from("example.com").expect("DNS name");
//!
//! let mut hello = Vec::new();
//! assert!(config.connect(name.clone()).build(&mut hello).is_err(), "no scope");
//! let connection = replay::scope(&stream, || config.connect(name).build(&mut hello))?;
//! # let _ = connection;
//! # }
//! # Ok::<(), rustls::Error>(())
//! ```

use alloc::sync::Arc;
use core::cell::RefCell;
use std::sync::Mutex;

use rscrypto::ChaCha20Rng;
use rustls::crypto::{GetRandomFailed, SecureRandom};

std::thread_local! {
  /// The stream of the innermost `scope` on this thread.
  static CURRENT: RefCell<Option<Arc<Mutex<ChaCha20Rng>>>> = const { RefCell::new(None) };
}

/// The entropy stream of one connection.
///
/// It owns its generator for its whole life, and drop clears the generator.
/// `Debug` shows no generator state.
#[derive(Debug)]
pub struct Stream(Arc<Mutex<ChaCha20Rng>>);

impl Stream {
  /// Use `rng` for every draw inside a [`scope`] of this stream.
  #[must_use]
  pub fn new(rng: ChaCha20Rng) -> Self {
    Self(Arc::new(Mutex::new(rng)))
  }
}

/// Entropy from the stream of the innermost [`scope`] on the calling thread.
///
/// Outside a scope, `fill` returns `GetRandomFailed`.
#[derive(Debug)]
pub struct ScopedRandom;

impl SecureRandom for ScopedRandom {
  fn fill(&self, output: &mut [u8]) -> Result<(), GetRandomFailed> {
    CURRENT.with_borrow(|current| {
      let stream = current.as_ref().ok_or(GetRandomFailed)?;
      // A poisoned lock fails closed. `fill_bytes` cannot panic, so no draw poisons it.
      stream.lock().map_err(|_| GetRandomFailed)?.fill_bytes(output);
      Ok(())
    })
  }
}

/// Run `f` with `stream` as the source of [`ScopedRandom`] on this thread.
///
/// When `f` returns or unwinds, the previous source of this thread returns:
/// the stream of the enclosing scope, or none.
pub fn scope<R>(stream: &Stream, f: impl FnOnce() -> R) -> R {
  /// Restores the previous stream when `scope` returns or unwinds.
  struct Restore(Option<Arc<Mutex<ChaCha20Rng>>>);

  impl Drop for Restore {
    fn drop(&mut self) {
      CURRENT.set(self.0.take());
    }
  }

  let _restore = Restore(CURRENT.replace(Some(Arc::clone(&stream.0))));
  f()
}

#[cfg(test)]
mod tests {
  use core::panic::AssertUnwindSafe;
  use std::panic::{catch_unwind, resume_unwind};

  use rscrypto::ChaCha20Rng;
  use rustls::crypto::SecureRandom as _;

  use super::{ScopedRandom, Stream, scope};

  fn stream(seed: u8) -> Stream {
    Stream::new(ChaCha20Rng::from_seed([seed; ChaCha20Rng::SEED_SIZE]))
  }

  /// The first `N` bytes of the generator that `stream(seed)` wraps.
  fn first<const N: usize>(seed: u8) -> [u8; N] {
    let mut bytes = [0; N];
    ChaCha20Rng::from_seed([seed; ChaCha20Rng::SEED_SIZE]).fill_bytes(&mut bytes);
    bytes
  }

  fn draw() -> [u8; 16] {
    let mut bytes = [0; 16];
    ScopedRandom.fill(&mut bytes).expect("draw inside a scope");
    bytes
  }

  #[test]
  fn nested_scopes_restore_the_outer_stream() {
    let (outer, inner) = (stream(1), stream(2));
    let (before, nested, after) = scope(&outer, || {
      let before = draw();
      let nested = scope(&inner, draw);
      (before, nested, draw())
    });
    assert_eq!([before, after].concat(), first::<32>(1));
    assert_eq!(nested, first::<16>(2));
    assert!(
      ScopedRandom.fill(&mut [0; 16]).is_err(),
      "no source after the outer scope"
    );
  }

  #[test]
  fn a_panic_inside_a_scope_restores_the_previous_stream() {
    let (outer, inner) = (stream(1), stream(2));
    let after = scope(&outer, || {
      let unwound = catch_unwind(AssertUnwindSafe(|| {
        scope(&inner, || resume_unwind(Box::new("rustls callback failed")))
      }));
      assert!(unwound.is_err());
      draw()
    });
    assert_eq!(after, first::<16>(1));
    assert!(
      ScopedRandom.fill(&mut [0; 16]).is_err(),
      "no source after the outer scope"
    );
  }
}
