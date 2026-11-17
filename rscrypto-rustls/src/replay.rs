//! Replayable TLS entropy from caller-owned generators.
//!
//! rustls draws entropy from `&'static` sources and passes no connection
//! identity to them. [`ScopedRandom`] is a stateless source for a `static`. It
//! draws from the [`Stream`] of the innermost [`scope`] on the calling thread.
//! Give each connection its own stream, such as
//! `Stream::new(endpoint.child(b"tls-con\0", n))` for the `n`th connection.
//! Run each rustls call of the connection inside a scope of that stream: the
//! call that builds the connection, and each call that processes peer data.
//! Outside a scope, [`ScopedRandom`] returns `GetRandomFailed`. A missed scope
//! therefore fails the handshake and never draws from the stream of another
//! connection.
//!
//! # Shared owners
//!
//! A shared owner is a value that many connections use. It never draws from
//! the stream of a connection:
//!
//! - A ticketer from [`ticketer_with_rng`](crate::ticket::ticketer_with_rng)
//!   owns the generator that you give it, such as
//!   `endpoint.child(b"tickets\0", 0)`. It draws its first key and each
//!   rotated key from that generator.
//! - An RSA key that [`Provider::scoped`](crate::sign::Provider::scoped) loads
//!   seeds its own blinding generator from 32 bytes of the scope that loads
//!   it. Each blinding refresh draws from that generator. For a recorded path,
//!   load the key in a scope of its own child, such as
//!   `endpoint.child(b"rsa-key\0", 0)`.
//!
//! [`ScopedRandom`] refuses a draw by a shared owner. So
//! [`ticketer_with`](crate::ticket::ticketer_with) over [`ScopedRandom`]
//! fails, and an RSA key that [`Provider::new`](crate::sign::Provider::new)
//! loads over [`ScopedRandom`] fails to sign.
//!
//! A signature draws its salt or nonce from the stream of the connection that
//! signs. So the position of a connection stream depends only on the
//! operations of that connection.
//!
//! # Replay
//!
//! A run replays byte for byte when the replay rebuilds the same generators
//! and makes the same rustls calls in the same order. This order includes the
//! calls that build the shared owners, because an RSA key takes its seed from
//! the stream position at load. One connection that runs alone is not
//! promised to replay. Shared state, such as the nonce counter and the rotated
//! keys of a ticketer, can change its bytes.
//!
//! # Process copies
//!
//! A process fork copies every stream. Each draw from a stream checks the
//! [origin](rscrypto::ChaCha20Rng::origin) of its generator: in a process other
//! than the one that seeded the root generator, the draw returns
//! `GetRandomFailed`. A child keeps the origin of its parent, so this also
//! holds for a stream that a fork child builds from a copied generator. After a
//! fork, a connection over a copied stream fails, an RSA key from
//! [`Provider::scoped`](crate::sign::Provider::scoped) fails to sign, and a
//! ticketer from [`ticketer_with_rng`](crate::ticket::ticketer_with_rng) seals
//! no tickets. Give each process its own root seed. Fork detection has the
//! targets and limits of [`ProcessOrigin`](rscrypto::platform::ProcessOrigin).
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
//!
//! // A server's shared owners: keys that seed their own blinding generators, and
//! // a ticketer that owns a child of the endpoint's generator.
//! static KEYS: rscrypto_rustls::sign::Provider = rscrypto_rustls::sign::Provider::scoped();
//! let tickets = rscrypto_rustls::ticket::ticketer_with_rng(
//!   root.child(b"tickets\0", 0),
//!   Arc::new(rustls::time_provider::DefaultTimeProvider),
//! )?;
//! # let _ = (&KEYS, tickets);
//! # }
//! # Ok::<(), rustls::Error>(())
//! ```

use alloc::sync::Arc;
use core::cell::{Cell, RefCell};
use std::sync::Mutex;

use rscrypto::ChaCha20Rng;
use rustls::crypto::{GetRandomFailed, SecureRandom};

std::thread_local! {
  /// The stream of the innermost `scope` on this thread.
  static CURRENT: RefCell<Option<Arc<Mutex<ChaCha20Rng>>>> = const { RefCell::new(None) };
  /// Whether a shared owner draws on this thread now, through [`owner_fill`].
  static OWNER: Cell<bool> = const { Cell::new(false) };
}

/// The source of [`Provider::scoped`](crate::sign::Provider::scoped).
pub(crate) static SCOPED: ScopedRandom = ScopedRandom;

/// The entropy stream of one connection.
///
/// It owns its generator for its whole life, and drop clears the generator.
/// `Debug` shows no generator state.
#[derive(Debug)]
pub struct Stream(Arc<Mutex<ChaCha20Rng>>);

impl Stream {
  /// Use `rng` for every draw inside a [`scope`] of this stream.
  ///
  /// Each draw refuses in a process other than the one that seeded the root
  /// of `rng`; see [Process copies](self#process-copies).
  #[must_use]
  pub fn new(rng: ChaCha20Rng) -> Self {
    Self(Arc::new(Mutex::new(rng)))
  }

  /// Fill `output` from this stream for the shared owner that owns it.
  #[cfg(any(feature = "aes-gcm", feature = "rsa"))]
  pub(crate) fn fill(&self, output: &mut [u8]) -> Result<(), GetRandomFailed> {
    draw(&self.0, output)
  }
}

/// Fill `output` from `stream`, or refuse in a copy that a process fork made.
fn draw(stream: &Mutex<ChaCha20Rng>, output: &mut [u8]) -> Result<(), GetRandomFailed> {
  // A poisoned lock fails closed. `fill_bytes` cannot panic, so no draw poisons it.
  let mut rng = stream.lock().map_err(|_| GetRandomFailed)?;
  if !rng.origin().is_current() {
    return Err(GetRandomFailed);
  }
  rng.fill_bytes(output);
  Ok(())
}

/// Entropy from the stream of the innermost [`scope`] on the calling thread.
///
/// Outside a scope, for a draw by a shared owner, and in a copy that a process
/// fork made, `fill` returns `GetRandomFailed`; see [Shared owners](self#shared-owners).
#[derive(Debug)]
pub struct ScopedRandom;

impl SecureRandom for ScopedRandom {
  fn fill(&self, output: &mut [u8]) -> Result<(), GetRandomFailed> {
    if OWNER.get() {
      return Err(GetRandomFailed);
    }
    CURRENT.with_borrow(|current| draw(current.as_ref().ok_or(GetRandomFailed)?, output))
  }
}

/// Fill `output` from `random` for a shared owner. [`ScopedRandom`] refuses,
/// because the innermost scope belongs to one connection. Other sources fill.
#[cfg(any(feature = "aes-gcm", feature = "rsa"))]
pub(crate) fn owner_fill(random: &dyn SecureRandom, output: &mut [u8]) -> Result<(), GetRandomFailed> {
  // Restore the previous flag when the draw returns or unwinds.
  let _restore = core::mem::DropGuard::new(OWNER.replace(true), |previous| OWNER.set(previous));
  random.fill(output)
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
