//! QUIC v1 workloads through the pinned production rustls and Quinn APIs.
//!
//! TLS credentials and policy come from the same fixtures as the TCP rows.
//! This module supplies transport and observations, never packet cryptography.

mod crypto;
mod network;

use alloc::sync::Arc;

use rustls::{ClientConfig, ServerConfig};

use crate::{Configs, Fixture, Profile, Workload};

pub use crypto::{CryptoFixture, HeaderFixture, PacketFixture};
pub use network::{NetworkFixture, NetworkMode, ResetFixture};

pub const TLS_BACKENDS: [&str; 4] = ["rscrypto", "rscrypto-webpki", "aws-lc-rs", "ring"];
pub const CRYPTO_BACKENDS: [&str; 3] = ["rscrypto", "aws-lc-rs", "ring"];

/// P-256 server authentication, X25519, one suite, no TLS resumption or early data.
/// The shared fixture also checks a TCP connection during untimed preparation.
#[expect(clippy::unreachable, reason = "the QUIC catalog selects only rustls backends")]
fn configs(backend: &str, suite: &str) -> (Arc<ClientConfig>, Arc<ServerConfig>) {
  let fixture = Fixture::with_workload(
    backend,
    Profile {
      credential: "p256",
      group: "X25519",
      suite,
      resumed: false,
    },
    Workload::ServerAuth,
  );
  let Configs::Rustls(client, server) = fixture.configs else {
    unreachable!("QUIC uses only the catalog's rustls providers")
  };
  (client, server)
}

fn oracle(backend: &str) -> &'static str {
  if backend == "aws-lc-rs" { "ring" } else { "aws-lc-rs" }
}
