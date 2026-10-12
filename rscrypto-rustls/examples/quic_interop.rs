//! Bounded loopback QUIC peer for scripts/test/provider_quic.py.

extern crate alloc;
mod common;

use alloc::sync::Arc;
use core::{error::Error, net::SocketAddr, time::Duration};
use rscrypto_rustls::quinn::{self as adapter, HandshakeData};
use std::{
  io::{self, Write},
  net::UdpSocket,
  path::Path,
};

const PAYLOAD: &[u8] = b"rscrypto QUIC application data\n";

fn invalid(message: &str) -> io::Error {
  io::Error::new(io::ErrorKind::InvalidInput, message)
}

async fn exchange(
  connection: &quinn::Connection,
  client: bool,
  early: Option<quinn::ZeroRttAccepted>,
  reset: bool,
) -> Result<(), Box<dyn Error>> {
  let expected = PAYLOAD.repeat(1024);
  let mut early = early;
  let mut accepted = false;
  for round in 0..2 {
    let (mut send, mut receive) = if client {
      connection.open_bi().await?
    } else {
      connection.accept_bi().await?
    };
    if client {
      send.write_all(&expected).await?;
      send.finish()?;
    }
    let received = receive.read_to_end(expected.len()).await?;
    if received != expected {
      return Err(invalid("QUIC payload differs").into());
    }
    if !client {
      send.write_all(&expected).await?;
      send.finish()?;
    }
    if let Some(done) = early.take() {
      accepted = done.await;
      if !accepted {
        return Err(invalid("0-RTT was rejected").into());
      }
    }
    if round == 0 {
      connection.force_key_update();
    }
  }
  let data = connection
    .handshake_data()
    .ok_or_else(|| invalid("missing negotiation"))?
    .downcast::<HandshakeData>()
    .map_err(|_| invalid("unexpected provider"))?;
  if data.protocol != b"provider-test" || connection.peer_identity().is_none() {
    return Err(invalid("QUIC ALPN or mutual authentication differs").into());
  }
  let mut exporter = [0; 32];
  connection
    .export_keying_material(&mut exporter, b"provider-quic", b"interop")
    .map_err(|_| invalid("QUIC exporter failed"))?;
  let hex = exporter.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
  println!(
    "RESULT {:?} {:?} {:?} early={accepted} {hex}",
    data.handshake_kind, data.cipher_suite, data.key_exchange_group
  );
  io::stdout().flush()?;
  if reset {
    return Ok(());
  }
  if client {
    connection.close(0u32.into(), b"complete");
  } else if !matches!(connection.closed().await, quinn::ConnectionError::ApplicationClosed(ref close) if close.error_code == 0u32.into())
  {
    return Err(invalid("expected clean peer shutdown").into());
  }
  Ok(())
}

async fn reset_signal(name: &str, write: bool) -> Result<(), Box<dyn Error>> {
  let directory = std::env::var_os("RSCRYPTO_QUIC_RESET_CONTROL")
    .ok_or_else(|| invalid("reset scenario needs a control directory"))?;
  let path = Path::new(&directory).join(name);
  if write {
    std::fs::write(path, [])?;
  } else {
    while !path.try_exists()? {
      tokio::time::sleep(Duration::from_millis(10)).await;
    }
  }
  Ok(())
}

async fn run() -> Result<(), Box<dyn Error>> {
  let args = std::env::args().skip(1).collect::<Vec<_>>();
  if args.len() != 8 {
    return Err(
      invalid(
        "usage: quic_interop client|server ADDRESS FIXTURES CREDENTIAL SUITE GROUP COUNT normal|early|retry|reset",
      )
      .into(),
    );
  }
  let address: SocketAddr = args[1].parse()?;
  let count: usize = args[6].parse()?;
  if !address.ip().is_loopback() || !(1..=4).contains(&count) {
    return Err(invalid("use loopback and 1–4 connections").into());
  }
  let (early, retry, reset) = match args[7].as_str() {
    "normal" => (false, false, false),
    "early" => (true, false, false),
    "retry" => (false, true, false),
    "reset" if count == 1 => (false, false, true),
    _ => return Err(invalid("unknown scenario").into()),
  };
  let provider = common::provider(&args[4], &args[5])?;
  let (roots, certificates, key) = common::credentials(Path::new(&args[2]), &args[3])?;
  let runtime = Arc::new(quinn::TokioRuntime);
  match args[0].as_str() {
    "client" => {
      let mut tls = rustls::ClientConfig::builder(provider)
        .with_root_certificates(roots)
        .with_client_auth_cert(certificates, key)?;
      tls.alpn_protocols = vec![b"provider-test".to_vec().into()];
      tls.enable_early_data = early;
      let mut endpoint = quinn::Endpoint::new(
        adapter::endpoint_config()?,
        None,
        UdpSocket::bind("127.0.0.1:0")?,
        runtime,
      )?;
      endpoint.set_default_client_config(adapter::client_config(tls)?);
      for index in 0..count {
        let connecting = endpoint.connect(address, "localhost")?;
        let (connection, accepted) = if early && index > 0 {
          let (connection, accepted) = connecting
            .into_0rtt()
            .map_err(|_| invalid("no cached QUIC early-data keys"))?;
          (connection, Some(accepted))
        } else {
          (connecting.await?, None)
        };
        exchange(&connection, true, accepted, reset).await?;
        if reset {
          reset_signal("exchanged", true).await?;
          reset_signal("ready", false).await?;
          let mut stream = connection.open_uni().await?;
          stream.write_all(b"stateless reset probe").await?;
          if !matches!(connection.closed().await, quinn::ConnectionError::Reset) {
            return Err(invalid("expected authenticated stateless reset").into());
          }
          println!("RESET_RECEIVED");
          reset_signal("received", true).await?;
        }
      }
      endpoint.wait_idle().await;
    }
    "server" => {
      let verifier = rustls::server::WebPkiClientVerifier::builder(Arc::new(roots), &provider).build()?;
      let mut tls = rustls::ServerConfig::builder(provider)
        .with_client_cert_verifier(Arc::new(verifier))
        .with_single_cert(certificates, key)?;
      tls.alpn_protocols = vec![b"provider-test".to_vec().into()];
      // Rustls requires stateful storage for 0-RTT anti-replay. Ordinary cases
      // exercise this provider's stateless ticket encryption instead.
      if early {
        tls.max_early_data_size = u32::MAX;
      } else {
        tls.ticketer = Some(rscrypto_rustls::ticket::ticketer()?);
      }
      let endpoint_config = adapter::endpoint_config()?;
      let server_config = adapter::server_config(tls)?;
      let endpoint = quinn::Endpoint::new(
        endpoint_config.clone(),
        Some(server_config.clone()),
        UdpSocket::bind(address)?,
        runtime.clone(),
      )?;
      println!("LISTEN {}", endpoint.local_addr()?);
      io::stdout().flush()?;
      let mut connections = 0usize;
      while connections < count {
        let incoming = endpoint.accept().await.ok_or_else(|| invalid("endpoint closed"))?;
        if retry && incoming.may_retry() {
          incoming.retry()?;
          println!("RETRY");
          continue;
        }
        let connection = incoming.await?;
        exchange(&connection, false, None, reset).await?;
        if reset {
          reset_signal("exchanged", false).await?;
          // Retain the old connection without polling its original socket.
          // The replacement knows the reset/CID keys, but no connection state.
          let address = endpoint.local_addr()?;
          endpoint.rebind(UdpSocket::bind("127.0.0.1:0")?)?;
          // Quinn retains one previous socket for migration. A second rebind
          // releases the original once its connection driver observes both events.
          endpoint.rebind(UdpSocket::bind("127.0.0.1:0")?)?;
          let socket = loop {
            match UdpSocket::bind(address) {
              Ok(socket) => break socket,
              Err(error) if error.kind() == io::ErrorKind::AddrInUse => {
                tokio::time::sleep(Duration::from_millis(10)).await;
              }
              Err(error) => return Err(error.into()),
            }
          };
          let replacement = quinn::Endpoint::new(
            endpoint_config.clone(),
            Some(server_config.clone()),
            socket,
            runtime.clone(),
          )?;
          reset_signal("ready", true).await?;
          reset_signal("received", false).await?;
          replacement.close(0u32.into(), b"complete");
          replacement.wait_idle().await;
        }
        connections = connections.strict_add(1);
      }
      endpoint.close(0u32.into(), b"complete");
      endpoint.wait_idle().await;
    }
    _ => return Err(invalid("expected client or server").into()),
  }
  Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
  tokio::time::timeout(Duration::from_secs(30), run()).await?
}
