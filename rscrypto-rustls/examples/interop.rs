//! Bounded loopback endpoint for scripts/test/provider_interop.py.

extern crate alloc;

use alloc::sync::Arc;
use core::error::Error;
use core::net::SocketAddr;
use core::time::Duration;
use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::Path;
use std::time::Instant;

use rustls::pki_types::ServerName;
use rustls::{ClientConfig, Connection, ServerConfig, SideData, Tcp};
use rustls_util::StreamOwned;

mod common;
use common::{credentials, provider};

const PAYLOAD: &[u8] = b"rscrypto provider interoperability\n";
const TIMEOUT: Duration = Duration::from_secs(10);

fn invalid(message: &str) -> io::Error {
  io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn exchange(
  connection: Connection<impl SideData, Tcp>,
  socket: TcpStream,
  output: Vec<u8>,
  client: bool,
  reverse: bool,
) -> Result<(), Box<dyn Error>> {
  // Accepted sockets can inherit the listener's nonblocking flag on BSD hosts.
  socket.set_nonblocking(false)?;
  socket.set_read_timeout(Some(TIMEOUT))?;
  socket.set_write_timeout(Some(TIMEOUT))?;
  let mut stream = StreamOwned::new(connection, socket, output);
  rustls_util::complete_io(
    &mut stream.sock,
    &mut stream.input,
    &mut stream.received_plaintext,
    &mut stream.output,
    &mut stream.conn,
  )?;
  if stream.conn.is_handshaking() || stream.conn.alpn_protocol().map(AsRef::as_ref) != Some(b"provider-test".as_slice())
  {
    return Err(invalid("handshake or ALPN mismatch").into());
  }
  for round in 0..2 {
    if client {
      stream.write_all(PAYLOAD)?;
      stream.flush()?;
    }
    let mut received = vec![0; PAYLOAD.len()];
    stream.read_exact(&mut received)?;
    let mut expected = PAYLOAD.to_vec();
    if reverse {
      expected[..PAYLOAD.len().strict_sub(1)].reverse();
    }
    if received != expected {
      return Err(invalid("peer payload mismatch").into());
    }
    if !client {
      stream.write_all(PAYLOAD)?;
    }
    // TLS 1.2 has no KeyUpdate.
    if round == 0 && stream.conn.protocol_version() == Some(rustls::enums::ProtocolVersion::TLSv1_3) {
      stream.conn.refresh_traffic_keys(&mut stream.output)?;
    }
    stream.flush()?;
  }
  let exported = stream.conn.exporter()?.derive(b"provider-interop", None, [0; 32])?;
  let hex = exported.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
  println!(
    "RESULT {:?} {:?} {:?} {hex}",
    stream.conn.handshake_kind(),
    stream.conn.negotiated_cipher_suite().map(|suite| suite.suite()),
    stream.conn.negotiated_key_exchange_group().map(|group| group.name())
  );
  io::stdout().flush()?;
  stream.conn.send_close_notify(&mut stream.output)?;
  stream.flush()?;
  // A peer record can still arrive after the last read, such as the reply to
  // our KeyUpdate. Closing a socket with unread data sends a TCP reset, which
  // can destroy the close_notify in flight, so drain until the peer closes.
  stream.sock.shutdown(Shutdown::Write)?;
  io::copy(&mut stream.sock, &mut io::sink())?;
  Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
  let args = std::env::args().skip(1).collect::<Vec<_>>();
  if args.len() != 8 {
    return Err(
      invalid("usage: interop client|server ADDRESS FIXTURES CREDENTIAL SUITE GROUP COUNT echo|reverse").into(),
    );
  }
  let address: SocketAddr = args[1].parse()?;
  let count: usize = args[6].parse()?;
  if !address.ip().is_loopback() || !(1..=4).contains(&count) {
    return Err(invalid("use a loopback address and 1–4 connections").into());
  }
  let reverse = match args[7].as_str() {
    "echo" => false,
    "reverse" => true,
    _ => return Err(invalid("expected echo or reverse").into()),
  };
  let provider = provider(&args[4], &args[5])?;
  let (roots, certificates, key) = credentials(Path::new(&args[2]), &args[3])?;
  match args[0].as_str() {
    "client" => {
      let mut config = ClientConfig::builder(provider)
        .with_root_certificates(roots)
        .with_client_auth_cert(certificates, key)?;
      config.alpn_protocols = vec![b"provider-test".to_vec().into()];
      // TLS 1.2 sessions require the extended master secret (RFC 7627).
      config.require_ems = true;
      let config = Arc::new(config);
      for _ in 0..count {
        let socket = TcpStream::connect_timeout(&address, TIMEOUT)?;
        let mut output = Vec::new();
        let connection = config.connect(ServerName::try_from("localhost")?).build(&mut output)?;
        exchange(connection, socket, output, true, reverse)?;
      }
    }
    "server" => {
      let verifier = rustls::server::WebPkiClientVerifier::builder(Arc::new(roots), &provider).build()?;
      let mut config = ServerConfig::builder(provider)
        .with_client_cert_verifier(Arc::new(verifier))
        .with_single_cert(certificates, key)?;
      config.alpn_protocols = vec![b"provider-test".to_vec().into()];
      config.require_ems = true;
      config.ticketer = Some(rscrypto_rustls::ticket::ticketer()?);
      config.session_storage = Arc::new(rustls::server::NoServerSessionStorage {});
      let config = Arc::new(config);
      let listener = TcpListener::bind(address)?;
      listener.set_nonblocking(true)?;
      println!("LISTEN {}", listener.local_addr()?);
      io::stdout().flush()?;
      for _ in 0..count {
        let deadline = Instant::now()
          .checked_add(TIMEOUT)
          .ok_or_else(|| invalid("deadline overflow"))?;
        let socket = loop {
          match listener.accept() {
            Ok((socket, _)) => break socket,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock && Instant::now() < deadline => {
              std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => return Err(error.into()),
          }
        };
        exchange(Connection::new(config.clone())?, socket, Vec::new(), false, false)?;
      }
    }
    _ => return Err(invalid("expected client or server").into()),
  }
  Ok(())
}
