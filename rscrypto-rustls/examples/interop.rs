//! Bounded loopback endpoint for scripts/test/provider_interop.py.

extern crate alloc;

use alloc::sync::Arc;
use core::error::Error;
use core::net::SocketAddr;
use core::time::Duration;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::time::Instant;

use rustls::pki_types::ServerName;
use rustls::{ClientConfig, ClientConnection, Connection, ServerConfig, ServerConnection};

mod common;
use common::{credentials, provider};

const PAYLOAD: &[u8] = b"rscrypto provider interoperability\n";
const TIMEOUT: Duration = Duration::from_secs(10);

fn invalid(message: &str) -> io::Error {
  io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn flush(connection: &mut Connection, socket: &mut TcpStream) -> io::Result<()> {
  while connection.wants_write() {
    connection.write_tls(socket)?;
  }
  socket.flush()
}

fn receive(connection: &mut Connection, socket: &mut TcpStream, output: &mut [u8]) -> io::Result<()> {
  let mut offset = 0;
  while offset < output.len() {
    match connection.reader().read(&mut output[offset..]) {
      Ok(0) => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated peer payload")),
      Ok(length) => offset = offset.strict_add(length),
      Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
        connection.complete_io(socket)?;
      }
      Err(error) => return Err(error),
    }
  }
  Ok(())
}

fn exchange(
  mut connection: Connection,
  mut socket: TcpStream,
  client: bool,
  reverse: bool,
) -> Result<(), Box<dyn Error>> {
  // Accepted sockets can inherit the listener's nonblocking flag on BSD hosts.
  socket.set_nonblocking(false)?;
  socket.set_read_timeout(Some(TIMEOUT))?;
  socket.set_write_timeout(Some(TIMEOUT))?;
  connection.complete_io(&mut socket)?;
  if connection.is_handshaking() || connection.alpn_protocol() != Some(b"provider-test") {
    return Err(invalid("handshake or ALPN mismatch").into());
  }
  for round in 0..2 {
    if client {
      connection.writer().write_all(PAYLOAD)?;
      flush(&mut connection, &mut socket)?;
    }
    let mut received = vec![0; PAYLOAD.len()];
    receive(&mut connection, &mut socket, &mut received)?;
    let mut expected = PAYLOAD.to_vec();
    if reverse {
      expected[..PAYLOAD.len().strict_sub(1)].reverse();
    }
    if received != expected {
      return Err(invalid("peer payload mismatch").into());
    }
    if !client {
      connection.writer().write_all(PAYLOAD)?;
    }
    if round == 0 {
      connection.refresh_traffic_keys()?;
    }
    flush(&mut connection, &mut socket)?;
  }
  let exported = connection.export_keying_material([0; 32], b"provider-interop", None)?;
  let hex = exported.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
  println!(
    "RESULT {:?} {:?} {:?} {hex}",
    connection.handshake_kind(),
    connection.negotiated_cipher_suite().map(|suite| suite.suite()),
    connection.negotiated_key_exchange_group().map(|group| group.name())
  );
  io::stdout().flush()?;
  connection.send_close_notify();
  flush(&mut connection, &mut socket)?;
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
      let mut config = ClientConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .with_root_certificates(roots)
        .with_client_auth_cert(certificates, key)?;
      config.alpn_protocols = vec![b"provider-test".to_vec()];
      let config = Arc::new(config);
      for _ in 0..count {
        let socket = TcpStream::connect_timeout(&address, TIMEOUT)?;
        let connection = ClientConnection::new(config.clone(), ServerName::try_from("localhost")?)?;
        exchange(Connection::Client(connection), socket, true, reverse)?;
      }
    }
    "server" => {
      let verifier =
        rustls::server::WebPkiClientVerifier::builder_with_provider(Arc::new(roots), provider.clone()).build()?;
      let mut config = ServerConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .with_client_cert_verifier(verifier)
        .with_single_cert(certificates, key)?;
      config.alpn_protocols = vec![b"provider-test".to_vec()];
      config.ticketer = rscrypto_rustls::ticket::ticketer()?;
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
        exchange(
          Connection::Server(ServerConnection::new(config.clone())?),
          socket,
          false,
          false,
        )?;
      }
    }
    _ => return Err(invalid("expected client or server").into()),
  }
  Ok(())
}
