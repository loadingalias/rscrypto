//! Blocking TLS record I/O for the provider examples.

use alloc::sync::Arc;
use core::{error::Error, net::SocketAddr, time::Duration};
use std::io::{self, Write};
use std::net::TcpStream;
use std::time::Instant;

use rustls::pki_types::ServerName;
use rustls::{ClientConfig, Connection, IoState, SideData, Tcp, VecInput};

pub(super) const TIMEOUT: Duration = Duration::from_secs(10);

/// One TLS connection over a blocking socket, with its record buffers.
pub(super) struct Stream<S: SideData> {
  pub(super) conn: Connection<S, Tcp>,
  pub(super) sock: TcpStream,
  input: VecInput,
  pub(super) output: Vec<u8>,
  pub(super) plaintext: Vec<u8>,
}

impl<S: SideData> Stream<S> {
  /// Sends every queued TLS record.
  pub(super) fn flush(&mut self) -> io::Result<()> {
    self.sock.write_all(&self.output)?;
    self.output.clear();
    Ok(())
  }

  /// Process received records and send any queued handshake messages or alerts.
  /// Report a TLS error even if the peer closes before the alert reaches it.
  pub(super) fn receive(&mut self) -> Result<IoState, Box<dyn Error>> {
    if self.input.read(&mut self.sock)? == 0 {
      return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
    }
    let processed = self
      .conn
      .read_tls(&mut self.input, &mut self.output)
      .handle_all(&mut self.plaintext);
    let sent = self.flush();
    let state = processed?;
    sent?;
    Ok(state)
  }
}

pub(super) fn handshake<S: SideData>(
  connection: Connection<S, Tcp>,
  socket: TcpStream,
  output: Vec<u8>,
) -> Result<Stream<S>, Box<dyn Error>> {
  // Accepted sockets can inherit the listener's nonblocking flag on BSD hosts.
  socket.set_nonblocking(false)?;
  socket.set_read_timeout(Some(TIMEOUT))?;
  socket.set_write_timeout(Some(TIMEOUT))?;
  let mut stream = Stream {
    conn: connection,
    sock: socket,
    input: VecInput::default(),
    output,
    plaintext: Vec::new(),
  };
  stream.flush()?;
  let started = Instant::now();
  while stream.conn.is_handshaking() {
    if started.elapsed() >= TIMEOUT {
      return Err(io::Error::from(io::ErrorKind::TimedOut).into());
    }
    stream.receive()?;
  }
  Ok(stream)
}

pub(super) fn connect(
  config: &Arc<ClientConfig>,
  address: SocketAddr,
  name: ServerName<'static>,
) -> Result<Stream<rustls::ClientSide>, Box<dyn Error>> {
  let socket = TcpStream::connect_timeout(&address, TIMEOUT)?;
  let mut output = Vec::new();
  let connection = config.connect(name).build(&mut output)?;
  handshake(connection, socket, output)
}
