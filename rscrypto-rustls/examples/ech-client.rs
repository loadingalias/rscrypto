//! Fetch an HTTPS response with a trusted ECH configuration and a trusted CA bundle.

extern crate alloc;

use alloc::sync::Arc;
use core::{error::Error, net::SocketAddr};
use std::io::{self, Write};
use std::time::Instant;

use rscrypto_rustls::hpke::ALL_SUPPORTED_SUITES;
use rustls::client::{EchConfig, EchStatus};
use rustls::pki_types::{CertificateDer, DnsName, ServerName, pem::PemObject};
use rustls::{ClientConfig, RootCertStore};

mod tls_stream;

fn main() -> Result<(), Box<dyn Error>> {
  let args: Vec<_> = std::env::args_os().skip(1).collect();
  let [origin, endpoint, config_file, ca_file] = args.as_slice() else {
    return Err("usage: ech-client ORIGIN IP:PORT ECH_CONFIG_LIST CA_BUNDLE_PEM".into());
  };
  let origin = origin.to_str().ok_or("the origin must be a DNS name")?;
  let name = DnsName::try_from(origin)?;
  let address: SocketAddr = endpoint.to_str().ok_or("use an IP address and port")?.parse()?;
  let ech = EchConfig::new(std::fs::read(config_file)?.into(), ALL_SUPPORTED_SUITES)?;
  let mut roots = RootCertStore::empty();
  for cert in CertificateDer::pem_file_iter(ca_file)? {
    roots.add(cert?)?;
  }
  if roots.is_empty() {
    return Err("the CA bundle contains no PEM certificates".into());
  }
  let roots = Arc::new(roots);
  let provider = Arc::new(rscrypto_rustls::default_provider());
  let connect = |ech: EchConfig| {
    let mut config = ClientConfig::builder(provider.clone())
      .with_ech(ech.into())
      .with_root_certificates(roots.clone())
      .with_no_client_auth()?;
    config.alpn_protocols = vec![b"http/1.1".to_vec().into()];
    tls_stream::connect(&Arc::new(config), address, ServerName::DnsName(name.to_owned()))
  };

  let mut stream = match connect(ech) {
    Ok(stream) => stream,
    Err(error) => match *error.downcast::<rustls::Error>()? {
      rustls::Error::RejectedEch(rejection) if rejection.can_retry() => {
        let retry = EchConfig::for_retry(rejection, ALL_SUPPORTED_SUITES)?;
        eprintln!("Retrying ECH on a fresh connection");
        connect(retry)?
      }
      error => return Err(error.into()),
    },
  };
  if stream.conn.side().ech_status() != EchStatus::Accepted {
    return Err("the server did not accept ECH".into());
  }
  eprintln!("ECH accepted");
  let request = format!("GET / HTTP/1.1\r\nHost: {origin}\r\nConnection: close\r\n\r\n");
  stream.conn.write(request.as_bytes().into(), &mut stream.output)?;
  stream.flush()?;

  let started = Instant::now();
  let mut stdout = io::stdout().lock();
  loop {
    if started.elapsed() >= tls_stream::TIMEOUT {
      return Err(io::Error::from(io::ErrorKind::TimedOut).into());
    }
    let state = stream.receive()?;
    stdout.write_all(&stream.plaintext)?;
    stream.plaintext.clear();
    if state.peer_has_closed() {
      break;
    }
  }
  stdout.flush()?;
  stream.conn.send_close_notify(&mut stream.output)?;
  Ok(stream.flush()?)
}
