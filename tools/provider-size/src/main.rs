//! Minimal HTTPS client whose linked size compares the provider constructors.
//!
//! Usage: `rscrypto-provider-size HOST ROOTS.pem`. It verifies HOST's chain
//! against the PEM bundle, requests `/`, and prints the response length.

extern crate alloc;

use alloc::sync::Arc;
use core::error::Error;
use std::io::{Read, Write};
use std::net::TcpStream;

use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, ServerName};
use rustls::{ClientConfig, RootCertStore};
use rustls_util::StreamOwned;

#[cfg(not(feature = "default-provider"))]
fn provider() -> rustls::crypto::CryptoProvider {
  static CLIENT: rscrypto_rustls::ClientProvider = rscrypto_rustls::ClientProvider::new(&entropy::SYSTEM);
  CLIENT.provider()
}

#[cfg(feature = "default-provider")]
fn provider() -> rustls::crypto::CryptoProvider {
  rscrypto_rustls::default_provider()
}

fn main() -> Result<(), Box<dyn Error>> {
  let mut args = std::env::args().skip(1);
  let (Some(host), Some(bundle)) = (args.next(), args.next()) else {
    return Err("usage: rscrypto-provider-size HOST ROOTS.pem".into());
  };
  let mut roots = RootCertStore::empty();
  // Skip any bundle root that webpki cannot parse, as rustls's loaders do.
  roots.add_parsable_certificates(CertificateDer::pem_file_iter(&bundle)?.collect::<Result<Vec<_>, _>>()?);
  let config = ClientConfig::builder(Arc::new(provider()))
    .with_root_certificates(roots)
    .with_no_client_auth()?;
  let mut output = Vec::new();
  let connection = Arc::new(config)
    .connect(ServerName::try_from(host.clone())?)
    .build(&mut output)?;
  let socket = TcpStream::connect((host.as_str(), 443))?;
  let mut stream = StreamOwned::new(connection, socket, output);
  write!(stream, "GET / HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n")?;
  let mut response = Vec::new();
  stream.read_to_end(&mut response)?;
  println!("{} response bytes", response.len());
  Ok(())
}

/// Operating-system entropy without the getrandom crate, as top-secret supplies it.
#[cfg(not(feature = "default-provider"))]
mod entropy {
  use rustls::crypto::{GetRandomFailed, SecureRandom};

  #[derive(Debug)]
  pub(crate) struct System;

  pub(crate) static SYSTEM: System = System;

  impl SecureRandom for System {
    fn fill(&self, output: &mut [u8]) -> Result<(), GetRandomFailed> {
      if fill(output) {
        Ok(())
      } else {
        output.fill(0);
        Err(GetRandomFailed)
      }
    }
  }

  #[cfg(target_os = "linux")]
  fn fill(output: &mut [u8]) -> bool {
    unsafe extern "C" {
      // glibc 2.25 and musl 1.1.20 export getrandom(2).
      fn getrandom(buffer: *mut u8, length: usize, flags: u32) -> isize;
    }
    let mut filled = 0;
    while filled < output.len() {
      let rest = &mut output[filled..];
      // SAFETY: `rest` is valid for writes of `rest.len()` bytes.
      let written = unsafe { getrandom(rest.as_mut_ptr(), rest.len(), 0) };
      match usize::try_from(written) {
        Ok(written) => filled += written,
        Err(_) if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted => {}
        Err(_) => return false,
      }
    }
    true
  }

  #[cfg(target_vendor = "apple")]
  fn fill(output: &mut [u8]) -> bool {
    unsafe extern "C" {
      fn getentropy(buffer: *mut u8, length: usize) -> i32;
    }
    // getentropy fills at most 256 bytes per call.
    output
      .chunks_mut(256)
      // SAFETY: each chunk is valid for writes of its length, which is at most 256.
      .all(|chunk| unsafe { getentropy(chunk.as_mut_ptr(), chunk.len()) } == 0)
  }

  #[cfg(windows)]
  fn fill(output: &mut [u8]) -> bool {
    #[link(name = "bcryptprimitives", kind = "raw-dylib")]
    unsafe extern "system" {
      fn ProcessPrng(buffer: *mut u8, length: usize) -> i32;
    }
    // SAFETY: `output` is valid for writes of `output.len()` bytes; ProcessPrng always succeeds.
    unsafe { ProcessPrng(output.as_mut_ptr(), output.len()) != 0 }
  }
}
