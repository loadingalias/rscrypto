//! Shared credential and profile input for bounded provider examples.
use alloc::sync::Arc;
use core::error::Error;
use rustls::RootCertStore;
use rustls::crypto::CryptoProvider;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use std::{fs, io, path::Path};

fn invalid(message: &str) -> io::Error {
  io::Error::new(io::ErrorKind::InvalidInput, message)
}

pub(super) fn credentials(
  directory: &Path,
  kind: &str,
) -> Result<(RootCertStore, Vec<CertificateDer<'static>>, PrivateKeyDer<'static>), Box<dyn Error>> {
  if !kind.bytes().all(|byte| byte.is_ascii_alphanumeric()) {
    return Err(invalid("invalid fixture name").into());
  }
  let mut roots = RootCertStore::empty();
  roots.add(CertificateDer::from(fs::read(
    directory.join(format!("{kind}-root.der")),
  )?))?;
  let certificates = vec![CertificateDer::from(fs::read(
    directory.join(format!("{kind}-leaf.der")),
  )?)];
  let key = PrivateKeyDer::try_from(fs::read(directory.join(format!("{kind}-leaf-key.der")))?).map_err(invalid)?;
  Ok((roots, certificates, key))
}

pub(super) fn provider(suite: &str, group: &str) -> Result<Arc<CryptoProvider>, Box<dyn Error>> {
  let mut provider = rscrypto_rustls::default_provider();
  provider
    .cipher_suites
    .retain(|item| format!("{:?}", item.suite()) == suite);
  provider.kx_groups.retain(|item| format!("{:?}", item.name()) == group);
  if provider.cipher_suites.len() != 1 || provider.kx_groups.len() != 1 {
    return Err(invalid("suite or group is unavailable").into());
  }
  Ok(Arc::new(provider))
}
