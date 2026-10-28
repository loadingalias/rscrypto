//! Shared credential and profile input for bounded provider examples.
use alloc::sync::Arc;
use core::error::Error;
use rustls::RootCertStore;
use rustls::crypto::{CryptoProvider, Identity};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use std::{fs, io, path::Path};

fn invalid(message: &str) -> io::Error {
  io::Error::new(io::ErrorKind::InvalidInput, message)
}

pub(super) fn credentials(
  directory: &Path,
  kind: &str,
) -> Result<(RootCertStore, Arc<Identity<'static>>, PrivateKeyDer<'static>), Box<dyn Error>> {
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
  Ok((roots, Arc::new(Identity::from_cert_chain(certificates)?), key))
}

/// The default provider restricted to one suite and the `+`-joined groups. A
/// TLS 1.2 suite selects TLS 1.2 only, as a caller opts in with
/// `ALL_TLS12_CIPHER_SUITES`.
pub(super) fn provider(suite: &str, groups: &str) -> Result<Arc<CryptoProvider>, Box<dyn Error>> {
  let groups = groups.split('+').collect::<Vec<_>>();
  let mut provider = rscrypto_rustls::default_provider();
  provider.tls12_cipher_suites = rscrypto_rustls::cipher_suite::ALL_TLS12_CIPHER_SUITES.into();
  provider
    .tls13_cipher_suites
    .to_mut()
    .retain(|item| format!("{:?}", item.common.suite) == suite);
  provider
    .tls12_cipher_suites
    .to_mut()
    .retain(|item| format!("{:?}", item.common.suite) == suite);
  // In the given order, which is the server's preference order.
  let all = provider.kx_groups.to_vec();
  provider.kx_groups = groups
    .iter()
    .filter_map(|name| all.iter().find(|item| format!("{:?}", item.name()) == *name).copied())
    .collect::<Vec<_>>()
    .into();
  if provider
    .tls13_cipher_suites
    .len()
    .strict_add(provider.tls12_cipher_suites.len())
    != 1
    || provider.kx_groups.len() != groups.len()
  {
    return Err(invalid("suite or group is unavailable").into());
  }
  Ok(Arc::new(provider))
}
