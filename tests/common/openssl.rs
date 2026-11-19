//! The `openssl` command-line oracle of the OpenSSL differential tests.

/// Skip an OpenSSL differential that cannot run here, or fail it when `RSCRYPTO_REQUIRE_OPENSSL` is set.
///
/// CI sets the variable where it puts the pinned OpenSSL peer on `PATH`, so a missing or older `openssl`
/// cannot let a differential pass without running.
pub(crate) fn skip(reason: core::fmt::Arguments<'_>) {
  assert!(
    std::env::var_os("RSCRYPTO_REQUIRE_OPENSSL").is_none(),
    "{reason}, and RSCRYPTO_REQUIRE_OPENSSL requires the oracle"
  );
  eprintln!("{reason}");
}
