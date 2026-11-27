//! Signature profiles and protocol adapters.

use super::*;

impl RsaJwtAlgorithm {
  /// Return the canonical case-sensitive JOSE `alg` value.
  #[inline]
  #[must_use]
  pub const fn as_str(self) -> &'static str {
    match self {
      Self::Rs256 => "RS256",
      Self::Rs384 => "RS384",
      Self::Rs512 => "RS512",
      Self::Ps256 => "PS256",
      Self::Ps384 => "PS384",
      Self::Ps512 => "PS512",
    }
  }

  /// Return the exact RSA signature profile fixed by this algorithm.
  #[inline]
  #[must_use]
  pub const fn signature_profile(self) -> RsaSignatureProfile {
    match self {
      Self::Rs256 => RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256),
      Self::Rs384 => RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha384),
      Self::Rs512 => RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha512),
      Self::Ps256 => RsaSignatureProfile::pss(RsaPssProfile::Sha256),
      Self::Ps384 => RsaSignatureProfile::pss(RsaPssProfile::Sha384),
      Self::Ps512 => RsaSignatureProfile::pss(RsaPssProfile::Sha512),
    }
  }
}

impl fmt::Display for RsaJwtAlgorithm {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(self.as_str())
  }
}

impl RsaSignatureProfile {
  /// Return a TLS 1.3-style PSS profile with salt length equal to digest length.
  #[inline]
  #[must_use]
  pub const fn pss(profile: RsaPssProfile) -> Self {
    Self::Pss {
      profile,
      salt_len: profile.digest_len(),
    }
  }

  /// Return a PSS profile with an explicitly parsed salt length.
  #[inline]
  #[must_use]
  pub const fn pss_with_salt_len(profile: RsaPssProfile, salt_len: usize) -> Self {
    Self::Pss { profile, salt_len }
  }

  /// Return a PKCS#1 v1.5 signature profile.
  #[inline]
  #[must_use]
  pub const fn pkcs1v15(profile: RsaPkcs1v15Profile) -> Self {
    Self::Pkcs1v15(profile)
  }

  /// Return the PSS profile and salt length when this is an RSASSA-PSS profile.
  #[inline]
  #[must_use]
  pub const fn pss_parts(self) -> Option<(RsaPssProfile, usize)> {
    match self {
      Self::Pss { profile, salt_len } => Some((profile, salt_len)),
      Self::Pkcs1v15(_) => None,
    }
  }

  /// Return the PKCS#1 v1.5 profile when this is an RSASSA-PKCS1-v1_5 profile.
  #[inline]
  #[must_use]
  pub const fn pkcs1v15_profile(self) -> Option<RsaPkcs1v15Profile> {
    match self {
      Self::Pss { .. } => None,
      Self::Pkcs1v15(profile) => Some(profile),
    }
  }

  /// Map a TLS 1.3 `SignatureScheme` for `CertificateVerify`.
  ///
  /// TLS 1.3 RSA handshake signatures are RSASSA-PSS only. Legacy
  /// RSASSA-PKCS1-v1_5 scheme IDs intentionally return
  /// [`RsaProtocolAlgorithmError::UnsupportedAlgorithm`] here, even though
  /// they may still be valid for certificate-chain signatures.
  ///
  /// # Errors
  ///
  /// Returns [`RsaProtocolAlgorithmError::UnsupportedAlgorithm`] for unknown,
  /// non-RSA, SHA-1, or PKCS#1 v1.5 TLS scheme IDs.
  #[inline]
  pub const fn from_tls13_signature_scheme(scheme: u16) -> Result<Self, RsaProtocolAlgorithmError> {
    match scheme {
      0x0804 | 0x0809 => Ok(Self::pss(RsaPssProfile::Sha256)),
      0x0805 | 0x080a => Ok(Self::pss(RsaPssProfile::Sha384)),
      0x0806 | 0x080b => Ok(Self::pss(RsaPssProfile::Sha512)),
      _ => Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm),
    }
  }

  /// Map a TLS certificate signature scheme ID to an RSA signature profile.
  ///
  /// This includes the SHA-2 PKCS#1 v1.5 identifiers used for legacy
  /// certificate-chain compatibility, plus the TLS 1.3 RSASSA-PSS identifiers.
  /// SHA-1 and MD5/SHA-1 identifiers are always rejected.
  ///
  /// # Errors
  ///
  /// Returns [`RsaProtocolAlgorithmError::UnsupportedAlgorithm`] for unknown,
  /// non-RSA, SHA-1, or MD5/SHA-1 TLS scheme IDs.
  #[inline]
  pub const fn from_tls_certificate_signature_scheme(scheme: u16) -> Result<Self, RsaProtocolAlgorithmError> {
    match scheme {
      0x0401 => Ok(Self::pkcs1v15(RsaPkcs1v15Profile::Sha256)),
      0x0501 => Ok(Self::pkcs1v15(RsaPkcs1v15Profile::Sha384)),
      0x0601 => Ok(Self::pkcs1v15(RsaPkcs1v15Profile::Sha512)),
      _ => Self::from_tls13_signature_scheme(scheme),
    }
  }

  /// Map a COSE algorithm ID to an RSA signature profile.
  ///
  /// Accepts the explicit COSE RSASSA-PKCS1-v1_5 SHA-2 IDs `-257`, `-258`,
  /// and `-259`, plus the RSASSA-PSS SHA-2 IDs `-37`, `-38`, and `-39`.
  /// SHA-1 and non-RSA IDs fail closed.
  ///
  /// This is a parser for locally configured or already trusted policy values.
  /// When the ID comes from a protected COSE header, use
  /// [`RsaPublicKey::verify_expected_cose_algorithm_id`] so protocol metadata
  /// cannot choose the verification profile.
  ///
  /// # Errors
  ///
  /// Returns [`RsaProtocolAlgorithmError::UnsupportedAlgorithm`] for unknown,
  /// non-RSA, or SHA-1 COSE algorithm IDs.
  #[inline]
  pub const fn from_cose_algorithm_id(algorithm: i64) -> Result<Self, RsaProtocolAlgorithmError> {
    match algorithm {
      -257 => Ok(Self::pkcs1v15(RsaPkcs1v15Profile::Sha256)),
      -258 => Ok(Self::pkcs1v15(RsaPkcs1v15Profile::Sha384)),
      -259 => Ok(Self::pkcs1v15(RsaPkcs1v15Profile::Sha512)),
      -37 => Ok(Self::pss(RsaPssProfile::Sha256)),
      -38 => Ok(Self::pss(RsaPssProfile::Sha384)),
      -39 => Ok(Self::pss(RsaPssProfile::Sha512)),
      _ => Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm),
    }
  }

  /// Return `true` if this profile is exactly the profile named by a COSE
  /// algorithm ID.
  #[inline]
  #[must_use]
  pub const fn matches_cose_algorithm_id(self, algorithm: i64) -> bool {
    match (self, algorithm) {
      (Self::Pkcs1v15(RsaPkcs1v15Profile::Sha256), -257)
      | (Self::Pkcs1v15(RsaPkcs1v15Profile::Sha384), -258)
      | (Self::Pkcs1v15(RsaPkcs1v15Profile::Sha512), -259) => true,
      (
        Self::Pss {
          profile: RsaPssProfile::Sha256,
          salt_len,
        },
        -37,
      ) => salt_len == RsaPssProfile::Sha256.digest_len(),
      (
        Self::Pss {
          profile: RsaPssProfile::Sha384,
          salt_len,
        },
        -38,
      ) => salt_len == RsaPssProfile::Sha384.digest_len(),
      (
        Self::Pss {
          profile: RsaPssProfile::Sha512,
          salt_len,
        },
        -39,
      ) => salt_len == RsaPssProfile::Sha512.digest_len(),
      _ => false,
    }
  }

  /// Return `true` if this profile is exactly the profile named by a TLS 1.3
  /// `CertificateVerify` RSA `SignatureScheme`.
  #[inline]
  #[must_use]
  pub const fn matches_tls13_signature_scheme(self, scheme: u16) -> bool {
    match (self, scheme) {
      (
        Self::Pss {
          profile: RsaPssProfile::Sha256,
          salt_len,
        },
        0x0804 | 0x0809,
      ) => salt_len == RsaPssProfile::Sha256.digest_len(),
      (
        Self::Pss {
          profile: RsaPssProfile::Sha384,
          salt_len,
        },
        0x0805 | 0x080a,
      ) => salt_len == RsaPssProfile::Sha384.digest_len(),
      (
        Self::Pss {
          profile: RsaPssProfile::Sha512,
          salt_len,
        },
        0x0806 | 0x080b,
      ) => salt_len == RsaPssProfile::Sha512.digest_len(),
      _ => false,
    }
  }

  /// Return `true` if this profile is exactly the profile named by a TLS
  /// certificate RSA `SignatureScheme`.
  #[inline]
  #[must_use]
  pub const fn matches_tls_certificate_signature_scheme(self, scheme: u16) -> bool {
    match (self, scheme) {
      (Self::Pkcs1v15(RsaPkcs1v15Profile::Sha256), 0x0401)
      | (Self::Pkcs1v15(RsaPkcs1v15Profile::Sha384), 0x0501)
      | (Self::Pkcs1v15(RsaPkcs1v15Profile::Sha512), 0x0601) => true,
      _ => self.matches_tls13_signature_scheme(scheme),
    }
  }

  /// Map an X.509 signature `AlgorithmIdentifier` DER value to an RSA profile.
  ///
  /// SHA-2 `sha*WithRSAEncryption` identifiers may carry absent or exact
  /// `NULL` parameters. RSASSA-PSS identifiers must carry explicit RFC 4055
  /// parameters with a supported SHA-2 hash, MGF1 using the same hash, absent
  /// or explicit salt length, and absent or explicit trailer field `1`.
  /// Missing salt length maps to the RFC 4055 default of 20 bytes. SHA-1
  /// defaults are rejected.
  ///
  /// # Errors
  ///
  /// Returns [`RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier`] for
  /// non-canonical DER and [`RsaProtocolAlgorithmError::UnsupportedAlgorithm`]
  /// for unknown, non-RSA, SHA-1, or otherwise unsupported parameters.
  #[inline]
  pub fn from_x509_signature_algorithm_der(der: &[u8]) -> Result<Self, RsaProtocolAlgorithmError> {
    parse_x509_signature_algorithm(der)
  }
}

impl RsaPrivateKey {
  /// Return a generic signer wrapper bound to `profile`.
  #[inline]
  #[must_use]
  pub const fn signer(&self, profile: RsaSignatureProfile) -> RsaSignatureSigner<'_> {
    RsaSignatureSigner::new(self, profile)
  }

  /// Return a signer bound to one typed JWT/JWS algorithm.
  ///
  /// Use [`RsaJwtAlgorithm::as_str`] when emitting the matching protected
  /// header. No runtime algorithm name is accepted by this API.
  #[inline]
  #[must_use]
  pub const fn jwt_signer(&self, algorithm: RsaJwtAlgorithm) -> RsaSignatureSigner<'_> {
    self.signer(algorithm.signature_profile())
  }

  /// Sign a message using a typed RSA signature profile and caller-provided entropy.
  ///
  /// Each blinding attempt requests one modulus-width blinding candidate and,
  /// for a valid candidate, one modulus-width inverse mask, redrawn while it is
  /// zero or not below the modulus. PSS profiles first request the profile's
  /// salt length. Invalid or non-invertible candidates are rejected for at most
  /// 128 attempts, and invalid masks for at most 128 draws per attempt.
  /// Requested buffer lengths depend only on public key and profile data. The
  /// callback count also depends on whether the supplied candidates and masks
  /// pass rejection sampling, subject to those public retry bounds.
  ///
  /// The entropy callback's error type is intentionally unconstrained and is
  /// never exposed. Any callback error becomes
  /// [`RsaPrivateOpError::EntropyUnavailable`].
  ///
  /// # Security
  ///
  /// On `Ok(())`, the callback must have overwritten every requested byte with
  /// independent, uniformly distributed output from a cryptographically secure
  /// random generator. A partial fill cannot be detected by this API and
  /// invalidates the PSS salt and blinding guarantees.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, lengths are
  /// invalid, the key is too small for the selected profile, no valid blinding
  /// factor is found within the retry bound, or the post-signing public fault
  /// check fails. `out` is cleared on every error.
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_signature_with_random_fill<E>(
    &self,
    profile: RsaSignatureProfile,
    message: &[u8],
    out: &mut [u8],
    mut fill_random: impl FnMut(&mut [u8]) -> Result<(), E>,
  ) -> Result<(), RsaPrivateOpError> {
    if out.len() != self.signature_len() {
      out.fill(0);
      return Err(RsaPrivateOpError::InvalidLength);
    }
    if let RsaSignatureProfile::Pss { profile, salt_len } = profile
      && !self.public_key().pss_salt_len_is_possible(profile, salt_len)
    {
      out.fill(0);
      return Err(RsaPrivateOpError::MessageTooLong);
    }

    let mut scratch = self.private_scratch();
    self.sign_signature_with_random_fill_and_scratch(profile, message, out, &mut scratch, &mut fill_random)
  }

  /// Sign using a typed RSA signature profile, caller-provided entropy, and scratch.
  ///
  /// This is the allocation-free steady-state signing path after `scratch`
  /// setup. The entropy request order, retry bound, and error normalization are
  /// the same as [`Self::sign_signature_with_random_fill`]. Scratch and `out`
  /// are cleared on every error; scratch is also cleared after success.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, lengths are
  /// invalid, `scratch` was allocated for a different modulus width, the key is
  /// too small for the selected profile, no valid blinding factor is found
  /// within the retry bound, or the post-signing public fault check fails.
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_signature_with_random_fill_and_scratch<E>(
    &self,
    profile: RsaSignatureProfile,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
    mut fill_random: impl FnMut(&mut [u8]) -> Result<(), E>,
  ) -> Result<(), RsaPrivateOpError> {
    let modulus_len = self.public_key().modulus().len();
    let result = scratch.ensure_len(modulus_len).and_then(|()| {
      if out.len() != modulus_len {
        return Err(RsaPrivateOpError::InvalidLength);
      }

      match profile {
        RsaSignatureProfile::Pss { profile, salt_len } => {
          if !self.public_key().pss_salt_len_is_possible(profile, salt_len) {
            return Err(RsaPrivateOpError::MessageTooLong);
          }
          {
            let salt = scratch
              .salt
              .as_mut_slice()
              .get_mut(..salt_len)
              .ok_or(RsaPrivateOpError::MessageTooLong)?;
            fill_random(salt).map_err(|_| RsaPrivateOpError::EntropyUnavailable)?;
          }
          self
            .components
            .random_blinding_factor_into_scratch_with(scratch, &mut fill_random)
            .and_then(|()| {
              self
                .components
                .sign_pss_with_stored_salt_and_blinding_and_scratch(profile, message, salt_len, out, scratch)
            })
        }
        RsaSignatureProfile::Pkcs1v15(profile) => self
          .components
          .random_blinding_factor_into_scratch_with(scratch, &mut fill_random)
          .and_then(|()| {
            self
              .components
              .sign_pkcs1v15_with_stored_blinding_and_scratch(profile, message, out, scratch)
          }),
      }
    });
    scratch.clear();
    clear_output_on_error(result, out)
  }

  /// Sign a message using a typed RSA signature profile and OS-backed randomness.
  ///
  /// PKCS#1 v1.5 profiles use deterministic EMSA-PKCS1-v1_5 encoding. PSS
  /// profiles generate a random salt with the profile's explicit salt length.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, the key is too
  /// small for the selected profile, or the post-signing public fault check
  /// fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_signature(
    &self,
    profile: RsaSignatureProfile,
    message: &[u8],
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    self.sign_signature_with_random_fill(profile, message, out, getrandom::fill)
  }

  /// Sign a message using a typed RSA signature profile, OS-backed randomness,
  /// and caller-owned private-operation scratch.
  ///
  /// Reusing scratch avoids top-level private-operation temporary allocation
  /// after setup. PKCS#1 v1.5 profiles use deterministic EMSA-PKCS1-v1_5
  /// encoding. PSS profiles generate a random salt with the profile's explicit
  /// salt length.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, `scratch` was
  /// allocated for a different modulus width, the key is too small for the
  /// selected profile, or the post-signing public fault check fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_signature_with_scratch(
    &self,
    profile: RsaSignatureProfile,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    self.sign_signature_with_random_fill_and_scratch(profile, message, out, scratch, getrandom::fill)
  }

  /// Sign using an X.509 signature `AlgorithmIdentifier` DER value.
  ///
  /// Primitive helper only: this does not build or validate certificates. It
  /// accepts the SHA-2 `sha*WithRSAEncryption` identifiers and supported
  /// RSASSA-PSS parameter sets.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::UnsupportedAlgorithm`] if `der` is malformed
  /// or not an accepted RSA SHA-2 signing algorithm. Other errors match
  /// [`Self::sign_signature`].
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_x509_signature_algorithm_der(
    &self,
    der: &[u8],
    message: &[u8],
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    let result = RsaSignatureProfile::from_x509_signature_algorithm_der(der)
      .map_err(|_| RsaPrivateOpError::UnsupportedAlgorithm)
      .and_then(|profile| self.sign_signature(profile, message, out));
    clear_output_on_error(result, out)
  }

  /// Sign using an X.509 signature `AlgorithmIdentifier` DER value and caller-owned scratch.
  ///
  /// Primitive helper only: this does not build or validate certificates. It
  /// accepts the SHA-2 `sha*WithRSAEncryption` identifiers and supported
  /// RSASSA-PSS parameter sets.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::UnsupportedAlgorithm`] if `der` is malformed
  /// or not an accepted RSA SHA-2 signing algorithm. Other errors match
  /// [`Self::sign_signature_with_scratch`].
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_x509_signature_algorithm_der_with_scratch(
    &self,
    der: &[u8],
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let result = RsaSignatureProfile::from_x509_signature_algorithm_der(der)
      .map_err(|_| RsaPrivateOpError::UnsupportedAlgorithm)
      .and_then(|profile| self.sign_signature_with_scratch(profile, message, out, scratch));
    clear_output_on_error(result, out)
  }

  /// Sign a TLS 1.3 `CertificateVerify` message with caller-provided entropy.
  ///
  /// Primitive helper only: this does not construct the TLS transcript message
  /// or enforce certificate-chain policy. TLS 1.3 RSA handshake signatures are
  /// RSASSA-PSS only; legacy PKCS#1 v1.5 scheme IDs are rejected before the
  /// entropy callback is invoked.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::UnsupportedAlgorithm`] if `scheme` is not an
  /// accepted TLS 1.3 RSA signing scheme. Other errors match
  /// [`Self::sign_signature_with_random_fill`].
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_tls13_signature_scheme_with_random_fill<E>(
    &self,
    scheme: u16,
    message: &[u8],
    out: &mut [u8],
    fill_random: impl FnMut(&mut [u8]) -> Result<(), E>,
  ) -> Result<(), RsaPrivateOpError> {
    let result = RsaSignatureProfile::from_tls13_signature_scheme(scheme)
      .map_err(|_| RsaPrivateOpError::UnsupportedAlgorithm)
      .and_then(|profile| self.sign_signature_with_random_fill(profile, message, out, fill_random));
    clear_output_on_error(result, out)
  }

  /// Sign a TLS 1.3 `CertificateVerify` message with caller entropy and scratch.
  ///
  /// Parsing and rejection happen before the entropy callback is invoked. The
  /// scratch and entropy contracts match
  /// [`Self::sign_signature_with_random_fill_and_scratch`].
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::UnsupportedAlgorithm`] if `scheme` is not an
  /// accepted TLS 1.3 RSA signing scheme. Other errors match
  /// [`Self::sign_signature_with_random_fill_and_scratch`].
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_tls13_signature_scheme_with_random_fill_and_scratch<E>(
    &self,
    scheme: u16,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
    fill_random: impl FnMut(&mut [u8]) -> Result<(), E>,
  ) -> Result<(), RsaPrivateOpError> {
    let result = RsaSignatureProfile::from_tls13_signature_scheme(scheme)
      .map_err(|_| RsaPrivateOpError::UnsupportedAlgorithm)
      .and_then(|profile| {
        self.sign_signature_with_random_fill_and_scratch(profile, message, out, scratch, fill_random)
      });
    clear_output_on_error(result, out)
  }

  /// Sign a TLS 1.3 `CertificateVerify` message using a parsed signature scheme.
  ///
  /// Primitive helper only: this does not construct the TLS transcript message
  /// or enforce certificate-chain policy. TLS 1.3 RSA handshake signatures are
  /// RSASSA-PSS only; legacy PKCS#1 v1.5 scheme IDs are rejected here.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::UnsupportedAlgorithm`] if `scheme` is not an
  /// accepted TLS 1.3 RSA signing scheme. Other errors match
  /// [`Self::sign_signature`].
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_tls13_signature_scheme(
    &self,
    scheme: u16,
    message: &[u8],
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    self.sign_tls13_signature_scheme_with_random_fill(scheme, message, out, getrandom::fill)
  }

  /// Sign a TLS 1.3 `CertificateVerify` message using a parsed signature scheme and caller-owned
  /// scratch.
  ///
  /// Primitive helper only: this does not construct the TLS transcript message
  /// or enforce certificate-chain policy. TLS 1.3 RSA handshake signatures are
  /// RSASSA-PSS only; legacy PKCS#1 v1.5 scheme IDs are rejected here.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::UnsupportedAlgorithm`] if `scheme` is not an
  /// accepted TLS 1.3 RSA signing scheme. Other errors match
  /// [`Self::sign_signature_with_scratch`].
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_tls13_signature_scheme_with_scratch(
    &self,
    scheme: u16,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    self.sign_tls13_signature_scheme_with_random_fill_and_scratch(scheme, message, out, scratch, getrandom::fill)
  }

  /// Sign using a TLS certificate signature scheme and caller-provided entropy.
  ///
  /// Primitive helper only: this does not build or validate certificates. It
  /// accepts the SHA-2 PKCS#1 v1.5 certificate-signature schemes and the TLS
  /// 1.3 RSA-PSS schemes. Unsupported schemes are rejected before the entropy
  /// callback is invoked.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::UnsupportedAlgorithm`] if `scheme` is not an
  /// accepted RSA certificate signing scheme. Other errors match
  /// [`Self::sign_signature_with_random_fill`].
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_tls_certificate_signature_scheme_with_random_fill<E>(
    &self,
    scheme: u16,
    message: &[u8],
    out: &mut [u8],
    fill_random: impl FnMut(&mut [u8]) -> Result<(), E>,
  ) -> Result<(), RsaPrivateOpError> {
    let result = RsaSignatureProfile::from_tls_certificate_signature_scheme(scheme)
      .map_err(|_| RsaPrivateOpError::UnsupportedAlgorithm)
      .and_then(|profile| self.sign_signature_with_random_fill(profile, message, out, fill_random));
    clear_output_on_error(result, out)
  }

  /// Sign using a TLS certificate signature scheme, caller entropy, and scratch.
  ///
  /// Parsing and rejection happen before the entropy callback is invoked. The
  /// scratch and entropy contracts match
  /// [`Self::sign_signature_with_random_fill_and_scratch`].
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::UnsupportedAlgorithm`] if `scheme` is not an
  /// accepted RSA certificate signing scheme. Other errors match
  /// [`Self::sign_signature_with_random_fill_and_scratch`].
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_tls_certificate_signature_scheme_with_random_fill_and_scratch<E>(
    &self,
    scheme: u16,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
    fill_random: impl FnMut(&mut [u8]) -> Result<(), E>,
  ) -> Result<(), RsaPrivateOpError> {
    let result = RsaSignatureProfile::from_tls_certificate_signature_scheme(scheme)
      .map_err(|_| RsaPrivateOpError::UnsupportedAlgorithm)
      .and_then(|profile| {
        self.sign_signature_with_random_fill_and_scratch(profile, message, out, scratch, fill_random)
      });
    clear_output_on_error(result, out)
  }

  /// Sign using a TLS certificate signature scheme ID.
  ///
  /// Primitive helper only: this does not build or validate certificates. It
  /// accepts the SHA-2 PKCS#1 v1.5 certificate-signature schemes and the TLS
  /// 1.3 RSA-PSS schemes.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::UnsupportedAlgorithm`] if `scheme` is not an
  /// accepted RSA certificate signing scheme. Other errors match
  /// [`Self::sign_signature`].
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_tls_certificate_signature_scheme(
    &self,
    scheme: u16,
    message: &[u8],
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    self.sign_tls_certificate_signature_scheme_with_random_fill(scheme, message, out, getrandom::fill)
  }

  /// Sign using a TLS certificate signature scheme ID and caller-owned scratch.
  ///
  /// Primitive helper only: this does not build or validate certificates. It
  /// accepts the SHA-2 PKCS#1 v1.5 certificate-signature schemes and the TLS
  /// 1.3 RSA-PSS schemes.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::UnsupportedAlgorithm`] if `scheme` is not an
  /// accepted RSA certificate signing scheme. Other errors match
  /// [`Self::sign_signature_with_scratch`].
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_tls_certificate_signature_scheme_with_scratch(
    &self,
    scheme: u16,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    self.sign_tls_certificate_signature_scheme_with_random_fill_and_scratch(
      scheme,
      message,
      out,
      scratch,
      getrandom::fill,
    )
  }

  /// Sign a COSE Sig_structure using an already-parsed COSE algorithm ID.
  ///
  /// Primitive helper only: this is not a COSE, CBOR, or CWT provider
  /// integration.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::UnsupportedAlgorithm`] if `algorithm` is not
  /// an accepted RSA SHA-2 COSE algorithm. Other errors match
  /// [`Self::sign_signature`].
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_cose_algorithm_id(
    &self,
    algorithm: i64,
    message: &[u8],
    out: &mut [u8],
  ) -> Result<(), RsaPrivateOpError> {
    let result = RsaSignatureProfile::from_cose_algorithm_id(algorithm)
      .map_err(|_| RsaPrivateOpError::UnsupportedAlgorithm)
      .and_then(|profile| self.sign_signature(profile, message, out));
    clear_output_on_error(result, out)
  }

  /// Sign a COSE Sig_structure using an already-parsed COSE algorithm ID and caller-owned scratch.
  ///
  /// Primitive helper only: this is not a COSE, CBOR, or CWT provider
  /// integration.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError::UnsupportedAlgorithm`] if `algorithm` is not
  /// an accepted RSA SHA-2 COSE algorithm. Other errors match
  /// [`Self::sign_signature_with_scratch`].
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[must_use = "RSA signing failure must be checked; a dropped Result silently discards a failed signature"]
  pub fn sign_cose_algorithm_id_with_scratch(
    &self,
    algorithm: i64,
    message: &[u8],
    out: &mut [u8],
    scratch: &mut RsaPrivateScratch,
  ) -> Result<(), RsaPrivateOpError> {
    let result = RsaSignatureProfile::from_cose_algorithm_id(algorithm)
      .map_err(|_| RsaPrivateOpError::UnsupportedAlgorithm)
      .and_then(|profile| self.sign_signature_with_scratch(profile, message, out, scratch));
    clear_output_on_error(result, out)
  }
}

impl RsaSignatureSigner<'_> {
  /// Construct a signer wrapper for `key` and `profile`.
  #[inline]
  #[must_use]
  pub const fn new(key: &RsaPrivateKey, profile: RsaSignatureProfile) -> RsaSignatureSigner<'_> {
    RsaSignatureSigner { key, profile }
  }

  /// Return the wrapped private key.
  #[inline]
  #[must_use]
  pub const fn key(&self) -> &RsaPrivateKey {
    self.key
  }

  /// Return the bound signature profile.
  #[inline]
  #[must_use]
  pub const fn profile(&self) -> RsaSignatureProfile {
    self.profile
  }

  /// Return the fixed signature length for this key.
  #[inline]
  #[must_use]
  pub fn signature_len(&self) -> usize {
    self.key.signature_len()
  }

  /// Sign `message` into `out` using the bound profile.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if entropy is unavailable, the key is too
  /// small for the selected profile, `out` has the wrong length, or the
  /// post-signing public fault check fails.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[inline]
  pub fn try_sign_into(&self, message: &[u8], out: &mut [u8]) -> Result<(), RsaPrivateOpError> {
    self.key.sign_signature(self.profile, message, out)
  }

  /// Allocate a signature buffer and sign `message` into it.
  ///
  /// # Errors
  ///
  /// Returns [`RsaPrivateOpError`] if signing fails. The temporary output buffer
  /// is cleared before returning an error.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[inline]
  pub fn try_sign_to_vec(&self, message: &[u8]) -> Result<Vec<u8>, RsaPrivateOpError> {
    <Self as crate::traits::TrySignerInto>::try_sign_to_vec(self, message)
  }
}

impl fmt::Debug for RsaSignatureSigner<'_> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("RsaSignatureSigner")
      .field("profile", &self.profile)
      .finish_non_exhaustive()
  }
}

#[cfg(feature = "getrandom")]
impl crate::traits::TrySignerInto for RsaSignatureSigner<'_> {
  type Error = RsaPrivateOpError;

  #[inline]
  fn signature_len(&self) -> usize {
    RsaSignatureSigner::signature_len(self)
  }

  #[inline]
  fn try_sign_into(&self, message: &[u8], out: &mut [u8]) -> Result<(), Self::Error> {
    RsaSignatureSigner::try_sign_into(self, message, out)
  }
}

#[cfg(feature = "getrandom")]
impl crate::traits::TrySigner for RsaSignatureSigner<'_> {
  type Signature = Vec<u8>;
  type Error = RsaPrivateOpError;

  #[inline]
  fn try_sign(&self, message: &[u8]) -> Result<Self::Signature, Self::Error> {
    self.try_sign_to_vec(message)
  }
}

impl RsaSignatureVerifier<'_> {
  /// Construct a verifier wrapper for `key` and `profile`.
  #[inline]
  #[must_use]
  pub const fn new(key: &RsaPublicKey, profile: RsaSignatureProfile) -> RsaSignatureVerifier<'_> {
    RsaSignatureVerifier { key, profile }
  }

  /// Return the wrapped public key.
  #[inline]
  #[must_use]
  pub const fn key(&self) -> &RsaPublicKey {
    self.key
  }

  /// Return the bound signature profile.
  #[inline]
  #[must_use]
  pub const fn profile(&self) -> RsaSignatureProfile {
    self.profile
  }

  /// Verify `signature` over `message` using the bound profile.
  ///
  /// # Errors
  ///
  /// Returns opaque [`VerificationError`] for any invalid signature.
  #[inline]
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify(&self, message: &[u8], signature: &[u8]) -> Result<(), VerificationError> {
    self.key.verify_signature(self.profile, message, signature)
  }
}

impl fmt::Debug for RsaSignatureVerifier<'_> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("RsaSignatureVerifier")
      .field("profile", &self.profile)
      .finish_non_exhaustive()
  }
}

impl crate::traits::Verifier<[u8]> for RsaSignatureVerifier<'_> {
  #[inline]
  fn verify(&self, message: &[u8], signature: &[u8]) -> Result<(), VerificationError> {
    RsaSignatureVerifier::verify(self, message, signature)
  }
}

impl<'a> RsaJwtVerifier<'a> {
  /// Construct a verifier bound to `key` and one locally selected algorithm.
  #[inline]
  #[must_use]
  pub const fn new(key: &'a RsaPublicKey, algorithm: RsaJwtAlgorithm) -> Self {
    Self { key, algorithm }
  }

  /// Return the bound public key.
  #[inline]
  #[must_use]
  pub const fn key(&self) -> &'a RsaPublicKey {
    self.key
  }

  /// Return the verifier-owned JWT/JWS algorithm policy.
  #[inline]
  #[must_use]
  pub const fn algorithm(&self) -> RsaJwtAlgorithm {
    self.algorithm
  }

  /// Verify a JWS signing input after matching the protected-header `alg`.
  ///
  /// `header_alg` must be the single decoded algorithm string produced by a
  /// strict JOSE parser. The parser remains responsible for rejecting duplicate
  /// header members, malformed JSON or UTF-8, invalid base64url, unsupported
  /// critical parameters, and malformed compact serialization.
  ///
  /// # Errors
  ///
  /// Returns opaque [`VerificationError`] when `header_alg` is not the exact
  /// canonical name of the bound algorithm or the RSA signature is invalid.
  #[inline]
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify(&self, header_alg: &str, message: &[u8], signature: &[u8]) -> Result<(), VerificationError> {
    if header_alg != self.algorithm.as_str() {
      return Err(VerificationError::new());
    }
    self
      .key
      .verify_signature(self.algorithm.signature_profile(), message, signature)
  }

  /// Verify with caller-owned RSA scratch after matching the protected-header `alg`.
  ///
  /// The same JOSE parsing requirements as [`Self::verify`] apply.
  ///
  /// # Errors
  ///
  /// Returns opaque [`VerificationError`] for an algorithm mismatch or invalid
  /// signature.
  #[inline]
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_with_scratch(
    &self,
    header_alg: &str,
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    if header_alg != self.algorithm.as_str() {
      return Err(VerificationError::new());
    }
    self
      .key
      .verify_signature_with_scratch(self.algorithm.signature_profile(), message, signature, scratch)
  }
}

impl fmt::Debug for RsaJwtVerifier<'_> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("RsaJwtVerifier")
      .field("algorithm", &self.algorithm)
      .finish_non_exhaustive()
  }
}

impl RsaTlsSignatureSchemes {
  /// Maximum number of RSA TLS schemes a single key algorithm can advertise.
  pub const MAX_LEN: usize = RSA_TLS_SIGNATURE_SCHEME_CAPACITY;

  const EMPTY: Self = Self {
    schemes: [0; RSA_TLS_SIGNATURE_SCHEME_CAPACITY],
    len: 0,
  };

  #[inline]
  pub(super) const fn new(schemes: [u16; RSA_TLS_SIGNATURE_SCHEME_CAPACITY], len: u8) -> Self {
    Self { schemes, len }
  }

  /// Number of advertised schemes.
  #[inline]
  #[must_use]
  pub const fn len(&self) -> usize {
    self.len as usize
  }

  /// Return `true` when no schemes are advertised.
  #[inline]
  #[must_use]
  pub const fn is_empty(&self) -> bool {
    self.len == 0
  }

  /// Iterate over the advertised scheme IDs.
  #[inline]
  pub fn iter(&self) -> impl Iterator<Item = u16> + '_ {
    self.schemes.iter().copied().take(self.len())
  }

  /// Return `true` if this list advertises `scheme`.
  #[inline]
  #[must_use]
  pub fn contains(&self, scheme: u16) -> bool {
    self.iter().any(|advertised| advertised == scheme)
  }
}

impl RsaX509PublicKeyAlgorithm {
  /// Check whether this key algorithm permits a signature profile.
  ///
  /// # Errors
  ///
  /// Returns [`RsaProtocolAlgorithmError::UnsupportedAlgorithm`] when the
  /// signature profile conflicts with X.509 key-algorithm constraints.
  #[inline]
  pub fn permits_signature_profile(self, profile: RsaSignatureProfile) -> Result<(), RsaProtocolAlgorithmError> {
    match self {
      Self::RsaEncryption => Ok(()),
      Self::RsaPss => match profile {
        RsaSignatureProfile::Pss { .. } => Ok(()),
        RsaSignatureProfile::Pkcs1v15(_) => Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm),
      },
      Self::RsaPssRestricted {
        profile: required,
        minimum_salt_len,
      } => match profile {
        RsaSignatureProfile::Pss { profile, salt_len } if profile == required && salt_len >= minimum_salt_len => Ok(()),
        RsaSignatureProfile::Pss { .. } | RsaSignatureProfile::Pkcs1v15(_) => {
          Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm)
        }
      },
    }
  }

  /// Map and constrain a TLS 1.3 `CertificateVerify` RSA `SignatureScheme`.
  ///
  /// `rsa_pss_rsae_*` schemes require an X.509 `rsaEncryption` public-key
  /// algorithm. `rsa_pss_pss_*` schemes require an X.509 `id-RSASSA-PSS`
  /// public-key algorithm. PKCS#1 v1.5, SHA-1, and RFC 9963 legacy
  /// client-certificate code points are rejected by the primitive default.
  ///
  /// # Errors
  ///
  /// Returns [`RsaProtocolAlgorithmError::UnsupportedAlgorithm`] for unknown,
  /// non-RSA, SHA-1, PKCS#1 v1.5, RFC 9963 legacy, or
  /// key-algorithm-confused scheme IDs.
  #[inline]
  pub fn signature_profile_from_tls13_signature_scheme(
    self,
    scheme: u16,
  ) -> Result<RsaSignatureProfile, RsaProtocolAlgorithmError> {
    let (profile, required_key_algorithm) = tls13_signature_scheme_profile_and_key_algorithm(scheme)?;
    self.permits_tls_key_algorithm(required_key_algorithm, profile)?;
    Ok(profile)
  }

  /// Map and constrain a TLS certificate RSA `SignatureScheme`.
  ///
  /// This includes legacy SHA-2 PKCS#1 v1.5 certificate-chain signatures for
  /// `rsaEncryption` keys, plus TLS 1.3 RSA-PSS RSAE/PSS schemes with their
  /// required X.509 public-key algorithm. RFC 9963 client `CertificateVerify`
  /// legacy code points are intentionally not certificate-chain schemes.
  ///
  /// # Errors
  ///
  /// Returns [`RsaProtocolAlgorithmError::UnsupportedAlgorithm`] for unknown,
  /// non-RSA, SHA-1, or key-algorithm-confused scheme IDs.
  #[inline]
  pub fn signature_profile_from_tls_certificate_signature_scheme(
    self,
    scheme: u16,
  ) -> Result<RsaSignatureProfile, RsaProtocolAlgorithmError> {
    let (profile, required_key_algorithm) = tls_certificate_signature_scheme_profile_and_key_algorithm(scheme)?;
    self.permits_tls_key_algorithm(required_key_algorithm, profile)?;
    Ok(profile)
  }

  /// Return TLS 1.3 RSA `CertificateVerify` schemes this key algorithm may advertise.
  ///
  /// Providers should use this list rather than advertising all RSA schemes:
  /// `rsa_pss_rsae_*` is valid only for `rsaEncryption` keys and
  /// `rsa_pss_pss_*` is valid only for `id-RSASSA-PSS` keys.
  #[inline]
  #[must_use]
  pub const fn advertised_tls13_signature_schemes(self) -> RsaTlsSignatureSchemes {
    match self {
      Self::RsaEncryption => RsaTlsSignatureSchemes::new([0x0804, 0x0805, 0x0806, 0, 0, 0], 3),
      Self::RsaPss => RsaTlsSignatureSchemes::new([0x0809, 0x080a, 0x080b, 0, 0, 0], 3),
      Self::RsaPssRestricted {
        profile,
        minimum_salt_len,
      } => advertised_restricted_pss_tls13_signature_scheme(profile, minimum_salt_len),
    }
  }

  /// Return TLS certificate RSA signature schemes this key algorithm may advertise.
  ///
  /// `rsaEncryption` keys may advertise SHA-2 PKCS#1 v1.5 compatibility
  /// schemes and RSAE-PSS schemes. PSS keys advertise only PSS-PSS schemes
  /// permitted by their SPKI parameters.
  #[inline]
  #[must_use]
  pub const fn advertised_tls_certificate_signature_schemes(self) -> RsaTlsSignatureSchemes {
    match self {
      Self::RsaEncryption => RsaTlsSignatureSchemes::new([0x0804, 0x0805, 0x0806, 0x0401, 0x0501, 0x0601], 6),
      Self::RsaPss => RsaTlsSignatureSchemes::new([0x0809, 0x080a, 0x080b, 0, 0, 0], 3),
      Self::RsaPssRestricted {
        profile,
        minimum_salt_len,
      } => advertised_restricted_pss_tls13_signature_scheme(profile, minimum_salt_len),
    }
  }

  fn permits_tls_key_algorithm(
    self,
    required: RsaTlsPublicKeyAlgorithm,
    profile: RsaSignatureProfile,
  ) -> Result<(), RsaProtocolAlgorithmError> {
    match required {
      RsaTlsPublicKeyAlgorithm::RsaEncryption => match self {
        Self::RsaEncryption => self.permits_signature_profile(profile),
        Self::RsaPss | Self::RsaPssRestricted { .. } => Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm),
      },
      RsaTlsPublicKeyAlgorithm::RsaPss => match self {
        Self::RsaEncryption => Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm),
        Self::RsaPss | Self::RsaPssRestricted { .. } => self.permits_signature_profile(profile),
      },
    }
  }
}

impl RsaX509PublicKey {
  /// Parse an X.509 RSA `SubjectPublicKeyInfo` DER object with the default policy.
  ///
  /// The default policy accepts RSA-3072 through RSA-8192 with exponent
  /// `65537`. Use [`RsaPublicKeyPolicy::modulus_2048_to_8192`] with
  /// [`Self::from_spki_der_with_policy`] only for deployed RSA-2048
  /// compatibility material.
  ///
  /// Unlike [`RsaPublicKey::from_spki_der`], this accepts both `rsaEncryption`
  /// and `id-RSASSA-PSS` SPKI algorithms while preserving PSS key constraints.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if the DER is malformed, the algorithm is not a
  /// supported RSA public-key algorithm, or the embedded key violates policy.
  pub fn from_spki_der(der: &[u8]) -> Result<Self, RsaKeyError> {
    Self::from_spki_der_with_policy(der, &RsaPublicKeyPolicy::default())
  }

  /// Parse an X.509 RSA `SubjectPublicKeyInfo` DER object with an explicit policy.
  ///
  /// # Errors
  ///
  /// Returns [`RsaKeyError`] if parsing or validation fails.
  pub fn from_spki_der_with_policy(der: &[u8], policy: &RsaPublicKeyPolicy) -> Result<Self, RsaKeyError> {
    let (algorithm_der, public_key_der) = parse_spki_der(der)?;
    let algorithm = parse_x509_public_key_algorithm_identifier(algorithm_der)?;
    let key = RsaPublicKey::from_pkcs1_der_with_policy(public_key_der, policy)?;
    Ok(Self { key, algorithm })
  }

  /// Return the validated RSA public key.
  #[inline]
  #[must_use]
  pub const fn public_key(&self) -> &RsaPublicKey {
    &self.key
  }

  /// Return the parsed X.509 public-key algorithm constraints.
  #[inline]
  #[must_use]
  pub const fn key_algorithm(&self) -> RsaX509PublicKeyAlgorithm {
    self.algorithm
  }

  /// Parse and constrain an X.509 signature `AlgorithmIdentifier`.
  ///
  /// # Errors
  ///
  /// Returns [`RsaProtocolAlgorithmError`] if the signature algorithm is
  /// malformed, unsupported, or conflicts with the SPKI key algorithm.
  #[inline]
  pub fn signature_profile_from_x509_algorithm_der(
    &self,
    der: &[u8],
  ) -> Result<RsaSignatureProfile, RsaProtocolAlgorithmError> {
    let profile = RsaSignatureProfile::from_x509_signature_algorithm_der(der)?;
    self.algorithm.permits_signature_profile(profile)?;
    Ok(profile)
  }

  /// Map and constrain a TLS 1.3 `CertificateVerify` RSA `SignatureScheme`.
  ///
  /// # Errors
  ///
  /// Returns [`RsaProtocolAlgorithmError`] if the scheme is unsupported or
  /// conflicts with this key's X.509 public-key algorithm.
  #[inline]
  pub fn signature_profile_from_tls13_signature_scheme(
    &self,
    scheme: u16,
  ) -> Result<RsaSignatureProfile, RsaProtocolAlgorithmError> {
    self.algorithm.signature_profile_from_tls13_signature_scheme(scheme)
  }

  /// Map and constrain a TLS certificate RSA `SignatureScheme`.
  ///
  /// # Errors
  ///
  /// Returns [`RsaProtocolAlgorithmError`] if the scheme is unsupported or
  /// conflicts with this key's X.509 public-key algorithm.
  #[inline]
  pub fn signature_profile_from_tls_certificate_signature_scheme(
    &self,
    scheme: u16,
  ) -> Result<RsaSignatureProfile, RsaProtocolAlgorithmError> {
    self
      .algorithm
      .signature_profile_from_tls_certificate_signature_scheme(scheme)
  }

  /// Verify a signature using an X.509 signature `AlgorithmIdentifier`.
  ///
  /// Primitive helper only: this is not a WebPKI, TLS, JWT, COSE, or
  /// certificate-chain provider integration.
  ///
  /// Algorithm mismatch and unsupported algorithm choices collapse to the same
  /// opaque verification error as invalid signatures.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if the algorithm is malformed, unsupported,
  /// conflicts with this key's SPKI constraints, or the signature is invalid.
  #[inline]
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_signature_from_x509_algorithm_der(
    &self,
    signature_algorithm_der: &[u8],
    message: &[u8],
    signature: &[u8],
  ) -> Result<(), VerificationError> {
    let profile = self
      .signature_profile_from_x509_algorithm_der(signature_algorithm_der)
      .map_err(|_| VerificationError::new())?;
    if !self.key.signature_profile_is_possible(profile) {
      return Err(VerificationError::new());
    }
    let mut scratch = self.key.public_scratch();
    self
      .key
      .verify_signature_with_scratch(profile, message, signature, &mut scratch)
  }

  /// Verify a signature using an X.509 signature `AlgorithmIdentifier` and caller-owned scratch.
  ///
  /// Primitive helper only: this is not a WebPKI, TLS, JWT, COSE, or
  /// certificate-chain provider integration.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if the algorithm is malformed, unsupported,
  /// conflicts with this key's SPKI constraints, or the signature is invalid.
  #[inline]
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_signature_from_x509_algorithm_der_with_scratch(
    &self,
    signature_algorithm_der: &[u8],
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    let profile = self
      .signature_profile_from_x509_algorithm_der(signature_algorithm_der)
      .map_err(|_| VerificationError::new())?;
    self
      .key
      .verify_signature_with_scratch(profile, message, signature, scratch)
  }

  /// Verify the signature on an X.509 certificate DER object.
  ///
  /// Primitive helper only: this is not a WebPKI or certificate-chain provider
  /// integration.
  ///
  /// This helper verifies only the certificate signature. It does not build or
  /// validate a certificate chain, names, validity periods, key usage, or other
  /// WebPKI policy. The caller supplies the issuer public key represented by
  /// `self`.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if the certificate DER is malformed, the
  /// TBSCertificate and outer signature algorithms differ, the signature
  /// algorithm is unsupported or conflicts with this issuer key, or the
  /// signature is invalid.
  #[inline]
  #[must_use = "certificate signature verification must be checked; a dropped Result silently accepts a forged \
                certificate"]
  pub fn verify_x509_certificate_signature_der(&self, certificate_der: &[u8]) -> Result<(), VerificationError> {
    let parsed = parse_x509_certificate_signature(certificate_der).map_err(|_| VerificationError::new())?;
    let profile = self
      .signature_profile_from_x509_algorithm_der(parsed.signature_algorithm_der)
      .map_err(|_| VerificationError::new())?;
    if !self.key.signature_profile_is_possible(profile) {
      return Err(VerificationError::new());
    }
    let mut scratch = self.key.public_scratch();
    self
      .key
      .verify_signature_with_scratch(profile, parsed.tbs_certificate_der, parsed.signature, &mut scratch)
  }

  /// Verify the signature on an X.509 certificate DER object using caller-owned scratch.
  ///
  /// Primitive helper only: this is not a WebPKI or certificate-chain provider
  /// integration.
  ///
  /// This helper verifies only the certificate signature. It does not build or
  /// validate a certificate chain, names, validity periods, key usage, or other
  /// WebPKI policy. The caller supplies the issuer public key represented by
  /// `self`.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if the certificate DER is malformed, the
  /// TBSCertificate and outer signature algorithms differ, the signature
  /// algorithm is unsupported or conflicts with this issuer key, or the
  /// signature is invalid.
  #[inline]
  #[must_use = "certificate signature verification must be checked; a dropped Result silently accepts a forged \
                certificate"]
  pub fn verify_x509_certificate_signature_der_with_scratch(
    &self,
    certificate_der: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    let parsed = parse_x509_certificate_signature(certificate_der).map_err(|_| VerificationError::new())?;
    self.verify_signature_from_x509_algorithm_der_with_scratch(
      parsed.signature_algorithm_der,
      parsed.tbs_certificate_der,
      parsed.signature,
      scratch,
    )
  }

  /// Verify a TLS 1.3 `CertificateVerify` RSA signature only if the parsed
  /// scheme matches the locally configured expected scheme and profile.
  ///
  /// Use this for TLS adapters that already know the negotiated or locally
  /// allowed `SignatureScheme` and [`RsaSignatureProfile`]. `presented_scheme`
  /// is the peer-controlled value parsed from the signed protocol message. A
  /// scheme or profile mismatch is rejected before signature verification.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if the schemes differ, the expected scheme
  /// is unsupported or conflicts with this key's X.509 public-key algorithm,
  /// or the signature is invalid.
  #[inline]
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_expected_tls13_signature_scheme(
    &self,
    expected_scheme: u16,
    presented_scheme: u16,
    profile: RsaSignatureProfile,
    message: &[u8],
    signature: &[u8],
  ) -> Result<(), VerificationError> {
    if presented_scheme != expected_scheme {
      return Err(VerificationError::new());
    }
    let mapped_profile = self
      .signature_profile_from_tls13_signature_scheme(expected_scheme)
      .map_err(|_| VerificationError::new())?;
    if mapped_profile != profile {
      return Err(VerificationError::new());
    }
    let mut scratch = self.key.public_scratch();
    self
      .key
      .verify_signature_with_scratch(profile, message, signature, &mut scratch)
  }

  /// Verify a TLS 1.3 `CertificateVerify` RSA signature with caller-owned
  /// scratch only if the parsed scheme matches the locally configured expected
  /// scheme and profile.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] for scheme mismatch, unsupported/conflicting
  /// expected schemes, or invalid signatures.
  #[inline]
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_expected_tls13_signature_scheme_with_scratch(
    &self,
    expected_scheme: u16,
    presented_scheme: u16,
    profile: RsaSignatureProfile,
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    if presented_scheme != expected_scheme {
      return Err(VerificationError::new());
    }
    let mapped_profile = self
      .signature_profile_from_tls13_signature_scheme(expected_scheme)
      .map_err(|_| VerificationError::new())?;
    if mapped_profile != profile {
      return Err(VerificationError::new());
    }
    self
      .key
      .verify_signature_with_scratch(profile, message, signature, scratch)
  }

  /// Verify a TLS 1.3 `CertificateVerify` RSA signature by mapping the
  /// supplied scheme directly.
  ///
  /// Compatibility helper only: prefer
  /// [`verify_expected_tls13_signature_scheme`](Self::verify_expected_tls13_signature_scheme)
  /// when the scheme was parsed from peer-controlled protocol metadata. The
  /// caller must construct the exact TLS `CertificateVerify` signed message and
  /// enforce handshake, certificate-chain, and policy state.
  ///
  /// Scheme mismatch and unsupported algorithm choices collapse to the same
  /// opaque verification error as invalid signatures.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if the scheme is unsupported, conflicts
  /// with this key's X.509 public-key algorithm, or the signature is invalid.
  #[inline]
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_tls13_signature_scheme(
    &self,
    scheme: u16,
    message: &[u8],
    signature: &[u8],
  ) -> Result<(), VerificationError> {
    let profile = self
      .signature_profile_from_tls13_signature_scheme(scheme)
      .map_err(|_| VerificationError::new())?;
    if !self.key.signature_profile_is_possible(profile) {
      return Err(VerificationError::new());
    }
    let mut scratch = self.key.public_scratch();
    self
      .key
      .verify_signature_with_scratch(profile, message, signature, &mut scratch)
  }

  /// Verify a TLS 1.3 `CertificateVerify` RSA signature with caller-owned scratch.
  ///
  /// Primitive helper only: this is not a TLS provider integration. The caller
  /// must construct the exact TLS `CertificateVerify` signed message and
  /// enforce handshake, certificate-chain, and policy state.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if the scheme is unsupported, conflicts
  /// with this key's X.509 public-key algorithm, or the signature is invalid.
  #[inline]
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_tls13_signature_scheme_with_scratch(
    &self,
    scheme: u16,
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    let profile = self
      .signature_profile_from_tls13_signature_scheme(scheme)
      .map_err(|_| VerificationError::new())?;
    self
      .key
      .verify_signature_with_scratch(profile, message, signature, scratch)
  }

  /// Verify a TLS certificate RSA signature only if the parsed scheme matches
  /// the locally configured expected scheme and profile.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] for scheme mismatch, unsupported/conflicting
  /// expected schemes, or invalid signatures.
  #[inline]
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_expected_tls_certificate_signature_scheme(
    &self,
    expected_scheme: u16,
    presented_scheme: u16,
    profile: RsaSignatureProfile,
    message: &[u8],
    signature: &[u8],
  ) -> Result<(), VerificationError> {
    if presented_scheme != expected_scheme {
      return Err(VerificationError::new());
    }
    let mapped_profile = self
      .signature_profile_from_tls_certificate_signature_scheme(expected_scheme)
      .map_err(|_| VerificationError::new())?;
    if mapped_profile != profile {
      return Err(VerificationError::new());
    }
    let mut scratch = self.key.public_scratch();
    self
      .key
      .verify_signature_with_scratch(profile, message, signature, &mut scratch)
  }

  /// Verify a TLS certificate RSA signature with caller-owned scratch only if
  /// the parsed scheme matches the locally configured expected scheme and
  /// profile.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] for scheme mismatch, unsupported/conflicting
  /// expected schemes, or invalid signatures.
  #[inline]
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_expected_tls_certificate_signature_scheme_with_scratch(
    &self,
    expected_scheme: u16,
    presented_scheme: u16,
    profile: RsaSignatureProfile,
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    if presented_scheme != expected_scheme {
      return Err(VerificationError::new());
    }
    let mapped_profile = self
      .signature_profile_from_tls_certificate_signature_scheme(expected_scheme)
      .map_err(|_| VerificationError::new())?;
    if mapped_profile != profile {
      return Err(VerificationError::new());
    }
    self
      .key
      .verify_signature_with_scratch(profile, message, signature, scratch)
  }

  /// Verify a TLS certificate RSA signature by mapping the supplied scheme directly.
  ///
  /// Compatibility helper only: prefer
  /// [`verify_expected_tls_certificate_signature_scheme`](Self::verify_expected_tls_certificate_signature_scheme)
  /// when the scheme was parsed from peer-controlled protocol metadata. The
  /// caller must construct the signed message and enforce handshake,
  /// certificate-chain, and policy state.
  ///
  /// This helper is for protocol surfaces that already mapped a certificate
  /// signature to a TLS `SignatureScheme`; raw X.509 certificate signatures
  /// should normally use [`Self::verify_signature_from_x509_algorithm_der`].
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if the scheme is unsupported, conflicts
  /// with this key's X.509 public-key algorithm, or the signature is invalid.
  #[inline]
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_tls_certificate_signature_scheme(
    &self,
    scheme: u16,
    message: &[u8],
    signature: &[u8],
  ) -> Result<(), VerificationError> {
    let profile = self
      .signature_profile_from_tls_certificate_signature_scheme(scheme)
      .map_err(|_| VerificationError::new())?;
    if !self.key.signature_profile_is_possible(profile) {
      return Err(VerificationError::new());
    }
    let mut scratch = self.key.public_scratch();
    self
      .key
      .verify_signature_with_scratch(profile, message, signature, &mut scratch)
  }

  /// Verify a TLS certificate RSA signature scheme with caller-owned scratch.
  ///
  /// Primitive helper only: this is not a TLS or WebPKI provider integration.
  /// The caller must construct the signed message and enforce handshake,
  /// certificate-chain, and policy state.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if the scheme is unsupported, conflicts
  /// with this key's X.509 public-key algorithm, or the signature is invalid.
  #[inline]
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_tls_certificate_signature_scheme_with_scratch(
    &self,
    scheme: u16,
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    let profile = self
      .signature_profile_from_tls_certificate_signature_scheme(scheme)
      .map_err(|_| VerificationError::new())?;
    self
      .key
      .verify_signature_with_scratch(profile, message, signature, scratch)
  }
}

impl RsaPublicKey {
  /// Return a generic verifier wrapper bound to `profile`.
  #[inline]
  #[must_use]
  pub const fn verifier(&self, profile: RsaSignatureProfile) -> RsaSignatureVerifier<'_> {
    RsaSignatureVerifier::new(self, profile)
  }

  /// Return a JWT/JWS verifier bound to one locally selected algorithm.
  ///
  /// The returned verifier treats the peer's protected-header `alg` value as
  /// an assertion to match, never as a signature-profile selector.
  #[inline]
  #[must_use]
  pub const fn jwt_verifier(&self, algorithm: RsaJwtAlgorithm) -> RsaJwtVerifier<'_> {
    RsaJwtVerifier::new(self, algorithm)
  }

  /// Verify an RSA signature under a typed signature profile.
  ///
  /// This is the preferred primitive entry point for protocol adapters after
  /// they map external algorithm identifiers and parameters to
  /// [`RsaSignatureProfile`]. Algorithm mismatches and malformed signatures all
  /// return the same opaque [`VerificationError`].
  ///
  /// # Errors
  ///
  /// Returns an opaque [`VerificationError`] for any invalid signature,
  /// malformed encoded message, unsupported representative, or profile
  /// mismatch.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_signature(
    &self,
    profile: RsaSignatureProfile,
    message: &[u8],
    signature: &[u8],
  ) -> Result<(), VerificationError> {
    if !self.signature_profile_is_possible(profile) {
      return Err(VerificationError::new());
    }
    let mut scratch = RsaPublicScratch::new(self);
    self.verify_signature_with_scratch(profile, message, signature, &mut scratch)
  }

  /// Verify an RSA signature under a typed profile with caller-owned scratch.
  ///
  /// # Errors
  ///
  /// Returns an opaque [`VerificationError`] for any invalid signature.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_signature_with_scratch(
    &self,
    profile: RsaSignatureProfile,
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    if !self.signature_profile_is_possible(profile) {
      return Err(VerificationError::new());
    }
    match profile {
      RsaSignatureProfile::Pss { profile, salt_len } => {
        self.verify_pss_with_salt_len_and_scratch(profile, salt_len, message, signature, scratch)
      }
      RsaSignatureProfile::Pkcs1v15(profile) => self.verify_pkcs1v15_with_scratch(profile, message, signature, scratch),
    }
  }

  /// Verify an RSA COSE signature only if the parsed algorithm ID matches the
  /// locally configured expected algorithm ID and profile.
  ///
  /// `expected_algorithm` and `profile` must come from local verifier
  /// configuration. `protected_algorithm` is the value parsed from the
  /// peer-controlled COSE protected header. A mismatch is rejected before
  /// signature verification.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if the algorithms differ, `profile` does
  /// not match `expected_algorithm`, or the signature is invalid.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_expected_cose_algorithm_id(
    &self,
    expected_algorithm: i64,
    protected_algorithm: i64,
    profile: RsaSignatureProfile,
    message: &[u8],
    signature: &[u8],
  ) -> Result<(), VerificationError> {
    if protected_algorithm != expected_algorithm || !profile.matches_cose_algorithm_id(expected_algorithm) {
      return Err(VerificationError::new());
    }
    self.verify_signature(profile, message, signature)
  }

  /// Verify an RSA COSE signature with caller-owned scratch only if the parsed
  /// algorithm ID matches the locally configured expected algorithm ID and
  /// profile.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] for algorithm mismatch, unsupported expected
  /// algorithms, or invalid signatures.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_expected_cose_algorithm_id_with_scratch(
    &self,
    expected_algorithm: i64,
    protected_algorithm: i64,
    profile: RsaSignatureProfile,
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    if protected_algorithm != expected_algorithm || !profile.matches_cose_algorithm_id(expected_algorithm) {
      return Err(VerificationError::new());
    }
    self.verify_signature_with_scratch(profile, message, signature, scratch)
  }

  /// Verify an RSA COSE signature by mapping the supplied algorithm ID directly.
  ///
  /// Compatibility helper only: prefer
  /// [`verify_expected_cose_algorithm_id`](Self::verify_expected_cose_algorithm_id)
  /// when the algorithm ID was parsed from peer-controlled COSE metadata. This
  /// is not a COSE, CBOR, or CWT provider integration.
  ///
  /// `message` is the caller-constructed COSE Sig_structure bytes. This helper
  /// does not parse CBOR or COSE objects; it only enforces the explicit RSA
  /// SHA-2 algorithm mapping before signature verification.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if `algorithm` is not an accepted RSA SHA-2
  /// COSE algorithm, or if the signature is invalid.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_cose_algorithm_id(
    &self,
    algorithm: i64,
    message: &[u8],
    signature: &[u8],
  ) -> Result<(), VerificationError> {
    let profile = RsaSignatureProfile::from_cose_algorithm_id(algorithm).map_err(|_| VerificationError::new())?;
    if !self.signature_profile_is_possible(profile) {
      return Err(VerificationError::new());
    }
    let mut scratch = RsaPublicScratch::new(self);
    self.verify_signature_with_scratch(profile, message, signature, &mut scratch)
  }

  /// Verify an RSA COSE signature using caller-owned RSA scratch space by
  /// mapping the supplied algorithm ID directly.
  ///
  /// Compatibility helper only: prefer
  /// [`verify_expected_cose_algorithm_id_with_scratch`](Self::verify_expected_cose_algorithm_id_with_scratch)
  /// when the algorithm ID was parsed from peer-controlled COSE metadata.
  ///
  /// # Errors
  ///
  /// Returns [`VerificationError`] if `algorithm` is unsupported or
  /// verification fails.
  #[must_use = "signature verification must be checked; a dropped Result silently accepts a forged signature"]
  pub fn verify_cose_algorithm_id_with_scratch(
    &self,
    algorithm: i64,
    message: &[u8],
    signature: &[u8],
    scratch: &mut RsaPublicScratch,
  ) -> Result<(), VerificationError> {
    let profile = RsaSignatureProfile::from_cose_algorithm_id(algorithm).map_err(|_| VerificationError::new())?;
    self.verify_signature_with_scratch(profile, message, signature, scratch)
  }

  /// Return `true` when this key is large enough for a signature profile.
  ///
  /// This is a cheap profile feasibility check for protocol adapters before
  /// allocating scratch or attempting an RSA public operation. It does not check
  /// X.509 key-algorithm constraints, parse external objects, or validate a
  /// signature.
  #[inline]
  #[must_use]
  pub fn signature_profile_is_possible(&self, profile: RsaSignatureProfile) -> bool {
    match profile {
      RsaSignatureProfile::Pss { profile, salt_len } => self.pss_salt_len_is_possible(profile, salt_len),
      RsaSignatureProfile::Pkcs1v15(_) => true,
    }
  }
}

const fn tls13_signature_scheme_profile_and_key_algorithm(
  scheme: u16,
) -> Result<(RsaSignatureProfile, RsaTlsPublicKeyAlgorithm), RsaProtocolAlgorithmError> {
  match scheme {
    0x0804 => Ok((
      RsaSignatureProfile::pss(RsaPssProfile::Sha256),
      RsaTlsPublicKeyAlgorithm::RsaEncryption,
    )),
    0x0805 => Ok((
      RsaSignatureProfile::pss(RsaPssProfile::Sha384),
      RsaTlsPublicKeyAlgorithm::RsaEncryption,
    )),
    0x0806 => Ok((
      RsaSignatureProfile::pss(RsaPssProfile::Sha512),
      RsaTlsPublicKeyAlgorithm::RsaEncryption,
    )),
    0x0809 => Ok((
      RsaSignatureProfile::pss(RsaPssProfile::Sha256),
      RsaTlsPublicKeyAlgorithm::RsaPss,
    )),
    0x080a => Ok((
      RsaSignatureProfile::pss(RsaPssProfile::Sha384),
      RsaTlsPublicKeyAlgorithm::RsaPss,
    )),
    0x080b => Ok((
      RsaSignatureProfile::pss(RsaPssProfile::Sha512),
      RsaTlsPublicKeyAlgorithm::RsaPss,
    )),
    _ => Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm),
  }
}

const fn tls_certificate_signature_scheme_profile_and_key_algorithm(
  scheme: u16,
) -> Result<(RsaSignatureProfile, RsaTlsPublicKeyAlgorithm), RsaProtocolAlgorithmError> {
  match scheme {
    0x0401 => Ok((
      RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha256),
      RsaTlsPublicKeyAlgorithm::RsaEncryption,
    )),
    0x0501 => Ok((
      RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha384),
      RsaTlsPublicKeyAlgorithm::RsaEncryption,
    )),
    0x0601 => Ok((
      RsaSignatureProfile::pkcs1v15(RsaPkcs1v15Profile::Sha512),
      RsaTlsPublicKeyAlgorithm::RsaEncryption,
    )),
    _ => tls13_signature_scheme_profile_and_key_algorithm(scheme),
  }
}

const fn advertised_restricted_pss_tls13_signature_scheme(
  profile: RsaPssProfile,
  minimum_salt_len: usize,
) -> RsaTlsSignatureSchemes {
  match profile {
    RsaPssProfile::Sha256 if minimum_salt_len <= RsaPssProfile::Sha256.digest_len() => {
      RsaTlsSignatureSchemes::new([0x0809, 0, 0, 0, 0, 0], 1)
    }
    RsaPssProfile::Sha384 if minimum_salt_len <= RsaPssProfile::Sha384.digest_len() => {
      RsaTlsSignatureSchemes::new([0x080a, 0, 0, 0, 0, 0], 1)
    }
    RsaPssProfile::Sha512 if minimum_salt_len <= RsaPssProfile::Sha512.digest_len() => {
      RsaTlsSignatureSchemes::new([0x080b, 0, 0, 0, 0, 0], 1)
    }
    RsaPssProfile::Sha256 | RsaPssProfile::Sha384 | RsaPssProfile::Sha512 => RsaTlsSignatureSchemes::EMPTY,
  }
}

fn parse_x509_public_key_algorithm_identifier(der: &[u8]) -> Result<RsaX509PublicKeyAlgorithm, RsaKeyError> {
  let mut reader = DerReader::new(der);
  let oid = reader.read_primitive(TAG_OBJECT_IDENTIFIER)?;
  match oid {
    RSA_ENCRYPTION_OID => {
      let null = reader.read_primitive(TAG_NULL)?;
      if !null.is_empty() {
        return Err(RsaKeyError::MalformedDer);
      }
      reader.finish()?;
      Ok(RsaX509PublicKeyAlgorithm::RsaEncryption)
    }
    ID_RSASSA_PSS_OID => {
      if reader.peek_byte().is_none() {
        reader.finish()?;
        return Ok(RsaX509PublicKeyAlgorithm::RsaPss);
      }
      let params = reader.read_constructed(TAG_SEQUENCE)?;
      reader.finish()?;
      let (profile, minimum_salt_len) = parse_x509_pss_parameters(params).map_err(protocol_algorithm_to_key_error)?;
      Ok(RsaX509PublicKeyAlgorithm::RsaPssRestricted {
        profile,
        minimum_salt_len,
      })
    }
    _ => Err(RsaKeyError::UnsupportedAlgorithm),
  }
}

fn parse_x509_signature_algorithm(der: &[u8]) -> Result<RsaSignatureProfile, RsaProtocolAlgorithmError> {
  let mut root = DerReader::new(der);
  let algorithm = protocol_der(root.read_constructed(TAG_SEQUENCE))?;
  protocol_der(root.finish())?;

  let mut reader = DerReader::new(algorithm);
  let oid = protocol_der(reader.read_primitive(TAG_OBJECT_IDENTIFIER))?;
  let profile = match oid {
    SHA256_WITH_RSA_ENCRYPTION_OID => parse_x509_pkcs1v15_algorithm(reader, RsaPkcs1v15Profile::Sha256)?,
    SHA384_WITH_RSA_ENCRYPTION_OID => parse_x509_pkcs1v15_algorithm(reader, RsaPkcs1v15Profile::Sha384)?,
    SHA512_WITH_RSA_ENCRYPTION_OID => parse_x509_pkcs1v15_algorithm(reader, RsaPkcs1v15Profile::Sha512)?,
    ID_RSASSA_PSS_OID => parse_x509_pss_algorithm(reader)?,
    SHA1_WITH_RSA_ENCRYPTION_OID => return Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm),
    _ => return Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm),
  };
  Ok(profile)
}

fn parse_x509_certificate_signature(
  der: &[u8],
) -> Result<ParsedX509CertificateSignature<'_>, RsaProtocolAlgorithmError> {
  let mut root = DerReader::new(der);
  let certificate = protocol_der(root.read_constructed(TAG_SEQUENCE))?;
  protocol_der(root.finish())?;

  let mut certificate = DerReader::new(certificate);
  let tbs_certificate_der = protocol_der(certificate.read_tlv(TAG_SEQUENCE))?;
  let signature_algorithm_der = protocol_der(certificate.read_tlv(TAG_SEQUENCE))?;
  let signature_value = protocol_der(certificate.read_primitive(TAG_BIT_STRING))?;
  protocol_der(certificate.finish())?;

  let tbs_signature_algorithm_der = parse_tbs_certificate_signature_algorithm(tbs_certificate_der)?;
  if tbs_signature_algorithm_der != signature_algorithm_der {
    return Err(RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier);
  }

  let (&unused_bits, signature) = signature_value
    .split_first()
    .ok_or(RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier)?;
  if unused_bits != 0 || signature.is_empty() {
    return Err(RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier);
  }

  Ok(ParsedX509CertificateSignature {
    tbs_certificate_der,
    signature_algorithm_der,
    signature,
  })
}

fn parse_tbs_certificate_signature_algorithm(tbs_certificate_der: &[u8]) -> Result<&[u8], RsaProtocolAlgorithmError> {
  let mut root = DerReader::new(tbs_certificate_der);
  let tbs_certificate = protocol_der(root.read_constructed(TAG_SEQUENCE))?;
  protocol_der(root.finish())?;

  let mut tbs_certificate = DerReader::new(tbs_certificate);
  if tbs_certificate.peek_byte() == Some(TAG_CONTEXT_0) {
    let _ = protocol_der(tbs_certificate.read_constructed(TAG_CONTEXT_0))?;
  }
  let _ = protocol_der(tbs_certificate.read_primitive(TAG_INTEGER))?;
  protocol_der(tbs_certificate.read_tlv(TAG_SEQUENCE))
}

fn parse_x509_pkcs1v15_algorithm(
  mut reader: DerReader<'_>,
  profile: RsaPkcs1v15Profile,
) -> Result<RsaSignatureProfile, RsaProtocolAlgorithmError> {
  if reader.peek_byte().is_some() {
    let null = protocol_der(reader.read_primitive(TAG_NULL))?;
    if !null.is_empty() {
      return Err(RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier);
    }
  }
  protocol_der(reader.finish())?;
  Ok(RsaSignatureProfile::pkcs1v15(profile))
}

fn parse_x509_pss_algorithm(mut reader: DerReader<'_>) -> Result<RsaSignatureProfile, RsaProtocolAlgorithmError> {
  let params = protocol_der(reader.read_constructed(TAG_SEQUENCE))?;
  protocol_der(reader.finish())?;

  let (profile, salt_len) = parse_x509_pss_parameters(params)?;
  Ok(RsaSignatureProfile::pss_with_salt_len(profile, salt_len))
}

fn parse_x509_pss_parameters(params: &[u8]) -> Result<(RsaPssProfile, usize), RsaProtocolAlgorithmError> {
  let mut params = DerReader::new(params);
  let hash = if params.peek_byte() == Some(TAG_CONTEXT_0) {
    let field = protocol_der(params.read_constructed(TAG_CONTEXT_0))?;
    Some(parse_x509_hash_algorithm(field)?)
  } else {
    None
  };
  let mgf = if params.peek_byte() == Some(TAG_CONTEXT_1) {
    let field = protocol_der(params.read_constructed(TAG_CONTEXT_1))?;
    Some(parse_x509_mgf1_algorithm(field)?)
  } else {
    None
  };
  let salt_len = if params.peek_byte() == Some(TAG_CONTEXT_2) {
    let field = protocol_der(params.read_constructed(TAG_CONTEXT_2))?;
    Some(parse_x509_nonnegative_integer(field)?)
  } else {
    None
  };
  if params.peek_byte() == Some(TAG_CONTEXT_3) {
    let field = protocol_der(params.read_constructed(TAG_CONTEXT_3))?;
    if parse_x509_nonnegative_integer(field)? != 1 {
      return Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm);
    }
  }
  protocol_der(params.finish())?;

  let Some(hash) = hash else {
    return Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm);
  };
  let Some(mgf) = mgf else {
    return Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm);
  };
  let salt_len = salt_len.unwrap_or(20);
  if hash != mgf {
    return Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm);
  }
  Ok((hash, salt_len))
}

fn parse_x509_hash_algorithm(der: &[u8]) -> Result<RsaPssProfile, RsaProtocolAlgorithmError> {
  let mut reader = DerReader::new(der);
  let algorithm = protocol_der(reader.read_constructed(TAG_SEQUENCE))?;
  protocol_der(reader.finish())?;
  parse_x509_hash_algorithm_body(algorithm)
}

fn parse_x509_hash_algorithm_body(algorithm: &[u8]) -> Result<RsaPssProfile, RsaProtocolAlgorithmError> {
  let mut algorithm = DerReader::new(algorithm);
  let oid = protocol_der(algorithm.read_primitive(TAG_OBJECT_IDENTIFIER))?;
  let profile = match oid {
    ID_SHA256_OID => RsaPssProfile::Sha256,
    ID_SHA384_OID => RsaPssProfile::Sha384,
    ID_SHA512_OID => RsaPssProfile::Sha512,
    ID_SHA1_OID => return Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm),
    _ => return Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm),
  };
  if algorithm.peek_byte().is_some() {
    let null = protocol_der(algorithm.read_primitive(TAG_NULL))?;
    if !null.is_empty() {
      return Err(RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier);
    }
  }
  protocol_der(algorithm.finish())?;
  Ok(profile)
}

fn parse_x509_mgf1_algorithm(der: &[u8]) -> Result<RsaPssProfile, RsaProtocolAlgorithmError> {
  let mut reader = DerReader::new(der);
  let algorithm = protocol_der(reader.read_constructed(TAG_SEQUENCE))?;
  protocol_der(reader.finish())?;

  let mut algorithm = DerReader::new(algorithm);
  let oid = protocol_der(algorithm.read_primitive(TAG_OBJECT_IDENTIFIER))?;
  if oid != ID_MGF1_OID {
    return Err(RsaProtocolAlgorithmError::UnsupportedAlgorithm);
  }
  let hash = parse_x509_hash_algorithm_body(protocol_der(algorithm.read_constructed(TAG_SEQUENCE))?)?;
  protocol_der(algorithm.finish())?;
  Ok(hash)
}

fn parse_x509_nonnegative_integer(der: &[u8]) -> Result<usize, RsaProtocolAlgorithmError> {
  let mut reader = DerReader::new(der);
  let integer = protocol_der(reader.read_primitive(TAG_INTEGER))?;
  protocol_der(reader.finish())?;
  let integer = protocol_der(read_positive_integer(integer))?;
  let mut value = 0usize;
  for &byte in integer {
    value = value
      .checked_mul(256)
      .and_then(|value| value.checked_add(usize::from(byte)))
      .ok_or(RsaProtocolAlgorithmError::UnsupportedAlgorithm)?;
  }
  Ok(value)
}

fn protocol_der<T>(result: Result<T, RsaKeyError>) -> Result<T, RsaProtocolAlgorithmError> {
  result.map_err(|_| RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier)
}

const fn protocol_algorithm_to_key_error(error: RsaProtocolAlgorithmError) -> RsaKeyError {
  match error {
    RsaProtocolAlgorithmError::MalformedAlgorithmIdentifier => RsaKeyError::MalformedDer,
    RsaProtocolAlgorithmError::UnsupportedAlgorithm => RsaKeyError::UnsupportedAlgorithm,
  }
}

#[cfg(test)]
mod tests;
