//! Routable connection IDs of QUIC-LB
//! ([draft-ietf-quic-load-balancers-22](https://datatracker.ietf.org/doc/draft-ietf-quic-load-balancers-22/)).

use core::fmt;

use rscrypto::SecretBytes;
use rscrypto::aead::expert::quic_lb::{QuicLbCipher, QuicLbKey};
use rustls::Error;
use rustls::crypto::{GetRandomFailed, SecureRandom};

/// Length of the cipher buffer: the largest server-ID and nonce total.
const BLOCK_LEN: usize = QuicLbCipher::MAX_PLAINTEXT_LEN;
/// Longest connection ID: the first octet and the largest block, the QUIC v1 limit
/// (RFC 9000, section 17.2).
const MAX_ID_LEN: usize = BLOCK_LEN + 1;
/// Longest server ID: the largest total less the shortest nonce.
const MAX_SERVER_ID_LEN: usize = BLOCK_LEN - QuicLbCipher::MIN_NONCE_LEN;
/// The five low bits of the first octet.
const LOW_BITS: u8 = 0x1f;

/// A QUIC-LB configuration rotation codepoint, 0 through 6 (section 3.1).
///
/// The codepoint 0b111 marks unroutable connection IDs (section 3.2), so no configuration uses it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RotationId(u8);

impl RotationId {
  /// The codepoint `id`, or `None` for 7 and above.
  #[must_use]
  pub const fn new(id: u8) -> Option<Self> {
    if id < 0b111 { Some(Self(id)) } else { None }
  }

  /// The codepoint as an integer.
  #[must_use]
  pub const fn get(self) -> u8 {
    self.0
  }
}

/// What the five low bits of the first octet carry (section 3.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FirstOctet {
  /// The connection-ID length less one, for devices that need IDs to describe their length.
  Length,
  /// Bits drawn from the caller's random source, so that the first octets of a connection's IDs
  /// have no observable relationship.
  Random,
}

/// The 16-octet QUIC-LB key of a configuration.
///
/// The key clears on drop. It is not `Clone`, and its `Debug` output names only the type.
pub struct LoadBalancerKey(QuicLbKey);

impl LoadBalancerKey {
  /// Length of the key.
  pub const LEN: usize = QuicLbKey::LENGTH;

  /// Take the key that the configuration agent distributes.
  #[must_use]
  pub fn from_bytes(bytes: [u8; QuicLbKey::LENGTH]) -> Self {
    Self(QuicLbKey::from_bytes(bytes))
  }

  /// Generate a key from 16 bytes of `random`, for servers that are their own load balancer.
  ///
  /// The source must provide cryptographically secure random bytes. A failure returns the error
  /// without fallback.
  pub fn generate(random: &dyn SecureRandom) -> Result<Self, GetRandomFailed> {
    let key = SecretBytes::<{ QuicLbKey::LENGTH }>::try_fill_with(|out| random.fill(out))?;
    Ok(Self(QuicLbKey::from_bytes(*key.as_bytes())))
  }
}

impl fmt::Debug for LoadBalancerKey {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("LoadBalancerKey").finish_non_exhaustive()
  }
}

/// One encrypted QUIC-LB configuration: issues connection IDs that route to a server, and
/// decodes the server ID of an ID.
///
/// A connection ID is the first octet, then the encrypted server ID and nonce (section 5.2). The
/// first octet carries the rotation codepoint in its three high bits and [`FirstOctet`] in the
/// rest. The encryption is the single AES-128 block of section 5.4.1 when the server ID and nonce
/// total 16 octets, and the four-pass Feistel network of section 5.4.2 otherwise. The ID carries
/// no integrity tag: every ID with this configuration's rotation and enough octets decodes to some
/// server ID, and the caller decides whether that server exists.
///
/// The caller owns the nonces. A nonce must not repeat under one key (section 8.6): a repeated
/// server ID and nonce repeats the connection ID. A counter from a random start meets that rule
/// until it wraps. A nonce and server ID are confidential: the encryption hides the route from
/// observers, and known plaintext weakens the four-pass form (section 8.7).
///
/// [`Self::encode`] and [`Self::decode`] do not allocate, and their timing depends only on the
/// configured lengths. The configuration clears its expanded key on drop. It is not `Clone`. Share
/// one configuration across threads with `Arc`.
///
/// Not implemented, because no consumer needs them: the no-key form, server-use octets after the
/// encrypted block, and issuance of unroutable (0b111) IDs.
pub struct LoadBalancerConfig {
  cipher: QuicLbCipher,
  rotation: RotationId,
  first_octet: FirstOctet,
}

impl fmt::Debug for LoadBalancerConfig {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("LoadBalancerConfig")
      .field("rotation", &self.rotation)
      .field("server_id_len", &self.server_id_len())
      .field("nonce_len", &self.nonce_len())
      .field("first_octet", &self.first_octet)
      .finish_non_exhaustive()
  }
}

impl LoadBalancerConfig {
  /// Build a configuration, or `None` for lengths that section 5.3 refuses: a server ID shorter
  /// than one octet, a nonce shorter than four, or a total above 19.
  #[must_use]
  pub fn new(
    rotation: RotationId,
    server_id_len: usize,
    nonce_len: usize,
    first_octet: FirstOctet,
    key: LoadBalancerKey,
  ) -> Option<Self> {
    Some(Self {
      cipher: QuicLbCipher::new(&key.0, server_id_len, nonce_len)?,
      rotation,
      first_octet,
    })
  }

  /// The rotation codepoint in every issued ID.
  #[must_use]
  pub fn rotation(&self) -> RotationId {
    self.rotation
  }

  /// Length of every server ID.
  #[must_use]
  pub fn server_id_len(&self) -> usize {
    self.cipher.server_id_len()
  }

  /// Length of every nonce.
  #[must_use]
  pub fn nonce_len(&self) -> usize {
    self.cipher.nonce_len()
  }

  /// Length of every issued connection ID: the first octet, the server ID, and the nonce.
  #[must_use]
  pub fn connection_id_len(&self) -> usize {
    self.plaintext_len().strict_add(1)
  }

  fn plaintext_len(&self) -> usize {
    self.server_id_len().strict_add(self.nonce_len())
  }

  /// Issue the connection ID that routes to `server_id` with `nonce`.
  ///
  /// With [`FirstOctet::Random`], draws one byte from `random` for the low bits of the first
  /// octet. With [`FirstOctet::Length`], draws nothing.
  ///
  /// # Errors
  ///
  /// Returns [`Error::EncryptError`] if `server_id` or `nonce` differs from the configured length,
  /// before drawing entropy, and [`Error::FailedToGetRandomBytes`] if the source fails. Nothing is
  /// issued after an error.
  pub fn encode(&self, server_id: &[u8], nonce: &[u8], random: &dyn SecureRandom) -> Result<ConnectionId, Error> {
    let server_id_len = self.server_id_len();
    if server_id.len() != server_id_len || nonce.len() != self.nonce_len() {
      return Err(Error::EncryptError);
    }
    let len = self.plaintext_len();
    let low_bits = match self.first_octet {
      FirstOctet::Length => u8::try_from(len).expect("QUIC-LB lengths fit one octet"),
      FirstOctet::Random => {
        let mut byte = [0u8; 1];
        random.fill(&mut byte)?;
        byte[0] & LOW_BITS
      }
    };

    let mut block = [0u8; BLOCK_LEN];
    block[..server_id_len].copy_from_slice(server_id);
    block[server_id_len..len].copy_from_slice(nonce);
    self.cipher.encrypt(&mut block);

    let mut id = ConnectionId {
      bytes: [0; MAX_ID_LEN],
      len: u8::try_from(len.strict_add(1)).expect("QUIC-LB lengths fit one octet"),
    };
    id.bytes[0] = (self.rotation.0 << 5) | low_bits;
    id.bytes[1..=len].copy_from_slice(&block[..len]);
    Ok(id)
  }

  /// The server ID that `connection_id` routes to.
  ///
  /// Returns `None`, with no other detail, when the ID is unroutable for this configuration: its
  /// rotation codepoint is 0b111 or another configuration's, or it is shorter than
  /// [`Self::connection_id_len`] (section 4.1). Octets after that length are server-use octets
  /// and are ignored. A caller with several configurations, for example during a rotation, tries
  /// each one. Each configuration refuses the other rotations from the first octet alone.
  #[must_use]
  pub fn decode(&self, connection_id: &[u8]) -> Option<ServerId> {
    let (&first, rest) = connection_id.split_first()?;
    if first >> 5 != self.rotation.0 {
      return None;
    }
    let len = self.plaintext_len();
    let mut block = [0u8; BLOCK_LEN];
    block[..len].copy_from_slice(rest.get(..len)?);
    self.cipher.decrypt_server_id(&mut block);

    let server_id_len = self.server_id_len();
    let mut server_id = ServerId {
      bytes: [0; MAX_SERVER_ID_LEN],
      len: u8::try_from(server_id_len).expect("QUIC-LB lengths fit one octet"),
    };
    server_id.bytes[..server_id_len].copy_from_slice(&block[..server_id_len]);
    Some(server_id)
  }
}

/// A connection ID that a [`LoadBalancerConfig`] issued, stored inline.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConnectionId {
  bytes: [u8; MAX_ID_LEN],
  len: u8,
}

impl ConnectionId {
  /// The ID's octets.
  #[must_use]
  pub fn as_slice(&self) -> &[u8] {
    &self.bytes[..usize::from(self.len)]
  }
}

impl AsRef<[u8]> for ConnectionId {
  fn as_ref(&self) -> &[u8] {
    self.as_slice()
  }
}

impl fmt::Debug for ConnectionId {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("ConnectionId(")?;
    write_hex(f, self.as_slice())?;
    f.write_str(")")
  }
}

/// A server ID that a [`LoadBalancerConfig`] decoded, stored inline.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ServerId {
  bytes: [u8; MAX_SERVER_ID_LEN],
  len: u8,
}

impl ServerId {
  /// The server ID's octets.
  #[must_use]
  pub fn as_slice(&self) -> &[u8] {
    &self.bytes[..usize::from(self.len)]
  }
}

impl AsRef<[u8]> for ServerId {
  fn as_ref(&self) -> &[u8] {
    self.as_slice()
  }
}

impl fmt::Debug for ServerId {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("ServerId(")?;
    write_hex(f, self.as_slice())?;
    f.write_str(")")
  }
}

fn write_hex(f: &mut fmt::Formatter<'_>, bytes: &[u8]) -> fmt::Result {
  bytes.iter().try_for_each(|byte| write!(f, "{byte:02x}"))
}
