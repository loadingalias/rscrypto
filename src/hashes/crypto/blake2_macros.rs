macro_rules! impl_blake2_params {
  ($params:ident, $key:ident) => {
    impl $params {
      /// Maximum key length (bytes).
      pub const MAX_KEY_LEN: usize = MAX_KEY_LEN;
      /// Maximum salt length (bytes).
      pub const SALT_LEN: usize = SALT_LEN;
      /// Maximum personalization length (bytes).
      pub const PERSONAL_LEN: usize = PERSONAL_LEN;

      /// Create a new params builder with no key, salt, or personalization.
      #[must_use]
      pub const fn new() -> Self {
        Self {
          key_buf: [0u8; MAX_KEY_LEN],
          key_len: 0,
          salt: [0u8; SALT_LEN],
          personal: [0u8; PERSONAL_LEN],
        }
      }

      /// Set a validated MAC key. Omit this method for unkeyed hashing.
      #[must_use]
      pub fn key(mut self, key: $key<'_>) -> Self {
        let key = key.as_bytes();
        self.key_buf = [0u8; MAX_KEY_LEN];
        self.key_buf[..key.len()].copy_from_slice(key);
        self.key_len = u8::try_from(key.len()).expect("validated BLAKE2 key length fits in u8");
        self
      }

      /// Set the RFC 7693 salt field.
      #[must_use]
      pub const fn salt(mut self, salt: [u8; SALT_LEN]) -> Self {
        self.salt = salt;
        self
      }

      /// Set the RFC 7693 personalization field.
      #[must_use]
      pub const fn personal(mut self, personal: [u8; PERSONAL_LEN]) -> Self {
        self.personal = personal;
        self
      }

      fn key_slice(&self) -> &[u8] {
        &self.key_buf[..usize::from(self.key_len)]
      }
    }

    impl Default for $params {
      fn default() -> Self {
        Self::new()
      }
    }

    impl fmt::Debug for $params {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(stringify!($params))
          .field("key_len", &self.key_len)
          .field("salt", &self.salt)
          .field("personal", &self.personal)
          .finish()
      }
    }

    impl Drop for $params {
      fn drop(&mut self) {
        ct::zeroize(&mut self.key_buf);
      }
    }
  };
}

macro_rules! impl_blake2_digest {
  ($digest:ident, $key:ident, $bytes:literal) => {
    impl fmt::Debug for $digest {
      fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(stringify!($digest)).finish_non_exhaustive()
      }
    }

    impl $digest {
      /// Output size in bytes.
      pub const OUTPUT_SIZE: usize = $bytes;

      /// Compute an unkeyed BLAKE2 hash in one shot.
      #[must_use]
      pub fn digest(data: &[u8]) -> [u8; $bytes] {
        oneshot_hash_array::<$bytes>($bytes, &[], data)
      }

      /// Create a keyed BLAKE2 streaming hasher.
      #[must_use]
      pub fn new_keyed(key: $key<'_>) -> Self {
        let key = key.as_bytes();
        Self(Core::new($bytes, key))
      }

      /// Compute a keyed BLAKE2 hash in one shot.
      #[must_use]
      pub fn keyed_digest(key: $key<'_>, data: &[u8]) -> [u8; $bytes] {
        let key = key.as_bytes();
        oneshot_hash_array::<$bytes>($bytes, key, data)
      }
    }

    impl Default for $digest {
      fn default() -> Self {
        Self(Core::new_unkeyed($bytes))
      }
    }

    impl Digest for $digest {
      const OUTPUT_SIZE: usize = $bytes;
      type Output = [u8; $bytes];

      #[inline]
      fn new() -> Self {
        Self(Core::new_unkeyed($bytes))
      }

      #[inline]
      fn update(&mut self, data: &[u8]) {
        self.0.update(data);
      }

      fn finalize(&self) -> Self::Output {
        finalize_array::<$bytes>(&self.0)
      }

      #[inline]
      fn reset(&mut self) {
        self.0.reset();
      }
    }

    impl Drop for $digest {
      fn drop(&mut self) {
        // Core::drop handles zeroization
      }
    }
  };
}
