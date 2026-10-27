//! Authenticated encryption with associated data traits.
//!
//! The core contract mirrors the rest of rscrypto:
//! explicit typed construction, byte-slice inputs, caller-provided output
//! buffers, and opaque verification failures.

#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};
use core::{fmt::Debug, mem::MaybeUninit};

#[cfg(feature = "getrandom")]
use crate::aead::RandomSealError;
use crate::aead::{AeadBufferError, AeadNonce, OpenError, SealError};

/// Zero-sized capability proving that a call came through a crate-owned nonce
/// issuer or the explicit expert extension.
#[doc(hidden)]
#[derive(Debug)]
pub struct SealToken(());

impl SealToken {
  const fn new() -> Self {
    Self(())
  }
}

/// Authenticated encryption with associated data.
///
/// The in-place methods mutate the payload buffer and return or consume the tag
/// separately. Combined ciphertext-plus-tag helpers are layered on top through
/// caller-provided output buffers so the trait remains `no_std` and `no_alloc`
/// friendly.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "getrandom")]
/// # {
/// use rscrypto::{Aead, ChaCha20Poly1305, ChaCha20Poly1305Key};
///
/// let key = ChaCha20Poly1305Key::from_bytes([0x42; 32]);
/// let cipher = ChaCha20Poly1305::new(&key);
///
/// // In-place encrypt with a fresh nonce and detached tag.
/// let mut buf = *b"hello";
/// let (nonce, tag) = cipher.seal_random_in_place(b"aad", &mut buf)?;
///
/// // In-place decrypt.
/// cipher.decrypt_in_place(&nonce, b"aad", &mut buf, &tag)?;
/// assert_eq!(&buf, b"hello");
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait Aead {
  /// Key size in bytes.
  const KEY_SIZE: usize;

  /// Nonce size in bytes.
  const NONCE_SIZE: usize;

  /// Authentication tag size in bytes.
  const TAG_SIZE: usize;

  /// Algorithm-specific key wrapper.
  type Key;

  /// Algorithm-specific nonce wrapper.
  type Nonce: AeadNonce;

  /// Algorithm-specific tag wrapper.
  type Tag: Copy + Debug + AsRef<[u8]>;

  /// Construct a new AEAD instance from `key`.
  #[must_use]
  fn new(key: &Self::Key) -> Self;

  /// Rebuild a typed tag from raw tag bytes.
  ///
  /// Implementations must reject slices whose length is not exactly
  /// [`TAG_SIZE`](Self::TAG_SIZE).
  fn tag_from_slice(bytes: &[u8]) -> Result<Self::Tag, AeadBufferError>;

  /// Internal sealing primitive used by fresh-nonce and expert APIs.
  #[doc(hidden)]
  fn __encrypt_in_place_with_nonce(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
    token: SealToken,
  ) -> Result<Self::Tag, SealError>;

  /// Encrypt `plaintext` into `out` as `ciphertext || tag` with a fresh random nonce.
  ///
  /// This is the default one-shot sealing API when the `getrandom` feature is
  /// available. The returned nonce must be stored or transmitted alongside the
  /// ciphertext and must be supplied to [`decrypt`](Self::decrypt).
  ///
  /// # Errors
  ///
  /// Returns [`RandomSealError`] if the operating-system entropy source fails,
  /// `out.len()` does not match `plaintext.len() + TAG_SIZE`, that addition
  /// overflows, or the input exceeds the algorithm's supported length bound.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[inline]
  fn seal_random(&self, aad: &[u8], plaintext: &[u8], out: &mut [u8]) -> Result<Self::Nonce, RandomSealError> {
    let nonce = Self::Nonce::try_random()?;
    AeadWithNonce::encrypt(self, &nonce, aad, plaintext, out)?;
    Ok(nonce)
  }

  /// Allocate `ciphertext || tag`, encrypt `plaintext` into it with a fresh random nonce, and
  /// return both values.
  ///
  /// Prefer the caller-buffer [`seal_random`](Self::seal_random) API for
  /// allocation-free paths. This helper exists for application layers that
  /// already own an allocator and want the common AEAD one-shot shape.
  ///
  /// # Errors
  ///
  /// Returns [`RandomSealError`] if nonce generation fails, allocation length
  /// calculation overflows, the output length would be invalid, or sealing
  /// fails.
  #[cfg(all(feature = "alloc", feature = "getrandom"))]
  #[cfg_attr(docsrs, doc(cfg(all(feature = "alloc", feature = "getrandom"))))]
  #[inline]
  fn seal_random_to_vec(&self, aad: &[u8], plaintext: &[u8]) -> Result<(Self::Nonce, Vec<u8>), RandomSealError> {
    let len = Self::ciphertext_len(plaintext.len()).map_err(SealError::from)?;
    let mut out = vec![0u8; len];
    match self.seal_random(aad, plaintext, &mut out) {
      Ok(nonce) => Ok((nonce, out)),
      Err(err) => {
        super::ct::zeroize(&mut out);
        Err(err)
      }
    }
  }

  /// Encrypt `buffer` in place with a fresh random nonce and return the nonce and detached tag.
  ///
  /// The returned nonce must be stored or transmitted alongside the ciphertext
  /// and supplied to [`decrypt_in_place`](Self::decrypt_in_place).
  ///
  /// # Errors
  ///
  /// Returns [`RandomSealError`] if the operating-system entropy source fails
  /// or the input exceeds the algorithm's supported length bound.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[inline]
  fn seal_random_in_place(&self, aad: &[u8], buffer: &mut [u8]) -> Result<(Self::Nonce, Self::Tag), RandomSealError> {
    let nonce = Self::Nonce::try_random()?;
    let tag = self.__encrypt_in_place_with_nonce(&nonce, aad, buffer, SealToken::new())?;
    Ok((nonce, tag))
  }

  /// Decrypt `buffer` in place and verify the detached authentication tag.
  ///
  /// # Errors
  ///
  /// Returns [`OpenError`] when authentication fails or the input exceeds the
  /// algorithm's supported length bound.
  fn decrypt_in_place(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Self::Tag,
  ) -> Result<(), OpenError>;

  /// Alias for [`decrypt_in_place`](Self::decrypt_in_place).
  #[inline]
  fn decrypt_in_place_detached(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
    tag: &Self::Tag,
  ) -> Result<(), OpenError> {
    self.decrypt_in_place(nonce, aad, buffer, tag)
  }

  /// Return the required combined ciphertext-plus-tag length.
  ///
  /// # Errors
  ///
  /// Returns [`AeadBufferError`] if the requested length overflows `usize`.
  #[inline]
  fn ciphertext_len(plaintext_len: usize) -> Result<usize, AeadBufferError> {
    plaintext_len
      .checked_add(Self::TAG_SIZE)
      .ok_or_else(AeadBufferError::new)
  }

  /// Return the plaintext length contained in a combined ciphertext-plus-tag.
  ///
  /// # Errors
  ///
  /// Returns [`AeadBufferError`] if `ciphertext_and_tag_len` is smaller than
  /// [`TAG_SIZE`](Self::TAG_SIZE).
  #[inline]
  fn plaintext_len(ciphertext_and_tag_len: usize) -> Result<usize, AeadBufferError> {
    if ciphertext_and_tag_len < Self::TAG_SIZE {
      return Err(AeadBufferError::new());
    }

    Ok(ciphertext_and_tag_len.strict_sub(Self::TAG_SIZE))
  }

  /// Decrypt a combined `ciphertext || tag` into `out`.
  ///
  /// # Errors
  ///
  /// Returns [`OpenError`] if the input is malformed, `out` has the wrong
  /// length, or authentication fails.
  #[inline]
  fn decrypt(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    ciphertext_and_tag: &[u8],
    out: &mut [u8],
  ) -> Result<(), OpenError> {
    let plaintext_len = Self::plaintext_len(ciphertext_and_tag.len())?;
    if out.len() != plaintext_len {
      return Err(OpenError::buffer());
    }

    let (ciphertext, tag_bytes) = ciphertext_and_tag.split_at(plaintext_len);
    out.copy_from_slice(ciphertext);
    let tag = Self::tag_from_slice(tag_bytes)?;
    if let Err(e) = self.decrypt_in_place(nonce, aad, out, &tag) {
      super::ct::zeroize(out);
      return Err(e);
    }
    Ok(())
  }

  /// Allocate plaintext output and decrypt a combined `ciphertext || tag`.
  ///
  /// # Errors
  ///
  /// Returns [`OpenError`] if the input is malformed or authentication fails.
  /// Any allocated plaintext buffer is cleared before returning an error.
  #[cfg(feature = "alloc")]
  #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
  #[inline]
  fn decrypt_to_vec(&self, nonce: &Self::Nonce, aad: &[u8], ciphertext_and_tag: &[u8]) -> Result<Vec<u8>, OpenError> {
    let len = Self::plaintext_len(ciphertext_and_tag.len())?;
    let mut out = vec![0u8; len];
    match self.decrypt(nonce, aad, ciphertext_and_tag, &mut out) {
      Ok(()) => Ok(out),
      Err(err) => {
        super::ct::zeroize(&mut out);
        Err(err)
      }
    }
  }
}

/// AEAD encryption from ordered plaintext slices into one ciphertext buffer.
///
/// Slice boundaries do not change the message, counter sequence, padding, or tag.
/// AES-128-GCM, AES-256-GCM, and ChaCha20-Poly1305 implement this capability.
/// Encryption reads the plaintext slices directly and does not stage plaintext
/// in the output. The core operation needs no allocation.
///
/// ```
/// # #[cfg(feature = "aes-gcm")]
/// # {
/// use rscrypto::{Aes128Gcm, Aes128GcmKey, aead::NonceCounter};
///
/// let cipher = Aes128Gcm::new(&Aes128GcmKey::from_bytes([0x42; 16]));
/// let mut nonces = NonceCounter::<Aes128Gcm>::new(*b"sess");
/// let mut ciphertext = [0; 5];
/// let (nonce, tag) = nonces.encrypt_gather(
///     &cipher, b"header", [b"he".as_slice(), b"llo".as_slice()], &mut ciphertext,
/// )?;
/// cipher.decrypt_in_place(&nonce, b"header", &mut ciphertext, &tag)?;
/// assert_eq!(&ciphertext, b"hello");
/// # }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait AeadGather: Aead {
  /// Internal public-length validation before output allocation or mutation.
  #[doc(hidden)]
  fn __validate_gather_lengths(aad_len: usize, plaintext_len: usize) -> Result<(), SealError>;

  /// Internal gather encryption used by fresh-nonce and expert APIs.
  #[doc(hidden)]
  fn __encrypt_gather_with_nonce<'input, 'output>(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    plaintext: impl IntoIterator<Item = &'input [u8]>,
    out: &'output mut [MaybeUninit<u8>],
    token: SealToken,
  ) -> Result<(Self::Tag, &'output mut [u8]), SealError>;

  /// Encrypt ordered plaintext slices with a fresh random nonce.
  ///
  /// `out` holds ciphertext only. Its length must equal the sum of the slice
  /// lengths. The returned nonce and detached tag must accompany the ciphertext.
  /// Empty slices are allowed, including an empty iterator for an empty message.
  ///
  /// # Errors
  ///
  /// Returns [`RandomSealError`] if nonce generation fails, an input length is
  /// unsupported, or the output length differs from the plaintext length.
  /// Entropy or input-limit rejection leaves `out` unchanged. A slice-length
  /// mismatch clears `out`, including any ciphertext already written.
  #[cfg(feature = "getrandom")]
  #[cfg_attr(docsrs, doc(cfg(feature = "getrandom")))]
  #[inline]
  fn seal_random_gather<'input>(
    &self,
    aad: &[u8],
    plaintext: impl IntoIterator<Item = &'input [u8]>,
    out: &mut [u8],
  ) -> Result<(Self::Nonce, Self::Tag), RandomSealError> {
    let nonce = Self::Nonce::try_random()?;
    let tag = AeadWithNonce::encrypt_gather(self, &nonce, aad, plaintext, out)?;
    Ok((nonce, tag))
  }
}

/// Initialize and clear every writable output slot without reading its old value.
#[inline]
pub(crate) fn clear_gather_output(output: &mut [MaybeUninit<u8>]) {
  for byte in output {
    // SAFETY: an exclusive MaybeUninit<u8> slot is writable regardless of its
    // prior initialization state. This store initializes it without reading it.
    unsafe { core::ptr::write_volatile(byte.as_mut_ptr(), 0) };
  }
  core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

struct InitializedGatherOutput<'a> {
  storage: &'a mut [MaybeUninit<u8>],
  complete: bool,
}

impl Drop for InitializedGatherOutput<'_> {
  fn drop(&mut self) {
    if !self.complete {
      clear_gather_output(self.storage);
    }
  }
}

/// Expert extension for protocols that define their own nonce derivation.
///
/// Import this trait explicitly from `rscrypto::aead::expert` only when the
/// protocol, persistent state, or test vector guarantees nonce uniqueness.
/// Reusing a nonce with the same key can destroy confidentiality and
/// authenticity.
pub trait AeadWithNonce: Aead {
  /// Encrypt ordered plaintext slices with a caller-supplied nonce.
  ///
  /// `out` holds ciphertext only; the tag is returned separately. The complete
  /// plaintext is the concatenation of the slices, including empty slices.
  /// A partial cipher block carries across slice boundaries without padding.
  ///
  /// # Errors
  ///
  /// Returns [`SealError`] if an input length is unsupported or `out.len()`
  /// differs from the sum of the slice lengths. Input-limit rejection leaves
  /// `out` unchanged. A slice-length mismatch clears the complete output.
  #[inline]
  fn encrypt_gather<'input>(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    plaintext: impl IntoIterator<Item = &'input [u8]>,
    out: &mut [u8],
  ) -> Result<Self::Tag, SealError>
  where
    Self: AeadGather,
  {
    Self::__validate_gather_lengths(aad.len(), out.len())?;
    let ptr = out.as_mut_ptr();
    let len = out.len();
    // SAFETY: MaybeUninit<u8> has the size and alignment of u8. The exclusive
    // borrow covers the same allocation. The guard restores initialization on
    // error or unwind even if a safe external implementation deinitializes it.
    // Success requires a valid initialized slice covering this exact allocation.
    let storage = unsafe { core::slice::from_raw_parts_mut(ptr.cast::<MaybeUninit<u8>>(), len) };
    let mut output = InitializedGatherOutput {
      storage,
      complete: false,
    };
    let (tag, initialized) = self.encrypt_gather_uninit(nonce, aad, plaintext, output.storage)?;
    if initialized.as_mut_ptr() != ptr || initialized.len() != len {
      return Err(SealError::buffer());
    }
    output.complete = true;
    Ok(tag)
  }

  /// Encrypt ordered plaintext slices directly into possibly uninitialized storage.
  ///
  /// This supports caller-owned storage such as `Vec::spare_capacity_mut()`
  /// without first initializing or copying plaintext into it. On success the
  /// returned slice covers the complete initialized ciphertext; the detached tag
  /// is returned separately. The storage length must equal the plaintext length.
  /// The caller must still guarantee nonce uniqueness for the key.
  ///
  /// # Errors
  ///
  /// Returns [`SealError`] for the same conditions as [`encrypt_gather`](Self::encrypt_gather).
  /// Input-limit rejection leaves storage unchanged. A slice-length mismatch
  /// initializes the complete output to zero before returning an error.
  #[inline]
  fn encrypt_gather_uninit<'input, 'output>(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    plaintext: impl IntoIterator<Item = &'input [u8]>,
    out: &'output mut [MaybeUninit<u8>],
  ) -> Result<(Self::Tag, &'output mut [u8]), SealError>
  where
    Self: AeadGather,
  {
    let _dit = super::ct::DataIndependentTiming::enter();
    self.__encrypt_gather_with_nonce(nonce, aad, plaintext, out, SealToken::new())
  }

  /// Append gathered ciphertext to `out` and return its detached tag.
  ///
  /// `plaintext_len` is the sum of the input slice lengths. Existing output
  /// bytes are preserved. New ciphertext is written directly into spare capacity.
  /// The vector grows only after the complete ciphertext is initialized.
  ///
  /// # Errors
  ///
  /// Returns [`SealError`] if the resulting vector length exceeds `isize::MAX`, the input exceeds its
  /// supported bound, or the slice lengths differ from `plaintext_len`.
  /// On error the vector keeps its original length and existing bytes.
  #[cfg(feature = "alloc")]
  #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
  #[inline]
  fn encrypt_gather_append<'input>(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    plaintext: impl IntoIterator<Item = &'input [u8]>,
    plaintext_len: usize,
    out: &mut Vec<u8>,
  ) -> Result<Self::Tag, SealError>
  where
    Self: AeadGather,
  {
    let end = out
      .len()
      .checked_add(plaintext_len)
      .filter(|end| *end <= isize::MAX.cast_unsigned())
      .ok_or_else(SealError::buffer)?;
    Self::__validate_gather_lengths(aad.len(), plaintext_len)?;
    out.reserve(plaintext_len);
    let spare = &mut out.spare_capacity_mut()[..plaintext_len];
    let expected = spare.as_ptr().cast::<u8>();
    let (tag, initialized) = self.encrypt_gather_uninit(nonce, aad, plaintext, spare)?;
    // A third-party implementation of this safe trait can return another slice.
    // Establish the initialized allocation and full length before extending Vec.
    if initialized.as_ptr() != expected || initialized.len() != plaintext_len {
      return Err(SealError::buffer());
    }
    // SAFETY: the returned initialized slice covers exactly the reserved tail
    // of this vector, as checked above. Existing bytes and allocation are intact.
    unsafe { out.set_len(end) };
    Ok(tag)
  }

  /// Encrypt `buffer` in place with a caller-supplied nonce.
  ///
  /// # Errors
  ///
  /// Returns [`SealError`] if the input exceeds the algorithm's supported
  /// length bound.
  #[inline]
  fn encrypt_in_place(&self, nonce: &Self::Nonce, aad: &[u8], buffer: &mut [u8]) -> Result<Self::Tag, SealError> {
    self.__encrypt_in_place_with_nonce(nonce, aad, buffer, SealToken::new())
  }

  /// Alias for [`encrypt_in_place`](Self::encrypt_in_place).
  #[inline]
  fn encrypt_in_place_detached(
    &self,
    nonce: &Self::Nonce,
    aad: &[u8],
    buffer: &mut [u8],
  ) -> Result<Self::Tag, SealError> {
    self.encrypt_in_place(nonce, aad, buffer)
  }

  /// Encrypt `plaintext` into `out` as `ciphertext || tag` with a caller-supplied nonce.
  ///
  /// # Errors
  ///
  /// Returns [`SealError`] if `out.len()` does not match `plaintext.len() +
  /// TAG_SIZE`, that addition overflows, or the input exceeds the algorithm's
  /// supported length bound.
  #[inline]
  fn encrypt(&self, nonce: &Self::Nonce, aad: &[u8], plaintext: &[u8], out: &mut [u8]) -> Result<(), SealError> {
    let expected = Self::ciphertext_len(plaintext.len()).map_err(SealError::from)?;
    if out.len() != expected {
      return Err(SealError::buffer());
    }

    let (ciphertext, tag_out) = out.split_at_mut(plaintext.len());
    ciphertext.copy_from_slice(plaintext);
    let tag = match self.encrypt_in_place(nonce, aad, ciphertext) {
      Ok(tag) => tag,
      Err(err) => {
        super::ct::zeroize(out);
        return Err(err);
      }
    };
    tag_out.copy_from_slice(tag.as_ref());
    Ok(())
  }

  /// Allocate `ciphertext || tag` and encrypt with a caller-supplied nonce.
  ///
  /// # Errors
  ///
  /// Returns [`SealError`] if length calculation fails or encryption fails.
  #[cfg(feature = "alloc")]
  #[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
  #[inline]
  fn encrypt_to_vec(&self, nonce: &Self::Nonce, aad: &[u8], plaintext: &[u8]) -> Result<Vec<u8>, SealError> {
    let len = Self::ciphertext_len(plaintext.len()).map_err(SealError::from)?;
    let mut out = vec![0u8; len];
    match self.encrypt(nonce, aad, plaintext, &mut out) {
      Ok(()) => Ok(out),
      Err(err) => {
        super::ct::zeroize(&mut out);
        Err(err)
      }
    }
  }
}

impl<T: Aead + ?Sized> AeadWithNonce for T {}
