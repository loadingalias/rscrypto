//! Canonical DER reading for key, signature, and certificate encodings.
//!
//! The reader accepts only definite, minimally encoded lengths and single-byte
//! tags. Each primitive family reports a rejection through its own public
//! malformed-encoding error.

pub(crate) const TAG_SEQUENCE: u8 = 0x30;
pub(crate) const TAG_BIT_STRING: u8 = 0x03;
pub(crate) const TAG_OBJECT_IDENTIFIER: u8 = 0x06;

/// A family error that reports malformed or non-canonical DER.
pub(crate) trait MalformedDer: Copy {
  /// The error returned for every reader rejection.
  const MALFORMED_DER: Self;
}

/// Sequential reader over one DER element's contents.
pub(crate) struct DerReader<'a, E> {
  input: &'a [u8],
  offset: usize,
  error: core::marker::PhantomData<E>,
}

impl<'a, E: MalformedDer> DerReader<'a, E> {
  pub(crate) const fn new(input: &'a [u8]) -> Self {
    Self {
      input,
      offset: 0,
      error: core::marker::PhantomData,
    }
  }

  /// Return the next tag byte without consuming it.
  pub(crate) fn peek_byte(&self) -> Option<u8> {
    self.input.get(self.offset).copied()
  }

  pub(crate) fn read_constructed(&mut self, tag: u8) -> Result<&'a [u8], E> {
    self.read_primitive(tag)
  }

  /// Read one element and return its complete tag-length-value encoding.
  #[cfg(feature = "rsa")]
  pub(crate) fn read_tlv(&mut self, tag: u8) -> Result<&'a [u8], E> {
    let start = self.offset;
    let _ = self.read_primitive(tag)?;
    self.input.get(start..self.offset).ok_or(E::MALFORMED_DER)
  }

  pub(crate) fn read_primitive(&mut self, tag: u8) -> Result<&'a [u8], E> {
    let actual = self.read_byte()?;
    if actual != tag {
      return Err(E::MALFORMED_DER);
    }
    let len = self.read_len()?;
    let end = self.offset.checked_add(len).ok_or(E::MALFORMED_DER)?;
    if end > self.input.len() {
      return Err(E::MALFORMED_DER);
    }
    let value = self.input.get(self.offset..end).ok_or(E::MALFORMED_DER)?;
    self.offset = end;
    Ok(value)
  }

  /// Reject any input left after the elements already read.
  pub(crate) fn finish(&self) -> Result<(), E> {
    if self.offset == self.input.len() {
      Ok(())
    } else {
      Err(E::MALFORMED_DER)
    }
  }

  fn read_byte(&mut self) -> Result<u8, E> {
    let byte = *self.input.get(self.offset).ok_or(E::MALFORMED_DER)?;
    self.offset = self.offset.strict_add(1);
    Ok(byte)
  }

  fn read_len(&mut self) -> Result<usize, E> {
    let first = self.read_byte()?;
    if first & 0x80 == 0 {
      return Ok(usize::from(first));
    }

    let len_len = usize::from(first & 0x7f);
    if len_len == 0 || len_len > core::mem::size_of::<usize>() {
      return Err(E::MALFORMED_DER);
    }

    let first_len_byte = self.read_byte()?;
    if first_len_byte == 0 {
      return Err(E::MALFORMED_DER);
    }

    let mut len = usize::from(first_len_byte);
    for _ in 1..len_len {
      len = len.checked_shl(8).ok_or(E::MALFORMED_DER)?;
      len |= usize::from(self.read_byte()?);
    }

    if len < 128 {
      return Err(E::MALFORMED_DER);
    }
    Ok(len)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[derive(Clone, Copy, Debug, Eq, PartialEq)]
  struct Malformed;

  impl MalformedDer for Malformed {
    const MALFORMED_DER: Self = Self;
  }

  type Reader<'a> = DerReader<'a, Malformed>;

  #[test]
  fn accepts_canonical_lengths() {
    for len in [0, 1, 127] {
      let encoded = [len];
      let mut reader = Reader::new(&encoded);
      assert_eq!(reader.read_len(), Ok(usize::from(len)));
      assert_eq!(reader.finish(), Ok(()));
    }

    for (encoded, expected) in [
      (&[0x81, 0x80][..], 128),
      (&[0x81, 0xff][..], 255),
      (&[0x82, 0x01, 0x00][..], 256),
    ] {
      let mut reader = Reader::new(encoded);
      assert_eq!(reader.read_len(), Ok(expected));
      assert_eq!(reader.finish(), Ok(()));
    }
  }

  #[test]
  fn rejects_noncanonical_lengths() {
    for encoded in [
      &[0x80, 0x80][..],
      &[0x81, 0x00][..],
      &[0x81, 0x7f][..],
      &[0x82, 0x00, 0x80][..],
    ] {
      let mut reader = Reader::new(encoded);
      assert_eq!(reader.read_len(), Err(Malformed));
    }

    let length_bytes = u8::try_from(core::mem::size_of::<usize>()).expect("usize width fits in one DER length byte");
    let oversized_len_len = [0x80 | length_bytes.strict_add(1)];
    let mut reader = Reader::new(&oversized_len_len);
    assert_eq!(reader.read_len(), Err(Malformed));
  }
}
