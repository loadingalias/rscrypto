//! Ciphertext output and partial-keystream ownership for gather encryption.

use core::mem::MaybeUninit;

use super::SealError;
use crate::secret::ZeroizingBytes;

struct Output<'a> {
  storage: &'a mut [MaybeUninit<u8>],
  written: usize,
  complete: bool,
}

impl<'a> Output<'a> {
  #[cfg(all(
    feature = "aes-gcm",
    any(
      all(feature = "aes-gcm", target_arch = "x86_64", target_os = "linux"),
      all(
        feature = "aes-gcm",
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux")
      )
    )
  ))]
  fn initialized_tail(&self, len: usize) -> &[u8] {
    let start = self.written.strict_sub(len);
    // SAFETY: `written` advances only after write_xor or a checked initializer
    // returns the initialized prefix of this exact storage. This shared borrow
    // ends before the next output write.
    unsafe { core::slice::from_raw_parts(self.storage.as_ptr().add(start).cast::<u8>(), len) }
  }

  fn finish(mut self) -> Result<&'a mut [u8], SealError> {
    if self.written != self.storage.len() {
      return Err(SealError::buffer());
    }
    self.complete = true;
    // SAFETY: write_xor and the checked fused initializer advance `written`
    // only across initialized output. The equality covers the complete exclusive
    // allocation. Drop skips clearing only on this successful ownership transfer.
    Ok(unsafe { core::slice::from_raw_parts_mut(self.storage.as_mut_ptr().cast::<u8>(), self.storage.len()) })
  }

  fn write_xor(&mut self, plaintext: &[u8], keystream: &[u8]) {
    let end = self.written.strict_add(plaintext.len());
    for ((out, plain), mask) in self.storage[self.written..end].iter_mut().zip(plaintext).zip(keystream) {
      out.write(plain ^ mask);
    }
    self.written = end;
  }
}

/// Borrowed input span passed to the integrated AArch64 seal boundary.
#[cfg(all(
  target_arch = "aarch64",
  target_endian = "little",
  any(target_os = "macos", target_os = "linux"),
  feature = "chacha20poly1305",
  any(not(debug_assertions), all(test, miri)),
  not(feature = "portable-only")
))]
#[repr(C)]
pub(super) struct Slice<'input> {
  pointer: *const u8,
  len: usize,
  borrow: core::marker::PhantomData<&'input [u8]>,
}

#[cfg(all(
  target_arch = "aarch64",
  target_endian = "little",
  any(target_os = "macos", target_os = "linux"),
  feature = "chacha20poly1305",
  any(not(debug_assertions), all(test, miri)),
  not(feature = "portable-only")
))]
impl<'input> Slice<'input> {
  pub(super) fn as_slice(&self) -> &'input [u8] {
    // SAFETY: construction retains the original shared input lifetime. The
    // pointer and length are copied together and never changed by the kernel.
    unsafe { core::slice::from_raw_parts(self.pointer, self.len) }
  }
}

/// Collect public span metadata for one integrated call, with bounded stack use.
/// A longer iterator continues through the ordinary gather implementation.
#[cfg(all(
  target_arch = "aarch64",
  target_endian = "little",
  any(target_os = "macos", target_os = "linux"),
  feature = "chacha20poly1305",
  any(not(debug_assertions), all(test, miri)),
  not(feature = "portable-only")
))]
pub(super) fn slices<'input, 'output, I, Tag>(
  plaintext: I,
  out: &'output mut [MaybeUninit<u8>],
  native: impl for<'a> FnOnce(&[Slice<'input>], &'a mut [MaybeUninit<u8>]) -> (Tag, &'a mut [u8]),
  fallback: impl for<'a> FnOnce(
    &[Slice<'input>],
    I::IntoIter,
    &'a mut [MaybeUninit<u8>],
  ) -> Result<(Tag, &'a mut [u8]), SealError>,
) -> Result<(Tag, &'output mut [u8]), SealError>
where
  I: IntoIterator<Item = &'input [u8]>,
{
  const CAPACITY: usize = 128;
  let mut output = Output {
    storage: out,
    written: 0,
    complete: false,
  };
  let mut spans: [MaybeUninit<Slice<'input>>; CAPACITY] = [const { MaybeUninit::uninit() }; CAPACITY];
  let mut count = 0;
  let mut total = 0usize;
  let mut chunks = plaintext.into_iter();
  let complete = loop {
    if count == CAPACITY {
      break false;
    }
    let Some(chunk) = chunks.next() else {
      break true;
    };
    if chunk.len() > output.storage.len().strict_sub(total) {
      return Err(SealError::buffer());
    }
    if chunk.is_empty() {
      continue;
    }
    spans[count].write(Slice {
      pointer: chunk.as_ptr(),
      len: chunk.len(),
      borrow: core::marker::PhantomData,
    });
    count = count.strict_add(1);
    total = total.strict_add(chunk.len());
  };
  // SAFETY: only the written prefix is exposed. Every span borrows initialized
  // input disjoint from the exclusive output, and metadata lives through the call.
  let spans = unsafe { core::slice::from_raw_parts(spans.as_ptr().cast::<Slice<'input>>(), count) };
  let pointer = output.storage.as_mut_ptr().cast::<u8>();
  let len = output.storage.len();
  let (tag, ciphertext) = if complete {
    if total != len {
      return Err(SealError::buffer());
    }
    native(spans, output.storage)
  } else {
    fallback(spans, chunks, output.storage)?
  };
  if ciphertext.as_mut_ptr() != pointer || ciphertext.len() != len {
    return Err(SealError::buffer());
  }
  output.written = len;
  Ok((tag, output.finish()?))
}

/// Pass the block-aligned prefix of the first nonempty span directly to an
/// integrated kernel, after the remaining at most `TAIL` bytes are encrypted
/// into their output range through the stream XOR. The kernel writes the
/// prefix ciphertext and authenticates the remainder's ciphertext after it.
/// Other shapes continue through `fallback` with the consumed span first.
/// No plaintext is copied; the output guard clears every slot on any failure.
#[cfg(all(
  target_arch = "x86_64",
  target_feature = "sse2",
  target_os = "linux",
  feature = "chacha20poly1305",
  not(debug_assertions),
  not(miri),
  not(feature = "portable-only")
))]
pub(super) fn head_tail<'input, 'output, I, Tag, const BLOCK: usize, const TAIL: usize>(
  plaintext: I,
  out: &'output mut [MaybeUninit<u8>],
  initial_counter: u32,
  generate: impl FnMut(u32, &mut [u8]) -> Result<(), SealError>,
  native: impl for<'a> FnOnce(&'input [u8], &'a mut [MaybeUninit<u8>], &[u8]) -> (Tag, &'a mut [u8]),
  fallback: impl for<'a> FnOnce(
    core::iter::Chain<core::iter::Once<&'input [u8]>, I::IntoIter>,
    &'a mut [MaybeUninit<u8>],
  ) -> Result<(Tag, &'a mut [u8]), SealError>,
) -> Result<(Tag, &'output mut [u8]), SealError>
where
  I: IntoIterator<Item = &'input [u8]>,
{
  const {
    assert!(BLOCK != 0 && TAIL >= BLOCK && TAIL.is_multiple_of(BLOCK));
  }
  let mut output = Output {
    storage: out,
    written: 0,
    complete: false,
  };
  let len = output.storage.len();
  let pointer = output.storage.as_mut_ptr().cast::<u8>();
  let mut chunks = plaintext.into_iter();
  let first = loop {
    match chunks.next() {
      Some([]) => {}
      Some(chunk) => break chunk,
      None => break &[][..],
    }
  };
  let head_len = first.len().strict_sub(first.len().strict_rem(BLOCK));
  let tag = match len.checked_sub(head_len) {
    Some(tail_len) if head_len != 0 && first.len() <= len && tail_len <= TAIL => {
      let (head, rest) = first.split_at(head_len);
      let (head_out, tail_out) = output.storage.split_at_mut(head_len);
      let counter = u32::try_from(head_len.strict_div(BLOCK))
        .ok()
        .and_then(|blocks| initial_counter.checked_add(blocks))
        .ok_or_else(SealError::too_large)?;
      // The tail checks the remaining slice sum against its exact range.
      let tail = xor::<BLOCK, TAIL>(core::iter::once(rest).chain(chunks), tail_out, counter, generate)?;
      let head_pointer = head_out.as_mut_ptr().cast::<u8>();
      let (tag, ciphertext) = native(head, head_out, tail);
      if ciphertext.as_mut_ptr() != head_pointer || ciphertext.len() != head_len {
        return Err(SealError::buffer());
      }
      tag
    }
    _ => {
      let (tag, ciphertext) = fallback(core::iter::once(first).chain(chunks), output.storage)?;
      if ciphertext.as_mut_ptr() != pointer || ciphertext.len() != len {
        return Err(SealError::buffer());
      }
      tag
    }
  };
  output.written = len;
  Ok((tag, output.finish()?))
}

/// Spans retained by [`head_tail_spans`]: the leading span, four tail spans,
/// and one span that ended the collection early.
#[cfg(all(
  target_arch = "x86_64",
  target_feature = "sse2",
  any(
    feature = "aes-gcm",
    all(feature = "chacha20poly1305", not(feature = "portable-only"))
  )
))]
const SHORT_SPANS: usize = 6;

/// The input that [`head_tail_spans`] passes on when the message has another shape:
/// every collected span, in order, then the rest of the iterator.
#[cfg(all(
  target_arch = "x86_64",
  target_feature = "sse2",
  any(
    feature = "aes-gcm",
    all(feature = "chacha20poly1305", not(feature = "portable-only"))
  )
))]
pub(super) type ShortRest<'input, I> = core::iter::Chain<core::array::IntoIter<&'input [u8], SHORT_SPANS>, I>;

/// Seal a message whose first nonempty span is followed by at most `max_tail`
/// bytes in at most four spans, such as a TLS record's payload and content type.
///
/// `native` receives that span, the tail spans, and the output address, and must
/// initialize the whole output. Other shapes continue through `fallback` with the
/// collected spans first. Only span lengths, which are public, select the path,
/// and this driver copies no plaintext. The output guard clears every slot on a
/// length mismatch, a native error, or an unwind.
#[cfg(all(
  target_arch = "x86_64",
  target_feature = "sse2",
  any(
    feature = "aes-gcm",
    all(feature = "chacha20poly1305", not(feature = "portable-only"))
  )
))]
pub(super) fn head_tail_spans<'input, 'output, I, Tag>(
  plaintext: I,
  out: &'output mut [MaybeUninit<u8>],
  max_tail: usize,
  native: impl FnOnce(&'input [u8], &[&'input [u8]], *mut u8) -> Result<Tag, SealError>,
  fallback: impl FnOnce(
    ShortRest<'input, I::IntoIter>,
    &'output mut [MaybeUninit<u8>],
  ) -> Result<(Tag, &'output mut [u8]), SealError>,
) -> Result<(Tag, &'output mut [u8]), SealError>
where
  I: IntoIterator<Item = &'input [u8]>,
{
  let mut spans: [&'input [u8]; SHORT_SPANS] = [&[]; SHORT_SPANS];
  let mut count = 0usize;
  let mut tail_len = 0usize;
  let mut chunks = plaintext.into_iter();
  let mut output = Output {
    storage: out,
    written: 0,
    complete: false,
  };
  let fits = loop {
    let Some(chunk) = chunks.next() else { break true };
    if chunk.is_empty() {
      continue;
    }
    spans[count] = chunk;
    count = count.strict_add(1);
    if count > 1 {
      tail_len = tail_len.strict_add(chunk.len());
      if count == SHORT_SPANS || tail_len > max_tail {
        break false;
      }
    }
  };
  if !fits {
    // Hand the untouched output to the general driver, which owns its guard.
    output.complete = true;
    let storage = core::mem::take(&mut output.storage);
    return fallback(spans.into_iter().chain(chunks), storage);
  }
  // With no nonempty span the message is empty and `spans[0]` is empty.
  let total = spans[0].len().strict_add(tail_len);
  if total != output.storage.len() {
    return Err(SealError::buffer());
  }
  let tail = spans.get(1..count).unwrap_or_default();
  let tag = native(spans[0], tail, output.storage.as_mut_ptr().cast())?;
  output.written = total;
  Ok((tag, output.finish()?))
}

/// Bounded public descriptors for full native input units. No plaintext is
/// copied. The final null pointer is an initialized, never-dereferenced sentinel
/// for the native kernel's read-cursor advance after its last complete unit.
#[cfg(all(
  feature = "aes-gcm",
  any(
    all(feature = "aes-gcm", target_arch = "x86_64", target_os = "linux"),
    all(
      feature = "aes-gcm",
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux")
    )
  )
))]
pub(super) struct InputBatch<'input, const UNIT: usize> {
  pointers: [MaybeUninit<*const u8>; 129],
  len: usize,
  input: core::marker::PhantomData<&'input [u8; UNIT]>,
}

#[cfg(all(
  feature = "aes-gcm",
  any(
    all(feature = "aes-gcm", target_arch = "x86_64", target_os = "linux"),
    all(
      feature = "aes-gcm",
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux")
    )
  )
))]
impl<'input, const UNIT: usize> InputBatch<'input, UNIT> {
  fn new() -> Self {
    Self {
      pointers: [MaybeUninit::uninit(); 129],
      len: 0,
      input: core::marker::PhantomData,
    }
  }

  #[cfg(any(target_arch = "x86_64", all(test, miri)))]
  #[inline(always)]
  fn extend_x86(&mut self, input: &'input [u8]) -> &'input [u8] {
    // A single native unit needs one descriptor, not a general fill loop.
    // Keep this hot operation visible inside the short-span traversal.
    if input.len() <= UNIT {
      if input.len() == UNIT && self.len < 128 {
        self.pointers[self.len].write(input.as_ptr());
        self.len = self.len.strict_add(1);
        return &input[UNIT..];
      }
      return input;
    }
    self.extend_many(input)
  }

  #[cfg(any(target_arch = "x86_64", all(test, miri)))]
  #[inline(never)]
  fn extend_many(&mut self, input: &'input [u8]) -> &'input [u8] {
    let count = (input.len().strict_div(UNIT)).min(128usize.strict_sub(self.len));
    let end = self.len.strict_add(count);
    for (index, pointer) in self.pointers[self.len..end].iter_mut().enumerate() {
      pointer.write(input[index.strict_mul(UNIT)..].as_ptr());
    }
    self.len = end;
    &input[count.strict_mul(UNIT)..]
  }

  #[cfg(any(not(target_arch = "x86_64"), all(test, miri)))]
  fn extend(&mut self, input: &'input [u8]) -> &'input [u8] {
    let count = (input.len().strict_div(UNIT)).min(128usize.strict_sub(self.len));
    let end = self.len.strict_add(count);
    for (index, pointer) in self.pointers[self.len..end].iter_mut().enumerate() {
      pointer.write(input[index.strict_mul(UNIT)..].as_ptr());
    }
    self.len = end;
    &input[count.strict_mul(UNIT)..]
  }

  fn is_full(&self) -> bool {
    self.len == 128
  }

  pub(super) fn byte_len(&self) -> usize {
    self.len.strict_mul(UNIT)
  }

  pub(super) fn as_ptr(&self) -> *const *const u8 {
    self.pointers.as_ptr().cast()
  }

  fn input(&mut self) -> Input<'_, UNIT> {
    // The native cursor reads one initialized sentinel after the final unit.
    // No other slot is exposed or read. Descriptors contain addresses only.
    self.pointers[self.len].write(core::ptr::null());
    Input::Batched(self)
  }
}

#[cfg(all(
  feature = "aes-gcm",
  any(
    all(feature = "aes-gcm", target_arch = "x86_64", target_os = "linux"),
    all(
      feature = "aes-gcm",
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux")
    )
  )
))]
pub(super) enum Input<'input, const UNIT: usize> {
  Contiguous(&'input [u8]),
  Batched(&'input InputBatch<'input, UNIT>),
}

#[cfg(all(
  feature = "aes-gcm",
  any(
    all(feature = "aes-gcm", target_arch = "x86_64", target_os = "linux"),
    all(
      feature = "aes-gcm",
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux")
    )
  )
))]
impl<const UNIT: usize> Input<'_, UNIT> {
  pub(super) fn byte_len(&self) -> usize {
    match self {
      Self::Contiguous(input) => input.len(),
      Self::Batched(batch) => batch.byte_len(),
    }
  }

  pub(super) fn read_cursor(&self) -> (*const u8, *const *const u8) {
    match self {
      Self::Contiguous(input) => (input.as_ptr(), core::ptr::null()),
      Self::Batched(batch) => (core::ptr::null(), batch.as_ptr()),
    }
  }
}

/// Keep large input spans intact and batch shorter spans for one native call.
/// One zeroizing block carries stream bytes through shorter seams.
#[cfg(all(
  feature = "aes-gcm",
  any(
    all(feature = "aes-gcm", target_arch = "x86_64", target_os = "linux"),
    all(
      feature = "aes-gcm",
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux")
    )
  )
))]
pub(super) fn fused<'input, 'output, const BLOCK: usize, const UNIT: usize, State>(
  plaintext: impl IntoIterator<Item = &'input [u8]>,
  out: &'output mut [MaybeUninit<u8>],
  initial_counter: u32,
  state: &mut State,
  mut encrypt_blocks: impl for<'source, 'a> FnMut(
    &mut State,
    u32,
    &Input<'source, UNIT>,
    &'a mut [MaybeUninit<u8>],
  ) -> Result<&'a mut [u8], SealError>,
  mut generate: impl FnMut(u32, &mut [u8; BLOCK]),
  mut authenticate: impl FnMut(&mut State, &[u8]),
) -> Result<&'output mut [u8], SealError> {
  const {
    assert!(BLOCK != 0 && UNIT >= BLOCK && UNIT.is_multiple_of(BLOCK));
  }
  let mut output = Output {
    storage: out,
    written: 0,
    complete: false,
  };
  let mut stream = ZeroizingBytes::<BLOCK>::zeroed();
  let mut consumed = 0;
  let mut counter = u64::from(initial_counter);
  let mut chunks = plaintext.into_iter();
  let mut chunk: &'input [u8] = &[];

  loop {
    if chunk.is_empty() {
      let Some(next) = chunks.next() else { break };
      chunk = next;
      if chunk.len() > output.storage.len().strict_sub(output.written) {
        return Err(SealError::buffer());
      }
      if chunk.is_empty() {
        continue;
      }
    }
    if consumed == 0 && chunk.len() >= UNIT {
      let mut finished = false;
      let mut batch;
      let contiguous_len = chunk.len().strict_sub(chunk.len().strict_rem(UNIT.strict_mul(8)));
      let prefer_contiguous = !cfg!(target_arch = "x86_64")
        || output
          .storage
          .len()
          .strict_sub(output.written)
          .strict_sub(contiguous_len)
          < UNIT;
      let input = if chunk.len() >= UNIT.strict_mul(16) && prefer_contiguous {
        // Full groups of eight units preserve the existing wide-kernel alignment.
        let group = UNIT.strict_mul(8);
        let len = chunk.len().strict_sub(chunk.len().strict_rem(group));
        let (head, tail) = chunk.split_at(len);
        chunk = tail;
        Input::Contiguous(head)
      } else {
        batch = InputBatch::new();
        loop {
          #[cfg(target_arch = "x86_64")]
          {
            chunk = batch.extend_x86(chunk);
          }
          #[cfg(not(target_arch = "x86_64"))]
          {
            chunk = batch.extend(chunk);
          }
          if batch.is_full() || !chunk.is_empty() {
            break;
          }
          match chunks.next() {
            Some(next) => {
              let remaining = output
                .storage
                .len()
                .strict_sub(output.written)
                .strict_sub(batch.byte_len());
              if next.len() > remaining {
                return Err(SealError::buffer());
              }
              chunk = next;
            }
            None => {
              finished = true;
              break;
            }
          }
        }
        batch.input()
      };
      let len = input.byte_len();
      let end = output.written.strict_add(len);
      let slots = &mut output.storage[output.written..end];
      let ptr = slots.as_mut_ptr().cast::<u8>();
      let ciphertext = encrypt_blocks(
        state,
        u32::try_from(counter).map_err(|_| SealError::too_large())?,
        &input,
        slots,
      )?;
      // The safe initializer proves initialization. Verify its exact output
      // allocation and complete batch before advancing the initialized prefix.
      if ciphertext.as_mut_ptr() != ptr || ciphertext.len() != len {
        return Err(SealError::buffer());
      }
      output.written = end;
      counter = counter.strict_add(len.strict_div(BLOCK) as u64);
      if finished {
        break;
      }
      continue;
    }
    if consumed == 0 {
      generate(
        u32::try_from(counter).map_err(|_| SealError::too_large())?,
        stream.as_mut_array(),
      );
      counter = counter.strict_add(1);
    }
    let take = chunk.len().min(BLOCK.strict_sub(consumed));
    let end = consumed.strict_add(take);
    output.write_xor(&chunk[..take], &stream.as_array()[consumed..end]);
    consumed = end;
    chunk = &chunk[take..];
    if consumed == BLOCK {
      authenticate(state, output.initialized_tail(BLOCK));
      consumed = 0;
    }
  }
  if consumed != 0 {
    authenticate(state, output.initialized_tail(consumed));
  }
  output.finish()
}

impl Drop for Output<'_> {
  fn drop(&mut self) {
    if !self.complete {
      crate::traits::aead::clear_gather_output(self.storage);
    }
  }
}

/// Carry unused keystream across arbitrary input slices. Only public lengths and
/// counters control iteration; plaintext is read once and ciphertext is written once.
pub(super) fn xor<'input, 'output, const BLOCK: usize, const BATCH: usize>(
  plaintext: impl IntoIterator<Item = &'input [u8]>,
  out: &'output mut [MaybeUninit<u8>],
  initial_counter: u32,
  mut generate: impl FnMut(u32, &mut [u8]) -> Result<(), SealError>,
) -> Result<&'output mut [u8], SealError> {
  let mut output = Output {
    storage: out,
    written: 0,
    complete: false,
  };
  let mut stream = ZeroizingBytes::<BATCH>::zeroed();
  let mut available = 0;
  let mut consumed = 0;
  let mut counter = u64::from(initial_counter);

  for mut chunk in plaintext {
    if chunk.len() > output.storage.len().strict_sub(output.written) {
      return Err(SealError::buffer());
    }
    while !chunk.is_empty() {
      if consumed == available {
        let remaining = output.storage.len().strict_sub(output.written);
        available = remaining.min(BATCH).div_ceil(BLOCK).strict_mul(BLOCK);
        consumed = 0;
        generate(
          u32::try_from(counter).map_err(|_| SealError::too_large())?,
          &mut stream.as_mut_array()[..available],
        )?;
        counter = counter.strict_add(available.strict_div(BLOCK) as u64);
      }
      let take = chunk.len().min(available.strict_sub(consumed));
      let end = consumed.strict_add(take);
      output.write_xor(&chunk[..take], &stream.as_array()[consumed..end]);
      consumed = end;
      chunk = &chunk[take..];
    }
  }
  output.finish()
}

#[cfg(all(test, miri))]
mod tests {
  use super::*;
  use crate::{
    AeadGather,
    aead::{Nonce96, expert::AeadWithNonce},
  };

  fn initialized<C: AeadGather<Nonce = Nonce96>>(cipher: C) {
    let plaintext = [0x31; 17];
    let chunks = [&plaintext[..1], &[], &plaintext[1..16], &plaintext[16..]];
    let nonce = Nonce96::from_bytes([0x42; 12]);
    let mut expected = plaintext;
    let expected_tag = cipher
      .encrypt_in_place(&nonce, b"aad", &mut expected)
      .expect("portable control");
    let mut storage = [MaybeUninit::uninit(); 17];
    let (tag, out) = cipher
      .encrypt_gather_uninit(&nonce, b"aad", chunks, &mut storage)
      .expect("initialize complete ciphertext");
    assert_eq!(out, expected);
    assert_eq!(tag.as_ref(), expected_tag.as_ref());

    let mut appended = alloc::vec![9, 7];
    let tag = cipher
      .encrypt_gather_append(&nonce, b"aad", chunks, 17, &mut appended)
      .expect("initialize vector spare capacity");
    assert_eq!(&appended[..2], &[9, 7]);
    assert_eq!(&appended[2..], expected);
    assert_eq!(tag.as_ref(), expected_tag.as_ref());

    let mut short = [MaybeUninit::uninit(); 16];
    assert!(
      cipher
        .encrypt_gather_uninit(&nonce, b"aad", chunks, &mut short)
        .is_err()
    );
    for byte in short {
      // SAFETY: the error contract initializes every supplied output slot to
      // zero on a chunk-length mismatch. Miri checks this production obligation.
      assert_eq!(unsafe { byte.assume_init() }, 0);
    }
  }

  fn unwind<C: AeadGather<Nonce = Nonce96>>(cipher: C) {
    let mut storage = [MaybeUninit::uninit(); 17];
    let chunks = core::iter::once(b"a".as_slice()).chain(core::iter::from_fn(|| -> Option<&[u8]> {
      panic!("iterator stops after one write")
    }));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
      let _ = cipher.encrypt_gather_uninit(&Nonce96::from_bytes([0x42; 12]), b"", chunks, &mut storage);
    }));
    assert!(result.is_err());
    for byte in storage {
      // SAFETY: the output owner initializes and clears all slots while
      // unwinding; Miri reports UB if any unwritten slot escapes that cleanup.
      assert_eq!(unsafe { byte.assume_init() }, 0);
    }
  }

  #[test]
  fn miri_gather_initialization_and_length_failure() {
    #[cfg(any(
      all(target_arch = "x86_64", target_os = "linux"),
      all(
        feature = "aes-gcm",
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux")
      )
    ))]
    {
      let first = [3u8; 256];
      let second = [7u8; 256];
      let mut batch = InputBatch::<256>::new();
      assert!(batch.extend(&first).is_empty());
      assert!(batch.extend(&second).is_empty());
      let input = batch.input();
      let (_, pointers) = input.read_cursor();
      // SAFETY: input exposes exactly two complete borrowed units and the
      // initialized terminal sentinel. Miri checks initialization/provenance.
      unsafe {
        assert_eq!(pointers.read(), first.as_ptr());
        assert_eq!(pointers.add(1).read(), second.as_ptr());
        assert!(pointers.add(2).read().is_null());
      }
      let mut fast = InputBatch::<256>::new();
      assert!(fast.extend_x86(&first).is_empty());
      assert!(fast.extend_x86(&second).is_empty());
      let input = fast.input();
      let (_, pointers) = input.read_cursor();
      // SAFETY: both production metadata paths retain the two borrowed units
      // and initialize exactly one sentinel. Expected addresses are independent.
      unsafe {
        assert_eq!(pointers.read(), first.as_ptr());
        assert_eq!(pointers.add(1).read(), second.as_ptr());
        assert!(pointers.add(2).read().is_null());
      }
      let full = [9u8; 129 * 256];
      let mut batch = InputBatch::<256>::new();
      assert_eq!(batch.extend(&full).len(), 256);
      let input = batch.input();
      let (_, pointers) = input.read_cursor();
      // SAFETY: the bounded production batch retains 128 complete units plus
      // one initialized sentinel; every expected unit is independently sliced.
      unsafe {
        for (index, unit) in full.as_chunks::<256>().0[..128].iter().enumerate() {
          assert_eq!(pointers.add(index).read(), unit.as_ptr());
        }
        assert!(pointers.add(128).read().is_null());
      }
    }
    #[cfg(all(
      target_arch = "aarch64",
      target_endian = "little",
      any(target_os = "macos", target_os = "linux"),
      not(feature = "portable-only")
    ))]
    {
      let cipher = crate::ChaCha20Poly1305::new(&crate::ChaCha20Poly1305Key::from_bytes([0x43; 32]));
      let nonce = Nonce96::from_bytes([0x27; 12]);
      let plain = [0x71; 129];
      let mut expected = plain;
      let expected_tag = cipher
        .encrypt_in_place(&nonce, b"span metadata", &mut expected)
        .unwrap();
      for width in [1, 65] {
        let mut storage = [MaybeUninit::uninit(); 129];
        let (tag, bytes) = slices(
          plain.chunks(width),
          &mut storage,
          |prefix, output| {
            cipher
              .encrypt_gather_uninit(&nonce, b"span metadata", prefix.iter().map(Slice::as_slice), output)
              .unwrap()
          },
          |prefix, rest, output| {
            cipher.encrypt_gather_uninit(
              &nonce,
              b"span metadata",
              prefix.iter().map(Slice::as_slice).chain(rest),
              output,
            )
          },
        )
        .unwrap();
        assert_eq!(tag.as_ref(), expected_tag.as_ref());
        assert_eq!(bytes, expected);
      }
      let mut storage = [MaybeUninit::uninit(); 128];
      assert!(
        slices(
          plain.chunks(65),
          &mut storage,
          |prefix, output| cipher
            .encrypt_gather_uninit(&nonce, b"", prefix.iter().map(Slice::as_slice), output)
            .unwrap(),
          |prefix, rest, output| cipher.encrypt_gather_uninit(
            &nonce,
            b"",
            prefix.iter().map(Slice::as_slice).chain(rest),
            output
          ),
        )
        .is_err()
      );
      for byte in storage {
        // SAFETY: the production output guard initializes every slot on error.
        assert_eq!(unsafe { byte.assume_init() }, 0);
      }
    }
    initialized(crate::Aes128Gcm::new(&crate::Aes128GcmKey::from_bytes([0x11; 16])));
    initialized(crate::Aes256Gcm::new(&crate::Aes256GcmKey::from_bytes([0x11; 32])));
    initialized(crate::ChaCha20Poly1305::new(&crate::ChaCha20Poly1305Key::from_bytes(
      [0x11; 32],
    )));
  }

  #[test]
  fn miri_gather_unwind_initializes_and_clears_output() {
    unwind(crate::Aes128Gcm::new(&crate::Aes128GcmKey::from_bytes([0x11; 16])));
    unwind(crate::Aes256Gcm::new(&crate::Aes256GcmKey::from_bytes([0x11; 32])));
    unwind(crate::ChaCha20Poly1305::new(&crate::ChaCha20Poly1305Key::from_bytes(
      [0x11; 32],
    )));
  }
}
