//! In-place decryption of one ciphertext across ordered segments.
//!
//! [`open`] passes the whole blocks of each segment to the backend's own
//! kernels. Only a block that crosses a segment boundary is staged, in one
//! clearing owner, and goes through the same kernels. Segment lengths and
//! boundaries are public and select only the control flow.

use crate::{secret::ZeroizingBytes, traits::ct};

/// The total length of `segments`, or `None` when it overflows `usize`.
#[inline]
pub(super) fn total_len(segments: &[&mut [u8]]) -> Option<usize> {
  segments
    .iter()
    .try_fold(0usize, |total, segment| total.checked_add(segment.len()))
}

/// The one segment that holds the whole message, when no other segment holds a byte.
#[inline]
pub(super) fn contiguous<'a>(segments: &'a mut [&mut [u8]]) -> Option<&'a mut [u8]> {
  let mut nonempty = segments.iter_mut().filter(|segment| !segment.is_empty());
  let first = nonempty.next();
  if nonempty.next().is_some() {
    return None;
  }
  Some(first.map_or(&mut [][..], |segment| &mut **segment))
}

/// One nonempty segment as the (pointer, length) pair of a native span-cursor kernel.
#[cfg(all(
  feature = "chacha20poly1305",
  any(
    all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      not(debug_assertions),
      not(feature = "portable-only")
    ),
    all(
      target_arch = "x86_64",
      target_feature = "sse2",
      target_os = "linux",
      all(any(test, not(debug_assertions)), not(feature = "portable-only"))
    )
  )
))]
#[repr(C)]
pub(super) struct Span {
  pointer: *mut u8,
  len: usize,
}

#[cfg(all(
  feature = "chacha20poly1305",
  any(
    all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      not(debug_assertions),
      not(feature = "portable-only")
    ),
    all(
      target_arch = "x86_64",
      target_feature = "sse2",
      target_os = "linux",
      all(any(test, not(debug_assertions)), not(feature = "portable-only"))
    )
  )
))]
impl Span {
  #[inline]
  pub(super) const fn len(&self) -> usize {
    self.len
  }
}

/// Pass the nonempty segments to `native` as spans, in order. Returns `None`,
/// and calls nothing, when more than 128 segments hold bytes. The spans keep
/// the exclusive borrow of `segments` for the call; their metadata is public.
#[cfg(all(
  feature = "chacha20poly1305",
  any(
    all(
      target_arch = "aarch64",
      any(target_os = "macos", target_os = "linux"),
      not(debug_assertions),
      not(feature = "portable-only")
    ),
    all(
      target_arch = "x86_64",
      target_feature = "sse2",
      target_os = "linux",
      all(any(test, not(debug_assertions)), not(feature = "portable-only"))
    )
  )
))]
#[inline]
pub(super) fn spans<T>(segments: &mut [&mut [u8]], native: impl FnOnce(&[Span]) -> T) -> Option<T> {
  const CAPACITY: usize = 128;
  let mut spans = [const { core::mem::MaybeUninit::<Span>::uninit() }; CAPACITY];
  let mut count = 0usize;
  for segment in segments.iter_mut().filter(|segment| !segment.is_empty()) {
    spans.get_mut(count)?.write(Span {
      pointer: segment.as_mut_ptr(),
      len: segment.len(),
    });
    count = count.strict_add(1);
  }
  // SAFETY: the first `count` slots are initialized above.
  let spans = unsafe { core::slice::from_raw_parts(spans.as_ptr().cast::<Span>(), count) };
  Some(native(spans))
}

/// Open the `len`-byte message of `segments`, at most `N` bytes, with the
/// contiguous `open` on a staged copy, and copy its plaintext back on a match.
/// The stage clears on drop.
#[cfg(feature = "chacha20poly1305")]
#[inline]
pub(super) fn staged<const N: usize>(
  segments: &mut [&mut [u8]],
  len: usize,
  open: impl FnOnce(&mut [u8]) -> bool,
) -> bool {
  let mut stage = ZeroizingBytes::<N>::zeroed();
  let message = &mut stage.as_mut_array()[..len];
  visit(segments, 0, 0, len, |part, at| {
    message[at..at.strict_add(part.len())].copy_from_slice(part);
  });
  let matched = open(message);
  if matched {
    visit(segments, 0, 0, len, |part, at| {
      part.copy_from_slice(&message[at..at.strict_add(part.len())]);
    });
  }
  matched
}

/// Return the open's result for `matched`; a mismatch first zeroes every segment.
#[inline]
pub(super) fn verdict(segments: &mut [&mut [u8]], matched: bool) -> Result<(), super::OpenError> {
  if matched {
    return Ok(());
  }
  for segment in segments {
    ct::zeroize(segment);
  }
  Err(super::OpenError::verification())
}

/// Open the `len`-byte ciphertext of `segments` in order and return whether its tag matched.
///
/// `body` decrypts whole `BLOCK`-byte blocks in place and folds their
/// ciphertext into `state`. `finish` takes `state` and the rest of the
/// message, decrypts it, checks the tag, and returns whether it matched. Each
/// call receives contiguous bytes: a segment's own range, or the staged block
/// that crosses a boundary. On a mismatch the segments hold partial plaintext;
/// [`verdict`] zeroes them.
///
/// `len` must equal the total length of `segments`.
#[inline(always)]
pub(super) fn open<const BLOCK: usize, S>(
  segments: &mut [&mut [u8]],
  len: usize,
  mut state: S,
  mut body: impl FnMut(&mut S, &mut [u8]),
  finish: impl FnOnce(S, &mut [u8]) -> bool,
) -> bool {
  let mut staged = ZeroizingBytes::<BLOCK>::zeroed();
  let mut rest = len;
  let mut index = 0usize;
  let mut offset = 0usize;
  loop {
    while let Some(segment) = segments.get(index)
      && offset == segment.len()
    {
      index = index.strict_add(1);
      offset = 0;
    }
    let span = match segments.get_mut(index) {
      Some(segment) => &mut segment[offset..],
      None => &mut [][..],
    };
    if span.len() == rest {
      break finish(state, span);
    }
    let whole = span.len().strict_sub(span.len().strict_rem(BLOCK));
    if whole != 0 {
      body(&mut state, &mut span[..whole]);
      offset = offset.strict_add(whole);
      rest = rest.strict_sub(whole);
      continue;
    }
    // Fewer than BLOCK bytes remain in this segment, and the message continues.
    let take = rest.min(BLOCK);
    let block = &mut staged.as_mut_array()[..take];
    visit(segments, index, offset, take, |part, at| {
      block[at..at.strict_add(part.len())].copy_from_slice(part);
    });
    let last = take == rest;
    if last {
      let matched = finish(state, block);
      if matched {
        visit(segments, index, offset, take, |part, at| {
          part.copy_from_slice(&block[at..at.strict_add(part.len())]);
        });
      }
      break matched;
    }
    body(&mut state, block);
    (index, offset) = visit(segments, index, offset, take, |part, at| {
      part.copy_from_slice(&block[at..at.strict_add(part.len())]);
    });
    rest = rest.strict_sub(take);
  }
}

/// Visit the `len` message bytes from segment `index` at `offset` on, in order.
/// `visit` receives each segment part and its offset within the visited bytes.
/// Returns the position after them.
#[inline(always)]
fn visit(
  segments: &mut [&mut [u8]],
  mut index: usize,
  mut offset: usize,
  len: usize,
  mut visit: impl FnMut(&mut [u8], usize),
) -> (usize, usize) {
  let mut done = 0usize;
  while done < len {
    let segment = &mut segments[index];
    let take = segment.len().strict_sub(offset).min(len.strict_sub(done));
    visit(&mut segment[offset..offset.strict_add(take)], done);
    done = done.strict_add(take);
    offset = offset.strict_add(take);
    if offset == segment.len() {
      index = index.strict_add(1);
      offset = 0;
    }
  }
  (index, offset)
}
