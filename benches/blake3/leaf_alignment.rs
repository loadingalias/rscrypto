//! Address alignment controls for existing full-chunk production workers.

use core::ops::Range;

fn range_at_offset(storage: &[u8], len: usize, offset: usize) -> Range<usize> {
  assert!(offset < 64);
  let base_offset = storage.as_ptr() as usize % 64;
  let start = (64usize.strict_sub(base_offset) % 64).strict_add(offset);
  let range = start..start.strict_add(len);
  assert_eq!(storage[range.clone()].as_ptr() as usize % 64, offset);
  range
}

pub(super) fn fixture(len: usize, offset: usize) -> (Vec<u8>, Range<usize>) {
  let mut storage = vec![0u8; len.strict_add(128)];
  let range = range_at_offset(&storage, len, offset);
  for (i, byte) in storage[range.clone()].iter_mut().enumerate() {
    *byte = u8::try_from(i.strict_add(1) % 251).expect("fixture remainder fits");
  }
  (storage, range)
}
