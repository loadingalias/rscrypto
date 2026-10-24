#![cfg(all(feature = "blake3", feature = "std"))]

extern crate alloc;

use alloc::rc::Rc;
use std::io::{self, Cursor, Read};

use blake3::hazmat::HasherExt as _;
use rscrypto::{Blake3, Blake3DeriveContext, Digest, Xof, hashes::expert::blake3_tree::Blake3Tree};

const KEY: [u8; 32] = [0xa7; 32];
const CONTEXT: &str = "rscrypto BLAKE3 buffered reader";
const MIB: usize = 1024 * 1024;

fn data(len: usize) -> Vec<u8> {
  (0..len).map(|i| u8::try_from(i % 251).expect("byte pattern")).collect()
}

fn states(mode: usize) -> (Blake3, blake3::Hasher, Blake3Tree) {
  match mode {
    0 => (Blake3::new(), blake3::Hasher::new(), Blake3Tree::new()),
    1 => (
      Blake3::new_keyed(&KEY),
      blake3::Hasher::new_keyed(&KEY),
      Blake3Tree::keyed(&KEY),
    ),
    _ => (
      Blake3::new_derive_key(CONTEXT),
      blake3::Hasher::new_derive_key(CONTEXT),
      Blake3Tree::derive_key(&Blake3DeriveContext::new(CONTEXT)),
    ),
  }
}

/// Exercise short reads, interruptions, and a recoverable I/O failure without
/// replacing any hashing operation. Error bytes were not successfully read.
struct Fragments<'a> {
  input: &'a [u8],
  position: usize,
  split: u64,
  interrupt: bool,
  fail_at: Option<usize>,
}

impl<'a> Fragments<'a> {
  fn new(input: &'a [u8]) -> Self {
    Self {
      input,
      position: 0,
      split: 0x517c_c1b7_2722_0a95,
      interrupt: true,
      fail_at: None,
    }
  }
}

impl Read for Fragments<'_> {
  fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
    if self.interrupt {
      self.interrupt = false;
      return Err(io::ErrorKind::Interrupted.into());
    }
    if self.fail_at == Some(self.position) {
      self.fail_at = None;
      buffer.fill(0xac);
      return Err(io::Error::from_raw_os_error(5));
    }
    self.split ^= self.split << 13;
    self.split ^= self.split >> 7;
    self.split ^= self.split << 17;
    let count = usize::try_from(self.split & 8191)
      .expect("split fits usize")
      .strict_add(1)
      .min(buffer.len())
      .min(self.input.len().strict_sub(self.position))
      .min(self.fail_at.map_or(usize::MAX, |at| at.strict_sub(self.position)));
    let end = self.position.strict_add(count);
    buffer[..count].copy_from_slice(&self.input[self.position..end]);
    self.position = end;
    self.interrupt = self.split & 3 == 0;
    Ok(count)
  }
}

/// A reader must stay on its caller even when hashing uses worker threads.
/// Rc also makes a newly introduced Send bound fail at compile time.
struct CallerFragments<'a> {
  inner: Fragments<'a>,
  caller: Rc<std::thread::ThreadId>,
  panic_at: Option<usize>,
}

impl<'a> CallerFragments<'a> {
  fn new(input: &'a [u8]) -> Self {
    #[cfg(feature = "parallel")]
    assert!(
      rayon::current_thread_index().is_none(),
      "exercise ordinary caller entry"
    );
    Self {
      inner: Fragments::new(input),
      caller: Rc::new(std::thread::current().id()),
      panic_at: None,
    }
  }
}

impl Read for CallerFragments<'_> {
  #[expect(
    clippy::panic,
    clippy::panic_in_result_fn,
    reason = "inject a reader panic and assert the reader stays on its caller"
  )]
  fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
    assert_eq!(*self.caller, std::thread::current().id(), "reader moved off its caller");
    if self.panic_at == Some(self.inner.position) {
      self.panic_at = None;
      buffer.fill(0xac);
      panic!("injected reader panic");
    }
    let count = self
      .panic_at
      .map_or(buffer.len(), |at| at.strict_sub(self.inner.position).min(buffer.len()));
    self.inner.read(&mut buffer[..count])
  }
}

/// Fills up to `max` bytes of each read, so reads end exactly where the
/// hasher's offered storage ends. Can interrupt every other call and fail once.
struct Exact<'a> {
  input: &'a [u8],
  position: usize,
  max: usize,
  interrupts: bool,
  interrupt_next: bool,
  fail_at: Option<usize>,
}

impl<'a> Exact<'a> {
  fn new(input: &'a [u8], max: usize) -> Self {
    Self {
      input,
      position: 0,
      max,
      interrupts: false,
      interrupt_next: false,
      fail_at: None,
    }
  }
}

impl Read for Exact<'_> {
  fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
    if self.interrupts {
      self.interrupt_next = !self.interrupt_next;
      if self.interrupt_next {
        return Err(io::ErrorKind::Interrupted.into());
      }
    }
    if self.fail_at == Some(self.position) {
      self.fail_at = None;
      buffer.fill(0xac);
      return Err(io::Error::from_raw_os_error(5));
    }
    let count = buffer
      .len()
      .min(self.max)
      .min(self.input.len().strict_sub(self.position))
      .min(self.fail_at.map_or(usize::MAX, |at| at.strict_sub(self.position)));
    let end = self.position.strict_add(count);
    buffer[..count].copy_from_slice(&self.input[self.position..end]);
    self.position = end;
    Ok(count)
  }
}

const KIB: usize = 1024;

/// Lengths around each internal read size: the buffer doubling to 16 KiB
/// (serial) or 1 MiB (threaded), serial growth to 32 and 64 KiB after 256 and
/// 512 KiB, the 64-byte end-of-input check past the 64 KiB stack buffer, the
/// move to the heap after 1 MiB of serial input, and whole 1 MiB buffers.
fn threshold_lengths() -> Vec<usize> {
  let mut lengths = Vec::new();
  for size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024].map(|kib: usize| kib.strict_mul(KIB)) {
    lengths.extend([size.strict_sub(1), size, size.strict_add(1)]);
  }
  lengths.extend([64 * KIB + 63, 64 * KIB + 64, 64 * KIB + 65]);
  lengths.extend([
    MIB + 64 * KIB - 1,
    MIB + 64 * KIB,
    MIB + 64 * KIB + 64,
    MIB + 64 * KIB + 65,
  ]);
  lengths
}

fn in_reader_pools(check: impl Fn() + Sync) {
  #[cfg(feature = "parallel")]
  for threads in [1, 4] {
    rayon::ThreadPoolBuilder::new()
      .num_threads(threads)
      .build()
      .expect("reader test pool")
      .install(&check);
  }
  #[cfg(not(feature = "parallel"))]
  check();
}

#[test]
fn reader_exact_reads_around_internal_sizes_match_upstream() {
  in_reader_pools(|| {
    let storage = data(MIB + 64 * KIB + 136);
    for len in threshold_lengths() {
      // A prefix moves the hasher off a 1 MiB boundary.
      for prefix in [0usize, 70] {
        let input = &storage[prefix..prefix + len];
        // Byte-at-a-time reads consume each offered window one byte at a time.
        let maxima: &[usize] = if len <= 130 * KIB {
          &[usize::MAX, 1]
        } else {
          &[usize::MAX]
        };
        for &max in maxima {
          for mode in 0..3 {
            let (mut ours, mut reference, _) = states(mode);
            ours.update(&storage[..prefix]);
            reference.update(&storage[..prefix]);
            let mut reader = Exact::new(input, max);
            reader.interrupts = max == 1;
            assert_eq!(
              ours.update_reader(&mut reader).expect("exact reads"),
              len as u64,
              "mode={mode} len={len} prefix={prefix} max={max}"
            );
            assert_eq!(reader.position, len);
            reference.update(input);
            assert_eq!(
              ours.clone().finalize(),
              *reference.finalize().as_bytes(),
              "mode={mode} len={len} prefix={prefix} max={max}"
            );
            ours.update(b"continued");
            reference.update(b"continued");
            assert_reader_state(&ours, &reference);
          }
        }
      }
    }
  });
}

#[test]
fn reader_serial_read_growth_boundaries_match_upstream() {
  // Serial reads grow to 128 KiB, 256 KiB, 512 KiB and 1 MiB after 1, 2, 4
  // and 8 MiB; the first 1 MiB serial read completes at 9 MiB.
  in_reader_pools(|| {
    let storage = data(9 * MIB + 1);
    for len in [2 * MIB, 4 * MIB, 8 * MIB, 9 * MIB]
      .into_iter()
      .flat_map(|size| [size - 1, size, size + 1])
    {
      let len = len.min(storage.len());
      let input = &storage[..len];
      for mode in 0..2 {
        let (mut ours, mut reference, _) = states(mode);
        assert_eq!(ours.update_reader(&mut &input[..]).expect("slice reads"), len as u64);
        reference.update(input);
        assert_reader_state(&ours, &reference);
      }
    }
  });
}

#[test]
fn reader_errors_at_internal_sizes_commit_consumed_bytes() {
  let storage = data(MIB + 64 * KIB + 200);
  let input = &storage[..];
  for fail_at in [
    KIB,
    KIB + 64,
    4 * KIB,
    64 * KIB,
    64 * KIB + 1,
    64 * KIB + 64,
    64 * KIB + 65,
    MIB,
    MIB + 64 * KIB,
    MIB + 64 * KIB + 64,
  ] {
    for mode in 0..3 {
      let (mut ours, mut reference, _) = states(mode);
      let mut reader = Exact::new(input, usize::MAX);
      reader.interrupts = true;
      reader.fail_at = Some(fail_at);
      assert_eq!(
        ours
          .update_reader(&mut reader)
          .expect_err("injected read error")
          .raw_os_error(),
        Some(5),
        "mode={mode} fail_at={fail_at}"
      );
      assert_eq!(reader.position, fail_at);
      reference.update(&input[..fail_at]);
      assert_reader_state(&ours, &reference);
      assert_eq!(
        ours.update_reader(&mut reader).expect("resumed read"),
        (input.len() - fail_at) as u64
      );
      reference.update(&input[fail_at..]);
      assert_reader_state(&ours, &reference);
    }
  }
}

#[test]
fn reader_take_and_subtree_capacity_stop_at_internal_sizes() {
  let storage = data(2 * MIB + 200);
  for limit in [
    KIB,
    64 * KIB,
    64 * KIB + 64,
    128 * KIB,
    MIB,
    MIB + 64 * KIB + 64,
    2 * MIB,
  ] {
    for mode in 0..3 {
      let (mut ours, mut reference, _) = states(mode);
      let mut reader = Exact::new(&storage, usize::MAX).take(limit as u64);
      assert_eq!(ours.update_reader(&mut reader).expect("taken reads"), limit as u64);
      assert_eq!(reader.into_inner().position, limit);
      reference.update(&storage[..limit]);
      assert_reader_state(&ours, &reference);
    }
  }
  // A subtree at offset `capacity` holds exactly `capacity` bytes. A prefix
  // leaves a remaining capacity that no internal read size divides.
  for capacity in [KIB, 64 * KIB, 128 * KIB, MIB, 2 * MIB] {
    for prefix in [0usize, 70] {
      for mode in 0..3 {
        let (_, mut reference, tree) = states(mode);
        let offset = capacity as u64;
        let mut subtree = tree.subtree(offset).expect("aligned offset");
        assert_eq!(subtree.max_len(), offset);
        subtree.update(&storage[..prefix]).expect("prefix fits");
        let mut reader = Exact::new(&storage[prefix..], usize::MAX);
        assert_eq!(
          subtree.update_reader(&mut reader).expect("bounded read"),
          (capacity - prefix) as u64
        );
        assert_eq!(reader.position, capacity - prefix, "no read past the subtree capacity");
        assert_eq!(subtree.update_reader(&mut reader).expect("full subtree"), 0);
        assert_eq!(reader.position, capacity - prefix);
        reference.set_input_offset(offset);
        reference.update(&storage[..capacity]);
        assert_eq!(
          subtree.finalize().expect("nonempty").as_bytes(),
          &reference.finalize_non_root(),
          "mode={mode} capacity={capacity} prefix={prefix}"
        );
      }
    }
  }
}

fn assert_reader_state(ours: &Blake3, reference: &blake3::Hasher) {
  assert_eq!(ours.finalize(), *reference.finalize().as_bytes());
  let mut actual = [0u8; 131];
  let mut expected = [0u8; 131];
  ours.finalize_xof().squeeze(&mut actual);
  reference.finalize_xof().fill(&mut expected);
  assert_eq!(actual, expected);
}

#[test]
fn reader_large_tails_keep_non_send_reader_on_caller() {
  // Exercise repeated full read buffers and nearby tails through the public API.
  let storage = data(18 * MIB + 71);
  for len in [16 * MIB, 17 * MIB - 1, 17 * MIB, 17 * MIB + 1, 18 * MIB + 70] {
    let input = &storage[1..len.strict_add(1)];
    let mut reader = CallerFragments::new(input);
    let mut ours = Blake3::new();
    let mut reference = blake3::Hasher::new();
    assert_eq!(
      ours.update_reader(&mut reader).expect("large fragmented read"),
      u64::try_from(len).expect("length fits")
    );
    reference.update(input);
    assert_reader_state(&ours, &reference);
    assert_eq!(ours.update_reader(&mut io::empty()).expect("empty continuation"), 0);
    ours.update(b"continued after large reader");
    reference.update(b"continued after large reader");
    assert_reader_state(&ours, &reference);
  }
}

#[test]
fn reader_large_errors_commit_full_and_partial_buffers() {
  let storage = data(18 * MIB + 2051);
  let input = &storage[1..];
  for fail_at in [17 * MIB, 18 * MIB + 70] {
    let mut reader = CallerFragments::new(input);
    reader.inner.fail_at = Some(fail_at);
    let mut ours = Blake3::new();
    let mut reference = blake3::Hasher::new();
    assert_eq!(
      ours
        .update_reader(&mut reader)
        .expect_err("injected read error")
        .raw_os_error(),
      Some(5)
    );
    reference.update(&input[..fail_at]);
    assert_reader_state(&ours, &reference);
    assert_eq!(reader.inner.position, fail_at);
    assert_eq!(
      ours.update_reader(&mut reader).expect("resume after large read error"),
      u64::try_from(input.len().strict_sub(fail_at)).expect("remaining length fits")
    );
    reference.update(&input[fail_at..]);
    assert_reader_state(&ours, &reference);
  }
}

// A caught reader panic can inspect completed state only when unwinding is enabled.
#[cfg(panic = "unwind")]
#[test]
fn reader_large_panic_preserves_completed_buffers() {
  let completed = 18 * MIB;
  let input = data(completed + 70);
  let mut reader = CallerFragments::new(&input);
  reader.panic_at = Some(completed);
  let mut ours = Blake3::new();
  let mut reference = blake3::Hasher::new();
  let panic = std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| ours.update_reader(&mut reader)))
    .expect_err("injected reader panic must propagate");
  assert_eq!(panic.downcast_ref::<&str>(), Some(&"injected reader panic"));
  reference.update(&input[..completed]);
  assert_reader_state(&ours, &reference);
  assert_eq!(reader.inner.position, completed);
  assert_eq!(ours.update_reader(&mut reader).expect("resume after reader panic"), 70);
  reference.update(&input[completed..]);
  assert_reader_state(&ours, &reference);
}

#[test]
fn reader_every_two_chunk_length_and_buffer_boundaries_match_upstream() {
  let storage = data(2 * MIB + 72);
  let boundaries = [MIB - 1, MIB, MIB + 1, 2 * MIB, 2 * MIB + 70];
  for len in (0usize..=2048).chain(boundaries) {
    let input = &storage[1..len.strict_add(1)];
    for mode in 0..3 {
      let (mut ours, mut reference, _) = states(mode);
      let prefix_len = [1, 24, 70, 1023][len % 4];
      ours.update(&storage[..prefix_len]);
      reference.update(&storage[..prefix_len]);
      let mut reader = Fragments::new(input);
      // Dynamic readers are supported as well as concrete files and adapters.
      let count = ours
        .update_reader(&mut reader as &mut dyn Read)
        .expect("fragmented reads");
      assert_eq!(count, u64::try_from(len).expect("length fits"));
      reference.update(input);
      assert_eq!(
        ours.finalize(),
        *reference.finalize().as_bytes(),
        "mode={mode} len={len}"
      );
      let mut actual = [0u8; 131];
      let mut expected = [0u8; 131];
      ours.finalize_xof().squeeze(&mut actual);
      reference.finalize_xof().fill(&mut expected);
      assert_eq!(actual, expected, "XOF mode={mode} len={len}");
      ours.update(b"continued");
      reference.update(b"continued");
      assert_eq!(ours.finalize(), *reference.finalize().as_bytes());
    }
  }
}

#[test]
fn reader_full_buffers_preserve_pending_roots_and_continuation() {
  fn check() {
    let storage = data(4 * MIB + 1);
    for len in [
      MIB - 1,
      MIB,
      MIB + 1,
      2 * MIB - 1,
      2 * MIB,
      2 * MIB + 70,
      3 * MIB + 1023,
      4 * MIB,
    ] {
      let input = &storage[1..len + 1];
      for mode in 0..3 {
        let (mut ours, mut reference, _) = states(mode);
        assert_eq!(
          ours
            .update_reader(&mut Fragments::new(input))
            .expect("full-buffer read"),
          len as u64
        );
        reference.update(input);
        assert_eq!(
          ours.clone().finalize(),
          *reference.finalize().as_bytes(),
          "mode={mode} len={len}"
        );
        assert_eq!(ours.update_reader(&mut io::empty()).expect("empty read"), 0);
        let mut actual = [0u8; 131];
        let mut expected = [0u8; 131];
        ours.finalize_xof().squeeze(&mut actual);
        reference.finalize_xof().fill(&mut expected);
        assert_eq!(actual, expected, "XOF mode={mode} len={len}");
        ours.update(b"continued");
        reference.update(b"continued");
        assert_eq!(ours.finalize(), *reference.finalize().as_bytes());
      }
    }
  }
  #[cfg(feature = "parallel")]
  for threads in [1, 4] {
    rayon::ThreadPoolBuilder::new()
      .num_threads(threads)
      .build()
      .expect("reader test pool")
      .install(check);
  }
  #[cfg(not(feature = "parallel"))]
  check();
}

#[test]
fn reader_error_after_full_buffers_preserves_pending_state() {
  let storage = data(2 * MIB + 1025);
  let input = &storage[1..];
  for fail_at in [MIB, MIB + 1, MIB + 70, 2 * MIB, 2 * MIB + 1] {
    for mode in 0..3 {
      let (mut ours, mut reference, _) = states(mode);
      let mut reader = Fragments::new(input);
      reader.fail_at = Some(fail_at);
      assert_eq!(
        ours
          .update_reader(&mut reader)
          .expect_err("injected read error")
          .raw_os_error(),
        Some(5)
      );
      reference.update(&input[..fail_at]);
      assert_eq!(ours.finalize(), *reference.finalize().as_bytes());
      assert_eq!(reader.position, fail_at);
      let mut actual = [0u8; 131];
      let mut expected = [0u8; 131];
      ours.finalize_xof().squeeze(&mut actual);
      reference.finalize_xof().fill(&mut expected);
      assert_eq!(actual, expected, "error XOF mode={mode} at={fail_at}");
      assert_eq!(
        ours.update_reader(&mut reader).expect("resumed read"),
        (input.len() - fail_at) as u64
      );
      reference.update(&input[fail_at..]);
      assert_eq!(ours.finalize(), *reference.finalize().as_bytes());
    }
  }
}

#[test]
fn reader_error_preserves_consumed_bytes_and_can_resume() {
  let input = data(MIB + 2050);
  for fail_at in [0, 70, 1023, 4097, MIB, MIB + 17] {
    for mode in 0..3 {
      let (mut ours, mut reference, tree) = states(mode);
      ours.update(b"prefix");
      reference.update(b"prefix");
      let mut reader = Fragments::new(&input);
      reader.fail_at = Some(fail_at);
      assert_eq!(
        ours
          .update_reader(&mut reader)
          .expect_err("injected error")
          .raw_os_error(),
        Some(5)
      );
      reference.update(&input[..fail_at]);
      assert_eq!(ours.finalize(), *reference.finalize().as_bytes());
      assert_eq!(reader.position, fail_at);
      assert_eq!(
        ours.update_reader(&mut reader).expect("resumed read"),
        u64::try_from(input.len() - fail_at).expect("length")
      );
      reference.update(&input[fail_at..]);
      assert_eq!(ours.finalize(), *reference.finalize().as_bytes());

      let mut subtree = tree.subtree(0).expect("offset zero");
      let mut reader = Fragments::new(&input);
      reader.fail_at = Some(fail_at);
      assert_eq!(
        subtree
          .update_reader(&mut reader)
          .expect_err("injected error")
          .raw_os_error(),
        Some(5)
      );
      assert_eq!(subtree.len(), u64::try_from(fail_at).expect("length"));
      subtree.update_reader(&mut reader).expect("resumed subtree");
      let (_, mut reference, _) = states(mode);
      reference.update(&input);
      assert_eq!(
        subtree.finalize().expect("nonempty").as_bytes(),
        &reference.finalize_non_root()
      );
    }
  }
}

#[test]
fn subtree_reader_stops_at_capacity_and_preserves_large_counters() {
  let input = data(4096 + 70);
  for mode in 0..3 {
    for offset in [1024u64, 6 * 1024, (1 << 32) * 1024, ((1 << 54) - 4) * 1024] {
      let (_, mut reference, tree) = states(mode);
      let mut subtree = tree.subtree(offset).expect("aligned offset");
      subtree.update(&input[..70]).expect("prefix fits");
      let mut reader = Cursor::new(&input[70..]);
      let count = subtree.update_reader(&mut reader).expect("bounded read");
      let len = input
        .len()
        .min(usize::try_from(subtree.max_len()).unwrap_or(usize::MAX));
      assert_eq!(count, u64::try_from(len - 70).expect("count"));
      assert_eq!(reader.position(), count);
      assert_eq!(subtree.len(), u64::try_from(len).expect("length"));
      reference.set_input_offset(offset);
      reference.update(&input[..len]);
      assert_eq!(
        subtree.finalize().expect("nonempty").as_bytes(),
        &reference.finalize_non_root()
      );
      if subtree.len() == subtree.max_len() {
        assert_eq!(subtree.update_reader(&mut reader).expect("full subtree"), 0);
        assert_eq!(reader.position(), count);
      }
    }
  }
}

#[test]
fn caller_scheduled_readers_merge_to_upstream_root_and_xof() {
  for len in [
    3 * 1024 + 70,
    5 * 1024 + 1,
    6 * 1024 + 1023,
    7 * 1024 + 70,
    2 * MIB + 17,
  ] {
    let input = data(len);
    for mode in 0..3 {
      let (_, mut reference, tree) = states(mode);
      let split =
        usize::try_from(Blake3Tree::left_subtree_len(u64::try_from(len).expect("length")).expect("multichunk"))
          .expect("split fits");
      let (left, right) = std::thread::scope(|scope| {
        let left = scope.spawn(|| {
          let mut subtree = tree.subtree(0).expect("left offset");
          let mut reader = Cursor::new(&input).take(u64::try_from(split).expect("split"));
          assert_eq!(
            subtree.update_reader(&mut reader).expect("left read"),
            u64::try_from(split).expect("split")
          );
          subtree.finalize().expect("left CV")
        });
        let right = scope.spawn(|| {
          let mut subtree = tree
            .subtree(u64::try_from(split).expect("split"))
            .expect("right offset");
          let mut reader = Fragments::new(&input[split..]);
          assert_eq!(
            subtree.update_reader(&mut reader).expect("right read"),
            u64::try_from(len - split).expect("right length")
          );
          subtree.finalize().expect("right CV")
        });
        (left.join().expect("left worker"), right.join().expect("right worker"))
      });
      reference.update(&input);
      assert_eq!(
        tree.merge_root(&left, &right).expect("valid root"),
        *reference.finalize().as_bytes()
      );
      let mut actual = [0u8; 131];
      let mut expected = [0u8; 131];
      tree
        .merge_root_xof(&left, &right)
        .expect("valid root")
        .squeeze(&mut actual);
      reference.finalize_xof().fill(&mut expected);
      assert_eq!(actual, expected);
    }
  }
}
