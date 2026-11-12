use rscrypto::{Adler32, Checksum};
use rscrypto_fuzz::{FuzzInput, assert_checksum_chunked, assert_checksum_combine, some_or_return, split_at_ratio};

pub(super) fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let split: u8 = some_or_return!(input.byte());
  let data = input.rest();

  let mut oracle = simd_adler32::Adler32::new();
  oracle.write(data);
  let oracle = oracle.finish();
  assert_eq!(Adler32::checksum(data), oracle, "adler32 oracle mismatch");

  assert_checksum_combine::<Adler32>(data, split);
  assert_checksum_chunked::<Adler32>(data, split);

  // Resuming from the checksum of a prefix continues it, and a reset from
  // that resumed state restores the standard initial state.
  let (head, tail) = split_at_ratio(data, split);
  let mut resumed = Adler32::with_initial(Adler32::checksum(head));
  resumed.update(tail);
  assert_eq!(resumed.finalize(), oracle, "adler32 resume mismatch");
  resumed.reset();
  resumed.update(data);
  assert_eq!(resumed.finalize(), oracle, "adler32 reset mismatch");
}
