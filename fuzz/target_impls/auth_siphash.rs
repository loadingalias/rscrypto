use core::hash::{BuildHasher, Hasher};

use rscrypto::{SipHash13, SipHash24, SipHashKey, SipRandomState};
use rscrypto_fuzz::{FuzzInput, some_or_return};

/// Differential SipHash-1-3 and SipHash-2-4 against `siphasher`.
///
/// The input supplies the key, then a schedule that splits the message into
/// byte-string and integer writes. Each integer is read from the message with
/// `from_ne_bytes`, so the writes hash exactly the message bytes, which the
/// oracle receives in one write.
pub(super) fn run(data: &[u8]) {
  let mut input = FuzzInput::new(data);
  let key_bytes: [u8; 16] = some_or_return!(input.bytes());
  let schedule: [u8; 16] = some_or_return!(input.bytes());
  let message = input.rest();
  let key = SipHashKey::from_bytes(key_bytes);

  let mut oracle13 = siphasher::sip::SipHasher13::new_with_key(&key_bytes);
  let mut oracle24 = siphasher::sip::SipHasher24::new_with_key(&key_bytes);
  oracle13.write(message);
  oracle24.write(message);
  let (expected13, expected24) = (oracle13.finish(), oracle24.finish());

  assert_eq!(SipHash13::hash(&key, message), expected13, "SipHash-1-3 one-shot");
  assert_eq!(SipHash24::hash(&key, message), expected24, "SipHash-2-4 one-shot");

  let mut hasher13 = SipHash13::new(&key);
  let mut hasher24 = SipHash24::new(&key);
  let mut table = SipRandomState::new(SipHashKey::from_bytes(key_bytes)).build_hasher();
  let mut offset = 0usize;
  let mut step = 0usize;
  while offset < message.len() {
    let choice = schedule[step & 15];
    step = step.strict_add(1);
    let rest = &message[offset..];
    let taken = write_next(choice, rest, &mut [&mut hasher13, &mut hasher24, &mut table]);
    offset = offset.strict_add(taken);
  }
  assert_eq!(hasher13.finish(), expected13, "SipHash-1-3 streamed");
  assert_eq!(hasher24.finish(), expected24, "SipHash-2-4 streamed");
  assert_eq!(table.finish(), expected13, "SipRandomState streamed");
}

/// Write the next piece of `rest` that `choice` selects to every hasher and
/// return its length. Integer writes need their full width; a shorter rest
/// goes as bytes.
fn write_next(choice: u8, rest: &[u8], hashers: &mut [&mut dyn Hasher]) -> usize {
  macro_rules! integer {
    ($ty:ty, $method:ident) => {
      if let Some(bytes) = rest.first_chunk::<{ size_of::<$ty>() }>() {
        let value = <$ty>::from_ne_bytes(*bytes);
        for hasher in hashers.iter_mut() {
          hasher.$method(value);
        }
        return size_of::<$ty>();
      }
    };
  }
  match choice & 7 {
    0 => integer!(u8, write_u8),
    1 => integer!(u16, write_u16),
    2 => integer!(u32, write_u32),
    3 => integer!(u64, write_u64),
    4 => integer!(u128, write_u128),
    5 => integer!(usize, write_usize),
    6 => integer!(i64, write_i64),
    _ => {}
  }
  // Byte strings of 0 to 31 bytes; an empty write must change nothing.
  let length = usize::from(choice >> 3).min(rest.len());
  for hasher in hashers.iter_mut() {
    hasher.write(&rest[..length]);
  }
  if length == 0 {
    // Progress past an empty write with one byte.
    for hasher in hashers.iter_mut() {
      hasher.write(&rest[..1]);
    }
    return 1;
  }
  length
}
