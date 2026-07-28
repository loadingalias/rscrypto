#![cfg(feature = "std")]

use rscrypto::platform::{Detected, expert};

#[test]
fn concurrent_override_writers_are_serialized() {
  std::thread::scope(|scope| {
    for _ in 0..8 {
      scope.spawn(|| {
        for _ in 0..128 {
          expert::try_set_override(Some(Detected::portable())).unwrap();
          expert::try_set_override(None).unwrap();
        }
      });
    }
  });

  expert::try_set_override(None).unwrap();
}
