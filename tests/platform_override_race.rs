#![cfg(feature = "std")]

use rscrypto::platform::{self, Detected};

#[test]
fn concurrent_override_writers_are_serialized() {
  std::thread::scope(|scope| {
    for _ in 0..8 {
      scope.spawn(|| {
        for _ in 0..128 {
          platform::try_set_override(Some(Detected::portable())).unwrap();
          platform::try_set_override(None).unwrap();
        }
      });
    }
  });

  platform::try_set_override(None).unwrap();
}
