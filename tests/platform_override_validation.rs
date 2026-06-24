#![cfg(feature = "std")]

use rscrypto::platform::{self, Arch, Caps, Detected, OverrideError};

#[test]
fn safe_override_rejects_impossible_caps_and_still_allows_portable() {
  let invalid = Detected {
    caps: Caps::from_words([u64::MAX; 4]),
    arch: Arch::current(),
  };

  assert_eq!(
    platform::try_set_override(Some(invalid)),
    Err(OverrideError::InvalidCapabilities)
  );

  platform::try_set_override(Some(Detected::portable())).unwrap();
  platform::try_set_override(None).unwrap();
}
