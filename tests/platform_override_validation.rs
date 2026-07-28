#![cfg(feature = "std")]

use rscrypto::platform::{
  self, Arch, Caps, Detected,
  expert::{self, OverrideError},
};

#[test]
fn safe_override_rejects_impossible_caps_and_still_allows_portable() {
  let invalid = Detected {
    caps: Caps::from_words([u64::MAX; 4]),
    arch: Arch::current(),
  };

  assert_eq!(
    expert::try_set_override(Some(invalid)),
    Err(OverrideError::InvalidCapabilities)
  );

  let portable = Detected::portable();
  expert::try_set_override(Some(portable)).unwrap();
  assert!(expert::has_override());
  assert_eq!(platform::get(), portable);

  #[cfg(not(miri))]
  {
    assert_eq!(expert::try_set_override(None), Err(OverrideError::AlreadyInitialized));
    assert_eq!(
      expert::try_set_override(Some(portable)),
      Err(OverrideError::AlreadyInitialized)
    );
  }

  #[cfg(miri)]
  {
    expert::try_set_override(None).unwrap();
    assert!(!expert::has_override());
  }
}
