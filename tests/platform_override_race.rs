#![cfg(feature = "std")]

use rscrypto::platform::{Detected, expert};

const CHILD_MODE: &str = "RSCRYPTO_PLATFORM_OVERRIDE_RACE_CHILD";

#[test]
fn concurrent_override_writers_are_serialized() {
  std::thread::scope(|scope| {
    for _ in 0..8 {
      scope.spawn(|| {
        for _ in 0..128 {
          expert::try_set_override(Some(Detected::portable()))
            .expect("portable override must be accepted before detection");
          expert::try_set_override(None).expect("override must be clearable before detection");
        }
      });
    }
  });

  expert::try_set_override(None).expect("override must be clearable after concurrent updates");
}

#[test]
#[cfg(not(miri))]
fn concurrent_detection_and_override_child() {
  if std::env::var_os(CHILD_MODE).is_none() {
    return;
  }

  let barrier = std::sync::Barrier::new(3);
  std::thread::scope(|scope| {
    let setter = scope.spawn(|| {
      barrier.wait();
      expert::try_set_override(Some(Detected::portable()))
    });
    let detector = scope.spawn(|| {
      barrier.wait();
      rscrypto::platform::get()
    });

    barrier.wait();
    let setter_result = setter.join().expect("override thread must not panic");
    let detected = detector.join().expect("detection thread must not panic");

    assert!(
      matches!(setter_result, Ok(()) | Err(expert::OverrideError::AlreadyInitialized)),
      "concurrent override returned an invalid result: {setter_result:?}"
    );
    if setter_result == Ok(()) {
      assert_eq!(detected, Detected::portable());
    }
  });
}

#[test]
#[cfg(not(miri))]
fn concurrent_detection_and_override_are_linearizable() {
  let executable = std::env::current_exe().expect("test executable path must be available");

  for _ in 0..32 {
    let status = std::process::Command::new(&executable)
      .arg("--exact")
      .arg("concurrent_detection_and_override_child")
      .arg("--quiet")
      .env(CHILD_MODE, "1")
      .status()
      .expect("race child process must start");
    assert!(status.success(), "race child failed with {status}");
  }
}
