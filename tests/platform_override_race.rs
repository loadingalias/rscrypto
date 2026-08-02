#![cfg(feature = "std")]

use rscrypto::platform::{Detected, expert};

const CHILD_MODE: &str = "RSCRYPTO_PLATFORM_OVERRIDE_RACE_CHILD";

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
    let setter_result = setter.join().unwrap();
    let detected = detector.join().unwrap();

    match setter_result {
      Ok(()) => assert_eq!(detected, Detected::portable()),
      Err(expert::OverrideError::AlreadyInitialized) => {}
      Err(error) => panic!("unexpected override result: {error:?}"),
    }
  });
}

#[test]
#[cfg(not(miri))]
fn concurrent_detection_and_override_are_linearizable() {
  let executable = std::env::current_exe().unwrap();

  for _ in 0..32 {
    let status = std::process::Command::new(&executable)
      .arg("--exact")
      .arg("concurrent_detection_and_override_child")
      .arg("--quiet")
      .env(CHILD_MODE, "1")
      .status()
      .unwrap();
    assert!(status.success(), "race child failed with {status}");
  }
}
