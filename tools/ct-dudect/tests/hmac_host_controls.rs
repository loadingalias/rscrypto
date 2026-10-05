use std::process::Command;

#[test]
fn unknown_control_mode_exits_with_a_cli_error() {
  let output = Command::new(env!("CARGO_BIN_EXE_hmac_host_controls"))
    .env("RSCRYPTO_CT_HMAC_CONTROL", "unsupported-control")
    .env("RSCRYPTO_CT_DUDECT_SAMPLES", "2")
    .args(["--filter", "hmac_sha256_control"])
    .output()
    .expect("run HMAC diagnostic binary");

  assert_eq!(output.status.code(), Some(2));
  assert!(String::from_utf8_lossy(&output.stderr).contains("unknown HMAC control: unsupported-control"));
}
