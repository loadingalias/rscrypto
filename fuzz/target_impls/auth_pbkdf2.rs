
use libfuzzer_sys::fuzz_target;
use rscrypto::{Pbkdf2Sha256, Pbkdf2Sha512};
use rscrypto_fuzz::{FuzzInput, some_or_return, split_at_ratio};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let split: u8 = some_or_return!(input.byte());
    let out_len_byte: u8 = some_or_return!(input.byte());
    let iterations_bytes: [u8; 2] = some_or_return!(input.bytes());
    let data = input.rest();

    let (password, salt) = split_at_ratio(data, split);
    let out_len = (out_len_byte as usize % 96).strict_add(1);
    let iterations = (u32::from(u16::from_le_bytes(iterations_bytes)) % 64).strict_add(1);

    let mut ours_256 = vec![0u8; out_len];
    let mut ours_256_state = vec![0u8; out_len];
    Pbkdf2Sha256::derive_key(password, salt, iterations, &mut ours_256).unwrap();
    Pbkdf2Sha256::new(password)
        .derive(salt, iterations, &mut ours_256_state)
        .unwrap();
    assert_eq!(ours_256, ours_256_state, "pbkdf2-sha256 state reuse mismatch");
    assert!(Pbkdf2Sha256::verify_password(password, salt, iterations, &ours_256).is_ok());

    let mut oracle_256 = vec![0u8; out_len];
    pbkdf2::pbkdf2_hmac::<sha2_010::Sha256>(password, salt, iterations, &mut oracle_256);
    assert_eq!(ours_256, oracle_256, "pbkdf2-sha256 oracle mismatch");

    let mut ours_512 = vec![0u8; out_len];
    let mut ours_512_state = vec![0u8; out_len];
    Pbkdf2Sha512::derive_key(password, salt, iterations, &mut ours_512).unwrap();
    Pbkdf2Sha512::new(password)
        .derive(salt, iterations, &mut ours_512_state)
        .unwrap();
    assert_eq!(ours_512, ours_512_state, "pbkdf2-sha512 state reuse mismatch");
    assert!(Pbkdf2Sha512::verify_password(password, salt, iterations, &ours_512).is_ok());

    let mut oracle_512 = vec![0u8; out_len];
    pbkdf2::pbkdf2_hmac::<sha2_010::Sha512>(password, salt, iterations, &mut oracle_512);
    assert_eq!(ours_512, oracle_512, "pbkdf2-sha512 oracle mismatch");
});
