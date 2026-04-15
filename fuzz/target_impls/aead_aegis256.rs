
use libfuzzer_sys::fuzz_target;
use rscrypto::{Aegis256, Aegis256Key, aead::Nonce256};
use rscrypto_fuzz::{FuzzInput, assert_aead_forgery, assert_aead_roundtrip, some_or_return};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let key_bytes: [u8; 32] = some_or_return!(input.bytes());
    let nonce_bytes: [u8; 32] = some_or_return!(input.bytes());
    let control: u8 = some_or_return!(input.byte());
    let (aad, plaintext) = some_or_return!(input.split_rest());

    let cipher = Aegis256::new(&Aegis256Key::from_bytes(key_bytes));
    let nonce = Nonce256::from_bytes(nonce_bytes);

    assert_aead_roundtrip(&cipher, &nonce, aad, plaintext);
    assert_aead_forgery(&cipher, &nonce, aad, plaintext, control);

    // Differential: rscrypto ↔ aegis crate (16-byte tag variant)
    {
        use aegis::aegis256::Aegis256 as OracleAegis;

        // rscrypto encrypt → oracle decrypt
        let mut ct = plaintext.to_vec();
        let tag = cipher.encrypt_in_place(&nonce, aad, &mut ct);
        let tag_arr: [u8; 16] = tag.as_ref().try_into().unwrap();
        let oracle = OracleAegis::<16>::new(&key_bytes, &nonce_bytes);
        let pt = oracle.decrypt(&ct, &tag_arr, aad).unwrap();
        assert_eq!(pt, plaintext, "oracle failed to decrypt our ciphertext");

        // oracle encrypt → rscrypto decrypt
        let oracle_enc = OracleAegis::<16>::new(&key_bytes, &nonce_bytes);
        let (oct, otag) = oracle_enc.encrypt(plaintext, aad);
        let mut buf = oct;
        cipher
            .decrypt_in_place(&nonce, aad, &mut buf, &Aegis256::tag_from_slice(&otag).unwrap())
            .expect("we failed to decrypt oracle ciphertext");
        assert_eq!(buf, plaintext, "decrypt mismatch on oracle ciphertext");
    }
});
