
use libfuzzer_sys::fuzz_target;
use rscrypto::{X25519Error, X25519PublicKey, X25519SecretKey, X25519SharedSecret};
use rscrypto_fuzz::{FuzzInput, some_or_return};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let secret_bytes: [u8; 32] = some_or_return!(input.bytes());
    let peer_bytes: [u8; 32] = some_or_return!(input.bytes());

    let secret = X25519SecretKey::from_bytes(secret_bytes);
    let peer = X25519PublicKey::from_bytes(peer_bytes);

    let ours_public = secret.public_key();
    let dalek_secret = x25519_dalek::StaticSecret::from(secret_bytes);
    let dalek_public = x25519_dalek::PublicKey::from(&dalek_secret);
    assert_eq!(ours_public.to_bytes(), dalek_public.to_bytes(), "x25519 public-key mismatch");

    let dalek_peer = x25519_dalek::PublicKey::from(peer_bytes);
    let dalek_shared = dalek_secret.diffie_hellman(&dalek_peer).to_bytes();
    let ours_shared = secret.diffie_hellman(&peer);
    let helper_shared = X25519SharedSecret::diffie_hellman(&secret, &peer);
    assert_eq!(ours_shared, helper_shared, "x25519 helper mismatch");

    if dalek_shared.iter().all(|&byte| byte == 0) {
        assert_eq!(ours_shared, Err(X25519Error::new()), "x25519 low-order rejection mismatch");
    } else {
        assert_eq!(
            ours_shared.expect("non-zero x25519 oracle output must succeed").to_bytes(),
            dalek_shared,
            "x25519 shared-secret mismatch"
        );
    }
});
