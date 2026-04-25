use libfuzzer_sys::fuzz_target;
use rscrypto::{
    Aes256Gcm, Aes256GcmKey,
    aead::NonceCounter,
};
use rscrypto_fuzz::{FuzzInput, some_or_return, split_at_ratio};

// Bound the per-iteration nonce burst so the fuzz job stays fast even when
// the cipher path runs the full AES-GCM round set per call.
const MAX_NONCES_PER_ITER: u32 = 64;

fn assert_monotonic_and_roundtrip(
    cipher: &Aes256Gcm,
    counter: &mut NonceCounter<Aes256Gcm>,
    aad: &[u8],
    plaintext: &[u8],
    burst: u32,
    initial: u64,
) {
    let prefix = counter.fixed_prefix();

    for i in 0..burst {
        let nonce = counter.next_nonce().expect("counter must not exhaust within bounded burst");

        // Property: prefix never mutates and the counter field equals the
        // expected absolute value (catches both off-by-one and duplicate-
        // emit bugs that monotonic-only checks would miss).
        let bytes = nonce.to_bytes();
        assert_eq!(
            &bytes[..NonceCounter::<Aes256Gcm>::FIXED_PREFIX_LEN],
            &prefix[..],
            "nonce counter mutated the fixed prefix"
        );
        let mut ctr_bytes = [0u8; 8];
        ctr_bytes.copy_from_slice(&bytes[NonceCounter::<Aes256Gcm>::FIXED_PREFIX_LEN..]);
        let ctr_value = u64::from_be_bytes(ctr_bytes);
        let expected = initial.strict_add(u64::from(i));
        assert_eq!(ctr_value, expected, "counter value drifted from monotonic sequence");

        // Property: the issued nonce produces a valid encrypt → decrypt roundtrip.
        let mut buf = plaintext.to_vec();
        let tag = cipher
            .encrypt_in_place(&nonce, aad, &mut buf)
            .expect("encrypt under fresh nonce must succeed");
        cipher
            .decrypt_in_place(&nonce, aad, &mut buf, &tag)
            .expect("decrypt with the same nonce must succeed");
        assert_eq!(buf, plaintext, "roundtrip plaintext mismatch");
    }

    // Property: issued + remaining = MAX_MESSAGES, and next_counter reflects the burst.
    assert_eq!(
        counter.next_counter(),
        initial.strict_add(u64::from(burst)),
        "next_counter did not advance by burst length"
    );
    assert_eq!(
        counter
            .issued()
            .strict_add(counter.remaining()),
        NonceCounter::<Aes256Gcm>::MAX_MESSAGES,
        "issued + remaining must always equal MAX_MESSAGES"
    );
}

fn assert_resume_equivalence(prefix: [u8; 4], advance: u32) {
    // Property: a counter resumed at `advance` produces the same nonce as a
    // fresh counter advanced to `advance` by issuing-and-discarding. Bounded
    // by `MAX_NONCES_PER_ITER` so the advance loop is always short.
    let mut resumed = NonceCounter::<Aes256Gcm>::with_counter(prefix, u64::from(advance))
        .expect("resume below cap must succeed");
    let mut fresh = NonceCounter::<Aes256Gcm>::new(prefix);
    for _ in 0..advance {
        fresh.next_nonce().expect("advance fresh counter to resume point");
    }
    let resumed_nonce = resumed.next_nonce().expect("resumed counter must issue");
    let fresh_nonce = fresh.next_nonce().expect("fresh-advanced counter must issue");
    assert_eq!(
        resumed_nonce, fresh_nonce,
        "resume vs advance produced different nonces"
    );
    // The fresh counter has now issued `advance + 1` nonces; assert the
    // monotonic invariant holds across the resume bridge.
    assert_eq!(
        fresh.next_counter(),
        u64::from(advance).strict_add(1),
        "fresh-advanced counter did not advance by `advance + 1`"
    );
}

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let key_bytes: [u8; 32] = some_or_return!(input.bytes());
    let prefix: [u8; 4] = some_or_return!(input.bytes());
    let initial_bytes: [u8; 8] = some_or_return!(input.bytes());
    let burst_byte: u8 = some_or_return!(input.byte());
    let aad_split: u8 = some_or_return!(input.byte());
    let rest = input.rest();

    // Bound `initial` so the burst cannot collide with MAX_MESSAGES — keeps
    // the monotonic-burst path on the success branch. Exhaustion semantics
    // are pinned by `aes_gcm_nonce_counter_exhausts_cleanly` and
    // `aes_gcm_nonce_counter_with_counter_rejects_max` in src/aead/nonce_counter.rs.
    let initial = u64::from_le_bytes(initial_bytes)
        % NonceCounter::<Aes256Gcm>::MAX_MESSAGES.strict_sub(u64::from(MAX_NONCES_PER_ITER));
    let burst = (u32::from(burst_byte) % MAX_NONCES_PER_ITER).strict_add(1);
    // Resume-equivalence advance bounded so the fresh-counter loop runs
    // ≤ MAX_NONCES_PER_ITER times — exercised every iteration rather than
    // gated on a 10⁻¹² fuzzer probability.
    let resume_advance = u32::from(burst_byte) % MAX_NONCES_PER_ITER;

    let cipher = Aes256Gcm::new(&Aes256GcmKey::from_bytes(key_bytes));
    let (aad, plaintext) = split_at_ratio(rest, aad_split);

    let mut counter = NonceCounter::<Aes256Gcm>::with_counter(prefix, initial)
        .expect("initial < MAX_MESSAGES by construction");

    assert_monotonic_and_roundtrip(&cipher, &mut counter, aad, plaintext, burst, initial);
    assert_resume_equivalence(prefix, resume_advance);
});
