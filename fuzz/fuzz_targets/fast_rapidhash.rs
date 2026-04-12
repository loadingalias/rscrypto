#![no_main]

use libfuzzer_sys::fuzz_target;
use rscrypto::{FastHash, RapidHash, RapidHash128, RapidHashFast64, RapidHashFast128};
use rscrypto_fuzz::{FuzzInput, some_or_return};

// No oracle crate available — the rapidhash crate exposes only a Hasher API,
// not standalone hash functions compatible with rscrypto's RapidHash.
// Test internal consistency properties instead.

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let seed_bytes: [u8; 8] = some_or_return!(input.bytes());
    let data = input.rest();
    let seed = u64::from_le_bytes(seed_bytes);

    // Property: default seed = seed(0)
    {
        let default_64 = RapidHash::hash(data);
        let seeded_64 = RapidHash::hash_with_seed(0, data);
        assert_eq!(default_64, seeded_64, "rapidhash-64: default vs seed=0");

        let default_128 = RapidHash128::hash(data);
        let seeded_128 = RapidHash128::hash_with_seed(0, data);
        assert_eq!(default_128, seeded_128, "rapidhash-128: default vs seed=0");
    }

    // Property: 128-bit hash embeds 64-bit hash
    // (The high/low 64 bits of rapidhash-128 should relate to rapidhash-64.)
    // Just exercise all variants with the fuzzed seed to shake out panics/UB.
    let _h64 = RapidHash::hash_with_seed(seed, data);
    let _h128 = RapidHash128::hash_with_seed(seed, data);
    let _f64 = RapidHashFast64::hash_with_seed(seed, data);
    let _f128 = RapidHashFast128::hash_with_seed(seed, data);
});
