
use libfuzzer_sys::fuzz_target;
use rscrypto::{FastHash, Xxh3, Xxh3_128};
use rscrypto_fuzz::{FuzzInput, some_or_return};

fuzz_target!(|data: &[u8]| {
    let mut input = FuzzInput::new(data);
    let seed_bytes: [u8; 8] = some_or_return!(input.bytes());
    let data = input.rest();
    let seed = u64::from_le_bytes(seed_bytes);

    // Differential: rscrypto ↔ xxhash-rust crate (64-bit)
    {
        let ours = Xxh3::hash_with_seed(seed, data);
        let oracle = xxhash_rust::xxh3::xxh3_64_with_seed(data, seed);
        assert_eq!(ours, oracle, "xxh3-64 oracle mismatch");
    }

    // Differential: rscrypto ↔ xxhash-rust crate (128-bit)
    {
        let ours = Xxh3_128::hash_with_seed(seed, data);
        let oracle = xxhash_rust::xxh3::xxh3_128_with_seed(data, seed);
        assert_eq!(ours, oracle, "xxh3-128 oracle mismatch");
    }

    // Default-seed (0) consistency
    {
        let default_64 = Xxh3::hash(data);
        let seeded_64 = Xxh3::hash_with_seed(0, data);
        assert_eq!(default_64, seeded_64, "xxh3-64: default vs seed=0 mismatch");

        let default_128 = Xxh3_128::hash(data);
        let seeded_128 = Xxh3_128::hash_with_seed(0, data);
        assert_eq!(
            default_128, seeded_128,
            "xxh3-128: default vs seed=0 mismatch"
        );
    }
});
