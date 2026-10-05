# API compatibility choices

These choices retain existing caller contracts after the API review.
Reopen them for a concrete caller requirement or correctness defect.
A naming preference alone does not justify breaking callers or adding another synonym.

## Hash outputs and collection seeds

`Blake3::digest` and `Blake3::digest_const` return `[u8; 32]`, consistent with the
fixed-size digest APIs. Keep that output type without adding a second unkeyed
`Blake3Hash` wrapper. Keyed output already has the distinct `Blake3KeyedHash` type
and its `ct_eq` operation.

`Xxh3BuildHasher::default()` and `new()` keep the deterministic zero seed.
`with_seed` lets a caller select another deterministic seed.
The type's Rustdoc restricts it to trusted collection keys and directs callers
with attacker-controlled keys to a randomized collision-resistant builder.
Changing `Default` to obtain randomness would change reproducibility and introduce
an entropy requirement into this allocation-free API.

## AEAD aliases

Keep `Aead::decrypt_in_place_detached` and
`AeadWithNonce::encrypt_in_place_detached` as aliases of their respective
`*_in_place` methods. Existing callers retain the explicit detached-tag spelling.
The aliases delegate to the same implementation, with the same bounds, errors,
and post-failure state. Caller-supplied nonce encryption remains on the expert trait.
The AEAD foundation and API consistency tests cover the aliases.

## Parameter and platform names

Keep the published `get_*` accessors on `Argon2Params` and `ScryptParams`.
Their immutable values, units, and validation rules are documented at the methods.
Renaming them or adding parallel getters would add migration work or duplicate API
without changing those contracts.

Keep `platform::get()` for the cached `Detected` value containing architecture
and capabilities. Use `platform::caps()` or `platform::arch()` for one field.
The existing names and initialization rules remain the caller contract.
