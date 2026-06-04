# Migration: `aegis` (jedisct1) → `rscrypto`

> Covers AEGIS-256 from the `aegis` crate. Replace `Aegis256::<TAG>::new(&key, &nonce).encrypt(msg, aad) -> (Vec<u8>, [u8; TAG])` with rscrypto's `Aead`-trait-style `encrypt(&nonce, aad, msg, &mut out)`. Same algorithm (draft-irtf-cfrg-aegis-aead), byte-identical ciphertext+tag.

Verified against `aegis = "0.9.12"` and the `rscrypto` 0.3.1 line.
Evidence: `tests/aegis256_oracle.rs` and `tests/aead_wycheproof.rs`.

## TL;DR

| | Before (`aegis` 0.9.x) | After (`rscrypto` 0.3.1) |
|---|---|---|
| Cargo dep | `aegis = "0.9"` | `rscrypto = { version = "0.3.1", features = ["aegis256"] }` |
| Import | `use aegis::aegis256::Aegis256;` | `use rscrypto::{Aead, Aegis256, Aegis256Key, aead::Nonce256};` |
| Encrypt | `Aegis256::<16>::new(&key, &nonce).encrypt(msg, aad) -> (Vec<u8>, [u8; 16])` | `cipher.encrypt(&nonce, aad, msg, &mut out)?` |

## Cargo.toml

```toml
# Before
[dependencies]
aegis = "0.9"
```

```toml
# After
[dependencies]
rscrypto = { version = "0.3.1", features = ["aegis256"] }
```

## Algorithm map

| `aegis` type | rscrypto type | Key | Nonce | Tag |
|---|---|---|---|---|
| `aegis::aegis256::Aegis256` | `Aegis256` | 32 bytes | 32 bytes | 16 bytes (or 32 in upstream) |
| `aegis::aegis128l::Aegis128L` | not currently mapped — open an issue | 16 bytes | 16 bytes | 16/32 bytes |
| `aegis::aegis128x2::Aegis128X2` | not currently mapped | 16 bytes | 16 bytes | 16/32 bytes |
| `aegis::aegis128x4::Aegis128X4` | not currently mapped | 16 bytes | 16 bytes | 16/32 bytes |

rscrypto ships only AEGIS-256 with a 16-byte tag. The `aegis` crate is parameterised over tag size (`Aegis256<16>` or `Aegis256<32>`) — fix the upstream tag at 16 to match.

## API patterns

### Combined encrypt — note the shape change

```rust
// Before
use aegis::aegis256::Aegis256;

let cipher = Aegis256::<16>::new(&[0u8; 32], &[0u8; 32]);   // (key, nonce) at construction
let (ciphertext, tag) = cipher.encrypt(plaintext, aad);     // separate (Vec<u8>, [u8; 16])
```

```rust
// After
use rscrypto::{Aead, Aegis256, Aegis256Key, aead::Nonce256};

let key = Aegis256Key::from_bytes([0u8; 32]);
let cipher = Aegis256::new(&key);                            // key at construction
let nonce = Nonce256::from_bytes([0u8; 32]);                 // nonce at call
let mut ct = vec![0u8; plaintext.len() + 16];
cipher.encrypt(&nonce, aad, plaintext, &mut ct)?;            // appended tag
```

Three structural differences:

| `aegis` | rscrypto |
|---|---|
| `(key, nonce)` together at construction; one cipher per message | `key` at construction; `nonce` per call (cipher reused across messages) |
| Returns `(ciphertext, tag)` separately | Returns single buffer `[ciphertext || tag]` |
| Construction-then-call shape | `Aead`-trait-style call shape |

To rebuild the exact `(ct, tag)` pair, take the last 16 bytes of `ct` as the tag:

```rust
// rscrypto, equivalent to (ct, tag) from aegis:
let tag: &[u8] = &ct[plaintext.len()..];
let ct_only: &[u8] = &ct[..plaintext.len()];
```

If you specifically want a detached `(ct, tag)` pair without the slice arithmetic, use `encrypt_in_place`:

```rust
// After (detached form matches aegis's return shape)
let mut buffer = plaintext.to_vec();          // becomes ciphertext in place
let tag = cipher.encrypt_in_place(&nonce, aad, &mut buffer)?;
// Now: buffer == ct, tag == aegis's [u8; 16].
```

### Decrypt

```rust
// Before
let plaintext = cipher.decrypt(&ciphertext, &tag, aad)?;     // returns Result<Vec<u8>, _>
```

```rust
// After (combined; ct already includes tag)
let mut plaintext = vec![0u8; ct.len() - 16];
cipher.decrypt(&nonce, aad, &ct, &mut plaintext)?;

// Or detached (matches aegis's separate ct + tag inputs):
cipher.decrypt_in_place(&nonce, aad, &mut buffer, &tag)?;
```

`Err(OpenError::Verification(_))` on tag mismatch — same shape as `aegis::Error::InvalidTag`.

## Notes

- **One cipher, many messages.** `aegis::Aegis256::new(&key, &nonce)` baked the nonce into the cipher and required reconstruction per message. rscrypto's `Aegis256::new(&key)` returns a reusable cipher; pass a fresh `Nonce256` per call. This is a small clarity win and aligns AEGIS with the rest of the rscrypto AEAD family.
- **Tag size fixed at 16 bytes.** `aegis` lets you pick `Aegis256<16>` or `Aegis256<32>`; rscrypto ships only the 16-byte-tag variant (the cfrg draft default). If you depend on 32-byte tags for a specific protocol, file an issue.
- **`(ct, tag)` vs. `[ct || tag]` layout.** Both layouts are trivially convertible; the harness verifies they produce the same bytes. Pick whichever shape your downstream protocol expects.
- **Hardware acceleration.** AEGIS uses AES round functions (not full AES). Both crates dispatch to AES-NI on x86_64 and AES-CE on aarch64. AEGIS is typically 2–3× faster than AES-GCM on hardware-accelerated targets — keep the win after migrating.
- **Random nonces are safe.** AEGIS-256 has a 256-bit nonce; random selection across `2^96` messages is safe.
- **Not nonce-misuse-resistant.** `(key, nonce)` reuse leaks plaintext XORs. If you need misuse resistance, use AES-GCM-SIV instead.
- **`no_std`.** Both crates support `no_std`.
