# Examples

These binaries cover complete workflows. One-call hashing, MAC, and checksum
operations stay in the API documentation.

Run every example with its minimum feature set:

```bash
just test-examples
```

## Workflows

| Example | Purpose | Features |
| --- | --- | --- |
| `aead_seal_open` | Generate a ChaCha20-Poly1305 key, seal with associated data, and open the ciphertext. | `alloc,chacha20poly1305,getrandom` |
| `argon2id_password_hashing` | Create and verify a bounded Argon2id PHC record. | `argon2,phc-strings,getrandom` |
| `ed25519_sign_verify` | Generate an Ed25519 keypair, sign a message, and verify the signature. | `ed25519,getrandom` |
| `rsa_pss_verify` | Verify a packaged RSA-PSS/SHA-256 fixture. | `rsa` |
| `mlkem_encapsulation` | Generate ML-KEM-768 keys and confirm encapsulation and decapsulation agree. | `ml-kem,getrandom` |
| `x25519_key_agreement` | Generate two X25519 keypairs and confirm both parties derive the same raw secret. | `x25519,getrandom` |
| `introspect` | Report platform capabilities and selected CRC, SHA-256, and AEAD backends. | `crc32,sha2,chacha20poly1305,diag` |

Run one example:

```bash
cargo run --example aead_seal_open --features alloc,chacha20poly1305,getrandom
```

Replace the example name and feature list with the matching row. X25519 returns
a raw shared secret that a protocol must bind to its transcript with a KDF.
ML-KEM encapsulation alone does not define a hybrid key-establishment protocol.

See [`docs/migration.md`](../docs/migration.md) when replacing another library.
