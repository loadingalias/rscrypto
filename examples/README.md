# Examples

These programs show complete workflows.
One-call hashing, MAC, and checksum operations are in the API documentation.

Run them with the source revision that they come from.

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
| `mldsa_sign_verify` | Generate ML-DSA-65 keys, sign with operating-system randomness and an explicit context, and verify. | `ml-dsa,getrandom` |
| `mlkem_encapsulation` | Generate ML-KEM-768 keys and confirm that encapsulation and decapsulation agree. | `ml-kem,getrandom` |
| `p256_ecdh` | Generate two ephemeral P-256 keys and confirm that both parties derive the same fixed-width ECC CDH output. | `p256-ecdh,getrandom` |
| `p384_ecdh` | Generate two ephemeral P-384 keys and confirm that both parties derive the same fixed-width ECC CDH output. | `p384-ecdh,getrandom` |
| `x25519_key_agreement` | Generate two X25519 keypairs and confirm that both parties derive the same raw secret. | `x25519,getrandom` |
| `introspect` | Show platform capabilities and the selected CRC, SHA-256, and AEAD backends. | `crc32,sha2,chacha20poly1305,diag` |

Run one example:

```bash
cargo run --example aead_seal_open --features alloc,chacha20poly1305,getrandom
```

Replace the example name and the feature list with the values from its row.

## Protocol notes

- P-256 ECDH, P-384 ECDH, and X25519 return raw shared secrets.
  A protocol must bind them to its transcript with a KDF.
  None of these operations authenticates the peer.
- ML-KEM encapsulation alone does not define a hybrid key-establishment protocol.
- ML-DSA qualification is still in progress.
  See its [status and resource contract](../docs/mldsa.md).
