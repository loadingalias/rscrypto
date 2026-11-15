# HPKE vectors

These files pin the HPKE test material that `tests/hpke.rs` and `tests/hpke_wycheproof.rs` use.
The RFC and draft files were retrieved on 2026-11-14. Each is the listed selection of its source, written as compact JSON
without other changes. The Wycheproof files were retrieved on 2026-11-15 and are the upstream files without change.

| File | Source | Selection |
| --- | --- | --- |
| `hpke-x25519-hkdfsha256-aes128gcm.json` | [`hpkewg/hpke`](https://github.com/hpkewg/hpke) `test-vectors/test-vectors.json` at `8668538c` (SHA-256 `2a3f653dc93f2446f8cc57c7f5a795ef9671c79a963879906186e147a3bb1e20`) | `mode` 0, `kem_id` 32, `kdf_id` 1, `aead_id` 1: one vector with 257 encryptions and 3 exports |
| `hpke-hpke-06-edge-cases.json` | draft-ietf-hpke-hpke-06 Appendix D.2 to D.4 (`draft-ietf-hpke-hpke-06.txt`, SHA-256 `ad685a1f0ae03ad79c4a8334e156c879ae6019317b1ede8339359b86a4d1497d`) | Empty and zero-byte `pt`, `aad`, and exporter contexts; empty `info`; `info` with zero bytes |
| `hpke-pq-xwing.json` | [`hpkewg/hpke-pq`](https://github.com/hpkewg/hpke-pq) `test-vectors.json` at `6433c8fc` (SHA-256 `35c59f4a0132e5631e50ac039d8ca3a72e99f5e92dfd94d45338d6ae243f613c`) | `kem_id` 25722 (X-Wing): A.5 with HKDF-SHA256 and ChaCha20Poly1305, and A.12 with the SHAKE256 KDF |
| `hpke_x25519_encap_test.json`, `hpke_x25519_decrypt_test.json`, `hpke_x25519_sequence_test.json` | [`C2SP/wycheproof`](https://github.com/C2SP/wycheproof) `testvectors_v1/` at `65ca4947` (added in `d5f724a2`, pull request 273) | The complete files, byte for byte; `tests/hpke_wycheproof.rs` runs every group with `kemId` 32, `kdfId` 1, and `aeadId` 1 |

The first file's values equal RFC 9180 Appendix A.1.1
(`rfc9180.txt`, SHA-256 `f45a8b7c30231b26d668b1e8168074e5ab514b174d656b67e859895117f1f8f6`).
The edge-case file was transcribed from the draft text into the same JSON fields:
`ikmR`, `ikmE`, `skRm`, `pkRm`, `enc`, `shared_secret`, `key`, `base_nonce`, `exporter_secret`,
`encryptions`, and `exports`. Its encryptions are listed in sequence order from 0.
rscrypto does not implement the SHAKE256 KDF, so the tests use only the KEM values of A.12:
the derived key pair, the encapsulation, and the shared secret.

Internet-Draft and RFC test data is subject to the IETF Trust Legal Provisions. The Wycheproof files are
licensed under Apache-2.0. These files contain test data only; no upstream implementation source is included.

`SHA256SUMS` is the complete payload inventory; verify it from this directory with `shasum -a 256 -c SHA256SUMS`.
