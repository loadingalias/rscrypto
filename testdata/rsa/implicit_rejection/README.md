# RSAES-PKCS1-v1_5 implicit-rejection vectors

`appendix_b.json` contains the four private keys and all 48 cases from Appendix B of
[draft-irtf-cfrg-rsa-guidance-10](https://www.ietf.org/archive/id/draft-irtf-cfrg-rsa-guidance-10.txt),
dated 2026-09-11. The source text has SHA-256
`9e2af1e3e15744ef3b6575184d97dcc5ca22af427f6c93235e1bf23bca9952e6`.
The generator removes page furniture, decodes PKCS #8 PEM to hex DER, and converts
ASCII messages to hex. It preserves every ciphertext and expected message.
Each case also records OpenSSL's raw RSA result as `encoded_message`, using
`openssl pkeyutl -decrypt -inkey key.der -keyform DER -pkeyopt rsa_padding_mode:none -in ciphertext.bin`.
That independent intermediate lets the unit tests exercise production
derivation and depadding for all 48 cases without requiring private-key import.
The draft is an IRTF Informational Internet-Draft, not an IETF standard.
Its copyright and licensing terms are the IETF Trust terms linked in the source.

`openssl_4_0_3.json` records independent results from OpenSSL 4.0.3 (2026-09-29).
Generation first requires OpenSSL to reproduce all 48 Appendix B messages, then
decrypts 128 seeded ciphertexts (32 per key) and 59 Wycheproof invalid-padding
cases. The command for each DER key and ciphertext is:

```sh
openssl pkeyutl -decrypt -inkey key.der -keyform DER \
  -pkeyopt rsa_padding_mode:pkcs1 -pkeyopt rsa_pkcs1_implicit_rejection:1 \
  -in ciphertext.bin
```

The random inputs use Python `random.Random(0x52534149)` and `randbytes(k - 1)`
with a zero leading octet, keeping each representative below its modulus.
The generator implements no RSA arithmetic.

The Wycheproof inputs are the checked-in, normalized files from
[`C2SP/wycheproof`](https://github.com/C2SP/wycheproof/tree/b61843a9a5115bb758134b6a1f5d5e502d445342/testvectors_v1)
commit `b61843a9a5115bb758134b6a1f5d5e502d445342`, licensed under Apache-2.0.
See [their provenance](../wycheproof/README.md).

| Input | SHA-256 |
| --- | --- |
| `rsa_pkcs1_2048_test.json` | `41ed2e3ca6ec1b35d1ff5f6dff3272319eb47b51a7edd0aa5323ce86a36f6159` |
| `rsa_pkcs1_3072_test.json` | `febd0293129218c2d8f35a88542c5d0b1995fb9964d5cb318cc88a528f8f2667` |
| `rsa_pkcs1_4096_test.json` | `d3690019fefa50cff6d60b303b32bbebb182a620a9c626beedc57121697183f3` |

Reproduce with OpenSSL 4.0.3 on `PATH` and the pinned draft text:

```sh
scripts/lib/python.sh scripts/test/rsa_implicit_rejection_vectors.py --draft /path/to/draft-irtf-cfrg-rsa-guidance-10.txt
```

`SHA256SUMS` inventories the two generated payloads. From this directory,
`shasum -a 256 -c SHA256SUMS` verifies them.

`tests/rsa_implicit_rejection.rs` checks all four decryption entry points against
36 Appendix B cases and 96 seeded ciphertexts for the 2048-, 3072-, and 4096-bit
keys. It also checks the 59 Wycheproof synthetic messages and retains public
format errors. Appendix B.2's 2049-bit key is outside rscrypto's existing
half-width-factor import policy: the test requires `InvalidModulus` and skips its
12 decryption cases and 32 random ciphertexts. The complete B.2 inputs remain
here, and OpenSSL cross-checks them during generation; rscrypto decryption of
that key remains a public-API coverage gap. The production derivation and
depadding unit test includes all 12 B.2 cases at their full 257-byte width.
A `src/traits/ct.rs` unit test imports the B.1 key to check that each RSA
decryption entry point enters data-independent timing.
