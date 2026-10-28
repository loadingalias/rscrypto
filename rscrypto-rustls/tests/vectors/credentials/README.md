# TLS credential and signature fixtures

These public test credentials and signatures were generated with OpenSSL 4.0.3 on 2026-10-11.
Each P-256, P-384, Ed25519, RSA-3072, and ML-DSA-44/65/87 root signs a distinct leaf for `localhost`,
usable for client and server authentication.
Certificates are valid from 2026-01-01 through 2036-01-01;
in-process tests use a fixed time within that interval.
All leaf keys are PKCS#8.
Additional SEC1 and PKCS#1 files exercise the alternate key containers
and raw RSA verification keys.

The `*-chain-*` and `rsa4096-*` fixtures were added on 2026-10-18 with the same OpenSSL version and validity interval.
Each chain fixture has three distinct keys: root → intermediate → leaf.
The root allows one intermediate; the intermediate allows no further CA.
Only the leaf private key is retained.
The RSA-4096 root signs a separate RSA-4096 leaf directly.
These fixtures support the TLS benchmark's chain-depth and RSA-key-size comparisons.

The `rsa2048-*` fixtures were regenerated on 2026-10-28 with OpenSSL 4.0.3 and the same validity
interval: an RSA-2048 root signs an RSA-2048 leaf, with PKCS#1 and PKCS#8 keys and the six RSA
signatures that RSA-3072 has. They exercise the provider's default 2,048-bit signing-key floor and
the stricter 3,072-bit policy. The earlier RSA-2048 key, public key, and PSS signature had no
certificate and were replaced.

The private keys are intentionally public and must never protect real traffic.
`SHA256SUMS` pins every payload.
Root private keys are not shipped.

ECDSA signatures cover SHA-256 and SHA-384.
RSA covers PKCS#1 v1.5 and PSS with SHA-256, SHA-384, and SHA-512; PSS uses a digest-length salt.
Ed25519 signs the same `message.txt` directly.
Tests check these independent signatures through both this provider and ring.
They also verify provider-generated signatures with ring.

ML-DSA fixtures include RFC 9881 combined and seed-only PKCS#8, SPKI,
pure signatures with an empty context, and wrong-context signatures.
Go's importer uses the seed-only form.
Tests cover both forms, exact parameter-set binding,
and independent AWS-LC verification of provider signatures.

Generate equivalent fixtures in separate new directories:

```bash
scripts/lib/python.sh scripts/test/provider_credentials.py /tmp/provider-credentials-new --profile classical
scripts/lib/python.sh scripts/test/provider_credentials.py /tmp/provider-mldsa-new --profile mldsa
scripts/lib/python.sh scripts/test/provider_credentials.py /tmp/provider-rsa2048-new --profile rsa2048
scripts/lib/python.sh scripts/test/provider_credentials.py /tmp/provider-tls-bench-new --profile tls-bench
```

The generator runs the OpenSSL 4.0.3 that `scripts/lib/openssl_peer.py` builds unless `--openssl` names one.

The generator records its commands beside that directory.
It uses fresh keys and randomized signatures, so it reproduces the contracts, not the pinned bytes.
The original P-256 certificate subjects omit the algorithm name;
that difference has no effect on certificate purpose, identity, or key usage.
