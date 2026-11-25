# Bouncy Castle 1.87 composite certificate chains

Bouncy Castle 1.87 issued these six public DER certificates on Temurin 25.0.5+8.
Each RFC 10073 profile has a root, intermediate, and `localhost` server certificate.
The generator used fresh keys and wrote only certificates. [RFC 10073](https://www.rfc-editor.org/rfc/rfc10073.html).

Run `just test-composite-certificates` from the repository root.
The Java peer checks rscrypto-issued chains. The Rust provider checks both Bouncy Castle chains.
Each verifier must reject a changed ML-DSA signature and a changed classical signature in each leaf.
The Rust tests also regenerate the rscrypto-issued chains and compare their DER bytes.

The Java runner checks the SHA-256 pins of `bcprov`, `bcutil`, and `bcpkix` from Maven Central.
Install the managed artifacts with `machine update all`. Set `JAVA_HOME` to Temurin 25 if needed.
To issue new fixtures, pass an empty directory to `scripts/lib/python.sh scripts/test/composite_certificates.py --issue DIRECTORY`.
New keys produce different certificates. `SHA256SUMS` identifies the fixtures from this run.
