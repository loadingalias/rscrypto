# Contributing

`rscrypto` is a cryptography primitives crate. Keep changes small, measured,
and scoped to the primitive or tool path they affect.

## Before Opening A PR

Run the narrowest checks that prove the change:

```bash
just check
just test
```

For release-facing or shared changes, run:

```bash
just check-all
just test --all
```

Use the deeper lanes when relevant:

| Change | Required validation |
|---|---|
| Parser, import, DER, PHC, hex, or untrusted input | `just test-fuzz <target>` or `just test-fuzz --all` |
| `unsafe`, SIMD, ASM, or dispatch | backend equivalence tests plus `just test-fuzz-asan --all` where the target runs natively |
| Portable unsafe path | `just test-miri` |
| Constant-time claim boundary | `just ct-full --target <triple>` and update `ct.toml` only with matching evidence |
| Public API change | `cargo semver-checks --package rscrypto --all-features` |
| Dependency or release change | `cargo deny check all` and `cargo audit --ignore RUSTSEC-2023-0071` |

## Security Boundaries

Do not broaden constant-time, FIPS, audit, or compliance claims without matching
evidence. The constant-time boundary is the `ct_claimed` set in
[`ct.toml`](ct.toml), interpreted by [`docs/constant-time.md`](docs/constant-time.md).
The external audit entry point is [`THREAT_MODEL.md`](THREAT_MODEL.md).

Report real vulnerabilities through GitHub Private Vulnerability Reporting, not
public issues. See [`SECURITY.md`](SECURITY.md).

## Fuzz Corpus

Fuzz targets live in [`fuzz/`](fuzz/) and feature-scoped packages live in
[`fuzz-packages/`](fuzz-packages/). Commit small, stable corpus seeds that
exercise real parser or primitive paths. Do not commit `target/`, `artifacts/`,
coverage output, crashers, or bulk local corpus output without minimization.

## Release Changes

Normal releases are tag-triggered and CI-published. Do not run `cargo publish`
locally for a normal release. Use [`docs/release.md`](docs/release.md).
