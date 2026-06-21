# Security Policy

## Reporting A Vulnerability

Please do not report real-world vulnerabilities through public GitHub issues.

Use GitHub's
[Private Vulnerability Reporting](https://github.com/loadingalias/rscrypto/security/advisories/new).

Include:

- Affected version and commit, if known.
- Minimal reproduction steps or proof of concept.
- Expected and actual behavior.
- Relevant platform and CPU details.
- Whether exploitability depends on features such as `std`, `alloc`,
  `getrandom`, `serde-secrets`, or hardware acceleration.

You should receive an acknowledgment within 72 hours. If you do not, follow up
on the private advisory thread.

## Scope

In scope:

- Cryptographic correctness failures.
- Timing side channels in claimed constant-time code paths.
- ML-KEM key generation, encapsulation, decapsulation, or implicit-rejection
  behavior that disagrees with FIPS 203 or the documented CT boundary.
- Memory-safety issues in `unsafe` code.
- Nonce, key, signature, or verification behavior that creates
  security-relevant API misuse.
- Parser or resource-exhaustion behavior triggered by untrusted input.
- Build, release, or supply-chain issues that can affect published artifacts.

Out of scope:

- Benchmark regressions without a security impact.
- Expensive caller-chosen parameters with no crash, undefined behavior, or
  untrusted-input component.
- Local-only development tooling issues that cannot affect users, releases, or
  generated artifacts.
- Bugs caused by downstream crates using `rscrypto` outside its documented
  contract.

## Supported Versions

| Version | Supported |
| ------- | --------- |
| `0.5.x` | Yes |

Only the latest published minor release receives security patches. Users should
stay current.

## Security Posture

`rscrypto` is a pure Rust primitive crate with no mandatory production C/FFI
dependency. External crypto crates used for testing, fuzzing, migration checks,
or benchmarks are not production dependencies.

Constant-time claims are scoped. They apply only to named secret-bearing
operations and target configurations, not to every API or every build. See
[`docs/security.md`](docs/security.md) for application guidance and
[`docs/constant-time.md`](docs/constant-time.md) for the exact claim model.

ML-KEM-512/768/1024 are in scope for the documented FIPS 203 correctness and
constant-time evidence model. The ML-KEM claim covers declared secret inputs in
key generation, encapsulation, decapsulation, and implicit rejection; public
keys, ciphertext lengths, parse errors, profile selection, and matrix seeds
remain public-input work.

No third-party security audit, FIPS 140-3 validation, or formal proof is claimed
today.

## AI-Assisted Reports

AI-assisted reports are welcome when they are reproducible. If AI or automated
analysis helped produce the report, disclose the tool or model used and include
the concrete inputs, outputs, traces, or reproduction steps that support the
finding.

## Response Process

1. Triage within 72 hours.
2. Reproduce the issue and assess severity.
3. Prepare, test, and release a fix when the finding is valid.
4. Publish an advisory and credit the reporter if requested.

The default disclosure window is 30 days from the initial report unless a
different timeline is agreed on in the private advisory.

## Safe Harbor

Good-faith research is welcome when it avoids privacy violations, data
destruction, service interruption, and access to third-party systems. Do not
exploit a vulnerability beyond what is necessary to demonstrate impact.

Researchers who follow this policy will not face legal action from this project
for the reported activity.

## Acknowledgments

Reporters are credited in release notes and advisories unless they request
anonymity.
