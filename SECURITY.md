# Security Policy

## Report a vulnerability

Report real-world vulnerabilities through GitHub
[Private Vulnerability Reporting](https://github.com/loadingalias/rscrypto/security/advisories/new).
Do not open a public issue.

Include:

- Affected version and commit, if known.
- Minimal reproduction steps or proof of concept.
- Expected and actual behavior.
- Relevant platform and CPU details.
- Whether exploitability depends on features such as `std`, `alloc`,
  `getrandom`, `serde`, `serde-secrets`, or hardware acceleration.

Do not include live keys, credentials, personal data, or other secrets. Replace
them with a minimal synthetic reproducer.

You should receive an acknowledgment within 72 hours. If you do not, follow up
on the private advisory thread.

## Scope

In scope:

- Cryptographic correctness failures, including behavior that disagrees with
  documented standards, profiles, official vectors, or crate security
  boundaries.
- Timing side channels in claimed constant-time code paths.
- Oracle behavior beyond the documented opaque failure shape in authentication,
  decryption, decapsulation, signature verification, or key-agreement APIs.
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

## Supported versions

| Version | Supported |
| ------- | --------- |
| `0.7.x` | Yes |

Only the latest patch release in the current minor line receives security
fixes. Upgrade to the latest published `0.7.x` release before reporting an issue
that may already be fixed.

## Security posture

`rscrypto` is a pure Rust primitive crate with no mandatory production C/FFI
dependency. External crypto crates used for testing, fuzzing, migration checks,
or benchmarks are not production dependencies.

Constant-time claims apply only to named secret-bearing operations and exact
release configurations. They do not cover every API or build. Read
[`docs/constant-time.md`](docs/constant-time.md) for the claim model and
[`THREAT_MODEL.md`](THREAT_MODEL.md) for the security boundary, adversaries,
and review priorities.

No third-party security audit, FIPS 140-3 validation, or formal proof is
claimed.

## Advisory contents

When a report is valid, the project advisory records:

- Affected `rscrypto` versions and feature flags.
- Impacted primitives or parsing surfaces.
- Whether exploitation requires `std`, `alloc`, `getrandom`, `serde`,
  hardware acceleration, or a specific target architecture.
- Severity, patched version, and credit preference.
- CVSS and CWE when they are clear enough to be useful.

## AI-assisted reports

AI-assisted reports are welcome when they are reproducible. If AI or automated
analysis helped produce the report, disclose the tool or model used and include
the concrete inputs, outputs, traces, or reproduction steps that support the
finding.

## Response process

1. Triage within 72 hours.
2. Reproduce the issue and assess severity.
3. Prepare, test, and release a fix when the finding is valid.
4. Publish an advisory and credit the reporter if requested.

The default disclosure window is 30 days from the initial report unless a
different timeline is agreed on in the private advisory.

## Safe harbor

Good-faith research is welcome when it avoids privacy violations, data
destruction, service interruption, and access to third-party systems. Do not
exploit a vulnerability beyond what is necessary to demonstrate impact.

The project does not intend to pursue legal action for research conducted and
reported in accordance with this policy. This statement cannot bind third
parties.

## Acknowledgments

Reporters are credited in release notes and advisories unless they request
anonymity.
