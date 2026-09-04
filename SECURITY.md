# Security Policy

Report suspected vulnerabilities through GitHub
[Private Vulnerability Reporting](https://github.com/loadingalias/rscrypto/security/advisories/new).
Do not open a public issue.

Include:

- The affected release or commit.
- A minimal reproducer or proof of concept.
- Expected behavior, actual behavior, and security impact.
- Relevant features, target, operating system, and CPU.

Automated and AI-assisted reports must include the inputs, outputs, traces, or
reproduction steps that establish the finding. Never send live keys,
credentials, personal data, or other secrets; use synthetic values.

## Supported releases

Security fixes target the latest published release. Reproduce the issue there
when possible. Reports about older releases remain welcome when the issue may
still affect current code.

## Scope

Report issues that can affect users or published artifacts, including:

- Cryptographic or protocol-profile correctness failures.
- Secret disclosure through timing, memory, output, or error behavior.
- Authentication, decryption, decapsulation, signature, or key-agreement
  failures that accept invalid input or expose an unintended oracle.
- Memory unsafety, hostile-input panics, or unbounded resource use.
- Security-relevant API misuse, dependency, build, or release-integrity
  defects.

The exact security boundary and constant-time claim model are defined by
[`THREAT_MODEL.md`](THREAT_MODEL.md), [`ct.toml`](ct.toml), and
[`docs/constant-time.md`](docs/constant-time.md).

Performance regressions without security impact, expensive caller-selected
parameters within documented bounds, local tooling defects with no user or
artifact impact, and downstream violations of the documented API contract are
not vulnerabilities in `rscrypto`.

## Response and disclosure

The project will acknowledge a report within 72 hours, reproduce the issue,
assess its impact, and coordinate a tested fix and advisory when required. The
default disclosure window is 30 days from the initial report unless the project
and reporter agree to another timeline in the private advisory.

Reporters receive credit in the advisory and release notes unless they request
anonymity.

## Safe harbor

Good-faith research is welcome when it avoids privacy violations, data
destruction, service interruption, and access to third-party systems. Do not
exploit a vulnerability beyond what is necessary to demonstrate impact.

The project does not intend to pursue legal action for research conducted and
reported under this policy. This statement cannot bind third parties.
