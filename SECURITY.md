# Security Policy

Report a suspected vulnerability through GitHub [Private Vulnerability Reporting](https://github.com/loadingalias/rscrypto/security/advisories/new).
Do not open a public issue.

Include:

- the affected release or commit;
- a minimal reproducer or proof of concept;
- the expected behavior, the actual behavior, and the security impact;
- the relevant features, target, operating system, and CPU.

Automated and AI-assisted reports must include the inputs, outputs, traces,
or reproduction steps that show the finding.
Never send live keys, credentials, personal data, or other secrets.
Use synthetic values.

## Supported releases

Security fixes go into the latest published release.
Reproduce the issue there if you can.
Reports about older releases are welcome when the issue can still affect current code.

## Scope

Report issues that can affect users or published artifacts, including:

- correctness failures in a primitive or a protocol profile;
- secret disclosure through timing, memory, output, or error behavior;
- authentication, decryption, decapsulation, signature, or key-agreement failures
  that accept invalid input or expose an unintended oracle;
- memory unsafety, panics from hostile input, or unbounded resource use;
- security-relevant defects in the API, dependencies, build, or release integrity.

[`THREAT_MODEL.md`](THREAT_MODEL.md), [`ct.toml`](ct.toml),
and [`docs/constant-time.md`](docs/constant-time.md) define the exact security boundary and the constant-time claim model.

These are not vulnerabilities in `rscrypto`:

- performance regressions without security impact;
- expensive parameters that the caller selects within the documented bounds;
- defects in local tooling with no effect on users or artifacts;
- downstream code that breaks the documented API contract.

## Response and disclosure

The project acknowledges a report within 72 hours.
It then reproduces the issue, assesses its impact,
and coordinates a tested fix and an advisory when one is needed.
The default disclosure window is 30 days from the first report,
unless the project and the reporter agree on another timeline in the private advisory.

Reporters get credit in the advisory and the release notes, unless they ask to stay anonymous.

## Safe harbor

Good-faith research is welcome when it avoids privacy violations, data destruction,
service interruption, and access to third-party systems.
Do not exploit a vulnerability beyond what you need to show its impact.

The project does not intend to take legal action for research done and reported under this policy.
This statement cannot bind third parties.
