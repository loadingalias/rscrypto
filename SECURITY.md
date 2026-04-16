# Security Policy

## Reporting a Vulnerability

I take the security of `rscrypto` seriously. If you believe you have found a security vulnerability, please report it as described below.

**Please DO NOT report security vulnerabilities through public GitHub issues.**

### How to Report

Use GitHub's [Private Vulnerability Reporting](../../security/advisories/new).

Include as much detail as possible:
- Affected version(s) and commit hash
- Minimal reproduction steps or proof-of-concept
- Expected vs. actual behavior
- Any relevant platform/CPU details (x86_64, aarch64, etc.)

You should receive an acknowledgment within **48 hours**. We will follow up with a more detailed response indicating next steps and an expected resolution timeline.

## Scope

### In Scope

- Cryptographic correctness (incorrect outputs, weak primitives)
- Timing side-channels in constant-time code paths
- Memory safety issues in `unsafe` blocks
- Nonce/key misuse that the API fails to prevent
- Supply chain or build system vulnerabilities

### Out of Scope

- Benchmark regressions or performance issues
- Denial of service via resource exhaustion (unless it triggers a crash/UB)
- Issues in optional dev-dependencies or CI tooling
- Vulnerabilities in downstream crates that use `rscrypto` incorrectly

## AI-Generated Reports

We welcome vulnerability reports regardless of how they were discovered — including those found with the assistance of AI systems (LLMs, automated analysis tools, fuzzing agents, etc.).

**If your report was generated or assisted by AI, please disclose:**
- The AI system(s) used (model name, version, provider)
- The prompts or instructions given to the AI
- Any context, code snippets, or specifications you provided as input
- Whether the AI produced the report autonomously or assisted your own analysis
- Any intermediate outputs, reasoning traces, or tool calls the AI generated

This information helps us:
- Reproduce and validate the finding efficiently
- Understand which attack surfaces or code paths the AI focused on
- Improve our own automated analysis and fuzzing pipelines
- Contribute to the broader security research community

We do not penalize or deprioritize AI-assisted reports. Full transparency benefits everyone.

## Response Process

1. **Triage** (0–48 hours): We confirm receipt and assess severity.
2. **Investigation** (1–7 days): We reproduce the issue and determine root cause.
3. **Remediation** (1–14 days): We develop and test a fix.
4. **Disclosure**: We coordinate a public advisory, credit the reporter (if desired), and publish a patched release.

We follow a **90-day disclosure deadline**. If a fix is not available within 90 days of the initial report, we will publish the vulnerability details publicly, regardless of fix status.

## Supported Versions

| Version | Supported |
| ------- | --------- |
| `0.1.x` | ✅ Yes |

Only the latest `0.1.x` release receives security patches. We strongly recommend staying current.

## Safe Harbor

We support safe harbor for security researchers who:
- Make a good-faith effort to avoid privacy violations, data destruction, and service interruption
- Do not exploit a vulnerability beyond what is necessary to demonstrate it
- Do not modify user data or access third-party systems
- Report the vulnerability to us promptly and allow reasonable time for remediation

We will not pursue legal action against researchers who follow these guidelines.

## Acknowledgments

We appreciate the security research community and will publicly credit reporters in release notes and advisories unless anonymity is requested.
