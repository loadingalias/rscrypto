# Security Policy

## Reporting a Vulnerability

I take the security of `rscrypto` seriously. If you believe you have found a security vulnerability, please report it as described below.

**Please DO NOT report security vulnerabilities through public GitHub issues.**

### How to Report

Use GitHub's [Private Vulnerability Reporting](../../security/advisories/new).

Include as much detail as possible:
- Affected version(s) and commit hash
- Minimal repro steps and/or proof-of-concept (POC)
- Expected vs. actual behavior
- Any relevant platform/CPU details (x86_64, aarch64, etc.)

You should receive an acknowledgment within **72 hours** at the latest. I will follow up with a more detailed response indicating next steps and an expected resolution timeline. If you do not receive an acknowledgment within 72 hours, please follow up directly.

## Scope

### In Scope

- Cryptographic correctness/soundness (incorrect outputs, weak primitives)
- Timing side-channels in constant-time code paths
- Memory safety issues in `unsafe` blocks
- Nonce/key misuse that the API fails to prevent
- Supply chain or build system vulnerabilities
- Resource exhaustion caused by untrusted inputs, encoded password-hash params, or parser behavior
- CI or release automation vulnerabilities that can affect published artifacts

### Out of Scope

- Benchmark regressions or performance issues
- Pure availability reports against caller-chosen expensive params with no untrusted-input or crash/UB component
- Issues in local-only dev tooling that cannot affect builds, releases, users, or generated artifacts
- Vulnerabilities in downstream crates that use `rscrypto` incorrectly

## AI-Generated Reports

I welcome vulnerability reports regardless of how they were discovered, including those found using AI (LLMs, automated analysis tools, fuzzing agents, etc.). This is not an issue here; it's welcomed.

**If your report was generated or assisted by AI, please disclose:**
- The AI system(s) used (model name, version, provider)
- The prompts or instructions given to the AI
- Any context, code snippets, or specifications you provided as input
- Whether the AI produced the report autonomously or assisted your own analysis
- Any intermediate outputs, reasoning traces, or tool calls the AI generated

This information helps me:
- Reproduce and validate the finding efficiently
- Understand which attack surfaces or code paths the AI focused on
- Improve our own automated analysis and fuzzing pipelines
- Contribute to the broader security research community

I will almost never penalize or deprioritize AI-assisted or generated reports. Full transparency benefits everyone and AI is an outstanding tool to that end.

## Response Process

1. **Triage** (0-72 hours): I, or someone I trust, will confirm receipt and assess severity.
2. **Investigation** (0-7 days): I will reproduce the issue and determine root cause.
3. **Remediation** (0-14 days): I will develop and test a fix.
4. **Disclosure**: I will coordinate a public advisory, credit the reporter (if desired), and publish a patched release.

I follow a **30-day disclosure deadline**. If a fix is not available within 30 days of the initial report, I will publish the vulnerability details publicly, regardless of fix status.

## Supported Versions

| Version | Supported |
| ------- | --------- |
| `0.3.x` | Yes |

Only the latest published minor release receives security patches. I strongly recommend staying current.

## Safe Harbor

I support safe harbor for security researchers who:
- Make a good-faith effort to avoid privacy violations, data destruction, and service interruption
- Do not exploit a vulnerability beyond what is necessary to demonstrate it
- Do not modify user data or access third-party systems
- Report the vulnerability to me promptly and allow reasonable time for remediation

I will not pursue legal action against researchers who follow these guidelines under any circumstances.

## Acknowledgments

I appreciate the security research community and will publicly credit reporters in release notes and advisories unless anonymity is requested. If you wish to remain anonymous, please let me know... I will fight for your privacy as if it were my own.
