# Contributing

Use a short-lived feature branch for every change. `main` is the protected,
releasable history; it is not a working branch.

## What the gates do

| Action | Purpose |
|---|---|
| Commit | Creates a local, reviewable checkpoint. |
| Push | Shares the branch after fast, change-aware local checks. |
| Pull request | Declares merge intent and runs CI on the proposed change. |
| Merge | Adds the reviewed change to protected `main`. |
| `main` CI | Records proof for the exact commit that can become a release. |

Pull-request and `main` CI may test the same source tree, but they authorize
different events. Pull-request CI decides whether a change may merge. `main`
CI proves the exact protected-branch commit used by the release gate.

## Daily branch workflow

Start from current `main`:

```bash
git switch main
git pull --ff-only
git switch -c <short-feature-name>
```

Make one focused change. Add a `.changes/*.md` file when crate users will
observe the result: an API, behavior, security, performance, compatibility, or
release-artifact change. Internal-only tooling and maintainer-documentation
changes normally do not need one.

```bash
just release-change patch "Describe the user-visible result."
```

Use `minor` or `major` instead of `patch` when the compatibility impact
requires it. The pre-push check is the final authority on whether release
intent is missing.

Run checks proportional to the change. Common starting points are:

```bash
just check
just test
```

For release-facing or broad shared changes, run:

```bash
just check-all
just test --all
```

Use deeper checks where the risk requires them:

| Change | Required validation |
|---|---|
| Parser, import, DER, PHC, hex, or untrusted input | `just test-fuzz <target>` or `just test-fuzz --all` |
| `unsafe`, SIMD, ASM, or dispatch | Backend equivalence tests and `just test-fuzz-asan --all` where the target runs natively |
| Portable unsafe path | `just test-miri` |
| Constant-time claim boundary | `just ct-full --target <triple>`; update `ct.toml` only with matching evidence |
| Public API change | `cargo semver-checks --package rscrypto --all-features` |
| Dependency or release change | `cargo deny check all` and `cargo audit --ignore RUSTSEC-2023-0071` |

Inspect and commit only the intended files:

```bash
git status --short
git diff --check
git add <files>
git diff --cached
git commit -m "module: imperative outcome"
```

Push the current branch with its upstream. No extra Git flags are needed:

```bash
just push
```

`just push` runs the light, change-aware pre-push plan and then preserves any
installed Git hooks. Use `just push-full` when the change is unusually broad
or release-sensitive.

Open the pull request. A branch push alone does not start the normal PR suite:

```bash
gh pr create --base main --fill
```

Wait for the required `Complete` check. Because the repository currently has
one maintainer, no second approval is required; review the final diff yourself,
resolve any open threads, and merge in the GitHub UI.

After the merge:

```bash
git switch main
git pull --ff-only
git branch -d <short-feature-name>
```

GitHub can delete the remote branch during the merge. Do not create release
tags during daily development.

## Security boundaries

Do not broaden constant-time, FIPS, audit, or compliance claims without
matching evidence. The constant-time boundary is the `ct_claimed` set in
[`ct.toml`](ct.toml), interpreted by
[`docs/constant-time.md`](docs/constant-time.md). The external audit entry
point is [`THREAT_MODEL.md`](THREAT_MODEL.md).

Report real vulnerabilities through GitHub Private Vulnerability Reporting,
not public issues. See [`SECURITY.md`](SECURITY.md).

## Fuzz corpus

Fuzz targets live in [`fuzz/`](fuzz/) and feature-scoped packages live in
[`fuzz-packages/`](fuzz-packages/). Commit small, stable corpus seeds that
exercise real parser or primitive paths. Do not commit `target/`, `artifacts/`,
coverage output, crashers, or bulk local corpus output without minimization.

## Releases

Releases add stronger identity, evidence, and publication gates to this daily
loop. Follow the canonical [release runbook](docs/release.md); do not publish
with a local crates.io token.
