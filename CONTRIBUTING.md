# Contributing

Use a short-lived feature branch for every change. `main` is the protected,
releasable history; it is not a working branch.

## Gate model

| Action       | Purpose                                                   |
| ------------ | --------------------------------------------------------- |
| Commit       | Creates a local, reviewable checkpoint.                   |
| Push         | Shares the branch after fast, change-aware local checks.  |
| Pull request | Declares merge intent and runs CI on the proposed change. |
| Merge        | Adds the reviewed change to protected `main`.             |

The protected branch requires an up-to-date `Complete` result and has no bypass
actors, so merged commits do not repeat the pull-request suite. Release
candidates get a separate exact-commit Weekly run with the complete CI suite,
Cargo graph assurance, and release evidence.

## Create a branch

Start from a clean worktree and current `main`:

```bash
git status --short
git switch main
git pull --ff-only
git switch -c <short-feature-name>
```

Do not discard unrelated local changes to run this sequence. Preserve them or
move them to their own branch first.

## Record release intent

Add a `.changes/*.md` file when crate users will observe an API, behavior,
security, performance, compatibility, or release-artifact change. Internal
tooling and maintainer-only documentation normally do not need one.

```bash
cargo rail change add rscrypto --bump patch --message "Describe the user-visible result."
```

Use `minor` or `major` instead of `patch` when the compatibility impact
requires it. The pre-push check is the final authority on whether release
intent is missing.

## Configure compiler reuse

Enable Cargo Rail once for the effective Cargo home on each development
machine:

```bash
cargo rail cache setup --check
cargo rail cache setup
cargo rail cache status --scope local --format json
```

The installed wrapper applies transparently to ordinary Cargo, nextest, Just,
and IDE invocations. Cargo Rail bypasses unsupported compiler operations; do
not add wrapper commands to recipes or scripts. To add a machine-owned remote
authority, preview and apply it explicitly:

```bash
cargo rail cache setup --check --remote '<provider-url>' --remote-mode read-write
cargo rail cache setup --remote '<provider-url>' --remote-mode read-write
```

Keep credentials outside the URL and repository. Use `read` for a consumer
identity and enforce the same restriction in the provider policy. Use
`CARGO_RAIL_CACHE=off` only when a check requires a deliberately cold compiler
process, such as machine-code zeroization evidence or Miri.

## Validate the change

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

| Change                                            | Required validation                                                                                       |
| ------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Parser, import, DER, PHC, hex, or untrusted input | `just test-fuzz <target>` or `just test-fuzz --all`                                                       |
| `unsafe`, SIMD, ASM, or dispatch                  | Backend equivalence tests and `just test-fuzz-asan --all` where the target runs natively                  |
| Portable unsafe path                              | `just test-miri`                                                                                          |
| Constant-time claim boundary                      | `just ct-full --target <triple>`; update `ct.toml` only with matching evidence                            |
| Apple Silicon RSA assembly                        | `just test-rsa-macos-asm` on a physical local Arm64 Mac; GitHub Actions intentionally has no macOS runner |
| Public API change                                 | Review callers, docs, migration guidance, and release intent; pre-1.0 SemVer enforcement is deferred      |
| Dependency or release change                      | `cargo deny check all` and `cargo audit --ignore RUSTSEC-2023-0071`                                       |

## Review and commit

Inspect and commit only the intended files:

```bash
git status --short
git diff --check
git add <files>
git diff --cached
git commit -m "module: imperative outcome"
```

## Push and open a pull request

Push the current branch with its upstream. No extra Git flags are needed:

```bash
just push
```

`just push` runs the light, change-aware pre-push plan once and then pushes the
current branch. No rscrypto Git-hook installation is required. Use `just
push-full` when the change is unusually broad or release-sensitive.

Open a draft pull request while the change is still in progress. Expensive jobs
wait until the pull request is ready for review; a branch push alone does not
start the normal pull-request suite:

```bash
gh pr create --base main --fill --draft
```

Mark the pull request ready only when its head is ready for CI, then wait for
the required `Complete` check. Resolve every open review thread and review the
final diff before merging in the GitHub UI. GitHub enforces the current
approval policy.

For a broad or release-sensitive pull request, run the slow physical assurance
lanes on the pushed branch before merging:

```bash
branch=$(git branch --show-current)
test -n "$branch" && test "$branch" != main
test "$(git rev-parse HEAD)" = "$(git rev-parse '@{upstream}')"
gh workflow run weekly.yaml --ref "$branch" -f mode=assurance
gh workflow run riscv.yaml --ref "$branch" -f mode=evidence
```

These branch runs expose platform, constant-time, and RISC-V failures before
they reach `main`. They do not replace the release runbook's post-merge,
exact-commit evidence: a squash or merge commit has a different SHA, and
ancestor evidence is never promoted into a release claim.

## Clean up after merge

After GitHub reports the pull request merged:

```bash
git switch main
git pull --ff-only
git branch -D <short-feature-name>
```

Squash merges do not place the topic branch tip in `main`'s ancestry, so
`git branch -d` can reject a branch whose pull request is already merged. Use
`-D` only after verifying that exact pull request. GitHub can delete the remote
branch during the merge; otherwise delete that exact branch with `git push
origin --delete <short-feature-name>`. Do not create release tags during daily
development.

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
