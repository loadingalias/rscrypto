# Contributing

Develop every change on a short-lived branch and merge it through a pull
request. The protected `main` branch is releasable history, not a working
branch.

## Start a change

Start from a clean, current `main`:

```bash
git status --short
git switch main
git pull --ff-only
git switch -c <short-feature-name>
```

Do not discard unrelated work to make the worktree clean. Preserve it or move
it to its own branch first.

## Record release intent

Add a `.changes/*.md` file when crate users will observe an API, behavior,
security, performance, compatibility, or release-artifact change:

```bash
cargo rail change add rscrypto --bump patch --message "Describe the user-visible result."
```

Use `minor` or `major` when compatibility requires it. Internal tooling and
maintainer-only documentation normally need no change file. The pre-push check
enforces this contract.

## Configure compiler reuse

Cargo Rail can reuse compiler results across Cargo, Nextest, Just, and IDE
invocations. For the first-party development fleet, choose a configured
rscrypto target from `~/dev-machines/dev-machine list rscrypto`, acquire a
short-lived credential, and install the canonical remapped policy into the
active Cargo home:

```bash
eval "$("$HOME/dev-machines/dev-machine" cache-env rscrypto <target>)"
just rail-cache-setup --max-size 10GiB
just cache-status
```

Run `cache-env` again when its short-lived R2 lease expires. `dev-machine ssh`
and `dev-machine just` refresh the corresponding remote-machine lease before
execution. Keep cache credentials outside repository configuration. Use
`CARGO_RAIL_CACHE=off` only when a check requires a cold compiler process,
including Miri and machine-code zeroization evidence.

## Validate

Run `just --list` to discover the current recipes. Start with:

```bash
just check
just test
```

For broad or release-facing changes, run:

```bash
just check-all
just test --all
```

Add the risk-specific evidence reached by the change:

| Change | Required evidence |
| --- | --- |
| Parser, import, DER, PHC, hex, or hostile input | `just test-fuzz <target>` or `just test-fuzz --all` |
| Unsafe Rust, SIMD, assembly, or dispatch | Backend differential tests; `just test-fuzz-asan --all` where native |
| Portable unsafe path | `just test-miri` |
| Constant-time claim boundary | `just ct-full --target <triple>`; update `ct.toml` only with matching evidence |
| Apple Silicon RSA assembly | `just test-rsa-macos-asm` on physical Apple Silicon |
| Public API, examples, or compatibility | Run `just test-examples`; review callers, tests, docs, migration guidance, and release intent |
| Dependency or release | `just check`; inspect the selected graph and the release contract |

Cross-compilation proves compilation, not runtime behavior, constant-time
execution, or performance. Record target lanes that cannot run.

## Review and submit

Inspect and commit only the intended files:

```bash
git status --short
git diff --check
git add <files>
git diff --cached
git commit -m "module: imperative outcome"
```

Push the current branch:

```bash
just push
```

`just push` checks the outgoing diff, changed shell and Just syntax, affected
Cargo graph inputs, and release-intent coverage. It does not replace explicit
builds, tests, or risk-specific evidence.

Open a draft pull request:

```bash
gh pr create --base main --fill --draft
```

Mark it ready only when the head is ready for CI. Before merging, resolve every
review thread, inspect the final diff, and require the protected `Complete`
check to pass.

## Security and test evidence

Do not broaden constant-time, audit, FIPS, compliance, secret-lifecycle, or
platform claims without matching evidence. Security boundaries are defined by
[`THREAT_MODEL.md`](THREAT_MODEL.md), [`ct.toml`](ct.toml), and the linked
evidence documents. Report vulnerabilities privately through
[`SECURITY.md`](SECURITY.md).

Use official vectors or an independent implementation as the oracle for
cryptographic correctness. Keep vector provenance, licensing, transforms, and
coverage reviewable. Fuzz targets live in [`fuzz/`](fuzz/) and
[`fuzz-packages/`](fuzz-packages/); commit only small, minimized seeds that
exercise production paths.

## Releases

Daily development does not create release tags or publish crates. Follow the
canonical [release runbook](docs/release.md), which owns exact-commit evidence,
publication, and post-release verification.
