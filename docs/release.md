# Release Process

`rscrypto` releases are human-tagged and CI-published.

`cargo-rail` owns the release mutation: version bump, changelog, signed tag,
and git push. GitHub Actions owns crates.io publishing and the GitHub Release.
Do not run `cargo publish` locally for normal releases.

## One-Time Setup

Configure the crate on crates.io:

| Field | Value |
|---|---|
| Repository owner | `loadingalias` |
| Repository name | `rscrypto` |
| Workflow filename | `release.yaml` |
| Environment | `crates-io` |

After the first successful Trusted Publishing release, enable crates.io
Trusted Publishing Only Mode for `rscrypto`. That disables traditional API
token publishing for new versions.

Configure the GitHub repository:

1. Create an environment named `crates-io`.
2. Add required reviewers for that environment.
3. Keep long-lived crates.io publish tokens out of repository secrets.

The environment name must match both crates.io and
[`.github/workflows/release.yaml`](../.github/workflows/release.yaml). If it
does not match, `rust-lang/crates-io-auth-action` cannot exchange the GitHub
OIDC token for a temporary crates.io token.

## Release Commands

Run the checks first:

```bash
git status --short
cargo rail release check rscrypto --extended
cargo rail release run rscrypto --bump patch --skip-publish --check
```

Use `--bump minor`, `--bump major`, or `--bump X.Y.Z` when that is the intended
version.

When the plan is correct, run the release command:

```bash
cargo rail release run rscrypto --bump patch --skip-publish
```

That command is allowed to push because `.config/rail.toml` has `push = true`.
It must not publish to crates.io; the `--skip-publish` flag is part of the
release contract.

## CI Release Gate

Pushing a `vX.Y.Z` tag starts the `Release` workflow. The workflow:

1. Requires an annotated SSH-signed tag trusted by
   [`.github/allowed-signers`](../.github/allowed-signers).
2. Requires the tag target to match the checked-out commit.
3. Requires the tag version, `Cargo.toml` version, and `CHANGELOG.md` heading
   to match.
4. Runs `cargo rail release check rscrypto --extended`.
5. Builds the `.crate` with `cargo package --locked`.
6. Rejects dirty VCS metadata and private or local-only package contents.
7. Waits for the `CI` workflow on the same commit to pass.
8. Attests the `.crate` artifact with GitHub build provenance.
9. Uses crates.io Trusted Publishing to get a temporary publish token.
10. Publishes with `cargo publish -p rscrypto --locked`.
11. Downloads the crate from crates.io and verifies the SHA-256 against the
    attested package.
12. Creates or updates the GitHub Release with the crate and `SHA256SUMS`.

The publish job uses the `crates-io` GitHub environment. GitHub requests
approval only after preflight and CI have passed, and before the OIDC token is
issued.

## Recovery

Re-running the workflow is safe after a partial failure. If crates.io already
has the version, the workflow downloads it and compares SHA-256 before touching
the GitHub Release. A mismatch is a hard stop.

If the signed-tag key changes, update `.github/allowed-signers` in a reviewed
commit before creating the next release tag.

## Verification

Consumers can verify the GitHub Release artifact:

```bash
gh release download vX.Y.Z --repo loadingalias/rscrypto -p 'rscrypto-*.crate' -p SHA256SUMS
shasum -a 256 -c SHA256SUMS
gh attestation verify rscrypto-X.Y.Z.crate --repo loadingalias/rscrypto
```

The crate downloaded from crates.io should have the same SHA-256 as the
attested release artifact.
