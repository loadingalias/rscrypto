# Release Process

`rscrypto` releases are human-tagged and CI-published.

`cargo-rail` owns the local release mutation: reviewed change file, version
bump, changelog, release commit, signed tag, and tag push. GitHub Actions owns
crates.io publishing and the GitHub Release. Do not run `cargo publish`
locally for normal releases.

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

Add release intent with a cargo-rail change file. Commit the change file with
the code change when possible; if the code was already committed, add the
change file before cutting the release. Change files live in `.changes/`.
Use `--name` to choose the filename slug. The file may be renamed after
creation as long as it remains a `.changes/*.md` file.

```bash
cargo rail change add rscrypto --bump patch --name release-workflow-cargo-rail-0-15 --message "Updated release workflow validation for cargo-rail 0.15."
cargo rail change status
```

Use `--bump minor` or `--bump major` for larger user-visible changes.

Run the pre-release checks from a clean tree:

```bash
git status --short
cargo rail config validate --strict
cargo rail release check rscrypto --extended
cargo rail release run rscrypto --bump auto --skip-publish --check
```

The same flow is available through the justfile:

```bash
just release-change patch "Fixed ECDSA oracle compatibility with p256."
just release-status
just release-check
```

When the plan is correct, run the release command:

```bash
cargo rail release run rscrypto --bump auto --yes --skip-publish
```

or:

```bash
just release-tag
```

The `release-tag` recipe runs strict config validation and the extended
pre-tag cargo-rail release check before it creates the release commit. That
command is allowed to create the release commit and push the signed tag because
`.config/rail.toml` has `push = true`. It must not publish to crates.io; the
`--skip-publish` flag is part of the release contract.

## CI Release Gate

Pushing a `vX.Y.Z` tag starts the `Release` workflow. The workflow:

1. Requires an annotated SSH-signed tag trusted by
   [`.github/allowed-signers`](../.github/allowed-signers).
2. Requires the tag target to match the checked-out commit.
3. Requires the tag version, `Cargo.toml` version, and `CHANGELOG.md` heading
   to match.
4. Validates `.config/rail.toml` with `cargo rail config validate --strict`.
5. Runs `cargo deny check all`, `cargo audit`, and `cargo semver-checks`.
6. Builds the `.crate` with `cargo package --locked`.
7. Rejects dirty VCS metadata and private or local-only package contents.
8. Waits for the `CI` workflow on the same commit to pass.
9. Attests the `.crate` artifact with GitHub build provenance.
10. Runs the release CT evidence workflow and attaches a versioned CT evidence
   bundle to the GitHub Release.
11. Uses crates.io Trusted Publishing to get a temporary publish token.
12. Publishes with `cargo publish -p rscrypto --locked`.
13. Downloads the crate from crates.io and verifies the SHA-256 against the
    attested package.
14. Creates or updates the GitHub Release with the crate, CT evidence bundle,
    and `SHA256SUMS`.

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
gh attestation verify rscrypto-X.Y.Z-ct-evidence.tar.gz --repo loadingalias/rscrypto
```

The crate downloaded from crates.io should have the same SHA-256 as the
attested release artifact.
