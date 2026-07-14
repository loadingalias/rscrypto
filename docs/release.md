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
cargo rail change add rscrypto --bump patch --name concise-change-name --message "Describe the user-visible change."
cargo rail change status
```

Use `--bump minor` or `--bump major` for larger user-visible changes.

Run the pre-release checks from a clean tree:

```bash
git status --short
cargo rail config validate --strict
cargo rail config sync --check
cargo rail unify --check --explain
cargo rail release check rscrypto --extended
cargo rail release run rscrypto --bump auto --skip-publish --check
```

The same flow is available through the justfile:

```bash
just release-change patch "Fixed ECDSA oracle compatibility with p256."
just release-status
just release-check
```

Pull-request CI uses cargo-rail-action v5 with cargo-rail 0.17.0. The action's
surface outputs decide whether the single-crate CI suite is required, and its
resolved base ref feeds `cargo rail change check --required`. Execution scope
comes from the planner contract; workflows and scripts must not reconstruct it
from diagnostic impact fields.

Compiler-backed unification runs in the dedicated blocking Cargo Graph
Assurance job, not the fast Quality lane. The planner selects it for Cargo,
Rust source, build, test, bench, example, toolchain, or rail configuration
changes. Pushes to `main`, Weekly, and release preflight always run the complete
19-target proof. The job retains cargo-rail's structured JSON result and its
identity-validated compiler-evidence cache.

When the plan is correct, prepare the release candidate without creating a
tag:

```bash
just release-prepare
```

`release-prepare` lets cargo-rail create and push the version/changelog release
commit with `--skip-tag`, refreshes all standalone CT tool lockfiles against
that new version, and pushes one normal `workspace:` follow-up commit. This
keeps the release candidate reproducible without modifying a signed tag or
hand-editing generated lockfiles.

Wait for CI on the resulting `main` commit, then manually dispatch
`weekly.yaml` on that exact commit. Weekly requires both the full CT matrix and
the dedicated RSA workflow. If either fails, fix the release candidate and run
both gates again. Do not create the tag.

Once CI and Weekly are green, finalize the release:

```bash
just release-tag
```

The `release-tag` recipe reruns strict configuration and Cargo-graph validation,
then uses `cargo rail release finalize` to validate the materialized release,
create the signed tag, and push it. It does not rerun `cargo rail release check`:
that command validates pending release intent and correctly fails after
`release-prepare` consumes the change files. Finalization must not publish to
crates.io; the `--skip-publish` flag is part of the release contract.

## CI Release Gate

Pushing a `vX.Y.Z` tag starts the `Release` workflow. The workflow:

1. Requires an annotated SSH-signed tag trusted by
   [`.github/allowed-signers`](../.github/allowed-signers).
2. Requires the tag target to match the checked-out commit.
3. Requires the tag version, `Cargo.toml` version, and `CHANGELOG.md` heading
   to match.
4. Validates and sync-checks `.config/rail.toml`, then proves the unified Cargo
   graph is clean with cargo-rail 0.17.0.
5. Runs `cargo deny check all`, `cargo audit`, and the single hard SemVer check
   against the finalized release version. Before tagging, cargo-rail's automatic
   release plan uses compatibility analysis to select the required version bump;
   the extended release check reports the advisory verdict.
6. Builds the `.crate` once with `cargo package --locked` and transfers that
   validated artifact to the publish job.
7. Rejects dirty VCS metadata and private or local-only package contents.
8. Waits for the `CI` workflow on the same commit to pass.
9. Verifies the transferred artifact's SHA-256, then attests it with GitHub
   build provenance.
10. Runs every release CT lane and the dedicated RSA evidence workflow, and
    rejects any failed required gate.
11. Validates that the complete CT lane set, version, commit, clean provenance,
    tool versions, timing coverage, BINSEC coverage, and raw artifact hashes
    agree before creating the versioned CT evidence bundle.
12. Attests the CT evidence bundle with GitHub build provenance.
13. Uses crates.io Trusted Publishing to get a temporary publish token.
14. Publishes with `cargo publish -p rscrypto --locked`.
15. Downloads the crate from crates.io and verifies the SHA-256 against the
    attested package.
16. Creates or updates the GitHub Release with the crate, CT evidence bundle,
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
gh release download vX.Y.Z --repo loadingalias/rscrypto \
  -p 'rscrypto-X.Y.Z.crate' \
  -p 'rscrypto-X.Y.Z-ct-evidence.tar.gz' \
  -p SHA256SUMS
sha256sum --check SHA256SUMS
gh attestation verify rscrypto-X.Y.Z.crate --repo loadingalias/rscrypto
gh attestation verify rscrypto-X.Y.Z-ct-evidence.tar.gz --repo loadingalias/rscrypto
mkdir ct-evidence && tar -xzf rscrypto-X.Y.Z-ct-evidence.tar.gz -C ct-evidence
(cd ct-evidence && sha256sum --check CT-EVIDENCE-MANIFEST.txt)
```

The crate downloaded from crates.io should have the same SHA-256 as the
attested release artifact.
