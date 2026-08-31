# Release process

Releases are prepared by Cargo Rail, approved by the maintainer, and published
by GitHub Actions. Do not run `cargo publish` locally.

## Release

1. Start from clean, current `main` after all intended changes have merged.
   Every user-visible change needs a reviewed `.changes/*.md` entry.

   ```sh
   git switch main
   git pull --ff-only
   git status --short
   cargo rail change status
   ```

2. Open the generated release pull request:

   ```sh
   just release-prepare
   ```

   Cargo Rail builds and atomically applies the release plan, consumes the
   required change files, updates the version and changelog, and opens the
   release pull request. The same transaction computes and binds every
   standalone Cargo lockfile declared by
   `release.auxiliary_cargo_manifests`. Wait for `Complete`, review the diff,
   and merge in GitHub.

3. Record the merged candidate and dispatch exact-commit evidence before
   another change reaches `main`:

   ```sh
   git switch main
   git pull --ff-only
   candidate=$(git rev-parse HEAD)
   gh workflow run weekly.yaml --ref main -f mode=release
   ```

   Confirm the Qualification run uses `$candidate`. Its single immutable Cargo
   Rail plan starts the platform, graph, feature, CT, RSA, coverage, and RISC-V
   evidence lanes concurrently. Scheduled or assurance-mode runs do not
   satisfy the release gate. Any change to source, dependencies, features,
   build inputs, or test policy creates a new candidate and requires a new
   release-mode Qualification run.

4. After Qualification passes, create the signed tag:

   ```sh
   test "$(git rev-parse HEAD)" = "$candidate"
   just release-tag
   ```

   The tag starts the `Release` workflow. Approve its `crates-io` environment
   only after prerequisite jobs pass. CI publishes an immutable, attested
   GitHub Release, then publishes the same crate through crates.io Trusted
   Publishing.

## Release intent

Add a change file with the smallest accurate bump:

```sh
cargo rail change add rscrypto --bump patch --message "Describe the user-visible result."
cargo rail change status
```

Use `minor` or `major` for compatibility changes. Before 1.0, reviewed minor
releases may change unstable public shapes, but callers, examples, migration
guidance, and release notes must still be updated.

## Repository setup

- Configure crates.io Trusted Publishing for owner `loadingalias`, repository
  `rscrypto`, workflow `release.yaml`, and environment `crates-io`.
- Enable the committed `protect-main` and `protect-release-tags` rulesets with
  no bypass actors.
- Enable immutable GitHub Releases.
- Require maintainer approval for the `crates-io` environment and disable
  administrator bypass.
- Keep long-lived crates.io tokens out of repository secrets.

The environment name must match crates.io and
`.github/workflows/release.yaml`. After the first successful trusted release,
enable crates.io Trusted Publishing Only Mode.

## Recovery

Rerun a transient failure on the same tag and commit:

```sh
gh run rerun RUN_ID --failed
```

If the committed workflow needs repair, merge the smallest fix through
`Complete`, then dispatch recovery from protected `main`:

```sh
gh workflow run release.yaml --ref main -f tag=vX.Y.Z
```

Recovery checks out and verifies the existing signed tag. It may repair a draft
release, but it cannot replace a published immutable release or publish bytes
that differ from an existing crates.io version.

If only the s390x constant-time artifact must be regenerated, run the complete
native lane against the existing tag, then pass that run to recovery:

```sh
gh workflow run ct.yaml --ref main \
  -f platforms=ibm-s390x \
  -f dudect_gate=required \
  -f upload_raw_artifacts=true \
  -f artifact_retention_days=90 \
  -f release_tag=vX.Y.Z

gh workflow run release.yaml --ref main \
  -f tag=vX.Y.Z \
  -f s390x_ct_run=RUN_ID
```

For x86_64, the recovery group is all four physical timing lanes and runs them
in parallel:

```sh
gh workflow run ct.yaml --ref main \
  -f platforms=amd-zen4,intel-spr,intel-icl,amd-zen5 \
  -f dudect_gate=required \
  -f upload_raw_artifacts=true \
  -f artifact_retention_days=90 \
  -f release_tag=vX.Y.Z

gh workflow run release.yaml --ref main \
  -f tag=vX.Y.Z \
  -f x86_64_ct_run=RUN_ID
```

Run both dispatches from the same reviewed `main` commit. The CT recovery is
limited to a complete supported platform group and checks out the immutable
tag; the release preflight rejects any replacement run from another workflow,
branch, repository, or commit. The normal release evidence packager then
validates every replacement artifact's tag commit, crate version, cases,
hashes, and target provenance before publication.

## Verify a release

```sh
gh release download vX.Y.Z --repo loadingalias/rscrypto \
  -p 'rscrypto-X.Y.Z.crate' \
  -p 'rscrypto-X.Y.Z-source.tar.gz' \
  -p 'rscrypto-X.Y.Z-ct-evidence.tar.gz' \
  -p 'rscrypto-X.Y.Z-release-manifest.json' \
  -p SHA256SUMS
sha256sum --check SHA256SUMS
gh release verify vX.Y.Z --repo loadingalias/rscrypto
gh attestation verify rscrypto-X.Y.Z.crate --repo loadingalias/rscrypto
gh attestation verify rscrypto-X.Y.Z-source.tar.gz --repo loadingalias/rscrypto
gh attestation verify rscrypto-X.Y.Z-ct-evidence.tar.gz --repo loadingalias/rscrypto
```

The crate downloaded from crates.io must have the same SHA-256 as the attested
GitHub Release artifact.
