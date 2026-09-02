# Release process

Cargo Rail owns release planning, mutation, exact-SHA readiness, and the signed
tag. GitHub Actions qualifies that tag and publishes it with short-lived
credentials. Do not run `cargo publish` locally.

## Release

After every intended change and its reviewed `.changes/*.md` entry has merged,
start from clean, current, green `main` and run:

```sh
cargo rail release run rscrypto --wait
```

Cargo Rail owns the complete local transaction. It infers the bump, consumes
release intent, updates the version, changelog, root lockfile, and every
standalone lockfile declared by `release.auxiliary_cargo_manifests`, commits the
exact mutation, and pushes it. It then waits for the normal `Complete` check on
that commit—and every other exact-SHA check—to succeed before creating and
pushing the signed tag. A dirty, stale, non-default, or rejected checkout fails
without creating the tag.

The tag starts the `Release` workflow. That run captures one all-work Cargo
Rail plan and runs every qualification domain concurrently while the release
package is validated in parallel. Publication waits for both results. Approve
the `crates-io` environment only then; CI publishes the exact validated crate
through Trusted Publishing and cuts the immutable, attested GitHub Release.

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
- Restrict the `main` bypass to the maintainer identity used by Cargo Rail;
  direct release commits require that authority. Keep force pushes, deletion,
  and release-tag updates blocked. Cargo Rail does not push the tag until the
  exact release commit's checks, including `Complete`, succeed.
- Enable immutable GitHub Releases.
- Require maintainer approval for the `crates-io` environment and disable
  administrator bypass.
- Keep long-lived crates.io tokens out of repository secrets.

The environment name must match crates.io and
`.github/workflows/release.yaml`. After the first successful trusted release,
enable crates.io Trusted Publishing Only Mode.

## Retry

Rerun a transient failure on the same tag and commit. Successful qualification
and package jobs remain authoritative; rerun only the failed jobs:

```sh
gh run rerun RUN_ID --failed
```

There is no second recovery protocol. A workflow defect requires a new reviewed
candidate, a new release-mode Qualification run, and a new signed tag. Published
immutable release assets are never replaced.

## Verify a release

```sh
gh release download vX.Y.Z --repo loadingalias/rscrypto \
   -p 'rscrypto-X.Y.Z.crate' \
   -p 'rscrypto-X.Y.Z-source.tar.gz' \
   -p 'rscrypto-X.Y.Z-ct-evidence.tar.gz' \
   -p SHA256SUMS
sha256sum --check SHA256SUMS
gh release verify vX.Y.Z --repo loadingalias/rscrypto
gh attestation verify rscrypto-X.Y.Z.crate --repo loadingalias/rscrypto
gh attestation verify rscrypto-X.Y.Z-source.tar.gz --repo loadingalias/rscrypto
gh attestation verify rscrypto-X.Y.Z-ct-evidence.tar.gz --repo loadingalias/rscrypto
```

The crate downloaded from crates.io must have the same SHA-256 as the attested
GitHub Release artifact.
