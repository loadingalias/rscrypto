# Release Process

`rscrypto` releases are approved by the maintainer and published by CI.
`cargo-rail` prepares a release pull request and creates the signed tag. GitHub
Actions builds, attests, and publishes the artifacts. A normal release must
never run `cargo publish` locally.

## Release at a glance

1. Start from a clean, current `main` after all intended feature pull requests
   have merged. Every user-visible change must already have a reviewed
   `.changes/*.md` file.

   ```bash
   git switch main
   git pull --ff-only
   git status --short
   just release-check
   ```

2. Prepare the release:

   ```bash
   just release-prepare
   ```

   This creates a `rail/release-*` branch, commits the generated version and
   changelog, opens a pull request, refreshes the standalone constant-time tool
   lockfiles, and pushes that follow-up commit. It does not tag or publish.

3. Wait for the release pull request's required `Complete` check. Review the
   version, changelog, and lockfile diff, then merge it in the GitHub UI.

4. Record the exact merged release candidate:

   ```bash
   git switch main
   git pull --ff-only
   candidate=$(git rev-parse HEAD)
   ```

5. Wait for `main` CI to pass, then dispatch the expensive release evidence on
   that commit. Do this before another pull request merges into `main`.

   ```bash
   gh workflow run weekly.yaml --ref main
   gh workflow run riscv.yaml --ref main -f mode=evidence
   ```

   Confirm that both runs report `$candidate` as their head SHA. If code,
   dependencies, features, build inputs, or test policy change afterward, rerun
   both workflows.

6. After exact-commit CI and both evidence workflows are green, create and push
   the signed tag:

   ```bash
   test "$(git rev-parse HEAD)" = "$candidate"
   just release-tag
   ```

   `release-tag` rechecks live repository controls, exact-commit CI, and release
   evidence before allowing the tag. It never publishes to crates.io locally.

7. The tag starts the `Release` workflow. Approve its `crates-io` environment
   job after the prerequisite jobs pass. CI publishes and verifies the immutable
   GitHub Release before publishing the same crate through crates.io Trusted
   Publishing.

8. Run the commands in [Verification](#verification).

## Why each gate exists

| Gate | What it prevents |
|---|---|
| Release pull request | An unreviewed version or changelog mutation reaching protected `main`. |
| `main` CI | Tagging a commit that was never proven in its final protected-branch identity. |
| Weekly and RISC-V evidence | Publishing cryptographic claims without the required platform and timing evidence. |
| Signed immutable tag | Moving a released version to different source later. |
| Immutable, attested GitHub Release | Publishing artifacts that cannot be tied back to the tag and build. |
| Environment approval | A tag or compromised workflow publishing to crates.io without a final human decision. |
| Trusted Publishing | Long-lived crates.io credentials becoming a repository secret. |

The repeated pull-request and `main` checks are intentional. The first answers
"may this change merge?" The second proves "this exact protected-branch commit
may become a release." The expensive Weekly and RISC-V evidence is not repeated
on every pull request; it runs only for a release candidate.

## One-time setup

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

1. Activate the `protect-main` branch ruleset described by
   [`.github/rulesets/protect-main.json`](../.github/rulesets/protect-main.json),
   with no bypass actors.
2. Activate the `protect-release-tags` tag ruleset described by
   [`.github/rulesets/protect-release-tags.json`](../.github/rulesets/protect-release-tags.json),
   with no bypass actors. It permits a new `v*` tag but prevents updating or
   deleting an existing one.
3. In **Settings → General → Releases**, enable release immutability as described
   by [`.github/repository-settings/release-immutability.json`](../.github/repository-settings/release-immutability.json).
   It applies only to releases published after the setting is enabled.
4. Create an environment named `crates-io` and add the current maintainer as its
   required reviewer. Permit self-review while the project has one maintainer,
   but disable administrator bypass. Require independent approval after a second
   trusted maintainer exists.
5. Keep long-lived crates.io publish tokens out of repository secrets.

The environment name must match crates.io and
[`.github/workflows/release.yaml`](../.github/workflows/release.yaml), or the
OIDC token exchange will fail.

## Release intent

Commit a cargo-rail change file with each user-visible change when possible.
Change files live in `.changes/`; their reviewed bodies become the changelog.
Commit subjects remain engineering history, not release notes.

```bash
just release-change patch "Describe the user-visible result."
just release-status
```

Use `minor` or `major` when compatibility requires it. Before preparing a
release, `just release-check` validates configuration, dependency unification,
pending intent, SemVer advice, and the generated release plan.

Pull-request CI uses cargo-rail's planner to select checks from the actual
changed surfaces. Pushes to `main`, Weekly, and release preflight run the full
Cargo graph proof because those commits can become release inputs.

`release-prepare` consumes the change files. After its pull request merges,
`release-tag` deliberately does not rerun the pending-intent check. Instead, it
validates the materialized release and uses `cargo rail release finalize
--skip-publish` to create and push the signed tag.

To inspect live repository controls without starting a release:

```bash
just check-repository-controls
```

This is the only routine local check that reads live GitHub settings. It writes
the captured JSON to `target/repository-controls.json`; normal checks and
pre-push validation remain offline.

## What the tag workflow proves

Pushing a `vX.Y.Z` tag starts the `Release` workflow. Before crates.io can
receive anything, the workflow:

1. Verifies the annotated SSH signature, tag target, crate version, and
   changelog version.
2. Revalidates configuration, the unified Cargo graph, dependency policy,
   audit results, SemVer, and successful CI for the exact commit.
3. Requires complete Weekly CT/RSA and RISC-V native/CT evidence from that
   commit or a mechanically proven release-tooling-only ancestor.
4. Builds the `.crate` once, reproduces the source archive from the tag, and
   rejects dirty, private, local-only, or mismatched package contents.
5. Captures repository controls and writes provenance attestations, an identity
   manifest, and `SHA256SUMS` for the artifacts and evidence.
6. Publishes and verifies the immutable GitHub Release, obtains a temporary
   crates.io token through OIDC, publishes the same crate, then downloads it
   from crates.io and verifies its SHA-256.

A release-tooling-only repair may reuse the newest successful ancestor with
paired Weekly and RISC-V evidence. The checker permits only changelog,
release/CT tooling, root-package version, and normalized local CT lockfile
version changes. Runtime, dependency, feature, build, or test changes invalidate
that exception.

## Recovery

Re-running the workflow is safe after a partial failure. If crates.io already has the version, the workflow downloads
it and compares SHA-256 before touching the GitHub Release. A draft release can be repaired and then published. A
published immutable release is never overwritten; the workflow verifies its release attestation and stable crate and
source assets before publishing to crates.io. A mismatch is a hard stop.

If the signed-tag key changes, update `.github/allowed-signers` in a reviewed
commit before creating the next release tag.

## Verification

Consumers can verify the GitHub Release artifact:

```bash
gh release download vX.Y.Z --repo loadingalias/rscrypto \
  -p 'rscrypto-X.Y.Z.crate' \
  -p 'rscrypto-X.Y.Z-source.tar.gz' \
  -p 'rscrypto-X.Y.Z-ct-evidence.tar.gz' \
  -p 'rscrypto-X.Y.Z-repository-controls.json' \
  -p 'rscrypto-X.Y.Z-release-manifest.json' \
  -p SHA256SUMS
sha256sum --check SHA256SUMS
gh release verify vX.Y.Z --repo loadingalias/rscrypto
gh release verify-asset vX.Y.Z rscrypto-X.Y.Z.crate --repo loadingalias/rscrypto
gh release verify-asset vX.Y.Z rscrypto-X.Y.Z-source.tar.gz --repo loadingalias/rscrypto
gh attestation verify rscrypto-X.Y.Z.crate --repo loadingalias/rscrypto
gh attestation verify rscrypto-X.Y.Z-source.tar.gz --repo loadingalias/rscrypto
gh attestation verify rscrypto-X.Y.Z-ct-evidence.tar.gz --repo loadingalias/rscrypto
gh attestation verify rscrypto-X.Y.Z-repository-controls.json --repo loadingalias/rscrypto
gh attestation verify rscrypto-X.Y.Z-release-manifest.json --repo loadingalias/rscrypto
gh attestation verify SHA256SUMS --repo loadingalias/rscrypto
mkdir ct-evidence && tar -xzf rscrypto-X.Y.Z-ct-evidence.tar.gz -C ct-evidence
(cd ct-evidence && sha256sum --check CT-EVIDENCE-MANIFEST.txt)
```

The crate downloaded from crates.io should have the same SHA-256 as the attested release artifact. The release
identity manifest is the machine-readable join between the release's source, artifacts, evidence, and toolchain. The
repository controls JSON records the expected policies, immutable-release setting, live branch and tag rulesets,
effective rules on the default branch, capture time, and release commit. Its validation fields state whether each
bypass list and the immutable-release setting were visible to the capturing token. The JSON is evidence of the
release-time configuration, not a claim that GitHub settings cannot change later.
