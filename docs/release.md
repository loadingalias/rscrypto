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
   cargo rail change status
   cargo rail release check rscrypto --extended
   ```

2. Prepare the release:

   ```bash
   just release-prepare
   ```

   This creates a `rail/release-*` branch, commits the generated version and
   changelog, opens a pull request, refreshes the standalone constant-time tool
   lockfiles, and pushes that follow-up commit. It does not tag or publish.
   The adapter is required because Cargo Rail does not yet include auxiliary
   workspace lockfiles in its release mutation. Running `cargo rail release
run rscrypto --bump auto --yes --pr` directly would leave the CT workspaces
   stale under `--locked`.

3. Wait for the release pull request's required `Complete` check. Review the
   version, changelog, and lockfile diff, then merge it in the GitHub UI.

4. Record the exact merged release candidate:

   ```bash
   git switch main
   git pull --ff-only
   candidate=$(git rev-parse HEAD)
   ```

5. Dispatch the expensive release evidence on that commit. Do this before
   another pull request merges into `main`. Release mode reruns the complete CI
   suite, including compiler-backed Cargo graph assurance, and retains raw CT
   evidence for 90 days.

   ```bash
   gh workflow run weekly.yaml --ref main -f mode=release
   gh workflow run riscv.yaml --ref main -f mode=evidence
   ```

   Confirm that both runs report `$candidate` as their head SHA. If code,
   dependencies, features, build inputs, or test policy change afterward, rerun
   both workflows. Do not substitute a scheduled Weekly or RISC-V run, or an
   assurance-mode Weekly dispatch: those produce compact reports with 14-day
   retention and cannot satisfy the release evidence gate.

6. After both exact-commit evidence workflows are green, create and push the
   signed tag:

   ```bash
   test "$(git rev-parse HEAD)" = "$candidate"
   just release-tag
   ```

   `release-tag` rechecks live repository controls and exact-commit release
   evidence before allowing the tag. It never publishes to crates.io locally.

7. The tag starts the `Release` workflow. Approve its `crates-io` environment
   job after the prerequisite jobs pass. CI publishes and verifies the immutable
   GitHub Release before publishing the same crate through crates.io Trusted
   Publishing.

8. Run the commands in [Verification](#verification).

## Why each gate exists

| Gate                               | What it prevents                                                                                                                                    |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Release pull request               | An unreviewed version or changelog mutation reaching protected `main`.                                                                              |
| Exact-commit Weekly release mode   | Tagging a candidate without an explicitly requested full suite, compiler-backed Cargo graph assurance, raw CT artifacts, and complete CT/RSA gates. |
| Weekly and RISC-V evidence         | Publishing cryptographic claims without the required platform and timing evidence.                                                                  |
| Signed immutable tag               | Moving a released version to different source later.                                                                                                |
| Immutable, attested GitHub Release | Publishing artifacts that cannot be tied back to the tag and build.                                                                                 |
| Environment approval               | A tag or compromised workflow publishing to crates.io without a final human decision.                                                               |
| Trusted Publishing                 | Long-lived crates.io credentials becoming a repository secret.                                                                                      |

Pull-request CI answers "may this change merge?" once. Scheduled Weekly assurance
keeps routine safety coverage current with compact, short-lived reports. Only a
manually dispatched exact-commit Weekly release run and RISC-V evidence answer
"may this protected-branch commit become a release?"

## One-time setup

Configure the crate on crates.io:

| Field             | Value          |
| ----------------- | -------------- |
| Repository owner  | `loadingalias` |
| Repository name   | `rscrypto`     |
| Workflow filename | `release.yaml` |
| Environment       | `crates-io`    |

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
cargo rail change add rscrypto --bump patch --message "Describe the user-visible result."
cargo rail change status
```

Use `minor` or `major` when compatibility requires it. Before preparing a
release, `cargo rail release check rscrypto --extended` validates the pending
release and its SemVer contract.

Pull-request CI uses cargo-rail's planner to select checks from the actual
changed surfaces. Weekly release mode runs the full Cargo graph proof for an
exact release candidate; scheduled assurance does not. Release preflight
consumes the release-mode result instead of recompiling it.

`release-prepare` delegates the version, changelog, branch, commit, and pull
request to Cargo Rail, then synchronizes the three standalone CT lockfiles.
After that pull request merges, `release-tag` deliberately does not rerun the
consumed pending-intent check. It proves live repository controls and
exact-commit release evidence before using `cargo rail release finalize
--skip-publish` to create and push the signed tag.

To inspect live repository controls without starting a release:

```bash
scripts/ci/repository-controls-evidence.sh \
  --commit "$(git rev-parse HEAD)" \
  --output target/repository-controls.json
```

This is the only routine local check that reads live GitHub settings. It writes
the captured JSON to `target/repository-controls.json`; normal checks and
pre-push validation remain offline.

## What the tag workflow verifies

Pushing a `vX.Y.Z` tag starts the `Release` workflow. Before crates.io can
receive anything, the workflow:

1. Verifies the annotated SSH signature, tag target, crate version, and
   changelog version.
2. Revalidates configuration, dependency policy, audit results, SemVer, and the
   exact-commit Weekly release-mode Cargo graph result.
3. Requires the Weekly release gate, live raw CT artifacts, complete Weekly
   CT/RSA, and manually dispatched RISC-V native/CT evidence from that exact
   commit and crate version.
4. Builds the `.crate` once, reproduces the source archive from the tag, and
   rejects dirty, private, local-only, or mismatched package contents.
5. Captures repository controls and writes provenance attestations, an identity
   manifest, and `SHA256SUMS` for the artifacts and evidence.
6. Publishes and verifies the immutable GitHub Release, obtains a temporary
   crates.io token through OIDC, publishes the same crate, then downloads it
   from crates.io and verifies its SHA-256.

Any change to the candidate after the evidence run—including a version-only,
dependency, build-input, or test-policy change—creates a new release candidate
and requires fresh paired Weekly and RISC-V evidence. Ancestor binaries are
never promoted into an exact-commit constant-time claim.

## Recovery

Rerun a transient or partial failure on the same tag and commit:

```bash
gh run rerun RUN_ID --failed
```

If the committed workflow or one of its pinned tools cannot complete, merge the
smallest repair through the required `Complete` check, then dispatch the
reviewed recovery path from `main`:

```bash
gh workflow run release.yaml --ref main -f tag=vX.Y.Z
```

If release packaging rejects only the s390x CT artifact, regenerate that lane
against the existing tag before dispatching recovery:

```bash
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

Run both dispatches from the same reviewed `main` commit. The CT recovery is
limited to the complete native s390x lane and checks out the immutable tag;
the release preflight rejects any replacement run from another workflow,
branch, repository, or commit. The normal release evidence packager then
validates the replacement artifact's tag commit, crate version, cases, hashes,
and target provenance before publication.

Recovery checks out the existing annotated tag, verifies its allowed signature,
and binds the package, evidence, release manifest, and release notes to the
tag's commit rather than the newer workflow commit. Confirm that Preflight
reports the intended tag commit before approving the `crates-io` environment.
The recovery path cannot run from an unprotected branch.

If crates.io already contains the version, the workflow downloads it and
compares its SHA-256 before touching the GitHub Release. The workflow can repair
and publish a draft release. It never overwrites a published immutable release;
it verifies the release attestation and stable crate and source assets before
publishing to crates.io. Any mismatch stops the release.

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
ct_evidence_dir=$(mktemp -d)
tar -xzf rscrypto-X.Y.Z-ct-evidence.tar.gz -C "$ct_evidence_dir"
(cd "$ct_evidence_dir" && sha256sum --check CT-EVIDENCE-MANIFEST.txt)
```

The crate downloaded from crates.io must have the same SHA-256 as the attested
release artifact. The release identity manifest joins the release source,
artifacts, evidence, and toolchain. The repository-controls JSON records the
expected policies, immutable-release setting, live branch and tag rulesets,
effective default-branch rules, capture time, and release commit. Its
validation fields state whether the capturing token could inspect each bypass
list and the immutable-release setting. The JSON records release-time
configuration; GitHub settings can change afterward.
