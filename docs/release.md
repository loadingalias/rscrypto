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

1. Create the active `protect-main` branch ruleset described by
   [`.github/rulesets/protect-main.json`](../.github/rulesets/protect-main.json), with no bypass actors.
2. Create the active `protect-release-tags` tag ruleset described by
   [`.github/rulesets/protect-release-tags.json`](../.github/rulesets/protect-release-tags.json), with no bypass
   actors. It permits a new `v*` tag but prevents updating or deleting an existing one.
3. In **Settings → General → Releases**, enable release immutability as described by
   [`.github/repository-settings/release-immutability.json`](../.github/repository-settings/release-immutability.json).
   It applies only to releases published after the setting is enabled.
4. Create an environment named `crates-io`.
5. Add the current maintainer as its required reviewer. While the project has one maintainer, permit self-review;
   enable independent review only after a second trusted maintainer exists.
6. Keep long-lived crates.io publish tokens out of repository secrets.

The environment name must match both crates.io and
[`.github/workflows/release.yaml`](../.github/workflows/release.yaml). If it
does not match, `rust-lang/crates-io-auth-action` cannot exchange the GitHub
OIDC token for a temporary crates.io token.

## Release Commands

Add release intent with a cargo-rail change file. Commit the change file with
the code change when possible; if the code was already committed, add the
change file before cutting the release. Change files live in `.changes/`.
Use `--name` to choose the filename slug. The file may be renamed after
creation as long as it remains a `.changes/*.md` file. Changelog sections use
only these reviewed change-file bodies; commit subjects are engineering history,
not release notes.

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

Pull-request CI uses cargo-rail-action v5 with cargo-rail 0.17.3. The action's
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
`weekly.yaml` and the `evidence` mode of `riscv.yaml` on that exact commit.
Weekly owns the non-RISC-V CT matrix and dedicated RSA workflow; RISC-V owns
its native and CT evidence. Both retain the raw CT artifacts needed for the
release bundle. If runtime code, dependencies, features, build inputs, or test
policy changes, both evidence workflows must pass again before tagging.

A release-tooling-only repair may promote the newest successful ancestor with
paired Weekly and RISC-V evidence. The evidence checker permits only changelog,
release/CT tooling, and root-package version changes. It parses and compares
`Cargo.toml` and every CT lockfile after normalizing only the local `rscrypto`
version, and rejects every other changed path. The evidence bundle records both
the release commit and the promoted evidence commit. This exception does not
apply to runtime, dependency, feature, build, or test changes.

Once CI and both evidence workflows are green, finalize the release:

```bash
just release-tag
```

The `release-tag` recipe reruns strict configuration and Cargo-graph validation,
uses the maintainer's GitHub access to check the complete live ruleset—including
the empty bypass list—against the checked-in policy, requires successful push CI
and Cargo Graph Assurance for the exact candidate, and refuses to tag without
complete CT and RSA evidence from either that commit or a proven release-only
ancestor. It then uses `cargo rail release finalize` to validate the materialized
release, create the signed tag, and push it. It does not rerun `cargo rail release
check`: that command validates pending release intent and correctly fails after
`release-prepare` consumes the change files. Finalization must not publish to
crates.io; the `--skip-publish` flag is part of the release contract.

To check GitHub controls without running the release flow, use:

```bash
just check-repository-controls
```

This is the only routine local check that reads live GitHub settings. Normal
checks and pre-push validation remain offline. The command writes the captured
JSON to `target/repository-controls.json` and prints its SHA-256.

## CI Release Gate

Pushing a `vX.Y.Z` tag starts the `Release` workflow. The workflow:

1. Requires an annotated SSH-signed tag trusted by
   [`.github/allowed-signers`](../.github/allowed-signers).
2. Requires the tag target to match the checked-out commit.
3. Requires the tag version, `Cargo.toml` version, and `CHANGELOG.md` heading
   to match.
4. Validates and sync-checks `.config/rail.toml`, then proves the unified Cargo
   graph is clean with cargo-rail 0.17.3.
5. Runs `cargo deny check all`, `cargo audit`, and the single hard SemVer check
   against the finalized release version. Before tagging, cargo-rail's automatic
   release plan uses compatibility analysis to select the required version bump;
   the extended release check reports the advisory verdict.
6. Builds the `.crate` once with `cargo package --locked` and creates a deterministic source archive from the exact
   tag commit, then transfers both validated artifacts to the publish job.
7. Rejects dirty or mismatched VCS metadata, private or local-only package contents, and source archives that cannot
   be reproduced from the tag commit.
8. Waits for the `CI` workflow on the same commit to pass.
9. Verifies both transferred SHA-256 values, then attests the crate and source archive with GitHub build provenance.
10. Compares every rule visible to the restricted workflow token with the
    checked-in default-branch and release-tag policies and attests the resulting repository controls JSON. GitHub
    redacts bypass actors from this token; the artifact labels those fields as redacted instead of pretending the
    workflow reverified the pre-tag gate.
11. Requires successful Weekly CT/RSA evidence and RISC-V native/CT evidence
    from the exact tag commit or the same mechanically proven release-only
    ancestor, then downloads raw CT artifacts from both runs. The tag workflow
    does not rerun either multi-hour suite.
12. Validates the evidence commit's CT lane set, version, clean provenance,
    tool versions, timing coverage, BINSEC coverage, and raw artifact hashes.
    The bundle separately records the release commit and evidence commit.
13. Attests the CT evidence bundle with GitHub build provenance.
14. Writes and attests a release identity manifest binding the tag, commit, tree, pinned and active Rust toolchain,
    Cargo lockfile, release workflow, source archive, crate, CT evidence, and repository controls.
15. Writes and attests `SHA256SUMS` for every published release asset.
16. Creates a draft GitHub Release, attaches every asset, publishes it as an immutable release, and verifies GitHub's
    release attestation before any irreversible registry change. A rerun may repair an unpublished draft, but an
    already-published release is verified instead of modified.
17. Uses crates.io Trusted Publishing to get a temporary publish token.
18. Publishes with `cargo publish -p rscrypto --locked`.
19. Downloads the crate from crates.io and verifies the SHA-256 against the attested package.

The publish job uses the `crates-io` GitHub environment. GitHub requests
approval only after preflight and CI have passed, and before the OIDC token is
issued.

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
