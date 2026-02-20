# Tuning

`rscrypto` tuning is intentionally narrow and cheap.

## Scope

- Supported tuning path: BLAKE3 boundary capture.
- Goal: identify practical kernel crossover boundaries on real hardware.
- Non-goal: automatic dispatch-table mutation from tuning runs.
- Checksums and other hashes: no active tuning pipeline. Use benchmark workflows instead.

## Local Usage

Run the frontdoor command:

```bash
just tune
```

Optional window overrides:

```bash
just tune 120 200
```

## Artifacts

Outputs are written to `tune-results/`:

- `tune-results/tune.log`: full run log
- `tune-results/boundary/*.csv`: raw boundary captures (auto + forced kernels)
- `tune-results/boundary/summary.txt`: best-kernel table + suggested plain dispatch boundaries

## CI Workflow

Manual workflow: `.github/workflows/tune.yaml`

- runs boundary capture only
- supports per-run warmup/measure overrides
- uploads `tune-results/` as artifact

## Benching For Perf Gaps

- Core BLAKE3 matrix (fast, actionable): `just bench-blake3-core`
- Full BLAKE3 diagnostics (slower): `just bench-blake3-diag`

## Contribution Flow

1. Run `just tune` on an idle machine.
2. Inspect `tune-results/boundary/summary.txt`.
3. Submit summary and platform metadata via `tuning-results` issue template.
