# Rust Copilot Metrics Fixture

Deterministic fixture workspace for `examples/rust_copilot_daemon` extraction metrics.

This fixture includes real Rust symbols and cross-file references:
- trait + impl
- struct + enum
- top-level functions with symbol usage across modules

## Source layout

- `src/lib.rs`: shared trait (`Rule`) and evaluation helpers.
- `src/types.rs`: rule implementation (`PositiveRule`) and value classification.
- `src/engine.rs`: fixture orchestration (`Engine`, `run_default`).
- `tests/smoke.rs`: minimal behavioral assertion used as a stability check.

Expected metrics are in:
- `examples/rust_copilot_daemon/eval/fixtures/rust_copilot_metrics_fixture.expected.json`
- Rust-analyzer source baseline: `examples/rust_copilot_daemon/eval/fixtures/rust_copilot_metrics_fixture.rust_analyzer_baseline.json`

## Local verification

Run fixture tests:

`cargo test -p rust-copilot-metrics-fixture`

## Validation flow

1. Start daemon.
2. `initialize` with `workspaceRoot=/.../examples/rust_copilot_metrics_fixture`.
3. Send one `scan.full` (default globs).
4. Poll `status` until `indexing_in_progress=false` and `queue_depth=0`.
5. Compare `status.extraction_metrics` with the expected JSON.

Use a fresh initialize/session because metrics are cumulative.

## Regenerate rust-analyzer baseline

Run:

`examples/rust_copilot_daemon/eval/scripts/collect_rust_analyzer_baseline.py`

This queries rust-analyzer directly over LSP and writes:

`examples/rust_copilot_daemon/eval/fixtures/rust_copilot_metrics_fixture.rust_analyzer_baseline.json`
