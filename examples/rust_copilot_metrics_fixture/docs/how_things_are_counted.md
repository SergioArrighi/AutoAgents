## Manual Static Source Scan: Counting Rules

This document records the manual static scan counts for `examples/rust_copilot_metrics_fixture`.

- Eligibility filter (same semantic set): `{5, 10, 11, 12, 23}`
- `[E]` = eligible for relation requests
- `[N]` = not eligible

## Symbols

- `[N]` module `engine` - `lib.rs` (line 1)
- `[N]` module `types` - `lib.rs` (line 2)
- `[E]` trait `Rule` - `lib.rs` (line 4)
- `[N]` trait method `allows` (declaration) - `lib.rs` (line 5)
- `[E]` function `evaluate` - `lib.rs` (line 8)
- `[E]` function `evaluate_pair` - `lib.rs` (line 12)
- `[E]` struct `PositiveRule` - `types.rs` (line 1)
- `[N]` impl block `impl crate::Rule for PositiveRule` - `types.rs` (line 3)
- `[N]` method `allows` (impl) - `types.rs` (line 4)
- `[E]` enum `ValueState` - `types.rs` (line 9)
- `[N]` enum variant `Positive` - `types.rs` (line 10)
- `[N]` enum variant `Zero` - `types.rs` (line 11)
- `[N]` enum variant `Negative` - `types.rs` (line 12)
- `[E]` function `classify` - `types.rs` (line 15)
- `[E]` struct `Engine` - `engine.rs` (line 4)
- `[N]` impl block `impl Engine` - `engine.rs` (line 6)
- `[E]` associated function `run` - `engine.rs` (line 7)
- `[E]` function `run_default` - `engine.rs` (line 12)
- `[E]` test function `run_default_returns_zero_for_mixed_values` - `smoke.rs` (line 5)

## References

- `Rule` (`lib.rs`, line 4)
- request_generated: `yes`
- non_empty: `yes`
- refs: `lib.rs` (line 8), `lib.rs` (line 12), `types.rs` (line 3), `engine.rs` (line 2), `engine.rs` (line 7)

- `evaluate` (`lib.rs`, line 8)
- request_generated: `yes`
- non_empty: `yes`
- refs: `lib.rs` (line 13), 2 occurrences

- `evaluate_pair` (`lib.rs`, line 12)
- request_generated: `yes`
- non_empty: `yes`
- refs: `engine.rs` (line 2), `engine.rs` (line 8)

- `PositiveRule` (`types.rs`, line 1)
- request_generated: `yes`
- non_empty: `yes`
- refs: `types.rs` (line 3), `engine.rs` (line 1), `engine.rs` (line 13)

- `ValueState` (`types.rs`, line 9)
- request_generated: `yes`
- non_empty: `yes`
- refs: `types.rs` (line 15), `types.rs` (line 17), `types.rs` (line 19), `types.rs` (line 21), `engine.rs` (line 1), `engine.rs` (line 12), `engine.rs` (line 18), `smoke.rs` (line 2), `smoke.rs` (line 6)

- `classify` (`types.rs`, line 15)
- request_generated: `yes`
- non_empty: `yes`
- refs: `engine.rs` (line 1), `engine.rs` (line 16)

- `Engine` (`engine.rs`, line 4)
- request_generated: `yes`
- non_empty: `yes`
- refs: `engine.rs` (line 14)

- `run` (`engine.rs`, line 7)
- request_generated: `yes`
- non_empty: `yes`
- refs: `engine.rs` (line 14)

- `run_default` (`engine.rs`, line 12)
- request_generated: `yes`
- non_empty: `yes`
- refs: `smoke.rs` (line 1), `smoke.rs` (line 6)

- `run_default_returns_zero_for_mixed_values` (`smoke.rs`, line 5)
- request_generated: `yes`
- non_empty: `no`
- refs: none

Totals:

- `references_requests_total = 10`
- `references_nonempty_total = 9`
- `references_locations_total = 28`

## Implementations

Rules used:

- Trait symbol -> `impl Trait for Type` blocks
- Type symbol -> inherent `impl Type` blocks
- Do not double-count `impl Trait for Type` under the concrete type symbol

Results:

- `Rule` (`lib.rs`, line 4)
- request_generated: `yes`
- non_empty: `yes`
- implementation locations: `types.rs` (line 3), `impl crate::Rule for PositiveRule`

- `Engine` (`engine.rs`, line 4)
- request_generated: `yes`
- non_empty: `yes`
- implementation locations: `engine.rs` (line 6), `impl Engine`

- `evaluate` (`lib.rs`, line 8)
- request_generated: `yes`
- non_empty: `no`

- `evaluate_pair` (`lib.rs`, line 12)
- request_generated: `yes`
- non_empty: `no`

- `PositiveRule` (`types.rs`, line 1)
- request_generated: `yes`
- non_empty: `no`

- `ValueState` (`types.rs`, line 9)
- request_generated: `yes`
- non_empty: `no`

- `classify` (`types.rs`, line 15)
- request_generated: `yes`
- non_empty: `no`

- `run` (`engine.rs`, line 7)
- request_generated: `yes`
- non_empty: `no`

- `run_default` (`engine.rs`, line 12)
- request_generated: `yes`
- non_empty: `no`

- `run_default_returns_zero_for_mixed_values` (`smoke.rs`, line 5)
- request_generated: `yes`
- non_empty: `no`

Totals:

- `implementations_requests_total = 10`
- `implementations_nonempty_total = 2`
- `implementations_locations_total = 2`

## Definitions

- `Rule` (`lib.rs`, line 4)
- request_generated: `yes`
- non_empty: `yes`
- definition: `lib.rs` (line 4)

- `evaluate` (`lib.rs`, line 8)
- request_generated: `yes`
- non_empty: `yes`
- definition: `lib.rs` (line 8)

- `evaluate_pair` (`lib.rs`, line 12)
- request_generated: `yes`
- non_empty: `yes`
- definition: `lib.rs` (line 12)

- `PositiveRule` (`types.rs`, line 1)
- request_generated: `yes`
- non_empty: `yes`
- definition: `types.rs` (line 1)

- `ValueState` (`types.rs`, line 9)
- request_generated: `yes`
- non_empty: `yes`
- definition: `types.rs` (line 9)

- `classify` (`types.rs`, line 15)
- request_generated: `yes`
- non_empty: `yes`
- definition: `types.rs` (line 15)

- `Engine` (`engine.rs`, line 4)
- request_generated: `yes`
- non_empty: `yes`
- definition: `engine.rs` (line 4)

- `run` (`engine.rs`, line 7)
- request_generated: `yes`
- non_empty: `yes`
- definition: `engine.rs` (line 7)

- `run_default` (`engine.rs`, line 12)
- request_generated: `yes`
- non_empty: `no`

- `run_default_returns_zero_for_mixed_values` (`smoke.rs`, line 5)
- request_generated: `yes`
- non_empty: `no`

Totals:

- `definitions_requests_total = 10`
- `definitions_nonempty_total = 8`
- `definitions_locations_total = 8`

## Type Definitions

- `Rule` (`lib.rs`, line 4)
- request_generated: `yes`
- non_empty: `yes`
- type_definition: `lib.rs` (line 4)

- `PositiveRule` (`types.rs`, line 1)
- request_generated: `yes`
- non_empty: `yes`
- type_definition: `types.rs` (line 1)

- `ValueState` (`types.rs`, line 9)
- request_generated: `yes`
- non_empty: `yes`
- type_definition: `types.rs` (line 9)

- `Engine` (`engine.rs`, line 4)
- request_generated: `yes`
- non_empty: `yes`
- type_definition: `engine.rs` (line 4)

- `evaluate` (`lib.rs`, line 8)
- request_generated: `yes`
- non_empty: `no`

- `evaluate_pair` (`lib.rs`, line 12)
- request_generated: `yes`
- non_empty: `no`

- `classify` (`types.rs`, line 15)
- request_generated: `yes`
- non_empty: `no`

- `run` (`engine.rs`, line 7)
- request_generated: `yes`
- non_empty: `no`

- `run_default` (`engine.rs`, line 12)
- request_generated: `yes`
- non_empty: `no`

- `run_default_returns_zero_for_mixed_values` (`smoke.rs`, line 5)
- request_generated: `yes`
- non_empty: `no`

Totals:

- `type_definitions_requests_total = 10`
- `type_definitions_nonempty_total = 4`
- `type_definitions_locations_total = 4`

## Per-file aggregate (manual static source scan)

| File | document_symbols | eligible_symbols | references_requests | references_nonempty | references_locations | implementations_requests | implementations_nonempty | implementations_locations | definitions_requests | definitions_nonempty | definitions_locations | type_definitions_requests | type_definitions_nonempty | type_definitions_locations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `examples/rust_copilot_metrics_fixture/src/lib.rs` | 6 | 3 | 3 | 3 | 9 | 3 | 1 | 1 | 3 | 3 | 3 | 3 | 1 | 1 |
| `examples/rust_copilot_metrics_fixture/src/types.rs` | 8 | 3 | 3 | 3 | 14 | 3 | 0 | 0 | 3 | 3 | 3 | 3 | 2 | 2 |
| `examples/rust_copilot_metrics_fixture/src/engine.rs` | 4 | 3 | 3 | 3 | 4 | 3 | 1 | 1 | 3 | 2 | 2 | 3 | 1 | 1 |
| `examples/rust_copilot_metrics_fixture/tests/smoke.rs` | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 |