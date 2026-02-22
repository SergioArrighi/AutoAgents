# Rust Copilot Retrieval Goals

This document defines the retrieval quality contract for
`examples/rust_copilot_daemon` on the deterministic fixture project
`examples/rust_copilot_metrics_fixture`.

## Scope

- Workspace: `examples/rust_copilot_metrics_fixture`
- Query taxonomy: `examples/rust_copilot_daemon/eval/query_taxonomy.json`
- Golden contract: `examples/rust_copilot_daemon/eval/goldens/rust_copilot_metrics_fixture.contract.json`
- Default observed snapshot: `examples/rust_copilot_daemon/eval/observed/rust_copilot_metrics_fixture.sample.json`

## Target Tasks

1. Behavior lookup: answer "what does this function do?" from indexed symbols/tests.
2. Symbol navigation: resolve symbol context and location quickly.
3. Relation lookup: return relation-kind-consistent results for intentful relation queries.
4. Metadata lookup: return crate/workspace metadata needed for code-agent planning.

## Quality Targets

1. Relation intent accuracy: for query patterns containing explicit relation intent terms
   (`implements`, `defined`, `references`, `type definition`), top result should match
   the intended relation kind in fixture eval queries.
2. Extraction reliability:
   - All `*_failed_total` counters are `0`.
   - Non-empty relation counters are positive (`references`, `implementations`,
     `definitions`, `type_definitions`).
3. Index completeness:
   - `total_files`, `total_chunks`, and `workspace_crates` match the fixture contract.
   - Qdrant point counts for `rust_copilot_symbols`, `rust_copilot_relations`,
     `rust_copilot_metadata`, `rust_copilot_files`, `rust_copilot_calls`,
     `rust_copilot_types`, and `rust_copilot_diagnostics` match the fixture contract.

## Latency Budgets (Local Dev)

Budgets are measured from MCP endpoints on localhost with warm services:

1. `search_code` p95 <= 400 ms
2. `search_relations` p95 <= 350 ms
3. `get_symbol_context` p95 <= 250 ms

Latency budgets are currently policy targets; they are not hard-gated by the offline scorer.

## Freshness SLA

After `scan.full`:

1. `indexing_in_progress=false`
2. `queue_depth=0`
3. `indexed_at_unix_ms` updated

These are required before collecting an observed snapshot for scoring.

## Evaluation Command

```bash
cargo test -p rust-copilot-daemon --test eval_contract -- --nocapture
```

Use `RUST_COPILOT_EVAL_OBSERVED_JSON` to score a custom observed snapshot.
