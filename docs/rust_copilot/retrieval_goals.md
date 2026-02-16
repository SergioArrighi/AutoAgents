# Rust Copilot Retrieval Goals

## Task Classes

- symbol_lookup: Find exact or near-exact symbol definitions quickly.
- api_usage: Find idiomatic usage examples and signatures.
- refactor_support: Find related definitions/usages across files.
- diagnostics_fixing: Find relevant symbols and relation context for fixes.
- test_authoring: Find existing tests/fixtures and target API context.

## Targets

- Recall@5
  - symbol_lookup: >= 0.92
  - api_usage: >= 0.88
  - refactor_support: >= 0.85
  - diagnostics_fixing: >= 0.82
  - test_authoring: >= 0.80
- P95 query latency (daemon-side retrieval only): <= 350ms at top_k=8.
- Index freshness SLA: <= 3s from file change event to searchable update.

## Indexing Contract

- Primary unit: symbol-level documents.
- Secondary unit: file-context windows.
- Graph unit: typed relation edges.
- Metadata unit: crate/workspace docs.

## Eval Inputs

- Taxonomy source: `examples/rust_copilot_daemon/eval/query_taxonomy.json`
- Goldens source: `examples/rust_copilot_daemon/eval/goldens/*.json`

## Reporting

- Report Recall@k by task class.
- Report median/p95 latency by query class.
- Report stale index age for each eval run.
