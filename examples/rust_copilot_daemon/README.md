# Rust Copilot Daemon (Example)

Agentic co-pilot daemon dedicated to Rust development.

This example shows a **single Rust process** with two planes:

- JSON-RPC over stdio (sync/ingestion plane, for VS Code extension sidecar events)
- MCP-style HTTP tool surface on localhost (query/interaction plane for agent/chat)

## Stack

- AutoAgents framework
- Ollama
- Qdrant
- Docker
- rust-analyzer + VSCode integration target

## Models

- LLM: `gpt-oss:20b`
- Embeddings: `dengcao/Qwen3-Embedding-8B:Q4_K_M`

## Run infra (Docker)

```bash
docker compose -f examples/rust_copilot_daemon/docker-compose.yml up -d
```

Then pull models in Ollama:

```bash
ollama pull gpt-oss:20b
ollama pull dengcao/Qwen3-Embedding-8B:Q4_K_M
```

## Run daemon

```bash
cargo run -p rust-copilot-daemon
```

To pin MCP to a fixed localhost port:

```bash
MCP_PORT=43891 cargo run -p rust-copilot-daemon
```

At startup it prints MCP endpoint, for example:

```text
rust-copilot-daemon started: jsonrpc=stdio mcp=http://127.0.0.1:43891 schema_version=2026-02-10
```

## JSON-RPC methods (stdio)

Input format: newline-delimited JSON-RPC 2.0 requests.

- `initialize({ workspaceRoot, qdrantUrl, collection, relationCollection?, metadataCollection?, fileCollection?, callEdgeCollection?, typeEdgeCollection?, diagnosticCollection?, config })`
- `scan.full({ globs, exclude, respectGitignore })`
- `file.changed({ path, version?, reason? })`
- `file.deleted({ path })`
- `workspace.renamed({ oldPath, newPath })`
- `status()`
- `shutdown()`

Example:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"workspaceRoot":"/path/to/repo"}}
```

## MCP-style tool routes (localhost HTTP)

- `GET /mcp/tools`
- `GET /mcp/tools/index_status`
- `GET /mcp/tools/workspace_metadata`
- `POST /mcp/tools/search_code`
- `POST /mcp/tools/search_relations`
- `POST /mcp/tools/search_files`
- `POST /mcp/tools/search_calls`
- `POST /mcp/tools/search_types`
- `POST /mcp/tools/search_diagnostics`
- `POST /mcp/tools/get_file_chunks`
- `POST /mcp/tools/get_file_context`
- `POST /mcp/tools/get_symbol_context`
- `POST /mcp/tools/get_symbol_relations`
- `POST /mcp/tools/explain_relevance`

Example:

```bash
curl -s -X POST http://127.0.0.1:43891/mcp/tools/search_code \
  -H 'content-type: application/json' \
  -d '{"query":"trait ToolT","top_k":5}'
```

## Architecture notes

- Ingestion is queue-based, with debounce + batch processing for file-change storms.
- Query plane is read-only and returns `indexing_in_progress`.
- Search now supports multi-vector semantic fusion by default. If `vector_name` is omitted, the daemon queries multiple named vectors, applies weighted score fusion, and merges by stable id. If `vector_name` is provided, it uses that single channel only (backward-compatible behavior).
- Default fusion channels:
  - `search_code`: `symbol` (0.45), `docs` (0.25), `signature` (0.15), `type` (0.15), plus lexical/semantic hybrid merge.
  - `search_relations`: `symbol` (0.30), `docs` (0.20), `signature` (0.15), `body` (0.10), `type` (0.10), `graph` (0.15), then intent-aware reranking.
  - `search_files`: `symbol` (0.55), `docs` (0.45).
  - `search_calls`: `symbol` (0.35), `docs` (0.30), `graph` (0.35).
  - `search_types`: `symbol` (0.25), `docs` (0.25), `type` (0.30), `graph` (0.20).
  - `search_diagnostics`: `symbol` (0.50), `docs` (0.50).
- `explain_relevance` uses the same fused semantic retrieval path (`symbol/docs/signature/type`) before evidence extraction.
- `search_relations` decouples candidate pool size from user `top_k`: it always pulls a larger semantic candidate set (`max(48, top_k*4)` capped at `128`) before reranking and returning `top_k`.
- MCP responses include `semantic_vector_names` when fused mode is used, and keep `semantic_vector_name` for compatibility.
- `search_relations` uses semantic retrieval plus intent-aware reranking: query terms like `implements`, `defined`, `references`, and `type definition` boost matching relation kinds.
- MCP responses now emit canonical schema payloads (`SymbolDoc` / typed graph edge docs).
- Canonical indexing schema includes `SymbolDoc`, `FileDoc`, typed graph edges, and `DiagnosticDoc`.
- JSON-RPC returns `{protocol_version, daemon_version}` on initialize.
- MCP responses include `schema_version`.
- HTTP binds only to `127.0.0.1`.
- MCP bind port comes from `MCP_PORT` (default `0`, meaning OS-assigned ephemeral port).
- rust-analyzer integration is session-based (persistent process per workspace) with `didOpen`/`didChange`/`didClose`.
- Symbol extraction is enriched with LSP `documentSymbol`, `hover`, and best-effort `workspace/symbol` path hints before vector upsert.
- Cross-file relations are extracted via LSP `textDocument/references` and `textDocument/implementation` and persisted in a dedicated relation collection.
- Additional relation edges are extracted via `textDocument/definition` and `textDocument/typeDefinition`.
- Workspace Cargo metadata (crate name, edition, features, optional deps) is indexed and exposed via MCP.

## Current limitation

Graph neighborhood expansion and edge-traversal ranking are not yet exposed as first-class query controls.

## Retrieval Eval Contract

Retrieval quality goals and fixture contract are versioned in:

- `docs/rust_copilot/retrieval_goals.md`
- `examples/rust_copilot_daemon/eval/query_taxonomy.json`
- `examples/rust_copilot_daemon/eval/goldens/rust_copilot_metrics_fixture.contract.json`

Offline scorer (CI/local):

```bash
cargo test -p rust-copilot-daemon --test eval_contract -- --nocapture
```

The scorer compares an observed fixture snapshot against contract gates:

- extraction metrics
- index cardinality (`total_files`, `total_chunks`, `workspace_crates`)
- Qdrant collection point counts
- retrieval intent assertions (for example relation query intent)

By default it uses:

- `examples/rust_copilot_daemon/eval/observed/rust_copilot_metrics_fixture.sample.json`

Override with a fresh captured snapshot:

```bash
RUST_COPILOT_EVAL_OBSERVED_JSON=/path/to/observed.json \
cargo test -p rust-copilot-daemon --test eval_contract -- --nocapture
```
