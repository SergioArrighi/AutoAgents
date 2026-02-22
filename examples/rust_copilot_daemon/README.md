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

- `initialize({ workspaceRoot, qdrantUrl, collection, config })`
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
- `POST /mcp/tools/get_file_chunks`
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
- `search_relations` uses semantic retrieval plus intent-aware reranking: query terms like `implements`, `defined`, `references`, and `type definition` boost matching relation kinds.
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

Type-level graph edges (`typeDefinition`, call hierarchy, trait bounds graph) are not yet persisted as dedicated relation documents.

## Deterministic Metrics Fixture

Use `examples/rust_copilot_metrics_fixture` to validate extraction behavior against a known baseline with real symbols and non-zero relation output.

Expected metrics:

- `examples/rust_copilot_daemon/eval/fixtures/rust_copilot_metrics_fixture.expected.json`

Validation flow:

1. `initialize` with `workspaceRoot` set to `examples/rust_copilot_metrics_fixture`.
2. Send one `scan.full` with default globs.
3. Wait for `status.indexing_in_progress=false` and `status.queue_depth=0`.
4. Compare `status.extraction_metrics` to the expected JSON.

Run against a fresh initialize/session because extraction metrics are cumulative.
