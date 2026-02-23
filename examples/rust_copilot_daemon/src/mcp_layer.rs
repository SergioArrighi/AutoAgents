//! MCP-style query plane.
//!
//! This module exposes localhost HTTP endpoints that map to tool-like operations
//! (`search_code`, `get_file_chunks`, `get_symbol_context`, etc.).

use super::*;

const RELATION_CANDIDATE_POOL_MIN: usize = 48;
const RELATION_CANDIDATE_POOL_MAX: usize = 128;

/// Runs the localhost MCP HTTP server loop until a stop signal is received.
pub(super) async fn run_mcp_server(
    app: App,
    listener: TcpListener,
    mut stop_rx: watch::Receiver<bool>,
) -> Result<()> {
    loop {
        tokio::select! {
            changed = stop_rx.changed() => {
                if changed.is_ok() && *stop_rx.borrow() {
                    break;
                }
            }
            accepted = listener.accept() => {
                let (stream, _) = accepted?;
                let app_clone = app.clone();
                tokio::spawn(async move {
                    let _ = handle_http_connection(app_clone, stream).await;
                });
            }
        }
    }

    Ok(())
}

/// Reads one HTTP request and writes one HTTP response on the accepted socket.
async fn handle_http_connection(app: App, mut stream: TcpStream) -> Result<()> {
    // Minimal HTTP parsing is enough for localhost daemon<->client communication.
    let mut buf = vec![0u8; 64 * 1024];
    let n = stream.read(&mut buf).await?;
    if n == 0 {
        return Ok(());
    }

    let req = String::from_utf8_lossy(&buf[..n]);
    let (headers, body) = split_http_request(&req);

    let mut lines = headers.lines();
    let request_line = lines.next().unwrap_or_default();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default();
    let path = parts.next().unwrap_or_default();

    let response = route_mcp(app, method, path, body.as_bytes()).await;
    stream.write_all(response.as_bytes()).await?;
    stream.flush().await?;

    Ok(())
}

/// Splits a raw HTTP request into header and body sections.
pub(super) fn split_http_request(raw: &str) -> (String, String) {
    if let Some(idx) = raw.find("\r\n\r\n") {
        (raw[..idx].to_string(), raw[idx + 4..].to_string())
    } else if let Some(idx) = raw.find("\n\n") {
        (raw[..idx].to_string(), raw[idx + 2..].to_string())
    } else {
        (raw.to_string(), String::new())
    }
}

/// Routes MCP HTTP method/path to the corresponding tool handler.
async fn route_mcp(app: App, method: &str, path: &str, body: &[u8]) -> String {
    let payload = match (method, path) {
        ("GET", "/health") => Ok(json!({"ok": true})),
        ("GET", "/mcp/tools") => Ok(json!({
            "schema_version": SCHEMA_VERSION,
            "capabilities": {
                "default_scope": "workspace",
                "scope_filter_field": "filters.workspace_id",
                "override_scope": "Set filters.workspace_id explicitly to query another workspace"
            },
            "tools": [
                "search_code",
                "search_relations",
                "search_files",
                "search_calls",
                "search_types",
                "search_diagnostics",
                "search_semantic_artifacts",
                "search_syntax_artifacts",
                "search_crate_graph",
                "workspace_metadata",
                "get_file_chunks",
                "get_file_context",
                "get_symbol_context",
                "get_symbol_relations",
                "explain_relevance",
                "index_status"
            ]
        })),
        ("GET", "/mcp/tools/index_status") => handle_mcp_index_status(app).await,
        ("GET", "/mcp/tools/workspace_metadata") => handle_mcp_workspace_metadata(app).await,
        ("POST", "/mcp/tools/search_code") => match parse_json_body::<SearchCodeRequest>(body) {
            Ok(req) => handle_mcp_search_code(app, req).await,
            Err(err) => Err(err),
        },
        ("POST", "/mcp/tools/search_relations") => {
            match parse_json_body::<SearchRelationsRequest>(body) {
                Ok(req) => handle_mcp_search_relations(app, req).await,
                Err(err) => Err(err),
            }
        }
        ("POST", "/mcp/tools/search_files") => match parse_json_body::<SearchFilesRequest>(body) {
            Ok(req) => handle_mcp_search_files(app, req).await,
            Err(err) => Err(err),
        },
        ("POST", "/mcp/tools/search_calls") => match parse_json_body::<SearchCallsRequest>(body) {
            Ok(req) => handle_mcp_search_calls(app, req).await,
            Err(err) => Err(err),
        },
        ("POST", "/mcp/tools/search_types") => match parse_json_body::<SearchTypesRequest>(body) {
            Ok(req) => handle_mcp_search_types(app, req).await,
            Err(err) => Err(err),
        },
        ("POST", "/mcp/tools/search_diagnostics") => {
            match parse_json_body::<SearchDiagnosticsRequest>(body) {
                Ok(req) => handle_mcp_search_diagnostics(app, req).await,
                Err(err) => Err(err),
            }
        }
        ("POST", "/mcp/tools/search_semantic_artifacts") => {
            match parse_json_body::<SearchSemanticArtifactsRequest>(body) {
                Ok(req) => handle_mcp_search_semantic_artifacts(app, req).await,
                Err(err) => Err(err),
            }
        }
        ("POST", "/mcp/tools/search_syntax_artifacts") => {
            match parse_json_body::<SearchSyntaxArtifactsRequest>(body) {
                Ok(req) => handle_mcp_search_syntax_artifacts(app, req).await,
                Err(err) => Err(err),
            }
        }
        ("POST", "/mcp/tools/search_crate_graph") => {
            match parse_json_body::<SearchCrateGraphRequest>(body) {
                Ok(req) => handle_mcp_search_crate_graph(app, req).await,
                Err(err) => Err(err),
            }
        }
        ("POST", "/mcp/tools/get_file_chunks") => {
            match parse_json_body::<GetFileChunksRequest>(body) {
                Ok(req) => handle_mcp_get_file_chunks(app, req).await,
                Err(err) => Err(err),
            }
        }
        ("POST", "/mcp/tools/get_file_context") => {
            match parse_json_body::<FileContextRequest>(body) {
                Ok(req) => handle_mcp_get_file_context(app, req).await,
                Err(err) => Err(err),
            }
        }
        ("POST", "/mcp/tools/get_symbol_context") => {
            match parse_json_body::<SymbolContextRequest>(body) {
                Ok(req) => handle_mcp_get_symbol_context(app, req).await,
                Err(err) => Err(err),
            }
        }
        ("POST", "/mcp/tools/get_symbol_relations") => {
            match parse_json_body::<SymbolRelationsRequest>(body) {
                Ok(req) => handle_mcp_get_symbol_relations(app, req).await,
                Err(err) => Err(err),
            }
        }
        ("POST", "/mcp/tools/explain_relevance") => {
            match parse_json_body::<ExplainRelevanceRequest>(body) {
                Ok(req) => handle_mcp_explain_relevance(app, req).await,
                Err(err) => Err(err),
            }
        }
        _ => Err(RpcError {
            code: 404,
            message: format!("not found: {method} {path}"),
        }),
    };

    let (status, value) = match payload {
        Ok(v) => ("200 OK", v),
        Err(err) => (
            if err.code == 404 {
                "404 Not Found"
            } else {
                "400 Bad Request"
            },
            json!({"error": err.message}),
        ),
    };

    http_json(status, &value)
}

/// Parses a JSON request body into a typed request structure.
fn parse_json_body<T: for<'de> Deserialize<'de>>(body: &[u8]) -> Result<T, RpcError> {
    serde_json::from_slice(body).map_err(|err| RpcError {
        code: 400,
        message: format!("invalid json body: {err}"),
    })
}

/// Builds a minimal HTTP JSON response payload.
fn http_json(status: &str, value: &Value) -> String {
    let body = serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string());
    format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

/// Implements `index_status` tool.
async fn handle_mcp_index_status(app: App) -> Result<Value, RpcError> {
    let status = app.state.read().await.status();
    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "indexing_in_progress": status.indexing_in_progress,
        "status": status,
    }))
}

async fn handle_mcp_workspace_metadata(app: App) -> Result<Value, RpcError> {
    let state = app.state.read().await;
    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "indexing_in_progress": state.indexing_in_progress,
        "workspace_metadata": state.workspace_metadata,
    }))
}

/// Implements `search_code` using lexical + semantic hybrid ranking.
async fn handle_mcp_search_code(app: App, req: SearchCodeRequest) -> Result<Value, RpcError> {
    let top_k = resolve_limit(req.limit, req.top_k, 8);
    let semantic_samples = (top_k.saturating_mul(4)).clamp(top_k, 64);
    let vector_name = req.vector_name.clone();
    let default_workspace_id = app.config.read().await.workspace_id.clone();

    let lexical = lexical_search(
        app.state.read().await.rust_items_by_file.clone(),
        &req.query,
        top_k,
    );

    let (semantic, used_semantic_vectors) = {
        let services = app.services.read().await.clone();
        semantic_search_fused(
            &services.store,
            &req.query,
            vector_name.as_deref(),
            semantic_samples,
            &[
                (SYMBOL_VECTOR_NAME, 0.38),
                (DOCS_VECTOR_NAME, 0.22),
                (SIGNATURE_VECTOR_NAME, 0.12),
                (TYPE_VECTOR_NAME, 0.12),
                (SEMANTIC_VECTOR_NAME, 0.08),
                (SYNTAX_VECTOR_NAME, 0.08),
            ],
            |item: &schema::SymbolDoc| {
                matches_filters(item, req.filters.as_ref(), &default_workspace_id)
            },
            "search request",
            "semantic search failed",
        )
        .await?
    };

    let mut merged: HashMap<String, (f64, schema::SymbolDoc)> = HashMap::new();

    for (score, item) in lexical {
        if matches_filters(&item, req.filters.as_ref(), &default_workspace_id) {
            merged
                .entry(item.id.clone())
                .and_modify(|entry| entry.0 += score * 0.4)
                .or_insert((score * 0.4, item));
        }
    }

    for (score, item) in semantic {
        merged
            .entry(item.id.clone())
            .and_modify(|entry| entry.0 += score * 0.6)
            .or_insert((score * 0.6, item));
    }

    let mut rows = merged.into_values().collect::<Vec<_>>();
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    rows.truncate(top_k);

    let state = app.state.read().await;

    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "indexing_in_progress": state.indexing_in_progress,
        "semantic_vector_name": vector_name.unwrap_or_else(|| "fused".to_string()),
        "semantic_vector_names": used_semantic_vectors,
        "results": rows
            .into_iter()
            .map(|(score, item)| json!({
                "score": score,
                "item": canonical_symbol_doc(&item),
            }))
            .collect::<Vec<_>>(),
    }))
}

/// Implements `search_relations` using semantic ranking from relation store.
async fn handle_mcp_search_relations(
    app: App,
    req: SearchRelationsRequest,
) -> Result<Value, RpcError> {
    let top_k = resolve_limit(req.limit, req.top_k, 8);
    let vector_name = req.vector_name.clone();
    let default_workspace_id = app.config.read().await.workspace_id.clone();
    let semantic_samples = top_k
        .saturating_mul(4)
        .clamp(RELATION_CANDIDATE_POOL_MIN, RELATION_CANDIDATE_POOL_MAX);

    let (semantic, used_semantic_vectors) = {
        let services = app.services.read().await.clone();
        semantic_search_fused(
            &services.relation_store,
            &req.query,
            vector_name.as_deref(),
            semantic_samples,
            &[
                (SYMBOL_VECTOR_NAME, 0.30),
                (DOCS_VECTOR_NAME, 0.20),
                (SIGNATURE_VECTOR_NAME, 0.15),
                (BODY_VECTOR_NAME, 0.10),
                (TYPE_VECTOR_NAME, 0.10),
                (GRAPH_VECTOR_NAME, 0.15),
            ],
            |item: &schema::GraphEdgeDoc| {
                matches_relation_filters(item, req.filters.as_ref(), &default_workspace_id)
            },
            "relation search request",
            "semantic relation search failed",
        )
        .await?
    };

    let rows = semantic
        .into_iter()
        .map(|(score, item)| (score, item))
        .collect::<Vec<_>>();
    let rows = rerank_relation_results(&req.query, rows, top_k);

    let state = app.state.read().await;
    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "indexing_in_progress": state.indexing_in_progress,
        "semantic_vector_name": vector_name.unwrap_or_else(|| "fused".to_string()),
        "semantic_vector_names": used_semantic_vectors,
        "results": rows
            .into_iter()
            .map(|(score, item)| json!({
                "score": score,
                "item": canonical_relation_doc(&item),
            }))
            .collect::<Vec<_>>(),
    }))
}

/// Implements `search_files` over file-level docs.
async fn handle_mcp_search_files(app: App, req: SearchFilesRequest) -> Result<Value, RpcError> {
    let top_k = resolve_limit(req.limit, req.top_k, 8);
    let semantic_samples = (top_k.saturating_mul(3)).clamp(top_k, 64);
    let vector_name = req.vector_name.clone();
    let default_workspace_id = app.config.read().await.workspace_id.clone();

    let (semantic, used_semantic_vectors) = {
        let services = app.services.read().await.clone();
        semantic_search_fused(
            &services.file_store,
            &req.query,
            vector_name.as_deref(),
            semantic_samples,
            &[(SYMBOL_VECTOR_NAME, 0.55), (DOCS_VECTOR_NAME, 0.45)],
            |item: &schema::FileDoc| {
                matches_file_filters(item, req.filters.as_ref(), &default_workspace_id)
            },
            "file search request",
            "semantic file search failed",
        )
        .await?
    };

    let rows = semantic
        .into_iter()
        .map(|(score, item)| (score, item))
        .take(top_k)
        .collect::<Vec<_>>();

    let state = app.state.read().await;
    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "indexing_in_progress": state.indexing_in_progress,
        "semantic_vector_name": vector_name.unwrap_or_else(|| "fused".to_string()),
        "semantic_vector_names": used_semantic_vectors,
        "results": rows
            .into_iter()
            .map(|(score, item)| json!({
                "score": score,
                "item": canonical_file_doc(&item),
            }))
            .collect::<Vec<_>>(),
    }))
}

/// Implements `search_calls` over typed call edges.
async fn handle_mcp_search_calls(app: App, req: SearchCallsRequest) -> Result<Value, RpcError> {
    let top_k = resolve_limit(req.limit, req.top_k, 8);
    let semantic_samples = (top_k.saturating_mul(3)).clamp(top_k, 64);
    let vector_name = req.vector_name.clone();
    let default_workspace_id = app.config.read().await.workspace_id.clone();

    let (semantic, used_semantic_vectors) = {
        let services = app.services.read().await.clone();
        semantic_search_fused(
            &services.call_edge_store,
            &req.query,
            vector_name.as_deref(),
            semantic_samples,
            &[
                (SYMBOL_VECTOR_NAME, 0.35),
                (DOCS_VECTOR_NAME, 0.30),
                (GRAPH_VECTOR_NAME, 0.35),
            ],
            |item: &schema::CallEdge| {
                matches_call_edge_filters(item, req.filters.as_ref(), &default_workspace_id)
            },
            "call search request",
            "semantic call search failed",
        )
        .await?
    };

    let rows = semantic
        .into_iter()
        .map(|(score, item)| (score, item))
        .take(top_k)
        .collect::<Vec<_>>();

    let state = app.state.read().await;
    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "indexing_in_progress": state.indexing_in_progress,
        "semantic_vector_name": vector_name.unwrap_or_else(|| "fused".to_string()),
        "semantic_vector_names": used_semantic_vectors,
        "results": rows
            .into_iter()
            .map(|(score, item)| json!({
                "score": score,
                "item": canonical_call_edge(&item),
            }))
            .collect::<Vec<_>>(),
    }))
}

/// Implements `search_types` over typed type edges.
async fn handle_mcp_search_types(app: App, req: SearchTypesRequest) -> Result<Value, RpcError> {
    let top_k = resolve_limit(req.limit, req.top_k, 8);
    let semantic_samples = (top_k.saturating_mul(3)).clamp(top_k, 64);
    let vector_name = req.vector_name.clone();
    let default_workspace_id = app.config.read().await.workspace_id.clone();

    let (semantic, used_semantic_vectors) = {
        let services = app.services.read().await.clone();
        semantic_search_fused(
            &services.type_edge_store,
            &req.query,
            vector_name.as_deref(),
            semantic_samples,
            &[
                (SYMBOL_VECTOR_NAME, 0.25),
                (DOCS_VECTOR_NAME, 0.25),
                (TYPE_VECTOR_NAME, 0.30),
                (GRAPH_VECTOR_NAME, 0.20),
            ],
            |item: &schema::TypeEdge| {
                matches_type_edge_filters(item, req.filters.as_ref(), &default_workspace_id)
            },
            "type-edge search request",
            "semantic type-edge search failed",
        )
        .await?
    };

    let rows = semantic
        .into_iter()
        .map(|(score, item)| (score, item))
        .take(top_k)
        .collect::<Vec<_>>();

    let state = app.state.read().await;
    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "indexing_in_progress": state.indexing_in_progress,
        "semantic_vector_name": vector_name.unwrap_or_else(|| "fused".to_string()),
        "semantic_vector_names": used_semantic_vectors,
        "results": rows
            .into_iter()
            .map(|(score, item)| json!({
                "score": score,
                "item": canonical_type_edge(&item),
            }))
            .collect::<Vec<_>>(),
    }))
}

/// Implements `search_diagnostics` over extracted diagnostics.
async fn handle_mcp_search_diagnostics(
    app: App,
    req: SearchDiagnosticsRequest,
) -> Result<Value, RpcError> {
    let top_k = resolve_limit(req.limit, req.top_k, 8);
    let semantic_samples = (top_k.saturating_mul(3)).clamp(top_k, 64);
    let vector_name = req.vector_name.clone();
    let default_workspace_id = app.config.read().await.workspace_id.clone();

    let (semantic, used_semantic_vectors) = {
        let services = app.services.read().await.clone();
        match semantic_search_fused(
            &services.diagnostic_store,
            &req.query,
            vector_name.as_deref(),
            semantic_samples,
            &[(SYMBOL_VECTOR_NAME, 0.50), (DOCS_VECTOR_NAME, 0.50)],
            |item: &schema::DiagnosticDoc| {
                matches_diagnostic_filters(item, req.filters.as_ref(), &default_workspace_id)
            },
            "diagnostic search request",
            "semantic diagnostic search failed",
        )
        .await
        {
            Ok(rows) => rows,
            Err(err) if is_missing_collection_rpc_error(&err.message) => {
                let state = app.state.read().await;
                return Ok(json!({
                    "schema_version": SCHEMA_VERSION,
                    "indexing_in_progress": state.indexing_in_progress,
                    "semantic_vector_name": vector_name.unwrap_or_else(|| "fused".to_string()),
                    "semantic_vector_names": Vec::<String>::new(),
                    "results": Vec::<Value>::new(),
                    "availability": {
                        "status": "unavailable",
                        "reason": "diagnostic_collection_not_initialized"
                    }
                }));
            }
            Err(err) => return Err(err),
        }
    };

    let rows = semantic
        .into_iter()
        .map(|(score, item)| (score, item))
        .take(top_k)
        .collect::<Vec<_>>();

    let state = app.state.read().await;
    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "indexing_in_progress": state.indexing_in_progress,
        "semantic_vector_name": vector_name.unwrap_or_else(|| "fused".to_string()),
        "semantic_vector_names": used_semantic_vectors,
        "results": rows
            .into_iter()
            .map(|(score, item)| json!({
                "score": score,
                "item": canonical_diagnostic_doc(&item),
            }))
            .collect::<Vec<_>>(),
    }))
}

/// Implements `search_semantic_artifacts` over semantic-token docs.
async fn handle_mcp_search_semantic_artifacts(
    app: App,
    req: SearchSemanticArtifactsRequest,
) -> Result<Value, RpcError> {
    let top_k = resolve_limit(req.limit, req.top_k, 8);
    let semantic_samples = (top_k.saturating_mul(3)).clamp(top_k, 64);
    let vector_name = req.vector_name.clone();
    let default_workspace_id = app.config.read().await.workspace_id.clone();

    let (semantic, used_semantic_vectors) = {
        let services = app.services.read().await.clone();
        semantic_search_fused(
            &services.semantic_store,
            &req.query,
            vector_name.as_deref(),
            semantic_samples,
            &[(SYMBOL_VECTOR_NAME, 0.30), (SEMANTIC_VECTOR_NAME, 0.70)],
            |item: &schema::SemanticTokenDoc| {
                matches_semantic_artifact_filters(item, req.filters.as_ref(), &default_workspace_id)
            },
            "semantic artifact search request",
            "semantic artifact search failed",
        )
        .await?
    };

    let rows = semantic.into_iter().take(top_k).collect::<Vec<_>>();
    let state = app.state.read().await;
    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "indexing_in_progress": state.indexing_in_progress,
        "semantic_vector_name": vector_name.unwrap_or_else(|| "fused".to_string()),
        "semantic_vector_names": used_semantic_vectors,
        "results": rows
            .into_iter()
            .map(|(score, item)| json!({
                "score": score,
                "item": canonical_semantic_artifact_doc(&item),
            }))
            .collect::<Vec<_>>(),
    }))
}

/// Implements `search_syntax_artifacts` over syntax-tree docs.
async fn handle_mcp_search_syntax_artifacts(
    app: App,
    req: SearchSyntaxArtifactsRequest,
) -> Result<Value, RpcError> {
    if !app.config.read().await.enable_syntax_artifacts {
        let state = app.state.read().await;
        return Ok(json!({
            "schema_version": SCHEMA_VERSION,
            "indexing_in_progress": state.indexing_in_progress,
            "semantic_vector_name": req.vector_name.unwrap_or_else(|| "fused".to_string()),
            "semantic_vector_names": Vec::<String>::new(),
            "results": Vec::<Value>::new(),
            "availability": {
                "status": "disabled",
                "reason": "syntax_artifacts_feature_flag_off"
            }
        }));
    }

    let top_k = resolve_limit(req.limit, req.top_k, 8);
    let semantic_samples = (top_k.saturating_mul(3)).clamp(top_k, 64);
    let vector_name = req.vector_name.clone();
    let default_workspace_id = app.config.read().await.workspace_id.clone();

    let (semantic, used_semantic_vectors) = {
        let services = app.services.read().await.clone();
        match semantic_search_fused(
            &services.syntax_store,
            &req.query,
            vector_name.as_deref(),
            semantic_samples,
            &[(SYMBOL_VECTOR_NAME, 0.30), (SYNTAX_VECTOR_NAME, 0.70)],
            |item: &schema::SyntaxTreeDoc| {
                matches_syntax_artifact_filters(item, req.filters.as_ref(), &default_workspace_id)
            },
            "syntax artifact search request",
            "syntax artifact search failed",
        )
        .await
        {
            Ok(rows) => rows,
            Err(err) if is_missing_collection_rpc_error(&err.message) => {
                let state = app.state.read().await;
                return Ok(json!({
                    "schema_version": SCHEMA_VERSION,
                    "indexing_in_progress": state.indexing_in_progress,
                    "semantic_vector_name": vector_name.unwrap_or_else(|| "fused".to_string()),
                    "semantic_vector_names": Vec::<String>::new(),
                    "results": Vec::<Value>::new(),
                    "availability": {
                        "status": "unavailable",
                        "reason": "syntax_collection_not_initialized"
                    }
                }));
            }
            Err(err) => return Err(err),
        }
    };

    let rows = semantic.into_iter().take(top_k).collect::<Vec<_>>();
    let state = app.state.read().await;
    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "indexing_in_progress": state.indexing_in_progress,
        "semantic_vector_name": vector_name.unwrap_or_else(|| "fused".to_string()),
        "semantic_vector_names": used_semantic_vectors,
        "results": rows
            .into_iter()
            .map(|(score, item)| json!({
                "score": score,
                "item": canonical_syntax_artifact_doc(&item),
            }))
            .collect::<Vec<_>>(),
    }))
}

/// Implements `search_crate_graph` over crate-graph docs.
async fn handle_mcp_search_crate_graph(
    app: App,
    req: SearchCrateGraphRequest,
) -> Result<Value, RpcError> {
    let top_k = resolve_limit(req.limit, req.top_k, 8);
    let semantic_samples = (top_k.saturating_mul(3)).clamp(top_k, 64);
    let vector_name = req.vector_name.clone();
    let default_workspace_id = app.config.read().await.workspace_id.clone();

    let (semantic, used_semantic_vectors) = {
        let services = app.services.read().await.clone();
        semantic_search_fused(
            &services.crate_graph_store,
            &req.query,
            vector_name.as_deref(),
            semantic_samples,
            &[(SYMBOL_VECTOR_NAME, 0.35), (GRAPH_VECTOR_NAME, 0.65)],
            |item: &schema::CrateGraphDoc| {
                matches_crate_graph_filters(item, req.filters.as_ref(), &default_workspace_id)
            },
            "crate-graph search request",
            "semantic crate-graph search failed",
        )
        .await?
    };

    let rows = semantic.into_iter().take(top_k).collect::<Vec<_>>();
    let state = app.state.read().await;
    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "indexing_in_progress": state.indexing_in_progress,
        "semantic_vector_name": vector_name.unwrap_or_else(|| "fused".to_string()),
        "semantic_vector_names": used_semantic_vectors,
        "results": rows
            .into_iter()
            .map(|(score, item)| json!({
                "score": score,
                "item": canonical_crate_graph_doc(&item),
            }))
            .collect::<Vec<_>>(),
    }))
}

/// Lexical scoring component used by `search_code`.
pub(super) fn lexical_search(
    items_by_file: HashMap<String, Vec<schema::SymbolDoc>>,
    query: &str,
    top_k: usize,
) -> Vec<(f64, schema::SymbolDoc)> {
    let tokens: Vec<String> = query
        .to_lowercase()
        .split_whitespace()
        .map(std::string::ToString::to_string)
        .collect();

    let mut scored = Vec::new();
    for items in items_by_file.into_values() {
        for item in items {
            let lower = rust_item_search_text(&item).to_lowercase();
            let mut score = 0.0f64;
            for token in &tokens {
                if lower.contains(token) {
                    score += 1.0;
                }
            }
            if score > 0.0 {
                scored.push((score / (tokens.len().max(1) as f64), item));
            }
        }
    }

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);
    scored
}

fn normalize_channel_weights(channels: &[(&str, f64)]) -> Vec<(String, f64)> {
    let positive_total = channels
        .iter()
        .filter_map(|(_, weight)| (*weight > 0.0).then_some(*weight))
        .sum::<f64>();

    if positive_total > 0.0 {
        return channels
            .iter()
            .filter_map(|(name, weight)| {
                (*weight > 0.0).then_some(((*name).to_string(), *weight / positive_total))
            })
            .collect::<Vec<_>>();
    }

    if channels.is_empty() {
        return Vec::new();
    }

    let default = 1.0 / channels.len() as f64;
    channels
        .iter()
        .map(|(name, _)| ((*name).to_string(), default))
        .collect::<Vec<_>>()
}

async fn semantic_search_fused<T, F>(
    store: &QdrantVectorStore,
    query: &str,
    vector_name_override: Option<&str>,
    samples: usize,
    channels: &[(&str, f64)],
    keep: F,
    invalid_request_context: &str,
    search_failure_context: &str,
) -> Result<(Vec<(f64, T)>, Vec<String>), RpcError>
where
    T: for<'de> Deserialize<'de> + Send + Sync + Clone,
    F: Fn(&T) -> bool,
{
    if let Some(vector_name) = vector_name_override {
        let request = VectorSearchRequest::builder()
            .query(query.to_string())
            .query_vector_name(vector_name.to_string())
            .samples(samples as u64)
            .build()
            .map_err(|err| RpcError {
                code: 400,
                message: format!("invalid {invalid_request_context}: {err}"),
            })?;

        let rows = store.top_n::<T>(request).await.map_err(|err| RpcError {
            code: 500,
            message: format!("{search_failure_context}: {err}"),
        })?;

        let filtered = rows
            .into_iter()
            .filter_map(|(score, _, item)| keep(&item).then_some((score, item)))
            .collect::<Vec<_>>();
        return Ok((filtered, vec![vector_name.to_string()]));
    }

    let normalized = normalize_channel_weights(channels);
    let mut merged: HashMap<String, (f64, T)> = HashMap::new();
    let mut used_vectors = Vec::<String>::new();

    for (vector_name, weight) in normalized {
        let request = VectorSearchRequest::builder()
            .query(query.to_string())
            .query_vector_name(vector_name.clone())
            .samples(samples as u64)
            .build()
            .map_err(|err| RpcError {
                code: 400,
                message: format!("invalid {invalid_request_context}: {err}"),
            })?;

        let rows = match store.top_n::<T>(request).await {
            Ok(rows) => rows,
            Err(err) if is_missing_vector_error(&err) => {
                // Channel not indexed yet in this collection: skip instead of failing
                // the whole fused search request.
                continue;
            }
            Err(err) => {
                return Err(RpcError {
                    code: 500,
                    message: format!("{search_failure_context}: {err}"),
                });
            }
        };
        used_vectors.push(vector_name);

        for (score, id, item) in rows {
            if !keep(&item) {
                continue;
            }
            let entry = merged.entry(id).or_insert_with(|| (0.0, item));
            entry.0 += score * weight;
        }
    }

    let mut rows = merged.into_values().collect::<Vec<_>>();
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok((rows, used_vectors))
}

fn is_missing_vector_error(err: &autoagents_core::vector_store::VectorStoreError) -> bool {
    let msg = err.to_string().to_lowercase();
    (msg.contains("vector") && msg.contains("not found"))
        || msg.contains("vector name")
        || msg.contains("no vector")
}

fn is_missing_collection_rpc_error(message: &str) -> bool {
    let msg = message.to_lowercase();
    msg.contains("collection") && (msg.contains("doesn't exist") || msg.contains("not found"))
}

fn resolve_limit(limit: Option<usize>, top_k: Option<usize>, default: usize) -> usize {
    limit.or(top_k).unwrap_or(default).max(1)
}

fn canonical_symbol_doc(item: &schema::SymbolDoc) -> schema::SymbolDoc {
    item.clone()
}

fn canonical_relation_doc(item: &schema::GraphEdgeDoc) -> schema::GraphEdgeDoc {
    item.clone()
}

fn canonical_file_doc(item: &schema::FileDoc) -> schema::FileDoc {
    item.clone()
}

fn canonical_call_edge(item: &schema::CallEdge) -> schema::CallEdge {
    item.clone()
}

fn canonical_type_edge(item: &schema::TypeEdge) -> schema::TypeEdge {
    item.clone()
}

fn canonical_diagnostic_doc(item: &schema::DiagnosticDoc) -> schema::DiagnosticDoc {
    item.clone()
}

fn canonical_semantic_artifact_doc(item: &schema::SemanticTokenDoc) -> schema::SemanticTokenDoc {
    item.clone()
}

fn canonical_syntax_artifact_doc(item: &schema::SyntaxTreeDoc) -> schema::SyntaxTreeDoc {
    item.clone()
}

fn canonical_crate_graph_doc(item: &schema::CrateGraphDoc) -> schema::CrateGraphDoc {
    item.clone()
}

fn rerank_relation_results(
    query: &str,
    mut rows: Vec<(f64, schema::GraphEdgeDoc)>,
    top_k: usize,
) -> Vec<(f64, schema::GraphEdgeDoc)> {
    let intent_kind = infer_relation_intent_kind(query);
    let query_tokens = query_tokens(query);

    rows.sort_by(|(left_score, left_item), (right_score, right_item)| {
        let left = relation_relevance_score(*left_score, left_item, intent_kind, &query_tokens);
        let right = relation_relevance_score(*right_score, right_item, intent_kind, &query_tokens);
        right
            .partial_cmp(&left)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(left_item.id.cmp(&right_item.id))
    });
    rows.truncate(top_k);
    rows
}

fn relation_relevance_score(
    base_score: f64,
    item: &schema::GraphEdgeDoc,
    intent_kind: Option<&'static str>,
    query_tokens: &HashSet<String>,
) -> f64 {
    const INTENT_KIND_BOOST: f64 = 0.12;
    const SOURCE_SYMBOL_BOOST: f64 = 0.05;

    let mut boosted = base_score;
    if intent_kind.is_some_and(|kind| item.relation_kind == kind) {
        boosted += INTENT_KIND_BOOST;
    }
    if source_symbol_token_overlap(item, query_tokens) {
        boosted += SOURCE_SYMBOL_BOOST;
    }
    boosted
}

fn infer_relation_intent_kind(query: &str) -> Option<&'static str> {
    let tokens = query_tokens(query);

    let has = |needle: &[&str]| needle.iter().any(|term| tokens.contains(*term));
    let has_type = has(&["type", "types", "typedef", "typedefs"]);
    let has_definition = has(&["definition", "definitions", "define", "defined"]);

    if has_type && has_definition || has(&["type_definition", "type_definitions"]) {
        Some("type_definitions")
    } else if has(&[
        "implement",
        "implements",
        "implemented",
        "implementation",
        "implementations",
        "impl",
    ]) {
        Some("implementations")
    } else if has_definition {
        Some("definitions")
    } else if has(&[
        "reference",
        "references",
        "referenced",
        "usage",
        "usages",
        "used",
        "calls",
        "called",
        "callers",
    ]) {
        Some("references")
    } else {
        None
    }
}

fn query_tokens(query: &str) -> HashSet<String> {
    query
        .to_lowercase()
        .split(|ch: char| !ch.is_alphanumeric() && ch != '_')
        .filter(|token| !token.is_empty())
        .map(std::string::ToString::to_string)
        .collect::<HashSet<_>>()
}

fn source_symbol_token_overlap(
    item: &schema::GraphEdgeDoc,
    query_tokens: &HashSet<String>,
) -> bool {
    if query_tokens.is_empty() {
        return false;
    }
    let source_symbol = item.source_symbol.to_lowercase();
    query_tokens
        .iter()
        .filter(|token| token.len() >= 3)
        .any(|token| source_symbol.contains(token.as_str()))
}

/// Applies optional search filters to a chunk.
fn matches_filters(
    item: &schema::SymbolDoc,
    filters: Option<&SearchFilters>,
    default_workspace_id: &str,
) -> bool {
    let workspace_id = filters
        .and_then(|filters| filters.workspace_id.as_deref())
        .unwrap_or(default_workspace_id);
    if item.workspace_id != workspace_id {
        return false;
    }

    let Some(filters) = filters else {
        return true;
    };

    if let Some(file_path) = &filters.file_path
        && !item.file_path.contains(file_path)
    {
        return false;
    }

    if let Some(kind) = &filters.kind
        && item.kind != *kind
    {
        return false;
    }

    true
}

fn matches_relation_filters(
    item: &schema::GraphEdgeDoc,
    filters: Option<&RelationSearchFilters>,
    default_workspace_id: &str,
) -> bool {
    let workspace_id = filters
        .and_then(|filters| filters.workspace_id.as_deref())
        .unwrap_or(default_workspace_id);
    if item.workspace_id != workspace_id {
        return false;
    }

    let Some(filters) = filters else {
        return true;
    };

    if let Some(kind) = &filters.relation_kind
        && item.relation_kind != *kind
    {
        return false;
    }
    if let Some(source_file_path) = &filters.source_file_path
        && !item.source_file_path.contains(source_file_path)
    {
        return false;
    }
    if let Some(target_file_path) = &filters.target_file_path
        && !item.target_file_path.contains(target_file_path)
    {
        return false;
    }

    true
}

fn matches_file_filters(
    item: &schema::FileDoc,
    filters: Option<&FileSearchFilters>,
    default_workspace_id: &str,
) -> bool {
    let workspace_id = filters
        .and_then(|filters| filters.workspace_id.as_deref())
        .unwrap_or(default_workspace_id);
    if item.workspace_id != workspace_id {
        return false;
    }

    let Some(filters) = filters else {
        return true;
    };

    if let Some(file_path) = &filters.file_path
        && !item.file_path.contains(file_path)
    {
        return false;
    }
    if let Some(module) = &filters.module
        && !item.module.contains(module)
    {
        return false;
    }
    if let Some(crate_name) = &filters.crate_name
        && item.crate_name != *crate_name
    {
        return false;
    }
    true
}

fn matches_call_edge_filters(
    item: &schema::CallEdge,
    filters: Option<&CallEdgeSearchFilters>,
    default_workspace_id: &str,
) -> bool {
    let workspace_id = filters
        .and_then(|filters| filters.workspace_id.as_deref())
        .unwrap_or(default_workspace_id);
    if item.workspace_id != workspace_id {
        return false;
    }

    let Some(filters) = filters else {
        return true;
    };

    if let Some(source_file_path) = &filters.source_file_path
        && !item.source_span.file_path.contains(source_file_path)
    {
        return false;
    }
    if let Some(target_file_path) = &filters.target_file_path
        && !item.target_span.file_path.contains(target_file_path)
    {
        return false;
    }
    if let Some(source_symbol_id) = &filters.source_symbol_id
        && item.source_symbol_id != *source_symbol_id
    {
        return false;
    }
    if let Some(target_symbol_id) = &filters.target_symbol_id
        && item.target_symbol_id != *target_symbol_id
    {
        return false;
    }
    true
}

fn matches_type_edge_filters(
    item: &schema::TypeEdge,
    filters: Option<&TypeEdgeSearchFilters>,
    default_workspace_id: &str,
) -> bool {
    let workspace_id = filters
        .and_then(|filters| filters.workspace_id.as_deref())
        .unwrap_or(default_workspace_id);
    if item.workspace_id != workspace_id {
        return false;
    }

    let Some(filters) = filters else {
        return true;
    };

    if let Some(relation_kind) = &filters.relation_kind
        && item.relation_kind != *relation_kind
    {
        return false;
    }
    if let Some(source_file_path) = &filters.source_file_path
        && !item.source_span.file_path.contains(source_file_path)
    {
        return false;
    }
    if let Some(target_file_path) = &filters.target_file_path
        && !item.target_span.file_path.contains(target_file_path)
    {
        return false;
    }
    if let Some(source_symbol_id) = &filters.source_symbol_id
        && item.source_symbol_id != *source_symbol_id
    {
        return false;
    }
    if let Some(target_symbol_id) = &filters.target_symbol_id
        && item.target_symbol_id != *target_symbol_id
    {
        return false;
    }
    true
}

fn matches_diagnostic_filters(
    item: &schema::DiagnosticDoc,
    filters: Option<&DiagnosticSearchFilters>,
    default_workspace_id: &str,
) -> bool {
    let workspace_id = filters
        .and_then(|filters| filters.workspace_id.as_deref())
        .unwrap_or(default_workspace_id);
    if item.workspace_id != workspace_id {
        return false;
    }

    let Some(filters) = filters else {
        return true;
    };

    if let Some(file_path) = &filters.file_path
        && !item.file_path.contains(file_path)
    {
        return false;
    }
    if let Some(severity) = &filters.severity
        && item.severity != *severity
    {
        return false;
    }
    if let Some(code) = &filters.code
        && item.code.as_deref() != Some(code.as_str())
    {
        return false;
    }
    true
}

fn matches_semantic_artifact_filters(
    item: &schema::SemanticTokenDoc,
    filters: Option<&SemanticArtifactSearchFilters>,
    default_workspace_id: &str,
) -> bool {
    let workspace_id = filters
        .and_then(|filters| filters.workspace_id.as_deref())
        .unwrap_or(default_workspace_id);
    if item.workspace_id != workspace_id {
        return false;
    }

    let Some(filters) = filters else {
        return true;
    };

    if let Some(file_path) = &filters.file_path
        && !item.file_path.contains(file_path)
    {
        return false;
    }
    if let Some(symbol_id) = &filters.symbol_id
        && item.symbol_id.as_deref() != Some(symbol_id.as_str())
    {
        return false;
    }
    true
}

fn matches_syntax_artifact_filters(
    item: &schema::SyntaxTreeDoc,
    filters: Option<&SyntaxArtifactSearchFilters>,
    default_workspace_id: &str,
) -> bool {
    let workspace_id = filters
        .and_then(|filters| filters.workspace_id.as_deref())
        .unwrap_or(default_workspace_id);
    if item.workspace_id != workspace_id {
        return false;
    }

    let Some(filters) = filters else {
        return true;
    };

    if let Some(file_path) = &filters.file_path
        && !item.file_path.contains(file_path)
    {
        return false;
    }
    if let Some(symbol_id) = &filters.symbol_id
        && item.symbol_id.as_deref() != Some(symbol_id.as_str())
    {
        return false;
    }
    true
}

fn matches_crate_graph_filters(
    item: &schema::CrateGraphDoc,
    filters: Option<&CrateGraphSearchFilters>,
    default_workspace_id: &str,
) -> bool {
    let workspace_id = filters
        .and_then(|filters| filters.workspace_id.as_deref())
        .unwrap_or(default_workspace_id);
    if item.workspace_id != workspace_id {
        return false;
    }

    let Some(filters) = filters else {
        return true;
    };

    if let Some(crate_name) = &filters.crate_name
        && item.crate_name != *crate_name
    {
        return false;
    }
    if let Some(crate_root) = &filters.crate_root
        && !item.crate_root.contains(crate_root)
    {
        return false;
    }
    true
}

/// Implements `get_file_context` tool.
async fn handle_mcp_get_file_context(app: App, req: FileContextRequest) -> Result<Value, RpcError> {
    let limit = req.limit.unwrap_or(16).max(1);
    let state = app.state.read().await;

    let file_doc = state
        .file_docs_by_file
        .get(&req.file_path)
        .map(canonical_file_doc);
    let symbols = state
        .rust_items_by_file
        .get(&req.file_path)
        .map(|items| {
            items
                .iter()
                .take(limit)
                .map(canonical_symbol_doc)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let relations = state
        .relations_by_file
        .get(&req.file_path)
        .map(|items| {
            items
                .iter()
                .take(limit)
                .map(canonical_relation_doc)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let call_edges = state
        .call_edges_by_file
        .get(&req.file_path)
        .map(|items| {
            items
                .iter()
                .take(limit)
                .map(canonical_call_edge)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let type_edges = state
        .type_edges_by_file
        .get(&req.file_path)
        .map(|items| {
            items
                .iter()
                .take(limit)
                .map(canonical_type_edge)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let diagnostics = state
        .diagnostics_by_file
        .get(&req.file_path)
        .map(|items| {
            items
                .iter()
                .take(limit)
                .map(canonical_diagnostic_doc)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "indexing_in_progress": state.indexing_in_progress,
        "file_path": req.file_path,
        "file_doc": file_doc,
        "symbols": symbols,
        "relations": relations,
        "call_edges": call_edges,
        "type_edges": type_edges,
        "diagnostics": diagnostics,
    }))
}

/// Implements `get_file_chunks` tool.
async fn handle_mcp_get_file_chunks(
    app: App,
    req: GetFileChunksRequest,
) -> Result<Value, RpcError> {
    let state = app.state.read().await;
    let items = state
        .rust_items_by_file
        .get(&req.file_path)
        .cloned()
        .unwrap_or_default();

    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "indexing_in_progress": state.indexing_in_progress,
        "items": items
            .iter()
            .map(canonical_symbol_doc)
            .collect::<Vec<_>>(),
    }))
}

/// Implements `get_symbol_context` tool.
async fn handle_mcp_get_symbol_context(
    app: App,
    req: SymbolContextRequest,
) -> Result<Value, RpcError> {
    let needle = req.symbol.to_lowercase();
    let limit = req.limit.unwrap_or(8).max(1);
    let state = app.state.read().await;
    let mut locations = Vec::new();

    for (file_path, items) in &state.rust_items_by_file {
        if let Some(filter_path) = &req.file_path
            && filter_path != file_path
        {
            continue;
        }

        for item in items {
            let search_text = rust_item_search_text(item).to_lowercase();
            if search_text.contains(&needle) {
                locations.push(json!({
                    "file_path": file_path,
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "item": canonical_symbol_doc(item),
                }));
            }
        }
    }

    locations.sort_by(|a, b| {
        let a_file = a
            .get("file_path")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let b_file = b
            .get("file_path")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let a_start = a
            .get("start_line")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let b_start = b
            .get("start_line")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let a_end = a
            .get("end_line")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let b_end = b
            .get("end_line")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let a_id = a
            .get("item")
            .and_then(|item| item.get("id"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let b_id = b
            .get("item")
            .and_then(|item| item.get("id"))
            .and_then(Value::as_str)
            .unwrap_or_default();

        a_file
            .cmp(b_file)
            .then(a_start.cmp(&b_start))
            .then(a_end.cmp(&b_end))
            .then(a_id.cmp(b_id))
    });
    locations.truncate(limit);

    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "indexing_in_progress": state.indexing_in_progress,
        "locations": locations,
    }))
}

/// Implements `get_symbol_relations` tool.
async fn handle_mcp_get_symbol_relations(
    app: App,
    req: SymbolRelationsRequest,
) -> Result<Value, RpcError> {
    let limit = req.limit.unwrap_or(8).max(1);
    let relation_kind = req.relation_kind.as_deref();
    let state = app.state.read().await;
    let mut relations = Vec::new();

    for (file_path, items) in &state.relations_by_file {
        if let Some(filter_path) = &req.file_path
            && filter_path != file_path
        {
            continue;
        }

        for item in items {
            if let Some(kind) = relation_kind
                && item.relation_kind != kind
            {
                continue;
            }
            if relation_matches_symbol(item, &req.symbol) {
                relations.push(json!({
                    "file_path": file_path,
                    "item": canonical_relation_doc(item),
                }));
            }
        }
    }

    relations.sort_by(|a, b| {
        let a_file = a
            .get("file_path")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let b_file = b
            .get("file_path")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let a_target_file = a
            .get("item")
            .and_then(|item| item.get("target_file_path"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let b_target_file = b
            .get("item")
            .and_then(|item| item.get("target_file_path"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let a_start = a
            .get("item")
            .and_then(|item| item.get("target_start_line"))
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let b_start = b
            .get("item")
            .and_then(|item| item.get("target_start_line"))
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let a_end = a
            .get("item")
            .and_then(|item| item.get("target_end_line"))
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let b_end = b
            .get("item")
            .and_then(|item| item.get("target_end_line"))
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let a_id = a
            .get("item")
            .and_then(|item| item.get("id"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let b_id = b
            .get("item")
            .and_then(|item| item.get("id"))
            .and_then(Value::as_str)
            .unwrap_or_default();

        a_file
            .cmp(b_file)
            .then(a_target_file.cmp(b_target_file))
            .then(a_start.cmp(&b_start))
            .then(a_end.cmp(&b_end))
            .then(a_id.cmp(b_id))
    });
    relations.truncate(limit);

    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "indexing_in_progress": state.indexing_in_progress,
        "relations": relations,
    }))
}

fn relation_matches_symbol(item: &schema::GraphEdgeDoc, query: &str) -> bool {
    let query = query.trim();
    if query.is_empty() {
        return false;
    }

    let query_leaf = query.rsplit("::").next().unwrap_or(query).trim();
    if item.source_symbol.eq_ignore_ascii_case(query)
        || item.source_symbol.eq_ignore_ascii_case(query_leaf)
    {
        return true;
    }

    let source_id = item.source_symbol_id.to_lowercase();
    let query_lower = query.to_lowercase();
    let query_leaf_lower = query_leaf.to_lowercase();
    source_id.contains(&format!(":{query_lower}:"))
        || source_id.contains(&format!(":{query_leaf_lower}:"))
}

fn explain_query_terms(query: &str) -> (Vec<String>, Vec<String>) {
    let mut kept = Vec::new();
    let mut ignored = Vec::new();

    for raw in query.split_whitespace() {
        for part in raw.split("::") {
            for token in part.split(|ch: char| !(ch.is_alphanumeric() || ch == '_')) {
                if token.is_empty() {
                    continue;
                }
                let normalized = token.to_lowercase();
                if normalized.len() < 3 || is_low_signal_query_term(&normalized) {
                    if !ignored.contains(&normalized) {
                        ignored.push(normalized);
                    }
                    continue;
                }
                if !kept.contains(&normalized) {
                    kept.push(normalized);
                }
            }
        }
    }

    (kept, ignored)
}

fn is_low_signal_query_term(token: &str) -> bool {
    // Keep this list short and high-confidence to avoid dropping meaningful code terms.
    const STOPWORDS: &[&str] = &[
        "a",
        "an",
        "and",
        "are",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "where",
        "with",
        "find",
        "show",
        "implemented",
        "implementation",
        "function",
        "code",
    ];
    STOPWORDS.contains(&token)
}

fn compute_query_term_idf<'a>(
    terms: &[String],
    corpus: impl IntoIterator<Item = &'a schema::SymbolDoc>,
) -> HashMap<String, f64> {
    let corpus_docs = corpus.into_iter().collect::<Vec<_>>();
    let n = corpus_docs.len() as f64;
    let mut idf = HashMap::new();

    for term in terms {
        let df = corpus_docs
            .iter()
            .filter(|doc| {
                rust_item_search_text(doc)
                    .to_lowercase()
                    .contains(term.as_str())
            })
            .count() as f64;
        let value = ((n + 1.0) / (df + 1.0)).ln() + 1.0;
        idf.insert(term.clone(), value);
    }

    idf
}

fn explain_term_evidence(
    item: &schema::SymbolDoc,
    terms: &[String],
    term_idf: &HashMap<String, f64>,
    score_threshold: f64,
) -> (
    Vec<String>,
    HashMap<String, f64>,
    HashMap<String, Vec<String>>,
) {
    const FIELD_WEIGHTS: [(&str, f64); 6] = [
        ("symbol", 3.0),
        ("symbol_path", 3.0),
        ("signature", 2.0),
        ("docs", 1.0),
        ("hover_summary", 1.0),
        ("body_excerpt", 1.0),
    ];

    let field_values = [
        ("symbol", item.symbol.to_lowercase()),
        ("symbol_path", item.symbol_path.to_lowercase()),
        ("signature", item.signature.to_lowercase()),
        ("docs", item.docs.to_lowercase()),
        ("hover_summary", item.hover_summary.to_lowercase()),
        ("body_excerpt", item.body_excerpt.to_lowercase()),
    ];

    let mut term_scores: HashMap<String, f64> = HashMap::new();
    let mut evidence_fields: HashMap<String, Vec<String>> = HashMap::new();

    for term in terms {
        let idf = term_idf.get(term).copied().unwrap_or(1.0);
        for (field_name, field_weight) in FIELD_WEIGHTS {
            if let Some((_, value)) = field_values.iter().find(|(name, _)| *name == field_name)
                && value.contains(term.as_str())
            {
                *term_scores.entry(term.clone()).or_insert(0.0) += field_weight * idf;
                let fields = evidence_fields.entry(term.clone()).or_default();
                if !fields.iter().any(|f| f == field_name) {
                    fields.push(field_name.to_string());
                }
            }
        }
    }

    let mut ranked = term_scores
        .iter()
        .filter(|(_, score)| **score >= score_threshold)
        .map(|(term, score)| (term.clone(), *score))
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    let matched_terms = ranked
        .iter()
        .map(|(term, _)| term.clone())
        .collect::<Vec<_>>();
    let matched_term_scores = ranked.into_iter().collect::<HashMap<_, _>>();

    (matched_terms, matched_term_scores, evidence_fields)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relation_doc(source_symbol: &str, source_symbol_id: &str) -> schema::GraphEdgeDoc {
        schema::GraphEdgeDoc {
            id: "relation:test".to_string(),
            workspace_id: "ws_test".to_string(),
            relation_kind: "references".to_string(),
            edge_type: "symbol_reference".to_string(),
            source_symbol_id: source_symbol_id.to_string(),
            source_symbol: source_symbol.to_string(),
            source_file_path: "src/lib.rs".to_string(),
            source_crate_name: "fixture".to_string(),
            source_uri: "file:///tmp/src/lib.rs".to_string(),
            target_file_path: "src/lib.rs".to_string(),
            target_uri: "file:///tmp/src/lib.rs".to_string(),
            target_start_line: 1,
            target_end_line: 1,
            target_excerpt: "pub enum ValueState {}".to_string(),
        }
    }

    #[test]
    fn relation_matches_symbol_requires_source_identity() {
        let value_state = relation_doc(
            "ValueState",
            "relation:ws_test:references:symbol:ws_test:src/types.rs:enum:ValueState:8:12:src/lib.rs:1:1",
        );
        let run_default = relation_doc(
            "run_default",
            "relation:ws_test:references:symbol:ws_test:src/engine.rs:fn:run_default:12:20:src/lib.rs:1:1",
        );

        assert!(relation_matches_symbol(&value_state, "ValueState"));
        assert!(relation_matches_symbol(&value_state, "types::ValueState"));
        assert!(!relation_matches_symbol(&run_default, "ValueState"));
    }

    #[test]
    fn explain_query_terms_filters_low_signal_words() {
        let (kept, ignored) = explain_query_terms("where is Rule implemented in the engine?");

        assert!(kept.contains(&"rule".to_string()));
        assert!(kept.contains(&"engine".to_string()));
        assert!(!kept.contains(&"where".to_string()));
        assert!(!kept.contains(&"implemented".to_string()));
        assert!(ignored.contains(&"where".to_string()));
        assert!(ignored.contains(&"implemented".to_string()));
    }

    #[test]
    fn explain_query_terms_keeps_code_identifiers() {
        let (kept, _ignored) = explain_query_terms("impl crate::Rule for PositiveRule");

        assert!(kept.contains(&"impl".to_string()));
        assert!(kept.contains(&"crate".to_string()));
        assert!(kept.contains(&"rule".to_string()));
        assert!(kept.contains(&"positiverule".to_string()));
    }

    fn sample_doc(
        id: &str,
        symbol: &str,
        signature: &str,
        body_excerpt: &str,
    ) -> schema::SymbolDoc {
        schema::SymbolDoc {
            id: id.to_string(),
            kind: "fn".to_string(),
            symbol: symbol.to_string(),
            file_path: "src/lib.rs".to_string(),
            workspace_id: "ws_test".to_string(),
            uri: "file:///tmp/ws/src/lib.rs".to_string(),
            module: "src::lib".to_string(),
            symbol_path: format!("src::lib::{symbol}"),
            crate_name: "fixture".to_string(),
            edition: "2021".to_string(),
            signature: signature.to_string(),
            docs: String::new(),
            hover_summary: String::new(),
            body_excerpt: body_excerpt.to_string(),
            start_line: 1,
            start_character: 0,
            end_line: 1,
            span: schema::Span::default(),
            visibility: None,
            generics: None,
            where_clause: None,
            receiver: None,
            return_type: None,
            attrs: Vec::new(),
            cfgs: Vec::new(),
            deprecated: false,
            stability: None,
            body_hash: String::new(),
            symbol_id_stable: id.to_string(),
            semantic_tokens_summary: String::new(),
            syntax_tree_summary: String::new(),
            inlay_hints_summary: String::new(),
        }
    }

    #[test]
    fn explain_term_evidence_applies_threshold_and_ranking() {
        let doc = sample_doc(
            "symbol:ws_test:src/lib.rs:fn:run:1:1",
            "run",
            "pub fn run(rule: &dyn Rule) -> bool {",
            "evaluate_pair(rule, left, right)",
        );
        let terms = vec!["rule".to_string(), "engine".to_string()];
        let mut idf = HashMap::new();
        idf.insert("rule".to_string(), 1.2);
        idf.insert("engine".to_string(), 2.0);

        let (matched, scores, fields) = explain_term_evidence(&doc, &terms, &idf, 2.0);

        assert_eq!(matched, vec!["rule".to_string()]);
        assert!(scores.get("rule").copied().unwrap_or_default() >= 2.0);
        assert!(fields.contains_key("rule"));
        assert!(!fields.contains_key("engine"));
    }

    #[test]
    fn infer_relation_intent_kind_detects_implementations() {
        assert_eq!(
            infer_relation_intent_kind("PositiveRule implements Rule"),
            Some("implementations")
        );
        assert_eq!(
            infer_relation_intent_kind("where is Rule defined"),
            Some("definitions")
        );
        assert_eq!(
            infer_relation_intent_kind("type definition for Rule"),
            Some("type_definitions")
        );
    }

    #[test]
    fn rerank_relation_results_prefers_intent_matching_kind() {
        let mut definition = relation_doc("PositiveRule", "relation:def");
        definition.id = "relation:def".to_string();
        definition.relation_kind = "definitions".to_string();

        let mut implementation = relation_doc("PositiveRule", "relation:impl");
        implementation.id = "relation:impl".to_string();
        implementation.relation_kind = "implementations".to_string();

        let rows = vec![(0.70, definition), (0.66, implementation)];
        let reranked = rerank_relation_results("PositiveRule implements Rule", rows, 2);

        assert_eq!(
            reranked
                .first()
                .map(|(_, item)| item.relation_kind.as_str()),
            Some("implementations")
        );
    }

    #[test]
    fn is_missing_collection_rpc_error_detects_qdrant_message() {
        let msg = "syntax artifact search failed: Datastore error: Error in the response: Some requested entity was not found Not found: Collection `rust_copilot_syntax_artifacts` doesn't exist!";
        assert!(is_missing_collection_rpc_error(msg));
    }
}

/// Implements `explain_relevance` tool.
async fn handle_mcp_explain_relevance(
    app: App,
    req: ExplainRelevanceRequest,
) -> Result<Value, RpcError> {
    let top_k = resolve_limit(req.limit, None, 5);
    let score_threshold = req.score_threshold.unwrap_or(1.2).max(0.0);
    let semantic_samples = (top_k.saturating_mul(4)).clamp(top_k, 64);
    let vector_name = req.vector_name.clone();
    let default_workspace_id = app.config.read().await.workspace_id.clone();

    let mut candidate_scores: HashMap<String, f64> = HashMap::new();
    let mut used_semantic_vectors = Vec::<String>::new();
    let resolved_point_ids =
        if let Some(point_ids) = req.point_ids.clone().filter(|ids| !ids.is_empty()) {
            point_ids
        } else {
            let lexical = lexical_search(
                app.state.read().await.rust_items_by_file.clone(),
                &req.query,
                top_k,
            );

            let (semantic, vectors) = {
                let services = app.services.read().await.clone();
                semantic_search_fused(
                    &services.store,
                    &req.query,
                    vector_name.as_deref(),
                    semantic_samples,
                    &[
                        (SYMBOL_VECTOR_NAME, 0.38),
                        (DOCS_VECTOR_NAME, 0.22),
                        (SIGNATURE_VECTOR_NAME, 0.12),
                        (TYPE_VECTOR_NAME, 0.12),
                        (SEMANTIC_VECTOR_NAME, 0.08),
                        (SYNTAX_VECTOR_NAME, 0.08),
                    ],
                    |item: &schema::SymbolDoc| {
                        matches_filters(item, req.filters.as_ref(), &default_workspace_id)
                    },
                    "explain request",
                    "semantic explain search failed",
                )
                .await?
            };
            used_semantic_vectors = vectors;

            let mut merged: HashMap<String, (f64, schema::SymbolDoc)> = HashMap::new();

            for (score, item) in lexical {
                if matches_filters(&item, req.filters.as_ref(), &default_workspace_id) {
                    merged
                        .entry(item.id.clone())
                        .and_modify(|entry| entry.0 += score * 0.4)
                        .or_insert((score * 0.4, item));
                }
            }

            for (score, item) in semantic {
                merged
                    .entry(item.id.clone())
                    .and_modify(|entry| entry.0 += score * 0.6)
                    .or_insert((score * 0.6, item));
            }

            let mut rows = merged.into_values().collect::<Vec<_>>();
            rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            rows.truncate(top_k);

            rows.into_iter()
                .map(|(score, item)| {
                    candidate_scores.insert(item.id.clone(), score);
                    item.id
                })
                .collect::<Vec<_>>()
        };

    let state = app.state.read().await;
    let (query_terms, ignored_terms) = explain_query_terms(&req.query);

    let mut items_by_id = HashMap::new();
    for items in state.rust_items_by_file.values() {
        for item in items {
            items_by_id.insert(item.id.clone(), item.clone());
        }
    }
    let corpus = items_by_id
        .values()
        .filter(|item| matches_filters(item, req.filters.as_ref(), &default_workspace_id))
        .collect::<Vec<_>>();
    let term_idf = compute_query_term_idf(&query_terms, corpus);

    let mut explanations = Vec::new();

    for point_id in &resolved_point_ids {
        if let Some(item) = items_by_id.get(point_id) {
            let (matched, matched_term_scores, evidence_fields) =
                explain_term_evidence(item, &query_terms, &term_idf, score_threshold);

            explanations.push(json!({
                "point_id": item.id,
                "file_path": item.file_path,
                "item": canonical_symbol_doc(item),
                "score": candidate_scores.get(&item.id).copied(),
                "matched_terms": matched,
                "matched_term_scores": matched_term_scores,
                "evidence_fields": evidence_fields,
                "ignored_query_terms": ignored_terms.clone(),
                "reason": "Item matched lexical terms and/or semantic embedding neighborhood.",
            }));
        }
    }

    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "indexing_in_progress": state.indexing_in_progress,
        "semantic_vector_name": vector_name.unwrap_or_else(|| "fused".to_string()),
        "semantic_vector_names": used_semantic_vectors,
        "score_threshold": score_threshold,
        "resolved_point_ids": resolved_point_ids,
        "explanation": explanations,
    }))
}
