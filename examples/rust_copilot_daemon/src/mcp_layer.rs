//! MCP-style query plane.
//!
//! This module exposes localhost HTTP endpoints that map to tool-like operations
//! (`search_code`, `get_file_chunks`, `get_symbol_context`, etc.).

use super::*;

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
                    if let Err(err) = handle_http_connection(app_clone, stream).await {
                        eprintln!("mcp connection error: {err}");
                    }
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
                "workspace_metadata",
                "get_file_chunks",
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
        ("POST", "/mcp/tools/get_file_chunks") => {
            match parse_json_body::<GetFileChunksRequest>(body) {
                Ok(req) => handle_mcp_get_file_chunks(app, req).await,
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
    let top_k = req.top_k.unwrap_or(8).max(1);
    let vector_name = req
        .vector_name
        .clone()
        .unwrap_or_else(|| SYMBOL_VECTOR_NAME.to_string());
    let default_workspace_id = app.config.read().await.workspace_id.clone();

    let lexical = lexical_search(
        app.state.read().await.rust_items_by_file.clone(),
        &req.query,
        top_k,
    );

    let semantic = {
        let services = app.services.read().await.clone();
        let request = VectorSearchRequest::builder()
            .query(req.query.clone())
            .query_vector_name(vector_name.clone())
            .samples(top_k as u64)
            .build()
            .map_err(|err| RpcError {
                code: 400,
                message: format!("invalid search request: {err}"),
            })?;

        services
            .store
            .top_n::<RustItemDoc>(request)
            .await
            .map_err(|err| RpcError {
                code: 500,
                message: format!("semantic search failed: {err}"),
            })?
    };

    let mut merged: HashMap<String, (f64, RustItemDoc)> = HashMap::new();

    for (score, item) in lexical {
        if matches_filters(&item, req.filters.as_ref(), &default_workspace_id) {
            merged
                .entry(item.id.clone())
                .and_modify(|entry| entry.0 += score * 0.4)
                .or_insert((score * 0.4, item));
        }
    }

    for (score, _, item) in semantic {
        if matches_filters(&item, req.filters.as_ref(), &default_workspace_id) {
            merged
                .entry(item.id.clone())
                .and_modify(|entry| entry.0 += score * 0.6)
                .or_insert((score * 0.6, item));
        }
    }

    let mut rows = merged.into_values().collect::<Vec<_>>();
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    rows.truncate(top_k);

    let state = app.state.read().await;

    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "indexing_in_progress": state.indexing_in_progress,
        "semantic_vector_name": vector_name,
        "results": rows
            .into_iter()
            .map(|(score, item)| json!({
                "score": score,
                "item": item,
            }))
            .collect::<Vec<_>>(),
    }))
}

/// Implements `search_relations` using semantic ranking from relation store.
async fn handle_mcp_search_relations(
    app: App,
    req: SearchRelationsRequest,
) -> Result<Value, RpcError> {
    let top_k = req.top_k.unwrap_or(8).max(1);
    let vector_name = req
        .vector_name
        .clone()
        .unwrap_or_else(|| SYMBOL_VECTOR_NAME.to_string());
    let default_workspace_id = app.config.read().await.workspace_id.clone();

    let semantic = {
        let services = app.services.read().await.clone();
        let request = VectorSearchRequest::builder()
            .query(req.query.clone())
            .query_vector_name(vector_name.clone())
            .samples(top_k as u64)
            .build()
            .map_err(|err| RpcError {
                code: 400,
                message: format!("invalid relation search request: {err}"),
            })?;

        services
            .relation_store
            .top_n::<SymbolRelationDoc>(request)
            .await
            .map_err(|err| RpcError {
                code: 500,
                message: format!("semantic relation search failed: {err}"),
            })?
    };

    let rows = semantic
        .into_iter()
        .filter_map(|(score, _, item)| {
            if matches_relation_filters(&item, req.filters.as_ref(), &default_workspace_id) {
                Some((score, item))
            } else {
                None
            }
        })
        .take(top_k)
        .collect::<Vec<_>>();

    let state = app.state.read().await;
    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "indexing_in_progress": state.indexing_in_progress,
        "semantic_vector_name": vector_name,
        "results": rows
            .into_iter()
            .map(|(score, item)| json!({
                "score": score,
                "item": item,
            }))
            .collect::<Vec<_>>(),
    }))
}

/// Lexical scoring component used by `search_code`.
pub(super) fn lexical_search(
    items_by_file: HashMap<String, Vec<RustItemDoc>>,
    query: &str,
    top_k: usize,
) -> Vec<(f64, RustItemDoc)> {
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

/// Applies optional search filters to a chunk.
fn matches_filters(
    item: &RustItemDoc,
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
    item: &SymbolRelationDoc,
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
        "items": items,
    }))
}

/// Implements `get_symbol_context` tool.
async fn handle_mcp_get_symbol_context(
    app: App,
    req: SymbolContextRequest,
) -> Result<Value, RpcError> {
    let needle = req.symbol.to_lowercase();
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
                    "item": item,
                }));
            }
        }
    }

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
    let needle = req.symbol.to_lowercase();
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
            let search_text = relation_search_text(item).to_lowercase();
            if search_text.contains(&needle) {
                relations.push(json!({
                    "file_path": file_path,
                    "item": item,
                }));
            }
        }
    }

    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "indexing_in_progress": state.indexing_in_progress,
        "relations": relations,
    }))
}

/// Implements `explain_relevance` tool.
async fn handle_mcp_explain_relevance(
    app: App,
    req: ExplainRelevanceRequest,
) -> Result<Value, RpcError> {
    let state = app.state.read().await;
    let query_tokens = req
        .query
        .to_lowercase()
        .split_whitespace()
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>();

    let mut explanations = Vec::new();

    for point_id in req.point_ids {
        let mut maybe_item: Option<RustItemDoc> = None;
        for items in state.rust_items_by_file.values() {
            if let Some(found) = items.iter().find(|item| item.id == point_id) {
                maybe_item = Some(found.clone());
                break;
            }
        }

        if let Some(item) = maybe_item {
            let lower = rust_item_search_text(&item).to_lowercase();
            let matched = query_tokens
                .iter()
                .filter(|token| lower.contains(token.as_str()))
                .cloned()
                .collect::<Vec<_>>();

            explanations.push(json!({
                "point_id": item.id,
                "file_path": item.file_path,
                "item": item,
                "matched_terms": matched,
                "reason": "Item matched lexical terms and/or semantic embedding neighborhood.",
            }));
        }
    }

    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "indexing_in_progress": state.indexing_in_progress,
        "explanation": explanations,
    }))
}
