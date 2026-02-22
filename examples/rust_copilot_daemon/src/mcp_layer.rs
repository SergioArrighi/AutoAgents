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
    let top_k = resolve_limit(req.limit, req.top_k, 8);
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
    let top_k = resolve_limit(req.limit, req.top_k, 8);
    let vector_name = req
        .vector_name
        .clone()
        .unwrap_or_else(|| SYMBOL_VECTOR_NAME.to_string());
    let default_workspace_id = app.config.read().await.workspace_id.clone();
    let semantic_samples = (top_k.saturating_mul(4)).clamp(top_k, 64);

    let semantic = {
        let services = app.services.read().await.clone();
        let request = VectorSearchRequest::builder()
            .query(req.query.clone())
            .query_vector_name(vector_name.clone())
            .samples(semantic_samples as u64)
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
        .collect::<Vec<_>>();
    let rows = rerank_relation_results(&req.query, rows, top_k);

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

fn resolve_limit(limit: Option<usize>, top_k: Option<usize>, default: usize) -> usize {
    limit.or(top_k).unwrap_or(default).max(1)
}

fn rerank_relation_results(
    query: &str,
    mut rows: Vec<(f64, SymbolRelationDoc)>,
    top_k: usize,
) -> Vec<(f64, SymbolRelationDoc)> {
    let intent_kind = infer_relation_intent_kind(query);
    let query_tokens = query_tokens(query);

    rows.sort_by(|(left_score, left_item), (right_score, right_item)| {
        let left = relation_relevance_score(*left_score, left_item, intent_kind, &query_tokens);
        let right =
            relation_relevance_score(*right_score, right_item, intent_kind, &query_tokens);
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
    item: &SymbolRelationDoc,
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

fn source_symbol_token_overlap(item: &SymbolRelationDoc, query_tokens: &HashSet<String>) -> bool {
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
                    "item": item,
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
                    "item": item,
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

fn relation_matches_symbol(item: &SymbolRelationDoc, query: &str) -> bool {
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
    corpus: impl IntoIterator<Item = &'a RustItemDoc>,
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
    item: &RustItemDoc,
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

    fn relation_doc(source_symbol: &str, source_symbol_id: &str) -> SymbolRelationDoc {
        SymbolRelationDoc {
            id: "relation:test".to_string(),
            workspace_id: "ws_test".to_string(),
            relation_kind: "references".to_string(),
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

    fn sample_doc(id: &str, symbol: &str, signature: &str, body_excerpt: &str) -> RustItemDoc {
        RustItemDoc {
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

        assert_eq!(reranked.first().map(|(_, item)| item.relation_kind.as_str()), Some("implementations"));
    }
}

/// Implements `explain_relevance` tool.
async fn handle_mcp_explain_relevance(
    app: App,
    req: ExplainRelevanceRequest,
) -> Result<Value, RpcError> {
    let top_k = resolve_limit(req.limit, None, 5);
    let score_threshold = req.score_threshold.unwrap_or(1.2).max(0.0);
    let vector_name = req
        .vector_name
        .clone()
        .unwrap_or_else(|| SYMBOL_VECTOR_NAME.to_string());
    let default_workspace_id = app.config.read().await.workspace_id.clone();

    let mut candidate_scores: HashMap<String, f64> = HashMap::new();
    let resolved_point_ids =
        if let Some(point_ids) = req.point_ids.clone().filter(|ids| !ids.is_empty()) {
            point_ids
        } else {
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
                        message: format!("invalid explain request: {err}"),
                    })?;

                services
                    .store
                    .top_n::<RustItemDoc>(request)
                    .await
                    .map_err(|err| RpcError {
                        code: 500,
                        message: format!("semantic explain search failed: {err}"),
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
                "item": item,
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
        "semantic_vector_name": vector_name,
        "score_threshold": score_threshold,
        "resolved_point_ids": resolved_point_ids,
        "explanation": explanations,
    }))
}
