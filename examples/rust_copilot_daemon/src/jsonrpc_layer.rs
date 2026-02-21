//! JSON-RPC sync plane.
//!
//! This module handles high-frequency ingestion/synchronization events over stdio
//! (workspace scans, file changes, status, lifecycle).

use super::*;

/// Runs newline-delimited JSON-RPC over stdin/stdout until a stop signal.
pub(super) async fn run_jsonrpc_stdio(app: App, mut stop_rx: watch::Receiver<bool>) -> Result<()> {
    let mut lines = BufReader::new(io::stdin()).lines();
    let stdout = Arc::new(Mutex::new(io::stdout()));

    loop {
        tokio::select! {
            changed = stop_rx.changed() => {
                if changed.is_ok() && *stop_rx.borrow() {
                    break;
                }
            }
            line = lines.next_line() => {
                let Some(line) = line.context("failed reading stdin line")? else {
                    break;
                };

                if line.trim().is_empty() {
                    continue;
                }

                let req: RpcRequest = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(err) => {
                        eprintln!("invalid json-rpc payload: {err}");
                        continue;
                    }
                };

                let response = handle_rpc_request(app.clone(), req).await;
                if let Some(resp) = response {
                    // newline-delimited JSON makes extension-side framing easy.
                    let mut out = stdout.lock().await;
                    out.write_all(serde_json::to_string(&resp)?.as_bytes()).await?;
                    out.write_all(b"\n").await?;
                    out.flush().await?;
                }
            }
        }
    }

    Ok(())
}

/// Dispatches one JSON-RPC request into method-specific handlers.
async fn handle_rpc_request(app: App, req: RpcRequest) -> Option<RpcResponse> {
    let id = req.id.clone().unwrap_or(Value::Null);

    let result = match req.method.as_str() {
        "initialize" => {
            let parsed: InitializeParams = parse_params(req.params)?;
            handle_initialize(app, parsed).await
        }
        "scan.full" => {
            let parsed: ScanFullParams = parse_params(req.params)?;
            handle_scan_full(app, parsed).await
        }
        "file.changed" => {
            let parsed: FileChangedParams = parse_params(req.params)?;
            handle_file_changed(app, parsed).await
        }
        "file.deleted" => {
            let parsed: FileDeletedParams = parse_params(req.params)?;
            handle_file_deleted(app, parsed).await
        }
        "workspace.renamed" => {
            let parsed: WorkspaceRenamedParams = parse_params(req.params)?;
            handle_workspace_renamed(app, parsed).await
        }
        "status" => Ok(json!(app.state.read().await.status())),
        "shutdown" => {
            let _ = app.stop_tx.send(true);
            Ok(json!({ "ok": true }))
        }
        unknown => Err(RpcError {
            code: -32601,
            message: format!("unknown method: {unknown}"),
        }),
    };

    req.id.as_ref()?;

    match result {
        Ok(value) => Some(RpcResponse {
            jsonrpc: "2.0",
            id,
            result: Some(value),
            error: None,
        }),
        Err(error) => Some(RpcResponse {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(error),
        }),
    }
}

/// Deserializes generic JSON params into the requested type.
fn parse_params<T: for<'de> Deserialize<'de>>(params: Option<Value>) -> Option<T> {
    let raw = params.unwrap_or(Value::Object(Default::default()));
    serde_json::from_value(raw).ok()
}

/// Handles `initialize`: applies runtime overrides and rebuilds service clients.
async fn handle_initialize(app: App, params: InitializeParams) -> Result<Value, RpcError> {
    let mut workspace_changed = false;
    {
        let mut cfg = app.config.write().await;
        if let Some(root) = params.workspace_root {
            let new_root = PathBuf::from(root);
            if cfg.workspace_root != new_root {
                workspace_changed = true;
            }
            cfg.workspace_root = new_root;
            cfg.workspace_id = derive_workspace_id(&cfg.workspace_root);
        }
        if let Some(qdrant_url) = params.qdrant_url {
            cfg.qdrant_url = qdrant_url;
        }
        if let Some(collection) = params.collection {
            cfg.qdrant_collection = collection;
        }
        if let Some(relation_collection) = params.relation_collection {
            cfg.qdrant_relation_collection = relation_collection;
        }
        if let Some(config) = params.config {
            if let Some(v) = config.ollama_base_url {
                cfg.ollama_base_url = v;
            }
            if let Some(v) = config.llm_model {
                cfg.llm_model = v;
            }
            if let Some(v) = config.embedding_model {
                cfg.embedding_model = v;
            }
            if let Some(v) = config.qdrant_api_key {
                cfg.qdrant_api_key = Some(v);
            }
            if let Some(v) = config.workspace_id {
                if cfg.workspace_id != v {
                    workspace_changed = true;
                }
                cfg.workspace_id = v;
            }
        }

        let rebuilt = build_services(&cfg).await.map_err(|err| RpcError {
            code: -32001,
            message: format!("failed to rebuild services: {err}"),
        })?;
        *app.services.write().await = rebuilt;
    }

    {
        let mut state = app.state.write().await;
        state.is_ra_warm = false;
    }

    if workspace_changed {
        let mut state = app.state.write().await;
        state.rust_items_by_file.clear();
        state.relations_by_file.clear();
        state.chunks_by_file.clear();
        state.indexed_ids_by_file.clear();
        state.indexed_relation_ids_by_file.clear();
        state.workspace_metadata = None;
        state.metadata_docs_by_crate.clear();
        state.indexed_metadata_ids.clear();
        state.extraction_metrics = ExtractionMetrics::default();
        state.is_ra_warm = false;
        state.queue_depth = 0;
        state.indexing_in_progress = false;
        state.last_error = None;
        drop(state);
        app.ra_manager.lock().await.reset().await;
    }

    if let Err(err) = refresh_workspace_metadata(&app).await {
        eprintln!("workspace metadata refresh failed: {err:#}");
    }

    Ok(json!({
        "protocol_version": PROTOCOL_VERSION,
        "daemon_version": DAEMON_VERSION,
        "schema_version": SCHEMA_VERSION,
        "workspace_id": app.config.read().await.workspace_id,
    }))
}

/// Handles `scan.full`: expands file set and enqueues indexing events.
async fn handle_scan_full(app: App, params: ScanFullParams) -> Result<Value, RpcError> {
    let cfg = app.config.read().await.clone();
    let include = params.globs.unwrap_or_else(|| vec!["**/*.rs".to_string()]);
    let exclude = params.exclude.unwrap_or_default();
    let respect_gitignore = params.respect_gitignore.unwrap_or(true);

    let files = list_workspace_files(&cfg.workspace_root, &include, &exclude, respect_gitignore)
        .map_err(|err| RpcError {
            code: -32002,
            message: format!("scan.full failed: {err}"),
        })?;

    let mut enqueued = 0usize;
    for file in files {
        if app.dirty_tx.send(DirtyEvent::Index(file)).await.is_ok() {
            enqueued += 1;
        }
    }

    {
        let mut state = app.state.write().await;
        state.queue_depth = state.queue_depth.saturating_add(enqueued);
    }

    Ok(json!({"enqueued": enqueued}))
}

/// Handles `file.changed`: enqueues a single reindex event.
async fn handle_file_changed(app: App, params: FileChangedParams) -> Result<Value, RpcError> {
    let path = resolve_path(&app.config.read().await.workspace_root, &params.path);
    app.dirty_tx
        .send(DirtyEvent::Index(path))
        .await
        .map_err(|_| RpcError {
            code: -32003,
            message: "index queue unavailable".to_string(),
        })?;

    let mut state = app.state.write().await;
    state.queue_depth = state.queue_depth.saturating_add(1);

    Ok(json!({"queued": true}))
}

/// Handles `file.deleted`: enqueues file removal from index/cache.
async fn handle_file_deleted(app: App, params: FileDeletedParams) -> Result<Value, RpcError> {
    let path = resolve_path(&app.config.read().await.workspace_root, &params.path);

    app.dirty_tx
        .send(DirtyEvent::Delete(path))
        .await
        .map_err(|_| RpcError {
            code: -32003,
            message: "index queue unavailable".to_string(),
        })?;

    let mut state = app.state.write().await;
    state.queue_depth = state.queue_depth.saturating_add(1);

    Ok(json!({"queued": true}))
}

/// Handles `workspace.renamed`: updates known workspace root.
async fn handle_workspace_renamed(
    app: App,
    params: WorkspaceRenamedParams,
) -> Result<Value, RpcError> {
    let mut cfg = app.config.write().await;
    if cfg.workspace_root == Path::new(&params.old_path) {
        cfg.workspace_root = PathBuf::from(params.new_path);
        cfg.workspace_id = derive_workspace_id(&cfg.workspace_root);
        drop(cfg);
        app.ra_manager.lock().await.reset().await;
        if let Err(err) = refresh_workspace_metadata(&app).await {
            eprintln!("workspace metadata refresh failed after rename: {err:#}");
        }
        return Ok(json!({"ok": true}));
    }

    Ok(json!({"ok": true}))
}
