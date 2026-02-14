use anyhow::{Context, Result};
use autoagents_core::embeddings::{Embed, EmbedError, TextEmbedder};
use autoagents_core::vector_store::request::VectorSearchRequest;
use autoagents_core::vector_store::{NamedVectorDocument, VectorStoreIndex};
use autoagents_llm::backends::ollama::Ollama;
use autoagents_llm::embedding::EmbeddingBuilder;
use autoagents_qdrant::QdrantVectorStore;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{self, AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{Mutex, RwLock, mpsc, watch};
use walkdir::WalkDir;

mod jsonrpc_layer;
mod mcp_layer;

/// JSON-RPC protocol version returned by the sync plane.
const PROTOCOL_VERSION: &str = "0.1.0";
/// Daemon package version, sourced from Cargo metadata.
const DAEMON_VERSION: &str = env!("CARGO_PKG_VERSION");
/// Schema contract version returned by query-plane responses.
const SCHEMA_VERSION: &str = "2026-02-10";
const SYMBOL_VECTOR_NAME: &str = "symbol";
const DOCS_VECTOR_NAME: &str = "docs";
const SIGNATURE_VECTOR_NAME: &str = "signature";
const BODY_VECTOR_NAME: &str = "body";
const LSP_REQUEST_TIMEOUT: Duration = Duration::from_secs(6);

/// Runtime configuration shared across JSON-RPC and MCP layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RuntimeConfig {
    /// Workspace root used for scans and relative path normalization.
    workspace_root: PathBuf,
    /// Stable workspace identity used for multi-repo isolation.
    workspace_id: String,
    /// Qdrant endpoint URL.
    qdrant_url: String,
    /// Qdrant collection name for code chunks.
    qdrant_collection: String,
    /// Optional Qdrant API key.
    qdrant_api_key: Option<String>,
    /// Ollama endpoint URL.
    ollama_base_url: String,
    /// Chat/completion model name.
    llm_model: String,
    /// Embedding model name.
    embedding_model: String,
}

impl Default for RuntimeConfig {
    /// Builds config from environment with local-development defaults.
    fn default() -> Self {
        let workspace_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self {
            workspace_id: derive_workspace_id(&workspace_root),
            workspace_root,
            qdrant_url: std::env::var("QDRANT_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:6334".to_string()),
            qdrant_collection: std::env::var("QDRANT_COLLECTION")
                .unwrap_or_else(|_| "rust_copilot_chunks".to_string()),
            qdrant_api_key: std::env::var("QDRANT_API_KEY").ok(),
            ollama_base_url: std::env::var("OLLAMA_BASE_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string()),
            llm_model: "gpt-oss:20b".to_string(),
            embedding_model: "dengcao/Qwen3-Embedding-8B:Q4_K_M".to_string(),
        }
    }
}

/// Indexed code chunk stored in Qdrant and cached in memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CodeChunk {
    /// Stable logical id (`file:start:end`).
    chunk_id: String,
    /// Workspace-relative file path.
    file_path: String,
    /// Workspace identity for query-time isolation.
    workspace_id: String,
    /// Absolute file URI (`file://...`) for editor/LSP navigation.
    uri: String,
    /// Inclusive chunk start line (1-based).
    start_line: usize,
    /// Inclusive chunk end line (1-based).
    end_line: usize,
    /// Lightweight inferred kind (`function`, `struct`, ...).
    kind: String,
    /// Chunk hash used for diagnostics/change tracking.
    hash: String,
    /// Raw chunk text.
    text: String,
}

impl Embed for CodeChunk {
    /// Embeds chunk metadata + text for semantic retrieval.
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.workspace_id.clone());
        embedder.embed(self.uri.clone());
        embedder.embed(self.file_path.clone());
        embedder.embed(self.kind.clone());
        embedder.embed(self.text.clone());
        Ok(())
    }
}

/// Symbol-aware document extracted from Rust source.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RustItemDoc {
    /// Stable id used for incremental upserts/deletes.
    id: String,
    /// Item kind: fn, struct, impl, trait, mod, enum, type, const, macro.
    kind: String,
    /// Best-effort symbol name (or fully qualified shape for impl blocks).
    symbol: String,
    /// Workspace-relative path.
    file_path: String,
    /// Workspace identity for query-time isolation.
    workspace_id: String,
    /// Absolute file URI (`file://...`) for editor/LSP navigation.
    uri: String,
    /// Best-effort module path from file path.
    module: String,
    /// Signature/header line.
    signature: String,
    /// Joined doc comments directly above the item.
    docs: String,
    /// Hover payload extracted from rust-analyzer.
    hover_summary: String,
    /// Signature help summary extracted from rust-analyzer.
    signature_help: String,
    /// Short source excerpt for additional context.
    body_excerpt: String,
    /// Inclusive start line (1-based).
    start_line: usize,
    /// Inclusive end line (1-based).
    end_line: usize,
}

impl Embed for RustItemDoc {
    /// Build embeddings from one canonical search text representation.
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(rust_item_search_text(self));
        Ok(())
    }
}

/// Snapshot returned by status endpoints.
#[derive(Debug, Clone, Serialize)]
struct StatusSnapshot {
    /// JSON-RPC protocol version.
    protocol_version: String,
    /// Daemon binary version.
    daemon_version: String,
    /// Query schema version.
    schema_version: String,
    /// Count of pending indexing events.
    queue_depth: usize,
    /// Whether indexing worker is currently processing a batch.
    indexing_in_progress: bool,
    /// Number of files represented in local in-memory cache.
    total_files: usize,
    /// Total number of chunks represented in local in-memory cache.
    total_chunks: usize,
    /// Last successful indexing timestamp (unix millis).
    indexed_at_unix_ms: Option<u128>,
    /// Last indexing error, if any.
    last_error: Option<String>,
}

/// Mutable daemon state shared by both planes.
#[derive(Default)]
struct State {
    /// Local in-memory view of indexed chunks by file path.
    chunks_by_file: HashMap<String, Vec<CodeChunk>>,
    /// Local in-memory view of indexed symbol docs by file path.
    rust_items_by_file: HashMap<String, Vec<RustItemDoc>>,
    /// Count of pending indexing events.
    queue_depth: usize,
    /// Whether worker is currently indexing.
    indexing_in_progress: bool,
    /// Last indexing completion timestamp.
    indexed_at_unix_ms: Option<u128>,
    /// Last indexing failure message.
    last_error: Option<String>,
    /// Currently indexed logical IDs by file, used for incremental cleanup.
    indexed_ids_by_file: HashMap<String, HashSet<String>>,
}

impl State {
    /// Converts internal state into a serializable status payload.
    fn status(&self) -> StatusSnapshot {
        let total_chunks = self
            .rust_items_by_file
            .values()
            .map(std::vec::Vec::len)
            .sum::<usize>();

        StatusSnapshot {
            protocol_version: PROTOCOL_VERSION.to_string(),
            daemon_version: DAEMON_VERSION.to_string(),
            schema_version: SCHEMA_VERSION.to_string(),
            queue_depth: self.queue_depth,
            indexing_in_progress: self.indexing_in_progress,
            total_files: self.rust_items_by_file.len(),
            total_chunks,
            indexed_at_unix_ms: self.indexed_at_unix_ms,
            last_error: self.last_error.clone(),
        }
    }
}

/// External services used by indexing/query flows.
#[derive(Clone)]
struct Services {
    /// Shared vector store client.
    store: QdrantVectorStore,
}

/// Root application context injected into all tasks and handlers.
#[derive(Clone)]
struct App {
    // Shared runtime config/state/services used by both planes.
    /// Runtime configuration.
    config: Arc<RwLock<RuntimeConfig>>,
    /// Mutable daemon state.
    state: Arc<RwLock<State>>,
    /// Bound service clients.
    services: Arc<RwLock<Services>>,
    /// Shared persistent rust-analyzer session manager.
    ra_manager: Arc<Mutex<RaSessionManager>>,
    /// Work queue sender for indexing operations.
    dirty_tx: mpsc::Sender<DirtyEvent>,
    /// Global stop signal broadcaster.
    stop_tx: watch::Sender<bool>,
}

/// Unit of indexing work.
#[derive(Debug, Clone)]
enum DirtyEvent {
    /// Re-index a file.
    Index(PathBuf),
    /// Delete file chunks from in-memory view.
    Delete(PathBuf),
}

struct RaSessionManager {
    workspace_root: Option<PathBuf>,
    session: Option<RaLspSession>,
}

impl RaSessionManager {
    fn new() -> Self {
        Self {
            workspace_root: None,
            session: None,
        }
    }

    async fn reset(&mut self) {
        if let Some(mut session) = self.session.take() {
            let _ = session.shutdown().await;
        }
        self.workspace_root = None;
    }

    async fn ensure_session(&mut self, workspace_root: &Path) -> Result<&mut RaLspSession> {
        if self.workspace_root.as_deref() != Some(workspace_root) {
            self.reset().await;
        }

        if self.session.is_none() {
            self.session = Some(RaLspSession::start(workspace_root).await?);
            self.workspace_root = Some(workspace_root.to_path_buf());
        }

        self.session
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("rust-analyzer session unavailable"))
    }

    async fn extract_symbol_docs(
        &mut self,
        workspace_root: &Path,
        workspace_id: &str,
        file_path_abs: &Path,
        file_path_rel: &str,
        file_uri: &str,
        content: &str,
    ) -> Result<Vec<RustItemDoc>> {
        let file_uri = if file_uri.is_empty() {
            path_to_file_uri(file_path_abs)?
        } else {
            file_uri.to_string()
        };

        let session = self.ensure_session(workspace_root).await?;
        session.sync_document(&file_uri, content).await?;
        let result = session
            .request(
                "textDocument/documentSymbol",
                json!({
                    "textDocument": { "uri": file_uri }
                }),
            )
            .await?;
        let mut docs = parse_lsp_symbols(result, workspace_id, file_path_rel, &file_uri, content);
        enrich_docs_with_lsp_metadata(session, &file_uri, &mut docs).await?;
        Ok(docs)
    }

    async fn close_document(&mut self, workspace_root: &Path, file_uri: &str) {
        if self.workspace_root.as_deref() != Some(workspace_root) {
            return;
        }
        if let Some(session) = self.session.as_mut() {
            let _ = session.close_document(file_uri).await;
        }
    }
}

struct RaLspSession {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
    next_id: i64,
    open_doc_versions: HashMap<String, i64>,
}

impl RaLspSession {
    async fn start(workspace_root: &Path) -> Result<Self> {
        let mut child = Command::new("rust-analyzer")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("failed to spawn rust-analyzer process")?;

        let stdin = child
            .stdin
            .take()
            .context("failed to acquire rust-analyzer stdin")?;
        let stdout = child
            .stdout
            .take()
            .context("failed to acquire rust-analyzer stdout")?;
        let mut session = Self {
            child,
            stdin,
            reader: BufReader::new(stdout),
            next_id: 1,
            open_doc_versions: HashMap::new(),
        };

        let workspace_uri = path_to_file_uri(workspace_root)?;
        let _ = session
            .request(
                "initialize",
                json!({
                    "processId": null,
                    "rootUri": workspace_uri,
                    "capabilities": {}
                }),
            )
            .await?;

        session.notify("initialized", json!({})).await?;
        Ok(session)
    }

    async fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        write_lsp_message(
            &mut self.stdin,
            &json!({
                "jsonrpc":"2.0",
                "method": method,
                "params": params,
            })
            .to_string(),
        )
        .await
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let req_id = self.next_id;
        self.next_id += 1;

        tokio::time::timeout(LSP_REQUEST_TIMEOUT, async {
            write_lsp_message(
                &mut self.stdin,
                &json!({
                    "jsonrpc":"2.0",
                    "id": req_id,
                    "method": method,
                    "params": params,
                })
                .to_string(),
            )
            .await?;
            let response = read_lsp_response_for_id(&mut self.reader, req_id).await?;
            Ok::<Value, anyhow::Error>(response.get("result").cloned().unwrap_or(Value::Null))
        })
        .await
        .map_err(|_| anyhow::anyhow!("rust-analyzer request timed out: {method}"))?
    }

    async fn sync_document(&mut self, file_uri: &str, content: &str) -> Result<()> {
        if let Some(current) = self.open_doc_versions.get(file_uri).copied() {
            let new_version = current + 1;
            self.open_doc_versions
                .insert(file_uri.to_string(), new_version);
            self.notify(
                "textDocument/didChange",
                json!({
                    "textDocument": {
                        "uri": file_uri,
                        "version": new_version,
                    },
                    "contentChanges": [{
                        "text": content
                    }]
                }),
            )
            .await?;
            return Ok(());
        }

        self.open_doc_versions.insert(file_uri.to_string(), 1);
        self.notify(
            "textDocument/didOpen",
            json!({
                "textDocument": {
                    "uri": file_uri,
                    "languageId": "rust",
                    "version": 1,
                    "text": content
                }
            }),
        )
        .await
    }

    async fn close_document(&mut self, file_uri: &str) -> Result<()> {
        if self.open_doc_versions.remove(file_uri).is_none() {
            return Ok(());
        }
        self.notify(
            "textDocument/didClose",
            json!({
                "textDocument": { "uri": file_uri }
            }),
        )
        .await
    }

    async fn shutdown(&mut self) -> Result<()> {
        let _ = self.request("shutdown", Value::Null).await;
        let _ = self.notify("exit", Value::Null).await;
        let _ = tokio::time::timeout(Duration::from_secs(2), self.child.wait()).await;
        Ok(())
    }
}

/// JSON-RPC request envelope.
#[derive(Debug, Deserialize)]
struct RpcRequest {
    #[allow(dead_code)]
    jsonrpc: Option<String>,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

/// JSON-RPC response envelope.
#[derive(Debug, Serialize)]
struct RpcResponse {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<RpcError>,
}

/// JSON-RPC error object.
#[derive(Debug, Serialize)]
struct RpcError {
    code: i64,
    message: String,
}

/// `initialize` params for JSON-RPC sync plane.
#[derive(Debug, Deserialize)]
struct InitializeParams {
    #[serde(rename = "workspaceRoot")]
    workspace_root: Option<String>,
    #[serde(rename = "qdrantUrl")]
    qdrant_url: Option<String>,
    collection: Option<String>,
    config: Option<InitializeConfig>,
}

/// Optional nested config overrides for `initialize`.
#[derive(Debug, Deserialize)]
struct InitializeConfig {
    #[serde(rename = "ollamaBaseUrl")]
    ollama_base_url: Option<String>,
    #[serde(rename = "llmModel")]
    llm_model: Option<String>,
    #[serde(rename = "embeddingModel")]
    embedding_model: Option<String>,
    #[serde(rename = "qdrantApiKey")]
    qdrant_api_key: Option<String>,
    #[serde(rename = "workspaceId")]
    workspace_id: Option<String>,
}

/// `scan.full` params.
#[derive(Debug, Deserialize)]
struct ScanFullParams {
    globs: Option<Vec<String>>,
    exclude: Option<Vec<String>>,
    #[serde(rename = "respectGitignore")]
    respect_gitignore: Option<bool>,
}

/// `file.changed` params.
#[derive(Debug, Deserialize)]
struct FileChangedParams {
    path: String,
    #[allow(dead_code)]
    version: Option<String>,
    #[allow(dead_code)]
    reason: Option<String>,
}

/// `file.deleted` params.
#[derive(Debug, Deserialize)]
struct FileDeletedParams {
    path: String,
}

/// `workspace.renamed` params.
#[derive(Debug, Deserialize)]
struct WorkspaceRenamedParams {
    #[serde(rename = "oldPath")]
    old_path: String,
    #[serde(rename = "newPath")]
    new_path: String,
}

/// `search_code` request body.
#[derive(Debug, Deserialize)]
struct SearchCodeRequest {
    query: String,
    top_k: Option<usize>,
    vector_name: Option<String>,
    filters: Option<SearchFilters>,
}

/// Optional filters accepted by `search_code`.
#[derive(Debug, Deserialize)]
struct SearchFilters {
    file_path: Option<String>,
    kind: Option<String>,
    workspace_id: Option<String>,
}

/// `get_file_chunks` request body.
#[derive(Debug, Deserialize)]
struct GetFileChunksRequest {
    file_path: String,
}

/// `get_symbol_context` request body.
#[derive(Debug, Deserialize)]
struct SymbolContextRequest {
    symbol: String,
    file_path: Option<String>,
}

/// `explain_relevance` request body.
#[derive(Debug, Deserialize)]
struct ExplainRelevanceRequest {
    query: String,
    point_ids: Vec<String>,
}

/// Entry point: starts shared services and both communication planes.
#[tokio::main]
async fn main() -> Result<()> {
    let config = RuntimeConfig::default();
    let services = build_services(&config)
        .await
        .context("failed to initialize Ollama embedding + Qdrant clients")?;

    let (dirty_tx, dirty_rx) = mpsc::channel::<DirtyEvent>(4096);
    let (stop_tx, stop_rx) = watch::channel(false);

    let app = App {
        config: Arc::new(RwLock::new(config)),
        state: Arc::new(RwLock::new(State::default())),
        services: Arc::new(RwLock::new(services)),
        ra_manager: Arc::new(Mutex::new(RaSessionManager::new())),
        dirty_tx,
        stop_tx: stop_tx.clone(),
    };

    let mcp_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("failed to bind MCP endpoint")?;
    let mcp_addr = mcp_listener.local_addr()?;

    eprintln!(
        "rust-copilot-daemon started: jsonrpc=stdio mcp=http://{} schema_version={}",
        mcp_addr, SCHEMA_VERSION
    );

    // Run both planes in one process:
    // - stdio JSON-RPC for extension->daemon sync events
    // - localhost HTTP MCP-style tools for agent/chat queries
    let indexer_task = tokio::spawn(indexer_loop(app.clone(), dirty_rx, stop_rx.clone()));
    let mcp_task = tokio::spawn(mcp_layer::run_mcp_server(
        app.clone(),
        mcp_listener,
        stop_rx.clone(),
    ));
    let rpc_task = tokio::spawn(jsonrpc_layer::run_jsonrpc_stdio(app.clone(), stop_rx));

    let rpc_result = rpc_task.await.context("jsonrpc task join failed")?;
    rpc_result?;

    let _ = stop_tx.send(true);

    indexer_task.abort();
    mcp_task.abort();

    Ok(())
}

/// Initializes external clients (Ollama embeddings + Qdrant vector store).
async fn build_services(config: &RuntimeConfig) -> Result<Services> {
    let provider = EmbeddingBuilder::<Ollama>::new()
        .base_url(&config.ollama_base_url)
        .model(&config.embedding_model)
        .build()
        .context("failed to build Ollama embedding client")?;

    let store = if let Some(api_key) = &config.qdrant_api_key {
        QdrantVectorStore::with_api_key(
            provider,
            config.qdrant_url.clone(),
            config.qdrant_collection.clone(),
            Some(api_key.clone()),
        )
    } else {
        QdrantVectorStore::new(
            provider,
            config.qdrant_url.clone(),
            config.qdrant_collection.clone(),
        )
    }
    .context("failed to initialize Qdrant store")?;

    Ok(Services { store })
}

/// Background worker that debounces and batches indexing events.
async fn indexer_loop(
    app: App,
    mut rx: mpsc::Receiver<DirtyEvent>,
    mut stop_rx: watch::Receiver<bool>,
) {
    // Debounce lets us absorb file-change storms and batch updates.
    let debounce = Duration::from_millis(250);

    loop {
        tokio::select! {
            changed = stop_rx.changed() => {
                if changed.is_ok() && *stop_rx.borrow() {
                    break;
                }
            }
            maybe_first = rx.recv() => {
                let Some(first) = maybe_first else {
                    break;
                };

                let mut batch: HashMap<PathBuf, DirtyEvent> = HashMap::new();
                insert_batch_event(&mut batch, first);

                let timer = tokio::time::sleep(debounce);
                tokio::pin!(timer);

                loop {
                    tokio::select! {
                        changed = stop_rx.changed() => {
                            if changed.is_ok() && *stop_rx.borrow() {
                                return;
                            }
                        }
                        _ = &mut timer => {
                            break;
                        }
                        maybe_event = rx.recv() => {
                            match maybe_event {
                                Some(event) => insert_batch_event(&mut batch, event),
                                None => break,
                            }
                        }
                    }
                }

                process_batch(app.clone(), batch.into_values().collect()).await;
            }
        }
    }
}

/// Inserts/overwrites an event in the current batch keyed by file path.
fn insert_batch_event(batch: &mut HashMap<PathBuf, DirtyEvent>, event: DirtyEvent) {
    // Last-write-wins per path inside a batch.
    match &event {
        DirtyEvent::Index(path) | DirtyEvent::Delete(path) => {
            batch.insert(path.clone(), event);
        }
    }
}

/// Processes one debounced batch and updates status state.
async fn process_batch(app: App, events: Vec<DirtyEvent>) {
    {
        let mut state = app.state.write().await;
        state.indexing_in_progress = true;
        state.queue_depth = state.queue_depth.saturating_sub(events.len());
    }

    let services = app.services.read().await.clone();

    for event in events {
        let result = match event {
            DirtyEvent::Index(path) => reindex_file(&app, &services, &path).await,
            DirtyEvent::Delete(path) => delete_file_chunks(&app, &path).await,
        };

        if let Err(err) = result {
            let mut state = app.state.write().await;
            state.last_error = Some(format!("{err:#}"));
        }
    }

    let mut state = app.state.write().await;
    state.indexing_in_progress = false;
    state.indexed_at_unix_ms = Some(unix_ms_now());
}

/// Reads and re-indexes one Rust file into Qdrant and local cache.
async fn reindex_file(app: &App, services: &Services, path: &Path) -> Result<()> {
    if !path.is_file() {
        return Ok(());
    }

    if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
        return Ok(());
    }

    let content = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("failed reading file {}", path.display()))?;

    let cfg = app.config.read().await.clone();
    let workspace_id = cfg.workspace_id.clone();
    let file_uri = path_to_file_uri(path)?;
    let file_rel = path
        .strip_prefix(&cfg.workspace_root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string();

    let previous_ids = app
        .state
        .read()
        .await
        .indexed_ids_by_file
        .get(&file_rel)
        .cloned()
        .unwrap_or_default();

    let coarse_chunks = chunk_rust_file(&workspace_id, &file_rel, &file_uri, &content);
    if coarse_chunks.is_empty() {
        if !previous_ids.is_empty() {
            let stale_ids = previous_ids.into_iter().collect::<Vec<_>>();
            services
                .store
                .delete_documents_by_ids(&stale_ids)
                .await
                .with_context(|| format!("qdrant delete failed for {}", file_rel))?;
        }
        let mut state = app.state.write().await;
        state.rust_items_by_file.remove(&file_rel);
        state.chunks_by_file.remove(&file_rel);
        state.indexed_ids_by_file.remove(&file_rel);
        return Ok(());
    }

    // Symbol-aware enrichment stage. When symbols are available, they replace
    // coarse chunk fallback docs for the same file.
    let symbol_docs = match app
        .ra_manager
        .lock()
        .await
        .extract_symbol_docs(
            &cfg.workspace_root,
            &workspace_id,
            path,
            &file_rel,
            &file_uri,
            &content,
        )
        .await
    {
        Ok(docs) if !docs.is_empty() => docs,
        Ok(_) => extract_symbol_docs_heuristic(&workspace_id, &file_rel, &file_uri, &content),
        Err(err) => {
            eprintln!(
                "rust-analyzer enrichment failed for {}: {err:#}. Falling back to heuristic extraction",
                file_rel
            );
            extract_symbol_docs_heuristic(&workspace_id, &file_rel, &file_uri, &content)
        }
    };

    let final_docs = if symbol_docs.is_empty() {
        coarse_chunks_to_symbol_docs(&coarse_chunks)
    } else {
        symbol_docs
    };

    let symbol_rows = final_docs
        .iter()
        .map(|doc| NamedVectorDocument {
            id: doc.id.clone(),
            raw: doc.clone(),
            vectors: rust_item_named_vectors(doc),
        })
        .collect::<Vec<_>>();

    services
        .store
        .insert_documents_with_named_vectors(symbol_rows)
        .await
        .with_context(|| format!("qdrant symbol upsert failed for {}", file_rel))?;

    let final_chunks = symbol_docs_to_chunks(&final_docs);
    let final_ids = final_docs
        .iter()
        .map(|doc| doc.id.clone())
        .collect::<HashSet<_>>();

    let stale_ids = previous_ids
        .difference(&final_ids)
        .cloned()
        .collect::<Vec<_>>();
    if !stale_ids.is_empty() {
        services
            .store
            .delete_documents_by_ids(&stale_ids)
            .await
            .with_context(|| format!("qdrant stale delete failed for {}", file_rel))?;
    }

    let mut state = app.state.write().await;
    state
        .rust_items_by_file
        .insert(file_rel.clone(), final_docs);
    state.chunks_by_file.insert(file_rel.clone(), final_chunks);
    state.indexed_ids_by_file.insert(file_rel, final_ids);

    Ok(())
}

/// Removes a file's chunks from local in-memory cache.
async fn delete_file_chunks(app: &App, path: &Path) -> Result<()> {
    let cfg = app.config.read().await.clone();
    if let Ok(file_uri) = path_to_file_uri_non_canonical(path) {
        app.ra_manager
            .lock()
            .await
            .close_document(&cfg.workspace_root, &file_uri)
            .await;
    }

    let file_rel = path
        .strip_prefix(&cfg.workspace_root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string();

    let ids_to_delete = {
        let mut state = app.state.write().await;
        state.rust_items_by_file.remove(&file_rel);
        state.chunks_by_file.remove(&file_rel);
        state
            .indexed_ids_by_file
            .remove(&file_rel)
            .unwrap_or_default()
            .into_iter()
            .collect::<Vec<_>>()
    };

    if !ids_to_delete.is_empty() {
        let services = app.services.read().await.clone();
        services
            .store
            .delete_documents_by_ids(&ids_to_delete)
            .await
            .with_context(|| format!("qdrant delete failed for {}", file_rel))?;
    }

    Ok(())
}

fn coarse_chunks_to_symbol_docs(chunks: &[CodeChunk]) -> Vec<RustItemDoc> {
    chunks
        .iter()
        .map(|chunk| RustItemDoc {
            id: chunk.chunk_id.clone(),
            kind: normalize_chunk_kind(&chunk.kind).to_string(),
            symbol: format!("chunk_{}_{}", chunk.start_line, chunk.end_line),
            file_path: chunk.file_path.clone(),
            workspace_id: chunk.workspace_id.clone(),
            uri: chunk.uri.clone(),
            module: module_from_file_path(&chunk.file_path),
            signature: chunk
                .text
                .lines()
                .next()
                .unwrap_or_default()
                .trim()
                .to_string(),
            docs: String::new(),
            hover_summary: String::new(),
            signature_help: String::new(),
            body_excerpt: chunk.text.clone(),
            start_line: chunk.start_line,
            end_line: chunk.end_line,
        })
        .collect()
}

fn normalize_chunk_kind(kind: &str) -> &str {
    if kind == "function" { "fn" } else { kind }
}

/// Splits a Rust file into overlapping fixed-size chunks.
fn chunk_rust_file(
    workspace_id: &str,
    file_path: &str,
    uri: &str,
    content: &str,
) -> Vec<CodeChunk> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let chunk_size = 80usize;
    let overlap = 10usize;
    let mut chunks = Vec::new();
    let mut i = 0usize;

    while i < lines.len() {
        let start = i;
        let end = (i + chunk_size).min(lines.len());
        let body = lines[start..end].join("\n");
        let start_line = start + 1;
        let end_line = end;
        let kind = infer_kind(&body).to_string();
        let chunk_id = format!("coarse:{workspace_id}:{file_path}:{start_line}:{end_line}");
        let hash = simple_hash(&body);

        chunks.push(CodeChunk {
            chunk_id,
            file_path: file_path.to_string(),
            workspace_id: workspace_id.to_string(),
            uri: uri.to_string(),
            start_line,
            end_line,
            kind,
            hash,
            text: body,
        });

        if end == lines.len() {
            break;
        }

        i = end.saturating_sub(overlap);
    }

    chunks
}

/// Heuristically infers chunk kind from text patterns.
fn infer_kind(content: &str) -> &'static str {
    if content.contains("\nstruct ") || content.starts_with("struct ") {
        "struct"
    } else if content.contains("\nenum ") || content.starts_with("enum ") {
        "enum"
    } else if content.contains("\ntrait ") || content.starts_with("trait ") {
        "trait"
    } else if content.contains("\nimpl ") || content.starts_with("impl ") {
        "impl"
    } else if content.contains("\nfn ") || content.starts_with("fn ") {
        "function"
    } else {
        "module"
    }
}

/// Returns a stable short hash for chunk content.
fn simple_hash(s: &str) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Recursively scans workspace files with include/exclude matching.
fn list_workspace_files(
    root: &Path,
    include: &[String],
    exclude: &[String],
    respect_gitignore: bool,
) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut seen = HashSet::new();

    for entry in WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| should_walk(e.path(), exclude, respect_gitignore))
    {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path().to_path_buf();
        let rel = path.strip_prefix(root).unwrap_or(path.as_path());
        let rel_str = rel.to_string_lossy();

        if matches_globs(&rel_str, include)
            && !matches_globs(&rel_str, exclude)
            && seen.insert(path.clone())
        {
            files.push(path);
        }
    }

    Ok(files)
}

/// Directory traversal predicate for scan walk.
fn should_walk(path: &Path, exclude: &[String], respect_gitignore: bool) -> bool {
    let name = path
        .file_name()
        .and_then(|v| v.to_str())
        .unwrap_or_default();
    if respect_gitignore && (name == ".git" || name == "target") {
        return false;
    }

    let s = path.to_string_lossy();
    !matches_globs(&s, exclude)
}

/// Lightweight glob matcher used by scan filters.
fn matches_globs(path: &str, globs: &[String]) -> bool {
    if globs.is_empty() {
        return true;
    }

    globs.iter().any(|glob| {
        if glob == "**/*.rs" {
            path.ends_with(".rs")
        } else if let Some(suffix) = glob.strip_prefix("**/*") {
            path.ends_with(suffix)
        } else if let Some(prefix) = glob.strip_suffix("/**") {
            path.starts_with(prefix)
        } else {
            path.contains(glob.trim_matches('*'))
        }
    })
}

/// Resolves relative paths against workspace root.
fn resolve_path(root: &Path, maybe_rel: &str) -> PathBuf {
    let candidate = PathBuf::from(maybe_rel);
    if candidate.is_absolute() {
        candidate
    } else {
        root.join(candidate)
    }
}

/// Returns the current timestamp as unix milliseconds.
fn unix_ms_now() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn derive_workspace_id(workspace_root: &Path) -> String {
    format!("ws_{}", simple_hash(&workspace_root.to_string_lossy()))
}

fn rust_item_search_text(doc: &RustItemDoc) -> String {
    format!(
        "{symbol}\n{signature}\n{signature_help}\n{docs}\n{hover_summary}\n{body_excerpt}\nmodule: {module}\npath: {file_path}\nkind: {kind}\nuri: {uri}\nworkspace_id: {workspace_id}",
        symbol = doc.symbol,
        signature = doc.signature,
        signature_help = doc.signature_help,
        docs = doc.docs,
        hover_summary = doc.hover_summary,
        body_excerpt = doc.body_excerpt,
        module = doc.module,
        file_path = doc.file_path,
        kind = doc.kind,
        uri = doc.uri,
        workspace_id = doc.workspace_id,
    )
}

fn rust_item_named_vectors(doc: &RustItemDoc) -> HashMap<String, String> {
    let mut vectors = HashMap::from([
        (
            SYMBOL_VECTOR_NAME.to_string(),
            format!(
                "{symbol}\nkind: {kind}\nmodule: {module}\npath: {file_path}\nworkspace_id: {workspace_id}",
                symbol = doc.symbol,
                kind = doc.kind,
                module = doc.module,
                file_path = doc.file_path,
                workspace_id = doc.workspace_id,
            ),
        ),
        (
            DOCS_VECTOR_NAME.to_string(),
            format!(
                "{docs}\nhover: {hover}\nsymbol: {symbol}\nmodule: {module}\npath: {file_path}\nworkspace_id: {workspace_id}",
                docs = doc.docs,
                hover = doc.hover_summary,
                symbol = doc.symbol,
                module = doc.module,
                file_path = doc.file_path,
                workspace_id = doc.workspace_id,
            ),
        ),
        (
            SIGNATURE_VECTOR_NAME.to_string(),
            format!(
                "{signature}\n{signature_help}\nsymbol: {symbol}\nkind: {kind}\nmodule: {module}\npath: {file_path}\nworkspace_id: {workspace_id}",
                signature = doc.signature,
                signature_help = doc.signature_help,
                symbol = doc.symbol,
                kind = doc.kind,
                module = doc.module,
                file_path = doc.file_path,
                workspace_id = doc.workspace_id,
            ),
        ),
    ]);

    if !doc.body_excerpt.trim().is_empty() {
        vectors.insert(
            BODY_VECTOR_NAME.to_string(),
            format!(
                "{body_excerpt}\nhover: {hover}\nsymbol: {symbol}\nkind: {kind}\nmodule: {module}\npath: {file_path}\nworkspace_id: {workspace_id}",
                body_excerpt = doc.body_excerpt,
                hover = doc.hover_summary,
                symbol = doc.symbol,
                kind = doc.kind,
                module = doc.module,
                file_path = doc.file_path,
                workspace_id = doc.workspace_id,
            ),
        );
    }

    vectors
}

async fn enrich_docs_with_lsp_metadata(
    session: &mut RaLspSession,
    file_uri: &str,
    docs: &mut [RustItemDoc],
) -> Result<()> {
    for doc in docs {
        let line = doc.start_line.saturating_sub(1);

        let hover = session
            .request(
                "textDocument/hover",
                json!({
                    "textDocument": { "uri": file_uri },
                    "position": { "line": line, "character": 0 }
                }),
            )
            .await?;
        doc.hover_summary = lsp_hover_to_text(&hover).unwrap_or_default();

        let signature_help = session
            .request(
                "textDocument/signatureHelp",
                json!({
                    "textDocument": { "uri": file_uri },
                    "position": { "line": line, "character": 0 }
                }),
            )
            .await?;
        doc.signature_help = lsp_signature_help_to_text(&signature_help).unwrap_or_default();
    }

    Ok(())
}

fn lsp_hover_to_text(result: &Value) -> Option<String> {
    let contents = result.get("contents")?;
    lsp_marked_content_to_text(contents)
}

fn lsp_signature_help_to_text(result: &Value) -> Option<String> {
    let signatures = result.get("signatures")?.as_array()?;
    let first = signatures.first()?;
    first
        .get("label")
        .and_then(Value::as_str)
        .map(|v| v.trim().to_string())
}

fn lsp_marked_content_to_text(contents: &Value) -> Option<String> {
    if let Some(v) = contents.as_str() {
        return Some(v.trim().to_string());
    }

    if let Some(v) = contents.get("value").and_then(Value::as_str) {
        return Some(v.trim().to_string());
    }

    if let Some(items) = contents.as_array() {
        let joined = items
            .iter()
            .filter_map(lsp_marked_content_to_text)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        if !joined.is_empty() {
            return Some(joined);
        }
    }

    None
}

async fn write_lsp_message<W>(writer: &mut W, payload: &str) -> Result<()>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    let framed = format!("Content-Length: {}\r\n\r\n{}", payload.len(), payload);
    writer.write_all(framed.as_bytes()).await?;
    writer.flush().await?;
    Ok(())
}

async fn read_lsp_message<R>(reader: &mut BufReader<R>) -> Result<Option<String>>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut content_length = None::<usize>;
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            if content_length.is_none() {
                return Ok(None);
            }
            anyhow::bail!("rust-analyzer closed stream while reading headers");
        }

        if line == "\r\n" {
            break;
        }

        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            content_length = Some(value.trim().parse()?);
        }
    }

    let len = content_length.ok_or_else(|| anyhow::anyhow!("missing Content-Length header"))?;
    let mut body = vec![0u8; len];
    reader.read_exact(&mut body).await?;
    Ok(Some(String::from_utf8(body)?))
}

async fn read_lsp_response_for_id<R>(reader: &mut BufReader<R>, expected_id: i64) -> Result<Value>
where
    R: tokio::io::AsyncRead + Unpin,
{
    while let Some(raw) = read_lsp_message(reader).await? {
        let msg: Value = serde_json::from_str(&raw)?;
        if msg.get("id").and_then(Value::as_i64) != Some(expected_id) {
            continue;
        }
        if let Some(error) = msg.get("error") {
            anyhow::bail!("rust-analyzer returned error: {}", error);
        }
        return Ok(msg);
    }

    anyhow::bail!("rust-analyzer closed before response id={expected_id}")
}

fn path_to_file_uri(path: &Path) -> Result<String> {
    let canonical = path
        .canonicalize()
        .with_context(|| format!("failed to canonicalize path {}", path.display()))?;
    let mut p = canonical.to_string_lossy().replace('\\', "/");
    if !p.starts_with('/') {
        p = format!("/{p}");
    }
    let encoded = p
        .replace('%', "%25")
        .replace(' ', "%20")
        .replace('#', "%23")
        .replace('?', "%3F");
    Ok(format!("file://{encoded}"))
}

fn path_to_file_uri_non_canonical(path: &Path) -> Result<String> {
    let raw = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("failed to read current directory for uri conversion")?
            .join(path)
    };
    let mut p = raw.to_string_lossy().replace('\\', "/");
    if !p.starts_with('/') {
        p = format!("/{p}");
    }
    let encoded = p
        .replace('%', "%25")
        .replace(' ', "%20")
        .replace('#', "%23")
        .replace('?', "%3F");
    Ok(format!("file://{encoded}"))
}

fn parse_lsp_symbols(
    result: Value,
    workspace_id: &str,
    file_path: &str,
    uri: &str,
    content: &str,
) -> Vec<RustItemDoc> {
    let Some(items) = result.as_array() else {
        return Vec::new();
    };

    let lines = content.lines().collect::<Vec<_>>();
    let module = module_from_file_path(file_path);
    let mut out = Vec::new();

    for item in items {
        collect_lsp_symbol_docs(
            item,
            workspace_id,
            file_path,
            uri,
            &module,
            &lines,
            &mut out,
        );
    }

    out
}

fn collect_lsp_symbol_docs(
    item: &Value,
    workspace_id: &str,
    file_path: &str,
    uri: &str,
    module: &str,
    lines: &[&str],
    out: &mut Vec<RustItemDoc>,
) {
    let Some(name) = item.get("name").and_then(Value::as_str) else {
        return;
    };

    let kind = item
        .get("kind")
        .and_then(Value::as_i64)
        .map(symbol_kind_to_kind)
        .unwrap_or("module")
        .to_string();

    if let Some((start_line, end_line)) = lsp_symbol_line_range(item) {
        let sig_idx = start_line
            .saturating_sub(1)
            .min(lines.len().saturating_sub(1));
        let signature = lines.get(sig_idx).copied().unwrap_or("").trim().to_string();
        let docs = collect_docs_above(lines, sig_idx);
        let excerpt = excerpt_from_range(lines, sig_idx, end_line.saturating_sub(1), 60);
        let id = format!("symbol:{workspace_id}:{file_path}:{kind}:{name}:{start_line}:{end_line}");

        out.push(RustItemDoc {
            id,
            kind,
            symbol: name.to_string(),
            file_path: file_path.to_string(),
            workspace_id: workspace_id.to_string(),
            uri: uri.to_string(),
            module: module.to_string(),
            signature,
            docs,
            hover_summary: String::new(),
            signature_help: String::new(),
            body_excerpt: excerpt,
            start_line,
            end_line,
        });
    }

    if let Some(children) = item.get("children").and_then(Value::as_array) {
        for child in children {
            collect_lsp_symbol_docs(child, workspace_id, file_path, uri, module, lines, out);
        }
    }
}

fn lsp_symbol_line_range(item: &Value) -> Option<(usize, usize)> {
    let range = item
        .get("range")
        .or_else(|| item.get("location").and_then(|loc| loc.get("range")))?;
    let start = range.get("start")?.get("line")?.as_u64()? as usize + 1;
    let end = range.get("end")?.get("line")?.as_u64()? as usize + 1;
    Some((start, end.max(start)))
}

fn symbol_kind_to_kind(kind: i64) -> &'static str {
    match kind {
        2 => "mod",
        5 => "struct",
        6 => "fn",
        10 => "enum",
        11 => "trait",
        12 => "fn",
        13 => "var",
        14 => "const",
        23 => "struct",
        _ => "module",
    }
}

fn extract_symbol_docs_heuristic(
    workspace_id: &str,
    file_path: &str,
    uri: &str,
    content: &str,
) -> Vec<RustItemDoc> {
    let lines = content.lines().collect::<Vec<_>>();
    if lines.is_empty() {
        return Vec::new();
    }

    let module = module_from_file_path(file_path);
    let mut docs = Vec::new();
    let mut line_idx = 0usize;
    while line_idx < lines.len() {
        let line = lines[line_idx].trim();
        let Some(kind) = detect_item_kind(line) else {
            line_idx += 1;
            continue;
        };

        let start_line = line_idx + 1;
        let end_line = find_item_end_line(&lines, line_idx);
        let signature = lines[line_idx].trim().to_string();
        let symbol = extract_symbol(kind, line).unwrap_or_else(|| format!("item_{start_line}"));
        let item_docs = collect_docs_above(&lines, line_idx);
        let body_excerpt = excerpt_from_range(&lines, line_idx, end_line.saturating_sub(1), 60);
        let id =
            format!("symbol:{workspace_id}:{file_path}:{kind}:{symbol}:{start_line}:{end_line}");

        docs.push(RustItemDoc {
            id,
            kind: kind.to_string(),
            symbol,
            file_path: file_path.to_string(),
            workspace_id: workspace_id.to_string(),
            uri: uri.to_string(),
            module: module.clone(),
            signature,
            docs: item_docs,
            hover_summary: String::new(),
            signature_help: String::new(),
            body_excerpt,
            start_line,
            end_line,
        });

        line_idx = end_line;
    }

    docs
}

fn symbol_docs_to_chunks(docs: &[RustItemDoc]) -> Vec<CodeChunk> {
    docs.iter()
        .map(|doc| {
            let text = format!(
                "symbol: {}\nkind: {}\nsignature: {}\nsignature_help: {}\ndocs:\n{}\nhover:\n{}\n\nexcerpt:\n{}",
                doc.symbol,
                doc.kind,
                doc.signature,
                doc.signature_help,
                doc.docs,
                doc.hover_summary,
                doc.body_excerpt
            );
            CodeChunk {
                chunk_id: doc.id.clone(),
                file_path: doc.file_path.clone(),
                workspace_id: doc.workspace_id.clone(),
                uri: doc.uri.clone(),
                start_line: doc.start_line,
                end_line: doc.end_line,
                kind: doc.kind.clone(),
                hash: simple_hash(&text),
                text,
            }
        })
        .collect()
}

fn module_from_file_path(file_path: &str) -> String {
    file_path
        .trim_end_matches(".rs")
        .split('/')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("::")
}

fn detect_item_kind(line: &str) -> Option<&'static str> {
    let stripped = strip_visibility(line);
    if stripped.starts_with("fn ") {
        Some("fn")
    } else if stripped.starts_with("struct ") {
        Some("struct")
    } else if stripped.starts_with("enum ") {
        Some("enum")
    } else if stripped.starts_with("trait ") {
        Some("trait")
    } else if stripped.starts_with("impl ") {
        Some("impl")
    } else if stripped.starts_with("mod ") {
        Some("mod")
    } else if stripped.starts_with("type ") {
        Some("type")
    } else if stripped.starts_with("const ") {
        Some("const")
    } else if stripped.starts_with("macro_rules!") {
        Some("macro")
    } else {
        None
    }
}

fn strip_visibility(line: &str) -> &str {
    let trimmed = line.trim_start();
    for prefix in ["pub(crate) ", "pub(super) ", "pub(self) ", "pub "] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            return rest;
        }
    }
    if let Some(rest) = trimmed.strip_prefix("pub(in ")
        && let Some(idx) = rest.find(')')
    {
        return rest[idx + 1..].trim_start();
    }
    trimmed
}

fn extract_symbol(kind: &str, line: &str) -> Option<String> {
    let stripped = strip_visibility(line);
    let raw = match kind {
        "fn" => stripped.strip_prefix("fn "),
        "struct" => stripped.strip_prefix("struct "),
        "enum" => stripped.strip_prefix("enum "),
        "trait" => stripped.strip_prefix("trait "),
        "mod" => stripped.strip_prefix("mod "),
        "type" => stripped.strip_prefix("type "),
        "const" => stripped.strip_prefix("const "),
        "macro" => stripped.strip_prefix("macro_rules!"),
        "impl" => stripped.strip_prefix("impl "),
        _ => None,
    }?;

    if kind == "impl" {
        let symbol = raw
            .split('{')
            .next()
            .unwrap_or(raw)
            .trim()
            .replace(" for ", " impl_for ");
        return Some(symbol);
    }

    let name = raw
        .trim_start_matches('!')
        .chars()
        .take_while(|ch| ch.is_alphanumeric() || *ch == '_')
        .collect::<String>();
    if name.is_empty() { None } else { Some(name) }
}

fn collect_docs_above(lines: &[&str], start_idx: usize) -> String {
    if start_idx == 0 {
        return String::new();
    }

    let mut docs = Vec::new();
    let mut i = start_idx;
    while i > 0 {
        let line = lines[i - 1].trim();
        if let Some(doc) = line.strip_prefix("///") {
            docs.push(doc.trim().to_string());
        } else if let Some(doc) = line.strip_prefix("//!") {
            docs.push(doc.trim().to_string());
        } else if line.is_empty() && !docs.is_empty() {
            i -= 1;
            continue;
        } else {
            break;
        }
        i -= 1;
    }
    docs.reverse();
    docs.join("\n")
}

fn find_item_end_line(lines: &[&str], start_idx: usize) -> usize {
    let mut brace_balance = 0i64;
    let mut saw_brace = false;

    for (offset, line) in lines[start_idx..].iter().enumerate() {
        for ch in line.chars() {
            if ch == '{' {
                brace_balance += 1;
                saw_brace = true;
            } else if ch == '}' {
                brace_balance -= 1;
            }
        }

        if saw_brace && brace_balance == 0 {
            return start_idx + offset + 1;
        }

        if !saw_brace && line.trim_end().ends_with(';') {
            return start_idx + offset + 1;
        }
    }

    lines.len()
}

fn excerpt_from_range(
    lines: &[&str],
    start_idx: usize,
    end_idx: usize,
    max_lines: usize,
) -> String {
    let end = end_idx.min(lines.len().saturating_sub(1));
    let mut excerpt = lines[start_idx..=end]
        .iter()
        .take(max_lines)
        .copied()
        .collect::<Vec<_>>()
        .join("\n");
    if excerpt.len() > 4000 {
        excerpt.truncate(4000);
    }
    excerpt
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_item(
        file_path: &str,
        id: &str,
        kind: &str,
        symbol: &str,
        excerpt: &str,
    ) -> RustItemDoc {
        RustItemDoc {
            id: id.to_string(),
            kind: kind.to_string(),
            symbol: symbol.to_string(),
            file_path: file_path.to_string(),
            workspace_id: "ws_test".to_string(),
            uri: "file:///tmp/ws/src/main.rs".to_string(),
            module: "src::main".to_string(),
            signature: format!("fn {symbol}() {{}}"),
            docs: String::new(),
            hover_summary: String::new(),
            signature_help: String::new(),
            body_excerpt: excerpt.to_string(),
            start_line: 1,
            end_line: 3,
        }
    }

    #[test]
    fn chunk_rust_file_creates_overlapping_chunks() {
        let content = (1..=170)
            .map(|i| format!("fn line_{i}() {{}}"))
            .collect::<Vec<_>>()
            .join("\n");

        let chunks = chunk_rust_file(
            "ws_test",
            "src/lib.rs",
            "file:///tmp/ws/src/lib.rs",
            &content,
        );

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 80);
        assert_eq!(chunks[1].start_line, 71);
        assert_eq!(chunks[1].end_line, 150);
        assert_eq!(chunks[2].start_line, 141);
        assert_eq!(chunks[2].end_line, 170);
    }

    #[test]
    fn infer_kind_detects_common_rust_items() {
        assert_eq!(infer_kind("struct A {}"), "struct");
        assert_eq!(infer_kind("enum A { B }"), "enum");
        assert_eq!(infer_kind("trait A {}"), "trait");
        assert_eq!(infer_kind("impl A {}"), "impl");
        assert_eq!(infer_kind("fn a() {}"), "function");
    }

    #[test]
    fn matches_globs_handles_expected_patterns() {
        assert!(matches_globs("src/main.rs", &[String::from("**/*.rs")]));
        assert!(matches_globs("src/lib.rs", &[String::from("**/*.rs")]));
        assert!(!matches_globs("src/main.ts", &[String::from("**/*.rs")]));
        assert!(matches_globs(
            "crates/autoagents/src/lib.rs",
            &[String::from("crates/**")]
        ));
    }

    #[test]
    fn split_http_request_parses_headers_and_body() {
        let raw = "POST /mcp/tools/search_code HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"query\":\"tool\"}";
        let (headers, body) = mcp_layer::split_http_request(raw);

        assert!(headers.starts_with("POST /mcp/tools/search_code HTTP/1.1"));
        assert_eq!(body, r#"{"query":"tool"}"#);
    }

    #[test]
    fn resolve_path_joins_relative_paths() {
        let root = Path::new("/tmp/workspace");
        let path = resolve_path(root, "src/main.rs");
        assert_eq!(path, PathBuf::from("/tmp/workspace/src/main.rs"));
    }

    #[test]
    fn lexical_search_ranks_more_matching_chunks_higher() {
        let mut map = HashMap::new();
        map.insert(
            "src/main.rs".to_string(),
            vec![
                sample_item(
                    "src/main.rs",
                    "c1",
                    "fn",
                    "search_code",
                    r#"fn search_code() { let query = "rust"; }"#,
                ),
                sample_item("src/main.rs", "c2", "mod", "parser", "mod parser;"),
            ],
        );

        let ranked = mcp_layer::lexical_search(map, "search rust", 5);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].1.id, "c1");
        assert!(ranked[0].0 > 0.0);
    }

    #[test]
    fn extract_symbol_docs_finds_functions_and_structs() {
        let src = r#"
/// Adds numbers
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub struct Thing {
    value: i32,
}
"#;
        let docs = extract_symbol_docs_heuristic(
            "ws_test",
            "src/lib.rs",
            "file:///tmp/ws/src/lib.rs",
            src,
        );
        assert!(docs.iter().any(|d| d.kind == "fn" && d.symbol == "add"));
        assert!(
            docs.iter()
                .any(|d| d.kind == "struct" && d.symbol == "Thing")
        );
    }

    #[test]
    fn symbol_docs_to_chunks_uses_stable_symbol_ids() {
        let docs = vec![RustItemDoc {
            id: "symbol:ws_test:src/lib.rs:fn:add:1:3".to_string(),
            kind: "fn".to_string(),
            symbol: "add".to_string(),
            file_path: "src/lib.rs".to_string(),
            workspace_id: "ws_test".to_string(),
            uri: "file:///tmp/ws/src/lib.rs".to_string(),
            module: "src::lib".to_string(),
            signature: "fn add(a: i32, b: i32) -> i32 {".to_string(),
            docs: "Adds numbers".to_string(),
            hover_summary: "hover add".to_string(),
            signature_help: "fn add(a: i32, b: i32) -> i32".to_string(),
            body_excerpt: "a + b".to_string(),
            start_line: 1,
            end_line: 3,
        }];

        let chunks = symbol_docs_to_chunks(&docs);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_id, "symbol:ws_test:src/lib.rs:fn:add:1:3");
    }

    #[test]
    fn rust_item_search_text_contains_canonical_fields() {
        let doc = RustItemDoc {
            id: "symbol:ws_test:src/lib.rs:fn:add:1:3".to_string(),
            kind: "fn".to_string(),
            symbol: "add".to_string(),
            file_path: "src/lib.rs".to_string(),
            workspace_id: "ws_test".to_string(),
            uri: "file:///tmp/ws/src/lib.rs".to_string(),
            module: "src::lib".to_string(),
            signature: "fn add(a: i32, b: i32) -> i32 {".to_string(),
            docs: "Adds numbers".to_string(),
            hover_summary: "hover add".to_string(),
            signature_help: "fn add(a: i32, b: i32) -> i32".to_string(),
            body_excerpt: "a + b".to_string(),
            start_line: 1,
            end_line: 3,
        };

        let search_text = rust_item_search_text(&doc);
        assert!(search_text.contains("add"));
        assert!(search_text.contains("module: src::lib"));
        assert!(search_text.contains("path: src/lib.rs"));
        assert!(search_text.contains("kind: fn"));
        assert!(search_text.contains("uri: file:///tmp/ws/src/lib.rs"));
        assert!(search_text.contains("workspace_id: ws_test"));
    }
}
