use anyhow::{Context, Result};
use autoagents_core::embeddings::{Embed, EmbedError, TextEmbedder};
use autoagents_core::vector_store::request::VectorSearchRequest;
use autoagents_core::vector_store::{NamedVectorDocument, VectorStoreIndex};
use autoagents_llm::backends::ollama::Ollama;
use autoagents_llm::embedding::EmbeddingBuilder;
use autoagents_qdrant::QdrantVectorStore;
use lsp_types::SymbolKind;
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
mod schema;

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
const TYPE_VECTOR_NAME: &str = "type";
const GRAPH_VECTOR_NAME: &str = "graph";
const SEMANTIC_VECTOR_NAME: &str = "semantic";
const SYNTAX_VECTOR_NAME: &str = "syntax";
const EMBED_OVERSIZE_DEBUG_CHARS: usize = 12_000;
const EMBED_HARD_CAP_CHARS: usize = 8_000;
const LSP_REQUEST_TIMEOUT: Duration = Duration::from_secs(6);
const LSP_HEAVY_REQUEST_TIMEOUT: Duration = Duration::from_secs(12);
const LSP_CONTENT_MODIFIED_RETRIES: usize = 1;
const LSP_WORK_DONE_SETTLE_TIMEOUT: Duration = Duration::from_millis(700);
const LSP_WORK_DONE_POLL_INTERVAL: Duration = Duration::from_millis(60);
const BULK_SCAN_QUEUE_DEPTH_THRESHOLD: usize = 32;
const WORKSPACE_REFRESH_MAX_RETRIES: u8 = 3;

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
    /// Qdrant collection name for cross-file symbol relations.
    qdrant_relation_collection: String,
    /// Qdrant collection name for workspace crate metadata.
    qdrant_metadata_collection: String,
    /// Qdrant collection name for file-level documents.
    qdrant_file_collection: String,
    /// Qdrant collection name for typed call edges.
    qdrant_call_edge_collection: String,
    /// Qdrant collection name for typed type edges.
    qdrant_type_edge_collection: String,
    /// Qdrant collection name for diagnostic documents.
    qdrant_diagnostic_collection: String,
    /// Qdrant collection name for semantic token artifacts.
    qdrant_semantic_collection: String,
    /// Qdrant collection name for syntax tree artifacts.
    qdrant_syntax_collection: String,
    /// Qdrant collection name for inlay hint artifacts.
    qdrant_inlay_collection: String,
    /// Qdrant collection name for crate graph artifacts.
    qdrant_crate_graph_collection: String,
    /// Optional Qdrant API key.
    qdrant_api_key: Option<String>,
    /// Ollama endpoint URL.
    ollama_base_url: String,
    /// Chat/completion model name.
    llm_model: String,
    /// Embedding model name.
    embedding_model: String,
    /// MCP HTTP bind port (`0` means ephemeral).
    mcp_port: u16,
    /// Enables syntax artifact extraction/indexing (`viewSyntaxTree` channel).
    enable_syntax_artifacts: bool,
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
                .unwrap_or_else(|_| "rust_copilot_symbols".to_string()),
            qdrant_relation_collection: std::env::var("QDRANT_RELATION_COLLECTION")
                .unwrap_or_else(|_| "rust_copilot_relations".to_string()),
            qdrant_metadata_collection: std::env::var("QDRANT_METADATA_COLLECTION")
                .unwrap_or_else(|_| "rust_copilot_metadata".to_string()),
            qdrant_file_collection: std::env::var("QDRANT_FILE_COLLECTION")
                .unwrap_or_else(|_| "rust_copilot_files".to_string()),
            qdrant_call_edge_collection: std::env::var("QDRANT_CALL_EDGE_COLLECTION")
                .unwrap_or_else(|_| "rust_copilot_calls".to_string()),
            qdrant_type_edge_collection: std::env::var("QDRANT_TYPE_EDGE_COLLECTION")
                .unwrap_or_else(|_| "rust_copilot_types".to_string()),
            qdrant_diagnostic_collection: std::env::var("QDRANT_DIAGNOSTIC_COLLECTION")
                .unwrap_or_else(|_| "rust_copilot_diagnostics".to_string()),
            qdrant_semantic_collection: std::env::var("QDRANT_SEMANTIC_COLLECTION")
                .unwrap_or_else(|_| "rust_copilot_semantic_artifacts".to_string()),
            qdrant_syntax_collection: std::env::var("QDRANT_SYNTAX_COLLECTION")
                .unwrap_or_else(|_| "rust_copilot_syntax_artifacts".to_string()),
            qdrant_inlay_collection: std::env::var("QDRANT_INLAY_COLLECTION")
                .unwrap_or_else(|_| "rust_copilot_inlay_artifacts".to_string()),
            qdrant_crate_graph_collection: std::env::var("QDRANT_CRATE_GRAPH_COLLECTION")
                .unwrap_or_else(|_| "rust_copilot_crate_graph".to_string()),
            qdrant_api_key: std::env::var("QDRANT_API_KEY").ok(),
            ollama_base_url: std::env::var("OLLAMA_BASE_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string()),
            llm_model: "gpt-oss:20b".to_string(),
            embedding_model: "dengcao/Qwen3-Embedding-8B:Q4_K_M".to_string(),
            mcp_port: parse_env_u16("MCP_PORT").unwrap_or(0),
            enable_syntax_artifacts: parse_env_bool("ENABLE_SYNTAX_ARTIFACTS").unwrap_or(false),
        }
    }
}

fn parse_env_u16(name: &str) -> Option<u16> {
    let raw = std::env::var(name).ok()?;
    match raw.parse::<u16>() {
        Ok(value) => Some(value),
        Err(_) => None,
    }
}

fn parse_env_bool(name: &str) -> Option<bool> {
    let raw = std::env::var(name).ok()?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
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

impl Embed for schema::SymbolDoc {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(rust_item_search_text(self));
        Ok(())
    }
}

impl Embed for schema::GraphEdgeDoc {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(relation_search_text(self));
        Ok(())
    }
}

impl Embed for schema::SemanticTokenDoc {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(semantic_artifact_search_text(self));
        Ok(())
    }
}

impl Embed for schema::SyntaxTreeDoc {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(syntax_artifact_search_text(self));
        Ok(())
    }
}

impl Embed for schema::InlayHintDoc {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(inlay_artifact_search_text(self));
        Ok(())
    }
}

impl Embed for schema::CrateGraphDoc {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(crate_graph_search_text(self));
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
    /// Extraction metrics for quality/observability.
    extraction_metrics: ExtractionMetrics,
    /// Count of indexed workspace crates.
    workspace_crates: usize,
    /// Current number of active diagnostics in the in-memory latest-pass snapshot.
    diagnostics_active_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct WorkspaceMetadata {
    workspace_root: String,
    crates: Vec<CrateMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct CrateMetadata {
    crate_name: String,
    edition: String,
    crate_root: String,
    manifest_path: String,
    features: Vec<String>,
    optional_dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CrateMetadataDoc {
    id: String,
    workspace_id: String,
    crate_name: String,
    edition: String,
    crate_root: String,
    manifest_path: String,
    features: Vec<String>,
    optional_dependencies: Vec<String>,
}

impl Embed for CrateMetadataDoc {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(crate_metadata_search_text(self));
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Default)]
struct ExtractionMetrics {
    files_reindexed_total: u64,
    files_fallback_heuristic_total: u64,
    symbols_indexed_total: u64,
    relations_indexed_total: u64,
    hover_success_total: u64,
    hover_nonempty_total: u64,
    hover_failed_total: u64,
    workspace_symbol_success_total: u64,
    workspace_symbol_failed_total: u64,
    workspace_symbol_nonempty_total: u64,
    // signature_help_success_total: u64,
    // signature_help_failed_total: u64,
    references_success_total: u64,
    references_failed_total: u64,
    references_nonempty_total: u64,
    implementations_success_total: u64,
    implementations_failed_total: u64,
    implementations_nonempty_total: u64,
    definitions_success_total: u64,
    definitions_failed_total: u64,
    definitions_nonempty_total: u64,
    type_definitions_success_total: u64,
    type_definitions_failed_total: u64,
    type_definitions_nonempty_total: u64,
    relations_references_emitted_total: u64,
    relations_implementations_emitted_total: u64,
    relations_definitions_emitted_total: u64,
    relations_type_definitions_emitted_total: u64,
    content_modified_retries_total: u64,
    request_timeouts_total: u64,
    semantic_tokens_success_total: u64,
    semantic_tokens_failed_total: u64,
    semantic_tokens_nonempty_total: u64,
    syntax_tree_success_total: u64,
    syntax_tree_failed_total: u64,
    syntax_tree_nonempty_total: u64,
    syntax_tree_unsupported_total: u64,
    inlay_hints_success_total: u64,
    inlay_hints_failed_total: u64,
    inlay_hints_nonempty_total: u64,
    workspace_refresh_requeued_total: u64,
    crate_graph_transient_requeued_total: u64,
    crate_graph_unsupported_total: u64,
    crate_graph_success_total: u64,
    crate_graph_failed_total: u64,
    crate_graph_nonempty_total: u64,
    diagnostics_indexed_total: u64,
}

#[derive(Debug, Default)]
struct FileExtractionMetrics {
    fallback_heuristic: bool,
    symbols_indexed: usize,
    relations_indexed: usize,
    hover_success: u64,
    hover_nonempty: u64,
    hover_failed: u64,
    workspace_symbol_success: u64,
    workspace_symbol_failed: u64,
    workspace_symbol_nonempty: u64,
    // signature_help_success: u64,
    // signature_help_failed: u64,
    references_success: u64,
    references_failed: u64,
    references_nonempty: u64,
    relations_references_emitted: u64,
    implementations_success: u64,
    implementations_failed: u64,
    implementations_nonempty: u64,
    relations_implementations_emitted: u64,
    definitions_success: u64,
    definitions_failed: u64,
    definitions_nonempty: u64,
    relations_definitions_emitted: u64,
    type_definitions_success: u64,
    type_definitions_failed: u64,
    type_definitions_nonempty: u64,
    relations_type_definitions_emitted: u64,
    content_modified_retries: u64,
    request_timeouts: u64,
    semantic_tokens_success: u64,
    semantic_tokens_failed: u64,
    semantic_tokens_nonempty: u64,
    syntax_tree_success: u64,
    syntax_tree_failed: u64,
    syntax_tree_nonempty: u64,
    syntax_tree_unsupported: u64,
    inlay_hints_success: u64,
    inlay_hints_failed: u64,
    inlay_hints_nonempty: u64,
    diagnostics_indexed: usize,
}

/// Mutable daemon state shared by both planes.
#[derive(Default)]
struct State {
    /// Local in-memory view of indexed chunks by file path.
    chunks_by_file: HashMap<String, Vec<CodeChunk>>,
    /// Local in-memory view of indexed symbol docs by file path.
    rust_items_by_file: HashMap<String, Vec<schema::SymbolDoc>>,
    /// Local in-memory view of indexed relations by source file path.
    relations_by_file: HashMap<String, Vec<schema::GraphEdgeDoc>>,
    /// Local in-memory file-level docs by file path.
    file_docs_by_file: HashMap<String, schema::FileDoc>,
    /// Local in-memory typed call edges by source file path.
    call_edges_by_file: HashMap<String, Vec<schema::CallEdge>>,
    /// Local in-memory typed type edges by source file path.
    type_edges_by_file: HashMap<String, Vec<schema::TypeEdge>>,
    /// Local in-memory diagnostics by file path.
    diagnostics_by_file: HashMap<String, Vec<schema::DiagnosticDoc>>,
    /// Local in-memory semantic-token artifacts by file path.
    semantic_tokens_by_file: HashMap<String, Vec<schema::SemanticTokenDoc>>,
    /// Local in-memory syntax-tree artifacts by file path.
    syntax_trees_by_file: HashMap<String, Vec<schema::SyntaxTreeDoc>>,
    /// Local in-memory inlay-hint artifacts by file path.
    inlay_hints_by_file: HashMap<String, Vec<schema::InlayHintDoc>>,
    /// Local in-memory crate graph artifacts by crate root.
    crate_graph_by_crate: HashMap<String, schema::CrateGraphDoc>,
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
    /// Currently indexed relation IDs by file, used for incremental cleanup.
    indexed_relation_ids_by_file: HashMap<String, HashSet<String>>,
    /// Currently indexed file-doc IDs by file, used for incremental cleanup.
    indexed_file_doc_ids_by_file: HashMap<String, HashSet<String>>,
    /// Currently indexed call-edge IDs by file, used for incremental cleanup.
    indexed_call_edge_ids_by_file: HashMap<String, HashSet<String>>,
    /// Currently indexed type-edge IDs by file, used for incremental cleanup.
    indexed_type_edge_ids_by_file: HashMap<String, HashSet<String>>,
    /// Currently indexed diagnostic IDs by file, used for incremental cleanup.
    indexed_diagnostic_ids_by_file: HashMap<String, HashSet<String>>,
    /// Currently indexed semantic-token IDs by file, used for incremental cleanup.
    indexed_semantic_ids_by_file: HashMap<String, HashSet<String>>,
    /// Currently indexed syntax-tree IDs by file, used for incremental cleanup.
    indexed_syntax_ids_by_file: HashMap<String, HashSet<String>>,
    /// Currently indexed inlay-hint IDs by file, used for incremental cleanup.
    indexed_inlay_ids_by_file: HashMap<String, HashSet<String>>,
    /// Parsed workspace metadata snapshot.
    workspace_metadata: Option<WorkspaceMetadata>,
    /// Indexed metadata docs by crate root.
    metadata_docs_by_crate: HashMap<String, CrateMetadataDoc>,
    /// Logical IDs for metadata docs in vector store.
    indexed_metadata_ids: HashSet<String>,
    /// Logical IDs for crate graph docs in vector store.
    indexed_crate_graph_ids: HashSet<String>,
    /// Aggregated extraction metrics.
    extraction_metrics: ExtractionMetrics,
    /// True after a post-initialize warm-up extraction has completed.
    is_ra_warm: bool,
    /// True after first implementation extraction pass has been discarded/requeued.
    is_ra_warm_impl: bool,
    /// True once crate-graph extraction has passed first-call warmup.
    is_ra_warm_crate_graph: bool,
    /// Number of consecutive crate-graph refresh retries.
    workspace_refresh_retry_count: u8,
    /// Set after first scan.full; graph refresh is triggered when queue drains.
    pending_initial_graph_refresh: bool,
    /// Guards one-time deferred graph refresh per workspace lifecycle.
    initial_graph_refresh_done: bool,
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
            extraction_metrics: self.extraction_metrics.clone(),
            workspace_crates: self
                .workspace_metadata
                .as_ref()
                .map(|w| w.crates.len())
                .unwrap_or(0),
            diagnostics_active_count: self
                .diagnostics_by_file
                .values()
                .map(std::vec::Vec::len)
                .sum::<usize>(),
        }
    }
}

/// External services used by indexing/query flows.
#[derive(Clone)]
struct Services {
    /// Shared vector store client.
    store: QdrantVectorStore,
    /// Dedicated vector store for relation documents.
    relation_store: QdrantVectorStore,
    /// Dedicated vector store for workspace metadata documents.
    metadata_store: QdrantVectorStore,
    /// Dedicated vector store for file-level documents.
    file_store: QdrantVectorStore,
    /// Dedicated vector store for typed call edges.
    call_edge_store: QdrantVectorStore,
    /// Dedicated vector store for typed type edges.
    type_edge_store: QdrantVectorStore,
    /// Dedicated vector store for diagnostics.
    diagnostic_store: QdrantVectorStore,
    /// Dedicated vector store for semantic-token artifacts.
    semantic_store: QdrantVectorStore,
    /// Dedicated vector store for syntax-tree artifacts.
    syntax_store: QdrantVectorStore,
    /// Dedicated vector store for inlay-hint artifacts.
    inlay_store: QdrantVectorStore,
    /// Dedicated vector store for crate-graph artifacts.
    crate_graph_store: QdrantVectorStore,
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
    /// Refresh workspace-level graph artifacts.
    RefreshWorkspace,
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
        bulk_mode: bool,
        metrics: &mut FileExtractionMetrics,
    ) -> Result<Vec<schema::SymbolDoc>> {
        let file_uri = if file_uri.is_empty() {
            path_to_file_uri(file_path_abs)?
        } else {
            file_uri.to_string()
        };

        let session = self.ensure_session(workspace_root).await?;
        session.sync_document(&file_uri, content).await?;
        let Some(result) = session
            .request_if_supported(
                "textDocument/documentSymbol",
                json!({
                    "textDocument": { "uri": file_uri }
                }),
                LSP_REQUEST_TIMEOUT,
                LSP_CONTENT_MODIFIED_RETRIES,
            )
            .await?
        else {
            return Ok(Vec::new());
        };
        let mut docs = parse_lsp_symbols(
            result.clone(),
            workspace_id,
            file_path_rel,
            &file_uri,
            content,
        );
        enrich_docs_with_lsp_metadata(session, &file_uri, content, &mut docs, bulk_mode, metrics)
            .await?;
        for doc in &mut docs {
            doc.finalize_derived_fields();
        }
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

    async fn extract_symbol_relations(
        &mut self,
        workspace_root: &Path,
        workspace_id: &str,
        file_path_rel: &str,
        file_uri: &str,
        content: &str,
        docs: &[schema::SymbolDoc],
        bulk_mode: bool,
        metrics: &mut FileExtractionMetrics,
    ) -> Result<Vec<schema::GraphEdgeDoc>> {
        let session = self.ensure_session(workspace_root).await?;
        session.sync_document(file_uri, content).await?;
        let relations = extract_symbol_relations_with_lsp(
            session,
            workspace_root,
            workspace_id,
            file_path_rel,
            file_uri,
            content,
            docs,
            bulk_mode,
            metrics,
        )
        .await?;
        Ok(relations)
    }

    async fn extract_file_artifacts(
        &mut self,
        workspace_root: &Path,
        workspace_id: &str,
        file_path_rel: &str,
        file_uri: &str,
        content: &str,
        docs: &[schema::SymbolDoc],
        enable_syntax_artifacts: bool,
        metrics: &mut FileExtractionMetrics,
    ) -> Result<(
        Vec<schema::SemanticTokenDoc>,
        Vec<schema::SyntaxTreeDoc>,
        Vec<schema::InlayHintDoc>,
    )> {
        let session = self.ensure_session(workspace_root).await?;
        session.sync_document(file_uri, content).await?;
        extract_lsp_artifact_docs(
            session,
            workspace_id,
            file_path_rel,
            file_uri,
            content,
            docs,
            enable_syntax_artifacts,
            metrics,
        )
        .await
    }

    async fn extract_file_diagnostics(
        &mut self,
        workspace_root: &Path,
        workspace_id: &str,
        file_path_rel: &str,
        file_uri: &str,
    ) -> Result<Vec<schema::DiagnosticDoc>> {
        let session = self.ensure_session(workspace_root).await?;
        let diagnostics = session.latest_diagnostics_for_uri(file_uri).await?;
        Ok(build_diagnostic_docs_from_lsp(
            workspace_id,
            file_path_rel,
            file_uri,
            &diagnostics,
        ))
    }

    async fn extract_crate_graph_doc(
        &mut self,
        workspace_root: &Path,
        workspace_id: &str,
        crate_name: &str,
        crate_root: &str,
    ) -> Result<Option<schema::CrateGraphDoc>> {
        let session = self.ensure_session(workspace_root).await?;
        let Some(payload) = session
            .request_if_supported(
                "rust-analyzer/viewCrateGraph",
                json!({ "full": true }),
                LSP_HEAVY_REQUEST_TIMEOUT,
                LSP_CONTENT_MODIFIED_RETRIES,
            )
            .await?
        else {
            return Ok(None);
        };
        let summary = summarize_crate_graph_payload(&payload, crate_name);
        if summary.trim().is_empty() {
            return Ok(None);
        }
        Ok(Some(schema::CrateGraphDoc {
            id: format!("crate_graph:{workspace_id}:{crate_root}"),
            workspace_id: workspace_id.to_string(),
            crate_name: crate_name.to_string(),
            crate_root: crate_root.to_string(),
            summary,
        }))
    }

    fn session_counters(&self) -> RaSessionCounters {
        self.session
            .as_ref()
            .map(RaLspSession::counters)
            .unwrap_or_default()
    }
}

fn summarize_crate_graph_payload(payload: &Value, crate_name: &str) -> String {
    let raw = payload
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| payload.to_string());

    let mut node_labels = HashMap::<String, String>::new();
    let mut edges = Vec::<(String, String)>::new();

    for line in raw.lines().map(str::trim) {
        if let Some((node_id, label)) = parse_dot_node_line(line) {
            node_labels.insert(node_id.to_string(), label.to_string());
            continue;
        }
        if let Some((src, dst)) = parse_dot_edge_line(line) {
            edges.push((src.to_string(), dst.to_string()));
        }
    }

    let nodes_total = node_labels.len();
    let edges_total = edges.len();
    let Some(crate_node_id) = node_labels
        .iter()
        .find_map(|(node_id, label)| (label == crate_name).then_some(node_id.clone()))
    else {
        return format!(
            "crate: {crate}\nnode_found: false\nnodes_total: {nodes}\nedges_total: {edges}",
            crate = crate_name,
            nodes = nodes_total,
            edges = edges_total
        );
    };

    let mut outbound = edges
        .iter()
        .filter_map(|(src, dst)| {
            (src == &crate_node_id)
                .then(|| node_labels.get(dst).cloned())
                .flatten()
        })
        .collect::<Vec<_>>();
    outbound.sort();
    outbound.dedup();

    let mut inbound = edges
        .iter()
        .filter_map(|(src, dst)| {
            (dst == &crate_node_id)
                .then(|| node_labels.get(src).cloned())
                .flatten()
        })
        .collect::<Vec<_>>();
    inbound.sort();
    inbound.dedup();

    format!(
        "crate: {crate}\nnode_found: true\nnodes_total: {nodes}\nedges_total: {edges}\noutbound_deps_count: {out_count}\noutbound_deps: {outbound}\ninbound_deps_count: {in_count}\ninbound_deps: {inbound}",
        crate = crate_name,
        nodes = nodes_total,
        edges = edges_total,
        out_count = outbound.len(),
        outbound = join_limited(&outbound, 24),
        in_count = inbound.len(),
        inbound = join_limited(&inbound, 24),
    )
}

fn parse_dot_node_line(line: &str) -> Option<(&str, &str)> {
    let id_end = line.find('[')?;
    let node_id = line[..id_end].trim();
    if node_id.is_empty() {
        return None;
    }
    let (label_start, terminator) = if let Some(idx) = line.find("label=\"") {
        (idx + "label=\"".len(), "\"")
    } else {
        let idx = line.find("label=\\\"")?;
        (idx + "label=\\\"".len(), "\\\"")
    };
    let rest = &line[label_start..];
    let label_end = rest.find(terminator)?;
    let label = &rest[..label_end];
    Some((node_id, label))
}

fn parse_dot_edge_line(line: &str) -> Option<(&str, &str)> {
    let arrow = line.find("->")?;
    let src = line[..arrow].trim();
    if src.is_empty() {
        return None;
    }
    let rest = line[(arrow + 2)..].trim_start();
    let dst_end = rest
        .find('[')
        .or_else(|| rest.find(';'))
        .unwrap_or(rest.len());
    let dst = rest[..dst_end].trim();
    if dst.is_empty() {
        return None;
    }
    Some((src, dst))
}

fn join_limited(items: &[String], max_items: usize) -> String {
    if items.is_empty() {
        return String::new();
    }
    if items.len() <= max_items {
        return items.join(",");
    }
    let mut out = items[..max_items].join(",");
    out.push_str(",...");
    out
}

struct RaLspSession {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
    next_id: i64,
    open_doc_versions: HashMap<String, i64>,
    unsupported_methods: HashSet<String>,
    content_modified_retries_total: u64,
    request_timeouts_total: u64,
    diagnostics_by_uri: HashMap<String, Vec<Value>>,
    work_done_in_flight: usize,
    saw_work_done_progress: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct RaSessionCounters {
    content_modified_retries_total: u64,
    request_timeouts_total: u64,
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
            unsupported_methods: HashSet::new(),
            content_modified_retries_total: 0,
            request_timeouts_total: 0,
            diagnostics_by_uri: HashMap::new(),
            work_done_in_flight: 0,
            saw_work_done_progress: false,
        };

        let workspace_uri = path_to_file_uri(workspace_root)?;
        let _ = session
            .request(
                "initialize",
                json!({
                    "processId": null,
                    "rootUri": workspace_uri,
                    "capabilities": {
                        "window": {
                            "workDoneProgress": true
                        },
                        "textDocument": {
                            "documentSymbol": {
                                "dynamicRegistration": false,
                                "hierarchicalDocumentSymbolSupport": true,
                                "labelSupport": true
                            }
                        }
                    },
                    "initializationOptions": {
                        "cargo": {
                            "allTargets": true,
                            "cfgs": ["test"]
                        }
                    }
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
        self.request_with_retry(
            method,
            params,
            LSP_REQUEST_TIMEOUT,
            LSP_CONTENT_MODIFIED_RETRIES,
        )
        .await
    }

    async fn request_with_retry(
        &mut self,
        method: &str,
        params: Value,
        timeout: Duration,
        retries: usize,
    ) -> Result<Value> {
        let mut attempts = 0usize;
        loop {
            let result = self
                .request_with_timeout(method, params.clone(), timeout)
                .await;
            match result {
                Ok(v) => return Ok(v),
                Err(err) if attempts < retries && is_content_modified_error(&err) => {
                    attempts += 1;
                    self.content_modified_retries_total += 1;
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                Err(err) => return Err(err),
            }
        }
    }

    async fn request_with_timeout(
        &mut self,
        method: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value> {
        let req_id = self.next_id;
        self.next_id += 1;

        tokio::time::timeout(timeout, async {
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
            let response = read_lsp_response_for_id(
                &mut self.reader,
                req_id,
                &mut self.work_done_in_flight,
                &mut self.saw_work_done_progress,
                &mut self.diagnostics_by_uri,
            )
            .await?;
            Ok::<Value, anyhow::Error>(response.get("result").cloned().unwrap_or(Value::Null))
        })
        .await
        .map_err(|_| {
            self.request_timeouts_total += 1;
            anyhow::anyhow!("rust-analyzer request timed out: {method}")
        })?
    }

    async fn request_if_supported(
        &mut self,
        method: &str,
        params: Value,
        timeout: Duration,
        retries: usize,
    ) -> Result<Option<Value>> {
        if self.unsupported_methods.contains(method) {
            return Ok(None);
        }

        match self
            .request_with_retry(method, params, timeout, retries)
            .await
        {
            Ok(v) => Ok(Some(v)),
            Err(err) if is_method_unsupported_error(&err) => {
                if method == "rust-analyzer/viewSyntaxTree" {
                    eprintln!(
                        "[syntax-tree-debug] method=rust-analyzer/viewSyntaxTree first_unsupported_response={:#}",
                        err
                    );
                }
                self.unsupported_methods.insert(method.to_string());
                Ok(None)
            }
            Err(err) => Err(err),
        }
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
        self.diagnostics_by_uri.remove(file_uri);
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

    async fn wait_for_work_done_settle(&mut self, timeout: Duration) -> Result<()> {
        if !self.saw_work_done_progress {
            self.collect_pending_messages(LSP_WORK_DONE_POLL_INTERVAL)
                .await?;
            return Ok(());
        }
        let deadline = tokio::time::Instant::now() + timeout;
        while tokio::time::Instant::now() < deadline {
            if self.work_done_in_flight == 0 {
                return Ok(());
            }
            match tokio::time::timeout(
                LSP_WORK_DONE_POLL_INTERVAL,
                read_lsp_message(&mut self.reader),
            )
            .await
            {
                Ok(Ok(Some(raw))) => {
                    if let Ok(msg) = serde_json::from_str::<Value>(&raw) {
                        update_lsp_side_channel_state(
                            &msg,
                            &mut self.work_done_in_flight,
                            &mut self.saw_work_done_progress,
                            &mut self.diagnostics_by_uri,
                        );
                    }
                }
                Ok(Ok(None)) => return Ok(()),
                Ok(Err(err)) => return Err(err),
                Err(_) => {}
            }
        }
        Ok(())
    }

    async fn collect_pending_messages(&mut self, wait: Duration) -> Result<()> {
        loop {
            match tokio::time::timeout(wait, read_lsp_message(&mut self.reader)).await {
                Ok(Ok(Some(raw))) => {
                    if let Ok(msg) = serde_json::from_str::<Value>(&raw) {
                        update_lsp_side_channel_state(
                            &msg,
                            &mut self.work_done_in_flight,
                            &mut self.saw_work_done_progress,
                            &mut self.diagnostics_by_uri,
                        );
                    }
                }
                Ok(Ok(None)) => return Ok(()),
                Ok(Err(err)) => return Err(err),
                Err(_) => return Ok(()),
            }
        }
    }

    async fn latest_diagnostics_for_uri(&mut self, file_uri: &str) -> Result<Vec<Value>> {
        self.wait_for_work_done_settle(LSP_WORK_DONE_SETTLE_TIMEOUT)
            .await?;
        self.collect_pending_messages(Duration::from_millis(120))
            .await?;
        Ok(self
            .diagnostics_by_uri
            .get(file_uri)
            .cloned()
            .unwrap_or_default())
    }

    fn counters(&self) -> RaSessionCounters {
        RaSessionCounters {
            content_modified_retries_total: self.content_modified_retries_total,
            request_timeouts_total: self.request_timeouts_total,
        }
    }
}

fn is_content_modified_error(err: &anyhow::Error) -> bool {
    let message = format!("{err:#}");
    message.contains("\"code\":-32801") || message.contains("content modified")
}

fn is_transient_lsp_error(err: &anyhow::Error) -> bool {
    let message = format!("{err:#}");
    is_content_modified_error(err)
        || message.contains("request timed out")
        || message.contains("\"code\":-32800")
        || message.contains("\"code\":-32802")
        || message.contains("server cancelled")
        || message.contains("canceled")
}

fn is_method_unsupported_error(err: &anyhow::Error) -> bool {
    let message = format!("{err:#}");
    message.contains("\"code\":-32601")
        || message.contains("Method not found")
        || message.contains("method not found")
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
    #[serde(rename = "relationCollection")]
    relation_collection: Option<String>,
    #[serde(rename = "metadataCollection")]
    metadata_collection: Option<String>,
    #[serde(rename = "fileCollection")]
    file_collection: Option<String>,
    #[serde(rename = "callEdgeCollection")]
    call_edge_collection: Option<String>,
    #[serde(rename = "typeEdgeCollection")]
    type_edge_collection: Option<String>,
    #[serde(rename = "diagnosticCollection")]
    diagnostic_collection: Option<String>,
    #[serde(rename = "semanticCollection")]
    semantic_collection: Option<String>,
    #[serde(rename = "syntaxCollection")]
    syntax_collection: Option<String>,
    #[serde(rename = "inlayCollection")]
    inlay_collection: Option<String>,
    #[serde(rename = "crateGraphCollection")]
    crate_graph_collection: Option<String>,
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
    #[serde(rename = "enableSyntaxArtifacts")]
    enable_syntax_artifacts: Option<bool>,
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
    limit: Option<usize>,
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

/// Optional filters accepted by `search_relations`.
#[derive(Debug, Deserialize)]
struct RelationSearchFilters {
    workspace_id: Option<String>,
    relation_kind: Option<String>,
    source_file_path: Option<String>,
    target_file_path: Option<String>,
}

/// Optional filters accepted by `search_files`.
#[derive(Debug, Deserialize)]
struct FileSearchFilters {
    workspace_id: Option<String>,
    file_path: Option<String>,
    module: Option<String>,
    crate_name: Option<String>,
}

/// Optional filters accepted by `search_calls`.
#[derive(Debug, Deserialize)]
struct CallEdgeSearchFilters {
    workspace_id: Option<String>,
    source_file_path: Option<String>,
    target_file_path: Option<String>,
    source_symbol_id: Option<String>,
    target_symbol_id: Option<String>,
}

/// Optional filters accepted by `search_types`.
#[derive(Debug, Deserialize)]
struct TypeEdgeSearchFilters {
    workspace_id: Option<String>,
    relation_kind: Option<String>,
    source_file_path: Option<String>,
    target_file_path: Option<String>,
    source_symbol_id: Option<String>,
    target_symbol_id: Option<String>,
}

/// Optional filters accepted by `search_diagnostics`.
#[derive(Debug, Deserialize)]
struct DiagnosticSearchFilters {
    workspace_id: Option<String>,
    file_path: Option<String>,
    severity: Option<String>,
    code: Option<String>,
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
    limit: Option<usize>,
    file_path: Option<String>,
}

/// `get_symbol_relations` request body.
#[derive(Debug, Deserialize)]
struct SymbolRelationsRequest {
    symbol: String,
    limit: Option<usize>,
    relation_kind: Option<String>,
    file_path: Option<String>,
}

/// `search_relations` request body.
#[derive(Debug, Deserialize)]
struct SearchRelationsRequest {
    query: String,
    limit: Option<usize>,
    top_k: Option<usize>,
    vector_name: Option<String>,
    filters: Option<RelationSearchFilters>,
}

/// `search_files` request body.
#[derive(Debug, Deserialize)]
struct SearchFilesRequest {
    query: String,
    limit: Option<usize>,
    top_k: Option<usize>,
    vector_name: Option<String>,
    filters: Option<FileSearchFilters>,
}

/// `search_calls` request body.
#[derive(Debug, Deserialize)]
struct SearchCallsRequest {
    query: String,
    limit: Option<usize>,
    top_k: Option<usize>,
    vector_name: Option<String>,
    filters: Option<CallEdgeSearchFilters>,
}

/// `search_types` request body.
#[derive(Debug, Deserialize)]
struct SearchTypesRequest {
    query: String,
    limit: Option<usize>,
    top_k: Option<usize>,
    vector_name: Option<String>,
    filters: Option<TypeEdgeSearchFilters>,
}

/// `search_diagnostics` request body.
#[derive(Debug, Deserialize)]
struct SearchDiagnosticsRequest {
    query: String,
    limit: Option<usize>,
    top_k: Option<usize>,
    vector_name: Option<String>,
    filters: Option<DiagnosticSearchFilters>,
}

/// `search_semantic_artifacts` request body.
#[derive(Debug, Deserialize)]
struct SearchSemanticArtifactsRequest {
    query: String,
    limit: Option<usize>,
    top_k: Option<usize>,
    vector_name: Option<String>,
    filters: Option<SemanticArtifactSearchFilters>,
}

/// Optional filters accepted by `search_semantic_artifacts`.
#[derive(Debug, Deserialize)]
struct SemanticArtifactSearchFilters {
    workspace_id: Option<String>,
    file_path: Option<String>,
    symbol_id: Option<String>,
}

/// `search_syntax_artifacts` request body.
#[derive(Debug, Deserialize)]
struct SearchSyntaxArtifactsRequest {
    query: String,
    limit: Option<usize>,
    top_k: Option<usize>,
    vector_name: Option<String>,
    filters: Option<SyntaxArtifactSearchFilters>,
}

/// Optional filters accepted by `search_syntax_artifacts`.
#[derive(Debug, Deserialize)]
struct SyntaxArtifactSearchFilters {
    workspace_id: Option<String>,
    file_path: Option<String>,
    symbol_id: Option<String>,
}

/// `search_crate_graph` request body.
#[derive(Debug, Deserialize)]
struct SearchCrateGraphRequest {
    query: String,
    limit: Option<usize>,
    top_k: Option<usize>,
    vector_name: Option<String>,
    filters: Option<CrateGraphSearchFilters>,
}

/// Optional filters accepted by `search_crate_graph`.
#[derive(Debug, Deserialize)]
struct CrateGraphSearchFilters {
    workspace_id: Option<String>,
    crate_name: Option<String>,
    crate_root: Option<String>,
}

/// `get_file_context` request body.
#[derive(Debug, Deserialize)]
struct FileContextRequest {
    file_path: String,
    limit: Option<usize>,
}

/// `explain_relevance` request body.
#[derive(Debug, Deserialize)]
struct ExplainRelevanceRequest {
    query: String,
    point_ids: Option<Vec<String>>,
    limit: Option<usize>,
    vector_name: Option<String>,
    filters: Option<SearchFilters>,
    score_threshold: Option<f64>,
}

/// Entry point: starts shared services and both communication planes.
#[tokio::main]
async fn main() -> Result<()> {
    let config = RuntimeConfig::default();
    let mcp_bind_addr = format!("127.0.0.1:{}", config.mcp_port);
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

    let mcp_listener: TcpListener = TcpListener::bind(&mcp_bind_addr)
        .await
        .with_context(|| format!("failed to bind MCP endpoint at {mcp_bind_addr}"))?;
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
    let relation_provider = EmbeddingBuilder::<Ollama>::new()
        .base_url(&config.ollama_base_url)
        .model(&config.embedding_model)
        .build()
        .context("failed to build relation embedding client")?;
    let metadata_provider = EmbeddingBuilder::<Ollama>::new()
        .base_url(&config.ollama_base_url)
        .model(&config.embedding_model)
        .build()
        .context("failed to build metadata embedding client")?;
    let file_provider = EmbeddingBuilder::<Ollama>::new()
        .base_url(&config.ollama_base_url)
        .model(&config.embedding_model)
        .build()
        .context("failed to build file embedding client")?;
    let call_edge_provider = EmbeddingBuilder::<Ollama>::new()
        .base_url(&config.ollama_base_url)
        .model(&config.embedding_model)
        .build()
        .context("failed to build call-edge embedding client")?;
    let type_edge_provider = EmbeddingBuilder::<Ollama>::new()
        .base_url(&config.ollama_base_url)
        .model(&config.embedding_model)
        .build()
        .context("failed to build type-edge embedding client")?;
    let diagnostic_provider = EmbeddingBuilder::<Ollama>::new()
        .base_url(&config.ollama_base_url)
        .model(&config.embedding_model)
        .build()
        .context("failed to build diagnostic embedding client")?;
    let semantic_provider = EmbeddingBuilder::<Ollama>::new()
        .base_url(&config.ollama_base_url)
        .model(&config.embedding_model)
        .build()
        .context("failed to build semantic artifact embedding client")?;
    let syntax_provider = EmbeddingBuilder::<Ollama>::new()
        .base_url(&config.ollama_base_url)
        .model(&config.embedding_model)
        .build()
        .context("failed to build syntax artifact embedding client")?;
    let inlay_provider = EmbeddingBuilder::<Ollama>::new()
        .base_url(&config.ollama_base_url)
        .model(&config.embedding_model)
        .build()
        .context("failed to build inlay artifact embedding client")?;
    let crate_graph_provider = EmbeddingBuilder::<Ollama>::new()
        .base_url(&config.ollama_base_url)
        .model(&config.embedding_model)
        .build()
        .context("failed to build crate-graph embedding client")?;

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

    let relation_store = if let Some(api_key) = &config.qdrant_api_key {
        QdrantVectorStore::with_api_key(
            relation_provider,
            config.qdrant_url.clone(),
            config.qdrant_relation_collection.clone(),
            Some(api_key.clone()),
        )
    } else {
        QdrantVectorStore::new(
            relation_provider,
            config.qdrant_url.clone(),
            config.qdrant_relation_collection.clone(),
        )
    }
    .context("failed to initialize relation Qdrant store")?;

    let metadata_store = if let Some(api_key) = &config.qdrant_api_key {
        QdrantVectorStore::with_api_key(
            metadata_provider,
            config.qdrant_url.clone(),
            config.qdrant_metadata_collection.clone(),
            Some(api_key.clone()),
        )
    } else {
        QdrantVectorStore::new(
            metadata_provider,
            config.qdrant_url.clone(),
            config.qdrant_metadata_collection.clone(),
        )
    }
    .context("failed to initialize metadata Qdrant store")?;

    let file_store = if let Some(api_key) = &config.qdrant_api_key {
        QdrantVectorStore::with_api_key(
            file_provider,
            config.qdrant_url.clone(),
            config.qdrant_file_collection.clone(),
            Some(api_key.clone()),
        )
    } else {
        QdrantVectorStore::new(
            file_provider,
            config.qdrant_url.clone(),
            config.qdrant_file_collection.clone(),
        )
    }
    .context("failed to initialize file Qdrant store")?;

    let call_edge_store = if let Some(api_key) = &config.qdrant_api_key {
        QdrantVectorStore::with_api_key(
            call_edge_provider,
            config.qdrant_url.clone(),
            config.qdrant_call_edge_collection.clone(),
            Some(api_key.clone()),
        )
    } else {
        QdrantVectorStore::new(
            call_edge_provider,
            config.qdrant_url.clone(),
            config.qdrant_call_edge_collection.clone(),
        )
    }
    .context("failed to initialize call-edge Qdrant store")?;

    let type_edge_store = if let Some(api_key) = &config.qdrant_api_key {
        QdrantVectorStore::with_api_key(
            type_edge_provider,
            config.qdrant_url.clone(),
            config.qdrant_type_edge_collection.clone(),
            Some(api_key.clone()),
        )
    } else {
        QdrantVectorStore::new(
            type_edge_provider,
            config.qdrant_url.clone(),
            config.qdrant_type_edge_collection.clone(),
        )
    }
    .context("failed to initialize type-edge Qdrant store")?;

    let diagnostic_store = if let Some(api_key) = &config.qdrant_api_key {
        QdrantVectorStore::with_api_key(
            diagnostic_provider,
            config.qdrant_url.clone(),
            config.qdrant_diagnostic_collection.clone(),
            Some(api_key.clone()),
        )
    } else {
        QdrantVectorStore::new(
            diagnostic_provider,
            config.qdrant_url.clone(),
            config.qdrant_diagnostic_collection.clone(),
        )
    }
    .context("failed to initialize diagnostic Qdrant store")?;
    let semantic_store = if let Some(api_key) = &config.qdrant_api_key {
        QdrantVectorStore::with_api_key(
            semantic_provider,
            config.qdrant_url.clone(),
            config.qdrant_semantic_collection.clone(),
            Some(api_key.clone()),
        )
    } else {
        QdrantVectorStore::new(
            semantic_provider,
            config.qdrant_url.clone(),
            config.qdrant_semantic_collection.clone(),
        )
    }
    .context("failed to initialize semantic artifact Qdrant store")?;
    let syntax_store = if let Some(api_key) = &config.qdrant_api_key {
        QdrantVectorStore::with_api_key(
            syntax_provider,
            config.qdrant_url.clone(),
            config.qdrant_syntax_collection.clone(),
            Some(api_key.clone()),
        )
    } else {
        QdrantVectorStore::new(
            syntax_provider,
            config.qdrant_url.clone(),
            config.qdrant_syntax_collection.clone(),
        )
    }
    .context("failed to initialize syntax artifact Qdrant store")?;
    let inlay_store = if let Some(api_key) = &config.qdrant_api_key {
        QdrantVectorStore::with_api_key(
            inlay_provider,
            config.qdrant_url.clone(),
            config.qdrant_inlay_collection.clone(),
            Some(api_key.clone()),
        )
    } else {
        QdrantVectorStore::new(
            inlay_provider,
            config.qdrant_url.clone(),
            config.qdrant_inlay_collection.clone(),
        )
    }
    .context("failed to initialize inlay artifact Qdrant store")?;
    let crate_graph_store = if let Some(api_key) = &config.qdrant_api_key {
        QdrantVectorStore::with_api_key(
            crate_graph_provider,
            config.qdrant_url.clone(),
            config.qdrant_crate_graph_collection.clone(),
            Some(api_key.clone()),
        )
    } else {
        QdrantVectorStore::new(
            crate_graph_provider,
            config.qdrant_url.clone(),
            config.qdrant_crate_graph_collection.clone(),
        )
    }
    .context("failed to initialize crate-graph Qdrant store")?;

    Ok(Services {
        store,
        relation_store,
        metadata_store,
        file_store,
        call_edge_store,
        type_edge_store,
        diagnostic_store,
        semantic_store,
        syntax_store,
        inlay_store,
        crate_graph_store,
    })
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

                let mut file_batch: HashMap<PathBuf, DirtyEvent> = HashMap::new();
                let mut workspace_refresh_queued = false;
                insert_batch_event(&mut file_batch, &mut workspace_refresh_queued, first);

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
                                Some(event) => insert_batch_event(
                                    &mut file_batch,
                                    &mut workspace_refresh_queued,
                                    event,
                                ),
                                None => break,
                            }
                        }
                    }
                }

                let mut events = file_batch.into_values().collect::<Vec<_>>();
                if workspace_refresh_queued {
                    events.push(DirtyEvent::RefreshWorkspace);
                }
                process_batch(app.clone(), events).await;
            }
        }
    }
}

/// Inserts/overwrites an event in the current batch keyed by file path.
fn insert_batch_event(
    file_batch: &mut HashMap<PathBuf, DirtyEvent>,
    workspace_refresh_queued: &mut bool,
    event: DirtyEvent,
) {
    // Last-write-wins per path inside a batch.
    match &event {
        DirtyEvent::Index(path) | DirtyEvent::Delete(path) => {
            file_batch.insert(path.clone(), event);
        }
        DirtyEvent::RefreshWorkspace => *workspace_refresh_queued = true,
    }
}

/// Processes one debounced batch and updates status state.
async fn process_batch(app: App, events: Vec<DirtyEvent>) {
    let batch_bulk_mode = {
        let mut state = app.state.write().await;
        let queued_before = state.queue_depth;
        state.indexing_in_progress = true;
        state.queue_depth = state.queue_depth.saturating_sub(events.len());
        queued_before > BULK_SCAN_QUEUE_DEPTH_THRESHOLD || events.len() > 1
    };

    let services = app.services.read().await.clone();

    for event in events {
        let result = match event {
            DirtyEvent::Index(path) => reindex_file(&app, &services, &path, batch_bulk_mode).await,
            DirtyEvent::Delete(path) => delete_file_chunks(&app, &path).await,
            DirtyEvent::RefreshWorkspace => refresh_workspace_graph_with_retry(&app).await,
        };

        if let Err(err) = result {
            let mut state = app.state.write().await;
            state.last_error = Some(format!("{err:#}"));
        }
    }

    let mut state = app.state.write().await;
    state.indexing_in_progress = false;
    state.indexed_at_unix_ms = Some(unix_ms_now());
    let should_enqueue_deferred_graph_refresh = state.queue_depth == 0
        && state.pending_initial_graph_refresh
        && !state.initial_graph_refresh_done;
    if should_enqueue_deferred_graph_refresh {
        state.pending_initial_graph_refresh = false;
        state.initial_graph_refresh_done = true;
        state.queue_depth = state.queue_depth.saturating_add(1);
    }
    drop(state);

    if should_enqueue_deferred_graph_refresh {
        if app
            .dirty_tx
            .send(DirtyEvent::RefreshWorkspace)
            .await
            .is_err()
        {
            let mut state = app.state.write().await;
            state.queue_depth = state.queue_depth.saturating_sub(1);
            state.last_error =
                Some("index queue unavailable for deferred workspace refresh".to_string());
        }
    }
}

/// Reads and re-indexes one Rust file into Qdrant and local cache.
async fn reindex_file(app: &App, services: &Services, path: &Path, bulk_mode: bool) -> Result<()> {
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

    let (
        previous_ids,
        previous_relation_ids,
        previous_file_doc_ids,
        previous_call_edge_ids,
        previous_type_edge_ids,
        previous_diagnostic_ids,
        previous_semantic_ids,
        previous_syntax_ids,
        previous_inlay_ids,
    ) = {
        let state = app.state.read().await;
        (
            state
                .indexed_ids_by_file
                .get(&file_rel)
                .cloned()
                .unwrap_or_default(),
            state
                .indexed_relation_ids_by_file
                .get(&file_rel)
                .cloned()
                .unwrap_or_default(),
            state
                .indexed_file_doc_ids_by_file
                .get(&file_rel)
                .cloned()
                .unwrap_or_default(),
            state
                .indexed_call_edge_ids_by_file
                .get(&file_rel)
                .cloned()
                .unwrap_or_default(),
            state
                .indexed_type_edge_ids_by_file
                .get(&file_rel)
                .cloned()
                .unwrap_or_default(),
            state
                .indexed_diagnostic_ids_by_file
                .get(&file_rel)
                .cloned()
                .unwrap_or_default(),
            state
                .indexed_semantic_ids_by_file
                .get(&file_rel)
                .cloned()
                .unwrap_or_default(),
            state
                .indexed_syntax_ids_by_file
                .get(&file_rel)
                .cloned()
                .unwrap_or_default(),
            state
                .indexed_inlay_ids_by_file
                .get(&file_rel)
                .cloned()
                .unwrap_or_default(),
        )
    };

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
        if !previous_relation_ids.is_empty() {
            let stale_relation_ids = previous_relation_ids.into_iter().collect::<Vec<_>>();
            services
                .relation_store
                .delete_documents_by_ids(&stale_relation_ids)
                .await
                .with_context(|| format!("qdrant relation delete failed for {}", file_rel))?;
        }
        if !previous_file_doc_ids.is_empty() {
            let stale_file_doc_ids = previous_file_doc_ids.into_iter().collect::<Vec<_>>();
            services
                .file_store
                .delete_documents_by_ids(&stale_file_doc_ids)
                .await
                .with_context(|| format!("qdrant file-doc delete failed for {}", file_rel))?;
        }
        if !previous_call_edge_ids.is_empty() {
            let stale_call_edge_ids = previous_call_edge_ids.into_iter().collect::<Vec<_>>();
            services
                .call_edge_store
                .delete_documents_by_ids(&stale_call_edge_ids)
                .await
                .with_context(|| format!("qdrant call-edge delete failed for {}", file_rel))?;
        }
        if !previous_type_edge_ids.is_empty() {
            let stale_type_edge_ids = previous_type_edge_ids.into_iter().collect::<Vec<_>>();
            services
                .type_edge_store
                .delete_documents_by_ids(&stale_type_edge_ids)
                .await
                .with_context(|| format!("qdrant type-edge delete failed for {}", file_rel))?;
        }
        if !previous_diagnostic_ids.is_empty() {
            let stale_diagnostic_ids = previous_diagnostic_ids.into_iter().collect::<Vec<_>>();
            services
                .diagnostic_store
                .delete_documents_by_ids(&stale_diagnostic_ids)
                .await
                .with_context(|| format!("qdrant diagnostic delete failed for {}", file_rel))?;
        }
        if !previous_semantic_ids.is_empty() {
            let stale_semantic_ids = previous_semantic_ids.into_iter().collect::<Vec<_>>();
            services
                .semantic_store
                .delete_documents_by_ids(&stale_semantic_ids)
                .await
                .with_context(|| format!("qdrant semantic delete failed for {}", file_rel))?;
        }
        if !previous_syntax_ids.is_empty() {
            let stale_syntax_ids = previous_syntax_ids.into_iter().collect::<Vec<_>>();
            services
                .syntax_store
                .delete_documents_by_ids(&stale_syntax_ids)
                .await
                .with_context(|| format!("qdrant syntax delete failed for {}", file_rel))?;
        }
        if !previous_inlay_ids.is_empty() {
            let stale_inlay_ids = previous_inlay_ids.into_iter().collect::<Vec<_>>();
            services
                .inlay_store
                .delete_documents_by_ids(&stale_inlay_ids)
                .await
                .with_context(|| format!("qdrant inlay delete failed for {}", file_rel))?;
        }
        let mut state = app.state.write().await;
        state.rust_items_by_file.remove(&file_rel);
        state.relations_by_file.remove(&file_rel);
        state.chunks_by_file.remove(&file_rel);
        state.file_docs_by_file.remove(&file_rel);
        state.call_edges_by_file.remove(&file_rel);
        state.type_edges_by_file.remove(&file_rel);
        state.diagnostics_by_file.remove(&file_rel);
        state.semantic_tokens_by_file.remove(&file_rel);
        state.syntax_trees_by_file.remove(&file_rel);
        state.inlay_hints_by_file.remove(&file_rel);
        state.indexed_ids_by_file.remove(&file_rel);
        state.indexed_relation_ids_by_file.remove(&file_rel);
        state.indexed_file_doc_ids_by_file.remove(&file_rel);
        state.indexed_call_edge_ids_by_file.remove(&file_rel);
        state.indexed_type_edge_ids_by_file.remove(&file_rel);
        state.indexed_diagnostic_ids_by_file.remove(&file_rel);
        state.indexed_semantic_ids_by_file.remove(&file_rel);
        state.indexed_syntax_ids_by_file.remove(&file_rel);
        state.indexed_inlay_ids_by_file.remove(&file_rel);
        return Ok(());
    }

    // Symbol-aware enrichment stage. When symbols are available, they replace
    // coarse chunk fallback docs for the same file.
    let mut file_metrics = FileExtractionMetrics::default();
    let (
        symbol_docs,
        mut relation_docs,
        diagnostics,
        semantic_token_docs,
        syntax_tree_docs,
        inlay_hint_docs,
    ) = {
        let mut ra = app.ra_manager.lock().await;
        let counters_before = ra.session_counters();
        match ra
            .extract_symbol_docs(
                &cfg.workspace_root,
                &workspace_id,
                path,
                &file_rel,
                &file_uri,
                &content,
                bulk_mode,
                &mut file_metrics,
            )
            .await
        {
            Ok(docs) if !docs.is_empty() => {
                let relations = match ra
                    .extract_symbol_relations(
                        &cfg.workspace_root,
                        &workspace_id,
                        &file_rel,
                        &file_uri,
                        &content,
                        &docs,
                        bulk_mode,
                        &mut file_metrics,
                    )
                    .await
                {
                    Ok(v) => v,
                    Err(_) => Vec::new(),
                };
                let (semantic_docs, syntax_docs, inlay_docs) = match ra
                    .extract_file_artifacts(
                        &cfg.workspace_root,
                        &workspace_id,
                        &file_rel,
                        &file_uri,
                        &content,
                        &docs,
                        cfg.enable_syntax_artifacts,
                        &mut file_metrics,
                    )
                    .await
                {
                    Ok(v) => v,
                    Err(_) => (Vec::new(), Vec::new(), Vec::new()),
                };
                let diagnostics = ra
                    .extract_file_diagnostics(
                        &cfg.workspace_root,
                        &workspace_id,
                        &file_rel,
                        &file_uri,
                    )
                    .await
                    .unwrap_or_default();
                let counters_after = ra.session_counters();
                file_metrics.content_modified_retries = counters_after
                    .content_modified_retries_total
                    .saturating_sub(counters_before.content_modified_retries_total);
                file_metrics.request_timeouts = counters_after
                    .request_timeouts_total
                    .saturating_sub(counters_before.request_timeouts_total);
                (
                    docs,
                    relations,
                    diagnostics,
                    semantic_docs,
                    syntax_docs,
                    inlay_docs,
                )
            }
            Ok(_) => {
                file_metrics.fallback_heuristic = true;
                let counters_after = ra.session_counters();
                file_metrics.content_modified_retries = counters_after
                    .content_modified_retries_total
                    .saturating_sub(counters_before.content_modified_retries_total);
                file_metrics.request_timeouts = counters_after
                    .request_timeouts_total
                    .saturating_sub(counters_before.request_timeouts_total);
                (
                    extract_symbol_docs_heuristic(&workspace_id, &file_rel, &file_uri, &content),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                )
            }
            Err(err) => {
                file_metrics.fallback_heuristic = true;
                let counters_after = ra.session_counters();
                file_metrics.content_modified_retries = counters_after
                    .content_modified_retries_total
                    .saturating_sub(counters_before.content_modified_retries_total);
                file_metrics.request_timeouts = counters_after
                    .request_timeouts_total
                    .saturating_sub(counters_before.request_timeouts_total);
                let _ = err;
                (
                    extract_symbol_docs_heuristic(&workspace_id, &file_rel, &file_uri, &content),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                )
            }
        }
    };

    let mut final_docs = if symbol_docs.is_empty() {
        coarse_chunks_to_symbol_docs(&coarse_chunks)
    } else {
        symbol_docs
    };
    let metadata_docs_by_crate = app.state.read().await.metadata_docs_by_crate.clone();
    if let Some(crate_meta) = find_crate_metadata_for_file(&metadata_docs_by_crate, &file_rel) {
        apply_crate_metadata_to_symbol_docs(&mut final_docs, crate_meta);
        apply_crate_metadata_to_relation_docs(&mut relation_docs, crate_meta);
    }
    apply_artifact_summaries_to_symbols(
        &mut final_docs,
        &semantic_token_docs,
        &syntax_tree_docs,
        &inlay_hint_docs,
    );

    let should_discard_for_warmup = {
        let state = app.state.read().await;
        !state.is_ra_warm
    };
    if should_discard_for_warmup {
        let requeue_needed = {
            let mut state = app.state.write().await;
            if state.is_ra_warm {
                false
            } else {
                state.is_ra_warm = true;
                state.queue_depth = state.queue_depth.saturating_add(1);
                true
            }
        };

        if requeue_needed {
            app.dirty_tx
                .send(DirtyEvent::Index(path.to_path_buf()))
                .await
                .map_err(|_| anyhow::anyhow!("index queue unavailable during warm-up requeue"))?;
        }

        return Ok(());
    }

    let should_discard_implementations_for_warmup = {
        let state = app.state.read().await;
        !state.is_ra_warm_impl && file_metrics.implementations_failed > 0
    };
    if should_discard_implementations_for_warmup {
        relation_docs.retain(|doc| doc.relation_kind != "implementations");
        file_metrics.implementations_success = 0;
        file_metrics.implementations_failed = 0;
        file_metrics.implementations_nonempty = 0;
        file_metrics.relations_implementations_emitted = 0;
        file_metrics.type_definitions_success = 0;
        file_metrics.type_definitions_failed = 0;
        file_metrics.type_definitions_nonempty = 0;
        file_metrics.relations_type_definitions_emitted = 0;
        let requeue_needed = {
            let mut state = app.state.write().await;
            if state.is_ra_warm_impl {
                false
            } else {
                state.is_ra_warm_impl = true;
                state.queue_depth = state.queue_depth.saturating_add(1);
                true
            }
        };

        if requeue_needed {
            app.dirty_tx
                .send(DirtyEvent::Index(path.to_path_buf()))
                .await
                .map_err(|_| {
                    anyhow::anyhow!("index queue unavailable during impl warm-up requeue")
                })?;
        }
    }

    let crate_name = find_crate_metadata_for_file(&metadata_docs_by_crate, &file_rel)
        .map(|m| m.crate_name.clone())
        .unwrap_or_default();
    let file_doc = build_file_doc(&workspace_id, &file_rel, &file_uri, &content, &crate_name);
    let (call_edges, type_edges) =
        build_typed_edges_from_relations(&workspace_id, &relation_docs, &final_docs);
    let symbol_rows = final_docs
        .iter()
        .map(|doc| NamedVectorDocument {
            id: doc.id.clone(),
            raw: doc.clone(),
            vectors: debug_oversize_named_vectors(&doc.id, rust_item_named_vectors(doc)),
        })
        .collect::<Vec<_>>();

    services
        .store
        .insert_documents_with_named_vectors(symbol_rows)
        .await
        .with_context(|| format!("qdrant symbol upsert failed for {}", file_rel))?;

    if !relation_docs.is_empty() {
        let relation_rows = relation_docs
            .iter()
            .map(|doc| NamedVectorDocument {
                id: doc.id.clone(),
                raw: doc.clone(),
                vectors: debug_oversize_named_vectors(&doc.id, relation_named_vectors(doc)),
            })
            .collect::<Vec<_>>();
        services
            .relation_store
            .insert_documents_with_named_vectors(relation_rows)
            .await
            .with_context(|| format!("qdrant relation upsert failed for {}", file_rel))?;
    }

    let file_rows = vec![NamedVectorDocument {
        id: file_doc.id.clone(),
        raw: file_doc.clone(),
        vectors: debug_oversize_named_vectors(&file_doc.id, file_doc_named_vectors(&file_doc)),
    }];
    services
        .file_store
        .insert_documents_with_named_vectors(file_rows)
        .await
        .with_context(|| format!("qdrant file-doc upsert failed for {}", file_rel))?;

    if !call_edges.is_empty() {
        let call_edge_rows = call_edges
            .iter()
            .map(|doc| NamedVectorDocument {
                id: doc.id.clone(),
                raw: doc.clone(),
                vectors: debug_oversize_named_vectors(&doc.id, call_edge_named_vectors(doc)),
            })
            .collect::<Vec<_>>();
        services
            .call_edge_store
            .insert_documents_with_named_vectors(call_edge_rows)
            .await
            .with_context(|| format!("qdrant call-edge upsert failed for {}", file_rel))?;
    }

    if !type_edges.is_empty() {
        let type_edge_rows = type_edges
            .iter()
            .map(|doc| NamedVectorDocument {
                id: doc.id.clone(),
                raw: doc.clone(),
                vectors: debug_oversize_named_vectors(&doc.id, type_edge_named_vectors(doc)),
            })
            .collect::<Vec<_>>();
        services
            .type_edge_store
            .insert_documents_with_named_vectors(type_edge_rows)
            .await
            .with_context(|| format!("qdrant type-edge upsert failed for {}", file_rel))?;
    }

    if !diagnostics.is_empty() {
        let diagnostic_rows = diagnostics
            .iter()
            .map(|doc| NamedVectorDocument {
                id: doc.id.clone(),
                raw: doc.clone(),
                vectors: debug_oversize_named_vectors(&doc.id, diagnostic_named_vectors(doc)),
            })
            .collect::<Vec<_>>();
        services
            .diagnostic_store
            .insert_documents_with_named_vectors(diagnostic_rows)
            .await
            .with_context(|| format!("qdrant diagnostic upsert failed for {}", file_rel))?;
    }
    if !semantic_token_docs.is_empty() {
        let semantic_rows = semantic_token_docs
            .iter()
            .map(|doc| NamedVectorDocument {
                id: doc.id.clone(),
                raw: doc.clone(),
                vectors: debug_oversize_named_vectors(
                    &doc.id,
                    semantic_artifact_named_vectors(doc),
                ),
            })
            .collect::<Vec<_>>();
        services
            .semantic_store
            .insert_documents_with_named_vectors(semantic_rows)
            .await
            .with_context(|| format!("qdrant semantic upsert failed for {}", file_rel))?;
    }
    if !syntax_tree_docs.is_empty() {
        let syntax_rows = syntax_tree_docs
            .iter()
            .map(|doc| NamedVectorDocument {
                id: doc.id.clone(),
                raw: doc.clone(),
                vectors: debug_oversize_named_vectors(&doc.id, syntax_artifact_named_vectors(doc)),
            })
            .collect::<Vec<_>>();
        services
            .syntax_store
            .insert_documents_with_named_vectors(syntax_rows)
            .await
            .with_context(|| format!("qdrant syntax upsert failed for {}", file_rel))?;
    }
    if !inlay_hint_docs.is_empty() {
        let inlay_rows = inlay_hint_docs
            .iter()
            .map(|doc| NamedVectorDocument {
                id: doc.id.clone(),
                raw: doc.clone(),
                vectors: debug_oversize_named_vectors(&doc.id, inlay_artifact_named_vectors(doc)),
            })
            .collect::<Vec<_>>();
        services
            .inlay_store
            .insert_documents_with_named_vectors(inlay_rows)
            .await
            .with_context(|| format!("qdrant inlay upsert failed for {}", file_rel))?;
    }

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
    let final_relation_ids = relation_docs
        .iter()
        .map(|doc| doc.id.clone())
        .collect::<HashSet<_>>();
    let final_file_doc_ids = [file_doc.id.clone()].into_iter().collect::<HashSet<_>>();
    let final_call_edge_ids = call_edges
        .iter()
        .map(|doc| doc.id.clone())
        .collect::<HashSet<_>>();
    let final_type_edge_ids = type_edges
        .iter()
        .map(|doc| doc.id.clone())
        .collect::<HashSet<_>>();
    let final_diagnostic_ids = diagnostics
        .iter()
        .map(|doc| doc.id.clone())
        .collect::<HashSet<_>>();
    let final_semantic_ids = semantic_token_docs
        .iter()
        .map(|doc| doc.id.clone())
        .collect::<HashSet<_>>();
    let final_syntax_ids = syntax_tree_docs
        .iter()
        .map(|doc| doc.id.clone())
        .collect::<HashSet<_>>();
    let final_inlay_ids = inlay_hint_docs
        .iter()
        .map(|doc| doc.id.clone())
        .collect::<HashSet<_>>();
    let stale_relation_ids = previous_relation_ids
        .difference(&final_relation_ids)
        .cloned()
        .collect::<Vec<_>>();
    if !stale_relation_ids.is_empty() {
        services
            .relation_store
            .delete_documents_by_ids(&stale_relation_ids)
            .await
            .with_context(|| format!("qdrant relation stale delete failed for {}", file_rel))?;
    }
    let stale_file_doc_ids = previous_file_doc_ids
        .difference(&final_file_doc_ids)
        .cloned()
        .collect::<Vec<_>>();
    if !stale_file_doc_ids.is_empty() {
        services
            .file_store
            .delete_documents_by_ids(&stale_file_doc_ids)
            .await
            .with_context(|| format!("qdrant file-doc stale delete failed for {}", file_rel))?;
    }
    let stale_call_edge_ids = previous_call_edge_ids
        .difference(&final_call_edge_ids)
        .cloned()
        .collect::<Vec<_>>();
    if !stale_call_edge_ids.is_empty() {
        services
            .call_edge_store
            .delete_documents_by_ids(&stale_call_edge_ids)
            .await
            .with_context(|| format!("qdrant call-edge stale delete failed for {}", file_rel))?;
    }
    let stale_type_edge_ids = previous_type_edge_ids
        .difference(&final_type_edge_ids)
        .cloned()
        .collect::<Vec<_>>();
    if !stale_type_edge_ids.is_empty() {
        services
            .type_edge_store
            .delete_documents_by_ids(&stale_type_edge_ids)
            .await
            .with_context(|| format!("qdrant type-edge stale delete failed for {}", file_rel))?;
    }
    let stale_diagnostic_ids = previous_diagnostic_ids
        .difference(&final_diagnostic_ids)
        .cloned()
        .collect::<Vec<_>>();
    if !stale_diagnostic_ids.is_empty() {
        services
            .diagnostic_store
            .delete_documents_by_ids(&stale_diagnostic_ids)
            .await
            .with_context(|| format!("qdrant diagnostic stale delete failed for {}", file_rel))?;
    }
    let stale_semantic_ids = previous_semantic_ids
        .difference(&final_semantic_ids)
        .cloned()
        .collect::<Vec<_>>();
    if !stale_semantic_ids.is_empty() {
        services
            .semantic_store
            .delete_documents_by_ids(&stale_semantic_ids)
            .await
            .with_context(|| format!("qdrant semantic stale delete failed for {}", file_rel))?;
    }
    let stale_syntax_ids = previous_syntax_ids
        .difference(&final_syntax_ids)
        .cloned()
        .collect::<Vec<_>>();
    if !stale_syntax_ids.is_empty() {
        services
            .syntax_store
            .delete_documents_by_ids(&stale_syntax_ids)
            .await
            .with_context(|| format!("qdrant syntax stale delete failed for {}", file_rel))?;
    }
    let stale_inlay_ids = previous_inlay_ids
        .difference(&final_inlay_ids)
        .cloned()
        .collect::<Vec<_>>();
    if !stale_inlay_ids.is_empty() {
        services
            .inlay_store
            .delete_documents_by_ids(&stale_inlay_ids)
            .await
            .with_context(|| format!("qdrant inlay stale delete failed for {}", file_rel))?;
    }

    file_metrics.symbols_indexed = final_docs.len();
    file_metrics.relations_indexed = relation_docs.len();
    file_metrics.diagnostics_indexed = diagnostics.len();
    let mut state = app.state.write().await;
    state
        .rust_items_by_file
        .insert(file_rel.clone(), final_docs);
    state
        .relations_by_file
        .insert(file_rel.clone(), relation_docs.clone());
    state.file_docs_by_file.insert(file_rel.clone(), file_doc);
    state
        .call_edges_by_file
        .insert(file_rel.clone(), call_edges);
    state
        .type_edges_by_file
        .insert(file_rel.clone(), type_edges);
    state
        .diagnostics_by_file
        .insert(file_rel.clone(), diagnostics);
    state
        .semantic_tokens_by_file
        .insert(file_rel.clone(), semantic_token_docs);
    state
        .syntax_trees_by_file
        .insert(file_rel.clone(), syntax_tree_docs);
    state
        .inlay_hints_by_file
        .insert(file_rel.clone(), inlay_hint_docs);
    state.chunks_by_file.insert(file_rel.clone(), final_chunks);
    state
        .indexed_ids_by_file
        .insert(file_rel.clone(), final_ids);
    state
        .indexed_relation_ids_by_file
        .insert(file_rel.clone(), final_relation_ids);
    state
        .indexed_file_doc_ids_by_file
        .insert(file_rel.clone(), final_file_doc_ids);
    state
        .indexed_call_edge_ids_by_file
        .insert(file_rel.clone(), final_call_edge_ids);
    state
        .indexed_type_edge_ids_by_file
        .insert(file_rel.clone(), final_type_edge_ids);
    state
        .indexed_diagnostic_ids_by_file
        .insert(file_rel.clone(), final_diagnostic_ids);
    state
        .indexed_semantic_ids_by_file
        .insert(file_rel.clone(), final_semantic_ids);
    state
        .indexed_syntax_ids_by_file
        .insert(file_rel.clone(), final_syntax_ids);
    state
        .indexed_inlay_ids_by_file
        .insert(file_rel, final_inlay_ids);
    let metrics = &mut state.extraction_metrics;
    metrics.files_reindexed_total += 1;
    if file_metrics.fallback_heuristic {
        metrics.files_fallback_heuristic_total += 1;
    }
    metrics.symbols_indexed_total += file_metrics.symbols_indexed as u64;
    metrics.relations_indexed_total += file_metrics.relations_indexed as u64;
    metrics.hover_success_total += file_metrics.hover_success;
    metrics.hover_nonempty_total += file_metrics.hover_nonempty;
    metrics.hover_failed_total += file_metrics.hover_failed;
    metrics.workspace_symbol_success_total += file_metrics.workspace_symbol_success;
    metrics.workspace_symbol_failed_total += file_metrics.workspace_symbol_failed;
    metrics.workspace_symbol_nonempty_total += file_metrics.workspace_symbol_nonempty;
    // metrics.signature_help_success_total += file_metrics.signature_help_success;
    // metrics.signature_help_failed_total += file_metrics.signature_help_failed;
    metrics.references_success_total += file_metrics.references_success;
    metrics.references_failed_total += file_metrics.references_failed;
    metrics.references_nonempty_total += file_metrics.references_nonempty;
    metrics.implementations_success_total += file_metrics.implementations_success;
    metrics.implementations_failed_total += file_metrics.implementations_failed;
    metrics.implementations_nonempty_total += file_metrics.implementations_nonempty;
    metrics.definitions_success_total += file_metrics.definitions_success;
    metrics.definitions_failed_total += file_metrics.definitions_failed;
    metrics.definitions_nonempty_total += file_metrics.definitions_nonempty;
    metrics.type_definitions_success_total += file_metrics.type_definitions_success;
    metrics.type_definitions_failed_total += file_metrics.type_definitions_failed;
    metrics.type_definitions_nonempty_total += file_metrics.type_definitions_nonempty;
    metrics.relations_references_emitted_total += file_metrics.relations_references_emitted;
    metrics.relations_implementations_emitted_total +=
        file_metrics.relations_implementations_emitted;
    metrics.relations_definitions_emitted_total += file_metrics.relations_definitions_emitted;
    metrics.relations_type_definitions_emitted_total +=
        file_metrics.relations_type_definitions_emitted;
    metrics.content_modified_retries_total += file_metrics.content_modified_retries;
    metrics.request_timeouts_total += file_metrics.request_timeouts;
    metrics.semantic_tokens_success_total += file_metrics.semantic_tokens_success;
    metrics.semantic_tokens_failed_total += file_metrics.semantic_tokens_failed;
    metrics.semantic_tokens_nonempty_total += file_metrics.semantic_tokens_nonempty;
    metrics.syntax_tree_success_total += file_metrics.syntax_tree_success;
    metrics.syntax_tree_failed_total += file_metrics.syntax_tree_failed;
    metrics.syntax_tree_nonempty_total += file_metrics.syntax_tree_nonempty;
    metrics.syntax_tree_unsupported_total += file_metrics.syntax_tree_unsupported;
    metrics.inlay_hints_success_total += file_metrics.inlay_hints_success;
    metrics.inlay_hints_failed_total += file_metrics.inlay_hints_failed;
    metrics.inlay_hints_nonempty_total += file_metrics.inlay_hints_nonempty;
    metrics.diagnostics_indexed_total += file_metrics.diagnostics_indexed as u64;

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

    let (
        relation_ids_to_delete,
        symbol_ids_to_delete,
        file_doc_ids_to_delete,
        call_edge_ids_to_delete,
        type_edge_ids_to_delete,
        diagnostic_ids_to_delete,
        semantic_ids_to_delete,
        syntax_ids_to_delete,
        inlay_ids_to_delete,
    ) = {
        let mut state = app.state.write().await;
        state.rust_items_by_file.remove(&file_rel);
        state.relations_by_file.remove(&file_rel);
        state.chunks_by_file.remove(&file_rel);
        state.file_docs_by_file.remove(&file_rel);
        state.call_edges_by_file.remove(&file_rel);
        state.type_edges_by_file.remove(&file_rel);
        state.diagnostics_by_file.remove(&file_rel);
        state.semantic_tokens_by_file.remove(&file_rel);
        state.syntax_trees_by_file.remove(&file_rel);
        state.inlay_hints_by_file.remove(&file_rel);
        let relation_ids_to_delete = state
            .indexed_relation_ids_by_file
            .remove(&file_rel)
            .unwrap_or_default()
            .into_iter()
            .collect::<Vec<_>>();
        let symbol_ids_to_delete = state
            .indexed_ids_by_file
            .remove(&file_rel)
            .unwrap_or_default()
            .into_iter()
            .collect::<Vec<_>>();
        let file_doc_ids_to_delete = state
            .indexed_file_doc_ids_by_file
            .remove(&file_rel)
            .unwrap_or_default()
            .into_iter()
            .collect::<Vec<_>>();
        let call_edge_ids_to_delete = state
            .indexed_call_edge_ids_by_file
            .remove(&file_rel)
            .unwrap_or_default()
            .into_iter()
            .collect::<Vec<_>>();
        let type_edge_ids_to_delete = state
            .indexed_type_edge_ids_by_file
            .remove(&file_rel)
            .unwrap_or_default()
            .into_iter()
            .collect::<Vec<_>>();
        let diagnostic_ids_to_delete = state
            .indexed_diagnostic_ids_by_file
            .remove(&file_rel)
            .unwrap_or_default()
            .into_iter()
            .collect::<Vec<_>>();
        let semantic_ids_to_delete = state
            .indexed_semantic_ids_by_file
            .remove(&file_rel)
            .unwrap_or_default()
            .into_iter()
            .collect::<Vec<_>>();
        let syntax_ids_to_delete = state
            .indexed_syntax_ids_by_file
            .remove(&file_rel)
            .unwrap_or_default()
            .into_iter()
            .collect::<Vec<_>>();
        let inlay_ids_to_delete = state
            .indexed_inlay_ids_by_file
            .remove(&file_rel)
            .unwrap_or_default()
            .into_iter()
            .collect::<Vec<_>>();
        (
            relation_ids_to_delete,
            symbol_ids_to_delete,
            file_doc_ids_to_delete,
            call_edge_ids_to_delete,
            type_edge_ids_to_delete,
            diagnostic_ids_to_delete,
            semantic_ids_to_delete,
            syntax_ids_to_delete,
            inlay_ids_to_delete,
        )
    };
    if !symbol_ids_to_delete.is_empty() {
        let services = app.services.read().await.clone();
        services
            .store
            .delete_documents_by_ids(&symbol_ids_to_delete)
            .await
            .with_context(|| format!("qdrant delete failed for {}", file_rel))?;
    }
    if !relation_ids_to_delete.is_empty() {
        let services = app.services.read().await.clone();
        services
            .relation_store
            .delete_documents_by_ids(&relation_ids_to_delete)
            .await
            .with_context(|| format!("qdrant relation delete failed for {}", file_rel))?;
    }
    if !file_doc_ids_to_delete.is_empty() {
        let services = app.services.read().await.clone();
        services
            .file_store
            .delete_documents_by_ids(&file_doc_ids_to_delete)
            .await
            .with_context(|| format!("qdrant file-doc delete failed for {}", file_rel))?;
    }
    if !call_edge_ids_to_delete.is_empty() {
        let services = app.services.read().await.clone();
        services
            .call_edge_store
            .delete_documents_by_ids(&call_edge_ids_to_delete)
            .await
            .with_context(|| format!("qdrant call-edge delete failed for {}", file_rel))?;
    }
    if !type_edge_ids_to_delete.is_empty() {
        let services = app.services.read().await.clone();
        services
            .type_edge_store
            .delete_documents_by_ids(&type_edge_ids_to_delete)
            .await
            .with_context(|| format!("qdrant type-edge delete failed for {}", file_rel))?;
    }
    if !diagnostic_ids_to_delete.is_empty() {
        let services = app.services.read().await.clone();
        services
            .diagnostic_store
            .delete_documents_by_ids(&diagnostic_ids_to_delete)
            .await
            .with_context(|| format!("qdrant diagnostic delete failed for {}", file_rel))?;
    }
    if !semantic_ids_to_delete.is_empty() {
        let services = app.services.read().await.clone();
        services
            .semantic_store
            .delete_documents_by_ids(&semantic_ids_to_delete)
            .await
            .with_context(|| format!("qdrant semantic delete failed for {}", file_rel))?;
    }
    if !syntax_ids_to_delete.is_empty() {
        let services = app.services.read().await.clone();
        services
            .syntax_store
            .delete_documents_by_ids(&syntax_ids_to_delete)
            .await
            .with_context(|| format!("qdrant syntax delete failed for {}", file_rel))?;
    }
    if !inlay_ids_to_delete.is_empty() {
        let services = app.services.read().await.clone();
        services
            .inlay_store
            .delete_documents_by_ids(&inlay_ids_to_delete)
            .await
            .with_context(|| format!("qdrant inlay delete failed for {}", file_rel))?;
    }

    Ok(())
}

fn coarse_chunks_to_symbol_docs(chunks: &[CodeChunk]) -> Vec<schema::SymbolDoc> {
    chunks
        .iter()
        .map(|chunk| {
            let mut doc = schema::SymbolDoc {
                id: chunk.chunk_id.clone(),
                kind: normalize_chunk_kind(&chunk.kind).to_string(),
                symbol: format!("chunk_{}_{}", chunk.start_line, chunk.end_line),
                file_path: chunk.file_path.clone(),
                workspace_id: chunk.workspace_id.clone(),
                uri: chunk.uri.clone(),
                module: module_from_file_path(&chunk.file_path),
                symbol_path: String::new(),
                crate_name: String::new(),
                edition: String::new(),
                signature: chunk
                    .text
                    .lines()
                    .next()
                    .unwrap_or_default()
                    .trim()
                    .to_string(),
                docs: String::new(),
                hover_summary: String::new(),
                body_excerpt: chunk.text.clone(),
                start_line: chunk.start_line,
                start_character: 0,
                end_line: chunk.end_line,
                ..Default::default()
            };
            doc.finalize_derived_fields();
            doc
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
    let mut chunks: Vec<CodeChunk> = Vec::new();
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

fn debug_oversize_named_vectors(
    doc_id: &str,
    vectors: HashMap<String, String>,
) -> HashMap<String, String> {
    let mut sanitized = HashMap::with_capacity(vectors.len());
    for (vector_name, text) in vectors {
        let chars = text.chars().count();
        if chars > EMBED_OVERSIZE_DEBUG_CHARS {
            eprintln!(
                "[embed-oversize-debug] doc_id={} vector={} chars={} threshold={}",
                doc_id, vector_name, chars, EMBED_OVERSIZE_DEBUG_CHARS
            );
        }
        if chars > EMBED_HARD_CAP_CHARS {
            eprintln!(
                "[embed-truncate-debug] doc_id={} vector={} chars={} cap={}",
                doc_id, vector_name, chars, EMBED_HARD_CAP_CHARS
            );
            sanitized.insert(
                vector_name,
                text.chars().take(EMBED_HARD_CAP_CHARS).collect::<String>(),
            );
        } else {
            sanitized.insert(vector_name, text);
        }
    }
    sanitized
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

fn rust_item_search_text(doc: &schema::SymbolDoc) -> String {
    format!(
        // "{symbol}\n{symbol_path}\n{signature}\n{signature_help}\n{docs}\n{hover_summary}\n{body_excerpt}\nmodule: {module}\ncrate: {crate_name}\nedition: {edition}\npath: {file_path}\nkind: {kind}\nuri: {uri}\nworkspace_id: {workspace_id}",
        "{symbol}\n{symbol_path}\n{signature}\n{docs}\n{hover_summary}\n{body_excerpt}\nmodule: {module}\ncrate: {crate_name}\nedition: {edition}\npath: {file_path}\nkind: {kind}\nuri: {uri}\nworkspace_id: {workspace_id}",
        symbol = doc.symbol,
        symbol_path = doc.symbol_path,
        signature = doc.signature,
        // signature_help = doc.signature_help,
        docs = doc.docs,
        hover_summary = doc.hover_summary,
        body_excerpt = doc.body_excerpt,
        module = doc.module,
        crate_name = doc.crate_name,
        edition = doc.edition,
        file_path = doc.file_path,
        kind = doc.kind,
        uri = doc.uri,
        workspace_id = doc.workspace_id,
    )
}

fn rust_item_named_vectors(doc: &schema::SymbolDoc) -> HashMap<String, String> {
    HashMap::from([
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
                // "{signature}\n{signature_help}\nsymbol: {symbol}\nkind: {kind}\nmodule: {module}\npath: {file_path}\nworkspace_id: {workspace_id}",
                "{signature}\nsymbol: {symbol}\nkind: {kind}\nmodule: {module}\npath: {file_path}\nworkspace_id: {workspace_id}",
                signature = doc.signature,
                //signature_help = doc.signature_help,
                symbol = doc.symbol,
                kind = doc.kind,
                module = doc.module,
                file_path = doc.file_path,
                workspace_id = doc.workspace_id,
            ),
        ),
        (TYPE_VECTOR_NAME.to_string(), symbol_type_vector_text(doc)),
        (
            SEMANTIC_VECTOR_NAME.to_string(),
            format!(
                "{semantic}\n{inlay}\nsymbol: {symbol}\npath: {path}\nworkspace_id: {workspace_id}",
                semantic = doc.semantic_tokens_summary,
                inlay = doc.inlay_hints_summary,
                symbol = doc.symbol,
                path = doc.file_path,
                workspace_id = doc.workspace_id,
            ),
        ),
        (
            SYNTAX_VECTOR_NAME.to_string(),
            format!(
                "{syntax}\nsymbol: {symbol}\nkind: {kind}\npath: {path}\nworkspace_id: {workspace_id}",
                syntax = doc.syntax_tree_summary,
                symbol = doc.symbol,
                kind = doc.kind,
                path = doc.file_path,
                workspace_id = doc.workspace_id,
            ),
        ),
    ])
}

fn symbol_type_vector_text(doc: &schema::SymbolDoc) -> String {
    format!(
        "kind: {kind}\nvisibility: {visibility}\ngenerics: {generics}\nwhere: {where_clause}\nreceiver: {receiver}\nreturn: {return_type}\nsymbol: {symbol}\npath: {symbol_path}\nworkspace_id: {workspace_id}",
        kind = doc.kind,
        visibility = doc.visibility.clone().unwrap_or_default(),
        generics = doc.generics.clone().unwrap_or_default(),
        where_clause = doc.where_clause.clone().unwrap_or_default(),
        receiver = doc.receiver.clone().unwrap_or_default(),
        return_type = doc.return_type.clone().unwrap_or_default(),
        symbol = doc.symbol,
        symbol_path = doc.symbol_path,
        workspace_id = doc.workspace_id,
    )
}

fn relation_search_text(doc: &schema::GraphEdgeDoc) -> String {
    format!(
        "{relation_kind}\nsource_symbol: {source_symbol}\nsource_crate: {source_crate}\nsource_file: {source_file}\ntarget_file: {target_file}\nexcerpt:\n{excerpt}\nworkspace_id: {workspace_id}",
        relation_kind = doc.relation_kind,
        source_symbol = doc.source_symbol,
        source_crate = doc.source_crate_name,
        source_file = doc.source_file_path,
        target_file = doc.target_file_path,
        excerpt = doc.target_excerpt,
        workspace_id = doc.workspace_id,
    )
}

fn relation_named_vectors(doc: &schema::GraphEdgeDoc) -> HashMap<String, String> {
    let mut vectors = HashMap::from([
        (
            SYMBOL_VECTOR_NAME.to_string(),
            format!(
                "{source_symbol}\nrelation: {relation_kind}\nsource_path: {source_file}\ntarget_path: {target_file}\nworkspace_id: {workspace_id}",
                source_symbol = doc.source_symbol,
                relation_kind = doc.relation_kind,
                source_file = doc.source_file_path,
                target_file = doc.target_file_path,
                workspace_id = doc.workspace_id,
            ),
        ),
        (
            DOCS_VECTOR_NAME.to_string(),
            format!(
                "relation: {relation_kind}\nsource_symbol: {source_symbol}\nsource: {source_file}\ntarget: {target_file}\nworkspace_id: {workspace_id}",
                relation_kind = doc.relation_kind,
                source_symbol = doc.source_symbol,
                source_file = doc.source_file_path,
                target_file = doc.target_file_path,
                workspace_id = doc.workspace_id,
            ),
        ),
        (
            SIGNATURE_VECTOR_NAME.to_string(),
            format!(
                "{relation_kind}\n{source_symbol}\n{source_file}:{line}\nworkspace_id: {workspace_id}",
                relation_kind = doc.relation_kind,
                source_symbol = doc.source_symbol,
                source_file = doc.source_file_path,
                line = doc.target_start_line,
                workspace_id = doc.workspace_id,
            ),
        ),
        (GRAPH_VECTOR_NAME.to_string(), relation_search_text(doc)),
        (
            TYPE_VECTOR_NAME.to_string(),
            format!(
                "edge_type: {edge_type}\nrelation_kind: {relation_kind}\nsource_symbol: {source_symbol}\nsource_crate: {source_crate}\nworkspace_id: {workspace_id}",
                edge_type = doc.edge_type,
                relation_kind = doc.relation_kind,
                source_symbol = doc.source_symbol,
                source_crate = doc.source_crate_name,
                workspace_id = doc.workspace_id,
            ),
        ),
    ]);

    if !doc.target_excerpt.trim().is_empty() {
        vectors.insert(
            BODY_VECTOR_NAME.to_string(),
            format!(
                "{excerpt}\nrelation: {relation_kind}\nsource_symbol: {source_symbol}\nsource: {source_file}\ntarget: {target_file}\nworkspace_id: {workspace_id}",
                excerpt = doc.target_excerpt,
                relation_kind = doc.relation_kind,
                source_symbol = doc.source_symbol,
                source_file = doc.source_file_path,
                target_file = doc.target_file_path,
                workspace_id = doc.workspace_id,
            ),
        );
    }

    vectors
}

fn crate_metadata_search_text(doc: &CrateMetadataDoc) -> String {
    format!(
        "crate: {crate_name}\nedition: {edition}\nroot: {crate_root}\nmanifest: {manifest}\nfeatures: {features}\noptional_dependencies: {optional}\nworkspace_id: {workspace_id}",
        crate_name = doc.crate_name,
        edition = doc.edition,
        crate_root = doc.crate_root,
        manifest = doc.manifest_path,
        features = doc.features.join(","),
        optional = doc.optional_dependencies.join(","),
        workspace_id = doc.workspace_id,
    )
}

fn crate_metadata_named_vectors(doc: &CrateMetadataDoc) -> HashMap<String, String> {
    HashMap::from([
        (
            SYMBOL_VECTOR_NAME.to_string(),
            format!(
                "{crate_name}\ncrate_root: {crate_root}\nworkspace_id: {workspace_id}",
                crate_name = doc.crate_name,
                crate_root = doc.crate_root,
                workspace_id = doc.workspace_id,
            ),
        ),
        (
            DOCS_VECTOR_NAME.to_string(),
            format!(
                "features: {features}\noptional_dependencies: {optional}\ncrate: {crate_name}\nworkspace_id: {workspace_id}",
                features = doc.features.join(","),
                optional = doc.optional_dependencies.join(","),
                crate_name = doc.crate_name,
                workspace_id = doc.workspace_id,
            ),
        ),
        (
            SIGNATURE_VECTOR_NAME.to_string(),
            format!(
                "edition: {edition}\nmanifest: {manifest}\ncrate: {crate_name}\nworkspace_id: {workspace_id}",
                edition = doc.edition,
                manifest = doc.manifest_path,
                crate_name = doc.crate_name,
                workspace_id = doc.workspace_id,
            ),
        ),
    ])
}

fn file_doc_search_text(doc: &schema::FileDoc) -> String {
    format!(
        "file: {file_path}\nmodule: {module}\ncrate: {crate_name}\nuri: {uri}\nworkspace_id: {workspace_id}",
        file_path = doc.file_path,
        module = doc.module,
        crate_name = doc.crate_name,
        uri = doc.uri,
        workspace_id = doc.workspace_id,
    )
}

fn file_doc_named_vectors(doc: &schema::FileDoc) -> HashMap<String, String> {
    HashMap::from([
        (
            SYMBOL_VECTOR_NAME.to_string(),
            format!(
                "{file_path}\nmodule: {module}\ncrate: {crate_name}\nworkspace_id: {workspace_id}",
                file_path = doc.file_path,
                module = doc.module,
                crate_name = doc.crate_name,
                workspace_id = doc.workspace_id,
            ),
        ),
        (DOCS_VECTOR_NAME.to_string(), file_doc_search_text(doc)),
    ])
}

fn call_edge_search_text(doc: &schema::CallEdge) -> String {
    format!(
        "call_edge\nsource_symbol_id: {source}\ntarget_symbol_id: {target}\nsource_file: {source_file}\ntarget_file: {target_file}\nworkspace_id: {workspace_id}",
        source = doc.source_symbol_id,
        target = doc.target_symbol_id,
        source_file = doc.source_span.file_path,
        target_file = doc.target_span.file_path,
        workspace_id = doc.workspace_id,
    )
}

fn call_edge_named_vectors(doc: &schema::CallEdge) -> HashMap<String, String> {
    HashMap::from([
        (
            SYMBOL_VECTOR_NAME.to_string(),
            format!(
                "{source}\n{target}\nworkspace_id: {workspace_id}",
                source = doc.source_symbol_id,
                target = doc.target_symbol_id,
                workspace_id = doc.workspace_id,
            ),
        ),
        (DOCS_VECTOR_NAME.to_string(), call_edge_search_text(doc)),
        (GRAPH_VECTOR_NAME.to_string(), call_edge_search_text(doc)),
    ])
}

fn type_edge_search_text(doc: &schema::TypeEdge) -> String {
    format!(
        "type_edge\nrelation_kind: {relation_kind}\nsource_symbol_id: {source}\ntarget_symbol_id: {target}\nsource_file: {source_file}\ntarget_file: {target_file}\nworkspace_id: {workspace_id}",
        relation_kind = doc.relation_kind,
        source = doc.source_symbol_id,
        target = doc.target_symbol_id,
        source_file = doc.source_span.file_path,
        target_file = doc.target_span.file_path,
        workspace_id = doc.workspace_id,
    )
}

fn type_edge_named_vectors(doc: &schema::TypeEdge) -> HashMap<String, String> {
    HashMap::from([
        (
            SYMBOL_VECTOR_NAME.to_string(),
            format!(
                "{source}\n{target}\nrelation_kind: {relation_kind}\nworkspace_id: {workspace_id}",
                source = doc.source_symbol_id,
                target = doc.target_symbol_id,
                relation_kind = doc.relation_kind,
                workspace_id = doc.workspace_id,
            ),
        ),
        (DOCS_VECTOR_NAME.to_string(), type_edge_search_text(doc)),
        (TYPE_VECTOR_NAME.to_string(), type_edge_search_text(doc)),
        (GRAPH_VECTOR_NAME.to_string(), type_edge_search_text(doc)),
    ])
}

fn diagnostic_search_text(doc: &schema::DiagnosticDoc) -> String {
    format!(
        "diagnostic\nseverity: {severity}\ncode: {code}\nmessage: {message}\nfile: {file_path}\nworkspace_id: {workspace_id}",
        severity = doc.severity,
        code = doc.code.clone().unwrap_or_default(),
        message = doc.message,
        file_path = doc.file_path,
        workspace_id = doc.workspace_id,
    )
}

fn diagnostic_named_vectors(doc: &schema::DiagnosticDoc) -> HashMap<String, String> {
    HashMap::from([
        (
            SYMBOL_VECTOR_NAME.to_string(),
            format!(
                "{severity}\n{file_path}\nworkspace_id: {workspace_id}",
                severity = doc.severity,
                file_path = doc.file_path,
                workspace_id = doc.workspace_id,
            ),
        ),
        (DOCS_VECTOR_NAME.to_string(), diagnostic_search_text(doc)),
    ])
}

fn semantic_artifact_search_text(doc: &schema::SemanticTokenDoc) -> String {
    format!(
        "semantic_tokens\nfile: {file_path}\nsymbol_id: {symbol_id}\ncount: {count}\nhist: {hist}\nsummary:\n{summary}\nworkspace_id: {workspace_id}",
        file_path = doc.file_path,
        symbol_id = doc.symbol_id.clone().unwrap_or_default(),
        count = doc.token_count,
        hist = doc.token_type_histogram.join(","),
        summary = doc.summary,
        workspace_id = doc.workspace_id,
    )
}

fn semantic_artifact_named_vectors(doc: &schema::SemanticTokenDoc) -> HashMap<String, String> {
    HashMap::from([
        (
            SYMBOL_VECTOR_NAME.to_string(),
            format!(
                "{file_path}\n{symbol_id}\nworkspace_id: {workspace_id}",
                file_path = doc.file_path,
                symbol_id = doc.symbol_id.clone().unwrap_or_default(),
                workspace_id = doc.workspace_id,
            ),
        ),
        (
            SEMANTIC_VECTOR_NAME.to_string(),
            semantic_artifact_search_text(doc),
        ),
    ])
}

fn syntax_artifact_search_text(doc: &schema::SyntaxTreeDoc) -> String {
    format!(
        "syntax_tree\nfile: {file_path}\nsymbol_id: {symbol_id}\nhist: {hist}\nsummary:\n{summary}\nworkspace_id: {workspace_id}",
        file_path = doc.file_path,
        symbol_id = doc.symbol_id.clone().unwrap_or_default(),
        hist = doc.node_kind_histogram.join(","),
        summary = doc.summary,
        workspace_id = doc.workspace_id,
    )
}

fn syntax_artifact_named_vectors(doc: &schema::SyntaxTreeDoc) -> HashMap<String, String> {
    HashMap::from([
        (
            SYMBOL_VECTOR_NAME.to_string(),
            format!(
                "{file_path}\n{symbol_id}\nworkspace_id: {workspace_id}",
                file_path = doc.file_path,
                symbol_id = doc.symbol_id.clone().unwrap_or_default(),
                workspace_id = doc.workspace_id,
            ),
        ),
        (
            SYNTAX_VECTOR_NAME.to_string(),
            syntax_artifact_search_text(doc),
        ),
    ])
}

fn inlay_artifact_search_text(doc: &schema::InlayHintDoc) -> String {
    format!(
        "inlay_hints\nfile: {file_path}\nsymbol_id: {symbol_id}\ncount: {count}\nsummary:\n{summary}\nworkspace_id: {workspace_id}",
        file_path = doc.file_path,
        symbol_id = doc.symbol_id.clone().unwrap_or_default(),
        count = doc.hint_count,
        summary = doc.summary,
        workspace_id = doc.workspace_id,
    )
}

fn inlay_artifact_named_vectors(doc: &schema::InlayHintDoc) -> HashMap<String, String> {
    HashMap::from([
        (
            SYMBOL_VECTOR_NAME.to_string(),
            format!(
                "{file_path}\n{symbol_id}\nworkspace_id: {workspace_id}",
                file_path = doc.file_path,
                symbol_id = doc.symbol_id.clone().unwrap_or_default(),
                workspace_id = doc.workspace_id,
            ),
        ),
        (
            SEMANTIC_VECTOR_NAME.to_string(),
            inlay_artifact_search_text(doc),
        ),
    ])
}

fn crate_graph_search_text(doc: &schema::CrateGraphDoc) -> String {
    format!(
        "crate_graph\ncrate: {crate_name}\nroot: {crate_root}\nsummary:\n{summary}\nworkspace_id: {workspace_id}",
        crate_name = doc.crate_name,
        crate_root = doc.crate_root,
        summary = doc.summary,
        workspace_id = doc.workspace_id,
    )
}

fn crate_graph_named_vectors(doc: &schema::CrateGraphDoc) -> HashMap<String, String> {
    HashMap::from([
        (
            SYMBOL_VECTOR_NAME.to_string(),
            format!(
                "{crate_name}\n{crate_root}\nworkspace_id: {workspace_id}",
                crate_name = doc.crate_name,
                crate_root = doc.crate_root,
                workspace_id = doc.workspace_id,
            ),
        ),
        (GRAPH_VECTOR_NAME.to_string(), crate_graph_search_text(doc)),
    ])
}

fn build_file_doc(
    workspace_id: &str,
    file_rel: &str,
    file_uri: &str,
    content: &str,
    crate_name: &str,
) -> schema::FileDoc {
    schema::FileDoc {
        id: format!("file:{workspace_id}:{file_rel}"),
        workspace_id: workspace_id.to_string(),
        crate_name: crate_name.to_string(),
        file_path: file_rel.to_string(),
        uri: file_uri.to_string(),
        module: module_from_file_path(file_rel),
        body_hash: simple_hash(content),
    }
}

fn build_typed_edges_from_relations(
    workspace_id: &str,
    relation_docs: &[schema::GraphEdgeDoc],
    symbol_docs: &[schema::SymbolDoc],
) -> (Vec<schema::CallEdge>, Vec<schema::TypeEdge>) {
    let mut call_edges = Vec::new();
    let mut type_edges = Vec::new();
    let by_id = symbol_docs
        .iter()
        .map(|doc| (doc.id.as_str(), &doc.span))
        .collect::<HashMap<_, _>>();

    for relation in relation_docs {
        let source_span = by_id
            .get(relation.source_symbol_id.as_str())
            .copied()
            .cloned()
            .unwrap_or(schema::Span {
                file_path: relation.source_file_path.clone(),
                uri: relation.source_uri.clone(),
                start_line: 0,
                start_character: 0,
                end_line: 0,
                end_character: None,
            });
        let target_span = schema::Span {
            file_path: relation.target_file_path.clone(),
            uri: relation.target_uri.clone(),
            start_line: relation.target_start_line,
            start_character: 0,
            end_line: relation.target_end_line,
            end_character: None,
        };
        let target_symbol_id = format!(
            "target:{workspace_id}:{file}:{start}:{end}",
            file = relation.target_file_path,
            start = relation.target_start_line,
            end = relation.target_end_line
        );

        if relation.relation_kind == "references" {
            call_edges.push(schema::CallEdge {
                id: format!(
                    "call_edge:{workspace_id}:{source}:{target}",
                    source = relation.source_symbol_id,
                    target = target_symbol_id
                ),
                workspace_id: workspace_id.to_string(),
                source_symbol_id: relation.source_symbol_id.clone(),
                target_symbol_id: target_symbol_id.clone(),
                source_span: source_span.clone(),
                target_span: target_span.clone(),
            });
        } else {
            type_edges.push(schema::TypeEdge {
                id: format!(
                    "type_edge:{workspace_id}:{kind}:{source}:{target}",
                    kind = relation.relation_kind,
                    source = relation.source_symbol_id,
                    target = target_symbol_id
                ),
                workspace_id: workspace_id.to_string(),
                source_symbol_id: relation.source_symbol_id.clone(),
                target_symbol_id: target_symbol_id.clone(),
                relation_kind: relation.relation_kind.clone(),
                source_span: source_span.clone(),
                target_span: target_span.clone(),
            });
        }
    }

    (call_edges, type_edges)
}

fn build_diagnostic_docs_from_lsp(
    workspace_id: &str,
    file_rel: &str,
    file_uri: &str,
    diagnostics: &[Value],
) -> Vec<schema::DiagnosticDoc> {
    diagnostics
        .iter()
        .filter_map(|diagnostic| {
            let message = diagnostic
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .to_string();
            if message.is_empty() {
                return None;
            }

            let range = diagnostic.get("range")?;
            let start_line = range
                .get("start")
                .and_then(|v| v.get("line"))
                .and_then(Value::as_u64)
                .map(|v| v as usize + 1)
                .unwrap_or(1);
            let start_character = range
                .get("start")
                .and_then(|v| v.get("character"))
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .unwrap_or(0);
            let end_line = range
                .get("end")
                .and_then(|v| v.get("line"))
                .and_then(Value::as_u64)
                .map(|v| v as usize + 1)
                .unwrap_or(start_line);
            let end_character = range
                .get("end")
                .and_then(|v| v.get("character"))
                .and_then(Value::as_u64)
                .map(|v| v as usize);

            let severity = diagnostic
                .get("severity")
                .and_then(Value::as_u64)
                .map(map_lsp_diagnostic_severity)
                .unwrap_or("unknown")
                .to_string();
            let code = lsp_diagnostic_code_to_string(diagnostic.get("code"));

            Some(schema::DiagnosticDoc {
                id: format!(
                    "diagnostic:{workspace_id}:{file_rel}:{start_line}:{start_character}:{hash}",
                    hash = simple_hash(&format!(
                        "{severity}:{code}:{message}",
                        code = code.as_deref().unwrap_or_default()
                    ))
                ),
                workspace_id: workspace_id.to_string(),
                file_path: file_rel.to_string(),
                uri: file_uri.to_string(),
                severity,
                code,
                message,
                span: schema::Span {
                    file_path: file_rel.to_string(),
                    uri: file_uri.to_string(),
                    start_line,
                    start_character,
                    end_line: end_line.max(start_line),
                    end_character,
                },
            })
        })
        .collect::<Vec<_>>()
}

fn map_lsp_diagnostic_severity(severity: u64) -> &'static str {
    match severity {
        1 => "error",
        2 => "warning",
        3 => "information",
        4 => "hint",
        _ => "unknown",
    }
}

fn lsp_diagnostic_code_to_string(code: Option<&Value>) -> Option<String> {
    let code = code?;
    if let Some(value) = code.as_str() {
        return Some(value.to_string());
    }
    code.as_i64().map(|value| value.to_string())
}

fn build_crate_metadata_docs(
    workspace_id: &str,
    metadata: &WorkspaceMetadata,
) -> Vec<CrateMetadataDoc> {
    metadata
        .crates
        .iter()
        .map(|c| CrateMetadataDoc {
            id: format!("metadata:{workspace_id}:{}", c.crate_root),
            workspace_id: workspace_id.to_string(),
            crate_name: c.crate_name.clone(),
            edition: c.edition.clone(),
            crate_root: c.crate_root.clone(),
            manifest_path: c.manifest_path.clone(),
            features: c.features.clone(),
            optional_dependencies: c.optional_dependencies.clone(),
        })
        .collect()
}

async fn refresh_workspace_metadata_static(app: &App) -> Result<()> {
    let cfg = app.config.read().await.clone();
    let metadata = load_workspace_metadata(&cfg.workspace_root)?;
    let docs = build_crate_metadata_docs(&cfg.workspace_id, &metadata);
    let new_ids = docs.iter().map(|d| d.id.clone()).collect::<HashSet<_>>();

    let previous_ids = app.state.read().await.indexed_metadata_ids.clone();
    let stale_ids = previous_ids
        .difference(&new_ids)
        .cloned()
        .collect::<Vec<_>>();
    if !stale_ids.is_empty() {
        app.services
            .read()
            .await
            .metadata_store
            .delete_documents_by_ids(&stale_ids)
            .await
            .context("qdrant metadata stale delete failed")?;
    }

    if !docs.is_empty() {
        let rows = docs
            .iter()
            .map(|doc| NamedVectorDocument {
                id: doc.id.clone(),
                raw: doc.clone(),
                vectors: debug_oversize_named_vectors(&doc.id, crate_metadata_named_vectors(doc)),
            })
            .collect::<Vec<_>>();
        app.services
            .read()
            .await
            .metadata_store
            .insert_documents_with_named_vectors(rows)
            .await
            .context("qdrant metadata upsert failed")?;
    }

    let mut by_crate = HashMap::new();
    for doc in docs {
        by_crate.insert(doc.crate_root.clone(), doc);
    }

    let mut state = app.state.write().await;
    state.workspace_metadata = Some(metadata);
    state.metadata_docs_by_crate = by_crate;
    state.indexed_metadata_ids = new_ids;
    Ok(())
}

async fn refresh_workspace_graph(app: &App) -> Result<()> {
    let cfg = app.config.read().await.clone();
    let by_crate = app.state.read().await.metadata_docs_by_crate.clone();
    let mut crate_graph_docs = Vec::<schema::CrateGraphDoc>::new();
    {
        let mut ra = app.ra_manager.lock().await;
        for crate_doc in by_crate.values() {
            match ra
                .extract_crate_graph_doc(
                    &cfg.workspace_root,
                    &cfg.workspace_id,
                    &crate_doc.crate_name,
                    &crate_doc.crate_root,
                )
                .await
            {
                Ok(Some(doc)) => crate_graph_docs.push(doc),
                Ok(None) => {}
                Err(err) => return Err(err),
            }
        }
    }
    let new_graph_ids = crate_graph_docs
        .iter()
        .map(|d| d.id.clone())
        .collect::<HashSet<_>>();
    let previous_graph_ids = app.state.read().await.indexed_crate_graph_ids.clone();
    let stale_graph_ids = previous_graph_ids
        .difference(&new_graph_ids)
        .cloned()
        .collect::<Vec<_>>();
    if !stale_graph_ids.is_empty() {
        app.services
            .read()
            .await
            .crate_graph_store
            .delete_documents_by_ids(&stale_graph_ids)
            .await
            .context("qdrant crate-graph stale delete failed")?;
    }
    if !crate_graph_docs.is_empty() {
        let rows = crate_graph_docs
            .iter()
            .map(|doc| NamedVectorDocument {
                id: doc.id.clone(),
                raw: doc.clone(),
                vectors: debug_oversize_named_vectors(&doc.id, crate_graph_named_vectors(doc)),
            })
            .collect::<Vec<_>>();
        app.services
            .read()
            .await
            .crate_graph_store
            .insert_documents_with_named_vectors(rows)
            .await
            .context("qdrant crate-graph upsert failed")?;
    }
    let mut crate_graph_by_crate = HashMap::new();
    for doc in crate_graph_docs {
        crate_graph_by_crate.insert(doc.crate_root.clone(), doc);
    }

    let mut state = app.state.write().await;
    state.crate_graph_by_crate = crate_graph_by_crate;
    state.indexed_crate_graph_ids = new_graph_ids.clone();
    state.extraction_metrics.crate_graph_success_total += new_graph_ids.len() as u64;
    if !new_graph_ids.is_empty() {
        state.extraction_metrics.crate_graph_nonempty_total += 1;
    }
    Ok(())
}

async fn refresh_workspace_graph_with_retry(app: &App) -> Result<()> {
    let requires_crate_graph_warmup = !app.state.read().await.is_ra_warm_crate_graph;
    if requires_crate_graph_warmup {
        {
            let mut state = app.state.write().await;
            state.is_ra_warm_crate_graph = true;
            state.workspace_refresh_retry_count = 0;
            state.extraction_metrics.workspace_refresh_requeued_total += 1;
            state.queue_depth = state.queue_depth.saturating_add(1);
        }
        app.dirty_tx
            .send(DirtyEvent::RefreshWorkspace)
            .await
            .map_err(|_| anyhow::anyhow!("index queue unavailable during crate-graph warmup"))?;
        return Ok(());
    }

    match refresh_workspace_graph(app).await {
        Ok(()) => {
            let mut state = app.state.write().await;
            state.is_ra_warm_crate_graph = true;
            state.workspace_refresh_retry_count = 0;
            Ok(())
        }
        Err(err) if is_method_unsupported_error(&err) => {
            let mut state = app.state.write().await;
            state.is_ra_warm_crate_graph = true;
            state.workspace_refresh_retry_count = 0;
            state.extraction_metrics.crate_graph_unsupported_total += 1;
            Ok(())
        }
        Err(err) if is_transient_lsp_error(&err) => {
            let mut should_requeue = false;
            {
                let mut state = app.state.write().await;
                state.is_ra_warm_crate_graph = true;
                if state.workspace_refresh_retry_count < WORKSPACE_REFRESH_MAX_RETRIES {
                    state.workspace_refresh_retry_count += 1;
                    state.extraction_metrics.workspace_refresh_requeued_total += 1;
                    state
                        .extraction_metrics
                        .crate_graph_transient_requeued_total += 1;
                    state.queue_depth = state.queue_depth.saturating_add(1);
                    should_requeue = true;
                } else {
                    state.workspace_refresh_retry_count = 0;
                    state.extraction_metrics.crate_graph_failed_total += 1;
                }
            }
            if should_requeue {
                app.dirty_tx
                    .send(DirtyEvent::RefreshWorkspace)
                    .await
                    .map_err(|_| {
                        anyhow::anyhow!("index queue unavailable during workspace transient retry")
                    })?;
                return Ok(());
            }
            Err(err)
        }
        Err(err) => {
            let mut state = app.state.write().await;
            state.workspace_refresh_retry_count = 0;
            state.extraction_metrics.crate_graph_failed_total += 1;
            Err(err)
        }
    }
}

fn load_workspace_metadata(root: &Path) -> Result<WorkspaceMetadata> {
    let mut crates = Vec::<CrateMetadata>::new();
    for entry in WalkDir::new(root).into_iter().filter_entry(|e| {
        let name = e.file_name().to_string_lossy();
        name != ".git" && name != "target"
    }) {
        let entry = entry?;
        if !entry.file_type().is_file() || entry.file_name() != "Cargo.toml" {
            continue;
        }

        let manifest_path = entry.path().to_path_buf();
        let content = std::fs::read_to_string(&manifest_path)
            .with_context(|| format!("failed reading {}", manifest_path.display()))?;
        let value: toml::Value = content
            .parse::<toml::Value>()
            .with_context(|| format!("invalid Cargo.toml: {}", manifest_path.display()))?;
        let Some(pkg) = value.get("package").and_then(|v| v.as_table()) else {
            continue;
        };
        let Some(name) = pkg.get("name").and_then(|v| v.as_str()) else {
            continue;
        };
        let edition = pkg
            .get("edition")
            .and_then(|v| v.as_str())
            .unwrap_or("2021")
            .to_string();
        let crate_root_rel = manifest_path
            .parent()
            .unwrap_or(root)
            .strip_prefix(root)
            .unwrap_or_else(|_| Path::new("."));
        let crate_root = normalize_crate_root(crate_root_rel);
        let manifest_rel = manifest_path
            .strip_prefix(root)
            .unwrap_or(manifest_path.as_path())
            .to_string_lossy()
            .to_string();

        let features = value
            .get("features")
            .and_then(|v| v.as_table())
            .map(|t| t.keys().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        let optional_dependencies = collect_optional_dependencies(&value);

        crates.push(CrateMetadata {
            crate_name: name.to_string(),
            edition,
            crate_root,
            manifest_path: manifest_rel,
            features,
            optional_dependencies,
        });
    }

    crates.sort_by(|a, b| a.crate_root.cmp(&b.crate_root));
    Ok(WorkspaceMetadata {
        workspace_root: root.to_string_lossy().to_string(),
        crates,
    })
}

fn collect_optional_dependencies(value: &toml::Value) -> Vec<String> {
    let mut out = Vec::<String>::new();
    for key in ["dependencies", "dev-dependencies", "build-dependencies"] {
        let Some(table) = value.get(key).and_then(|v| v.as_table()) else {
            continue;
        };
        for (dep, cfg) in table {
            if cfg
                .as_table()
                .and_then(|t| t.get("optional"))
                .and_then(|v| v.as_bool())
                == Some(true)
            {
                out.push(dep.clone());
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

fn normalize_crate_root(path: &Path) -> String {
    if path.as_os_str().is_empty() || path == Path::new(".") {
        ".".to_string()
    } else {
        path.to_string_lossy().to_string()
    }
}

async fn extract_symbol_relations_with_lsp(
    session: &mut RaLspSession,
    workspace_root: &Path,
    workspace_id: &str,
    file_path_rel: &str,
    file_uri: &str,
    content: &str,
    docs: &[schema::SymbolDoc],
    bulk_mode: bool,
    metrics: &mut FileExtractionMetrics,
) -> Result<Vec<schema::GraphEdgeDoc>> {
    let source_lines = content.lines().collect::<Vec<_>>();
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut cache = HashMap::<PathBuf, Vec<String>>::new();

    for doc in docs {
        if !should_extract_relations_for_symbol(doc, bulk_mode) {
            continue;
        }
        session
            .wait_for_work_done_settle(LSP_WORK_DONE_SETTLE_TIMEOUT)
            .await?;
        let position = json!({
            "line": doc.start_line.saturating_sub(1),
            "character": doc.start_character
        });
        let _ = session
            .request_if_supported(
                "textDocument/definition",
                json!({
                    "textDocument": { "uri": file_uri },
                    "position": position
                }),
                LSP_HEAVY_REQUEST_TIMEOUT,
                LSP_CONTENT_MODIFIED_RETRIES,
            )
            .await;

        let references = session
            .request_if_supported(
                "textDocument/references",
                json!({
                    "textDocument": { "uri": file_uri },
                    "position": position,
                    "context": { "includeDeclaration": false }
                }),
                LSP_HEAVY_REQUEST_TIMEOUT,
                LSP_CONTENT_MODIFIED_RETRIES,
            )
            .await;
        let mut references_payload: Option<Value> = None;
        if let Ok(Some(references)) = references {
            references_payload = Some(references.clone());
            metrics.references_success += 1;
            let reference_locations = lsp_locations_from_value(&references).len();
            let emitted = append_relation_targets(
                &mut out,
                &mut seen,
                &mut cache,
                workspace_root,
                workspace_id,
                file_path_rel,
                doc,
                "references",
                references,
                &source_lines,
            )
            .await?;
            if emitted > 0 {
                metrics.references_nonempty += 1;
            }
            let _ = reference_locations;
            metrics.relations_references_emitted += emitted as u64;
        } else if references.is_err() {
            metrics.references_failed += 1;
        }
        if should_request_implementations_for_symbol(doc) {
            let implementations = session
                .request_if_supported(
                    "textDocument/implementation",
                    json!({
                        "textDocument": { "uri": file_uri },
                        "position": position
                    }),
                    LSP_HEAVY_REQUEST_TIMEOUT,
                    LSP_CONTENT_MODIFIED_RETRIES,
                )
                .await;
            if let Ok(Some(implementations)) = implementations {
                metrics.implementations_success += 1;
                let mut emitted = append_relation_targets(
                    &mut out,
                    &mut seen,
                    &mut cache,
                    workspace_root,
                    workspace_id,
                    file_path_rel,
                    doc,
                    "implementations",
                    implementations,
                    &source_lines,
                )
                .await?;
                if emitted == 0
                    && should_fallback_implementations_from_references(doc)
                    && let Some(payload) = references_payload.clone()
                {
                    emitted += append_relation_targets_with_filter(
                        &mut out,
                        &mut seen,
                        &mut cache,
                        workspace_root,
                        workspace_id,
                        file_path_rel,
                        doc,
                        "implementations",
                        payload,
                        &source_lines,
                        |excerpt| excerpt_is_impl_for_symbol(excerpt, &doc.symbol),
                    )
                    .await?;
                }
                if emitted > 0 {
                    metrics.implementations_nonempty += 1;
                }
                metrics.relations_implementations_emitted += emitted as u64;
            } else if let Err(err) = implementations {
                metrics.implementations_failed += 1;
                let _ = err;
            }
        }
        let definitions = session
            .request_if_supported(
                "textDocument/definition",
                json!({
                    "textDocument": { "uri": file_uri },
                    "position": position
                }),
                LSP_HEAVY_REQUEST_TIMEOUT,
                LSP_CONTENT_MODIFIED_RETRIES,
            )
            .await;
        if let Ok(Some(definitions)) = definitions {
            metrics.definitions_success += 1;
            let emitted = append_relation_targets(
                &mut out,
                &mut seen,
                &mut cache,
                workspace_root,
                workspace_id,
                file_path_rel,
                doc,
                "definitions",
                definitions,
                &source_lines,
            )
            .await?;
            if emitted > 0 {
                metrics.definitions_nonempty += 1;
            }
            metrics.relations_definitions_emitted += emitted as u64;
        } else if definitions.is_err() {
            metrics.definitions_failed += 1;
        }
        if should_request_type_definitions_for_symbol(doc) {
            let type_definitions = session
                .request_if_supported(
                    "textDocument/typeDefinition",
                    json!({
                        "textDocument": { "uri": file_uri },
                        "position": position
                    }),
                    LSP_HEAVY_REQUEST_TIMEOUT,
                    LSP_CONTENT_MODIFIED_RETRIES,
                )
                .await;
            if let Ok(Some(type_definitions)) = type_definitions {
                metrics.type_definitions_success += 1;
                let mut emitted = append_relation_targets(
                    &mut out,
                    &mut seen,
                    &mut cache,
                    workspace_root,
                    workspace_id,
                    file_path_rel,
                    doc,
                    "type_definitions",
                    type_definitions,
                    &source_lines,
                )
                .await?;
                if emitted == 0
                    && let Some((usage_line, usage_character)) = symbol_usage_position_in_file(
                        &source_lines,
                        &doc.symbol,
                        doc.start_line,
                        doc.end_line,
                    )
                {
                    let usage_type_definitions = session
                        .request_if_supported(
                            "textDocument/typeDefinition",
                            json!({
                                "textDocument": { "uri": file_uri },
                                "position": {
                                    "line": usage_line.saturating_sub(1),
                                    "character": usage_character
                                }
                            }),
                            LSP_HEAVY_REQUEST_TIMEOUT,
                            LSP_CONTENT_MODIFIED_RETRIES,
                        )
                        .await;
                    if let Ok(Some(payload)) = usage_type_definitions {
                        emitted += append_relation_targets(
                            &mut out,
                            &mut seen,
                            &mut cache,
                            workspace_root,
                            workspace_id,
                            file_path_rel,
                            doc,
                            "type_definitions",
                            payload,
                            &source_lines,
                        )
                        .await?;
                    }
                }
                if emitted > 0 {
                    metrics.type_definitions_nonempty += 1;
                }
                metrics.relations_type_definitions_emitted += emitted as u64;
            } else if type_definitions.is_err() {
                metrics.type_definitions_failed += 1;
            }
        }
    }

    Ok(out)
}

async fn append_relation_targets(
    out: &mut Vec<schema::GraphEdgeDoc>,
    seen: &mut HashSet<String>,
    cache: &mut HashMap<PathBuf, Vec<String>>,
    workspace_root: &Path,
    workspace_id: &str,
    source_file_path: &str,
    source_doc: &schema::SymbolDoc,
    relation_kind: &str,
    payload: Value,
    source_lines: &[&str],
) -> Result<usize> {
    append_relation_targets_with_filter(
        out,
        seen,
        cache,
        workspace_root,
        workspace_id,
        source_file_path,
        source_doc,
        relation_kind,
        payload,
        source_lines,
        |_| true,
    )
    .await
}

async fn append_relation_targets_with_filter(
    out: &mut Vec<schema::GraphEdgeDoc>,
    seen: &mut HashSet<String>,
    cache: &mut HashMap<PathBuf, Vec<String>>,
    workspace_root: &Path,
    workspace_id: &str,
    source_file_path: &str,
    source_doc: &schema::SymbolDoc,
    relation_kind: &str,
    payload: Value,
    source_lines: &[&str],
    keep_target: impl Fn(&str) -> bool,
) -> Result<usize> {
    let locations = lsp_locations_from_value(&payload);
    let mut emitted = 0usize;
    for loc in locations {
        let Some(target_path) = uri_to_workspace_relative_path(workspace_root, &loc.uri) else {
            continue;
        };
        let key = format!(
            "{kind}:{source}:{target}:{start}:{end}",
            kind = relation_kind,
            source = source_doc.id,
            target = target_path,
            start = loc.start_line,
            end = loc.end_line
        );
        if !seen.insert(key) {
            continue;
        }

        let excerpt = if target_path == source_file_path {
            excerpt_from_range(
                source_lines,
                loc.start_line.saturating_sub(1),
                loc.end_line.saturating_sub(1),
                20,
            )
        } else {
            excerpt_for_target(
                cache,
                workspace_root,
                &target_path,
                loc.start_line,
                loc.end_line,
            )
            .await
        };
        if relation_target_is_low_signal(&excerpt) {
            continue;
        }
        if !keep_target(&excerpt) {
            continue;
        }

        let id = format!(
            "relation:{workspace_id}:{kind}:{source}:{target}:{start}:{end}",
            workspace_id = workspace_id,
            kind = relation_kind,
            source = source_doc.id,
            target = target_path,
            start = loc.start_line,
            end = loc.end_line
        );
        out.push(schema::GraphEdgeDoc {
            id,
            workspace_id: workspace_id.to_string(),
            relation_kind: relation_kind.to_string(),
            edge_type: schema::GraphEdgeDoc::edge_type_for_relation_kind(relation_kind).to_string(),
            source_symbol_id: source_doc.id.clone(),
            source_symbol: source_doc.symbol.clone(),
            source_file_path: source_file_path.to_string(),
            source_crate_name: source_doc.crate_name.clone(),
            source_uri: source_doc.uri.clone(),
            target_file_path: target_path,
            target_uri: loc.uri,
            target_start_line: loc.start_line,
            target_end_line: loc.end_line,
            target_excerpt: excerpt,
        });
        emitted += 1;
    }

    Ok(emitted)
}

fn excerpt_is_impl_for_symbol(excerpt: &str, symbol: &str) -> bool {
    let Some(first) = excerpt.lines().find_map(|line| {
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    }) else {
        return false;
    };

    first.starts_with("impl ")
        && first.contains(symbol)
        && (first.contains(" for ") || first.contains('{'))
}

fn relation_target_is_low_signal(excerpt: &str) -> bool {
    let Some(first) = excerpt.lines().find_map(|line| {
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    }) else {
        return false;
    };

    first.starts_with("use ")
        || first.starts_with("pub use ")
        || first.starts_with("mod ")
        || first.starts_with("pub mod ")
        || first.starts_with("extern crate ")
}

#[derive(Debug)]
struct LspLocation {
    uri: String,
    start_line: usize,
    end_line: usize,
}

fn lsp_locations_from_value(payload: &Value) -> Vec<LspLocation> {
    if let Some(items) = payload.as_array() {
        return items
            .iter()
            .filter_map(lsp_location_from_item)
            .collect::<Vec<_>>();
    }

    payload
        .get("uri")
        .and_then(|_| lsp_location_from_item(payload))
        .into_iter()
        .collect::<Vec<_>>()
}

fn lsp_location_from_item(item: &Value) -> Option<LspLocation> {
    let uri = item.get("uri")?.as_str()?.to_string();
    let range = item.get("range")?.as_object()?;
    let start = range.get("start")?.get("line")?.as_u64()? as usize + 1;
    let end = range.get("end")?.get("line")?.as_u64()? as usize + 1;
    Some(LspLocation {
        uri,
        start_line: start,
        end_line: end.max(start),
    })
}

fn uri_to_workspace_relative_path(workspace_root: &Path, uri: &str) -> Option<String> {
    let path = file_uri_to_path(uri)?;
    let rel = path.strip_prefix(workspace_root).ok()?;
    Some(rel.to_string_lossy().to_string())
}

fn file_uri_to_path(uri: &str) -> Option<PathBuf> {
    let raw = uri.strip_prefix("file://")?;
    Some(PathBuf::from(percent_decode(raw)))
}

fn percent_decode(input: &str) -> String {
    input
        .replace("%20", " ")
        .replace("%23", "#")
        .replace("%3F", "?")
        .replace("%25", "%")
}

async fn excerpt_for_target(
    cache: &mut HashMap<PathBuf, Vec<String>>,
    workspace_root: &Path,
    target_file_path: &str,
    start_line: usize,
    end_line: usize,
) -> String {
    let path = workspace_root.join(target_file_path);
    if !cache.contains_key(&path) {
        let lines = match tokio::fs::read_to_string(&path).await {
            Ok(content) => content
                .lines()
                .map(std::string::ToString::to_string)
                .collect(),
            Err(_) => Vec::new(),
        };
        cache.insert(path.clone(), lines);
    }

    let Some(lines) = cache.get(&path) else {
        return String::new();
    };
    if lines.is_empty() {
        return String::new();
    }
    let refs = lines.iter().map(String::as_str).collect::<Vec<_>>();
    excerpt_from_range(
        &refs,
        start_line.saturating_sub(1),
        end_line.saturating_sub(1),
        20,
    )
}

async fn enrich_docs_with_lsp_metadata(
    session: &mut RaLspSession,
    file_uri: &str,
    content: &str,
    docs: &mut [schema::SymbolDoc],
    bulk_mode: bool,
    metrics: &mut FileExtractionMetrics,
) -> Result<()> {
    session
        .wait_for_work_done_settle(LSP_WORK_DONE_SETTLE_TIMEOUT)
        .await?;
    let lines = content.lines().collect::<Vec<_>>();

    for doc in docs {
        let primary_line = doc.start_line.saturating_sub(1);
        let primary_character = doc.start_character;
        let position = json!({
            "line": primary_line,
            "character": primary_character
        });

        let hover = session
            .request_if_supported(
                "textDocument/hover",
                json!({
                    "textDocument": { "uri": file_uri },
                    "position": position
                }),
                LSP_REQUEST_TIMEOUT,
                LSP_CONTENT_MODIFIED_RETRIES,
            )
            .await;

        if let Ok(Some(hover)) = hover {
            metrics.hover_success += 1;
            doc.hover_summary = lsp_hover_to_text(&hover).unwrap_or_default();
            if doc.hover_summary.trim().is_empty()
                && let Some((retry_line, retry_character)) =
                    symbol_position_in_lsp_range(&lines, &doc.symbol, doc.start_line, doc.end_line)
                && (retry_line.saturating_sub(1) != primary_line
                    || retry_character != primary_character)
            {
                let retry_hover = session
                    .request_if_supported(
                        "textDocument/hover",
                        json!({
                            "textDocument": { "uri": file_uri },
                            "position": {
                                "line": retry_line.saturating_sub(1),
                                "character": retry_character
                            }
                        }),
                        LSP_REQUEST_TIMEOUT,
                        LSP_CONTENT_MODIFIED_RETRIES,
                    )
                    .await;
                if let Ok(Some(hover)) = retry_hover {
                    let retry_summary = lsp_hover_to_text(&hover).unwrap_or_default();
                    if !retry_summary.trim().is_empty() {
                        doc.hover_summary = retry_summary;
                    }
                }
            }
        } else if hover.is_err() {
            metrics.hover_failed += 1;
        }
        if !doc.hover_summary.trim().is_empty() {
            metrics.hover_nonempty += 1;
        }
        // signature_help is more often than not null, mainly usefull for IDE scenarios
        /*
        if doc.kind == "fn" {
            let signature_help = session
                .request_with_retry(
                    "textDocument/signatureHelp",
                    json!({
                        "textDocument": { "uri": file_uri },
                        "position": position
                    }),
                    LSP_REQUEST_TIMEOUT,
                    LSP_CONTENT_MODIFIED_RETRIES,
                )
                .await;
            if let Ok(signature_help) = signature_help {
                metrics.signature_help_success += 1;
                doc.signature_help = lsp_signature_help_to_text(&signature_help).unwrap_or_default();
            } else {
                metrics.signature_help_failed += 1;
            }
        } */

        if !bulk_mode {
            let ws_symbol = session
                .request_if_supported(
                    "workspace/symbol",
                    json!({
                        "query": doc.symbol
                    }),
                    LSP_HEAVY_REQUEST_TIMEOUT,
                    LSP_CONTENT_MODIFIED_RETRIES,
                )
                .await;
            if let Ok(Some(result)) = ws_symbol {
                metrics.workspace_symbol_success += 1;
                if let Some(path) = lsp_workspace_symbol_path_for_item(
                    &result,
                    file_uri,
                    &doc.symbol,
                    doc.start_line,
                ) {
                    maybe_upgrade_symbol_path(doc, &path);
                    metrics.workspace_symbol_nonempty += 1;
                }
            } else if ws_symbol.is_err() {
                metrics.workspace_symbol_failed += 1;
            }
        }
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct DecodedSemanticToken {
    line: usize,
    start_character: usize,
    length: usize,
    token_type_idx: usize,
}

#[derive(Debug, Clone)]
struct DecodedInlayHint {
    line: usize,
    character: usize,
    label: String,
}

async fn extract_lsp_artifact_docs(
    session: &mut RaLspSession,
    workspace_id: &str,
    file_path_rel: &str,
    file_uri: &str,
    content: &str,
    docs: &[schema::SymbolDoc],
    enable_syntax_artifacts: bool,
    metrics: &mut FileExtractionMetrics,
) -> Result<(
    Vec<schema::SemanticTokenDoc>,
    Vec<schema::SyntaxTreeDoc>,
    Vec<schema::InlayHintDoc>,
)> {
    let mut semantic_docs = Vec::<schema::SemanticTokenDoc>::new();
    let mut syntax_docs = Vec::<schema::SyntaxTreeDoc>::new();
    let mut inlay_docs = Vec::<schema::InlayHintDoc>::new();
    let line_count = content.lines().count().max(1);

    let semantic_payload = session
        .request_if_supported(
            "textDocument/semanticTokens/full",
            json!({ "textDocument": { "uri": file_uri } }),
            LSP_HEAVY_REQUEST_TIMEOUT,
            LSP_CONTENT_MODIFIED_RETRIES,
        )
        .await;
    match semantic_payload {
        Ok(Some(payload)) => {
            metrics.semantic_tokens_success += 1;
            let tokens = decode_semantic_tokens(&payload);
            if !tokens.is_empty() {
                metrics.semantic_tokens_nonempty += 1;
            }
            semantic_docs.push(build_file_semantic_doc(
                workspace_id,
                file_path_rel,
                file_uri,
                &tokens,
                line_count,
            ));
            for doc in docs {
                semantic_docs.push(build_symbol_semantic_doc(
                    workspace_id,
                    file_path_rel,
                    file_uri,
                    doc,
                    &tokens,
                ));
            }
        }
        Ok(None) => {}
        Err(_) => metrics.semantic_tokens_failed += 1,
    }

    if enable_syntax_artifacts {
        let syntax_payload = session
            .request_if_supported(
                "rust-analyzer/viewSyntaxTree",
                json!({ "textDocument": { "uri": file_uri } }),
                LSP_HEAVY_REQUEST_TIMEOUT,
                LSP_CONTENT_MODIFIED_RETRIES,
            )
            .await;
        match syntax_payload {
            Ok(Some(payload)) => {
                metrics.syntax_tree_success += 1;
                let normalized = payload
                    .as_str()
                    .map(normalize_syntax_tree_text)
                    .unwrap_or_else(|| normalize_syntax_tree_text(&payload.to_string()));
                if !normalized.trim().is_empty() {
                    metrics.syntax_tree_nonempty += 1;
                }
                let node_kind_histogram = syntax_histogram(&normalized);
                syntax_docs.push(schema::SyntaxTreeDoc {
                    id: format!("syntax:{workspace_id}:{file_path_rel}:file"),
                    workspace_id: workspace_id.to_string(),
                    file_path: file_path_rel.to_string(),
                    uri: file_uri.to_string(),
                    symbol_id: None,
                    node_kind_histogram: node_kind_histogram.clone(),
                    summary: compact_syntax_summary(&node_kind_histogram, None),
                    span: schema::Span {
                        file_path: file_path_rel.to_string(),
                        uri: file_uri.to_string(),
                        start_line: 1,
                        start_character: 0,
                        end_line: line_count,
                        end_character: None,
                    },
                });
                for doc in docs {
                    syntax_docs.push(schema::SyntaxTreeDoc {
                        id: format!(
                            "syntax:{workspace_id}:{file_path_rel}:{}",
                            doc.symbol_id_stable
                        ),
                        workspace_id: workspace_id.to_string(),
                        file_path: file_path_rel.to_string(),
                        uri: file_uri.to_string(),
                        symbol_id: Some(doc.symbol_id_stable.clone()),
                        node_kind_histogram: node_kind_histogram.clone(),
                        summary: syntax_summary_for_symbol(&node_kind_histogram, doc),
                        span: doc.span.clone(),
                    });
                }
            }
            Ok(None) => {
                metrics.syntax_tree_unsupported += 1;
                eprintln!(
                    "[syntax-tree-debug] file={} result=unsupported_or_none",
                    file_path_rel
                );
            }
            Err(_) => metrics.syntax_tree_failed += 1,
        }
    }

    let inlay_payload = session
        .request_if_supported(
            "textDocument/inlayHint",
            json!({
                "textDocument": { "uri": file_uri },
                "range": {
                    "start": { "line": 0, "character": 0 },
                    "end": { "line": line_count, "character": 0 }
                }
            }),
            LSP_HEAVY_REQUEST_TIMEOUT,
            LSP_CONTENT_MODIFIED_RETRIES,
        )
        .await;
    match inlay_payload {
        Ok(Some(payload)) => {
            metrics.inlay_hints_success += 1;
            let hints = decode_inlay_hints(&payload);
            if !hints.is_empty() {
                metrics.inlay_hints_nonempty += 1;
            }
            inlay_docs.push(schema::InlayHintDoc {
                id: format!("inlay:{workspace_id}:{file_path_rel}:file"),
                workspace_id: workspace_id.to_string(),
                file_path: file_path_rel.to_string(),
                uri: file_uri.to_string(),
                symbol_id: None,
                hint_count: hints.len(),
                summary: summarize_inlay_hints(&hints),
                span: schema::Span {
                    file_path: file_path_rel.to_string(),
                    uri: file_uri.to_string(),
                    start_line: 1,
                    start_character: 0,
                    end_line: line_count,
                    end_character: None,
                },
            });
            for doc in docs {
                let symbol_hints = hints
                    .iter()
                    .filter(|h| h.line >= doc.start_line && h.line <= doc.end_line)
                    .cloned()
                    .collect::<Vec<_>>();
                inlay_docs.push(schema::InlayHintDoc {
                    id: format!(
                        "inlay:{workspace_id}:{file_path_rel}:{}",
                        doc.symbol_id_stable
                    ),
                    workspace_id: workspace_id.to_string(),
                    file_path: file_path_rel.to_string(),
                    uri: file_uri.to_string(),
                    symbol_id: Some(doc.symbol_id_stable.clone()),
                    hint_count: symbol_hints.len(),
                    summary: summarize_inlay_hints(&symbol_hints),
                    span: doc.span.clone(),
                });
            }
        }
        Ok(None) => {}
        Err(_) => metrics.inlay_hints_failed += 1,
    }

    Ok((semantic_docs, syntax_docs, inlay_docs))
}

fn decode_semantic_tokens(payload: &Value) -> Vec<DecodedSemanticToken> {
    let raw = payload
        .get("data")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut out = Vec::new();
    let mut line = 0usize;
    let mut start = 0usize;
    for chunk in raw.chunks_exact(5) {
        let delta_line = chunk[0].as_u64().unwrap_or(0) as usize;
        let delta_start = chunk[1].as_u64().unwrap_or(0) as usize;
        let length = chunk[2].as_u64().unwrap_or(0) as usize;
        let token_type_idx = chunk[3].as_u64().unwrap_or(0) as usize;
        if delta_line == 0 {
            start = start.saturating_add(delta_start);
        } else {
            line = line.saturating_add(delta_line);
            start = delta_start;
        }
        out.push(DecodedSemanticToken {
            line: line.saturating_add(1),
            start_character: start,
            length,
            token_type_idx,
        });
    }
    out
}

fn build_file_semantic_doc(
    workspace_id: &str,
    file_path_rel: &str,
    file_uri: &str,
    tokens: &[DecodedSemanticToken],
    line_count: usize,
) -> schema::SemanticTokenDoc {
    schema::SemanticTokenDoc {
        id: format!("semantic:{workspace_id}:{file_path_rel}:file"),
        workspace_id: workspace_id.to_string(),
        file_path: file_path_rel.to_string(),
        uri: file_uri.to_string(),
        symbol_id: None,
        token_count: tokens.len(),
        token_type_histogram: semantic_histogram(tokens),
        summary: summarize_semantic_tokens(tokens),
        span: schema::Span {
            file_path: file_path_rel.to_string(),
            uri: file_uri.to_string(),
            start_line: 1,
            start_character: 0,
            end_line: line_count,
            end_character: None,
        },
    }
}

fn build_symbol_semantic_doc(
    workspace_id: &str,
    file_path_rel: &str,
    file_uri: &str,
    symbol: &schema::SymbolDoc,
    tokens: &[DecodedSemanticToken],
) -> schema::SemanticTokenDoc {
    let symbol_tokens = tokens
        .iter()
        .filter(|t| t.line >= symbol.start_line && t.line <= symbol.end_line)
        .cloned()
        .collect::<Vec<_>>();
    schema::SemanticTokenDoc {
        id: format!(
            "semantic:{workspace_id}:{file_path_rel}:{}",
            symbol.symbol_id_stable
        ),
        workspace_id: workspace_id.to_string(),
        file_path: file_path_rel.to_string(),
        uri: file_uri.to_string(),
        symbol_id: Some(symbol.symbol_id_stable.clone()),
        token_count: symbol_tokens.len(),
        token_type_histogram: semantic_histogram(&symbol_tokens),
        summary: summarize_semantic_tokens(&symbol_tokens),
        span: symbol.span.clone(),
    }
}

fn semantic_histogram(tokens: &[DecodedSemanticToken]) -> Vec<String> {
    let mut by_type = HashMap::<usize, usize>::new();
    for token in tokens {
        *by_type.entry(token.token_type_idx).or_default() += 1;
    }
    let mut rows = by_type.into_iter().map(|(k, v)| (k, v)).collect::<Vec<_>>();
    rows.sort_by_key(|(k, _)| *k);
    rows.into_iter()
        .map(|(k, v)| format!("type_{k}:{v}"))
        .collect()
}

fn summarize_semantic_tokens(tokens: &[DecodedSemanticToken]) -> String {
    if tokens.is_empty() {
        return String::new();
    }
    let avg_len = tokens.iter().map(|t| t.length as u64).sum::<u64>() as f64 / tokens.len() as f64;
    let first_line = tokens.first().map(|t| t.line).unwrap_or(0);
    let last_line = tokens.last().map(|t| t.line).unwrap_or(first_line);
    let min_char = tokens.iter().map(|t| t.start_character).min().unwrap_or(0);
    format!(
        "semantic_tokens count={} avg_len={:.2} lines={}..{} min_char={}",
        tokens.len(),
        avg_len,
        first_line,
        last_line,
        min_char
    )
}

fn normalize_syntax_tree_text(raw: &str) -> String {
    raw.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(400)
        .collect::<Vec<_>>()
        .join("\n")
}

fn syntax_histogram(summary: &str) -> Vec<String> {
    let mut counts = HashMap::<String, usize>::new();
    for line in summary.lines() {
        let key = line
            .split('@')
            .next()
            .unwrap_or_default()
            .trim()
            .to_string();
        if !key.is_empty() {
            *counts.entry(key).or_default() += 1;
        }
    }
    let mut rows = counts.into_iter().collect::<Vec<_>>();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    rows.into_iter().map(|(k, v)| format!("{k}:{v}")).collect()
}

fn compact_syntax_summary(histogram: &[String], span: Option<(usize, usize)>) -> String {
    let top = histogram
        .iter()
        .take(24)
        .cloned()
        .collect::<Vec<_>>()
        .join(",");
    if let Some((start, end)) = span {
        return format!(
            "syntax_kinds_count: {}\nspan: {}..{}\ntop_kinds: {}",
            histogram.len(),
            start,
            end,
            top
        );
    }
    format!(
        "syntax_kinds_count: {}\ntop_kinds: {}",
        histogram.len(),
        top
    )
}

fn syntax_summary_for_symbol(histogram: &[String], symbol: &schema::SymbolDoc) -> String {
    format!(
        "symbol: {}\nkind: {}\n{}",
        symbol.symbol,
        symbol.kind,
        compact_syntax_summary(histogram, Some((symbol.start_line, symbol.end_line)))
    )
}

fn decode_inlay_hints(payload: &Value) -> Vec<DecodedInlayHint> {
    let mut out = Vec::new();
    let Some(items) = payload.as_array() else {
        return out;
    };
    for item in items {
        let line = item
            .get("position")
            .and_then(|v| v.get("line"))
            .and_then(Value::as_u64)
            .map(|v| v as usize + 1)
            .unwrap_or(0);
        let character = item
            .get("position")
            .and_then(|v| v.get("character"))
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(0);
        let label = match item.get("label") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(parts)) => parts
                .iter()
                .filter_map(|p| p.get("value").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        };
        out.push(DecodedInlayHint {
            line,
            character,
            label,
        });
    }
    out
}

fn summarize_inlay_hints(hints: &[DecodedInlayHint]) -> String {
    if hints.is_empty() {
        return String::new();
    }
    hints
        .iter()
        .take(80)
        .map(|h| format!("{}:{} {}", h.line, h.character, h.label))
        .collect::<Vec<_>>()
        .join("\n")
}

fn apply_artifact_summaries_to_symbols(
    symbols: &mut [schema::SymbolDoc],
    semantic_docs: &[schema::SemanticTokenDoc],
    syntax_docs: &[schema::SyntaxTreeDoc],
    inlay_docs: &[schema::InlayHintDoc],
) {
    let semantic_by_symbol = semantic_docs
        .iter()
        .filter_map(|doc| {
            doc.symbol_id
                .as_ref()
                .map(|id| (id.as_str(), doc.summary.as_str()))
        })
        .collect::<HashMap<_, _>>();
    let syntax_by_symbol = syntax_docs
        .iter()
        .filter_map(|doc| {
            doc.symbol_id
                .as_ref()
                .map(|id| (id.as_str(), doc.summary.as_str()))
        })
        .collect::<HashMap<_, _>>();
    let inlay_by_symbol = inlay_docs
        .iter()
        .filter_map(|doc| {
            doc.symbol_id
                .as_ref()
                .map(|id| (id.as_str(), doc.summary.as_str()))
        })
        .collect::<HashMap<_, _>>();

    for symbol in symbols {
        if let Some(summary) = semantic_by_symbol.get(symbol.symbol_id_stable.as_str()) {
            symbol.semantic_tokens_summary = (*summary).to_string();
        }
        if let Some(summary) = syntax_by_symbol.get(symbol.symbol_id_stable.as_str()) {
            symbol.syntax_tree_summary = (*summary).to_string();
        }
        if let Some(summary) = inlay_by_symbol.get(symbol.symbol_id_stable.as_str()) {
            symbol.inlay_hints_summary = (*summary).to_string();
        }
    }
}

fn should_extract_relations_for_symbol(doc: &schema::SymbolDoc, bulk_mode: bool) -> bool {
    match doc.kind.as_str() {
        // Skip noisy/low-value leaf symbols for relation extraction.
        "var" | "module" => false,
        // During full/bulk scans, prioritize core API/type graph symbols.
        "fn" | "struct" | "enum" | "trait" => true,
        _ => !bulk_mode,
    }
}

fn should_request_implementations_for_symbol(doc: &schema::SymbolDoc) -> bool {
    matches!(doc.kind.as_str(), "trait" | "struct" | "method")
}

fn should_fallback_implementations_from_references(doc: &schema::SymbolDoc) -> bool {
    matches!(doc.kind.as_str(), "struct" | "enum")
}

fn should_request_type_definitions_for_symbol(doc: &schema::SymbolDoc) -> bool {
    matches!(doc.kind.as_str(), "trait" | "struct" | "enum")
}

fn lsp_hover_to_text(result: &Value) -> Option<String> {
    let contents = result.get("contents")?;
    lsp_marked_content_to_text(contents)
}

/*
fn lsp_signature_help_to_text(result: &Value) -> Option<String> {
    let signatures = result.get("signatures")?.as_array()?;
    let first = signatures.first()?;
    first
        .get("label")
        .and_then(Value::as_str)
        .map(|v| v.trim().to_string())
}*/

fn lsp_workspace_symbol_path_for_item(
    result: &Value,
    file_uri: &str,
    symbol: &str,
    start_line: usize,
) -> Option<String> {
    let items = result.as_array()?;
    let mut best: Option<(usize, String)> = None;
    for item in items {
        let name = item.get("name").and_then(Value::as_str)?;
        if name != symbol {
            continue;
        }
        let uri = item
            .get("location")
            .and_then(|v| v.get("uri"))
            .and_then(Value::as_str)
            .or_else(|| {
                item.get("location")
                    .and_then(|v| v.get("targetUri"))
                    .and_then(Value::as_str)
            })?;
        if uri != file_uri {
            continue;
        }
        let line = item
            .get("location")
            .and_then(|v| v.get("range"))
            .and_then(|v| v.get("start"))
            .and_then(|v| v.get("line"))
            .and_then(Value::as_u64)
            .map(|v| v as usize + 1)
            .unwrap_or(start_line);
        let distance = line.abs_diff(start_line);
        let candidate = item
            .get("containerName")
            .and_then(Value::as_str)
            .filter(|v| !v.is_empty())
            .map(|container| format!("{container}::{name}"))
            .unwrap_or_else(|| name.to_string());
        match &best {
            Some((d, _)) if *d <= distance => {}
            _ => best = Some((distance, candidate)),
        }
    }
    best.map(|(_, v)| v)
}

fn maybe_upgrade_symbol_path(doc: &mut schema::SymbolDoc, workspace_path: &str) {
    let candidate = normalize_symbol_path_candidate(&doc.module, &doc.symbol, workspace_path);
    if candidate.is_empty() {
        return;
    }
    if symbol_path_leaf(&candidate) != doc.symbol {
        return;
    }

    let current = doc.symbol_path.trim();
    let should_replace = current.is_empty()
        || symbol_path_leaf(current) != doc.symbol
        || symbol_path_quality(&candidate) > symbol_path_quality(current);

    if should_replace {
        doc.symbol_path = candidate;
    }
}

fn normalize_symbol_path_candidate(module: &str, symbol: &str, candidate: &str) -> String {
    let trimmed = candidate.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    if trimmed.contains("::") {
        return trimmed.to_string();
    }

    if !module.is_empty() && trimmed == symbol {
        return format!("{module}::{trimmed}");
    }

    trimmed.to_string()
}

fn symbol_path_quality(path: &str) -> (usize, usize) {
    let path = path.trim();
    if path.is_empty() {
        return (0, 0);
    }
    let segments = path.split("::").filter(|part| !part.is_empty()).count();
    (segments, path.len())
}

fn symbol_path_leaf(path: &str) -> &str {
    path.rsplit("::").next().unwrap_or(path)
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

async fn read_lsp_response_for_id<R>(
    reader: &mut BufReader<R>,
    expected_id: i64,
    work_done_in_flight: &mut usize,
    saw_work_done_progress: &mut bool,
    diagnostics_by_uri: &mut HashMap<String, Vec<Value>>,
) -> Result<Value>
where
    R: tokio::io::AsyncRead + Unpin,
{
    while let Some(raw) = read_lsp_message(reader).await? {
        let msg: Value = serde_json::from_str(&raw)?;
        update_lsp_side_channel_state(
            &msg,
            work_done_in_flight,
            saw_work_done_progress,
            diagnostics_by_uri,
        );
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

fn update_lsp_side_channel_state(
    msg: &Value,
    in_flight: &mut usize,
    saw_progress: &mut bool,
    diagnostics_by_uri: &mut HashMap<String, Vec<Value>>,
) {
    if msg.get("method").and_then(Value::as_str) == Some("textDocument/publishDiagnostics")
        && let Some(params) = msg.get("params")
        && let Some(uri) = params.get("uri").and_then(Value::as_str)
    {
        let diagnostics = params
            .get("diagnostics")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        diagnostics_by_uri.insert(uri.to_string(), diagnostics);
    }

    if msg.get("method").and_then(Value::as_str) != Some("$/progress") {
        return;
    }
    let kind = msg
        .get("params")
        .and_then(|v| v.get("value"))
        .and_then(|v| v.get("kind"))
        .and_then(Value::as_str);
    match kind {
        Some("begin") => {
            *saw_progress = true;
            *in_flight = in_flight.saturating_add(1);
        }
        Some("end") => {
            *saw_progress = true;
            *in_flight = in_flight.saturating_sub(1);
        }
        Some("report") => {
            *saw_progress = true;
        }
        _ => {}
    }
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
) -> Vec<schema::SymbolDoc> {
    let Some(items) = result.as_array() else {
        return Vec::new();
    };

    let lines: Vec<&str> = content.lines().collect::<Vec<_>>();
    let module = module_from_file_path(file_path);
    let mut out = Vec::new();

    for item in items {
        //dbg!(item);
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
    out: &mut Vec<schema::SymbolDoc>,
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

    if let Some((start_line, start_character, end_line)) = lsp_symbol_position_range(item) {
        let start_idx = start_line.saturating_sub(1);
        let end_idx = end_line.saturating_sub(1);
        let signature = collect_signature(lines, start_idx, end_idx);
        let docs = collect_docs_above(lines, start_idx);
        let excerpt = excerpt_from_range(lines, start_idx, end_idx, 60);
        let id = format!("symbol:{workspace_id}:{file_path}:{kind}:{name}:{start_line}:{end_line}");

        let mut doc = schema::SymbolDoc {
            id,
            kind,
            symbol: name.to_string(),
            file_path: file_path.to_string(),
            workspace_id: workspace_id.to_string(),
            uri: uri.to_string(),
            module: module.to_string(),
            symbol_path: format!("{module}::{name}"),
            crate_name: String::new(),
            edition: String::new(),
            signature,
            docs,
            hover_summary: String::new(),
            body_excerpt: excerpt,
            start_line,
            start_character,
            end_line,
            ..Default::default()
        };
        doc.finalize_derived_fields();
        out.push(doc);
    }

    if let Some(children) = item.get("children").and_then(Value::as_array) {
        for child in children {
            collect_lsp_symbol_docs(child, workspace_id, file_path, uri, module, lines, out);
        }
    }
}

fn lsp_symbol_position_range(item: &Value) -> Option<(usize, usize, usize)> {
    let range = item
        .get("range")
        .or_else(|| item.get("location").and_then(|loc| loc.get("range")))?;

    let selection_range = item.get("selectionRange").or_else(|| {
        item.get("location")
            .and_then(|loc| loc.get("selectionRange"))
    });

    let start = selection_range
        .and_then(|sel| sel.get("start"))
        .and_then(|start| start.get("line"))
        .and_then(Value::as_u64)
        .or_else(|| {
            range
                .get("start")
                .and_then(|start| start.get("line"))
                .and_then(Value::as_u64)
        })
        .unwrap_or(0) as usize
        + 1;
    let start_character = selection_range
        .and_then(|sel| sel.get("start"))
        .and_then(|start| start.get("character"))
        .and_then(Value::as_u64)
        .or_else(|| {
            range
                .get("start")
                .and_then(|start| start.get("character"))
                .and_then(Value::as_u64)
        })
        .unwrap_or(0) as usize;
    let end = range.get("end")?.get("line")?.as_u64()? as usize + 1;
    Some((start, start_character, end.max(start)))
}

fn symbol_kind_to_kind(kind: i64) -> &'static str {
    let symbol_kind = serde_json::from_value::<SymbolKind>(Value::from(kind)).ok();
    match symbol_kind {
        Some(SymbolKind::FILE) => "file",
        Some(SymbolKind::MODULE) => "mod",
        Some(SymbolKind::NAMESPACE) => "namespace",
        Some(SymbolKind::PACKAGE) => "package",
        Some(SymbolKind::CLASS) => "struct",
        Some(SymbolKind::METHOD) => "method",
        Some(SymbolKind::PROPERTY) => "property",
        Some(SymbolKind::FIELD) => "field",
        Some(SymbolKind::CONSTRUCTOR) => "ctor",
        Some(SymbolKind::ENUM) => "enum",
        Some(SymbolKind::INTERFACE) => "trait",
        Some(SymbolKind::FUNCTION) => "fn",
        Some(SymbolKind::VARIABLE) => "var",
        Some(SymbolKind::CONSTANT) => "const",
        Some(SymbolKind::STRING) => "string",
        Some(SymbolKind::NUMBER) => "number",
        Some(SymbolKind::BOOLEAN) => "bool",
        Some(SymbolKind::ARRAY) => "array",
        Some(SymbolKind::OBJECT) => "object",
        Some(SymbolKind::KEY) => "key",
        Some(SymbolKind::NULL) => "null",
        Some(SymbolKind::ENUM_MEMBER) => "enum_member",
        Some(SymbolKind::STRUCT) => "struct",
        Some(SymbolKind::EVENT) => "event",
        Some(SymbolKind::OPERATOR) => "operator",
        Some(SymbolKind::TYPE_PARAMETER) => "type_param",
        _ => "symbol",
    }
}

fn extract_symbol_docs_heuristic(
    workspace_id: &str,
    file_path: &str,
    uri: &str,
    content: &str,
) -> Vec<schema::SymbolDoc> {
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
        let signature = collect_signature(&lines, line_idx, end_line.saturating_sub(1));
        let symbol = extract_symbol(kind, line).unwrap_or_else(|| format!("item_{start_line}"));
        let item_docs = collect_docs_above(&lines, line_idx);
        let body_excerpt = excerpt_from_range(&lines, line_idx, end_line.saturating_sub(1), 60);
        let id =
            format!("symbol:{workspace_id}:{file_path}:{kind}:{symbol}:{start_line}:{end_line}");

        let mut doc = schema::SymbolDoc {
            id,
            kind: kind.to_string(),
            symbol: symbol.clone(),
            file_path: file_path.to_string(),
            workspace_id: workspace_id.to_string(),
            uri: uri.to_string(),
            module: module.clone(),
            symbol_path: format!("{module}::{symbol}"),
            crate_name: String::new(),
            edition: String::new(),
            signature,
            docs: item_docs,
            hover_summary: String::new(),
            body_excerpt,
            start_line,
            start_character: symbol_start_character(lines[line_idx], &symbol),
            end_line,
            ..Default::default()
        };
        doc.finalize_derived_fields();
        docs.push(doc);

        line_idx = end_line;
    }

    docs
}

fn symbol_docs_to_chunks(docs: &[schema::SymbolDoc]) -> Vec<CodeChunk> {
    docs.iter()
        .map(|doc| {
            let text = format!(
                // "symbol: {}\nkind: {}\nsignature: {}\nsignature_help: {}\ndocs:\n{}\nhover:\n{}\n\nexcerpt:\n{}",
                "symbol: {}\nkind: {}\nsignature: {}\ndocs:\n{}\nhover:\n{}\n\nexcerpt:\n{}",
                doc.symbol,
                doc.kind,
                doc.signature,
                // doc.signature_help,
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

fn find_crate_metadata_for_file<'a>(
    metadata_by_crate: &'a HashMap<String, CrateMetadataDoc>,
    file_path: &str,
) -> Option<&'a CrateMetadataDoc> {
    metadata_by_crate
        .iter()
        .filter(|(root, _)| {
            root.is_empty() || root.as_str() == "." || file_path.starts_with(root.as_str())
        })
        .max_by_key(|(root, _)| root.len())
        .map(|(_, doc)| doc)
}

fn apply_crate_metadata_to_symbol_docs(docs: &mut [schema::SymbolDoc], meta: &CrateMetadataDoc) {
    for doc in docs {
        doc.crate_name = meta.crate_name.clone();
        doc.edition = meta.edition.clone();
        if doc.symbol_path.is_empty() {
            doc.symbol_path = if doc.module.is_empty() {
                doc.symbol.clone()
            } else {
                format!("{}::{}", doc.module, doc.symbol)
            };
        }
    }
}

fn apply_crate_metadata_to_relation_docs(
    docs: &mut [schema::GraphEdgeDoc],
    meta: &CrateMetadataDoc,
) {
    for doc in docs {
        doc.source_crate_name = meta.crate_name.clone();
    }
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

fn collect_signature(lines: &[&str], start_idx: usize, end_idx: usize) -> String {
    if lines.is_empty() || start_idx >= lines.len() {
        return String::new();
    }

    let stop_idx = end_idx
        .min(lines.len().saturating_sub(1))
        .min(start_idx + 8);
    let mut parts = Vec::<String>::new();

    for idx in start_idx..=stop_idx {
        let line = lines[idx].trim();
        if line.is_empty() {
            if !parts.is_empty() {
                break;
            }
            continue;
        }

        parts.push(line.to_string());

        if line.contains('{') || line.ends_with(';') {
            break;
        }
    }

    parts.join(" ")
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

fn symbol_start_character(line: &str, symbol: &str) -> usize {
    if symbol.is_empty() {
        return first_non_whitespace_character(line);
    }

    for (idx, _) in line.match_indices(symbol) {
        if symbol_match_is_isolated(line, idx, symbol.len()) {
            return utf16_character_offset(line, idx);
        }
    }

    line.find(symbol)
        .map(|idx| utf16_character_offset(line, idx))
        .unwrap_or_else(|| first_non_whitespace_character(line))
}

fn symbol_position_in_lsp_range(
    lines: &[&str],
    symbol: &str,
    start_line: usize,
    end_line: usize,
) -> Option<(usize, usize)> {
    if lines.is_empty() || symbol.is_empty() {
        return None;
    }

    let start_idx = start_line
        .saturating_sub(1)
        .min(lines.len().saturating_sub(1));
    let end_idx = end_line
        .saturating_sub(1)
        .min(lines.len().saturating_sub(1));
    if start_idx > end_idx {
        return None;
    }

    for idx in start_idx..=end_idx {
        let line = lines[idx];
        for (byte_idx, _) in line.match_indices(symbol) {
            if symbol_match_is_isolated(line, byte_idx, symbol.len()) {
                return Some((idx + 1, utf16_character_offset(line, byte_idx)));
            }
        }
    }

    None
}

fn symbol_usage_position_in_file(
    lines: &[&str],
    symbol: &str,
    start_line: usize,
    end_line: usize,
) -> Option<(usize, usize)> {
    if lines.is_empty() || symbol.is_empty() {
        return None;
    }

    let start_idx = start_line
        .saturating_sub(1)
        .min(lines.len().saturating_sub(1));
    let end_idx = end_line
        .saturating_sub(1)
        .min(lines.len().saturating_sub(1));

    for (idx, line) in lines.iter().enumerate() {
        if idx >= start_idx && idx <= end_idx {
            continue;
        }
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") {
            continue;
        }
        for (byte_idx, _) in line.match_indices(symbol) {
            if symbol_match_is_isolated(line, byte_idx, symbol.len()) {
                return Some((idx + 1, utf16_character_offset(line, byte_idx)));
            }
        }
    }

    None
}

fn symbol_match_is_isolated(line: &str, start_idx: usize, symbol_len: usize) -> bool {
    let prev = line[..start_idx].chars().next_back();
    let next = line[start_idx + symbol_len..].chars().next();
    !is_rust_ident_char(prev) && !is_rust_ident_char(next)
}

fn is_rust_ident_char(ch: Option<char>) -> bool {
    matches!(ch, Some('_')) || ch.is_some_and(char::is_alphanumeric)
}

fn first_non_whitespace_character(line: &str) -> usize {
    let byte_idx = line
        .char_indices()
        .find(|(_, ch)| !ch.is_whitespace())
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    utf16_character_offset(line, byte_idx)
}

fn utf16_character_offset(line: &str, byte_idx: usize) -> usize {
    line[..byte_idx].encode_utf16().count()
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
    ) -> schema::SymbolDoc {
        let mut doc = schema::SymbolDoc {
            id: id.to_string(),
            kind: kind.to_string(),
            symbol: symbol.to_string(),
            file_path: file_path.to_string(),
            workspace_id: "ws_test".to_string(),
            uri: "file:///tmp/ws/src/main.rs".to_string(),
            module: "src::main".to_string(),
            symbol_path: format!("src::main::{symbol}"),
            crate_name: "example_crate".to_string(),
            edition: "2024".to_string(),
            signature: format!("fn {symbol}() {{}}"),
            docs: String::new(),
            hover_summary: String::new(),
            body_excerpt: excerpt.to_string(),
            start_line: 1,
            start_character: 0,
            end_line: 3,
            ..Default::default()
        };
        doc.finalize_derived_fields();
        doc
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
        let c1 = sample_item(
            "src/main.rs",
            "c1",
            "fn",
            "search_code",
            r#"fn search_code() { let query = "rust"; }"#,
        );
        let c2 = sample_item("src/main.rs", "c2", "mod", "parser", "mod parser;");
        map.insert("src/main.rs".to_string(), vec![c1, c2]);

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
    fn parse_lsp_symbols_uses_one_based_line_numbers() {
        let src = include_str!("../../rust_copilot_metrics_fixture/src/engine.rs");
        let symbols = json!([
            {
                "name": "Engine",
                "kind": 23,
                "range": {
                    "start": { "line": 3, "character": 0 },
                    "end": { "line": 9, "character": 1 }
                },
                "selectionRange": {
                    "start": { "line": 3, "character": 11 },
                    "end": { "line": 3, "character": 17 }
                }
            },
            {
                "name": "run_default",
                "kind": 12,
                "range": {
                    "start": { "line": 11, "character": 0 },
                    "end": { "line": 19, "character": 1 }
                },
                "selectionRange": {
                    "start": { "line": 11, "character": 7 },
                    "end": { "line": 11, "character": 18 }
                }
            }
        ]);

        let docs = parse_lsp_symbols(
            symbols,
            "ws_test",
            "src/engine.rs",
            "file:///tmp/ws/src/engine.rs",
            src,
        );

        let engine = docs
            .iter()
            .find(|doc| doc.symbol == "Engine")
            .expect("Engine symbol should be indexed");
        assert_eq!(engine.start_line, 4);

        let run_default = docs
            .iter()
            .find(|doc| doc.symbol == "run_default")
            .expect("run_default symbol should be indexed");
        assert_eq!(run_default.start_line, 12);
    }

    #[test]
    fn extract_symbol_docs_heuristic_keeps_where_clause_in_signature() {
        let src = r#"
pub fn pick_left<T>(left: T, _right: T) -> T
where
    T: Copy,
{
    left
}
"#;
        let docs = extract_symbol_docs_heuristic(
            "ws_test",
            "src/types.rs",
            "file:///tmp/ws/src/types.rs",
            src,
        );
        let pick_left = docs
            .iter()
            .find(|doc| doc.symbol == "pick_left")
            .expect("pick_left symbol should exist");
        assert!(pick_left.signature.contains("where"));
        assert!(pick_left.signature.contains("T: Copy"));
    }

    #[test]
    fn normalize_crate_root_maps_workspace_root_to_dot() {
        assert_eq!(normalize_crate_root(Path::new("")), ".");
        assert_eq!(normalize_crate_root(Path::new(".")), ".");
        assert_eq!(
            normalize_crate_root(Path::new("crates/core")),
            "crates/core"
        );
    }

    #[test]
    fn find_crate_metadata_for_file_supports_dot_root() {
        let root_doc = CrateMetadataDoc {
            id: "metadata:ws_test:.".to_string(),
            workspace_id: "ws_test".to_string(),
            crate_name: "root".to_string(),
            edition: "2021".to_string(),
            crate_root: ".".to_string(),
            manifest_path: "Cargo.toml".to_string(),
            features: Vec::new(),
            optional_dependencies: Vec::new(),
        };
        let nested_doc = CrateMetadataDoc {
            id: "metadata:ws_test:crates/nested".to_string(),
            workspace_id: "ws_test".to_string(),
            crate_name: "nested".to_string(),
            edition: "2021".to_string(),
            crate_root: "crates/nested".to_string(),
            manifest_path: "crates/nested/Cargo.toml".to_string(),
            features: Vec::new(),
            optional_dependencies: Vec::new(),
        };

        let mut metadata_by_crate = HashMap::new();
        metadata_by_crate.insert(root_doc.crate_root.clone(), root_doc);
        metadata_by_crate.insert(nested_doc.crate_root.clone(), nested_doc);

        let nested = find_crate_metadata_for_file(&metadata_by_crate, "crates/nested/src/lib.rs")
            .expect("nested crate metadata should resolve");
        assert_eq!(nested.crate_name, "nested");

        let root = find_crate_metadata_for_file(&metadata_by_crate, "src/main.rs")
            .expect("root crate metadata should resolve");
        assert_eq!(root.crate_name, "root");
    }

    #[test]
    fn symbol_docs_to_chunks_uses_stable_symbol_ids() {
        let docs = vec![schema::SymbolDoc {
            id: "symbol:ws_test:src/lib.rs:fn:add:1:3".to_string(),
            kind: "fn".to_string(),
            symbol: "add".to_string(),
            file_path: "src/lib.rs".to_string(),
            workspace_id: "ws_test".to_string(),
            uri: "file:///tmp/ws/src/lib.rs".to_string(),
            module: "src::lib".to_string(),
            symbol_path: "src::lib::add".to_string(),
            crate_name: "example_crate".to_string(),
            edition: "2024".to_string(),
            signature: "fn add(a: i32, b: i32) -> i32 {".to_string(),
            docs: "Adds numbers".to_string(),
            hover_summary: "hover add".to_string(),
            body_excerpt: "a + b".to_string(),
            start_line: 1,
            start_character: 3,
            end_line: 3,
            ..Default::default()
        }];
        let chunks = symbol_docs_to_chunks(&docs);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_id, "symbol:ws_test:src/lib.rs:fn:add:1:3");
    }

    #[test]
    fn canonical_symbol_doc_derives_expected_fields() {
        let mut doc = schema::SymbolDoc {
            id: "symbol:ws_test:src/lib.rs:fn:add:1:3".to_string(),
            kind: "fn".to_string(),
            symbol: "add".to_string(),
            file_path: "src/lib.rs".to_string(),
            workspace_id: "ws_test".to_string(),
            uri: "file:///tmp/ws/src/lib.rs".to_string(),
            module: "src::lib".to_string(),
            symbol_path: "src::lib::add".to_string(),
            crate_name: "example_crate".to_string(),
            edition: "2024".to_string(),
            signature: "fn add(a: i32, b: i32) -> i32 {".to_string(),
            docs: "Adds numbers".to_string(),
            hover_summary: "hover add".to_string(),
            body_excerpt: "a + b".to_string(),
            start_line: 1,
            start_character: 3,
            end_line: 3,
            ..Default::default()
        };
        doc.finalize_derived_fields();
        assert_eq!(doc.symbol_id_stable, doc.id);
        assert!(!doc.body_hash.is_empty());
        assert_eq!(doc.span.file_path, doc.file_path);
    }

    #[test]
    fn canonical_graph_edge_sets_edge_type() {
        let edge = schema::GraphEdgeDoc {
            id: "relation:ws_test:implementations:symbol:ws_test:src/lib.rs:trait:Rule:1:1:src/types.rs:4:4".to_string(),
            workspace_id: "ws_test".to_string(),
            relation_kind: "implementations".to_string(),
            edge_type: schema::GraphEdgeDoc::edge_type_for_relation_kind("implementations").to_string(),
            source_symbol_id: "symbol:ws_test:src/lib.rs:trait:Rule:1:1".to_string(),
            source_symbol: "Rule".to_string(),
            source_file_path: "src/lib.rs".to_string(),
            source_crate_name: "example_crate".to_string(),
            source_uri: "file:///tmp/ws/src/lib.rs".to_string(),
            target_file_path: "src/types.rs".to_string(),
            target_uri: "file:///tmp/ws/src/types.rs".to_string(),
            target_start_line: 4,
            target_end_line: 4,
            target_excerpt: "impl crate::Rule for PositiveRule {".to_string(),
        };
        assert_eq!(edge.edge_type, "impl_to_trait");
        assert_eq!(edge.relation_kind, "implementations");
    }

    #[test]
    fn rust_item_search_text_contains_canonical_fields() {
        let doc = schema::SymbolDoc {
            id: "symbol:ws_test:src/lib.rs:fn:add:1:3".to_string(),
            kind: "fn".to_string(),
            symbol: "add".to_string(),
            file_path: "src/lib.rs".to_string(),
            workspace_id: "ws_test".to_string(),
            uri: "file:///tmp/ws/src/lib.rs".to_string(),
            module: "src::lib".to_string(),
            symbol_path: "src::lib::add".to_string(),
            crate_name: "example_crate".to_string(),
            edition: "2024".to_string(),
            signature: "fn add(a: i32, b: i32) -> i32 {".to_string(),
            docs: "Adds numbers".to_string(),
            hover_summary: "hover add".to_string(),
            body_excerpt: "a + b".to_string(),
            start_line: 1,
            start_character: 3,
            end_line: 3,
            ..Default::default()
        };
        let search_text = rust_item_search_text(&doc);
        assert!(search_text.contains("add"));
        assert!(search_text.contains("module: src::lib"));
        assert!(search_text.contains("path: src/lib.rs"));
        assert!(search_text.contains("kind: fn"));
        assert!(search_text.contains("uri: file:///tmp/ws/src/lib.rs"));
        assert!(search_text.contains("workspace_id: ws_test"));
    }

    #[test]
    fn method_not_found_error_is_detected() {
        let err = anyhow::anyhow!(
            "rust-analyzer returned error: {{\"code\":-32601,\"message\":\"Method not found\"}}"
        );
        assert!(is_method_unsupported_error(&err));
    }

    #[test]
    fn timeout_error_is_treated_as_transient() {
        let err = anyhow::anyhow!("rust-analyzer request timed out: rust-analyzer/viewCrateGraph");
        assert!(is_transient_lsp_error(&err));
    }

    #[test]
    fn explain_relevance_request_accepts_query_only_payload() {
        let req: ExplainRelevanceRequest = serde_json::from_value(json!({
            "query": "find trait impls",
            "limit": 3,
            "score_threshold": 1.3
        }))
        .expect("query-only explain_relevance payload should deserialize");

        assert_eq!(req.query, "find trait impls");
        assert_eq!(req.limit, Some(3));
        assert_eq!(req.score_threshold, Some(1.3));
        assert!(req.point_ids.is_none());
    }

    #[test]
    fn maybe_upgrade_symbol_path_does_not_downgrade_to_unqualified_name() {
        let mut doc = sample_item("src/lib.rs", "c1", "trait", "Rule", "trait Rule");
        doc.module = "src::lib".to_string();
        doc.symbol_path = "src::lib::Rule".to_string();

        maybe_upgrade_symbol_path(&mut doc, "Rule");

        assert_eq!(doc.symbol_path, "src::lib::Rule");
    }

    #[test]
    fn maybe_upgrade_symbol_path_updates_empty_path_with_workspace_value() {
        let mut doc = sample_item("src/lib.rs", "c2", "fn", "evaluate", "fn evaluate()");
        doc.module = "src::lib".to_string();
        doc.symbol_path.clear();

        maybe_upgrade_symbol_path(&mut doc, "evaluate");

        assert_eq!(doc.symbol_path, "src::lib::evaluate");
    }

    #[test]
    fn relation_target_is_low_signal_for_import_and_module_lines() {
        assert!(relation_target_is_low_signal(
            "use crate::types::ValueState;"
        ));
        assert!(relation_target_is_low_signal(
            "pub use crate::engine::run_default;"
        ));
        assert!(relation_target_is_low_signal("mod engine;"));
        assert!(relation_target_is_low_signal("pub mod engine;"));
        assert!(relation_target_is_low_signal("extern crate alloc;"));
    }

    #[test]
    fn relation_target_is_not_low_signal_for_real_code_targets() {
        assert!(!relation_target_is_low_signal(
            "pub fn run_default(left: i32, right: i32) -> ValueState {"
        ));
        assert!(!relation_target_is_low_signal(
            "impl Engine {\n    pub fn run(...) -> bool {"
        ));
    }

    #[test]
    fn excerpt_is_impl_for_symbol_detects_impl_blocks() {
        assert!(excerpt_is_impl_for_symbol(
            "impl crate::Rule for PositiveRule {",
            "PositiveRule"
        ));
        assert!(excerpt_is_impl_for_symbol("impl Engine {", "Engine"));
        assert!(!excerpt_is_impl_for_symbol(
            "let rule = PositiveRule;",
            "PositiveRule"
        ));
    }

    #[test]
    fn summarize_crate_graph_payload_compacts_dot_for_target_crate() {
        let payload = json!(
            "digraph rust_analyzer_crate_graph {\n\
             _1[label=\\\"fixture\\\"][shape=\\\"box\\\"];\n\
             _2[label=\\\"helper\\\"][shape=\\\"box\\\"];\n\
             _3[label=\\\"serde\\\"][shape=\\\"box\\\"];\n\
             _1 -> _2[label=\\\"\\\"];\n\
             _1 -> _3[label=\\\"\\\"];\n\
             _2 -> _1[label=\\\"\\\"];\n\
             }\n"
        );
        let summary = summarize_crate_graph_payload(&payload, "fixture");
        assert!(summary.contains("crate: fixture"));
        assert!(summary.contains("node_found: true"));
        assert!(summary.contains("outbound_deps_count:"));
        assert!(summary.contains("inbound_deps_count:"));
        assert!(summary.contains("outbound_deps:"));
        assert!(summary.contains("inbound_deps:"));
        assert!(!summary.contains("digraph rust_analyzer_crate_graph"));
    }

    #[test]
    fn summarize_crate_graph_payload_handles_missing_crate_node() {
        let payload = json!(
            "digraph rust_analyzer_crate_graph {\n\
             _1[label=\\\"other\\\"][shape=\\\"box\\\"];\n\
             }\n"
        );
        let summary = summarize_crate_graph_payload(&payload, "fixture");
        assert!(summary.contains("crate: fixture"));
        assert!(summary.contains("node_found: false"));
        assert!(!summary.contains("digraph rust_analyzer_crate_graph"));
    }

    #[test]
    fn build_diagnostic_docs_from_lsp_uses_latest_lsp_shape() {
        let docs = build_diagnostic_docs_from_lsp(
            "ws_test",
            "src/lib.rs",
            "file:///tmp/ws/src/lib.rs",
            &[json!({
                "severity": 1,
                "code": "E0308",
                "message": "mismatched types",
                "range": {
                    "start": { "line": 9, "character": 4 },
                    "end": { "line": 9, "character": 12 }
                }
            })],
        );
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].severity, "error");
        assert_eq!(docs[0].code.as_deref(), Some("E0308"));
        assert_eq!(docs[0].span.start_line, 10);
        assert_eq!(docs[0].span.end_line, 10);
    }

    #[test]
    fn update_lsp_side_channel_state_tracks_publish_diagnostics() {
        let mut in_flight = 0usize;
        let mut saw_progress = false;
        let mut diagnostics_by_uri = HashMap::<String, Vec<Value>>::new();

        update_lsp_side_channel_state(
            &json!({
                "jsonrpc": "2.0",
                "method": "textDocument/publishDiagnostics",
                "params": {
                    "uri": "file:///tmp/ws/src/lib.rs",
                    "diagnostics": [
                        {
                            "severity": 2,
                            "message": "unused variable",
                            "range": {
                                "start": { "line": 1, "character": 0 },
                                "end": { "line": 1, "character": 3 }
                            }
                        }
                    ]
                }
            }),
            &mut in_flight,
            &mut saw_progress,
            &mut diagnostics_by_uri,
        );

        assert_eq!(
            diagnostics_by_uri
                .get("file:///tmp/ws/src/lib.rs")
                .map(std::vec::Vec::len),
            Some(1)
        );
    }
}
