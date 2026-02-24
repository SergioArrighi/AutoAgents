use serde::{Deserialize, Serialize};

/// Canonical span for symbols/edges.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub(crate) struct Span {
    pub(crate) file_path: String,
    pub(crate) uri: String,
    pub(crate) start_line: usize,
    pub(crate) start_character: usize,
    pub(crate) end_line: usize,
    pub(crate) end_character: Option<usize>,
}

/// Canonical symbol schema.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub(crate) struct SymbolDoc {
    pub(crate) id: String,
    pub(crate) kind: String,
    pub(crate) symbol: String,
    pub(crate) file_path: String,
    pub(crate) workspace_id: String,
    pub(crate) uri: String,
    pub(crate) module: String,
    pub(crate) symbol_path: String,
    pub(crate) crate_name: String,
    pub(crate) edition: String,
    pub(crate) signature: String,
    pub(crate) docs: String,
    pub(crate) hover_summary: String,
    pub(crate) body_excerpt: String,
    pub(crate) start_line: usize,
    pub(crate) start_character: usize,
    pub(crate) end_line: usize,
    #[serde(default)]
    pub(crate) span: Span,
    #[serde(default)]
    pub(crate) visibility: Option<String>,
    #[serde(default)]
    pub(crate) generics: Option<String>,
    #[serde(default)]
    pub(crate) where_clause: Option<String>,
    #[serde(default)]
    pub(crate) receiver: Option<String>,
    #[serde(default)]
    pub(crate) return_type: Option<String>,
    #[serde(default)]
    pub(crate) attrs: Vec<String>,
    #[serde(default)]
    pub(crate) cfgs: Vec<String>,
    #[serde(default)]
    pub(crate) deprecated: bool,
    #[serde(default)]
    pub(crate) stability: Option<String>,
    #[serde(default)]
    pub(crate) body_hash: String,
    #[serde(default)]
    pub(crate) symbol_id_stable: String,
    #[serde(default)]
    pub(crate) semantic_tokens_summary: String,
    #[serde(default)]
    pub(crate) syntax_tree_summary: String,
    #[serde(default)]
    pub(crate) inlay_hints_summary: String,
}

impl SymbolDoc {
    pub(crate) fn finalize_derived_fields(&mut self) {
        self.span = Span {
            file_path: self.file_path.clone(),
            uri: self.uri.clone(),
            start_line: self.start_line,
            start_character: self.start_character,
            end_line: self.end_line,
            end_character: None,
        };
        self.visibility = infer_visibility(&self.signature);
        self.generics = infer_generics(&self.signature, &self.symbol);
        self.where_clause = infer_where_clause(&self.signature);
        self.receiver = infer_receiver(&self.signature);
        self.return_type = infer_return_type(&self.signature);
        self.body_hash = crate::simple_hash(&self.body_excerpt);
        self.symbol_id_stable = self.id.clone();
    }
}

fn infer_visibility(signature: &str) -> Option<String> {
    let s = signature.trim_start();
    if s.starts_with("pub(") {
        if let Some(end) = s.find(')') {
            return Some(s[..=end].to_string());
        }
    }
    if s.starts_with("pub ") {
        return Some("pub".to_string());
    }
    None
}

fn infer_generics(signature: &str, symbol: &str) -> Option<String> {
    let s = signature.trim();
    let idx = s.find(symbol)?;
    let mut tail = &s[idx + symbol.len()..];
    tail = tail.trim_start();
    if !tail.starts_with('<') {
        return None;
    }

    let mut depth = 0usize;
    let mut end_byte = None;
    for (i, ch) in tail.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
                if depth == 0 {
                    end_byte = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = end_byte?;
    Some(tail[..=end].to_string())
}

fn infer_where_clause(signature: &str) -> Option<String> {
    let s = signature.trim();
    let where_idx = s.find(" where ")?;
    let mut clause = s[where_idx + 1..].trim().to_string();
    if let Some(pos) = clause.find('{') {
        clause = clause[..pos].trim().to_string();
    }
    if clause.is_empty() {
        None
    } else {
        Some(clause)
    }
}

fn infer_receiver(signature: &str) -> Option<String> {
    let s = signature.trim();
    let open = s.find('(')?;
    let close = s[open + 1..].find(')')?;
    let params = &s[open + 1..open + 1 + close];
    let first = params.split(',').next()?.trim();
    let normalized = first.replace(' ', "");
    if normalized.contains("self") {
        Some(first.to_string())
    } else {
        None
    }
}

fn infer_return_type(signature: &str) -> Option<String> {
    let s = signature.trim();
    let arrow = s.find("->")?;
    let mut tail = s[arrow + 2..].trim().to_string();
    if let Some(idx) = tail.find(" where ") {
        tail = tail[..idx].trim().to_string();
    }
    if let Some(idx) = tail.find('{') {
        tail = tail[..idx].trim().to_string();
    }
    if tail.is_empty() { None } else { Some(tail) }
}

/// Canonical typed graph edge.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub(crate) struct GraphEdgeDoc {
    pub(crate) id: String,
    pub(crate) workspace_id: String,
    pub(crate) relation_kind: String,
    pub(crate) edge_type: String,
    pub(crate) source_symbol_id: String,
    pub(crate) source_symbol: String,
    pub(crate) source_file_path: String,
    pub(crate) source_crate_name: String,
    pub(crate) source_uri: String,
    pub(crate) target_file_path: String,
    pub(crate) target_uri: String,
    pub(crate) target_start_line: usize,
    pub(crate) target_end_line: usize,
    pub(crate) target_excerpt: String,
}

impl GraphEdgeDoc {
    pub(crate) fn edge_type_for_relation_kind(relation_kind: &str) -> &'static str {
        match relation_kind {
            "references" => "symbol_reference",
            "implementations" => "impl_to_trait",
            "definitions" => "definition",
            "type_definitions" => "type_usage",
            _ => "other",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub(crate) struct FileDoc {
    pub(crate) id: String,
    pub(crate) workspace_id: String,
    pub(crate) crate_name: String,
    pub(crate) file_path: String,
    pub(crate) uri: String,
    pub(crate) module: String,
    pub(crate) body_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub(crate) struct CallEdge {
    pub(crate) id: String,
    pub(crate) workspace_id: String,
    pub(crate) source_symbol_id: String,
    pub(crate) target_symbol_id: String,
    pub(crate) source_span: Span,
    pub(crate) target_span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub(crate) struct TypeEdge {
    pub(crate) id: String,
    pub(crate) workspace_id: String,
    pub(crate) source_symbol_id: String,
    pub(crate) target_symbol_id: String,
    pub(crate) relation_kind: String,
    pub(crate) source_span: Span,
    pub(crate) target_span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub(crate) struct DiagnosticDoc {
    pub(crate) id: String,
    pub(crate) workspace_id: String,
    pub(crate) source: String,
    pub(crate) file_path: String,
    pub(crate) uri: String,
    pub(crate) severity: String,
    pub(crate) code: Option<String>,
    pub(crate) message: String,
    pub(crate) span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub(crate) struct SemanticTokenDoc {
    pub(crate) id: String,
    pub(crate) workspace_id: String,
    pub(crate) file_path: String,
    pub(crate) uri: String,
    pub(crate) symbol_id: Option<String>,
    pub(crate) token_count: usize,
    pub(crate) token_type_histogram: Vec<String>,
    pub(crate) summary: String,
    pub(crate) span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub(crate) struct SyntaxTreeDoc {
    pub(crate) id: String,
    pub(crate) workspace_id: String,
    pub(crate) file_path: String,
    pub(crate) uri: String,
    pub(crate) symbol_id: Option<String>,
    pub(crate) node_kind_histogram: Vec<String>,
    pub(crate) summary: String,
    pub(crate) span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub(crate) struct InlayHintDoc {
    pub(crate) id: String,
    pub(crate) workspace_id: String,
    pub(crate) file_path: String,
    pub(crate) uri: String,
    pub(crate) symbol_id: Option<String>,
    pub(crate) hint_count: usize,
    pub(crate) summary: String,
    pub(crate) span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub(crate) struct CrateGraphDoc {
    pub(crate) id: String,
    pub(crate) workspace_id: String,
    pub(crate) crate_name: String,
    pub(crate) crate_root: String,
    pub(crate) summary: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_visibility_from_signature() {
        assert_eq!(infer_visibility("pub fn run() {}"), Some("pub".to_string()));
        assert_eq!(
            infer_visibility("pub(crate) fn run() {}"),
            Some("pub(crate)".to_string())
        );
        assert_eq!(infer_visibility("fn run() {}"), None);
    }

    #[test]
    fn infer_signature_parts() {
        let sig = "pub fn run<T>(self, value: T) -> Result<T, E> where T: Clone {";
        assert_eq!(infer_generics(sig, "run"), Some("<T>".to_string()));
        assert_eq!(infer_where_clause(sig), Some("where T: Clone".to_string()));
        assert_eq!(infer_receiver(sig), Some("self".to_string()));
        assert_eq!(infer_return_type(sig), Some("Result<T, E>".to_string()));
    }
}
