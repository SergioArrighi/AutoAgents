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
///
/// Compatibility: legacy `RustItemDoc` fields are kept at top level so existing
/// readers can deserialize this payload without changes.
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
}

impl SymbolDoc {
    pub(crate) fn from_legacy(item: &crate::RustItemDoc) -> Self {
        let visibility = infer_visibility(&item.signature);
        let generics = infer_generics(&item.signature, &item.symbol);
        let where_clause = infer_where_clause(&item.signature);
        let receiver = infer_receiver(&item.signature);
        let return_type = infer_return_type(&item.signature);
        Self {
            id: item.id.clone(),
            kind: item.kind.clone(),
            symbol: item.symbol.clone(),
            file_path: item.file_path.clone(),
            workspace_id: item.workspace_id.clone(),
            uri: item.uri.clone(),
            module: item.module.clone(),
            symbol_path: item.symbol_path.clone(),
            crate_name: item.crate_name.clone(),
            edition: item.edition.clone(),
            signature: item.signature.clone(),
            docs: item.docs.clone(),
            hover_summary: item.hover_summary.clone(),
            body_excerpt: item.body_excerpt.clone(),
            start_line: item.start_line,
            start_character: item.start_character,
            end_line: item.end_line,
            span: Span {
                file_path: item.file_path.clone(),
                uri: item.uri.clone(),
                start_line: item.start_line,
                start_character: item.start_character,
                end_line: item.end_line,
                end_character: None,
            },
            visibility,
            generics,
            where_clause,
            receiver,
            return_type,
            attrs: Vec::new(),
            cfgs: Vec::new(),
            deprecated: false,
            stability: None,
            body_hash: crate::simple_hash(&item.body_excerpt),
            symbol_id_stable: item.id.clone(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn to_legacy(&self) -> crate::RustItemDoc {
        crate::RustItemDoc {
            id: self.id.clone(),
            kind: self.kind.clone(),
            symbol: self.symbol.clone(),
            file_path: self.file_path.clone(),
            workspace_id: self.workspace_id.clone(),
            uri: self.uri.clone(),
            module: self.module.clone(),
            symbol_path: self.symbol_path.clone(),
            crate_name: self.crate_name.clone(),
            edition: self.edition.clone(),
            signature: self.signature.clone(),
            docs: self.docs.clone(),
            hover_summary: self.hover_summary.clone(),
            body_excerpt: self.body_excerpt.clone(),
            start_line: self.start_line,
            start_character: self.start_character,
            end_line: self.end_line,
        }
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
///
/// Compatibility: legacy `SymbolRelationDoc` fields are kept at top level so
/// existing readers can deserialize this payload without changes.
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
    pub(crate) fn from_legacy(item: &crate::SymbolRelationDoc) -> Self {
        let edge_type = match item.relation_kind.as_str() {
            "references" => "symbol_reference",
            "implementations" => "impl_to_trait",
            "definitions" => "definition",
            "type_definitions" => "type_usage",
            _ => "other",
        };
        Self {
            id: item.id.clone(),
            workspace_id: item.workspace_id.clone(),
            relation_kind: item.relation_kind.clone(),
            edge_type: edge_type.to_string(),
            source_symbol_id: item.source_symbol_id.clone(),
            source_symbol: item.source_symbol.clone(),
            source_file_path: item.source_file_path.clone(),
            source_crate_name: item.source_crate_name.clone(),
            source_uri: item.source_uri.clone(),
            target_file_path: item.target_file_path.clone(),
            target_uri: item.target_uri.clone(),
            target_start_line: item.target_start_line,
            target_end_line: item.target_end_line,
            target_excerpt: item.target_excerpt.clone(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn to_legacy(&self) -> crate::SymbolRelationDoc {
        crate::SymbolRelationDoc {
            id: self.id.clone(),
            workspace_id: self.workspace_id.clone(),
            relation_kind: self.relation_kind.clone(),
            source_symbol_id: self.source_symbol_id.clone(),
            source_symbol: self.source_symbol.clone(),
            source_file_path: self.source_file_path.clone(),
            source_crate_name: self.source_crate_name.clone(),
            source_uri: self.source_uri.clone(),
            target_file_path: self.target_file_path.clone(),
            target_uri: self.target_uri.clone(),
            target_start_line: self.target_start_line,
            target_end_line: self.target_end_line,
            target_excerpt: self.target_excerpt.clone(),
        }
    }
}

#[allow(dead_code)]
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

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub(crate) struct CallEdge {
    pub(crate) id: String,
    pub(crate) workspace_id: String,
    pub(crate) source_symbol_id: String,
    pub(crate) target_symbol_id: String,
    pub(crate) source_span: Span,
    pub(crate) target_span: Span,
}

#[allow(dead_code)]
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

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub(crate) struct DiagnosticDoc {
    pub(crate) id: String,
    pub(crate) workspace_id: String,
    pub(crate) file_path: String,
    pub(crate) uri: String,
    pub(crate) severity: String,
    pub(crate) code: Option<String>,
    pub(crate) message: String,
    pub(crate) span: Span,
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
        assert_eq!(
            infer_where_clause(sig),
            Some("where T: Clone".to_string())
        );
        assert_eq!(infer_receiver(sig), Some("self".to_string()));
        assert_eq!(
            infer_return_type(sig),
            Some("Result<T, E>".to_string())
        );
    }
}
