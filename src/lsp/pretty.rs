//! Type pretty-printing for display in IDE hovers.
//!
//! This module provides AST-based signature formatting for the LSP server.

use lsp_types::{
    Documentation, MarkupContent, MarkupKind, ParameterInformation, ParameterLabel, SignatureHelp,
    SignatureInformation,
};
/// Format a function signature for LSP signature help (AST-based).
pub fn format_ast_signature(
    sig: &super::completion_index::FunctionSignature,
    doc_comment: Option<&str>,
    active_param: u32,
) -> SignatureHelp {
    // Build parameter information
    let mut param_infos = Vec::with_capacity(sig.params.len());
    let mut label_parts = Vec::with_capacity(sig.params.len());

    for (name, ty_str) in &sig.params {
        let label = if let Some(ty) = ty_str {
            format!("{}: {}", name, ty)
        } else {
            name.to_string()
        };

        label_parts.push(label.clone());

        param_infos.push(ParameterInformation {
            label: ParameterLabel::Simple(label),
            documentation: None,
        });
    }

    // Build the full signature label
    let params_str = label_parts.join(", ");
    let return_str = sig.return_ty.as_deref().unwrap_or("_");
    let signature_label = if let Some(effects) = &sig.effects {
        format!(
            "fn {}({}) ->{{{}}} {}",
            sig.name, params_str, effects, return_str
        )
    } else {
        format!("fn {}({}) -> {}", sig.name, params_str, return_str)
    };

    let documentation = doc_comment.map(|doc| {
        Documentation::MarkupContent(MarkupContent {
            kind: MarkupKind::Markdown,
            value: doc.to_string(),
        })
    });

    let signature = SignatureInformation {
        label: signature_label,
        documentation,
        parameters: Some(param_infos),
        active_parameter: Some(active_param),
    };

    SignatureHelp {
        signatures: vec![signature],
        active_signature: Some(0),
        active_parameter: Some(active_param),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::{Span, Symbol};

    #[test]
    fn test_format_ast_signature_basic() {
        use super::super::completion_index::FunctionSignature;

        let sig = FunctionSignature {
            name: Symbol::new("add"),
            params: vec![
                (Symbol::new("a"), Some("Int".to_string())),
                (Symbol::new("b"), Some("Int".to_string())),
            ],
            return_ty: Some("Int".to_string()),
            effects: None,
            span: Span::default(),
        };

        let result = format_ast_signature(&sig, None, 0);
        assert_eq!(result.signatures.len(), 1);
        assert_eq!(result.signatures[0].label, "fn add(a: Int, b: Int) -> Int");
        assert_eq!(result.active_parameter, Some(0));
    }

    #[test]
    fn test_format_ast_signature_with_effects() {
        use super::super::completion_index::FunctionSignature;

        let sig = FunctionSignature {
            name: Symbol::new("greet"),
            params: vec![(Symbol::new("name"), Some("Text".to_string()))],
            return_ty: Some("Nil".to_string()),
            effects: Some("Console".to_string()),
            span: Span::default(),
        };

        let result = format_ast_signature(&sig, None, 0);
        assert_eq!(result.signatures.len(), 1);
        assert_eq!(
            result.signatures[0].label,
            "fn greet(name: Text) ->{Console} Nil"
        );
    }

    #[test]
    fn test_format_ast_signature_multiple_effects() {
        use super::super::completion_index::FunctionSignature;

        let sig = FunctionSignature {
            name: Symbol::new("fetch"),
            params: vec![(Symbol::new("url"), Some("Text".to_string()))],
            return_ty: Some("Response".to_string()),
            effects: Some("Http, Async".to_string()),
            span: Span::default(),
        };

        let result = format_ast_signature(&sig, None, 0);
        assert_eq!(result.signatures.len(), 1);
        assert_eq!(
            result.signatures[0].label,
            "fn fetch(url: Text) ->{Http, Async} Response"
        );
    }

    #[test]
    fn test_format_ast_signature_with_doc_comment() {
        use super::super::completion_index::FunctionSignature;

        let sig = FunctionSignature {
            name: Symbol::new("hello"),
            params: vec![],
            return_ty: Some("Nil".to_string()),
            effects: None,
            span: Span::default(),
        };

        let result = format_ast_signature(&sig, Some("Says hello"), 0);
        assert_eq!(result.signatures.len(), 1);
        assert!(result.signatures[0].documentation.is_some());
    }

    #[test]
    fn test_format_ast_signature_active_param() {
        use super::super::completion_index::FunctionSignature;

        let sig = FunctionSignature {
            name: Symbol::new("add"),
            params: vec![
                (Symbol::new("a"), Some("Int".to_string())),
                (Symbol::new("b"), Some("Int".to_string())),
            ],
            return_ty: Some("Int".to_string()),
            effects: None,
            span: Span::default(),
        };

        let result = format_ast_signature(&sig, None, 1);
        assert_eq!(result.active_parameter, Some(1));
        assert_eq!(result.signatures[0].active_parameter, Some(1));
    }
}
