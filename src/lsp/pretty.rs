//! Type pretty-printing for display in IDE hovers.
//!
//! This module provides a thin wrapper around `trunk_ir::type_interface::Printable`
//! for use in the LSP server.

// TrunkIR-based functions are used only in tests
#![allow(dead_code)]

use lsp_types::{
    Documentation, MarkupContent, MarkupKind, ParameterInformation, ParameterLabel, SignatureHelp,
    SignatureInformation,
};
use trunk_ir::dialect::core::{EffectRowType, Func};
use trunk_ir::type_interface;
use trunk_ir::{DialectType, Symbol, Type};

/// Pretty-print a type to a user-friendly string.
pub fn print_type(db: &dyn salsa::Database, ty: Type<'_>) -> String {
    type_interface::print_type(db, ty)
}

/// Format a function signature for LSP signature help.
pub fn format_signature(
    db: &dyn salsa::Database,
    func_ty: Type<'_>,
    func_name: &str,
    param_names: &[Option<Symbol>],
    doc_comment: Option<&str>,
    active_param: u32,
) -> SignatureHelp {
    let Some(func) = Func::from_type(db, func_ty) else {
        // Not a function type, return empty signature help
        return SignatureHelp {
            signatures: vec![],
            active_signature: None,
            active_parameter: None,
        };
    };

    let params = func.params(db);
    let result = func.result(db);
    let effect = func.effect(db);

    // Build parameter information
    let mut param_infos = Vec::with_capacity(params.len());
    let mut label_parts = Vec::with_capacity(params.len());

    for (i, &param_ty) in params.iter().enumerate() {
        let param_name = param_names
            .get(i)
            .and_then(|n| n.as_ref())
            .map(|s| s.to_string());

        let type_str = print_type(db, param_ty);
        let label = if let Some(name) = &param_name {
            format!("{}: {}", name, type_str)
        } else {
            type_str.clone()
        };

        label_parts.push(label.clone());

        param_infos.push(ParameterInformation {
            label: ParameterLabel::Simple(label),
            documentation: None, // TODO: parameter-level doc comments
        });
    }

    // Build the full signature label
    let params_str = label_parts.join(", ");
    let result_str = print_type(db, result);

    let signature_label = if let Some(eff) = effect {
        if let Some(row) = EffectRowType::from_type(db, eff) {
            if row.is_empty(db) {
                format!("fn {}({}) -> {}", func_name, params_str, result_str)
            } else {
                let effect_str = format_effect_row(db, &row);
                format!(
                    "fn {}({}) ->{{{}}} {}",
                    func_name, params_str, effect_str, result_str
                )
            }
        } else {
            format!("fn {}({}) -> {}", func_name, params_str, result_str)
        }
    } else {
        format!("fn {}({}) -> {}", func_name, params_str, result_str)
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

/// Format an effect row for display.
fn format_effect_row(db: &dyn salsa::Database, row: &EffectRowType<'_>) -> String {
    let abilities = row.abilities(db);
    let tail = row.tail_var(db);

    let ability_strs: Vec<String> = abilities.iter().map(|&a| print_type(db, a)).collect();

    if let Some(tail_id) = tail {
        let tail_name = type_var_name(tail_id);
        if ability_strs.is_empty() {
            tail_name
        } else {
            format!("{} | {}", ability_strs.join(", "), tail_name)
        }
    } else {
        ability_strs.join(", ")
    }
}

/// Generate a type variable name from an ID.
fn type_var_name(id: u64) -> String {
    if id < 26 {
        // a-z
        char::from_u32('a' as u32 + id as u32)
            .map(|c| c.to_string())
            .unwrap_or_else(|| format!("t{}", id))
    } else {
        // t0, t1, ...
        format!("t{}", id - 26)
    }
}

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
    let signature_label = format!("fn {}({}) -> {}", sig.name, params_str, return_str);

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
    use salsa_test_macros::salsa_test;
    use tribute_ir::dialect::{tribute, tribute_rt};
    use trunk_ir::dialect::core::{AbilityRefType, EffectRowType, Func, Nil};
    use trunk_ir::{IdVec, Symbol, idvec};

    #[salsa_test]
    fn test_print_basic_types(db: &salsa::DatabaseImpl) {
        // Int (31-bit signed)
        let int_ty = tribute_rt::int_type(db);
        assert_eq!(print_type(db, int_ty), "Int");

        // Nil
        let nil_ty = *Nil::new(db);
        assert_eq!(print_type(db, nil_ty), "()");
    }

    #[salsa_test]
    fn test_print_function_type(db: &salsa::DatabaseImpl) {
        let int_ty = tribute_rt::int_type(db);

        // fn(Int, Int) -> Int
        let func_ty = *Func::new(db, idvec![int_ty, int_ty], int_ty);
        assert_eq!(print_type(db, func_ty), "fn(Int, Int) -> Int");

        // fn() -> ()
        let nil_ty = *Nil::new(db);
        let unit_func = *Func::new(db, IdVec::new(), nil_ty);
        assert_eq!(print_type(db, unit_func), "fn() -> ()");
    }

    #[salsa_test]
    fn test_print_effect_row(db: &salsa::DatabaseImpl) {
        // Empty row
        let empty = *EffectRowType::empty(db);
        assert_eq!(print_type(db, empty), "{}");

        // Row with ability
        let console = *AbilityRefType::simple(db, Symbol::new("Console"));
        let row = *EffectRowType::concrete(db, idvec![console]);
        assert_eq!(print_type(db, row), "{Console}");

        // Row with tail variable
        let open_row = *EffectRowType::with_tail(db, idvec![console], 4); // 'e' = id 4
        assert_eq!(print_type(db, open_row), "{Console | e}");
    }

    #[salsa_test]
    fn test_print_type_var(db: &salsa::DatabaseImpl) {
        let var_a = tribute::type_var_with_id(db, 0);
        assert_eq!(print_type(db, var_a), "a");

        let var_z = tribute::type_var_with_id(db, 25);
        assert_eq!(print_type(db, var_z), "z");

        let var_t0 = tribute::type_var_with_id(db, 26);
        assert_eq!(print_type(db, var_t0), "t0");
    }

    #[salsa_test]
    fn test_format_signature_basic(db: &salsa::DatabaseImpl) {
        let int_ty = tribute_rt::int_type(db);
        let func_ty = *Func::new(db, idvec![int_ty, int_ty], int_ty);

        let param_names = vec![Some(Symbol::new("x")), Some(Symbol::new("y"))];
        let result = format_signature(db, func_ty, "add", &param_names, None, 0);

        assert_eq!(result.signatures.len(), 1);
        let sig = &result.signatures[0];
        assert_eq!(sig.label, "fn add(x: Int, y: Int) -> Int");
        assert_eq!(result.active_parameter, Some(0));
    }

    #[salsa_test]
    fn test_format_signature_with_doc(db: &salsa::DatabaseImpl) {
        let nil_ty = *Nil::new(db);
        let func_ty = *Func::new(db, IdVec::new(), nil_ty);

        let result = format_signature(db, func_ty, "hello", &[], Some("Says hello"), 0);

        assert_eq!(result.signatures.len(), 1);
        let sig = &result.signatures[0];
        assert_eq!(sig.label, "fn hello() -> ()");
        assert!(sig.documentation.is_some());
    }

    #[salsa_test]
    fn test_format_signature_non_function(db: &salsa::DatabaseImpl) {
        let int_ty = tribute_rt::int_type(db);

        // Passing a non-function type should return empty signatures
        let result = format_signature(db, int_ty, "not_a_func", &[], None, 0);

        assert!(result.signatures.is_empty());
    }

    #[test]
    fn test_type_var_name() {
        assert_eq!(type_var_name(0), "a");
        assert_eq!(type_var_name(1), "b");
        assert_eq!(type_var_name(25), "z");
        assert_eq!(type_var_name(26), "t0");
        assert_eq!(type_var_name(27), "t1");
    }
}
