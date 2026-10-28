//! Source-position index for compiler-owned bindings without declarations.

use tribute_front::SourceCst;
use tribute_front::ast::{
    AbilityId, Decl, DeclTable, SpanMap, TypeAnnotation, TypeAnnotationKind, TypedRef,
};
use tribute_front::query as ast_query;
use trunk_ir::{Span, Symbol};

/// A compiler-owned symbol referenced from source.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BuiltinSymbolEntry {
    pub span: Span,
    pub name: Symbol,
    pub qualified: Symbol,
}

/// Compiler-owned symbol references in a source file.
#[salsa::tracked(returns(deref))]
pub fn builtin_symbols(db: &dyn salsa::Database, source: SourceCst) -> Vec<BuiltinSymbolEntry> {
    let Some(output) = tribute::parse_and_lower_ast(db, source) else {
        return Vec::new();
    };
    let module = output.module(db);
    let Some(span_map) = ast_query::span_map(db, source) else {
        return Vec::new();
    };

    let mut entries = Vec::new();
    collect_module(db, module, span_map, &mut entries);
    entries.sort_by_key(|entry| (entry.span.start, entry.span.end));
    entries
}

pub fn builtin_at(entries: &[BuiltinSymbolEntry], offset: usize) -> Option<&BuiltinSymbolEntry> {
    entries
        .iter()
        .filter(|entry| entry.span.range().contains(&offset))
        .min_by_key(|entry| entry.span.end - entry.span.start)
}

fn collect_module(
    db: &dyn salsa::Database,
    module: &DeclTable<TypedRef<'_>>,
    span_map: &SpanMap,
    entries: &mut Vec<BuiltinSymbolEntry>,
) {
    for decl in module.decls() {
        if let Decl::Function(func) = decl {
            for effect in func.effects.iter().flatten() {
                collect_effect(db, effect, span_map, entries);
            }
        }
    }
}

fn collect_effect(
    db: &dyn salsa::Database,
    annotation: &TypeAnnotation,
    span_map: &SpanMap,
    entries: &mut Vec<BuiltinSymbolEntry>,
) {
    let ability_annotation = match &annotation.kind {
        TypeAnnotationKind::App { ctor, .. } => ctor.as_ref(),
        _ => annotation,
    };
    let TypeAnnotationKind::Path(path) = &ability_annotation.kind else {
        return;
    };

    // An annotation without a span here is in a file module's file.
    let Some(span) = span_map.get(ability_annotation.id) else {
        return;
    };
    if let Some(ability) = AbilityId::builtin_from_path(db, path) {
        entries.push(BuiltinSymbolEntry {
            span,
            name: ability.name(db),
            qualified: ability.qualified(db).clone(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexes_imported_builtin_io_effect() {
        let db = salsa::DatabaseImpl::default();
        let source = SourceCst::from_source_str(
            &db,
            "builtin_io.trb",
            "use std::io::Io\nfn main() ->{Io} Nil { Nil }",
        );

        let entries = builtin_symbols(&db, source);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].qualified, Symbol::new("std::io::Io"));
    }

    #[test]
    fn does_not_index_shadowing_source_ability() {
        let db = salsa::DatabaseImpl::default();
        let source = SourceCst::from_source_str(
            &db,
            "source_io.trb",
            "use std::io::Io\nability Io {}\nfn main() ->{Io} Nil { Nil }",
        );

        assert!(builtin_symbols(&db, source).is_empty());
    }
}
