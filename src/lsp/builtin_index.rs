//! Source-position index for compiler-owned bindings without declarations.

use tribute_front::SourceCst;
use tribute_front::ast::{
    AbilityId, Decl, Module, SpanMap, TypeAnnotation, TypeAnnotationKind, TypedRef,
};
use tribute_front::query as ast_query;
use trunk_ir::{Span, Symbol};

/// A compiler-owned symbol referenced from source.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct BuiltinSymbolEntry {
    pub span: Span,
    pub name: Symbol,
    pub qualified: Symbol,
}

/// Compiler-owned symbol references in a source file.
#[salsa::tracked(returns(deref))]
pub fn builtin_symbols(db: &dyn salsa::Database, source: SourceCst) -> Vec<BuiltinSymbolEntry> {
    let Some(module) = ast_query::tdnr_module(db, source) else {
        return Vec::new();
    };
    let Some(span_map) = ast_query::span_map(db, source) else {
        return Vec::new();
    };

    let mut entries = Vec::new();
    collect_module(db, &module, &span_map, &mut entries);
    entries.sort_by_key(|entry| (entry.span.start, entry.span.end));
    entries
}

pub fn builtin_at(entries: &[BuiltinSymbolEntry], offset: usize) -> Option<&BuiltinSymbolEntry> {
    entries
        .iter()
        .filter(|entry| entry.span.start <= offset && offset < entry.span.end)
        .min_by_key(|entry| entry.span.end - entry.span.start)
}

fn collect_module(
    db: &dyn salsa::Database,
    module: &Module<TypedRef<'_>>,
    span_map: &SpanMap,
    entries: &mut Vec<BuiltinSymbolEntry>,
) {
    for decl in &module.decls {
        match decl {
            Decl::Function(func) => {
                for effect in func.effects.iter().flatten() {
                    collect_effect(db, effect, span_map, entries);
                }
            }
            Decl::Module(module) => {
                if let Some(body) = &module.body {
                    let nested = Module::new(module.id, Some(module.name), body.clone());
                    collect_module(db, &nested, span_map, entries);
                }
            }
            Decl::ExternFunction(_)
            | Decl::Struct(_)
            | Decl::Enum(_)
            | Decl::Ability(_)
            | Decl::Use(_) => {}
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

    if let Some(ability) = AbilityId::builtin_from_path(db, path) {
        entries.push(BuiltinSymbolEntry {
            span: span_map.get_or_default(ability_annotation.id),
            name: ability.name(db),
            qualified: ability.qualified(db),
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
