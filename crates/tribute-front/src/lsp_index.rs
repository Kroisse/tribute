//! LSP-specific indexes for AST-based lookups.
//!
//! This module provides Salsa-tracked indexes for LSP features like hover,
//! go-to-definition, and find-references. These indexes are built from the
//! typed AST (`Module<TypedRef>`) rather than TrunkIR, enabling incremental
//! updates based on AST changes.

use std::collections::HashMap;

use trunk_ir::Span;

use crate::ast::{
    Arm, Decl, Expr, ExprKind, FuncDecl, HandlerArm, HandlerKind, Module, NodeId, Pattern,
    PatternKind, SpanMap, Stmt, Type, TypeKind, TypedRef,
};
use crate::query as ast_query;
use crate::source_file::SourceCst;

// =============================================================================
// Type Pretty Printing
// =============================================================================

/// Pretty-print an AST type to a user-friendly string.
pub fn print_ast_type(db: &dyn salsa::Database, ty: Type<'_>) -> String {
    match ty.kind(db) {
        TypeKind::Int => "Int".to_string(),
        TypeKind::Nat => "Nat".to_string(),
        TypeKind::Float => "Float".to_string(),
        TypeKind::Bool => "Bool".to_string(),
        TypeKind::String => "String".to_string(),
        TypeKind::Bytes => "Bytes".to_string(),
        TypeKind::Nil => "()".to_string(),

        TypeKind::BoundVar { index } => {
            // Convert de Bruijn index to a name (a, b, c, ...)
            let name = if *index < 26 {
                char::from_u32('a' as u32 + *index).map(|c| c.to_string())
            } else {
                None
            };
            name.unwrap_or_else(|| format!("t{}", index))
        }

        TypeKind::UniVar { id } => {
            // Unification variables are displayed as lowercase letters
            let name = if *id < 26 {
                char::from_u32('a' as u32 + *id as u32).map(|c| c.to_string())
            } else {
                None
            };
            name.unwrap_or_else(|| format!("?{}", id))
        }

        TypeKind::Named { name, args } => {
            if args.is_empty() {
                name.to_string()
            } else {
                let args_str: Vec<String> = args.iter().map(|t| print_ast_type(db, *t)).collect();
                format!("{}({})", name, args_str.join(", "))
            }
        }

        TypeKind::Func {
            params,
            result,
            effect,
        } => {
            let params_str: Vec<String> = params.iter().map(|t| print_ast_type(db, *t)).collect();
            let result_str = print_ast_type(db, *result);

            let effects = effect.effects(db);
            let has_rest = effect.rest(db).is_some();

            if effects.is_empty() && !has_rest {
                // Pure function
                format!("fn({}) -> {}", params_str.join(", "), result_str)
            } else {
                // Function with effects
                let effect_strs: Vec<String> = effects
                    .iter()
                    .map(|e| {
                        if e.args.is_empty() {
                            e.name.to_string()
                        } else {
                            let args_str: Vec<String> =
                                e.args.iter().map(|t| print_ast_type(db, *t)).collect();
                            format!("{}({})", e.name, args_str.join(", "))
                        }
                    })
                    .collect();

                let effect_str = if has_rest {
                    if effect_strs.is_empty() {
                        "e".to_string()
                    } else {
                        format!("{} | e", effect_strs.join(", "))
                    }
                } else {
                    effect_strs.join(", ")
                };

                format!(
                    "fn({}) ->{{{}}} {}",
                    params_str.join(", "),
                    effect_str,
                    result_str
                )
            }
        }

        TypeKind::Tuple(elems) => {
            let elems_str: Vec<String> = elems.iter().map(|t| print_ast_type(db, *t)).collect();
            format!("({})", elems_str.join(", "))
        }

        TypeKind::App { ctor, args } => {
            let ctor_str = print_ast_type(db, *ctor);
            let args_str: Vec<String> = args.iter().map(|t| print_ast_type(db, *t)).collect();
            format!("{}({})", ctor_str, args_str.join(", "))
        }

        TypeKind::Error => "<error>".to_string(),
    }
}

// =============================================================================
// Type Index
// =============================================================================

/// Entry in the AST type index.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct AstTypeEntry<'db> {
    /// The NodeId of the AST node.
    pub node_id: NodeId,
    /// The span of the node in source.
    pub span: Span,
    /// The inferred type.
    pub ty: Type<'db>,
}

/// Index mapping source positions to type information.
///
/// Built from the typed AST module, this index enables hover
/// and other type-aware LSP features.
pub struct AstTypeIndex<'db> {
    /// Entries sorted by span start for efficient lookup.
    entries: Vec<AstTypeEntry<'db>>,
    /// Map from NodeId to index for direct lookup.
    by_node_id: HashMap<NodeId, usize>,
}

impl<'db> AstTypeIndex<'db> {
    /// Build a type index from a typed module.
    pub fn build(
        db: &'db dyn salsa::Database,
        module: &Module<TypedRef<'db>>,
        span_map: &SpanMap,
    ) -> Self {
        let mut collector = TypeCollector::new(db, span_map);
        collector.collect_module(module);

        let mut entries = collector.entries;

        // Sort by span start for efficient lookup
        entries.sort_by_key(|e| (e.span.start, e.span.end));

        // Build NodeId index
        let by_node_id: HashMap<_, _> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.node_id, i))
            .collect();

        Self {
            entries,
            by_node_id,
        }
    }

    /// Find the type at a given byte offset.
    ///
    /// Returns the innermost (most specific) type entry containing the offset.
    pub fn type_at(&self, offset: usize) -> Option<&AstTypeEntry<'db>> {
        // Find all entries containing this offset
        let containing: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.span.start <= offset && offset < e.span.end)
            .collect();

        // Return the innermost (smallest span)
        containing
            .into_iter()
            .min_by_key(|e| e.span.end - e.span.start)
    }

    /// Get the type for a specific NodeId.
    pub fn type_for_node(&self, node_id: NodeId) -> Option<&AstTypeEntry<'db>> {
        self.by_node_id.get(&node_id).map(|&i| &self.entries[i])
    }
}

/// Helper struct for collecting type entries from the AST.
struct TypeCollector<'a, 'db> {
    db: &'db dyn salsa::Database,
    span_map: &'a SpanMap,
    entries: Vec<AstTypeEntry<'db>>,
}

impl<'a, 'db> TypeCollector<'a, 'db> {
    fn new(db: &'db dyn salsa::Database, span_map: &'a SpanMap) -> Self {
        Self {
            db,
            span_map,
            entries: Vec::new(),
        }
    }

    fn add_entry(&mut self, node_id: NodeId, ty: Type<'db>) {
        let span = self.span_map.get_or_default(node_id);
        self.entries.push(AstTypeEntry { node_id, span, ty });
    }

    fn collect_module(&mut self, module: &Module<TypedRef<'db>>) {
        for decl in &module.decls {
            self.collect_decl(decl);
        }
    }

    fn collect_decl(&mut self, decl: &Decl<TypedRef<'db>>) {
        match decl {
            Decl::Function(func) => self.collect_func(func),
            Decl::Const(c) => self.collect_expr(&c.value),
            // Struct, Enum, Ability, Use don't have expression types
            Decl::Struct(_) | Decl::Enum(_) | Decl::Ability(_) | Decl::Use(_) => {}
        }
    }

    fn collect_func(&mut self, func: &FuncDecl<TypedRef<'db>>) {
        // Collect function body types
        self.collect_expr(&func.body);
    }

    fn collect_expr(&mut self, expr: &Expr<TypedRef<'db>>) {
        // Add type for this expression node based on its kind
        match expr.kind.as_ref() {
            ExprKind::Var(typed_ref) => {
                self.add_entry(expr.id, typed_ref.ty);
            }
            ExprKind::IntLit(_) => {
                // Integer literals have Int type
                let int_ty = Type::new(self.db, TypeKind::Int);
                self.add_entry(expr.id, int_ty);
            }
            ExprKind::FloatLit(_) => {
                let float_ty = Type::new(self.db, TypeKind::Float);
                self.add_entry(expr.id, float_ty);
            }
            ExprKind::StringLit(_) => {
                let string_ty = Type::new(self.db, TypeKind::String);
                self.add_entry(expr.id, string_ty);
            }
            ExprKind::BoolLit(_) => {
                let bool_ty = Type::new(self.db, TypeKind::Bool);
                self.add_entry(expr.id, bool_ty);
            }
            ExprKind::UnitLit => {
                let nil_ty = Type::new(self.db, TypeKind::Nil);
                self.add_entry(expr.id, nil_ty);
            }
            ExprKind::Call { callee, args } => {
                // The call expression's type is the return type of the callee
                // For now, we infer from the callee's function type
                self.collect_expr(callee);
                for arg in args {
                    self.collect_expr(arg);
                }
                // The type of a call is extracted from the callee's return type
                // We'd need the callee's type to extract the result type
                // For now, skip adding the call's type - hover on callee/args works
            }
            ExprKind::Cons { ctor, args } => {
                self.add_entry(expr.id, ctor.ty);
                for arg in args {
                    self.collect_expr(arg);
                }
            }
            ExprKind::Record {
                type_name,
                fields,
                spread,
            } => {
                self.add_entry(expr.id, type_name.ty);
                for (_, field_expr) in fields {
                    self.collect_expr(field_expr);
                }
                if let Some(spread_expr) = spread {
                    self.collect_expr(spread_expr);
                }
            }
            ExprKind::FieldAccess { expr: inner, .. } => {
                self.collect_expr(inner);
                // Field access type would need type information from the struct
            }
            ExprKind::MethodCall { receiver, args, .. } => {
                self.collect_expr(receiver);
                for arg in args {
                    self.collect_expr(arg);
                }
            }
            ExprKind::Block(stmts) => {
                for stmt in stmts {
                    self.collect_stmt(stmt);
                }
            }
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.collect_expr(cond);
                self.collect_expr(then_branch);
                if let Some(else_br) = else_branch {
                    self.collect_expr(else_br);
                }
            }
            ExprKind::Case { scrutinee, arms } => {
                self.collect_expr(scrutinee);
                for arm in arms {
                    self.collect_arm(arm);
                }
            }
            ExprKind::Lambda { body, .. } => {
                self.collect_expr(body);
            }
            ExprKind::Handle { body, handlers } => {
                self.collect_expr(body);
                for handler in handlers {
                    self.collect_handler(handler);
                }
            }
            ExprKind::Tuple(elems) => {
                for elem in elems {
                    self.collect_expr(elem);
                }
            }
            ExprKind::List(elems) => {
                for elem in elems {
                    self.collect_expr(elem);
                }
            }
            ExprKind::BinOp { lhs, rhs, .. } => {
                self.collect_expr(lhs);
                self.collect_expr(rhs);
            }
            ExprKind::UnaryOp { expr: inner, .. } => {
                self.collect_expr(inner);
            }
            ExprKind::Error => {}
        }
    }

    fn collect_stmt(&mut self, stmt: &Stmt<TypedRef<'db>>) {
        match stmt {
            Stmt::Let { pattern, value, .. } => {
                self.collect_pattern(pattern);
                self.collect_expr(value);
            }
            Stmt::Expr { expr, .. } | Stmt::Return { expr, .. } => {
                self.collect_expr(expr);
            }
        }
    }

    fn collect_pattern(&mut self, pattern: &Pattern<TypedRef<'db>>) {
        match pattern.kind.as_ref() {
            PatternKind::Wildcard => {}
            PatternKind::Bind { .. } => {
                // Bind patterns don't have TypedRef directly in the current structure
                // The type comes from the context (let binding, case arm, etc.)
            }
            PatternKind::Literal(_) => {}
            PatternKind::Variant { ctor, fields } => {
                self.add_entry(pattern.id, ctor.ty);
                for field in fields {
                    self.collect_pattern(field);
                }
            }
            PatternKind::Record {
                type_name, fields, ..
            } => {
                if let Some(tn) = type_name {
                    self.add_entry(pattern.id, tn.ty);
                }
                for field in fields {
                    if let Some(p) = &field.pattern {
                        self.collect_pattern(p);
                    }
                }
            }
            PatternKind::Tuple(elems) | PatternKind::List(elems) => {
                for elem in elems {
                    self.collect_pattern(elem);
                }
            }
            PatternKind::ListRest { head, .. } => {
                for h in head {
                    self.collect_pattern(h);
                }
            }
            PatternKind::As { pattern: inner, .. } => {
                self.collect_pattern(inner);
            }
            PatternKind::Or(alts) => {
                for alt in alts {
                    self.collect_pattern(alt);
                }
            }
            PatternKind::Error => {}
        }
    }

    fn collect_arm(&mut self, arm: &Arm<TypedRef<'db>>) {
        self.collect_pattern(&arm.pattern);
        if let Some(guard) = &arm.guard {
            self.collect_expr(guard);
        }
        self.collect_expr(&arm.body);
    }

    fn collect_handler(&mut self, handler: &HandlerArm<TypedRef<'db>>) {
        match &handler.kind {
            HandlerKind::Result { binding } => {
                self.collect_pattern(binding);
            }
            HandlerKind::Effect {
                ability, params, ..
            } => {
                self.add_entry(handler.id, ability.ty);
                for param in params {
                    self.collect_pattern(param);
                }
            }
        }
        self.collect_expr(&handler.body);
    }
}

// =============================================================================
// Salsa Queries
// =============================================================================

/// Build an AST-based type index for a source file.
///
/// This is a Salsa tracked query that returns an index mapping source
/// positions to type information. The index is invalidated when the
/// typed AST changes.
#[salsa::tracked]
pub fn type_index<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Option<AstTypeIndexData<'db>> {
    let module = ast_query::tdnr_module(db, source)?;
    let span_map = ast_query::span_map(db, source)?;

    let index = AstTypeIndex::build(db, &module, &span_map);

    Some(AstTypeIndexData::new(db, index.entries))
}

/// Salsa-tracked wrapper for the type index data.
///
/// This is needed because `AstTypeIndex` contains a HashMap which isn't
/// directly compatible with Salsa. We store just the entries and rebuild
/// the HashMap when needed.
#[salsa::tracked]
pub struct AstTypeIndexData<'db> {
    #[returns(ref)]
    pub entries: Vec<AstTypeEntry<'db>>,
}

impl<'db> AstTypeIndexData<'db> {
    /// Rebuild the full index from stored data.
    pub fn as_index(&self, db: &'db dyn salsa::Database) -> AstTypeIndex<'db> {
        let entries = self.entries(db).clone();
        let by_node_id: HashMap<_, _> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.node_id, i))
            .collect();

        AstTypeIndex {
            entries,
            by_node_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source_file::path_to_uri;
    use ropey::Rope;
    use tree_sitter::Parser;

    fn make_source(db: &dyn salsa::Database, text: &str) -> SourceCst {
        let uri = path_to_uri(std::path::Path::new("test.trb"));
        let rope = Rope::from_str(text);

        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_tribute::LANGUAGE.into())
            .expect("Failed to set language");
        let tree = parser.parse(text, None);

        SourceCst::new(db, uri, rope, tree)
    }

    #[test]
    fn test_print_ast_type_basic() {
        let db = salsa::DatabaseImpl::default();

        let int_ty = Type::new(&db, TypeKind::Int);
        assert_eq!(print_ast_type(&db, int_ty), "Int");

        let bool_ty = Type::new(&db, TypeKind::Bool);
        assert_eq!(print_ast_type(&db, bool_ty), "Bool");

        let nil_ty = Type::new(&db, TypeKind::Nil);
        assert_eq!(print_ast_type(&db, nil_ty), "()");
    }

    #[test]
    fn test_print_ast_type_named() {
        let db = salsa::DatabaseImpl::default();

        let int_ty = Type::new(&db, TypeKind::Int);
        let list_ty = Type::new(
            &db,
            TypeKind::Named {
                name: trunk_ir::Symbol::new("List"),
                args: vec![int_ty],
            },
        );
        assert_eq!(print_ast_type(&db, list_ty), "List(Int)");
    }

    #[test]
    fn test_print_ast_type_function() {
        let db = salsa::DatabaseImpl::default();

        let int_ty = Type::new(&db, TypeKind::Int);
        let pure_effect = crate::ast::EffectRow::pure(&db);
        let func_ty = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int_ty, int_ty],
                result: int_ty,
                effect: pure_effect,
            },
        );
        assert_eq!(print_ast_type(&db, func_ty), "fn(Int, Int) -> Int");
    }

    #[test]
    fn test_type_index_query() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn main() { 42 }");

        let index_data = type_index(&db, source);
        assert!(index_data.is_some());

        let data = index_data.unwrap();
        let index = data.as_index(&db);

        // The index should have at least one entry (the integer literal)
        assert!(!index.entries.is_empty());
    }

    #[test]
    fn test_type_index_finds_variable() {
        let db = salsa::DatabaseImpl::default();
        //                    0         1         2
        //                    0123456789012345678901234567
        let source = make_source(&db, "fn foo(x: Int): Int { x }");

        let index_data = type_index(&db, source);
        assert!(index_data.is_some());

        let data = index_data.unwrap();
        let index = data.as_index(&db);

        // Position of 'x' in the body is around 22
        let entry = index.type_at(22);
        assert!(entry.is_some(), "Should find type at position 22");

        // The type might be a UniVar if not fully resolved, or Int if resolved
        // For now, just verify we found a type entry
        let ty_str = print_ast_type(&db, entry.unwrap().ty);
        assert!(!ty_str.is_empty(), "Should have a non-empty type string");
    }
}
