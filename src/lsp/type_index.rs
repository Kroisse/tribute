//! Type index for LSP hover support.
//!
//! Maps source positions to inferred types, enabling hover and other
//! type-aware LSP features.

use std::collections::BTreeMap;

use trunk_ir::Span;

use tribute_front::SourceCst;
use tribute_front::ast::{
    Arm, Decl, Expr, ExprKind, FuncDecl, HandlerArm, HandlerKind, Module, NodeId, Pattern,
    PatternKind, SpanMap, Stmt, Type, TypeKind, TypedRef, UniVarSource,
};
use tribute_front::query as ast_query;

// =============================================================================
// Type Pretty Printing
// =============================================================================

/// Pretty-print an AST type to a user-friendly string.
pub fn print_ast_type(db: &dyn salsa::Database, ty: Type<'_>) -> String {
    let kind = ty.kind(db);

    if let Some(name) = kind.primitive_name() {
        return name.to_string();
    }

    match kind {
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
            // Use the index for polymorphic sources, or the counter for anonymous/function-local sources
            let display_index = match id.source(db) {
                UniVarSource::FunctionLocal { index, .. } => index as u32 + id.index(db),
                UniVarSource::Anonymous(counter) => counter as u32 + id.index(db),
            };
            let name = if display_index < 26 {
                char::from_u32('a' as u32 + display_index).map(|c| c.to_string())
            } else {
                None
            };
            name.unwrap_or_else(|| format!("?{}", display_index))
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
                            e.ability_id.name(db).to_string()
                        } else {
                            let args_str: Vec<String> =
                                e.args.iter().map(|t| print_ast_type(db, *t)).collect();
                            format!("{}({})", e.ability_id.name(db), args_str.join(", "))
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

        // Primitives are handled by early return above
        _ => unreachable!(),
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
#[salsa::tracked]
pub struct AstTypeIndex<'db> {
    /// Entries sorted by span start for efficient lookup.
    #[returns(deref)]
    entries: Vec<AstTypeEntry<'db>>,
    /// Map from NodeId to index for direct lookup.
    #[returns(ref)]
    by_node_id: BTreeMap<NodeId, usize>,
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
        let by_node_id = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.node_id, i))
            .collect();

        Self::new(db, entries, by_node_id)
    }

    /// Find the type at a given byte offset.
    ///
    /// Returns the innermost (most specific) type entry containing the offset.
    pub fn type_at(
        &self,
        db: &'db dyn salsa::Database,
        offset: usize,
    ) -> Option<&AstTypeEntry<'db>> {
        // Find all entries containing this offset
        let containing: Vec<_> = self
            .entries(db)
            .iter()
            .filter(|e| e.span.start <= offset && offset < e.span.end)
            .collect();

        // Return the innermost (smallest span)
        containing
            .into_iter()
            .min_by_key(|e| e.span.end - e.span.start)
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
            // ExternFunction, Struct, Enum, Ability, Use don't have expression types
            Decl::ExternFunction(_)
            | Decl::Struct(_)
            | Decl::Enum(_)
            | Decl::Ability(_)
            | Decl::Use(_) => {}
            Decl::Module(m) => {
                // Recursively collect from nested declarations
                if let Some(body) = &m.body {
                    for inner_decl in body {
                        self.collect_decl(inner_decl);
                    }
                }
            }
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
            ExprKind::NatLit(_) => {
                // Natural literals have Nat type
                let nat_ty = Type::new(self.db, TypeKind::Nat);
                self.add_entry(expr.id, nat_ty);
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
            ExprKind::BytesLit(_) => {
                let bytes_ty = Type::new(self.db, TypeKind::Bytes);
                self.add_entry(expr.id, bytes_ty);
            }
            ExprKind::BoolLit(_) => {
                let bool_ty = Type::new(self.db, TypeKind::Bool);
                self.add_entry(expr.id, bool_ty);
            }
            ExprKind::Nil => {
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
            // (Field access is now MethodCall)
            ExprKind::MethodCall { receiver, args, .. } => {
                self.collect_expr(receiver);
                for arg in args {
                    self.collect_expr(arg);
                }
            }
            ExprKind::Block { stmts, value } => {
                for stmt in stmts {
                    self.collect_stmt(stmt);
                }
                self.collect_expr(value);
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
            ExprKind::RuneLit(_) => {
                // Rune literals have Rune type (Unicode code point)
                let rune_ty = Type::new(self.db, TypeKind::Rune);
                self.add_entry(expr.id, rune_ty);
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
            Stmt::Expr { expr, .. } => {
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
) -> Option<AstTypeIndex<'db>> {
    let module = ast_query::tdnr_module(db, source)?;
    let span_map = ast_query::span_map(db, source)?;

    Some(AstTypeIndex::build(db, &module, &span_map))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ropey::Rope;
    use tree_sitter::Parser;
    use tribute_front::ast::UniVarId;
    use tribute_front::path_to_uri;

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
        assert_eq!(print_ast_type(&db, nil_ty), "Nil");
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
        use tribute_front::ast::EffectRow;

        let db = salsa::DatabaseImpl::default();

        let int_ty = Type::new(&db, TypeKind::Int);
        let pure_effect = EffectRow::pure(&db);
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

        let index = type_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();

        // The index should have at least one entry (the integer literal)
        assert!(!index.entries(&db).is_empty());
    }

    #[test]
    fn test_type_index_finds_variable() {
        let db = salsa::DatabaseImpl::default();
        //                    0         1         2
        //                    0123456789012345678901234567
        let source = make_source(&db, "fn foo(x: Int): Int { x }");

        let index = type_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();

        // Position of 'x' in the body is around 22
        let entry = index.type_at(&db, 22);
        assert!(entry.is_some(), "Should find type at position 22");

        // The type might be a UniVar if not fully resolved, or Int if resolved
        // For now, just verify we found a type entry
        let ty_str = print_ast_type(&db, entry.unwrap().ty);
        assert!(!ty_str.is_empty(), "Should have a non-empty type string");
    }

    // Additional print_ast_type tests

    #[test]
    fn test_print_ast_type_bound_var() {
        let db = salsa::DatabaseImpl::default();

        let ty = Type::new(&db, TypeKind::BoundVar { index: 0 });
        assert_eq!(print_ast_type(&db, ty), "a");

        let ty = Type::new(&db, TypeKind::BoundVar { index: 1 });
        assert_eq!(print_ast_type(&db, ty), "b");

        let ty = Type::new(&db, TypeKind::BoundVar { index: 25 });
        assert_eq!(print_ast_type(&db, ty), "z");

        // Large index should fallback to t{index}
        let ty = Type::new(&db, TypeKind::BoundVar { index: 26 });
        assert_eq!(print_ast_type(&db, ty), "t26");
    }

    #[test]
    fn test_print_ast_type_uni_var() {
        let db = salsa::DatabaseImpl::default();

        // Helper to create UniVarId with Anonymous source
        let make_uni_var = |counter: u64| {
            let source = UniVarSource::Anonymous(counter);
            UniVarId::new(&db, source, 0)
        };

        let ty = Type::new(
            &db,
            TypeKind::UniVar {
                id: make_uni_var(0),
            },
        );
        assert_eq!(print_ast_type(&db, ty), "a");

        let ty = Type::new(
            &db,
            TypeKind::UniVar {
                id: make_uni_var(25),
            },
        );
        assert_eq!(print_ast_type(&db, ty), "z");

        // Large id should fallback to ?{id}
        let ty = Type::new(
            &db,
            TypeKind::UniVar {
                id: make_uni_var(100),
            },
        );
        assert_eq!(print_ast_type(&db, ty), "?100");
    }

    #[test]
    fn test_print_ast_type_tuple() {
        let db = salsa::DatabaseImpl::default();

        let int_ty = Type::new(&db, TypeKind::Int);
        let bool_ty = Type::new(&db, TypeKind::Bool);
        let tuple_ty = Type::new(&db, TypeKind::Tuple(vec![int_ty, bool_ty]));
        assert_eq!(print_ast_type(&db, tuple_ty), "(Int, Bool)");
    }

    #[test]
    fn test_print_ast_type_app() {
        let db = salsa::DatabaseImpl::default();

        let int_ty = Type::new(&db, TypeKind::Int);
        let ctor_ty = Type::new(
            &db,
            TypeKind::Named {
                name: trunk_ir::Symbol::new("List"),
                args: vec![],
            },
        );
        let app_ty = Type::new(
            &db,
            TypeKind::App {
                ctor: ctor_ty,
                args: vec![int_ty],
            },
        );
        assert_eq!(print_ast_type(&db, app_ty), "List(Int)");
    }

    #[test]
    fn test_print_ast_type_error() {
        let db = salsa::DatabaseImpl::default();

        let ty = Type::new(&db, TypeKind::Error);
        assert_eq!(print_ast_type(&db, ty), "<error>");
    }

    #[test]
    fn test_print_ast_type_function_with_effects() {
        use tribute_front::ast::{AbilityId, Effect, EffectRow};
        use trunk_ir::SymbolVec;

        let db = salsa::DatabaseImpl::default();

        let int_ty = Type::new(&db, TypeKind::Int);
        let io_id = AbilityId::new(&db, SymbolVec::new(), trunk_ir::Symbol::new("IO"));
        let effect = Effect {
            ability_id: io_id,
            args: vec![],
        };
        let effect_row = EffectRow::new(&db, vec![effect], None);
        let func_ty = Type::new(
            &db,
            TypeKind::Func {
                params: vec![int_ty],
                result: int_ty,
                effect: effect_row,
            },
        );
        assert_eq!(print_ast_type(&db, func_ty), "fn(Int) ->{IO} Int");
    }

    // Type Collection Tests

    #[test]
    fn test_type_index_let_binding() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"fn main() -> Int {
    let x: Int = 42
    x
}"#,
        );

        let index = type_index(&db, source);
        assert!(index.is_some());
    }

    #[test]
    fn test_type_index_case_expression() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"enum Option { Some(Int), None }

fn test(opt: Option) -> Int {
    case opt {
        Some(v) -> v
        None -> 0
    }
}"#,
        );

        let index = type_index(&db, source);
        assert!(index.is_some());
    }

    #[test]
    fn test_type_index_function_params() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"fn add(a: Int, b: Int) -> Int {
    a + b
}"#,
        );

        let index = type_index(&db, source);
        assert!(index.is_some());
    }

    #[test]
    fn test_type_index_tuple_pattern() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"fn main() -> Int {
    let pair: #(Int, Int) = #(1, 2)
    let #(a, b) = pair
    a + b
}"#,
        );

        let index = type_index(&db, source);
        assert!(index.is_some());
    }

    // Rune Literal Type Tests

    #[test]
    fn test_rune_type_printed() {
        let db = salsa::DatabaseImpl::default();
        let rune_ty = Type::new(&db, TypeKind::Rune);
        assert_eq!(print_ast_type(&db, rune_ty), "Rune");
    }

    #[test]
    fn test_rune_literal_expression() {
        let db = salsa::DatabaseImpl::default();
        // Rune literal ?a in the source
        let source = make_source(&db, "fn main() -> Rune { ?a }");

        let index = type_index(&db, source);
        assert!(
            index.is_some(),
            "Type index should be available for valid source"
        );
    }
}
