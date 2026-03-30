//! TDNR resolver implementation.
//!
//! Transforms `MethodCall` expressions into `Call` expressions by resolving
//! the method name using the receiver's type.

use std::collections::HashMap;

use trunk_ir::Symbol;

use crate::ast::{
    Arm, Decl, Expr, ExprKind, FuncDecl, FuncDefId, HandlerArm, HandlerKind, Module, Pattern,
    ResolvedRef, Stmt, Type, TypeAnnotation, TypeAnnotationKind, TypeKind, TypedRef,
};
use crate::typeck::{MethodEntry, receiver_type_matches};
use crate::{push_prefix, qualified_symbol};

/// TDNR resolver for AST expressions.
///
/// Resolves method calls by finding functions where the first parameter
/// type matches the receiver's type.
pub struct TdnrResolver<'db> {
    db: &'db dyn salsa::Database,
    /// Map from method_name to candidate entries.
    /// Keyed by the method name only; receiver type filtering happens at lookup time.
    /// Multiple candidates with different receiver types can share the same method name
    /// (e.g., `String::len` and `Bytes::len` both register under `len`).
    method_index: HashMap<Symbol, Vec<MethodEntry<'db>>>,
}

impl<'db> TdnrResolver<'db> {
    /// Create a new TDNR resolver.
    pub fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            db,
            method_index: HashMap::new(),
        }
    }

    /// Resolve method calls in a module.
    pub fn resolve_module(mut self, module: Module<TypedRef<'db>>) -> Module<TypedRef<'db>> {
        // Phase 1: Build method index from function declarations
        self.build_method_index(&module);

        // Phase 2: Resolve method calls in all declarations
        let decls = module
            .decls
            .into_iter()
            .map(|decl| self.resolve_decl(decl))
            .collect();

        Module {
            id: module.id,
            name: module.name,
            decls,
        }
    }

    // =========================================================================
    // Method index building
    // =========================================================================

    /// Build an index of methods by method name.
    ///
    /// This indexes functions by their method name only; the receiver type is
    /// used for filtering at lookup time, not as part of the index key.
    fn build_method_index(&mut self, module: &Module<TypedRef<'db>>) {
        // Start with empty prefix — matches resolve::build_env convention.
        // The top-level module name (derived from filename) is not part of
        // internal qualified names. Only nested `pub mod` blocks extend the path.
        let mut prefix = String::new();
        self.index_decls(&module.decls, &mut prefix);
    }

    /// Pre-populate the method index from an external module (e.g., prelude).
    ///
    /// This allows TDNR to resolve UFCS calls to methods defined in imported modules.
    /// Must be called before `resolve_module` so that external methods are available
    /// when indexing the target module.
    ///
    /// External module functions are indexed with their original qualified names
    /// (no additional prefix), so FuncDefIds match what the rest of the pipeline expects.
    pub fn index_external_module(&mut self, module: &Module<TypedRef<'db>>) {
        let mut prefix = String::new();
        self.index_decls(&module.decls, &mut prefix);
    }

    /// Recursively index declarations, including nested modules.
    fn index_decls(&mut self, decls: &[Decl<TypedRef<'db>>], prefix: &mut String) {
        for decl in decls {
            match decl {
                Decl::Function(func) => {
                    // Check if this function can be a method (has at least one parameter
                    // with a determinable receiver type). Functions with bare type variables
                    // as their first parameter cannot be UFCS targets.
                    if func.params.is_empty() {
                        continue;
                    }

                    // Skip if receiver type is not determinable (e.g., bare type variable).
                    // This also prevents build_func_type from hitting open effect rows.
                    let first_param = &func.params[0];
                    if self
                        .extract_receiver_type_from_annotation(&first_param.ty)
                        .is_none()
                    {
                        continue;
                    }

                    let func_name = func.name;

                    // Create FuncDefId with qualified name
                    let qualified = qualified_symbol(prefix, func_name);
                    let func_id = FuncDefId::new(self.db, qualified);

                    // Build function type from parameter and return type annotations
                    let func_ty = self.build_func_type(func);

                    // Register under method name only; receiver type filtering at lookup time
                    self.method_index
                        .entry(func_name)
                        .or_default()
                        .push(MethodEntry { func_id, func_ty });
                }
                Decl::Struct(s) => {
                    // Register each field as an accessor method
                    // e.g., struct Point { x: Int, y: Int } registers:
                    //   - x → fn x(self: Point) -> Int  (receiver type filtering at lookup)
                    //   - y → fn y(self: Point) -> Int

                    let struct_name = s.name;

                    // Push struct name to build qualified field accessor names
                    let saved = push_prefix(prefix, struct_name);

                    for field in &s.fields {
                        let Some(field_name) = field.name else {
                            continue; // Skip unnamed fields
                        };

                        // Create synthetic FuncDefId for the accessor
                        let field_qualified = qualified_symbol(prefix, field_name);
                        let func_id = FuncDefId::new(self.db, field_qualified);

                        // Build accessor function type: fn(self: StructType) -> FieldType
                        // Note: Reusing struct's NodeId for synthetic type annotation.
                        // This is safe for type construction but won't be used for span lookups.
                        let self_annotation = if s.type_params.is_empty() {
                            TypeAnnotation {
                                id: s.id,
                                kind: TypeAnnotationKind::Named(struct_name),
                            }
                        } else {
                            // Include type parameters: StructName(a, b, ...)
                            let ctor = TypeAnnotation {
                                id: s.id,
                                kind: TypeAnnotationKind::Named(struct_name),
                            };
                            let args: Vec<TypeAnnotation> = s
                                .type_params
                                .iter()
                                .map(|tp| TypeAnnotation {
                                    id: tp.id,
                                    kind: TypeAnnotationKind::Named(tp.name),
                                })
                                .collect();
                            TypeAnnotation {
                                id: s.id,
                                kind: TypeAnnotationKind::App {
                                    ctor: Box::new(ctor),
                                    args,
                                },
                            }
                        };
                        let self_ty = self.annotation_to_type(&Some(self_annotation));
                        let field_ty = self.annotation_to_type(&Some(field.ty.clone()));

                        let effect = crate::ast::EffectRow::pure(self.db);
                        let func_ty = Type::new(
                            self.db,
                            TypeKind::Func {
                                params: vec![self_ty],
                                result: field_ty,
                                effect,
                            },
                        );

                        // Register under field name only
                        self.method_index
                            .entry(field_name)
                            .or_default()
                            .push(MethodEntry { func_id, func_ty });
                    }

                    prefix.truncate(saved);
                }
                Decl::Module(m) => {
                    if let Some(body) = &m.body {
                        // Build nested module path by appending current module name
                        let saved = push_prefix(prefix, m.name);
                        self.index_decls(body, prefix);
                        prefix.truncate(saved);
                    }
                }
                _ => {}
            }
        }
    }

    /// Build a function type from parameter and return type annotations.
    fn build_func_type(&self, func: &FuncDecl<TypedRef<'db>>) -> Type<'db> {
        use crate::ast::TypeKind;

        // Convert parameter types from annotations
        let params: Vec<Type<'db>> = func
            .params
            .iter()
            .map(|p| self.annotation_to_type(&p.ty))
            .collect();

        // Get return type from annotation or infer from body
        let result = func
            .return_ty
            .as_ref()
            .map(|ann| self.annotation_to_type(&Some(ann.clone())))
            .unwrap_or_else(|| {
                // Try to get return type from body expression
                self.get_expr_type(&func.body)
                    .unwrap_or_else(|| Type::new(self.db, TypeKind::Nil))
            });

        // Convert effect annotations to EffectRow
        let effect = self.annotations_to_effect_row(&func.effects);

        Type::new(
            self.db,
            TypeKind::Func {
                params,
                result,
                effect,
            },
        )
    }

    /// Convert effect annotations to an EffectRow.
    fn annotations_to_effect_row(
        &self,
        annotations: &Option<Vec<crate::ast::TypeAnnotation>>,
    ) -> crate::ast::EffectRow<'db> {
        use crate::ast::EffectRow;

        let Some(anns) = annotations else {
            return EffectRow::pure(self.db);
        };

        if anns.is_empty() {
            return EffectRow::pure(self.db);
        }

        // TDNR builds closed rows only (no fresh row variables).
        // Lowercase names and Infer annotations are filtered out by the shared helper.
        // We use an empty prefix since TDNR works after resolution phase.
        crate::ast::abilities_to_effect_row(
            self.db,
            anns,
            "",
            &mut |ann| self.annotation_to_type(&Some(ann.clone())),
            || unreachable!("TDNR does not support open effect rows"),
        )
    }

    /// Convert a type annotation to a Type.
    fn annotation_to_type(&self, annotation: &Option<crate::ast::TypeAnnotation>) -> Type<'db> {
        use crate::ast::{TypeAnnotationKind, TypeKind};

        let Some(ann) = annotation else {
            // No annotation - use a fresh type variable placeholder
            return Type::new(self.db, TypeKind::Error);
        };

        match &ann.kind {
            TypeAnnotationKind::Named(name) => {
                if let Some(kind) = name.with_str(TypeKind::from_primitive_name) {
                    Type::new(self.db, kind)
                } else {
                    // User-defined type
                    Type::new(
                        self.db,
                        TypeKind::Named {
                            name: *name,
                            args: vec![],
                        },
                    )
                }
            }
            TypeAnnotationKind::Path(path) if !path.is_empty() => {
                let name = path.last().copied().unwrap();
                Type::new(self.db, TypeKind::Named { name, args: vec![] })
            }
            TypeAnnotationKind::App { ctor, args } => {
                let ctor_ty = self.annotation_to_type(&Some((**ctor).clone()));
                let arg_tys: Vec<Type<'db>> = args
                    .iter()
                    .map(|a| self.annotation_to_type(&Some(a.clone())))
                    .collect();
                Type::new(
                    self.db,
                    TypeKind::App {
                        ctor: ctor_ty,
                        args: arg_tys,
                    },
                )
            }
            _ => Type::new(self.db, TypeKind::Error),
        }
    }

    /// Check if a type annotation has a determinable type constructor name.
    ///
    /// Returns `Some(name)` for Named, Path, and App annotations.
    /// Returns `None` for bare type variables and other non-determinable annotations.
    /// Used as a guard before calling `build_func_type` to avoid open effect row panics.
    fn extract_receiver_type_from_annotation(
        &self,
        annotation: &Option<crate::ast::TypeAnnotation>,
    ) -> Option<Symbol> {
        use crate::ast::TypeAnnotationKind;
        let ann = annotation.as_ref()?;
        match &ann.kind {
            TypeAnnotationKind::Named(name) => Some(*name),
            TypeAnnotationKind::Path(path) if !path.is_empty() => path.last().copied(),
            TypeAnnotationKind::App { ctor, .. } => {
                self.extract_receiver_type_from_annotation(&Some((**ctor).clone()))
            }
            _ => None,
        }
    }

    // =========================================================================
    // Declaration resolution
    // =========================================================================

    /// Resolve method calls in a declaration.
    fn resolve_decl(&mut self, decl: Decl<TypedRef<'db>>) -> Decl<TypedRef<'db>> {
        match decl {
            Decl::Function(func) => Decl::Function(self.resolve_func_decl(func)),
            // Other declarations don't contain expressions
            Decl::ExternFunction(e) => Decl::ExternFunction(e),
            Decl::Struct(s) => Decl::Struct(s),
            Decl::Enum(e) => Decl::Enum(e),
            Decl::Ability(a) => Decl::Ability(a),
            Decl::Use(u) => Decl::Use(u),
            Decl::Module(m) => Decl::Module(self.resolve_module_decl(m)),
        }
    }

    /// Resolve method calls in a module declaration.
    fn resolve_module_decl(
        &mut self,
        module: crate::ast::ModuleDecl<TypedRef<'db>>,
    ) -> crate::ast::ModuleDecl<TypedRef<'db>> {
        let body = module
            .body
            .map(|decls| decls.into_iter().map(|d| self.resolve_decl(d)).collect());

        crate::ast::ModuleDecl {
            id: module.id,
            name: module.name,
            is_pub: module.is_pub,
            body,
        }
    }

    /// Resolve method calls in a function declaration.
    fn resolve_func_decl(&mut self, func: FuncDecl<TypedRef<'db>>) -> FuncDecl<TypedRef<'db>> {
        FuncDecl {
            id: func.id,
            is_pub: func.is_pub,
            name: func.name,
            type_params: func.type_params,
            params: func.params,
            return_ty: func.return_ty,
            effects: func.effects,
            body: self.resolve_expr(func.body),
        }
    }

    // =========================================================================
    // Expression resolution
    // =========================================================================

    /// Resolve method calls in an expression.
    fn resolve_expr(&mut self, expr: Expr<TypedRef<'db>>) -> Expr<TypedRef<'db>> {
        let kind = match *expr.kind {
            // The main case: resolve method calls
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => self.resolve_method_call(expr.id, receiver, method, args),

            // Recursively process other expressions
            ExprKind::NatLit(n) => ExprKind::NatLit(n),
            ExprKind::IntLit(n) => ExprKind::IntLit(n),
            ExprKind::FloatLit(f) => ExprKind::FloatLit(f),
            ExprKind::BoolLit(b) => ExprKind::BoolLit(b),
            ExprKind::StringLit(s) => ExprKind::StringLit(s),
            ExprKind::BytesLit(b) => ExprKind::BytesLit(b),
            ExprKind::Nil => ExprKind::Nil,
            ExprKind::RuneLit(c) => ExprKind::RuneLit(c),
            ExprKind::Var(v) => ExprKind::Var(v),

            ExprKind::Call { callee, args } => ExprKind::Call {
                callee: self.resolve_expr(callee),
                args: args.into_iter().map(|a| self.resolve_expr(a)).collect(),
            },

            ExprKind::Cons { ctor, args } => ExprKind::Cons {
                ctor,
                args: args.into_iter().map(|a| self.resolve_expr(a)).collect(),
            },

            ExprKind::Record {
                type_name,
                fields,
                spread,
            } => ExprKind::Record {
                type_name,
                fields: fields
                    .into_iter()
                    .map(|(name, e)| (name, self.resolve_expr(e)))
                    .collect(),
                spread: spread.map(|e| self.resolve_expr(e)),
            },

            ExprKind::BinOp { op, lhs, rhs } => ExprKind::BinOp {
                op,
                lhs: self.resolve_expr(lhs),
                rhs: self.resolve_expr(rhs),
            },

            ExprKind::Block { stmts, value } => ExprKind::Block {
                stmts: stmts.into_iter().map(|s| self.resolve_stmt(s)).collect(),
                value: self.resolve_expr(value),
            },

            ExprKind::Case { scrutinee, arms } => ExprKind::Case {
                scrutinee: self.resolve_expr(scrutinee),
                arms: arms.into_iter().map(|a| self.resolve_arm(a)).collect(),
            },

            ExprKind::Lambda { params, body } => ExprKind::Lambda {
                params,
                body: self.resolve_expr(body),
            },

            ExprKind::Handle { body, handlers } => ExprKind::Handle {
                body: self.resolve_expr(body),
                handlers: handlers
                    .into_iter()
                    .map(|h| self.resolve_handler_arm(h))
                    .collect(),
            },

            ExprKind::Resume { arg, local_id } => ExprKind::Resume {
                arg: self.resolve_expr(arg),
                local_id,
            },

            ExprKind::Tuple(elements) => {
                ExprKind::Tuple(elements.into_iter().map(|e| self.resolve_expr(e)).collect())
            }

            ExprKind::List(elements) => {
                ExprKind::List(elements.into_iter().map(|e| self.resolve_expr(e)).collect())
            }

            ExprKind::Error => ExprKind::Error,
        };

        Expr::new(expr.id, kind)
    }

    /// Resolve a method call expression.
    ///
    /// Transforms `receiver.method(args)` into `method(receiver, args)`
    /// by looking up the method in the type's namespace.
    fn resolve_method_call(
        &mut self,
        id: crate::ast::NodeId,
        receiver: Expr<TypedRef<'db>>,
        method: Symbol,
        args: Vec<Expr<TypedRef<'db>>>,
    ) -> ExprKind<TypedRef<'db>> {
        let receiver = self.resolve_expr(receiver);
        let args: Vec<Expr<TypedRef<'db>>> =
            args.into_iter().map(|a| self.resolve_expr(a)).collect();

        // Get the receiver's type from the TypedRef
        let receiver_ty = self.get_expr_type(&receiver);

        // Try to find a matching method
        if let Some(entry) = self.lookup_method(receiver_ty, method) {
            // Found the method - transform to a Call
            let callee_ref = TypedRef {
                resolved: ResolvedRef::Function { id: entry.func_id },
                ty: entry.func_ty,
            };

            // Build callee expression (a Var referencing the function)
            // TODO: This reuses the MethodCall's NodeId, which makes SpanMap
            // point to the whole `receiver.method(args)` expression instead of
            // just the method name. A fresh/synthetic NodeId would be better
            // once the resolver has access to a NodeId generator.
            let callee = Expr::new(id, ExprKind::Var(callee_ref));

            // Prepend receiver to args
            let mut all_args = vec![receiver];
            all_args.extend(args);

            ExprKind::Call {
                callee,
                args: all_args,
            }
        } else {
            // Could not resolve - keep as MethodCall
            // This may be an error, but we leave it for later passes to report
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            }
        }
    }

    /// Get the type of an expression.
    ///
    /// Currently this handles expressions that directly contain TypedRef (Var, Cons).
    /// For other expressions, a type map would be needed to look up inferred types.
    fn get_expr_type(&self, expr: &Expr<TypedRef<'db>>) -> Option<Type<'db>> {
        match &*expr.kind {
            // For Var expressions, we can get the type from the TypedRef
            ExprKind::Var(typed_ref) => Some(typed_ref.ty),

            // Constructor expressions: extract the constructed value type
            // (the return type of the constructor function), not the
            // constructor function type itself.
            ExprKind::Cons { ctor, .. } => {
                if let crate::ast::TypeKind::Func { result, .. } = ctor.ty.kind(self.db) {
                    Some(*result)
                } else {
                    Some(ctor.ty)
                }
            }

            // For Call expressions, the return type of the callee would be needed
            // This requires looking at the callee's function type
            ExprKind::Call { callee, .. } => {
                // Try to get the return type from the callee's type
                if let Some(callee_ty) = self.get_expr_type(callee)
                    && let crate::ast::TypeKind::Func { result, .. } = callee_ty.kind(self.db)
                {
                    return Some(*result);
                }
                None
            }

            // Literals have known types
            ExprKind::NatLit(_) => Some(Type::new(self.db, crate::ast::TypeKind::Nat)),
            ExprKind::IntLit(_) => Some(Type::new(self.db, crate::ast::TypeKind::Int)),
            ExprKind::FloatLit(_) => Some(Type::new(self.db, crate::ast::TypeKind::Float)),
            ExprKind::BoolLit(_) => Some(Type::new(self.db, crate::ast::TypeKind::Bool)),
            ExprKind::StringLit(_) => Some(Type::new(self.db, crate::ast::TypeKind::string())),
            ExprKind::BytesLit(_) => Some(Type::new(self.db, crate::ast::TypeKind::Bytes)),
            ExprKind::RuneLit(_) => Some(Type::new(self.db, crate::ast::TypeKind::Rune)),
            ExprKind::Nil => Some(Type::new(self.db, crate::ast::TypeKind::Nil)),

            // Binary operations: only boolean ops remain
            ExprKind::BinOp { op, .. } => {
                use crate::ast::BinOpKind;
                match op {
                    BinOpKind::And | BinOpKind::Or => {
                        Some(Type::new(self.db, crate::ast::TypeKind::Bool))
                    }
                }
            }

            // Tuple: the type is a tuple of element types
            ExprKind::Tuple(elements) => {
                let elem_types: Vec<_> = elements
                    .iter()
                    .filter_map(|e| self.get_expr_type(e))
                    .collect();
                if elem_types.len() == elements.len() {
                    Some(Type::new(self.db, crate::ast::TypeKind::Tuple(elem_types)))
                } else {
                    None
                }
            }

            // Block: type is the type of the value expression
            ExprKind::Block { value, .. } => self.get_expr_type(value),

            // Record: get the return type from the constructor's type
            ExprKind::Record { type_name, .. } => {
                // type_name.ty is the constructor function type: fn(fields...) -> StructType
                if let crate::ast::TypeKind::Func { result, .. } = type_name.ty.kind(self.db) {
                    Some(*result)
                } else {
                    // If not a function type, use it directly
                    Some(type_name.ty)
                }
            }

            // Lambda: we can't easily determine the type without full analysis
            // For other expressions, we'd need a type map from typechecking
            _ => None,
        }
    }

    /// Look up a method for a given receiver type.
    ///
    /// Returns `None` when no candidates exist **or** when multiple candidates
    /// are found (ambiguous). In the ambiguous case the `MethodCall` is kept
    /// so that later passes can report an unresolved-method diagnostic.
    fn lookup_method(
        &self,
        receiver_ty: Option<Type<'db>>,
        method: Symbol,
    ) -> Option<&MethodEntry<'db>> {
        let receiver_ty = receiver_ty?;
        let candidates = self.method_index.get(&method)?;

        let mut iter = candidates
            .iter()
            .filter(|entry| receiver_type_matches(self.db, entry, receiver_ty));
        let matched = iter.next()?;
        if iter.next().is_some() {
            return None; // ambiguous — keep as MethodCall for error reporting
        }
        Some(matched)
    }

    // =========================================================================
    // Statement resolution
    // =========================================================================

    /// Resolve method calls in a statement.
    fn resolve_stmt(&mut self, stmt: Stmt<TypedRef<'db>>) -> Stmt<TypedRef<'db>> {
        match stmt {
            Stmt::Let {
                id,
                pattern,
                ty,
                value,
            } => Stmt::Let {
                id,
                pattern: self.resolve_pattern(pattern),
                ty,
                value: self.resolve_expr(value),
            },
            Stmt::Expr { id, expr } => Stmt::Expr {
                id,
                expr: self.resolve_expr(expr),
            },
        }
    }

    // =========================================================================
    // Pattern resolution (patterns don't have method calls, but we traverse them)
    // =========================================================================

    /// Resolve method calls in a pattern (patterns themselves don't have method calls,
    /// but we need this for completeness).
    fn resolve_pattern(&self, pattern: Pattern<TypedRef<'db>>) -> Pattern<TypedRef<'db>> {
        // Patterns don't contain method calls, so just return as-is
        // But we could have guards that need resolution
        pattern
    }

    // =========================================================================
    // Arm resolution
    // =========================================================================

    /// Resolve method calls in a case arm.
    fn resolve_arm(&mut self, arm: Arm<TypedRef<'db>>) -> Arm<TypedRef<'db>> {
        Arm {
            id: arm.id,
            pattern: self.resolve_pattern(arm.pattern),
            guard: arm.guard.map(|g| self.resolve_expr(g)),
            body: self.resolve_expr(arm.body),
        }
    }

    /// Resolve method calls in a handler arm.
    fn resolve_handler_arm(&mut self, arm: HandlerArm<TypedRef<'db>>) -> HandlerArm<TypedRef<'db>> {
        let kind = match arm.kind {
            HandlerKind::Do { binding } => HandlerKind::Do {
                binding: self.resolve_pattern(binding),
            },
            HandlerKind::Fn {
                ability,
                op,
                params,
            } => HandlerKind::Fn {
                ability,
                op,
                params: params
                    .into_iter()
                    .map(|p| self.resolve_pattern(p))
                    .collect(),
            },
            HandlerKind::Op {
                ability,
                op,
                params,
                resume_local_id,
            } => HandlerKind::Op {
                ability,
                op,
                params: params
                    .into_iter()
                    .map(|p| self.resolve_pattern(p))
                    .collect(),
                resume_local_id,
            },
        };
        HandlerArm {
            id: arm.id,
            kind,
            body: self.resolve_expr(arm.body),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinOpKind, CtorId, EffectRow, NodeId, ResolvedRef, TypeKind, TypedRef};
    use crate::typeck::extract_type_name_from_type;
    use salsa_test_macros::salsa_test;

    fn test_db() -> salsa::DatabaseImpl {
        salsa::DatabaseImpl::new()
    }

    fn fresh_node_id() -> NodeId {
        NodeId::from_raw(1)
    }

    #[test]
    fn test_tdnr_resolver_creation() {
        let db = test_db();
        let _resolver = TdnrResolver::new(&db);
    }

    #[test]
    fn test_get_expr_type_int_literal() {
        let db = test_db();
        let resolver = TdnrResolver::new(&db);

        let expr = Expr::new(fresh_node_id(), ExprKind::IntLit(42));
        let ty = resolver.get_expr_type(&expr);

        assert!(ty.is_some());
        assert!(matches!(*ty.unwrap().kind(&db), TypeKind::Int));
    }

    #[test]
    fn test_get_expr_type_nat_literal() {
        let db = test_db();
        let resolver = TdnrResolver::new(&db);

        let expr = Expr::new(fresh_node_id(), ExprKind::NatLit(42));
        let ty = resolver.get_expr_type(&expr);

        assert!(ty.is_some());
        assert!(matches!(*ty.unwrap().kind(&db), TypeKind::Nat));
    }

    #[test]
    fn test_get_expr_type_bool_literal() {
        let db = test_db();
        let resolver = TdnrResolver::new(&db);

        let expr = Expr::new(fresh_node_id(), ExprKind::BoolLit(true));
        let ty = resolver.get_expr_type(&expr);

        assert!(ty.is_some());
        assert!(matches!(*ty.unwrap().kind(&db), TypeKind::Bool));
    }

    #[test]
    fn test_get_expr_type_binop_and_returns_bool() {
        let db = test_db();
        let resolver = TdnrResolver::new(&db);

        // Create: true && false
        let lhs = Expr::new(fresh_node_id(), ExprKind::BoolLit(true));
        let rhs = Expr::new(fresh_node_id(), ExprKind::BoolLit(false));
        let expr = Expr::new(
            fresh_node_id(),
            ExprKind::BinOp {
                op: BinOpKind::And,
                lhs,
                rhs,
            },
        );

        let ty = resolver.get_expr_type(&expr);
        assert!(ty.is_some());
        assert!(matches!(*ty.unwrap().kind(&db), TypeKind::Bool));
    }

    #[test]
    fn test_get_expr_type_tuple() {
        let db = test_db();
        let resolver = TdnrResolver::new(&db);

        // Create: (1, true)
        let elements = vec![
            Expr::new(fresh_node_id(), ExprKind::IntLit(1)),
            Expr::new(fresh_node_id(), ExprKind::BoolLit(true)),
        ];
        let expr = Expr::new(fresh_node_id(), ExprKind::Tuple(elements));

        let ty = resolver.get_expr_type(&expr);
        assert!(ty.is_some());
        if let TypeKind::Tuple(elems) = ty.unwrap().kind(&db) {
            assert_eq!(elems.len(), 2);
        } else {
            panic!("Expected Tuple type");
        }
    }

    #[test]
    fn test_get_expr_type_block() {
        let db = test_db();
        let resolver = TdnrResolver::new(&db);

        // Create: { 42 }
        let value = Expr::new(fresh_node_id(), ExprKind::IntLit(42));
        let expr = Expr::new(
            fresh_node_id(),
            ExprKind::Block {
                stmts: vec![],
                value,
            },
        );

        let ty = resolver.get_expr_type(&expr);
        assert!(ty.is_some());
        assert!(matches!(*ty.unwrap().kind(&db), TypeKind::Int));
    }

    #[test]
    fn test_get_expr_type_nil() {
        let db = test_db();
        let resolver = TdnrResolver::new(&db);

        let expr = Expr::new(fresh_node_id(), ExprKind::Nil);
        let ty = resolver.get_expr_type(&expr);

        assert!(ty.is_some());
        assert!(matches!(*ty.unwrap().kind(&db), TypeKind::Nil));
    }

    #[test]
    fn test_annotation_to_type_rune() {
        use crate::ast::TypeAnnotationKind;

        let db = test_db();
        let resolver = TdnrResolver::new(&db);

        let ann = Some(crate::ast::TypeAnnotation {
            id: fresh_node_id(),
            kind: TypeAnnotationKind::Named(Symbol::new("Rune")),
        });

        let ty = resolver.annotation_to_type(&ann);
        assert!(
            matches!(*ty.kind(&db), TypeKind::Rune),
            "Expected Rune type, got {:?}",
            ty.kind(&db)
        );
    }

    #[test]
    fn test_get_expr_type_rune_literal() {
        let db = test_db();
        let resolver = TdnrResolver::new(&db);

        let expr = Expr::new(fresh_node_id(), ExprKind::RuneLit('a'));
        let ty = resolver.get_expr_type(&expr);

        assert!(ty.is_some());
        assert!(matches!(*ty.unwrap().kind(&db), TypeKind::Rune));
    }

    #[salsa::tracked]
    fn test_cons_constructed_type_inner<'db>(db: &'db dyn salsa::Database) -> bool {
        let resolver = TdnrResolver::new(db);

        let option_name = Symbol::new("Option");
        let int_ty = Type::new(db, TypeKind::Int);
        let option_int = Type::new(
            db,
            TypeKind::Named {
                name: option_name,
                args: vec![int_ty],
            },
        );

        let ctor_func_ty = Type::new(
            db,
            TypeKind::Func {
                params: vec![int_ty],
                result: option_int,
                effect: EffectRow::pure(db),
            },
        );

        let some_name = Symbol::new("Some");
        let ctor_id = CtorId::new(db, some_name);
        let ctor_ref = TypedRef {
            resolved: ResolvedRef::Constructor {
                id: ctor_id,
                variant: some_name,
            },
            ty: ctor_func_ty,
        };

        let expr = Expr::new(
            fresh_node_id(),
            ExprKind::Cons {
                ctor: ctor_ref,
                args: vec![Expr::new(fresh_node_id(), ExprKind::IntLit(42))],
            },
        );

        let ty = resolver.get_expr_type(&expr);
        let Some(ty) = ty else { return false };
        matches!(ty.kind(db), TypeKind::Named { name, args } if *name == option_name && args.len() == 1 && matches!(args[0].kind(db), TypeKind::Int))
    }

    #[salsa_test]
    fn test_get_expr_type_cons_returns_constructed_type(db: &dyn salsa::Database) {
        assert!(
            test_cons_constructed_type_inner(db),
            "Cons should return the constructed value type, not the constructor function type"
        );
    }

    #[salsa::tracked]
    fn test_cons_non_func_ctor_inner<'db>(db: &'db dyn salsa::Database) -> bool {
        let resolver = TdnrResolver::new(db);

        let point_name = Symbol::new("Point");
        let point_ty = Type::new(
            db,
            TypeKind::Named {
                name: point_name,
                args: vec![],
            },
        );

        let ctor_id = CtorId::new(db, point_name);
        let ctor_ref = TypedRef {
            resolved: ResolvedRef::Constructor {
                id: ctor_id,
                variant: Symbol::new("Point"),
            },
            ty: point_ty,
        };

        let expr = Expr::new(
            fresh_node_id(),
            ExprKind::Cons {
                ctor: ctor_ref,
                args: vec![],
            },
        );

        let ty = resolver.get_expr_type(&expr);
        let Some(ty) = ty else { return false };
        matches!(ty.kind(db), TypeKind::Named { name, .. } if *name == point_name)
    }

    #[salsa_test]
    fn test_get_expr_type_cons_non_func_ctor(db: &dyn salsa::Database) {
        assert!(
            test_cons_non_func_ctor_inner(db),
            "Cons with non-function ctor type should return the ctor type directly"
        );
    }

    // =========================================================================
    // extract_type_name_from_type tests (수정 2: App type support)
    // =========================================================================

    #[test]
    fn test_extract_type_name_from_named_type() {
        let db = test_db();

        let ty = Type::new(
            &db,
            TypeKind::Named {
                name: Symbol::new("Foo"),
                args: vec![],
            },
        );
        assert_eq!(
            extract_type_name_from_type(&db, ty),
            Some(Symbol::new("Foo"))
        );
    }

    #[test]
    fn test_extract_type_name_from_app_type() {
        let db = test_db();
        // App { ctor: Named("List"), args: [Int] }
        let list_named = Type::new(
            &db,
            TypeKind::Named {
                name: Symbol::new("List"),
                args: vec![],
            },
        );
        let int_ty = Type::new(&db, TypeKind::Int);
        let app_ty = Type::new(
            &db,
            TypeKind::App {
                ctor: list_named,
                args: vec![int_ty],
            },
        );

        assert_eq!(
            extract_type_name_from_type(&db, app_ty),
            Some(Symbol::new("List"))
        );
    }

    #[test]
    fn test_extract_type_name_from_nested_app_type() {
        let db = test_db();
        // App { ctor: App { ctor: Named("Map"), args: [Int] }, args: [String] }
        let map_named = Type::new(
            &db,
            TypeKind::Named {
                name: Symbol::new("Map"),
                args: vec![],
            },
        );
        let int_ty = Type::new(&db, TypeKind::Int);
        let inner_app = Type::new(
            &db,
            TypeKind::App {
                ctor: map_named,
                args: vec![int_ty],
            },
        );
        let string_ty = Type::new(&db, TypeKind::string());
        let outer_app = Type::new(
            &db,
            TypeKind::App {
                ctor: inner_app,
                args: vec![string_ty],
            },
        );

        assert_eq!(
            extract_type_name_from_type(&db, outer_app),
            Some(Symbol::new("Map"))
        );
    }

    #[test]
    fn test_extract_type_name_from_primitive_types() {
        let db = test_db();
        let cases: Vec<(TypeKind, &str)> = vec![
            (TypeKind::Int, "Int"),
            (TypeKind::Nat, "Nat"),
            (TypeKind::Float, "Float"),
            (TypeKind::Bool, "Bool"),
            (TypeKind::string(), "String"),
            (TypeKind::Bytes, "Bytes"),
            (TypeKind::Rune, "Rune"),
            (TypeKind::Nil, "Nil"),
        ];

        for (kind, expected_name) in cases {
            let ty = Type::new(&db, kind);
            assert_eq!(
                extract_type_name_from_type(&db, ty),
                Some(Symbol::new(expected_name)),
                "Expected type name '{}' for primitive type",
                expected_name,
            );
        }
    }

    #[test]
    fn test_extract_type_name_from_unsupported_type_returns_none() {
        let db = test_db();
        // Tuple type has no single type name
        let ty = Type::new(
            &db,
            TypeKind::Tuple(vec![
                Type::new(&db, TypeKind::Int),
                Type::new(&db, TypeKind::Bool),
            ]),
        );
        assert_eq!(extract_type_name_from_type(&db, ty), None);
    }

    // =========================================================================
    // lookup_method conflict detection tests (수정 1: ambiguity)
    // =========================================================================

    #[salsa::tracked]
    fn test_lookup_single_candidate_inner<'db>(db: &'db dyn salsa::Database) -> bool {
        let mut resolver = TdnrResolver::new(db);

        let type_name = Symbol::new("Foo");
        let method_name = Symbol::new("bar");
        let func_id = FuncDefId::new(db, method_name);
        // func_ty must be a Func type with Foo as the first param so receiver_ty() works
        let foo_ty = Type::new(
            db,
            TypeKind::Named {
                name: type_name,
                args: vec![],
            },
        );
        let effect = crate::ast::EffectRow::pure(db);
        let func_ty = Type::new(
            db,
            TypeKind::Func {
                params: vec![foo_ty],
                result: Type::new(db, TypeKind::Int),
                effect,
            },
        );

        resolver
            .method_index
            .entry(method_name)
            .or_default()
            .push(MethodEntry { func_id, func_ty });

        let receiver_ty = Some(Type::new(
            db,
            TypeKind::Named {
                name: type_name,
                args: vec![],
            },
        ));

        let result = resolver.lookup_method(receiver_ty, method_name);
        result.is_some()
    }

    #[salsa_test]
    fn test_lookup_single_candidate_resolves(db: &dyn salsa::Database) {
        assert!(
            test_lookup_single_candidate_inner(db),
            "Single candidate should resolve successfully"
        );
    }

    #[salsa::tracked]
    fn test_lookup_ambiguous_inner<'db>(db: &'db dyn salsa::Database) -> bool {
        let mut resolver = TdnrResolver::new(db);

        let type_name = Symbol::new("Foo");
        let method_name = Symbol::new("bar");

        // Two different functions with the same receiver type — ambiguous
        let foo_ty = Type::new(
            db,
            TypeKind::Named {
                name: type_name,
                args: vec![],
            },
        );
        let effect = crate::ast::EffectRow::pure(db);
        let make_func_ty = |result_ty| {
            Type::new(
                db,
                TypeKind::Func {
                    params: vec![foo_ty],
                    result: result_ty,
                    effect,
                },
            )
        };
        let func_id_1 = FuncDefId::new(db, Symbol::new("bar1"));
        let func_ty_1 = make_func_ty(Type::new(db, TypeKind::Int));
        let func_id_2 = FuncDefId::new(db, Symbol::new("bar2"));
        let func_ty_2 = make_func_ty(Type::new(db, TypeKind::Float));

        let candidates = resolver.method_index.entry(method_name).or_default();
        candidates.push(MethodEntry {
            func_id: func_id_1,
            func_ty: func_ty_1,
        });
        candidates.push(MethodEntry {
            func_id: func_id_2,
            func_ty: func_ty_2,
        });

        let receiver_ty = Some(Type::new(
            db,
            TypeKind::Named {
                name: type_name,
                args: vec![],
            },
        ));

        let result = resolver.lookup_method(receiver_ty, method_name);
        // Should be None due to ambiguity
        result.is_none()
    }

    #[salsa_test]
    fn test_lookup_ambiguous_returns_none(db: &dyn salsa::Database) {
        assert!(
            test_lookup_ambiguous_inner(db),
            "Ambiguous candidates should return None"
        );
    }

    #[salsa::tracked]
    fn test_lookup_no_candidates_inner<'db>(db: &'db dyn salsa::Database) -> bool {
        let resolver = TdnrResolver::new(db);

        let receiver_ty = Some(Type::new(
            db,
            TypeKind::Named {
                name: Symbol::new("Foo"),
                args: vec![],
            },
        ));

        let result = resolver.lookup_method(receiver_ty, Symbol::new("nonexistent"));
        result.is_none()
    }

    #[salsa_test]
    fn test_lookup_no_candidates_returns_none(db: &dyn salsa::Database) {
        assert!(
            test_lookup_no_candidates_inner(db),
            "No candidates should return None"
        );
    }

    #[salsa::tracked]
    fn test_lookup_app_receiver_inner<'db>(db: &'db dyn salsa::Database) -> bool {
        let mut resolver = TdnrResolver::new(db);

        let type_name = Symbol::new("List");
        let method_name = Symbol::new("map");
        let func_id = FuncDefId::new(db, method_name);
        // func_ty must have Named("List") as the first param for receiver_type_matches
        let list_named = Type::new(
            db,
            TypeKind::Named {
                name: type_name,
                args: vec![],
            },
        );
        let effect = crate::ast::EffectRow::pure(db);
        let func_ty = Type::new(
            db,
            TypeKind::Func {
                params: vec![list_named],
                result: Type::new(db, TypeKind::Int),
                effect,
            },
        );

        resolver
            .method_index
            .entry(method_name)
            .or_default()
            .push(MethodEntry { func_id, func_ty });

        // Receiver type is App { ctor: Named("List"), args: [Int] }
        let int_ty = Type::new(db, TypeKind::Int);
        let app_ty = Type::new(
            db,
            TypeKind::App {
                ctor: list_named,
                args: vec![int_ty],
            },
        );

        let result = resolver.lookup_method(Some(app_ty), method_name);
        result.is_some()
    }

    #[salsa_test]
    fn test_lookup_app_receiver_resolves(db: &dyn salsa::Database) {
        assert!(
            test_lookup_app_receiver_inner(db),
            "App type receiver should resolve method via constructor type name"
        );
    }
}
