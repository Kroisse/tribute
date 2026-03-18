//! TDNR resolver implementation.
//!
//! Transforms `MethodCall` expressions into `Call` expressions by resolving
//! the method name using the receiver's type.

use std::collections::HashMap;

use trunk_ir::{Symbol, smallvec::SmallVec};

use crate::ast::{
    Arm, Decl, Expr, ExprKind, FuncDecl, FuncDefId, HandlerArm, HandlerKind, Module, Pattern,
    ResolvedRef, Stmt, Type, TypeAnnotation, TypeAnnotationKind, TypeKind, TypedRef,
};
use crate::build_field_module_path;

/// TDNR resolver for AST expressions.
///
/// Resolves method calls by finding functions where the first parameter
/// type matches the receiver's type.
pub struct TdnrResolver<'db> {
    db: &'db dyn salsa::Database,
    /// Map from (type_name, method_name) to candidate (FuncDefId, function type) pairs.
    /// Multiple candidates arise when different modules define methods with the same name
    /// on the same type. Ambiguity is checked at lookup time.
    method_index: HashMap<(Symbol, Symbol), Vec<(FuncDefId<'db>, Type<'db>)>>,
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

    /// Build an index of methods by (type_name, method_name).
    ///
    /// This indexes functions by their first parameter's type name,
    /// enabling efficient UFCS resolution.
    fn build_method_index(&mut self, module: &Module<TypedRef<'db>>) {
        // Build module path from the module name (if any)
        let module_path: Vec<Symbol> = module.name.into_iter().collect();
        self.index_decls(&module.decls, &module_path);
    }

    /// Recursively index declarations, including nested modules.
    fn index_decls(&mut self, decls: &[Decl<TypedRef<'db>>], module_path: &[Symbol]) {
        for decl in decls {
            match decl {
                Decl::Function(func) => {
                    // Check if this function can be a method (has at least one parameter)
                    if func.params.is_empty() {
                        continue;
                    }

                    let func_name = func.name;

                    // Create FuncDefId with module path
                    let path_vec = SmallVec::from_slice(module_path);
                    let func_id = FuncDefId::new(self.db, path_vec, func_name);

                    // Try to extract the type name from the first parameter's type annotation
                    let first_param = &func.params[0];
                    let type_name = self.extract_type_name(&first_param.ty);

                    // Skip if we can't determine the receiver type - no "_any" fallback
                    let Some(type_name) = type_name else {
                        continue;
                    };

                    // Build function type from parameter and return type annotations
                    let func_ty = self.build_func_type(func);

                    // Register with the actual type name for precise UFCS lookup
                    self.method_index
                        .entry((type_name, func_name))
                        .or_default()
                        .push((func_id, func_ty));
                }
                Decl::Struct(s) => {
                    // Register each field as an accessor method
                    // e.g., struct Point { x: Int, y: Int } registers:
                    //   - (Point, x) → fn x(self: Point) -> Int
                    //   - (Point, y) → fn y(self: Point) -> Int

                    let struct_name = s.name;

                    for field in &s.fields {
                        let Some(field_name) = field.name else {
                            continue; // Skip unnamed fields
                        };

                        // Create synthetic FuncDefId for the accessor
                        // Use module_path + struct_name to avoid collisions across modules
                        let field_path = build_field_module_path(module_path, struct_name);
                        let func_id = FuncDefId::new(self.db, field_path, field_name);

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

                        // Register in method index
                        self.method_index
                            .entry((struct_name, field_name))
                            .or_default()
                            .push((func_id, func_ty));
                    }
                }
                Decl::Module(m) => {
                    if let Some(body) = &m.body {
                        // Build nested module path by appending current module name
                        let mut nested_path = module_path.to_vec();
                        nested_path.push(m.name);
                        self.index_decls(body, &nested_path);
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
        // We use an empty module path since TDNR works after resolution phase.
        let empty_path = trunk_ir::SymbolVec::new();
        crate::ast::abilities_to_effect_row(
            self.db,
            anns,
            &empty_path,
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

    /// Extract the type name from a type annotation.
    fn extract_type_name(&self, annotation: &Option<crate::ast::TypeAnnotation>) -> Option<Symbol> {
        use crate::ast::TypeAnnotationKind;
        let ann = annotation.as_ref()?;
        match &ann.kind {
            TypeAnnotationKind::Named(name) => Some(*name),
            TypeAnnotationKind::Path(path) if !path.is_empty() => {
                // Use the last segment of the path as the type name
                path.last().copied()
            }
            TypeAnnotationKind::App { ctor, .. } => {
                // Recursively extract from the constructor
                self.extract_type_name_from_annotation(ctor)
            }
            _ => None,
        }
    }

    /// Extract the type name from a TypeAnnotation (not Option).
    fn extract_type_name_from_annotation(
        &self,
        annotation: &crate::ast::TypeAnnotation,
    ) -> Option<Symbol> {
        use crate::ast::TypeAnnotationKind;
        match &annotation.kind {
            TypeAnnotationKind::Named(name) => Some(*name),
            TypeAnnotationKind::Path(path) if !path.is_empty() => path.last().copied(),
            TypeAnnotationKind::App { ctor, .. } => self.extract_type_name_from_annotation(ctor),
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
        if let Some((func_id, func_ty)) = self.lookup_method(receiver_ty, method) {
            // Found the method - transform to a Call
            let callee_ref = TypedRef {
                resolved: ResolvedRef::Function { id: func_id },
                ty: func_ty,
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
            ExprKind::StringLit(_) => Some(Type::new(self.db, crate::ast::TypeKind::String)),
            ExprKind::BytesLit(_) => Some(Type::new(self.db, crate::ast::TypeKind::Bytes)),
            ExprKind::RuneLit(_) => Some(Type::new(self.db, crate::ast::TypeKind::Rune)),
            ExprKind::Nil => Some(Type::new(self.db, crate::ast::TypeKind::Nil)),

            // Binary operations: comparison ops return Bool, arithmetic returns operand type
            ExprKind::BinOp { op, lhs, .. } => {
                use crate::ast::BinOpKind;
                match op {
                    BinOpKind::Eq
                    | BinOpKind::Ne
                    | BinOpKind::Lt
                    | BinOpKind::Le
                    | BinOpKind::Gt
                    | BinOpKind::Ge
                    | BinOpKind::And
                    | BinOpKind::Or => Some(Type::new(self.db, crate::ast::TypeKind::Bool)),
                    // Arithmetic operations return the same type as their operands
                    _ => self.get_expr_type(lhs),
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
    /// Returns (FuncDefId, function_type) if found.
    ///
    /// Returns `None` when no candidates exist **or** when multiple candidates
    /// are found (ambiguous). In the ambiguous case the `MethodCall` is kept
    /// so that later passes can report an unresolved-method diagnostic.
    fn lookup_method(
        &self,
        receiver_ty: Option<Type<'db>>,
        method: Symbol,
    ) -> Option<(FuncDefId<'db>, Type<'db>)> {
        let ty = receiver_ty?;
        let type_name = self.extract_type_name_from_type(ty)?;

        let candidates = self.method_index.get(&(type_name, method))?;
        match candidates.as_slice() {
            [] => None,
            [single] => Some(*single),
            _multiple => None, // ambiguous — keep as MethodCall for error reporting
        }
    }

    /// Extract the type name from a `Type` value.
    ///
    /// Handles `Named`, `App` (recursing into the constructor), and primitive
    /// type kinds.
    fn extract_type_name_from_type(&self, ty: Type<'db>) -> Option<Symbol> {
        use crate::ast::TypeKind;
        let kind = ty.kind(self.db);
        if let Some(name) = kind.primitive_name() {
            return Some(Symbol::new(name));
        }
        match kind {
            TypeKind::Named { name, .. } => Some(*name),
            TypeKind::App { ctor, .. } => self.extract_type_name_from_type(*ctor),
            _ => None,
        }
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
            HandlerKind::Result { binding } => HandlerKind::Result {
                binding: self.resolve_pattern(binding),
            },
            HandlerKind::Effect {
                ability,
                op,
                params,
                continuation,
                continuation_local_id,
            } => HandlerKind::Effect {
                ability,
                op,
                params: params
                    .into_iter()
                    .map(|p| self.resolve_pattern(p))
                    .collect(),
                continuation,
                continuation_local_id,
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
    use salsa_test_macros::salsa_test;
    use trunk_ir::SymbolVec;

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
    fn test_get_expr_type_binop_comparison_returns_bool() {
        let db = test_db();
        let resolver = TdnrResolver::new(&db);

        // Create: 1 < 2
        let lhs = Expr::new(fresh_node_id(), ExprKind::IntLit(1));
        let rhs = Expr::new(fresh_node_id(), ExprKind::IntLit(2));
        let expr = Expr::new(
            fresh_node_id(),
            ExprKind::BinOp {
                op: BinOpKind::Lt,
                lhs,
                rhs,
            },
        );

        let ty = resolver.get_expr_type(&expr);
        assert!(ty.is_some());
        assert!(matches!(*ty.unwrap().kind(&db), TypeKind::Bool));
    }

    #[test]
    fn test_get_expr_type_binop_eq_returns_bool() {
        let db = test_db();
        let resolver = TdnrResolver::new(&db);

        // Create: 1 == 2
        let lhs = Expr::new(fresh_node_id(), ExprKind::IntLit(1));
        let rhs = Expr::new(fresh_node_id(), ExprKind::IntLit(2));
        let expr = Expr::new(
            fresh_node_id(),
            ExprKind::BinOp {
                op: BinOpKind::Eq,
                lhs,
                rhs,
            },
        );

        let ty = resolver.get_expr_type(&expr);
        assert!(ty.is_some());
        assert!(matches!(*ty.unwrap().kind(&db), TypeKind::Bool));
    }

    #[test]
    fn test_get_expr_type_binop_add_returns_operand_type() {
        let db = test_db();
        let resolver = TdnrResolver::new(&db);

        // Create: 1 + 2
        let lhs = Expr::new(fresh_node_id(), ExprKind::IntLit(1));
        let rhs = Expr::new(fresh_node_id(), ExprKind::IntLit(2));
        let expr = Expr::new(
            fresh_node_id(),
            ExprKind::BinOp {
                op: BinOpKind::Add,
                lhs,
                rhs,
            },
        );

        let ty = resolver.get_expr_type(&expr);
        assert!(ty.is_some());
        // Add returns the type of the lhs (Int)
        assert!(matches!(*ty.unwrap().kind(&db), TypeKind::Int));
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

        let ctor_id = CtorId::new(db, SymbolVec::new(), option_name);
        let ctor_ref = TypedRef {
            resolved: ResolvedRef::Constructor {
                id: ctor_id,
                variant: Symbol::new("Some"),
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

        let ctor_id = CtorId::new(db, SymbolVec::new(), point_name);
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
        let resolver = TdnrResolver::new(&db);

        let ty = Type::new(
            &db,
            TypeKind::Named {
                name: Symbol::new("Foo"),
                args: vec![],
            },
        );
        assert_eq!(
            resolver.extract_type_name_from_type(ty),
            Some(Symbol::new("Foo"))
        );
    }

    #[test]
    fn test_extract_type_name_from_app_type() {
        let db = test_db();
        let resolver = TdnrResolver::new(&db);

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
            resolver.extract_type_name_from_type(app_ty),
            Some(Symbol::new("List"))
        );
    }

    #[test]
    fn test_extract_type_name_from_nested_app_type() {
        let db = test_db();
        let resolver = TdnrResolver::new(&db);

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
        let string_ty = Type::new(&db, TypeKind::String);
        let outer_app = Type::new(
            &db,
            TypeKind::App {
                ctor: inner_app,
                args: vec![string_ty],
            },
        );

        assert_eq!(
            resolver.extract_type_name_from_type(outer_app),
            Some(Symbol::new("Map"))
        );
    }

    #[test]
    fn test_extract_type_name_from_primitive_types() {
        let db = test_db();
        let resolver = TdnrResolver::new(&db);

        let cases: Vec<(TypeKind, &str)> = vec![
            (TypeKind::Int, "Int"),
            (TypeKind::Nat, "Nat"),
            (TypeKind::Float, "Float"),
            (TypeKind::Bool, "Bool"),
            (TypeKind::String, "String"),
            (TypeKind::Bytes, "Bytes"),
            (TypeKind::Rune, "Rune"),
            (TypeKind::Nil, "Nil"),
        ];

        for (kind, expected_name) in cases {
            let ty = Type::new(&db, kind);
            assert_eq!(
                resolver.extract_type_name_from_type(ty),
                Some(Symbol::new(expected_name)),
                "Expected type name '{}' for primitive type",
                expected_name,
            );
        }
    }

    #[test]
    fn test_extract_type_name_from_unsupported_type_returns_none() {
        let db = test_db();
        let resolver = TdnrResolver::new(&db);

        // Tuple type has no single type name
        let ty = Type::new(
            &db,
            TypeKind::Tuple(vec![
                Type::new(&db, TypeKind::Int),
                Type::new(&db, TypeKind::Bool),
            ]),
        );
        assert_eq!(resolver.extract_type_name_from_type(ty), None);
    }

    // =========================================================================
    // lookup_method conflict detection tests (수정 1: ambiguity)
    // =========================================================================

    #[salsa::tracked]
    fn test_lookup_single_candidate_inner<'db>(db: &'db dyn salsa::Database) -> bool {
        let mut resolver = TdnrResolver::new(db);

        let type_name = Symbol::new("Foo");
        let method_name = Symbol::new("bar");
        let func_id = FuncDefId::new(db, SymbolVec::new(), method_name);
        let func_ty = Type::new(db, TypeKind::Int);

        resolver
            .method_index
            .entry((type_name, method_name))
            .or_default()
            .push((func_id, func_ty));

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

        // Two different functions with the same (type_name, method_name) key
        let func_id_1 = FuncDefId::new(db, SymbolVec::new(), Symbol::new("bar"));
        let func_ty_1 = Type::new(db, TypeKind::Int);
        let func_id_2 = FuncDefId::new(db, SymbolVec::new(), Symbol::new("bar"));
        let func_ty_2 = Type::new(db, TypeKind::Float);

        let candidates = resolver
            .method_index
            .entry((type_name, method_name))
            .or_default();
        candidates.push((func_id_1, func_ty_1));
        candidates.push((func_id_2, func_ty_2));

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
        let func_id = FuncDefId::new(db, SymbolVec::new(), method_name);
        let func_ty = Type::new(db, TypeKind::Int);

        resolver
            .method_index
            .entry((type_name, method_name))
            .or_default()
            .push((func_id, func_ty));

        // Receiver type is App { ctor: Named("List"), args: [Int] }
        let list_named = Type::new(
            db,
            TypeKind::Named {
                name: type_name,
                args: vec![],
            },
        );
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
