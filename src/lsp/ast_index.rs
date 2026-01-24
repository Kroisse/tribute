// TODO: Remove once migration is complete
#![allow(dead_code)]

//! LSP-specific indexes for AST-based lookups.
//!
//! This module provides Salsa-tracked indexes for LSP features like hover,
//! go-to-definition, and find-references. These indexes are built from the
//! typed AST (`Module<TypedRef>`) rather than TrunkIR, enabling incremental
//! updates based on AST changes.
//!
//! ## Indexes
//!
//! - **Type Index**: Maps source positions to inferred types (for hover)
//! - **Definition Index**: Maps names to definitions and references (for go-to-definition)
//! - **Function Signatures**: Collects function signature info (for signature help)
//! - **Completion Index**: Provides completion candidates
//! - **Document Symbols**: Provides document outline

use std::collections::BTreeMap;

use trunk_ir::{Span, Symbol};

use tribute_front::SourceCst;
use tribute_front::ast::{
    AbilityDecl, Arm, ConstDecl, Decl, EnumDecl, Expr, ExprKind, FuncDecl, HandlerArm, HandlerKind,
    LocalId, Module, NodeId, ParamDecl, Pattern, PatternKind, ResolvedRef, SpanMap, Stmt,
    StructDecl, Type, TypeAnnotation, TypeAnnotationKind, TypeKind, TypedRef,
};
use tribute_front::query as ast_query;

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

    /// Get the type for a specific NodeId.
    pub fn type_for_node(
        &self,
        db: &'db dyn salsa::Database,
        node_id: NodeId,
    ) -> Option<&AstTypeEntry<'db>> {
        self.by_node_id(db)
            .get(&node_id)
            .map(|&i| &self.entries(db)[i])
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
) -> Option<AstTypeIndex<'db>> {
    let module = ast_query::tdnr_module(db, source)?;
    let span_map = ast_query::span_map(db, source)?;

    Some(AstTypeIndex::build(db, &module, &span_map))
}

// =============================================================================
// Definition Index (Phase 1)
// =============================================================================

/// Kind of definition for LSP.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DefinitionKind {
    /// A function definition.
    Function,
    /// A struct definition.
    Struct,
    /// An enum definition.
    Enum,
    /// An ability definition.
    Ability,
    /// A local variable binding.
    Local,
    /// A function parameter.
    Parameter,
    /// A constant definition.
    Const,
    /// A struct/enum field.
    Field,
}

/// Entry representing a definition in the AST.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct AstDefinitionEntry {
    /// The NodeId of the definition.
    pub node_id: NodeId,
    /// The span of the definition name.
    pub span: Span,
    /// The name of the definition.
    pub name: Symbol,
    /// The kind of definition.
    pub kind: DefinitionKind,
}

/// Entry representing a reference to a definition.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct AstReferenceEntry {
    /// The NodeId of the reference.
    pub node_id: NodeId,
    /// The span of the reference.
    pub span: Span,
    /// The resolved target of the reference.
    pub target: ResolvedTarget,
}

/// Target of a resolved reference.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub enum ResolvedTarget {
    /// Reference to a local variable.
    Local { id: LocalId, name: Symbol },
    /// Reference to a function.
    Function { name: Symbol },
    /// Reference to a constructor.
    Constructor { type_name: Symbol, variant: Symbol },
    /// Reference to a type.
    Type { name: Symbol },
    /// Reference to an ability.
    Ability { name: Symbol },
    /// Unresolved reference (for error recovery).
    Unresolved { name: Symbol },
    /// Reference to something else (builtin, module, etc.).
    #[allow(dead_code)]
    Other { name: Symbol },
}

impl ResolvedTarget {
    /// Get the name of the target.
    pub fn name(&self) -> Symbol {
        match self {
            ResolvedTarget::Local { name, .. } => *name,
            ResolvedTarget::Function { name } => *name,
            ResolvedTarget::Constructor { variant, .. } => *variant,
            ResolvedTarget::Type { name } => *name,
            ResolvedTarget::Ability { name } => *name,
            ResolvedTarget::Unresolved { name } => *name,
            ResolvedTarget::Other { name } => *name,
        }
    }
}

/// Index for go-to-definition and find-references.
#[salsa::tracked]
pub struct AstDefinitionIndex<'db> {
    /// All definitions, sorted by span.
    #[returns(deref)]
    pub definitions: Vec<AstDefinitionEntry>,
    /// All references, sorted by span.
    #[returns(deref)]
    pub references: Vec<AstReferenceEntry>,
    /// Map from name to definition indices.
    #[returns(ref)]
    by_name: BTreeMap<Symbol, Vec<usize>>,
}

impl<'db> AstDefinitionIndex<'db> {
    /// Build a definition index from an AST module.
    pub fn build(
        db: &'db dyn salsa::Database,
        module: &Module<TypedRef<'db>>,
        span_map: &SpanMap,
    ) -> Self {
        let mut collector = DefinitionCollector::new(db, span_map);
        collector.collect_module(module);

        let mut definitions = collector.definitions;
        let mut references = collector.references;

        // Sort by span
        definitions.sort_by_key(|e| (e.span.start, e.span.end));
        references.sort_by_key(|e| (e.span.start, e.span.end));

        // Build name index
        let mut by_name = BTreeMap::<_, Vec<_>>::new();
        for (i, def) in definitions.iter().enumerate() {
            by_name.entry(def.name).or_default().push(i);
        }

        Self::new(db, definitions, references, by_name)
    }

    /// Find the definition at a given offset (when cursor is on a definition).
    pub fn definition_at_position(
        &self,
        db: &'db dyn salsa::Database,
        offset: usize,
    ) -> Option<&AstDefinitionEntry> {
        self.definitions(db)
            .iter()
            .filter(|e| e.span.start <= offset && offset < e.span.end)
            .min_by_key(|e| e.span.end - e.span.start)
    }

    /// Find the reference at a given offset.
    pub fn reference_at(
        &self,
        db: &'db dyn salsa::Database,
        offset: usize,
    ) -> Option<&AstReferenceEntry> {
        self.references(db)
            .iter()
            .filter(|e| e.span.start <= offset && offset < e.span.end)
            .min_by_key(|e| e.span.end - e.span.start)
    }

    /// Get the definition for a reference at the given offset.
    pub fn definition_at(
        &self,
        db: &'db dyn salsa::Database,
        offset: usize,
    ) -> Option<&AstDefinitionEntry> {
        let reference = self.reference_at(db, offset)?;
        let name = reference.target.name();
        self.definition_of(db, name)
    }

    /// Find a definition by name.
    pub fn definition_of(
        &self,
        db: &'db dyn salsa::Database,
        name: Symbol,
    ) -> Option<&AstDefinitionEntry> {
        self.by_name(db)
            .get(&name)
            .and_then(|indices| indices.first())
            .map(|&i| &self.definitions(db)[i])
    }
    /// Find all references to a symbol.
    pub fn references_of(
        &self,
        db: &'db dyn salsa::Database,
        name: Symbol,
    ) -> Vec<&AstReferenceEntry> {
        self.references(db)
            .iter()
            .filter(|r| r.target.name() == name)
            .collect()
    }

    /// Find references from a position (works for both definitions and references).
    pub fn references_at(
        &self,
        db: &'db dyn salsa::Database,
        offset: usize,
    ) -> Option<(Symbol, Vec<&AstReferenceEntry>)> {
        // Try reference first (more specific than definitions which may span large areas)
        if let Some(reference) = self.reference_at(db, offset) {
            let name = reference.target.name();
            let refs = self.references_of(db, name);
            return Some((name, refs));
        }

        // Fall back to definition
        if let Some(def) = self.definition_at_position(db, offset) {
            let refs = self.references_of(db, def.name);
            return Some((def.name, refs));
        }

        None
    }

    /// Check if renaming is possible at the given position.
    pub fn can_rename(
        &self,
        db: &'db dyn salsa::Database,
        offset: usize,
    ) -> Option<(&AstDefinitionEntry, Span)> {
        // Try reference first (more specific than definitions which may span large areas)
        if let Some(reference) = self.reference_at(db, offset) {
            let name = reference.target.name();
            if let Some(def) = self.definition_of(db, name) {
                return Some((def, reference.span));
            }
        }

        // Fall back to definition
        if let Some(def) = self.definition_at_position(db, offset) {
            return Some((def, def.span));
        }

        None
    }
}

/// Collector for definitions and references.
struct DefinitionCollector<'a, 'db> {
    db: &'db dyn salsa::Database,
    span_map: &'a SpanMap,
    definitions: Vec<AstDefinitionEntry>,
    references: Vec<AstReferenceEntry>,
}

impl<'a, 'db> DefinitionCollector<'a, 'db> {
    fn new(db: &'db dyn salsa::Database, span_map: &'a SpanMap) -> Self {
        Self {
            db,
            span_map,
            definitions: Vec::new(),
            references: Vec::new(),
        }
    }

    fn add_definition(&mut self, node_id: NodeId, name: Symbol, kind: DefinitionKind) {
        let span = self.span_map.get_or_default(node_id);
        self.definitions.push(AstDefinitionEntry {
            node_id,
            span,
            name,
            kind,
        });
    }

    fn add_reference(&mut self, node_id: NodeId, target: ResolvedTarget) {
        let span = self.span_map.get_or_default(node_id);
        self.references.push(AstReferenceEntry {
            node_id,
            span,
            target,
        });
    }

    fn collect_module(&mut self, module: &Module<TypedRef<'db>>) {
        for decl in &module.decls {
            self.collect_decl(decl);
        }
    }

    fn collect_decl(&mut self, decl: &Decl<TypedRef<'db>>) {
        match decl {
            Decl::Function(func) => self.collect_func(func),
            Decl::Struct(s) => self.collect_struct(s),
            Decl::Enum(e) => self.collect_enum(e),
            Decl::Ability(a) => self.collect_ability(a),
            Decl::Const(c) => self.collect_const(c),
            Decl::Use(_) => {}
        }
    }

    fn collect_func(&mut self, func: &FuncDecl<TypedRef<'db>>) {
        // Add function definition
        self.add_definition(func.id, func.name, DefinitionKind::Function);

        // Add parameter definitions
        for param in &func.params {
            self.collect_param(param);
        }

        // Collect references in body
        self.collect_expr(&func.body);
    }

    fn collect_param(&mut self, param: &ParamDecl) {
        self.add_definition(param.id, param.name, DefinitionKind::Parameter);
    }

    fn collect_struct(&mut self, s: &StructDecl) {
        self.add_definition(s.id, s.name, DefinitionKind::Struct);

        // Add field definitions
        for field in &s.fields {
            if let Some(name) = field.name {
                self.add_definition(field.id, name, DefinitionKind::Field);
            }
        }
    }

    fn collect_enum(&mut self, e: &EnumDecl) {
        self.add_definition(e.id, e.name, DefinitionKind::Enum);

        // Add variant definitions
        for variant in &e.variants {
            self.add_definition(variant.id, variant.name, DefinitionKind::Field);
        }
    }

    fn collect_ability(&mut self, a: &AbilityDecl) {
        self.add_definition(a.id, a.name, DefinitionKind::Ability);

        // Add operation definitions
        for op in &a.operations {
            self.add_definition(op.id, op.name, DefinitionKind::Function);
        }
    }

    fn collect_const(&mut self, c: &ConstDecl<TypedRef<'db>>) {
        self.add_definition(c.id, c.name, DefinitionKind::Const);
        self.collect_expr(&c.value);
    }

    fn collect_expr(&mut self, expr: &Expr<TypedRef<'db>>) {
        match expr.kind.as_ref() {
            ExprKind::Var(typed_ref) => {
                let target = self.resolve_typed_ref(typed_ref);
                self.add_reference(expr.id, target);
            }
            ExprKind::Call { callee, args } => {
                self.collect_expr(callee);
                for arg in args {
                    self.collect_expr(arg);
                }
            }
            ExprKind::Cons { ctor, args } => {
                let target = self.resolve_typed_ref(ctor);
                self.add_reference(expr.id, target);
                for arg in args {
                    self.collect_expr(arg);
                }
            }
            ExprKind::Record {
                type_name,
                fields,
                spread,
            } => {
                let target = self.resolve_typed_ref(type_name);
                self.add_reference(expr.id, target);
                for (_, field_expr) in fields {
                    self.collect_expr(field_expr);
                }
                if let Some(spread_expr) = spread {
                    self.collect_expr(spread_expr);
                }
            }
            ExprKind::FieldAccess { expr: inner, .. } => {
                self.collect_expr(inner);
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
            ExprKind::Lambda { params, body } => {
                // Lambda params don't have ParamDecl, just Param
                for param in params {
                    self.add_definition(param.id, param.name, DefinitionKind::Parameter);
                }
                self.collect_expr(body);
            }
            ExprKind::Handle { body, handlers } => {
                self.collect_expr(body);
                for handler in handlers {
                    self.collect_handler(handler);
                }
            }
            ExprKind::Tuple(elems) | ExprKind::List(elems) => {
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
            ExprKind::IntLit(_)
            | ExprKind::FloatLit(_)
            | ExprKind::StringLit(_)
            | ExprKind::BoolLit(_)
            | ExprKind::UnitLit
            | ExprKind::Error => {}
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
            PatternKind::Bind { name } => {
                self.add_definition(pattern.id, *name, DefinitionKind::Local);
            }
            PatternKind::Variant { ctor, fields } => {
                let target = self.resolve_typed_ref(ctor);
                self.add_reference(pattern.id, target);
                for field in fields {
                    self.collect_pattern(field);
                }
            }
            PatternKind::Record {
                type_name, fields, ..
            } => {
                if let Some(tn) = type_name {
                    let target = self.resolve_typed_ref(tn);
                    self.add_reference(pattern.id, target);
                }
                for field in fields {
                    if let Some(p) = &field.pattern {
                        self.collect_pattern(p);
                    } else {
                        // Shorthand: `{ name }` binds `name`
                        self.add_definition(pattern.id, field.name, DefinitionKind::Local);
                    }
                }
            }
            PatternKind::Tuple(elems) | PatternKind::List(elems) => {
                for elem in elems {
                    self.collect_pattern(elem);
                }
            }
            PatternKind::ListRest { head, rest } => {
                for h in head {
                    self.collect_pattern(h);
                }
                if let Some(name) = rest {
                    self.add_definition(pattern.id, *name, DefinitionKind::Local);
                }
            }
            PatternKind::As {
                pattern: inner,
                name,
            } => {
                self.collect_pattern(inner);
                self.add_definition(pattern.id, *name, DefinitionKind::Local);
            }
            PatternKind::Or(alts) => {
                for alt in alts {
                    self.collect_pattern(alt);
                }
            }
            PatternKind::Wildcard | PatternKind::Literal(_) | PatternKind::Error => {}
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
                ability,
                params,
                continuation,
                ..
            } => {
                let target = self.resolve_typed_ref(ability);
                self.add_reference(handler.id, target);
                for param in params {
                    self.collect_pattern(param);
                }
                if let Some(k) = continuation {
                    self.add_definition(handler.id, *k, DefinitionKind::Local);
                }
            }
        }
        self.collect_expr(&handler.body);
    }

    fn resolve_typed_ref(&self, typed_ref: &TypedRef<'db>) -> ResolvedTarget {
        self.resolve_resolved_ref(&typed_ref.resolved)
    }

    fn resolve_resolved_ref(&self, resolved: &ResolvedRef<'db>) -> ResolvedTarget {
        match resolved {
            ResolvedRef::Local { id, name } => ResolvedTarget::Local {
                id: *id,
                name: *name,
            },
            ResolvedRef::Function { id } => ResolvedTarget::Function {
                name: id.name(self.db),
            },
            ResolvedRef::Constructor { id, variant } => ResolvedTarget::Constructor {
                type_name: id.type_name(self.db),
                variant: *variant,
            },
            ResolvedRef::Builtin(_) => ResolvedTarget::Other {
                name: Symbol::new("builtin"),
            },
            ResolvedRef::Module { .. } => ResolvedTarget::Other {
                name: Symbol::new("module"),
            },
        }
    }
}

/// Build a definition index for a source file.
#[salsa::tracked]
pub fn definition_index<'db>(
    db: &'db dyn salsa::Database,
    source: SourceCst,
) -> Option<AstDefinitionIndex<'db>> {
    let module = ast_query::tdnr_module(db, source)?;
    let span_map = ast_query::span_map(db, source)?;

    Some(AstDefinitionIndex::build(db, &module, &span_map))
}

// =============================================================================
// Function Signatures (Phase 2)
// =============================================================================

/// Function signature information for signature help.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunctionSignature {
    /// Function name.
    pub name: Symbol,
    /// Parameter names and type strings.
    pub params: Vec<(Symbol, Option<String>)>,
    /// Return type string (if specified).
    pub return_ty: Option<String>,
    /// Span of the function definition.
    pub span: Span,
}

/// Pretty-print a type annotation to a string.
fn print_type_annotation(ty: &TypeAnnotation) -> String {
    match &ty.kind {
        TypeAnnotationKind::Named(name) => name.to_string(),
        TypeAnnotationKind::Path(parts) => parts
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("::"),
        TypeAnnotationKind::App { ctor, args } => {
            let ctor_str = print_type_annotation(ctor);
            let args_str: Vec<_> = args.iter().map(print_type_annotation).collect();
            format!("{}({})", ctor_str, args_str.join(", "))
        }
        TypeAnnotationKind::Func { params, result } => {
            let params_str: Vec<_> = params.iter().map(print_type_annotation).collect();
            let result_str = print_type_annotation(result);
            format!("({}) -> {}", params_str.join(", "), result_str)
        }
        TypeAnnotationKind::Tuple(elems) => {
            let elems_str: Vec<_> = elems.iter().map(print_type_annotation).collect();
            format!("({})", elems_str.join(", "))
        }
        TypeAnnotationKind::WithEffects { inner, effects } => {
            let inner_str = print_type_annotation(inner);
            let effects_str: Vec<_> = effects.iter().map(print_type_annotation).collect();
            format!("{} ->{{{}}}", inner_str, effects_str.join(", "))
        }
        TypeAnnotationKind::Infer => "_".to_string(),
        TypeAnnotationKind::Error => "?".to_string(),
    }
}

/// Build function signatures from a typed module.
pub fn function_signatures(db: &dyn salsa::Database, source: SourceCst) -> Vec<FunctionSignature> {
    let Some(module) = ast_query::tdnr_module(db, source) else {
        return Vec::new();
    };
    let Some(span_map) = ast_query::span_map(db, source) else {
        return Vec::new();
    };

    let mut signatures = Vec::new();

    for decl in &module.decls {
        if let Decl::Function(func) = decl {
            let params: Vec<_> = func
                .params
                .iter()
                .map(|p| {
                    let ty_str = p.ty.as_ref().map(print_type_annotation);
                    (p.name, ty_str)
                })
                .collect();

            let return_ty = func.return_ty.as_ref().map(print_type_annotation);

            signatures.push(FunctionSignature {
                name: func.name,
                params,
                return_ty,
                span: span_map.get_or_default(func.id),
            });
        }
    }

    signatures
}

/// Find a function signature by name.
pub fn find_signature(
    signatures: &[FunctionSignature],
    name: Symbol,
) -> Option<&FunctionSignature> {
    signatures.iter().find(|s| s.name == name)
}

// =============================================================================
// Completion (Phase 3)
// =============================================================================

/// Completion item kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompletionKind {
    Function,
    Struct,
    Enum,
    Ability,
    Const,
    Variable,
    Keyword,
    Constructor,
}

impl From<CompletionKind> for lsp_types::CompletionItemKind {
    fn from(kind: CompletionKind) -> Self {
        match kind {
            CompletionKind::Function => lsp_types::CompletionItemKind::FUNCTION,
            CompletionKind::Struct => lsp_types::CompletionItemKind::STRUCT,
            CompletionKind::Enum => lsp_types::CompletionItemKind::ENUM,
            CompletionKind::Ability => lsp_types::CompletionItemKind::CLASS,
            CompletionKind::Const => lsp_types::CompletionItemKind::CONSTANT,
            CompletionKind::Variable => lsp_types::CompletionItemKind::VARIABLE,
            CompletionKind::Keyword => lsp_types::CompletionItemKind::KEYWORD,
            CompletionKind::Constructor => lsp_types::CompletionItemKind::CONSTRUCTOR,
        }
    }
}

/// Completion item.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct AstCompletionItem {
    /// Name of the item.
    pub name: Symbol,
    /// Kind of completion.
    pub kind: CompletionKind,
    /// Brief documentation or type hint.
    pub detail: Option<String>,
}

/// Reserved keywords in Tribute.
pub const KEYWORDS: &[&str] = &[
    "fn", "let", "case", "struct", "enum", "ability", "const", "pub", "use", "mod", "if", "handle",
    "as", "True", "False", "Nil",
];

/// Get keyword completions filtered by prefix.
pub fn complete_keywords(prefix: &str) -> Vec<AstCompletionItem> {
    KEYWORDS
        .iter()
        .filter(|kw| kw.starts_with(prefix))
        .map(|kw| AstCompletionItem {
            name: Symbol::new(kw),
            kind: CompletionKind::Keyword,
            detail: None,
        })
        .collect()
}

/// Build completion items from a typed module.
#[salsa::tracked(returns(deref))]
pub fn completion_items(db: &dyn salsa::Database, source: SourceCst) -> Vec<AstCompletionItem> {
    let Some(module) = ast_query::tdnr_module(db, source) else {
        return Vec::new();
    };

    let mut items = Vec::new();

    for decl in &module.decls {
        match decl {
            Decl::Function(func) => {
                items.push(AstCompletionItem {
                    name: func.name,
                    kind: CompletionKind::Function,
                    detail: None,
                });
            }
            Decl::Struct(s) => {
                items.push(AstCompletionItem {
                    name: s.name,
                    kind: CompletionKind::Struct,
                    detail: None,
                });
            }
            Decl::Enum(e) => {
                items.push(AstCompletionItem {
                    name: e.name,
                    kind: CompletionKind::Enum,
                    detail: None,
                });
                // Add variant constructors
                for variant in &e.variants {
                    items.push(AstCompletionItem {
                        name: variant.name,
                        kind: CompletionKind::Constructor,
                        detail: Some(format!("{}::{}", e.name, variant.name)),
                    });
                }
            }
            Decl::Ability(a) => {
                items.push(AstCompletionItem {
                    name: a.name,
                    kind: CompletionKind::Ability,
                    detail: None,
                });
            }
            Decl::Const(c) => {
                items.push(AstCompletionItem {
                    name: c.name,
                    kind: CompletionKind::Const,
                    detail: None,
                });
            }
            Decl::Use(_) => {}
        }
    }

    items
}

/// Filter completion items by prefix.
pub fn filter_completions<'a>(
    items: &'a [AstCompletionItem],
    prefix: &'a str,
) -> impl Iterator<Item = &'a AstCompletionItem> {
    items
        .iter()
        .filter(move |item| item.name.with_str(|s| s.starts_with(prefix)))
}

// =============================================================================
// Document Symbols (Phase 4)
// =============================================================================

/// Symbol kind for document outline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SymbolKind {
    Function,
    Struct,
    Enum,
    Ability,
    Const,
    Field,
    Variant,
}

/// Document symbol information.
#[derive(Clone, Debug, PartialEq, Eq, Hash, salsa::Update)]
pub struct DocumentSymbolInfo {
    /// Symbol name.
    pub name: Symbol,
    /// Symbol kind.
    pub kind: SymbolKind,
    /// Full span of the symbol definition.
    pub span: Span,
    /// Children symbols (e.g., struct fields, enum variants).
    pub children: Vec<DocumentSymbolInfo>,
}

/// Build document symbols from a parsed module.
///
/// Uses the parsed module (before type checking) for faster response.
#[salsa::tracked]
pub fn document_symbols(db: &dyn salsa::Database, source: SourceCst) -> Vec<DocumentSymbolInfo> {
    let Some(module) = ast_query::parsed_module(db, source) else {
        return Vec::new();
    };
    let Some(span_map) = ast_query::span_map(db, source) else {
        return Vec::new();
    };

    let mut symbols = Vec::new();

    for decl in &module.decls {
        match decl {
            Decl::Function(func) => {
                symbols.push(DocumentSymbolInfo {
                    name: func.name,
                    kind: SymbolKind::Function,
                    span: span_map.get_or_default(func.id),
                    children: vec![],
                });
            }
            Decl::Struct(s) => {
                let children: Vec<_> = s
                    .fields
                    .iter()
                    .filter_map(|f| {
                        f.name.map(|name| DocumentSymbolInfo {
                            name,
                            kind: SymbolKind::Field,
                            span: span_map.get_or_default(f.id),
                            children: vec![],
                        })
                    })
                    .collect();

                symbols.push(DocumentSymbolInfo {
                    name: s.name,
                    kind: SymbolKind::Struct,
                    span: span_map.get_or_default(s.id),
                    children,
                });
            }
            Decl::Enum(e) => {
                let children: Vec<_> = e
                    .variants
                    .iter()
                    .map(|v| DocumentSymbolInfo {
                        name: v.name,
                        kind: SymbolKind::Variant,
                        span: span_map.get_or_default(v.id),
                        children: vec![],
                    })
                    .collect();

                symbols.push(DocumentSymbolInfo {
                    name: e.name,
                    kind: SymbolKind::Enum,
                    span: span_map.get_or_default(e.id),
                    children,
                });
            }
            Decl::Ability(a) => {
                let children: Vec<_> = a
                    .operations
                    .iter()
                    .map(|op| DocumentSymbolInfo {
                        name: op.name,
                        kind: SymbolKind::Function,
                        span: span_map.get_or_default(op.id),
                        children: vec![],
                    })
                    .collect();

                symbols.push(DocumentSymbolInfo {
                    name: a.name,
                    kind: SymbolKind::Ability,
                    span: span_map.get_or_default(a.id),
                    children,
                });
            }
            Decl::Const(c) => {
                symbols.push(DocumentSymbolInfo {
                    name: c.name,
                    kind: SymbolKind::Const,
                    span: span_map.get_or_default(c.id),
                    children: vec![],
                });
            }
            Decl::Use(_) => {}
        }
    }

    symbols
}

// =============================================================================
// Identifier Validation (for rename)
// =============================================================================

/// Error type for rename validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenameError {
    EmptyName,
    InvalidIdentifier,
    InvalidTypeIdentifier,
    InvalidCharacter,
    ReservedKeyword,
}

impl std::fmt::Display for RenameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenameError::EmptyName => write!(f, "Name cannot be empty"),
            RenameError::InvalidIdentifier => {
                write!(
                    f,
                    "Identifier must start with lowercase letter or underscore"
                )
            }
            RenameError::InvalidTypeIdentifier => {
                write!(f, "Type identifier must start with uppercase letter")
            }
            RenameError::InvalidCharacter => write!(f, "Name contains invalid characters"),
            RenameError::ReservedKeyword => write!(f, "Name is a reserved keyword"),
        }
    }
}

impl std::error::Error for RenameError {}

/// Check if a name is a reserved keyword.
pub fn is_keyword(name: &str) -> bool {
    KEYWORDS.contains(&name)
}

/// Validate an identifier for renaming.
pub fn validate_identifier(name: &str, kind: DefinitionKind) -> Result<(), RenameError> {
    if name.is_empty() {
        return Err(RenameError::EmptyName);
    }

    let first = name.chars().next().unwrap();

    match kind {
        DefinitionKind::Struct | DefinitionKind::Enum | DefinitionKind::Ability => {
            if !first.is_ascii_uppercase() {
                return Err(RenameError::InvalidTypeIdentifier);
            }
        }
        _ => {
            if !first.is_ascii_lowercase() && first != '_' {
                return Err(RenameError::InvalidIdentifier);
            }
        }
    }

    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(RenameError::InvalidCharacter);
    }

    if is_keyword(name) {
        return Err(RenameError::ReservedKeyword);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ropey::Rope;
    use tree_sitter::Parser;
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

    #[test]
    fn test_definition_index_local_variable() {
        let db = salsa::DatabaseImpl::default();
        // Variable definition and uses (newline-separated)
        let source = make_source(
            &db,
            r#"fn main() {
    let foo = 1
    foo + foo
}"#,
        );

        let index = definition_index(&db, source);
        assert!(index.is_some(), "Should build definition index");

        let index = index.unwrap();

        // Find the definition of "foo"
        let foo_sym = trunk_ir::Symbol::new("foo");
        let foo_def = index.definition_of(&db, foo_sym);
        assert!(foo_def.is_some(), "Should find definition of 'foo'");

        // Find references to "foo" (should have 2: foo + foo)
        let foo_refs = index.references_of(&db, foo_sym);
        assert_eq!(foo_refs.len(), 2, "Should have 2 references to 'foo'");
    }
}
