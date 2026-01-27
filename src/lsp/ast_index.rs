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
    AbilityDecl, Arm, Decl, EnumDecl, Expr, ExprKind, FuncDecl, HandlerArm, HandlerKind, LocalId,
    Module, NodeId, ParamDecl, Pattern, PatternKind, ResolvedRef, SpanMap, Stmt, StructDecl, Type,
    TypeAnnotation, TypeAnnotationKind, TypeKind, TypedRef, UniVarSource,
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
        TypeKind::Rune => "Rune".to_string(),
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
            // Use the index for polymorphic sources, or the counter for anonymous sources
            let display_index = match id.source(db) {
                UniVarSource::Anonymous(counter) => counter as u32 + id.index(db),
                _ => id.index(db),
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
            // Struct, Enum, Ability, Use don't have expression types
            Decl::Struct(_) | Decl::Enum(_) | Decl::Ability(_) | Decl::Use(_) => {}
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

// =============================================================================
// Definition Index (Phase 1)
// =============================================================================

/// Kind of definition for LSP.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DefinitionKind {
    /// A function definition.
    Function,
    /// A struct definition.
    Struct,
    /// An enum definition.
    Enum,
    /// An enum variant (constructor).
    EnumVariant {
        /// The owning enum type name.
        owner: Symbol,
    },
    /// An ability definition.
    Ability,
    /// A local variable binding.
    Local,
    /// A function parameter.
    Parameter,
    /// A constant definition.
    Const,
    /// A struct field.
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
    /// The LocalId for local bindings (enables shadowed variable disambiguation).
    pub local_id: Option<LocalId>,
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

    /// Get the LocalId if this is a local reference.
    pub fn local_id(&self) -> Option<LocalId> {
        match self {
            ResolvedTarget::Local { id, .. } => Some(*id),
            _ => None,
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
    ///
    /// Uses LocalId-based matching for precise shadowed variable disambiguation.
    pub fn definition_at(
        &self,
        db: &'db dyn salsa::Database,
        offset: usize,
    ) -> Option<&AstDefinitionEntry> {
        let reference = self.reference_at(db, offset)?;
        self.definition_of_target(db, &reference.target)
    }

    /// Find a definition by name.
    ///
    /// Note: For locals with shadowing, this may return the wrong definition.
    /// Use `definition_of_target` with a `ResolvedTarget` for precise matching.
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

    /// Find a definition by resolved target.
    ///
    /// For local references, this uses LocalId to precisely match the definition,
    /// enabling correct shadowed variable disambiguation.
    pub fn definition_of_target(
        &self,
        db: &'db dyn salsa::Database,
        target: &ResolvedTarget,
    ) -> Option<&AstDefinitionEntry> {
        match target {
            ResolvedTarget::Local { id, name } => {
                // For locals, match by LocalId for precise scoping
                self.definitions(db)
                    .iter()
                    .find(|d| d.name == *name && d.local_id == Some(*id))
            }
            _ => {
                // For non-locals, fall back to name-based lookup
                self.definition_of(db, target.name())
            }
        }
    }

    /// Find all references to a symbol.
    ///
    /// Note: For locals with shadowing, this may include references to different
    /// bindings with the same name. Use `references_of_target` for precise results.
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

    /// Find all references to a resolved target.
    ///
    /// For local references, this uses LocalId to precisely match only references
    /// to the same binding, excluding references to shadowed variables with the same name.
    pub fn references_of_target(
        &self,
        db: &'db dyn salsa::Database,
        target: &ResolvedTarget,
    ) -> Vec<&AstReferenceEntry> {
        match target {
            ResolvedTarget::Local { id, name } => {
                // For locals, match by LocalId for precise scoping
                self.references(db)
                    .iter()
                    .filter(|r| r.target.name() == *name && r.target.local_id() == Some(*id))
                    .collect()
            }
            _ => {
                // For non-locals, fall back to name-based lookup
                self.references_of(db, target.name())
            }
        }
    }

    /// Find references from a position (works for both definitions and references).
    ///
    /// Uses LocalId-based matching for precise shadowed variable disambiguation.
    pub fn references_at(
        &self,
        db: &'db dyn salsa::Database,
        offset: usize,
    ) -> Option<(Symbol, Vec<&AstReferenceEntry>)> {
        // Try reference first (more specific than definitions which may span large areas)
        if let Some(reference) = self.reference_at(db, offset) {
            let name = reference.target.name();
            let refs = self.references_of_target(db, &reference.target);
            return Some((name, refs));
        }

        // Fall back to definition
        if let Some(def) = self.definition_at_position(db, offset) {
            let refs = if let Some(local_id) = def.local_id {
                // For locals with LocalId, use precise matching
                let target = ResolvedTarget::Local {
                    id: local_id,
                    name: def.name,
                };
                self.references_of_target(db, &target)
            } else {
                self.references_of(db, def.name)
            };
            return Some((def.name, refs));
        }

        None
    }

    /// Check if renaming is possible at the given position.
    ///
    /// Uses LocalId-based matching for precise shadowed variable disambiguation.
    pub fn can_rename(
        &self,
        db: &'db dyn salsa::Database,
        offset: usize,
    ) -> Option<(&AstDefinitionEntry, Span)> {
        // Try reference first (more specific than definitions which may span large areas)
        if let Some(reference) = self.reference_at(db, offset)
            && let Some(def) = self.definition_of_target(db, &reference.target)
        {
            return Some((def, reference.span));
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

    fn add_definition(
        &mut self,
        node_id: NodeId,
        name: Symbol,
        kind: DefinitionKind,
        local_id: Option<LocalId>,
    ) {
        let span = self.span_map.get_or_default(node_id);
        self.definitions.push(AstDefinitionEntry {
            node_id,
            span,
            name,
            kind,
            local_id,
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
            Decl::Use(_) => {}
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
        // Add function definition
        self.add_definition(func.id, func.name, DefinitionKind::Function, None);

        // Add parameter definitions
        for param in &func.params {
            self.collect_param(param);
        }

        // Collect references in body
        self.collect_expr(&func.body);
    }

    fn collect_param(&mut self, param: &ParamDecl) {
        self.add_definition(param.id, param.name, DefinitionKind::Parameter, None);
    }

    fn collect_struct(&mut self, s: &StructDecl) {
        self.add_definition(s.id, s.name, DefinitionKind::Struct, None);

        // Add field definitions
        for field in &s.fields {
            if let Some(name) = field.name {
                self.add_definition(field.id, name, DefinitionKind::Field, None);
            }
        }
    }

    fn collect_enum(&mut self, e: &EnumDecl) {
        self.add_definition(e.id, e.name, DefinitionKind::Enum, None);

        // Add variant definitions
        for variant in &e.variants {
            self.add_definition(
                variant.id,
                variant.name,
                DefinitionKind::EnumVariant { owner: e.name },
                None,
            );
        }
    }

    fn collect_ability(&mut self, a: &AbilityDecl) {
        self.add_definition(a.id, a.name, DefinitionKind::Ability, None);

        // Add operation definitions
        for op in &a.operations {
            self.add_definition(op.id, op.name, DefinitionKind::Function, None);
        }
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
            ExprKind::Lambda { params, body } => {
                // Lambda params don't have ParamDecl, just Param
                for param in params {
                    self.add_definition(param.id, param.name, DefinitionKind::Parameter, None);
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
            ExprKind::NatLit(_)
            | ExprKind::IntLit(_)
            | ExprKind::FloatLit(_)
            | ExprKind::StringLit(_)
            | ExprKind::BytesLit(_)
            | ExprKind::BoolLit(_)
            | ExprKind::RuneLit(_)
            | ExprKind::Nil
            | ExprKind::Error => {}
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
            PatternKind::Bind { name, local_id } => {
                // Use LocalId for scope-aware definition tracking (shadowed variable disambiguation)
                self.add_definition(pattern.id, *name, DefinitionKind::Local, *local_id);
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
                        // Shorthand: `{ name }` binds `name` (no LocalId available)
                        // Use field.id for per-field span tracking
                        self.add_definition(field.id, field.name, DefinitionKind::Local, None);
                    }
                }
            }
            PatternKind::Tuple(elems) | PatternKind::List(elems) => {
                for elem in elems {
                    self.collect_pattern(elem);
                }
            }
            PatternKind::ListRest {
                head,
                rest,
                rest_local_id,
            } => {
                for h in head {
                    self.collect_pattern(h);
                }
                if let Some(name) = rest {
                    // Rest binding has LocalId from name resolution
                    self.add_definition(pattern.id, *name, DefinitionKind::Local, *rest_local_id);
                }
            }
            PatternKind::As {
                pattern: inner,
                name,
                local_id,
            } => {
                self.collect_pattern(inner);
                // As binding has LocalId from name resolution
                self.add_definition(pattern.id, *name, DefinitionKind::Local, *local_id);
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
                    // Continuation binding doesn't have LocalId
                    self.add_definition(handler.id, *k, DefinitionKind::Local, None);
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
            Decl::Use(_) | Decl::Module(_) => {}
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
            Decl::Use(_) | Decl::Module(_) => {}
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
        DefinitionKind::Struct
        | DefinitionKind::Enum
        | DefinitionKind::EnumVariant { .. }
        | DefinitionKind::Ability => {
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

    // =========================================================================
    // Additional print_ast_type tests
    // =========================================================================

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
        use tribute_front::ast::{Effect, EffectRow};

        let db = salsa::DatabaseImpl::default();

        let int_ty = Type::new(&db, TypeKind::Int);
        let effect = Effect {
            name: trunk_ir::Symbol::new("IO"),
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

    // =========================================================================
    // Completion tests
    // =========================================================================

    #[test]
    fn test_complete_keywords_fn() {
        let completions = complete_keywords("fn");
        assert_eq!(completions.len(), 1);
        assert!(completions[0].name == trunk_ir::Symbol::new("fn"));
        assert_eq!(completions[0].kind, CompletionKind::Keyword);
    }

    #[test]
    fn test_complete_keywords_empty_prefix() {
        let completions = complete_keywords("");
        assert_eq!(completions.len(), KEYWORDS.len());
    }

    #[test]
    fn test_complete_keywords_no_match() {
        let completions = complete_keywords("xyz");
        assert!(completions.is_empty());
    }

    #[test]
    fn test_complete_keywords_partial() {
        let completions = complete_keywords("st");
        assert_eq!(completions.len(), 1);
        assert!(completions[0].name == trunk_ir::Symbol::new("struct"));
    }

    #[test]
    fn test_filter_completions() {
        let items = vec![
            AstCompletionItem {
                name: trunk_ir::Symbol::new("foo"),
                kind: CompletionKind::Function,
                detail: None,
            },
            AstCompletionItem {
                name: trunk_ir::Symbol::new("bar"),
                kind: CompletionKind::Function,
                detail: None,
            },
            AstCompletionItem {
                name: trunk_ir::Symbol::new("foobar"),
                kind: CompletionKind::Function,
                detail: None,
            },
        ];

        let filtered: Vec<_> = filter_completions(&items, "foo").collect();
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_completion_items_function() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn hello() { 1 }");

        let items = completion_items(&db, source);
        assert!(!items.is_empty());

        let hello_item = items
            .iter()
            .find(|i| i.name == trunk_ir::Symbol::new("hello"));
        assert!(hello_item.is_some());
        assert_eq!(hello_item.unwrap().kind, CompletionKind::Function);
    }

    #[test]
    fn test_completion_items_struct() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "struct Point { x: Int, y: Int }");

        let items = completion_items(&db, source);
        let point_item = items
            .iter()
            .find(|i| i.name == trunk_ir::Symbol::new("Point"));
        assert!(point_item.is_some());
        assert_eq!(point_item.unwrap().kind, CompletionKind::Struct);
    }

    #[test]
    fn test_completion_items_enum_with_variants() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "enum Color { Red, Green, Blue }");

        let items = completion_items(&db, source);

        // Should have the enum
        let color_item = items
            .iter()
            .find(|i| i.name == trunk_ir::Symbol::new("Color"));
        assert!(color_item.is_some());
        assert_eq!(color_item.unwrap().kind, CompletionKind::Enum);

        // Should have the variants
        let red_item = items
            .iter()
            .find(|i| i.name == trunk_ir::Symbol::new("Red"));
        assert!(red_item.is_some());
        assert_eq!(red_item.unwrap().kind, CompletionKind::Constructor);
    }

    // =========================================================================
    // Document symbols tests
    // =========================================================================

    #[test]
    fn test_document_symbols_function() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn main() { 1 }");

        let symbols = document_symbols(&db, source);
        assert!(!symbols.is_empty());

        let main_sym = symbols
            .iter()
            .find(|s| s.name == trunk_ir::Symbol::new("main"));
        assert!(main_sym.is_some());
        assert_eq!(main_sym.unwrap().kind, SymbolKind::Function);
    }

    #[test]
    fn test_document_symbols_struct_with_fields() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "struct Point { x: Int, y: Int }");

        let symbols = document_symbols(&db, source);

        let point_sym = symbols
            .iter()
            .find(|s| s.name == trunk_ir::Symbol::new("Point"));
        assert!(point_sym.is_some());
        assert_eq!(point_sym.unwrap().kind, SymbolKind::Struct);

        // Should have field children
        let point = point_sym.unwrap();
        assert_eq!(point.children.len(), 2);
        assert!(
            point
                .children
                .iter()
                .any(|c| c.name == trunk_ir::Symbol::new("x"))
        );
        assert!(
            point
                .children
                .iter()
                .any(|c| c.name == trunk_ir::Symbol::new("y"))
        );
    }

    #[test]
    fn test_document_symbols_enum_with_variants() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "enum Option { Some(a), None }");

        let symbols = document_symbols(&db, source);

        let option_sym = symbols
            .iter()
            .find(|s| s.name == trunk_ir::Symbol::new("Option"));
        assert!(option_sym.is_some());
        assert_eq!(option_sym.unwrap().kind, SymbolKind::Enum);

        // Should have variant children
        let option = option_sym.unwrap();
        assert_eq!(option.children.len(), 2);
    }

    // =========================================================================
    // Validate identifier tests
    // =========================================================================

    #[test]
    fn test_validate_identifier_valid() {
        assert!(validate_identifier("foo", DefinitionKind::Local).is_ok());
        assert!(validate_identifier("_private", DefinitionKind::Function).is_ok());
        assert!(validate_identifier("foo_bar", DefinitionKind::Parameter).is_ok());
    }

    #[test]
    fn test_validate_identifier_type_must_be_uppercase() {
        assert!(validate_identifier("point", DefinitionKind::Struct).is_err());
        assert!(validate_identifier("Point", DefinitionKind::Struct).is_ok());
        assert!(validate_identifier("option", DefinitionKind::Enum).is_err());
        assert!(validate_identifier("Option", DefinitionKind::Enum).is_ok());
    }

    #[test]
    fn test_validate_identifier_empty() {
        let result = validate_identifier("", DefinitionKind::Local);
        assert!(matches!(result, Err(RenameError::EmptyName)));
    }

    #[test]
    fn test_validate_identifier_reserved_keyword() {
        let result = validate_identifier("fn", DefinitionKind::Local);
        assert!(matches!(result, Err(RenameError::ReservedKeyword)));
    }

    #[test]
    fn test_validate_identifier_invalid_chars() {
        let result = validate_identifier("foo-bar", DefinitionKind::Local);
        assert!(matches!(result, Err(RenameError::InvalidCharacter)));
    }

    #[test]
    fn test_is_keyword() {
        assert!(is_keyword("fn"));
        assert!(is_keyword("let"));
        assert!(is_keyword("struct"));
        assert!(!is_keyword("foo"));
        assert!(!is_keyword("main"));
    }

    // =========================================================================
    // Function signatures tests
    // =========================================================================

    #[test]
    fn test_function_signatures_simple() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn add(a: Int, b: Int) -> Int { a + b }");

        let signatures = function_signatures(&db, source);
        assert_eq!(signatures.len(), 1);

        let sig = &signatures[0];
        assert!(sig.name == trunk_ir::Symbol::new("add"));
        assert_eq!(sig.params.len(), 2);
        assert_eq!(sig.params[0].1, Some("Int".to_string()));
        assert_eq!(sig.params[1].1, Some("Int".to_string()));
        assert_eq!(sig.return_ty, Some("Int".to_string()));
    }

    #[test]
    fn test_function_signatures_no_annotations() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn identity(x) { x }");

        let signatures = function_signatures(&db, source);
        assert_eq!(signatures.len(), 1);

        let sig = &signatures[0];
        assert!(sig.params[0].1.is_none());
        assert!(sig.return_ty.is_none());
    }

    #[test]
    fn test_find_signature() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn foo() { 1 }\nfn bar() { 2 }");

        let signatures = function_signatures(&db, source);
        assert_eq!(signatures.len(), 2);

        let foo = find_signature(&signatures, trunk_ir::Symbol::new("foo"));
        assert!(foo.is_some());

        let baz = find_signature(&signatures, trunk_ir::Symbol::new("baz"));
        assert!(baz.is_none());
    }

    // =========================================================================
    // Definition index additional tests
    // =========================================================================

    #[test]
    fn test_definition_index_function() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn hello() { 1 }");

        let index = definition_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();
        let hello_sym = trunk_ir::Symbol::new("hello");
        let def = index.definition_of(&db, hello_sym);
        assert!(def.is_some());
        assert_eq!(def.unwrap().kind, DefinitionKind::Function);
    }

    #[test]
    fn test_definition_index_struct() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "struct Point { x: Int }");

        let index = definition_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();

        // Check struct definition
        let point_def = index.definition_of(&db, trunk_ir::Symbol::new("Point"));
        assert!(point_def.is_some());
        assert_eq!(point_def.unwrap().kind, DefinitionKind::Struct);

        // Check field definition
        let x_def = index.definition_of(&db, trunk_ir::Symbol::new("x"));
        assert!(x_def.is_some());
        assert_eq!(x_def.unwrap().kind, DefinitionKind::Field);
    }

    #[test]
    fn test_definition_index_parameter() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn foo(x: Int) { x }");

        let index = definition_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();
        let x_def = index.definition_of(&db, trunk_ir::Symbol::new("x"));
        assert!(x_def.is_some());
        assert_eq!(x_def.unwrap().kind, DefinitionKind::Parameter);
    }

    #[test]
    fn test_definition_at_position() {
        let db = salsa::DatabaseImpl::default();
        //       0         1
        //       0123456789012345
        let source = make_source(&db, "fn foo() { 1 }");

        let index = definition_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();

        // Position 3-6 is "foo"
        let def = index.definition_at_position(&db, 4);
        assert!(def.is_some());
        assert!(def.unwrap().name == trunk_ir::Symbol::new("foo"));
    }

    #[test]
    fn test_references_at() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"fn main() {
    let x = 1
    x + x
}"#,
        );

        let index = definition_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();
        let x_sym = trunk_ir::Symbol::new("x");

        // Find all references
        let refs = index.references_of(&db, x_sym);
        assert_eq!(refs.len(), 2);
    }

    #[test]
    fn test_can_rename() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn foo() { 1 }");

        let index = definition_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();

        // Position 4 is in "foo"
        let result = index.can_rename(&db, 4);
        assert!(result.is_some());

        let (def, _span) = result.unwrap();
        assert!(def.name == trunk_ir::Symbol::new("foo"));
    }

    // =========================================================================
    // Pattern Collection Tests
    // =========================================================================

    #[test]
    fn test_pattern_bind_simple() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"fn main() -> Int {
    let x = 42
    x
}"#,
        );

        let index = definition_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();
        let x_def = index.definition_of(&db, trunk_ir::Symbol::new("x"));
        assert!(x_def.is_some());
        assert_eq!(x_def.unwrap().kind, DefinitionKind::Local);
    }

    #[test]
    fn test_pattern_tuple() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"fn main() -> Int {
    let #(a, b) = #(1, 2)
    a + b
}"#,
        );

        let index = definition_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();

        // Both a and b should be defined as locals
        let a_def = index.definition_of(&db, trunk_ir::Symbol::new("a"));
        assert!(a_def.is_some());
        assert_eq!(a_def.unwrap().kind, DefinitionKind::Local);

        let b_def = index.definition_of(&db, trunk_ir::Symbol::new("b"));
        assert!(b_def.is_some());
        assert_eq!(b_def.unwrap().kind, DefinitionKind::Local);
    }

    #[test]
    fn test_pattern_nested_tuple() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"fn main() -> Int {
    let #(#(a, b), c) = #(#(1, 2), 3)
    a + b + c
}"#,
        );

        let index = definition_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();

        // All three variables should be defined
        for name in ["a", "b", "c"] {
            let def = index.definition_of(&db, trunk_ir::Symbol::new(name));
            assert!(def.is_some(), "Expected definition for '{}'", name);
            assert_eq!(def.unwrap().kind, DefinitionKind::Local);
        }
    }

    #[test]
    fn test_pattern_wildcard() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"fn main() -> Int {
    let _ = 42
    0
}"#,
        );

        let index = definition_index(&db, source);
        assert!(index.is_some());

        // Wildcard should not create a definition
        let index = index.unwrap();
        let underscore_def = index.definition_of(&db, trunk_ir::Symbol::new("_"));
        assert!(underscore_def.is_none());
    }

    #[test]
    fn test_pattern_case_variant() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"enum Option { Some(Int), None }

fn unwrap(opt: Option) -> Int {
    case opt {
        Some(v) -> v
        None -> 0
    }
}"#,
        );

        let index = definition_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();

        // 'v' should be defined in the Some pattern
        let v_def = index.definition_of(&db, trunk_ir::Symbol::new("v"));
        assert!(v_def.is_some());
        assert_eq!(v_def.unwrap().kind, DefinitionKind::Local);

        // 'opt' should be a parameter
        let opt_def = index.definition_of(&db, trunk_ir::Symbol::new("opt"));
        assert!(opt_def.is_some());
        assert_eq!(opt_def.unwrap().kind, DefinitionKind::Parameter);

        // Enum variants should be defined with owner
        let some_def = index.definition_of(&db, trunk_ir::Symbol::new("Some"));
        assert!(some_def.is_some());
        assert!(matches!(
            some_def.unwrap().kind,
            DefinitionKind::EnumVariant { .. }
        ));

        let none_def = index.definition_of(&db, trunk_ir::Symbol::new("None"));
        assert!(none_def.is_some());
        assert!(matches!(
            none_def.unwrap().kind,
            DefinitionKind::EnumVariant { .. }
        ));
    }

    #[test]
    fn test_pattern_shadowing_with_local_id() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"fn main() -> Int {
    let x = 1
    let y = {
        let x = 2
        x
    }
    x + y
}"#,
        );

        let index = definition_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();
        let x_sym = trunk_ir::Symbol::new("x");

        // There should be two definitions of 'x'
        let x_defs: Vec<_> = index
            .definitions(&db)
            .iter()
            .filter(|d| d.name == x_sym)
            .collect();
        assert_eq!(x_defs.len(), 2, "Expected 2 definitions of 'x'");

        // Both should have LocalId set (for disambiguation)
        for def in &x_defs {
            assert!(
                def.local_id.is_some(),
                "LocalId should be set for local binding"
            );
        }

        // The LocalIds should be different
        assert_ne!(
            x_defs[0].local_id, x_defs[1].local_id,
            "Shadowed variables should have different LocalIds"
        );
    }

    #[test]
    fn test_pattern_references_with_shadowing() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"fn main() -> Int {
    let x = 1
    let y = {
        let x = 2
        x
    }
    x + y
}"#,
        );

        let index = definition_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();
        let x_sym = trunk_ir::Symbol::new("x");

        // Get all references to 'x'
        let all_refs = index.references_of(&db, x_sym);
        assert_eq!(all_refs.len(), 2, "Expected 2 references to 'x'");

        // Get the two different definitions
        let x_defs: Vec<_> = index
            .definitions(&db)
            .iter()
            .filter(|d| d.name == x_sym)
            .collect();
        assert_eq!(x_defs.len(), 2);

        // Each definition should have exactly one reference when using precise matching
        for def in &x_defs {
            if let Some(local_id) = def.local_id {
                let target = ResolvedTarget::Local {
                    id: local_id,
                    name: x_sym,
                };
                let refs = index.references_of_target(&db, &target);
                assert_eq!(
                    refs.len(),
                    1,
                    "Each shadowed variable should have exactly 1 reference"
                );
            }
        }
    }

    #[test]
    fn test_pattern_list() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"fn main() -> Int {
    let [a, b, c] = [1, 2, 3]
    a + b + c
}"#,
        );

        let index = definition_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();

        // All list elements should be defined
        for name in ["a", "b", "c"] {
            let def = index.definition_of(&db, trunk_ir::Symbol::new(name));
            assert!(def.is_some(), "Expected definition for '{}'", name);
            assert_eq!(def.unwrap().kind, DefinitionKind::Local);
        }
    }

    #[test]
    fn test_pattern_multiple_case_arms() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"fn classify(n: Int) -> Int {
    case n {
        0 -> 0
        1 -> 1
        x -> x * 2
    }
}"#,
        );

        let index = definition_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();

        // 'x' should be defined from the third arm
        let x_def = index.definition_of(&db, trunk_ir::Symbol::new("x"));
        assert!(x_def.is_some());
        assert_eq!(x_def.unwrap().kind, DefinitionKind::Local);

        // 'n' should be a parameter
        let n_def = index.definition_of(&db, trunk_ir::Symbol::new("n"));
        assert!(n_def.is_some());
        assert_eq!(n_def.unwrap().kind, DefinitionKind::Parameter);
    }

    // =========================================================================
    // Type Collection Tests
    // =========================================================================

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

    #[test]
    fn test_definition_of_target_local() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"fn main() -> Int {
    let x = 1
    x
}"#,
        );

        let index = definition_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();

        // Get a reference to x
        let refs = index.references_of(&db, trunk_ir::Symbol::new("x"));
        assert!(!refs.is_empty());

        // Use definition_of_target with the reference target
        let def = index.definition_of_target(&db, &refs[0].target);
        assert!(def.is_some());
        assert_eq!(def.unwrap().kind, DefinitionKind::Local);
    }

    #[test]
    fn test_definition_of_target_function() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"fn helper() -> Int { 42 }

fn main() -> Int {
    helper()
}"#,
        );

        let index = definition_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();

        // Get reference to helper function
        let refs = index.references_of(&db, trunk_ir::Symbol::new("helper"));
        assert!(!refs.is_empty());

        // Use definition_of_target
        let def = index.definition_of_target(&db, &refs[0].target);
        assert!(def.is_some());
        assert_eq!(def.unwrap().kind, DefinitionKind::Function);
    }

    // =========================================================================
    // Rune Literal Type Tests
    // =========================================================================

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

    // =========================================================================
    // Shorthand Record Pattern Field Tracking Tests
    // =========================================================================

    #[test]
    fn test_pattern_record_shorthand_fields_have_definitions() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"struct Point { x: Int, y: Int }

fn use_point(p: Point) -> Int {
    let Point { x, y } = p
    x + y
}"#,
        );

        let index = definition_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();

        // Both x and y should be defined as locals from the shorthand pattern
        let x_def = index.definition_of(&db, trunk_ir::Symbol::new("x"));
        assert!(x_def.is_some(), "Expected definition for 'x'");
        // x appears as both a field and a local binding; check we have at least one
        let x_defs: Vec<_> = index
            .definitions(&db)
            .iter()
            .filter(|d| d.name == trunk_ir::Symbol::new("x"))
            .collect();
        assert!(
            x_defs.len() >= 2,
            "Expected at least 2 definitions for 'x' (field + local), got {}",
            x_defs.len()
        );

        let y_def = index.definition_of(&db, trunk_ir::Symbol::new("y"));
        assert!(y_def.is_some(), "Expected definition for 'y'");
    }

    #[test]
    fn test_pattern_record_shorthand_fields_have_distinct_spans() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"struct Point { x: Int, y: Int }

fn f(p: Point) -> Int {
    let Point { x, y } = p
    x + y
}"#,
        );

        let index = definition_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();

        // Find the local definitions for x and y (from the shorthand pattern)
        let x_locals: Vec<_> = index
            .definitions(&db)
            .iter()
            .filter(|d| d.name == trunk_ir::Symbol::new("x") && d.kind == DefinitionKind::Local)
            .collect();
        let y_locals: Vec<_> = index
            .definitions(&db)
            .iter()
            .filter(|d| d.name == trunk_ir::Symbol::new("y") && d.kind == DefinitionKind::Local)
            .collect();

        assert!(!x_locals.is_empty(), "Expected local definition for 'x'");
        assert!(!y_locals.is_empty(), "Expected local definition for 'y'");

        // The key test: x and y should have different spans (not sharing parent pattern's span)
        let x_span = x_locals[0].span;
        let y_span = y_locals[0].span;
        assert_ne!(
            x_span, y_span,
            "Shorthand fields x and y should have different spans, but both have {:?}",
            x_span
        );
    }

    // =========================================================================
    // EnumVariant DefinitionKind Tests
    // =========================================================================

    #[test]
    fn test_enum_variants_have_enum_variant_kind() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "enum Color { Red, Green, Blue }");

        let index = definition_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();

        // The enum itself should have Enum kind
        let color_def = index.definition_of(&db, trunk_ir::Symbol::new("Color"));
        assert!(color_def.is_some());
        assert_eq!(color_def.unwrap().kind, DefinitionKind::Enum);

        // Variants should have EnumVariant kind with owner, not Field
        let red_def = index.definition_of(&db, trunk_ir::Symbol::new("Red"));
        assert!(red_def.is_some());
        assert!(matches!(
            &red_def.unwrap().kind,
            DefinitionKind::EnumVariant { owner } if *owner == "Color"
        ));

        let green_def = index.definition_of(&db, trunk_ir::Symbol::new("Green"));
        assert!(green_def.is_some());
        assert!(matches!(
            &green_def.unwrap().kind,
            DefinitionKind::EnumVariant { owner } if *owner == "Color"
        ));

        let blue_def = index.definition_of(&db, trunk_ir::Symbol::new("Blue"));
        assert!(blue_def.is_some());
        assert!(matches!(
            &blue_def.unwrap().kind,
            DefinitionKind::EnumVariant { owner } if *owner == "Color"
        ));
    }

    #[test]
    fn test_struct_fields_have_field_kind() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "struct Point { x: Int, y: Int }");

        let index = definition_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();

        // Struct fields should still have Field kind
        let x_def = index.definition_of(&db, trunk_ir::Symbol::new("x"));
        assert!(x_def.is_some());
        assert_eq!(x_def.unwrap().kind, DefinitionKind::Field);

        let y_def = index.definition_of(&db, trunk_ir::Symbol::new("y"));
        assert!(y_def.is_some());
        assert_eq!(y_def.unwrap().kind, DefinitionKind::Field);
    }

    #[test]
    fn test_validate_identifier_enum_variant_uppercase() {
        // EnumVariant should require uppercase (like Struct, Enum, Ability)
        let variant_kind = DefinitionKind::EnumVariant {
            owner: trunk_ir::Symbol::new("TestEnum"),
        };
        assert!(validate_identifier("Red", variant_kind.clone()).is_ok());
        assert!(validate_identifier("SomeValue", variant_kind.clone()).is_ok());

        // Lowercase should be rejected for EnumVariant
        let result = validate_identifier("red", variant_kind);
        assert!(matches!(result, Err(RenameError::InvalidTypeIdentifier)));
    }

    #[test]
    fn test_validate_identifier_field_lowercase() {
        // Field (struct field) should require lowercase
        assert!(validate_identifier("x", DefinitionKind::Field).is_ok());
        assert!(validate_identifier("my_field", DefinitionKind::Field).is_ok());

        // Uppercase should be rejected for Field
        let result = validate_identifier("X", DefinitionKind::Field);
        assert!(matches!(result, Err(RenameError::InvalidIdentifier)));
    }

    #[test]
    fn test_enum_variant_owner_tracking() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(
            &db,
            r#"
enum Option(T) {
    Some(T),
    None
}
enum Result(T, E) {
    Ok(T),
    Err(E)
}
"#,
        );

        let index = definition_index(&db, source);
        assert!(index.is_some());
        let index = index.unwrap();

        // Some's owner should be Option
        let some_def = index.definition_of(&db, trunk_ir::Symbol::new("Some"));
        assert!(some_def.is_some());
        assert!(matches!(
            &some_def.unwrap().kind,
            DefinitionKind::EnumVariant { owner } if *owner == "Option"
        ));

        // None's owner should be Option
        let none_def = index.definition_of(&db, trunk_ir::Symbol::new("None"));
        assert!(none_def.is_some());
        assert!(matches!(
            &none_def.unwrap().kind,
            DefinitionKind::EnumVariant { owner } if *owner == "Option"
        ));

        // Ok's owner should be Result
        let ok_def = index.definition_of(&db, trunk_ir::Symbol::new("Ok"));
        assert!(ok_def.is_some());
        assert!(matches!(
            &ok_def.unwrap().kind,
            DefinitionKind::EnumVariant { owner } if *owner == "Result"
        ));

        // Err's owner should be Result
        let err_def = index.definition_of(&db, trunk_ir::Symbol::new("Err"));
        assert!(err_def.is_some());
        assert!(matches!(
            &err_def.unwrap().kind,
            DefinitionKind::EnumVariant { owner } if *owner == "Result"
        ));
    }
}
