//! Definition index for go-to-definition, find-references, and rename.
//!
//! Maps names to definitions and references, enabling navigation and
//! refactoring LSP features.

use std::collections::BTreeMap;

use trunk_ir::{Span, Symbol};

use tribute_front::SourceCst;
use tribute_front::ast::{
    AbilityDecl, Arm, Decl, EnumDecl, Expr, ExprKind, FuncDecl, HandlerArm, HandlerKind, LocalId,
    Module, NodeId, ParamDecl, Pattern, PatternKind, ResolvedRef, SpanMap, Stmt, StructDecl,
    TypedRef,
};
use tribute_front::query as ast_query;

use super::completion_index::KEYWORDS;

// =============================================================================
// Definition Index
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
    /// A struct field.
    Field {
        /// The owning struct type name.
        owner: Symbol,
    },
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
    /// Reference to a struct field.
    Field { owner: Symbol, name: Symbol },
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
            ResolvedTarget::Field { name, .. } => *name,
            ResolvedTarget::Other { name } => *name,
        }
    }

    /// Get the LocalId if this is a local reference.
    pub fn local_id(&self) -> Option<LocalId> {
        match self {
            ResolvedTarget::Local { id, .. } => Some(*id),
            ResolvedTarget::Field { .. }
            | ResolvedTarget::Function { .. }
            | ResolvedTarget::Constructor { .. }
            | ResolvedTarget::Type { .. }
            | ResolvedTarget::Ability { .. }
            | ResolvedTarget::Unresolved { .. }
            | ResolvedTarget::Other { .. } => None,
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
    /// Returns the resolved target and all references to it.
    /// Uses LocalId-based matching for precise shadowed variable disambiguation.
    pub fn references_at(
        &self,
        db: &'db dyn salsa::Database,
        offset: usize,
    ) -> Option<(ResolvedTarget, Vec<&AstReferenceEntry>)> {
        // Try reference first (more specific than definitions which may span large areas)
        if let Some(reference) = self.reference_at(db, offset) {
            let target = reference.target.clone();
            let refs = self.references_of_target(db, &target);
            return Some((target, refs));
        }

        // Fall back to definition
        if let Some(def) = self.definition_at_position(db, offset) {
            let target = self.target_from_definition(def);
            let refs = self.references_of_target(db, &target);
            return Some((target, refs));
        }

        None
    }

    /// Create a ResolvedTarget from a definition entry.
    ///
    /// This is useful for converting a definition lookup result into a target
    /// that can be used with `references_of_target` for precise matching.
    pub fn target_from_definition(&self, def: &AstDefinitionEntry) -> ResolvedTarget {
        match &def.kind {
            DefinitionKind::Local | DefinitionKind::Parameter => {
                if let Some(local_id) = def.local_id {
                    ResolvedTarget::Local {
                        id: local_id,
                        name: def.name,
                    }
                } else {
                    // Fallback for locals without LocalId
                    ResolvedTarget::Unresolved { name: def.name }
                }
            }
            DefinitionKind::Function => ResolvedTarget::Function { name: def.name },
            DefinitionKind::Struct | DefinitionKind::Enum => {
                ResolvedTarget::Type { name: def.name }
            }
            DefinitionKind::EnumVariant { owner } => ResolvedTarget::Constructor {
                type_name: *owner,
                variant: def.name,
            },
            DefinitionKind::Ability => ResolvedTarget::Ability { name: def.name },
            DefinitionKind::Field { owner } => ResolvedTarget::Field {
                owner: *owner,
                name: def.name,
            },
        }
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
        self.add_definition(
            param.id,
            param.name,
            DefinitionKind::Parameter,
            param.local_id,
        );
    }

    fn collect_struct(&mut self, s: &StructDecl) {
        self.add_definition(s.id, s.name, DefinitionKind::Struct, None);

        // Add field definitions
        for field in &s.fields {
            if let Some(name) = field.name {
                self.add_definition(
                    field.id,
                    name,
                    DefinitionKind::Field { owner: s.name },
                    None,
                );
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
                // Lambda params have local_id assigned during name resolution
                for param in params {
                    self.add_definition(
                        param.id,
                        param.name,
                        DefinitionKind::Parameter,
                        param.local_id,
                    );
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
        DefinitionKind::Function
        | DefinitionKind::Local
        | DefinitionKind::Parameter
        | DefinitionKind::Field { .. } => {
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
        assert!(matches!(x_def.unwrap().kind, DefinitionKind::Field { .. }));
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

    // Pattern Collection Tests

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

    // Validate identifier tests

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

    #[test]
    fn test_definition_of_target_local() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn foo() { let x = 1; x }");

        let index = definition_index(&db, source);
        assert!(index.is_some());
        let index = index.unwrap();

        let x_def = index.definition_of(&db, trunk_ir::Symbol::new("x"));
        assert!(x_def.is_some());

        let target = index.target_from_definition(x_def.unwrap());
        assert!(
            matches!(target, ResolvedTarget::Local { .. }),
            "Local definition should produce Local target"
        );
    }

    #[test]
    fn test_definition_of_target_function() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn foo() { 1 }");

        let index = definition_index(&db, source);
        assert!(index.is_some());
        let index = index.unwrap();

        let foo_def = index.definition_of(&db, trunk_ir::Symbol::new("foo"));
        assert!(foo_def.is_some());

        let target = index.target_from_definition(foo_def.unwrap());
        assert!(
            matches!(target, ResolvedTarget::Function { name } if name == trunk_ir::Symbol::new("foo")),
            "Function definition should produce Function target"
        );
    }

    #[test]
    fn test_definition_of_target_enum_variant() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "enum Option { Some(a), None }");

        let index = definition_index(&db, source);
        assert!(index.is_some());
        let index = index.unwrap();

        let some_def = index.definition_of(&db, trunk_ir::Symbol::new("Some"));
        assert!(some_def.is_some());

        let target = index.target_from_definition(some_def.unwrap());
        assert!(
            matches!(target, ResolvedTarget::Constructor { type_name, variant }
                if type_name == trunk_ir::Symbol::new("Option")
                && variant == trunk_ir::Symbol::new("Some")),
            "EnumVariant definition should produce Constructor target"
        );
    }

    // Shorthand Record Pattern Field Tracking Tests

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

    // EnumVariant DefinitionKind Tests

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

        // Struct fields should still have Field kind with owner
        let x_def = index.definition_of(&db, trunk_ir::Symbol::new("x"));
        assert!(x_def.is_some());
        assert!(matches!(
            &x_def.unwrap().kind,
            DefinitionKind::Field { owner } if *owner == "Point"
        ));

        let y_def = index.definition_of(&db, trunk_ir::Symbol::new("y"));
        assert!(y_def.is_some());
        assert!(matches!(
            &y_def.unwrap().kind,
            DefinitionKind::Field { owner } if *owner == "Point"
        ));
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
        let field_kind = DefinitionKind::Field {
            owner: trunk_ir::Symbol::new("TestStruct"),
        };
        assert!(validate_identifier("x", field_kind.clone()).is_ok());
        assert!(validate_identifier("my_field", field_kind.clone()).is_ok());

        // Uppercase should be rejected for Field
        let result = validate_identifier("X", field_kind);
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

    // Parameter LocalId tests

    #[test]
    fn test_parameter_has_local_id() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn add(x: Int, y: Int) -> Int { x + y }");

        let index = definition_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();

        // Parameters should be registered as definitions with LocalIds
        let x_def = index.definition_of(&db, trunk_ir::Symbol::new("x"));
        assert!(x_def.is_some(), "Parameter 'x' should be in definitions");
        assert_eq!(x_def.unwrap().kind, DefinitionKind::Parameter);

        let y_def = index.definition_of(&db, trunk_ir::Symbol::new("y"));
        assert!(y_def.is_some(), "Parameter 'y' should be in definitions");
        assert_eq!(y_def.unwrap().kind, DefinitionKind::Parameter);

        // LocalIds should be set (not None)
        assert!(
            x_def.unwrap().local_id.is_some(),
            "Parameter 'x' should have a LocalId"
        );
        assert!(
            y_def.unwrap().local_id.is_some(),
            "Parameter 'y' should have a LocalId"
        );
    }

    #[test]
    fn test_parameter_reference_matches_definition() {
        let db = salsa::DatabaseImpl::default();
        // Simple function that uses its parameter
        let source = make_source(&db, "fn identity(x: Int) -> Int { x }");

        let index = definition_index(&db, source);
        assert!(index.is_some());

        let index = index.unwrap();

        // Get the parameter definition
        let x_def = index.definition_of(&db, trunk_ir::Symbol::new("x"));
        assert!(x_def.is_some());
        let x_local_id = x_def.unwrap().local_id;
        assert!(x_local_id.is_some(), "Parameter should have LocalId");

        // Get references to x (the one in the body)
        let x_refs = index.references_of(&db, trunk_ir::Symbol::new("x"));
        assert_eq!(x_refs.len(), 1, "Should have 1 reference to 'x' in body");

        // The reference's target LocalId should match the definition's LocalId
        let ref_target = &x_refs[0].target;
        if let ResolvedTarget::Local { id: local_id, .. } = ref_target {
            assert_eq!(
                Some(*local_id),
                x_local_id,
                "Reference LocalId should match definition LocalId"
            );
        } else {
            panic!("Expected local reference target");
        }
    }

    // Target-aware reference tests (shadowing)

    #[test]
    fn test_references_at_returns_target() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn foo(x: Int) -> Int { x }");

        let index = definition_index(&db, source);
        assert!(index.is_some());
        let index = index.unwrap();

        // Find references from the 'x' in body (position ~24)
        let result = index.references_at(&db, 24);
        assert!(result.is_some(), "Should find references at position");

        let (target, refs) = result.unwrap();

        // Target should be a Local with the correct name
        assert!(
            matches!(&target, ResolvedTarget::Local { name, .. } if *name == trunk_ir::Symbol::new("x")),
            "Target should be Local with name 'x'"
        );

        // Should have exactly 1 reference
        assert_eq!(refs.len(), 1, "Should have 1 reference to 'x'");
    }

    #[test]
    fn test_target_from_definition_local() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn foo() { let x = 1; x }");

        let index = definition_index(&db, source);
        assert!(index.is_some());
        let index = index.unwrap();

        let x_def = index.definition_of(&db, trunk_ir::Symbol::new("x"));
        assert!(x_def.is_some());

        let target = index.target_from_definition(x_def.unwrap());
        assert!(
            matches!(target, ResolvedTarget::Local { .. }),
            "Local definition should produce Local target"
        );
    }

    #[test]
    fn test_target_from_definition_function() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "fn foo() { 1 }");

        let index = definition_index(&db, source);
        assert!(index.is_some());
        let index = index.unwrap();

        let foo_def = index.definition_of(&db, trunk_ir::Symbol::new("foo"));
        assert!(foo_def.is_some());

        let target = index.target_from_definition(foo_def.unwrap());
        assert!(
            matches!(target, ResolvedTarget::Function { name } if name == trunk_ir::Symbol::new("foo")),
            "Function definition should produce Function target"
        );
    }

    #[test]
    fn test_target_from_definition_enum_variant() {
        let db = salsa::DatabaseImpl::default();
        let source = make_source(&db, "enum Option { Some(a), None }");

        let index = definition_index(&db, source);
        assert!(index.is_some());
        let index = index.unwrap();

        let some_def = index.definition_of(&db, trunk_ir::Symbol::new("Some"));
        assert!(some_def.is_some());

        let target = index.target_from_definition(some_def.unwrap());
        assert!(
            matches!(target, ResolvedTarget::Constructor { type_name, variant }
                if type_name == trunk_ir::Symbol::new("Option")
                && variant == trunk_ir::Symbol::new("Some")),
            "EnumVariant definition should produce Constructor target"
        );
    }
}
