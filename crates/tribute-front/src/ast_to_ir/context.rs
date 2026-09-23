//! IR lowering context.
//!
//! Manages state during AST-to-IR transformation.
//! Emits arena IR (`IrContext` / `TypeRef` / `ValueRef`) directly.

use std::collections::{HashMap, HashSet};
use std::ops::{Deref, DerefMut};

use tribute_ir::dialect::tribute_rt;
use trunk_ir::Symbol;
use trunk_ir::SymbolVec;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::refs::{BlockRef, PathRef, TypeRef, ValueRef};
use trunk_ir::types::{Attribute, Location, TypeDataBuilder};

use crate::ast::{
    AbilityId, CallingConvention, CtorId, LocalId, NodeId, SpanMap, TypeKind, TypeScheme,
};

/// Encode a tagged type-shape node without relying on separator characters in
/// source symbols. Each component carries its byte length, so nested keys are
/// readable in printed IR while still being uniquely decodable.
fn logical_key<I, S>(tag: &str, parts: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let parts = parts.into_iter().collect::<Vec<_>>();
    let mut key = format!("{tag}_{}", parts.len());
    for part in parts {
        let part = part.as_ref();
        key.push('_');
        key.push_str(&part.len().to_string());
        key.push('_');
        key.push_str(part);
    }
    key
}

fn logical_convention_key(convention: CallingConvention) -> String {
    match convention {
        CallingConvention::Direct => "direct".into(),
        CallingConvention::EvidenceDirect => "evidence_direct".into(),
        CallingConvention::Cps => "cps".into(),
    }
}

/// A source-visible callable synthesized by the logical frontend, such as a
/// struct-field accessor. These signatures are semantic lowering metadata,
/// not reconstructed by inspecting emitted operations.
#[derive(Clone)]
pub(crate) struct LogicalGeneratedSignature {
    pub param_types: Vec<TypeRef>,
    pub return_type: TypeRef,
    pub convention: CallingConvention,
}

/// Context for lowering AST to arena TrunkIR.
pub struct IrLoweringCtx<'db> {
    pub db: &'db dyn salsa::Database,
    pub path: PathRef,
    /// Span map for looking up source locations.
    span_map: SpanMap,
    /// Stack of scopes, each mapping LocalId to (name, SSA value).
    scopes: Vec<HashMap<LocalId, (Symbol, ValueRef)>>,
    local_callable_values: Vec<HashMap<(NodeId, crate::ast::Type<'db>, TypeRef), ValueRef>>,
    /// Scoped tags identifying locals whose SSA value is a suspended handler
    /// continuation rather than a source value.
    resume_scopes: Vec<HashSet<LocalId>>,
    /// Function type schemes from type checking, keyed by function name.
    function_types: HashMap<Symbol, TypeScheme<'db>>,
    /// Source-visible generated callables that have no source TypeScheme.
    logical_generated_signatures: HashMap<Symbol, LogicalGeneratedSignature>,
    /// Source declarations collected before logical lowering begins.
    logical_source_functions: HashSet<Symbol>,
    /// Extern declarations synthesized for referenced prelude functions.
    logical_emitted_externs: HashSet<Symbol>,
    /// Ability-level calling-convention requirements.
    ability_conventions: HashMap<AbilityId<'db>, CallingConvention>,
    /// Physical worker conventions for named function definitions.
    ///
    /// These are intentionally separate from semantic function-type
    /// conventions: an effect-polymorphic pure definition may have a Direct
    /// worker while its first-class function type requires a CPS adapter.
    definition_conventions: HashMap<Symbol, CallingConvention>,
    /// Module path as a vector of segments (e.g., ["std", "Option"]).
    module_path: SymbolVec,
    /// Module's top-level block, used for in-place insertion of lifted lambdas.
    module_block: Option<BlockRef>,
    /// Struct field order: CtorId → [field_names in definition order].
    /// Used for lowering Record expressions to adt.struct_new.
    struct_fields: HashMap<CtorId<'db>, Vec<Symbol>>,
    /// Type map: type name → arena TypeRef for adt.struct / adt.enum.
    /// Used for named structs, tuples, and (future) enum variants.
    type_map: im::HashMap<Symbol, TypeRef>,
    /// All source nominal identities collected before source-logical layouts
    /// are built. This lets recursive and forward fields retain `adt.typeref`
    /// while their layout is still incomplete.
    logical_nominal_declarations: HashSet<Symbol>,
    /// Exact intrinsic-directive declaration ID to canonical identity.
    compiler_intrinsics: HashMap<NodeId, Symbol>,
    /// Node types from type checking, keyed by NodeId.
    /// Used to get the effect type of lambda expressions.
    node_types: HashMap<NodeId, crate::ast::Type<'db>>,
}

impl<'db> IrLoweringCtx<'db> {
    pub(crate) fn db(&self) -> &'db dyn salsa::Database {
        self.db
    }

    /// Create a new IR lowering context.
    pub fn new(
        db: &'db dyn salsa::Database,
        path: PathRef,
        span_map: SpanMap,
        function_types: HashMap<Symbol, TypeScheme<'db>>,
        ability_conventions: HashMap<AbilityId<'db>, CallingConvention>,
        module_path: SymbolVec,
        node_types: HashMap<NodeId, crate::ast::Type<'db>>,
    ) -> Self {
        Self {
            db,
            path,
            span_map,
            scopes: vec![HashMap::new()],
            local_callable_values: vec![HashMap::new()],
            resume_scopes: vec![HashSet::new()],
            function_types,
            logical_generated_signatures: HashMap::new(),
            logical_source_functions: HashSet::new(),
            logical_emitted_externs: HashSet::new(),
            ability_conventions,
            definition_conventions: HashMap::new(),
            module_path,
            module_block: None,
            struct_fields: HashMap::new(),
            type_map: im::HashMap::new(),
            logical_nominal_declarations: HashSet::new(),
            compiler_intrinsics: HashMap::new(),

            node_types,
        }
    }

    pub(crate) fn with_compiler_intrinsics(
        mut self,
        compiler_intrinsics: HashMap<NodeId, Symbol>,
    ) -> Self {
        self.compiler_intrinsics = compiler_intrinsics;
        self
    }

    pub(crate) fn compiler_intrinsic(&self, declaration: NodeId) -> Option<Symbol> {
        self.compiler_intrinsics.get(&declaration).copied()
    }

    /// Get the current module path.
    pub fn module_path(&self) -> &SymbolVec {
        &self.module_path
    }

    /// Enter a nested module, updating the module path.
    pub fn enter_module(&mut self, name: Symbol) {
        self.module_path.push(name);
    }

    /// Exit a nested module, restoring the parent module path.
    pub fn exit_module(&mut self) {
        self.module_path.pop();
    }

    /// Set the module-level block for in-place insertion of lifted functions.
    pub fn set_module_block(&mut self, block: BlockRef) {
        self.module_block = Some(block);
    }

    /// Get the module-level block.
    pub fn module_block(&self) -> Option<BlockRef> {
        self.module_block
    }

    /// Create an arena Location for a node.
    pub fn location(&self, node_id: NodeId) -> Location {
        let span = self.span_map.get_or_default(node_id);
        Location::new(self.path, span)
    }

    /// Enter a new scope, returning a guard that exits on drop.
    ///
    /// The guard dereferences to `IrLoweringCtx`, so callers can use it
    /// in place of `self`/`ctx`. The scope is automatically exited when
    /// the guard is dropped, even on early returns or `?`.
    #[must_use]
    pub fn scope(&mut self) -> ScopeGuard<'_, 'db> {
        self.enter_scope();
        ScopeGuard { ctx: self }
    }

    /// Enter a new scope (internal — use `scope()` guard instead).
    fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
        self.local_callable_values.push(HashMap::new());
        self.resume_scopes.push(HashSet::new());
    }

    /// Exit the current scope (internal — use `scope()` guard instead).
    fn exit_scope(&mut self) {
        self.scopes.pop();
        self.local_callable_values.pop();
        self.resume_scopes.pop();
    }

    /// Bind a local variable to an SSA value.
    pub fn bind(&mut self, local_id: LocalId, name: Symbol, value: ValueRef) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(local_id, (name, value));
        }
    }

    pub(crate) fn bind_local_callable(
        &mut self,
        key: (NodeId, crate::ast::Type<'db>, TypeRef),
        value: ValueRef,
    ) {
        self.local_callable_values
            .last_mut()
            .expect("active scope")
            .insert(key, value);
    }

    pub(crate) fn lookup_local_callable(
        &self,
        key: (NodeId, crate::ast::Type<'db>, TypeRef),
    ) -> Option<ValueRef> {
        self.local_callable_values
            .iter()
            .rev()
            .find_map(|scope| scope.get(&key).copied())
    }

    pub(crate) fn bind_resume(&mut self, local_id: LocalId, name: Symbol, value: ValueRef) {
        self.bind(local_id, name, value);
        if let Some(scope) = self.resume_scopes.last_mut() {
            scope.insert(local_id);
        }
    }

    pub(crate) fn lookup_resume(&self, local_id: LocalId) -> Option<ValueRef> {
        self.resume_scopes
            .iter()
            .rev()
            .any(|scope| scope.contains(&local_id))
            .then(|| self.lookup(local_id))
            .flatten()
    }

    /// Look up a function's type scheme by name.
    pub fn lookup_function_type(&self, name: Symbol) -> Option<&TypeScheme<'db>> {
        self.function_types.get(&name)
    }

    pub(crate) fn register_logical_generated_signature(
        &mut self,
        name: Symbol,
        param_types: Vec<TypeRef>,
        return_type: TypeRef,
        convention: CallingConvention,
    ) {
        let signature = LogicalGeneratedSignature {
            param_types,
            return_type,
            convention,
        };
        if let Some(existing) = self
            .logical_generated_signatures
            .insert(name, signature.clone())
        {
            assert!(
                existing.param_types == signature.param_types
                    && existing.return_type == signature.return_type
                    && existing.convention == signature.convention,
                "conflicting logical generated signature for {name}"
            );
        }
    }

    pub(crate) fn lookup_logical_generated_signature(
        &self,
        name: Symbol,
    ) -> Option<&LogicalGeneratedSignature> {
        self.logical_generated_signatures.get(&name)
    }

    pub(crate) fn register_logical_source_function(&mut self, name: Symbol) {
        self.logical_source_functions.insert(name);
    }

    pub(crate) fn is_logical_source_function(&self, name: Symbol) -> bool {
        self.logical_source_functions.contains(&name)
    }

    /// Returns true exactly once for each prelude declaration that must be
    /// materialized in the logical module.
    pub(crate) fn mark_logical_extern_emitted(&mut self, name: Symbol) -> bool {
        self.logical_emitted_externs.insert(name)
    }

    /// Derive a function convention from both its effect row and ABI lower bound.
    pub(crate) fn calling_convention_for_type(
        &self,
        ty: crate::ast::Type<'db>,
    ) -> Option<CallingConvention> {
        crate::ast::calling_convention_for_function_type(self.db, ty, &self.ability_conventions)
    }

    /// Derive a convention from an effect row without a function-level ABI bound.
    pub(crate) fn calling_convention_for_effect_row(
        &self,
        effect: crate::ast::EffectRow<'db>,
    ) -> CallingConvention {
        crate::ast::calling_convention_for_effect_row(self.db, effect, &self.ability_conventions)
    }

    /// Look up a function definition and derive its ABI convention.
    pub(crate) fn function_calling_convention(&self, name: Symbol) -> Option<CallingConvention> {
        if let Some(convention) = self.definition_conventions.get(&name) {
            return Some(*convention);
        }
        let scheme = self.lookup_function_type(name)?;
        let body = scheme.body(self.db);
        self.calling_convention_for_type(body)
    }

    /// Register the physical worker convention for a named definition.
    pub(crate) fn register_definition_convention(
        &mut self,
        name: Symbol,
        convention: CallingConvention,
    ) {
        self.definition_conventions.insert(name, convention);
    }

    /// Look up a local variable.
    pub fn lookup(&self, local_id: LocalId) -> Option<ValueRef> {
        for scope in self.scopes.iter().rev() {
            if let Some((_, value)) = scope.get(&local_id) {
                return Some(*value);
            }
        }
        None
    }

    /// Qualify a name with the current module path.
    ///
    /// Matches resolve::build_env convention: the top-level module name
    /// (derived from filename) is NOT part of internal qualified names.
    /// Only nested `pub mod` blocks contribute to the path.
    ///
    /// - `["test", "String"]` + `"to_bytes"` → `"String::to_bytes"`
    /// - `["test"]` + `"print_line"` → `"print_line"` (top-level, unchanged)
    /// - `"Nested::Box::value"` → `"Nested::Box::value"` (already qualified,
    ///   returned unchanged)
    pub fn qualify_name(&self, name: Symbol) -> Symbol {
        // Synthetic monomorphized declarations already carry their resolved
        // source path. Adding the current module again would change the
        // identity used by convention lookup and logical declaration emission.
        if name.with_str(|text| text.contains("::")) {
            return name;
        }
        // Skip the first segment (top-level module name from filename).
        // Only nested module segments contribute to the qualified name.
        let nested: Vec<_> = self.module_path.iter().skip(1).collect();
        if nested.is_empty() {
            return name;
        }
        let mut prefix = nested
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("::");
        prefix.push_str("::");
        name.with_str(|s| prefix.push_str(s));
        Symbol::from_dynamic(&prefix)
    }

    /// Register struct field order for lowering Record expressions.
    pub fn register_struct_fields(&mut self, ctor_id: CtorId<'db>, field_names: Vec<Symbol>) {
        self.struct_fields.insert(ctor_id, field_names);
    }

    /// Get struct field order (for lowering Record → adt.struct_new).
    pub fn get_struct_field_order(&self, ctor_id: CtorId<'db>) -> Option<&Vec<Symbol>> {
        self.struct_fields.get(&ctor_id)
    }

    /// Register a type (named struct, tuple, etc.) in the type map.
    pub fn register_type(&mut self, name: Symbol, ty: TypeRef) {
        self.type_map.insert(name, ty);
    }

    /// Get a registered type by name.
    pub fn get_type(&self, name: Symbol) -> Option<TypeRef> {
        self.type_map.get(&name).copied()
    }

    /// Mark a nominal source type as known before its layout is complete.
    pub fn declare_logical_nominal(&mut self, name: Symbol) {
        self.logical_nominal_declarations.insert(name);
    }

    /// Get the type of an AST node by NodeId.
    ///
    /// Returns the type assigned during type checking.
    /// Used to get the effect type of lambda expressions.
    pub fn resolve_adt_type(&self, ty: crate::ast::Type<'db>) -> Option<TypeRef> {
        match ty.kind(self.db) {
            TypeKind::Named { name, .. } => self.get_type(*name),
            _ => None,
        }
    }

    pub fn get_node_type(&self, node_id: NodeId) -> Option<&crate::ast::Type<'db>> {
        self.node_types.get(&node_id)
    }

    /// Get all bindings visible in the current scope (for capture analysis).
    ///
    /// The stable local-ID order keeps generated closure capture lists and IR
    /// snapshots deterministic; each local ID is unique even across scopes.
    pub fn all_bindings(&self) -> impl Iterator<Item = (LocalId, Symbol, ValueRef)> + '_ {
        let mut bindings: Vec<_> = self
            .scopes
            .iter()
            .rev()
            .flat_map(|scope| scope.iter().map(|(&id, &(name, value))| (id, name, value)))
            .collect();
        bindings.sort_unstable_by_key(|(local_id, _, _)| local_id.raw());
        bindings.into_iter()
    }

    // =========================================================================
    // Arena type conversion
    // =========================================================================

    /// Convert a source type at the source-logical `tribute_control` boundary.
    pub fn convert_logical_type(&self, ir: &mut IrContext, ty: crate::ast::Type<'db>) -> TypeRef {
        match ty.kind(self.db) {
            TypeKind::Int | TypeKind::Nat | TypeKind::Rune => self.i32_type(ir),
            TypeKind::Float => self.f64_type(ir),
            TypeKind::Bool => self.bool_type(ir),
            TypeKind::Bytes => self.bytes_type(ir),
            TypeKind::Nil | TypeKind::Error => self.nil_type(ir),
            TypeKind::Never => core::never(ir).as_type_ref(),
            TypeKind::BoundVar { .. }
            | TypeKind::LocalBoundVar { .. }
            | TypeKind::UniVar { .. } => self.anyref_type(ir),
            TypeKind::Named { id, .. } if id.is_builtin_list(self.db) => self.anyref_type(ir),
            TypeKind::Named { id, .. }
                if self.get_type(id.qualified(self.db)).is_some()
                    || self
                        .logical_nominal_declarations
                        .contains(&id.qualified(self.db)) =>
            {
                self.adt_typeref(ir, id.qualified(self.db))
            }
            TypeKind::Named { .. } => self.anyref_type(ir),
            TypeKind::Func { params, result, .. } => {
                let params = params
                    .iter()
                    .map(|param| self.convert_logical_type(ir, *param))
                    .collect::<Vec<_>>();
                let result = self.convert_logical_type(ir, *result);
                let convention = self
                    .calling_convention_for_type(ty)
                    .unwrap_or(crate::ast::CallingConvention::Cps);
                tribute_ir::dialect::tribute_control::func_sig(
                    ir,
                    result,
                    params,
                    match convention {
                        crate::ast::CallingConvention::Direct => {
                            tribute_ir::dialect::tribute_control::CallingConvention::Direct
                        }
                        crate::ast::CallingConvention::EvidenceDirect => {
                            tribute_ir::dialect::tribute_control::CallingConvention::EvidenceDirect
                        }
                        crate::ast::CallingConvention::Cps => {
                            tribute_ir::dialect::tribute_control::CallingConvention::Cps
                        }
                    },
                )
                .as_type_ref()
            }
            TypeKind::Tuple(elements) => {
                let fields = elements
                    .iter()
                    .enumerate()
                    .map(|(index, element)| {
                        (
                            Symbol::from_dynamic(&index.to_string()),
                            self.convert_logical_type(ir, *element),
                        )
                    })
                    .collect::<Vec<_>>();
                let name = self.logical_tuple_name(ty);
                let layout = self.adt_struct_type(ir, name, &fields);
                ir.register_type_alias(name, layout);
                self.adt_typeref(ir, name)
            }
            TypeKind::App { ctor, .. } => self.convert_logical_type(ir, *ctor),
            TypeKind::Continuation { arg, result, .. } => {
                let arg = self.convert_logical_type(ir, *arg);
                let result = self.convert_logical_type(ir, *result);
                ir.intern_type(
                    TypeDataBuilder::new(
                        Symbol::new("tribute_control"),
                        Symbol::new("resume_token"),
                    )
                    .param(arg)
                    .param(result)
                    .build(),
                )
            }
        }
    }

    /// Canonical nominal name for a source-logical tuple layout.  Do not use
    /// arena dialect type names here: distinct callable signatures all share
    /// the `tribute_control.func_sig` dialect name.
    pub fn logical_tuple_name(&self, ty: crate::ast::Type<'db>) -> Symbol {
        Symbol::from_dynamic(&format!("__logical_tuple_{}", self.logical_type_key(ty)))
    }

    fn logical_type_key(&self, ty: crate::ast::Type<'db>) -> String {
        match ty.kind(self.db) {
            TypeKind::Int => "int".into(),
            TypeKind::Nat => "nat".into(),
            TypeKind::Float => "float".into(),
            TypeKind::Bool => "bool".into(),
            TypeKind::Bytes => "bytes".into(),
            TypeKind::Rune => "rune".into(),
            TypeKind::Nil => "nil".into(),
            TypeKind::Never => "never".into(),
            TypeKind::Error => "error".into(),
            TypeKind::BoundVar { index } => logical_key("bound", [index.to_string()]),
            TypeKind::LocalBoundVar { scope, index } => {
                logical_key("local_bound", [scope.raw().to_string(), index.to_string()])
            }
            TypeKind::UniVar { id } => self.logical_univar_key(*id),
            TypeKind::Named { id, name, args } => {
                let mut parts = vec![self.logical_nominal_key(*id, *name)];
                parts.extend(args.iter().map(|arg| self.logical_type_key(*arg)));
                logical_key("named", parts)
            }
            TypeKind::Func { params, result, .. } => {
                let mut parts = vec![params.len().to_string()];
                parts.extend(params.iter().map(|param| self.logical_type_key(*param)));
                parts.push(self.logical_type_key(*result));
                parts.push(logical_convention_key(
                    self.calling_convention_for_type(ty)
                        .unwrap_or(crate::ast::CallingConvention::Cps),
                ));
                logical_key("callable", parts)
            }
            TypeKind::Tuple(elements) => {
                let mut parts = vec![elements.len().to_string()];
                parts.extend(
                    elements
                        .iter()
                        .map(|element| self.logical_type_key(*element)),
                );
                logical_key("tuple", parts)
            }
            TypeKind::App { ctor, args } => {
                let mut parts = vec![args.len().to_string(), self.logical_type_key(*ctor)];
                parts.extend(args.iter().map(|arg| self.logical_type_key(*arg)));
                logical_key("app", parts)
            }
            TypeKind::Continuation { arg, result, .. } => logical_key(
                "resume",
                [self.logical_type_key(*arg), self.logical_type_key(*result)],
            ),
        }
    }

    fn logical_nominal_key(&self, id: crate::ast::TypeDefId<'db>, name: Symbol) -> String {
        let origin = match id.origin(self.db) {
            crate::ast::TypeOrigin::Source(_) => "source".into(),
            crate::ast::TypeOrigin::Builtin(crate::ast::BuiltinType::List) => "builtin_list".into(),
            crate::ast::TypeOrigin::Synthetic => "synthetic".into(),
        };
        logical_key("nominal", [origin, name.with_str(str::to_owned)])
    }

    fn logical_univar_key(&self, id: crate::ast::UniVarId<'db>) -> String {
        use crate::ast::UniVarSource;

        let source = match id.source(self.db) {
            UniVarSource::FunctionLocal { func_id, index } => logical_key(
                "function_local",
                [
                    func_id.qualified(self.db).with_str(str::to_owned),
                    index.to_string(),
                ],
            ),
            UniVarSource::Anonymous(index) => logical_key("anonymous", [index.to_string()]),
            UniVarSource::Solver { index } => logical_key("solver", [index.to_string()]),
        };
        logical_key("univar", [source, id.index(self.db).to_string()])
    }

    // =========================================================================
    // Arena type helpers
    // =========================================================================

    /// Get the `core.i32` type.
    pub fn i32_type(&self, ir: &mut IrContext) -> TypeRef {
        ir.intern_type(TypeDataBuilder::new("core", "i32").build())
    }

    /// Get the `core.nil` type.
    pub fn nil_type(&self, ir: &mut IrContext) -> TypeRef {
        core::nil(ir).as_type_ref()
    }

    /// Get the `core.i1` (bool) type.
    pub fn bool_type(&self, ir: &mut IrContext) -> TypeRef {
        ir.intern_type(TypeDataBuilder::new("core", "i1").build())
    }

    /// Get the `core.f64` type.
    pub fn f64_type(&self, ir: &mut IrContext) -> TypeRef {
        ir.intern_type(TypeDataBuilder::new("core", "f64").build())
    }

    /// Get the `core.bytes` type.
    pub fn bytes_type(&self, ir: &mut IrContext) -> TypeRef {
        core::bytes(ir).as_type_ref()
    }

    /// Get the `tribute_rt.anyref` type.
    pub fn anyref_type(&self, ir: &mut IrContext) -> TypeRef {
        tribute_rt::anyref(ir).as_type_ref()
    }

    /// Create a `core.ability_ref` type.
    pub fn ability_ref_type(
        &self,
        ir: &mut IrContext,
        ability_name: Symbol,
        params: &[TypeRef],
    ) -> TypeRef {
        let mut builder = TypeDataBuilder::new(Symbol::new("core"), Symbol::new("ability_ref"))
            .attr("name", Attribute::Symbol(ability_name));
        for &p in params {
            builder = builder.param(p);
        }
        ir.intern_type(builder.build())
    }

    /// Create an `adt.struct` type with name and fields.
    pub fn adt_struct_type(
        &self,
        ir: &mut IrContext,
        name: Symbol,
        fields: &[(Symbol, TypeRef)],
    ) -> TypeRef {
        let fields_attr: Vec<Attribute> = fields
            .iter()
            .map(|(field_name, field_type)| {
                Attribute::List(vec![
                    Attribute::Symbol(*field_name),
                    Attribute::Type(*field_type),
                ])
            })
            .collect();

        ir.intern_type(
            TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("struct"))
                .attr("name", Attribute::Symbol(name))
                .attr("fields", Attribute::List(fields_attr))
                .build(),
        )
    }

    /// Create an `adt.enum` type with name and variants.
    pub fn adt_enum_type(
        &self,
        ir: &mut IrContext,
        name: Symbol,
        variants: &[(Symbol, Vec<TypeRef>)],
    ) -> TypeRef {
        let variants_attr: Vec<Attribute> = variants
            .iter()
            .map(|(variant_name, field_types)| {
                let field_attrs: Vec<Attribute> =
                    field_types.iter().map(|t| Attribute::Type(*t)).collect();
                Attribute::List(vec![
                    Attribute::Symbol(*variant_name),
                    Attribute::List(field_attrs),
                ])
            })
            .collect();

        ir.intern_type(
            TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("enum"))
                .attr("name", Attribute::Symbol(name))
                .attr("variants", Attribute::List(variants_attr))
                .build(),
        )
    }

    /// Create an `adt.enum` type with a stable source declaration identity.
    pub fn adt_enum_type_with_definition(
        &self,
        ir: &mut IrContext,
        name: Symbol,
        variants: &[(Symbol, Vec<TypeRef>)],
        definition: crate::typeck::DefinitionIdentity,
    ) -> TypeRef {
        let variants_attr: Vec<Attribute> = variants
            .iter()
            .map(|(variant_name, field_types)| {
                let field_attrs: Vec<Attribute> =
                    field_types.iter().map(|ty| Attribute::Type(*ty)).collect();
                Attribute::List(vec![
                    Attribute::Symbol(*variant_name),
                    Attribute::List(field_attrs),
                ])
            })
            .collect();

        ir.intern_type(
            TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("enum"))
                .attr("name", Attribute::Symbol(name))
                .attr("variants", Attribute::List(variants_attr))
                .attr(
                    "tribute.definition.source",
                    Attribute::Int(definition.source as i128),
                )
                .attr(
                    "tribute.definition.start",
                    Attribute::Int(definition.start as i128),
                )
                .attr(
                    "tribute.definition.end",
                    Attribute::Int(definition.end as i128),
                )
                .build(),
        )
    }

    /// Create an `adt.typeref` type — a reference to a named type.
    pub fn adt_typeref(&self, ir: &mut IrContext, name: Symbol) -> TypeRef {
        ir.intern_type(
            TypeDataBuilder::new(Symbol::new("adt"), Symbol::new("typeref"))
                .attr("name", Attribute::Symbol(name))
                .build(),
        )
    }
}

/// RAII guard that exits a scope on drop.
///
/// Created by [`IrLoweringCtx::scope()`]. Dereferences to `IrLoweringCtx`
/// so it can be used as a drop-in replacement for `&mut ctx`.
pub struct ScopeGuard<'a, 'db> {
    ctx: &'a mut IrLoweringCtx<'db>,
}

impl<'db> Deref for ScopeGuard<'_, 'db> {
    type Target = IrLoweringCtx<'db>;
    fn deref(&self) -> &Self::Target {
        self.ctx
    }
}

impl<'db> DerefMut for ScopeGuard<'_, 'db> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.ctx
    }
}

impl Drop for ScopeGuard<'_, '_> {
    fn drop(&mut self) {
        self.ctx.exit_scope();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Type as AstType, TypeKind};
    use trunk_ir::ops::DialectType;

    fn test_db() -> salsa::DatabaseImpl {
        salsa::DatabaseImpl::new()
    }

    #[test]
    fn test_convert_logical_bound_var_to_any() {
        let db = test_db();
        let mut ir = IrContext::new();
        let path = ir.intern_path("test.trb".to_owned());
        let ctx = IrLoweringCtx::new(
            &db,
            path,
            crate::ast::SpanMap::default(),
            HashMap::new(),
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );

        let ty = AstType::new(&db, TypeKind::BoundVar { index: 0 });
        let ir_ty = ctx.convert_logical_type(&mut ir, ty);
        let expected = ctx.anyref_type(&mut ir);
        assert_eq!(ir_ty, expected);
    }

    /// The logical frontend boundary has its own recursive conversion: source
    /// callables, bottom values, resumptions, tuple layouts, and forward
    /// nominals must never fall back to the physical `func.func_sig` carrier.
    #[test]
    fn test_convert_logical_types_preserves_recursive_control_shapes() {
        let db = test_db();
        let mut ir = IrContext::new();
        let path = ir.intern_path("test.trb".to_owned());
        let mut ctx = IrLoweringCtx::new(
            &db,
            path,
            crate::ast::SpanMap::default(),
            HashMap::new(),
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );
        let int = AstType::new(&db, TypeKind::Int);
        let effect = crate::ast::EffectRow::pure(&db);
        let callable = AstType::new(
            &db,
            TypeKind::Func {
                params: vec![int],
                result: int,
                effect,
                minimum_convention: CallingConvention::Direct,
            },
        );
        let logical_callable = ctx.convert_logical_type(&mut ir, callable);
        let i32_ty = ctx.i32_type(&mut ir);
        let signature =
            tribute_ir::dialect::tribute_control::FuncSig::from_type_ref(&ir, logical_callable)
                .expect("logical function type must be a control callable");
        assert_eq!(signature.result(&ir), i32_ty);
        assert_eq!(signature.inputs(&ir), &[i32_ty]);
        assert_eq!(
            tribute_ir::dialect::tribute_control::func_sig_convention(&ir, logical_callable),
            Some(tribute_ir::dialect::tribute_control::CallingConvention::Direct)
        );

        let never = AstType::new(&db, TypeKind::Never);
        assert_eq!(
            ctx.convert_logical_type(&mut ir, never),
            core::never(&mut ir).as_type_ref()
        );

        let resume = AstType::new(
            &db,
            TypeKind::Continuation {
                arg: int,
                result: int,
                effect,
            },
        );
        let resume = ctx.convert_logical_type(&mut ir, resume);
        let resume_data = ir.get_type(resume);
        assert_eq!(resume_data.dialect, Symbol::new("tribute_control"));
        assert_eq!(resume_data.name, Symbol::new("resume_token"));

        let tuple = AstType::new(&db, TypeKind::Tuple(vec![callable, never]));
        let tuple_name = ctx.logical_tuple_name(tuple);
        assert_eq!(
            ctx.convert_logical_type(&mut ir, tuple),
            ctx.adt_typeref(&mut ir, tuple_name)
        );

        let nominal_name = Symbol::new("Nested::Forward");
        ctx.declare_logical_nominal(nominal_name);
        let forward = AstType::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::synthetic(&db, nominal_name),
                name: Symbol::new("Forward"),
                args: vec![],
            },
        );
        assert_eq!(
            ctx.convert_logical_type(&mut ir, forward),
            ctx.adt_typeref(&mut ir, nominal_name)
        );

        // A source declaration named List must not capture the builtin type,
        // including when it occurs recursively inside a callable or tuple.
        let list_name = Symbol::new("List");
        ctx.declare_logical_nominal(list_name);
        let source_list = AstType::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::synthetic(&db, list_name),
                name: list_name,
                args: vec![],
            },
        );
        let builtin_list = AstType::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::builtin_list(&db),
                name: list_name,
                args: vec![int],
            },
        );
        let list_callable = AstType::new(
            &db,
            TypeKind::Func {
                params: vec![builtin_list, source_list],
                result: builtin_list,
                effect,
                minimum_convention: CallingConvention::Direct,
            },
        );
        let callable_ir = ctx.convert_logical_type(&mut ir, list_callable);
        let list_signature =
            tribute_ir::dialect::tribute_control::FuncSig::from_type_ref(&ir, callable_ir).unwrap();
        let anyref = ctx.anyref_type(&mut ir);
        let nominal_list = ctx.adt_typeref(&mut ir, list_name);
        assert_eq!(list_signature.inputs(&ir), &[anyref, nominal_list]);
        assert_eq!(list_signature.result(&ir), anyref);
        let tuple = AstType::new(&db, TypeKind::Tuple(vec![builtin_list, source_list]));
        ctx.convert_logical_type(&mut ir, tuple);
        let tuple_name = ctx.logical_tuple_name(tuple);
        let (_, layout) = ir
            .types()
            .iter()
            .find(|(_, data)| {
                data.dialect == Symbol::new("adt")
                    && data.name == Symbol::new("struct")
                    && data.attrs.get_symbol("name") == Some(tuple_name)
            })
            .expect("logical tuple layout");
        assert_eq!(
            layout.attrs.get("fields"),
            Some(&Attribute::List(vec![
                Attribute::List(vec![
                    Attribute::Symbol(Symbol::new("0")),
                    Attribute::Type(anyref)
                ]),
                Attribute::List(vec![
                    Attribute::Symbol(Symbol::new("1")),
                    Attribute::Type(nominal_list)
                ]),
            ]))
        );

        let generated = Symbol::new("Forward::value");
        ctx.register_logical_generated_signature(
            generated,
            vec![i32_ty],
            i32_ty,
            CallingConvention::Direct,
        );
        // Re-registering the exact semantic signature is a deterministic
        // no-op; a conflicting signature remains fail-closed in the lowering
        // context rather than being silently replaced.
        ctx.register_logical_generated_signature(
            generated,
            vec![i32_ty],
            i32_ty,
            CallingConvention::Direct,
        );
        let generated_signature = ctx
            .lookup_logical_generated_signature(generated)
            .expect("generated logical signature must be retained");
        assert_eq!(generated_signature.param_types, vec![i32_ty]);
        assert_eq!(generated_signature.return_type, i32_ty);
        assert_eq!(generated_signature.convention, CallingConvention::Direct);

        let source_function = Symbol::new("Forward::run");
        ctx.register_logical_source_function(source_function);
        assert!(ctx.is_logical_source_function(source_function));
        assert!(ctx.mark_logical_extern_emitted(Symbol::new("prelude::id")));
        assert!(!ctx.mark_logical_extern_emitted(Symbol::new("prelude::id")));

        ctx.enter_module(Symbol::new("Nested"));
        assert_eq!(
            ctx.qualify_name(Symbol::new("run")),
            Symbol::new("Nested::run")
        );
        ctx.exit_module();

        let erased_cases = [
            AstType::new(&db, TypeKind::BoundVar { index: 0 }),
            AstType::new(
                &db,
                TypeKind::UniVar {
                    id: crate::ast::UniVarId::new(&db, crate::ast::UniVarSource::Anonymous(0), 0),
                },
            ),
            AstType::new(
                &db,
                TypeKind::Named {
                    id: crate::ast::TypeDefId::builtin_list(&db),
                    name: Symbol::new("Undeclared"),
                    args: vec![],
                },
            ),
        ];
        let anyref = ctx.anyref_type(&mut ir);
        for source_ty in erased_cases {
            assert_eq!(ctx.convert_logical_type(&mut ir, source_ty), anyref);
        }
        assert_eq!(
            ctx.convert_logical_type(&mut ir, AstType::new(&db, TypeKind::Error)),
            ctx.nil_type(&mut ir)
        );
        let applied_forward = AstType::new(
            &db,
            TypeKind::App {
                ctor: forward,
                args: vec![int],
            },
        );
        assert_eq!(
            ctx.convert_logical_type(&mut ir, applied_forward),
            ctx.adt_typeref(&mut ir, nominal_name)
        );

        let evidence_callable = AstType::new(
            &db,
            TypeKind::Func {
                params: vec![int],
                result: int,
                effect,
                minimum_convention: CallingConvention::EvidenceDirect,
            },
        );
        let cps_callable = AstType::new(
            &db,
            TypeKind::Func {
                params: vec![int],
                result: int,
                effect,
                minimum_convention: CallingConvention::Cps,
            },
        );
        let evidence_ir = ctx.convert_logical_type(&mut ir, evidence_callable);
        let cps_ir = ctx.convert_logical_type(&mut ir, cps_callable);
        assert_eq!(
            tribute_ir::dialect::tribute_control::func_sig_convention(&ir, evidence_ir),
            Some(tribute_ir::dialect::tribute_control::CallingConvention::EvidenceDirect)
        );
        assert_eq!(
            tribute_ir::dialect::tribute_control::func_sig_convention(&ir, cps_ir),
            Some(tribute_ir::dialect::tribute_control::CallingConvention::Cps)
        );

        let continuation_key = AstType::new(
            &db,
            TypeKind::Continuation {
                arg: int,
                result: int,
                effect,
            },
        );
        let univar_key = AstType::new(
            &db,
            TypeKind::UniVar {
                id: crate::ast::UniVarId::new(&db, crate::ast::UniVarSource::Anonymous(1), 1),
            },
        );
        let error_key = AstType::new(&db, TypeKind::Error);
        let list_key = AstType::new(
            &db,
            TypeKind::Named {
                id: crate::ast::TypeDefId::builtin_list(&db),
                name: Symbol::new("List"),
                args: vec![int],
            },
        );
        let function_local_key = AstType::new(
            &db,
            TypeKind::UniVar {
                id: crate::ast::UniVarId::new(
                    &db,
                    crate::ast::UniVarSource::FunctionLocal {
                        func_id: crate::ast::FuncDefId::new(&db, Symbol::new("Nested::key")),
                        index: 2,
                    },
                    0,
                ),
            },
        );
        let solver_key = AstType::new(
            &db,
            TypeKind::UniVar {
                id: crate::ast::UniVarId::new(
                    &db,
                    crate::ast::UniVarSource::Solver { index: 2 },
                    0,
                ),
            },
        );
        let structural_tuple = AstType::new(
            &db,
            TypeKind::Tuple(vec![
                evidence_callable,
                cps_callable,
                applied_forward,
                continuation_key,
                univar_key,
                error_key,
                list_key,
                function_local_key,
                solver_key,
            ]),
        );
        let structural_name = ctx.logical_tuple_name(structural_tuple);
        let structural_text = structural_name.with_str(str::to_owned);
        assert!(
            structural_text.contains("callable")
                && structural_text.contains("evidence_direct")
                && structural_text.contains("cps")
                && structural_text.contains("app")
                && structural_text.contains("resume")
                && structural_text.contains("univar")
                && structural_text.contains("error")
                && structural_text.contains("builtin_list")
                && structural_text.contains("function_local")
                && structural_text.contains("solver"),
            "logical tuple keys must retain full recursive callable and value shapes"
        );
        let function_local_name =
            ctx.logical_tuple_name(AstType::new(&db, TypeKind::Tuple(vec![function_local_key])));
        let solver_name =
            ctx.logical_tuple_name(AstType::new(&db, TypeKind::Tuple(vec![solver_key])));
        assert_ne!(function_local_name, solver_name);
    }

    #[test]
    fn test_convert_logical_primitives() {
        let db = test_db();
        let mut ir = IrContext::new();
        let path = ir.intern_path("test.trb".to_owned());
        let ctx = IrLoweringCtx::new(
            &db,
            path,
            crate::ast::SpanMap::default(),
            HashMap::new(),
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );

        // Int → I32
        let int_ty = AstType::new(&db, TypeKind::Int);
        assert_eq!(
            ctx.convert_logical_type(&mut ir, int_ty),
            ctx.i32_type(&mut ir)
        );

        // Bool → I1
        let bool_ty = AstType::new(&db, TypeKind::Bool);
        assert_eq!(
            ctx.convert_logical_type(&mut ir, bool_ty),
            ctx.bool_type(&mut ir)
        );

        // Float → F64
        let float_ty = AstType::new(&db, TypeKind::Float);
        assert_eq!(
            ctx.convert_logical_type(&mut ir, float_ty),
            ctx.f64_type(&mut ir)
        );

        // Nil → Nil
        let nil_ty = AstType::new(&db, TypeKind::Nil);
        assert_eq!(
            ctx.convert_logical_type(&mut ir, nil_ty),
            ctx.nil_type(&mut ir)
        );
    }

    #[test]
    fn test_lookup_function_type() {
        let db = test_db();
        let mut ir = IrContext::new();
        let path = ir.intern_path("test.trb".to_owned());
        let name = Symbol::new("foo");
        let body = AstType::new(&db, TypeKind::Int);
        let scheme = TypeScheme::new(&db, vec![], vec![], body);

        let mut ft = HashMap::new();
        ft.insert(name, scheme);

        let ctx = IrLoweringCtx::new(
            &db,
            path,
            crate::ast::SpanMap::default(),
            ft,
            HashMap::new(),
            smallvec::smallvec![Symbol::new("test")],
            HashMap::new(),
        );
        assert_eq!(ctx.lookup_function_type(name), Some(&scheme));
        assert_eq!(ctx.lookup_function_type(Symbol::new("missing")), None);
    }
}
