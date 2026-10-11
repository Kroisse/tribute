//! Type checker implementation.
//!
//! Performs bidirectional type checking on the AST, transforming
//! `Module<ResolvedRef<'db>>` into `Module<TypedRef<'db>>`.
//!
//! ## Architecture
//!
//! The type checker uses a two-level context system:
//!
//! - `ModuleTypeEnv`: Module-level type information (function signatures, constructors, type defs).
//!   Declaration collection initializes it; function checking only reads it, since each
//!   module-level function's declared signature is its final scheme.
//!
//! - `FunctionInferenceContext`: Per-function type inference state (local variables, constraints,
//!   type variable counters). Each function gets its own context, ensuring type inference is
//!   isolated and UniVars are fully resolved within each function.
//!
//! ## Modules
//!
//! - `collect`: Declaration collection (Phase 1) - populates ModuleTypeEnv
//! - `func_check`: Function type checking (Phase 2) - per-function inference
//! - `finalize`: Solved body substitution and binder-variable collection
//! - `diagnostics`: Source-oriented rendering of inference failures
//! - `expr`: Expression type checking - uses FunctionInferenceContext

mod become_check;
mod collect;
mod diagnostics;
mod exhaustiveness;
mod expr;
mod finalize;
mod func_check;

use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;

use trunk_ir::{Span, Symbol};

use crate::SortedMap;
use crate::ast::{
    Decl, FuncDefId, Module, NodeId, ResolvedRef, SpanMap, Type, TypeScheme, TypedRef, UniVarId,
};

use super::context::ModuleTypeEnv;
use super::{
    DefinitionIdentity, PreludeExports, StringType, WellKnownType, WellKnownTypeKey, WellKnownTypes,
};

/// Result of module type checking.
pub struct ModuleCheckResult<'db> {
    /// The typed module AST.
    pub module: Module<TypedRef<'db>>,
    /// Function type schemes (name → polymorphic type).
    pub function_types: Vec<(Symbol, TypeScheme<'db>)>,
    /// Constructor type schemes retain exact nominal field types for logical
    /// layout construction.
    pub constructor_types: Vec<(crate::ast::CtorId<'db>, TypeScheme<'db>)>,
    /// Node types for IR lowering (NodeId → monomorphic type).
    pub node_types: SortedMap<NodeId, Type<'db>>,
    /// Exact instantiated types selected for direct call callees.
    pub function_instances: SortedMap<NodeId, super::FunctionInstance<'db>>,
    pub local_instances: SortedMap<NodeId, super::LocalCallableInstance<'db>>,
    /// Non-identity evidence selections of calls and resumes.
    pub evidence_plans: SortedMap<NodeId, Vec<super::EvidenceStep<'db>>>,
    /// Exact semantic operation instances for handler arms.
    pub handler_operations: SortedMap<NodeId, crate::typeck::InstantiatedHandlerOperation<'db>>,
    /// Exact semantic operation instances for ability-operation calls.
    pub perform_operations: SortedMap<NodeId, crate::typeck::InstantiatedPerformOperation<'db>>,
    /// Fully solved source-callable signatures for lambda expressions.
    pub lambda_signatures: SortedMap<NodeId, crate::typeck::LambdaSignature<'db>>,
    /// Case expression nodes whose coverage was proved exhaustive.
    pub exhaustive_cases: Vec<NodeId>,
    /// Resolved ability operation schemas retained for source-logical IR
    /// declaration metadata.
    pub ability_definitions: Vec<(
        crate::ast::AbilityId<'db>,
        crate::typeck::context::AbilityInfo<'db>,
    )>,
    /// Prelude-defined semantic type identities.
    pub well_known_types: WellKnownTypes<'db>,
}

/// What checking one module-level function produces: its body's node
/// metadata, keyed by nodes only that function owns. Function checking only
/// reads the module environment, so the module merges these results in any
/// order.
#[derive(Default)]
pub(crate) struct FunctionCheck<'db> {
    pub(super) node_types: HashMap<NodeId, Type<'db>>,
    pub(super) function_instances: HashMap<NodeId, super::FunctionInstance<'db>>,
    pub(super) local_instances: HashMap<NodeId, super::LocalCallableInstance<'db>>,
    pub(super) evidence_plans: HashMap<NodeId, Vec<super::EvidenceStep<'db>>>,
    pub(super) handler_operations:
        HashMap<NodeId, crate::typeck::InstantiatedHandlerOperation<'db>>,
    pub(super) perform_operations:
        HashMap<NodeId, crate::typeck::InstantiatedPerformOperation<'db>>,
    pub(super) lambda_signatures: HashMap<NodeId, crate::typeck::LambdaSignature<'db>>,
    pub(super) exhaustive_cases: Vec<NodeId>,
    /// Quantifiers owned by generalized local schemes in this function. They
    /// remain separate from exported function schemes.
    pub(super) local_generalizations: HashMap<UniVarId<'db>, (NodeId, u32)>,
    /// Case scrutinees whose exhaustiveness diagnostics were already reported.
    pub(super) exhaustiveness_reported: HashSet<NodeId>,
}

/// Type checking mode.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum Mode<'db> {
    /// Infer the type of an expression.
    Infer,
    /// Check that an expression has a specific type.
    Check(Type<'db>),
}

/// Type checker for AST expressions.
///
/// Uses `ModuleTypeEnv` for module-level type information and creates
/// `FunctionInferenceContext` per function for isolated type inference.
pub struct TypeChecker<'db> {
    /// Module-level type environment (function signatures, constructors, type defs).
    pub(crate) env: ModuleTypeEnv<'db>,
    /// Current module prefix for qualified function names (e.g., "foo::bar").
    pub(crate) prefix: String,
    /// Span map for converting NodeId to Span in diagnostics.
    pub(crate) span_map: SpanMap,
    /// Accumulated node types from all functions.
    /// Collects NodeId → Type mappings during type checking.
    node_types: HashMap<NodeId, Type<'db>>,
    function_instances: HashMap<NodeId, super::FunctionInstance<'db>>,
    local_instances: HashMap<NodeId, super::LocalCallableInstance<'db>>,
    evidence_plans: HashMap<NodeId, Vec<super::EvidenceStep<'db>>>,
    /// Exact handler operation instances collected from each checked function.
    handler_operations: HashMap<NodeId, crate::typeck::InstantiatedHandlerOperation<'db>>,
    perform_operations: HashMap<NodeId, crate::typeck::InstantiatedPerformOperation<'db>>,
    lambda_signatures: HashMap<NodeId, crate::typeck::LambdaSignature<'db>>,
    exhaustive_cases: Vec<NodeId>,
    /// Source origins for concrete effects in each collected function signature.
    effect_annotation_origins: HashMap<FuncDefId<'db>, crate::ast::EffectAnnotationOrigins>,
    signature_row_names: HashMap<FuncDefId<'db>, HashMap<Symbol, crate::ast::EffectVar>>,
    signature_type_names: HashMap<FuncDefId<'db>, HashMap<Symbol, u32>>,
    /// Functions of one parameter collected so far, to be checked against
    /// the fields of that parameter's struct once it is collected too.
    unary_functions: Vec<(NodeId, Symbol, FuncDefId<'db>, Type<'db>)>,
}

impl<'db> TypeChecker<'db> {
    fn prelude_well_known_type(
        &self,
        module: &Module<ResolvedRef<'db>>,
        key: impl WellKnownTypeKey,
    ) -> Option<WellKnownType<'db>> {
        let name = key.name();
        // The prelude's root items are inside its package module.
        let mut prefix = String::new();
        let mut decls = &module.decls;
        if let [Decl::Module(package)] = decls.as_slice()
            && let Some(body) = &package.body
        {
            crate::push_prefix(&mut prefix, &package.name);
            decls = body;
        }
        let declaration = decls.iter().find_map(|decl| match decl {
            Decl::Struct(decl) if decl.name == name => Some(decl.id),
            Decl::Enum(decl) if decl.name == name => Some(decl.id),
            _ => None,
        })?;
        let qualified = crate::qualified_symbol(&mut prefix, &name);
        let ty = self.env.lookup_type_def(&qualified)?.body(self.db());
        Some(WellKnownType {
            ty,
            definition: DefinitionIdentity::new(
                declaration,
                self.span_map.get_or_default(declaration),
            ),
        })
    }

    /// The semantic identities the prelude supplies to later phases. Literal
    /// pattern equalities are the `==` methods of their types, resolved the
    /// same way as an `==` expression on such a receiver.
    fn prelude_well_known_types(&self, module: &Module<ResolvedRef<'db>>) -> WellKnownTypes<'db> {
        let string = self.prelude_well_known_type(module, StringType);
        let equality = |ty: Type<'db>| {
            self.env
                .lookup_method(&Symbol::new("=="), ty)
                .map(|entry| entry.func_id)
        };
        WellKnownTypes {
            string,
            string_equality: string.and_then(|string| equality(string.ty)),
            bytes_equality: equality(self.env.bytes_type()),
        }
    }

    /// Create a new type checker with the given span map.
    pub fn new(db: &'db dyn salsa::Database, span_map: SpanMap) -> Self {
        Self {
            env: ModuleTypeEnv::new(db),
            prefix: String::new(),
            span_map,
            node_types: HashMap::default(),
            function_instances: HashMap::default(),
            local_instances: HashMap::default(),
            evidence_plans: HashMap::default(),
            handler_operations: HashMap::default(),
            perform_operations: HashMap::default(),
            lambda_signatures: HashMap::default(),
            exhaustive_cases: Vec::new(),
            effect_annotation_origins: HashMap::default(),
            signature_row_names: HashMap::default(),
            signature_type_names: HashMap::default(),
            unary_functions: Vec::new(),
        }
    }

    /// Get the span for a NodeId, falling back to Span::new(0, 0) if not found.
    pub(crate) fn get_span(&self, node_id: crate::ast::NodeId) -> Span {
        self.span_map.get_or_default(node_id)
    }

    /// Get the current module prefix string (e.g., "foo::bar").
    pub(crate) fn current_prefix(&self) -> &str {
        &self.prefix
    }

    /// Create a FuncDefId from the current prefix and function name.
    pub(crate) fn func_def_id(&self, name: &Symbol) -> FuncDefId<'db> {
        FuncDefId::new(
            self.db(),
            crate::qualified_symbol(&mut self.prefix.clone(), name),
        )
    }

    /// Get the database.
    pub(crate) fn db(&self) -> &'db dyn salsa::Database {
        self.env.db()
    }

    /// Inject prelude's resolved type information before type checking.
    ///
    /// This makes prelude's types (Option, Some, None, etc.) available
    /// to user code without sharing a ModuleTypeEnv, avoiding UniVar conflicts.
    pub fn inject_prelude(&mut self, exports: &PreludeExports<'db>) {
        self.env.inject_prelude(exports);
    }

    /// Type check a module.
    ///
    /// Returns the typed module, function type schemes, and node types.
    pub fn check_module(self, module: &Module<ResolvedRef<'db>>) -> ModuleCheckResult<'db> {
        self.check_module_inner(module)
    }

    /// Type check the prelude module once.
    ///
    /// Returns the typed prelude, and the exports that user modules inject:
    /// function types, constructors, type definitions, and the method index.
    pub fn check_prelude(
        mut self,
        module: &Module<ResolvedRef<'db>>,
    ) -> (ModuleCheckResult<'db>, PreludeExports<'db>) {
        self.collect_declarations(module);
        let well_known_types = self.prelude_well_known_types(module);
        self.env.set_prelude_well_known_types(well_known_types);
        let decls = self.check_decls(module);
        let exports = self.prelude_exports(well_known_types);
        (self.into_module_result(module, decls), exports)
    }

    /// Internal implementation for module type checking.
    ///
    /// Uses per-function type inference:
    /// 1. Collect all declarations into ModuleTypeEnv
    /// 2. For each function, create an isolated FunctionInferenceContext
    /// 3. Check the function body, solve constraints, and apply substitution
    /// 4. No global solve needed - each function's UniVars are resolved independently
    fn check_module_inner(mut self, module: &Module<ResolvedRef<'db>>) -> ModuleCheckResult<'db> {
        // Phase 1: Collect type definitions and function signatures into ModuleTypeEnv
        // Note: module_path starts empty because module.name is the file-derived name,
        // which is for external references, not internal function naming.
        self.collect_declarations(module);
        self.check_collected_module(module)
    }

    fn check_collected_module(
        mut self,
        module: &Module<ResolvedRef<'db>>,
    ) -> ModuleCheckResult<'db> {
        let decls = self.check_decls(module);
        self.into_module_result(module, decls)
    }

    /// Phase 2: Type check each declaration with per-function inference.
    /// Each function gets its own FunctionInferenceContext with isolated
    /// constraints, so no global solve follows.
    fn check_decls(&mut self, module: &Module<ResolvedRef<'db>>) -> Vec<Decl<TypedRef<'db>>> {
        module
            .decls
            .iter()
            .map(|decl| self.check_decl(decl))
            .collect()
    }

    /// Assemble the result of checking `module` from its checked declarations.
    fn into_module_result(
        mut self,
        module: &Module<ResolvedRef<'db>>,
        decls: Vec<Decl<TypedRef<'db>>>,
    ) -> ModuleCheckResult<'db> {
        // Export the function types (already finalized during per-function checking)
        let function_types = self.env.export_function_types();
        let constructor_types = self.env.export_constructor_types();
        let ability_definitions = self.env.export_ability_defs();
        let well_known_types = self.env.well_known_types();

        self.exhaustive_cases.sort();

        ModuleCheckResult {
            module: Module {
                id: module.id,
                name: module.name.clone(),
                decls,
            },
            function_types,
            constructor_types,
            node_types: self.node_types.into_iter().collect(),
            function_instances: self.function_instances.into_iter().collect(),
            local_instances: self.local_instances.into_iter().collect(),
            evidence_plans: self.evidence_plans.into_iter().collect(),
            handler_operations: self.handler_operations.into_iter().collect(),
            perform_operations: self.perform_operations.into_iter().collect(),
            lambda_signatures: self.lambda_signatures.into_iter().collect(),
            exhaustive_cases: self.exhaustive_cases,
            ability_definitions,
            well_known_types,
        }
    }

    /// Export the prelude's finalized types for injection into user modules.
    fn prelude_exports(&self, well_known_types: WellKnownTypes<'db>) -> PreludeExports<'db> {
        PreludeExports::new(
            self.db(),
            self.env.export_function_types_with_ids(),
            self.env.export_extern_functions(),
            self.env.export_constructor_types(),
            self.env.export_type_defs(),
            self.env.export_struct_fields(),
            self.env.export_enum_variants(),
            self.env.export_constructor_field_names(),
            self.env.export_method_index(),
            self.env.export_ability_defs_for_prelude(),
            well_known_types,
        )
    }

    // =========================================================================
    // Declaration checking (Phase 2)
    // =========================================================================

    /// Type check a declaration.
    fn check_decl(&mut self, decl: &Decl<ResolvedRef<'db>>) -> Decl<TypedRef<'db>> {
        match decl {
            Decl::Function(func) => {
                let (func, checked) = self.check_func_decl(func);
                self.merge_function(checked);
                Decl::Function(func)
            }
            // These declarations contain no expressions to check.
            Decl::ExternFunction(e) => Decl::ExternFunction(e.clone()),
            Decl::Struct(s) => Decl::Struct(s.clone()),
            Decl::Enum(e) => Decl::Enum(e.clone()),
            Decl::Ability(a) => Decl::Ability(a.clone()),
            Decl::Use(u) => Decl::Use(u.clone()),
            Decl::Module(m) => Decl::Module(self.check_module_decl(m)),
        }
    }

    /// Merge one function's results into the module's.
    fn merge_function(&mut self, checked: FunctionCheck<'db>) {
        self.node_types.extend(checked.node_types);
        self.function_instances.extend(checked.function_instances);
        self.local_instances.extend(checked.local_instances);
        self.evidence_plans.extend(checked.evidence_plans);
        self.handler_operations.extend(checked.handler_operations);
        self.perform_operations.extend(checked.perform_operations);
        self.lambda_signatures.extend(checked.lambda_signatures);
        self.exhaustive_cases.extend(checked.exhaustive_cases);
    }

    /// Type check a module declaration.
    fn check_module_decl(
        &mut self,
        module: &crate::ast::ModuleDecl<ResolvedRef<'db>>,
    ) -> crate::ast::ModuleDecl<TypedRef<'db>> {
        // Push module name to prefix
        let prev_len = crate::push_prefix(&mut self.prefix, &module.name);

        let body = module
            .body
            .as_ref()
            .map(|decls| decls.iter().map(|d| self.check_decl(d)).collect());

        // Restore prefix
        self.prefix.truncate(prev_len);

        crate::ast::ModuleDecl {
            id: module.id,
            name: module.name.clone(),
            is_pub: module.is_pub,
            body,
        }
    }
}
