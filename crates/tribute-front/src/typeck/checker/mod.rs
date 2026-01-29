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
//!   This is populated during `collect_declarations` and is read-only afterward.
//!
//! - `FunctionInferenceContext`: Per-function type inference state (local variables, constraints,
//!   type variable counters). Each function gets its own context, ensuring type inference is
//!   isolated and UniVars are fully resolved within each function.
//!
//! ## Modules
//!
//! - `collect`: Declaration collection (Phase 1) - populates ModuleTypeEnv
//! - `func_check`: Function type checking (Phase 2) - per-function inference
//! - `expr`: Expression type checking - uses FunctionInferenceContext

mod collect;
mod expr;
mod func_check;

use trunk_ir::Symbol;

use crate::ast::{Decl, Module, ResolvedRef, Type, TypeScheme, TypedRef};

use super::PreludeExports;
use super::context::ModuleTypeEnv;

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
    /// Current module path for qualified function names.
    pub(crate) module_path: Vec<Symbol>,
}

impl<'db> TypeChecker<'db> {
    /// Create a new type checker.
    pub fn new(db: &'db dyn salsa::Database) -> Self {
        Self {
            env: ModuleTypeEnv::new(db),
            module_path: Vec::new(),
        }
    }

    /// Build a qualified function name from the current module path and function name.
    pub(crate) fn qualified_func_name(&self, name: Symbol) -> Symbol {
        if self.module_path.is_empty() {
            name
        } else {
            // Join module path with "::" separator
            let mut qualified = String::new();
            for part in &self.module_path {
                if !qualified.is_empty() {
                    qualified.push_str("::");
                }
                qualified.push_str(&part.to_string());
            }
            qualified.push_str("::");
            qualified.push_str(&name.to_string());
            Symbol::from_dynamic(&qualified)
        }
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
    /// Returns the typed module and a list of exported function type schemes.
    pub fn check_module(
        self,
        module: Module<ResolvedRef<'db>>,
    ) -> (Module<TypedRef<'db>>, Vec<(Symbol, TypeScheme<'db>)>) {
        self.check_module_inner(module)
    }

    /// Type check the prelude module.
    ///
    /// Returns PreludeExports containing function types, constructors, and type defs.
    pub fn check_module_for_prelude(self, module: Module<ResolvedRef<'db>>) -> PreludeExports<'db> {
        self.check_module_inner_for_prelude(module)
    }

    /// Internal implementation for module type checking.
    ///
    /// Uses per-function type inference:
    /// 1. Collect all declarations into ModuleTypeEnv
    /// 2. For each function, create an isolated FunctionInferenceContext
    /// 3. Check the function body, solve constraints, and apply substitution
    /// 4. No global solve needed - each function's UniVars are resolved independently
    fn check_module_inner(
        mut self,
        module: Module<ResolvedRef<'db>>,
    ) -> (Module<TypedRef<'db>>, Vec<(Symbol, TypeScheme<'db>)>) {
        // Phase 1: Collect type definitions and function signatures into ModuleTypeEnv
        // Note: module_path starts empty because module.name is the file-derived name,
        // which is for external references, not internal function naming.
        self.collect_declarations(&module);

        // Phase 2: Type check each declaration with per-function inference
        // Each function gets its own FunctionInferenceContext with isolated constraints
        let decls: Vec<Decl<TypedRef<'db>>> = module
            .decls
            .into_iter()
            .map(|decl| self.check_decl(decl))
            .collect();

        // Phase 3: No global solve needed!
        // Each function's constraints were solved in check_func_decl

        // Export the function types (already finalized during per-function checking)
        let function_types = self.env.export_function_types();

        let typed_module = Module {
            id: module.id,
            name: module.name,
            decls,
        };

        (typed_module, function_types)
    }

    /// Internal implementation for prelude module type checking.
    ///
    /// Similar to check_module_inner but returns PreludeExports with FuncDefId keys.
    /// Uses per-function type inference just like regular module checking.
    fn check_module_inner_for_prelude(
        mut self,
        module: Module<ResolvedRef<'db>>,
    ) -> PreludeExports<'db> {
        // Phase 1: Collect type definitions and function signatures
        // Note: module_path starts empty - prelude functions use simple names internally.
        self.collect_declarations(&module);

        // Phase 2: Type check all declarations with per-function inference
        let _decls: Vec<Decl<TypedRef<'db>>> = module
            .decls
            .into_iter()
            .map(|decl| self.check_decl(decl))
            .collect();

        // Phase 3: No global solve needed - each function was solved independently

        // Export the finalized types (already substituted and generalized per-function)
        let function_types = self.env.export_function_types_with_ids();
        let constructor_types = self.env.export_constructor_types();
        let type_defs = self.env.export_type_defs();

        PreludeExports::new(self.db(), function_types, constructor_types, type_defs)
    }

    // =========================================================================
    // Declaration checking (Phase 2)
    // =========================================================================

    /// Type check a declaration.
    fn check_decl(&mut self, decl: Decl<ResolvedRef<'db>>) -> Decl<TypedRef<'db>> {
        match decl {
            Decl::Function(func) => Decl::Function(self.check_func_decl(func)),
            Decl::ExternFunction(e) => Decl::ExternFunction(e),
            Decl::Struct(s) => Decl::Struct(self.check_struct_decl(s)),
            Decl::Enum(e) => Decl::Enum(self.check_enum_decl(e)),
            Decl::Ability(a) => Decl::Ability(self.check_ability_decl(a)),
            Decl::Use(u) => Decl::Use(self.check_use_decl(u)),
            Decl::Module(m) => Decl::Module(self.check_module_decl(m)),
        }
    }

    /// Type check a module declaration.
    fn check_module_decl(
        &mut self,
        module: crate::ast::ModuleDecl<ResolvedRef<'db>>,
    ) -> crate::ast::ModuleDecl<TypedRef<'db>> {
        // Push module name to path
        self.module_path.push(module.name);

        let body = module
            .body
            .map(|decls| decls.into_iter().map(|d| self.check_decl(d)).collect());

        // Pop module name from path
        self.module_path.pop();

        crate::ast::ModuleDecl {
            id: module.id,
            name: module.name,
            is_pub: module.is_pub,
            body,
        }
    }

    /// Type check a struct declaration (no body to check).
    fn check_struct_decl(&mut self, s: crate::ast::StructDecl) -> crate::ast::StructDecl {
        s // Struct declarations don't contain expressions
    }

    /// Type check an enum declaration (no body to check).
    fn check_enum_decl(&mut self, e: crate::ast::EnumDecl) -> crate::ast::EnumDecl {
        e // Enum declarations don't contain expressions
    }

    /// Type check an ability declaration (no body to check).
    fn check_ability_decl(&mut self, a: crate::ast::AbilityDecl) -> crate::ast::AbilityDecl {
        a // Ability declarations don't contain expressions
    }

    /// Type check a use declaration (nothing to check).
    fn check_use_decl(&mut self, u: crate::ast::UseDecl) -> crate::ast::UseDecl {
        u
    }
}
