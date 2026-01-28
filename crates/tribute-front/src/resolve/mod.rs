//! Name resolution for the AST.
//!
//! This module transforms `Module<UnresolvedName>` into `Module<ResolvedRef<'db>>`.
//!
//! ## Pipeline
//!
//! 1. Build a `ModuleEnv` from declarations
//! 2. Walk the AST, resolving each name reference
//!
//! ## Resolution Strategy
//!
//! Names are resolved in this order:
//! 1. Local variables (function parameters, let bindings, pattern bindings)
//! 2. Builtin operations
//! 3. Module-level definitions (functions, types, constructors)

mod env;
mod resolver;

pub use env::{Binding, ModuleEnv};
pub use resolver::Resolver;

use trunk_ir::Symbol;

use crate::ast::{CtorId, Decl, FuncDefId, Module, ResolvedRef, SpanMap, UnresolvedName};

/// Resolve names in a module.
///
/// This is the main entry point for name resolution.
pub fn resolve_module<'db>(
    db: &'db dyn salsa::Database,
    module: Module<UnresolvedName>,
    span_map: SpanMap,
) -> Module<ResolvedRef<'db>> {
    // Build the module environment from declarations
    let env = build_env(db, &module);

    // Create resolver and process the module
    let mut resolver = Resolver::new(db, env, span_map);
    resolver.resolve_module(module)
}

/// Build a module environment from AST declarations.
fn build_env<'db>(db: &'db dyn salsa::Database, module: &Module<UnresolvedName>) -> ModuleEnv<'db> {
    let mut env = ModuleEnv::new();

    for decl in &module.decls {
        collect_definition(db, &mut env, decl);
    }

    env
}

/// Collect a definition from a declaration into the environment.
fn collect_definition<'db>(
    db: &'db dyn salsa::Database,
    env: &mut ModuleEnv<'db>,
    decl: &Decl<UnresolvedName>,
) {
    match decl {
        Decl::Function(func) => {
            let id = FuncDefId::new(db, func.name);
            env.add_function(func.name, id);
        }

        Decl::ExternFunction(func) => {
            let id = FuncDefId::new(db, func.name);
            env.add_function(func.name, id);
        }

        Decl::Struct(s) => {
            // Struct is both a type and a constructor
            let ctor_id = CtorId::new(db, s.name);
            env.add_type(s.name, ctor_id);
            env.add_constructor(s.name, ctor_id, None, s.fields.len());
        }

        Decl::Enum(e) => {
            // Enum is a type, and each variant is a constructor
            let ctor_id = CtorId::new(db, e.name);
            env.add_type(e.name, ctor_id);

            // Add each variant as a constructor in the enum's namespace
            for variant in &e.variants {
                let variant_id = CtorId::new(db, variant.name);
                let binding = Binding::Constructor {
                    id: variant_id,
                    tag: Some(variant.name),
                    arity: variant.fields.len(),
                };
                // Add to namespace (e.g., Option::Some)
                env.add_to_namespace(e.name, variant.name, binding.clone());
                // Also add directly for unqualified access (e.g., Some)
                env.add_constructor(
                    variant.name,
                    variant_id,
                    Some(variant.name),
                    variant.fields.len(),
                );
            }
        }

        Decl::Ability(a) => {
            // Ability operations are added to the ability's namespace
            for op in &a.operations {
                let func_id = FuncDefId::new(
                    db,
                    Symbol::from_dynamic(&format!("{}::{}", a.name, op.name)),
                );
                let binding = Binding::Function { id: func_id };
                env.add_to_namespace(a.name, op.name, binding);
            }
        }

        Decl::Use(u) => {
            // Import the last segment of the path
            if let Some(&name) = u.path.last() {
                let binding = Binding::Module {
                    path: u.path.clone(),
                };
                let import_name = u.alias.unwrap_or(name);
                env.add_import(import_name, binding);
            }
        }

        Decl::Module(m) => {
            // For inline modules, collect bindings into a temporary environment
            // then register them under the module's namespace
            if let Some(body) = &m.body {
                // Collect inner declarations into a temporary environment
                let mut inner_env = ModuleEnv::new();
                for inner_decl in body {
                    collect_definition(db, &mut inner_env, inner_decl);
                }

                // Register each inner definition under the module's namespace
                // e.g., `mod Foo { fn bar() {} }` makes `Foo::bar` available
                for (name, binding) in inner_env.iter_definitions() {
                    env.add_to_namespace(m.name, name, binding.clone());
                }

                // Also transfer any nested namespaces
                // e.g., `mod Foo { enum Bar { Baz } }` makes `Foo::Bar::Baz` available
                for (inner_ns, inner_bindings) in inner_env.iter_namespaces() {
                    // Create qualified namespace path: Foo::Bar
                    let qualified_ns = Symbol::from_dynamic(&format!("{}::{}", m.name, inner_ns));
                    for (name, binding) in inner_bindings {
                        env.add_to_namespace(qualified_ns, name, binding.clone());
                    }
                    // Also add the inner namespace itself under the module
                    // so Foo::Bar resolves to the Bar namespace
                    env.add_to_namespace(
                        m.name,
                        inner_ns,
                        Binding::Module {
                            path: vec![m.name, inner_ns],
                        },
                    );
                }

                // Register the module itself as a namespace binding
                env.add_import(m.name, Binding::Module { path: vec![m.name] });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{
        EnumDecl, Expr, ExprKind, ExternFuncDecl, FieldDecl, FuncDecl, Module, ModuleDecl, NodeId,
        ParamDecl, StructDecl, TypeAnnotation, TypeAnnotationKind, VariantDecl,
    };
    use salsa_test_macros::salsa_test;

    fn fresh_node_id() -> NodeId {
        NodeId::from_raw(0)
    }

    /// Input wrapper for Module to use in tracked functions.
    #[salsa::input]
    struct TestModuleInput {
        #[returns(ref)]
        module: Module<UnresolvedName>,
    }

    // =========================================================================
    // Tracked test functions - one per test case
    // =========================================================================

    #[salsa::tracked]
    fn verify_module_function_namespace(db: &dyn salsa::Database, input: TestModuleInput) {
        let env = build_env(db, input.module(db));

        // Math::add should be accessible
        let math_add = env.lookup_qualified(Symbol::new("Math"), Symbol::new("add"));
        assert!(math_add.is_some(), "Math::add should be in namespace");
        assert!(
            matches!(math_add, Some(Binding::Function { .. })),
            "Math::add should be a function"
        );

        // "add" should NOT be directly accessible (not leaked to parent scope)
        let add_direct = env.lookup(Symbol::new("add"));
        assert!(
            add_direct.is_none(),
            "add should not be leaked to parent scope"
        );

        // "Math" itself should be accessible as a module binding
        let math_module = env.lookup(Symbol::new("Math"));
        assert!(math_module.is_some(), "Math should be accessible");
        assert!(
            matches!(math_module, Some(Binding::Module { .. })),
            "Math should be a module binding"
        );
    }

    #[salsa::tracked]
    fn verify_module_struct_namespace(db: &dyn salsa::Database, input: TestModuleInput) {
        let env = build_env(db, input.module(db));

        // Types::Point should be accessible as a constructor
        let types_point = env.lookup_qualified(Symbol::new("Types"), Symbol::new("Point"));
        assert!(types_point.is_some(), "Types::Point should be in namespace");

        // "Point" should NOT be directly accessible
        let point_direct = env.lookup(Symbol::new("Point"));
        assert!(
            point_direct.is_none(),
            "Point should not be leaked to parent scope"
        );
    }

    #[salsa::tracked]
    fn verify_module_enum_namespace(db: &dyn salsa::Database, input: TestModuleInput) {
        let env = build_env(db, input.module(db));

        // Types::Color should be accessible
        let types_color = env.lookup_qualified(Symbol::new("Types"), Symbol::new("Color"));
        assert!(types_color.is_some(), "Types::Color should be in namespace");

        // Types::Red should also be accessible (variant is in module's namespace)
        let types_red = env.lookup_qualified(Symbol::new("Types"), Symbol::new("Red"));
        assert!(types_red.is_some(), "Types::Red should be in namespace");

        // "Color" and "Red" should NOT be directly accessible
        assert!(
            env.lookup(Symbol::new("Color")).is_none(),
            "Color should not be leaked"
        );
        assert!(
            env.lookup(Symbol::new("Red")).is_none(),
            "Red should not be leaked"
        );
    }

    #[salsa::tracked]
    fn verify_module_multiple_definitions(db: &dyn salsa::Database, input: TestModuleInput) {
        let env = build_env(db, input.module(db));

        // All should be accessible via Utils::
        assert!(
            env.lookup_qualified(Symbol::new("Utils"), Symbol::new("helper"))
                .is_some()
        );
        assert!(
            env.lookup_qualified(Symbol::new("Utils"), Symbol::new("another"))
                .is_some()
        );
        assert!(
            env.lookup_qualified(Symbol::new("Utils"), Symbol::new("Data"))
                .is_some()
        );

        // None should be directly accessible
        assert!(env.lookup(Symbol::new("helper")).is_none());
        assert!(env.lookup(Symbol::new("another")).is_none());
        assert!(env.lookup(Symbol::new("Data")).is_none());
    }

    #[salsa::tracked]
    fn verify_multiple_modules_namespaces(db: &dyn salsa::Database, input: TestModuleInput) {
        let env = build_env(db, input.module(db));

        // Both A::foo and B::foo should exist
        let a_foo = env.lookup_qualified(Symbol::new("A"), Symbol::new("foo"));
        let b_foo = env.lookup_qualified(Symbol::new("B"), Symbol::new("foo"));

        assert!(a_foo.is_some(), "A::foo should exist");
        assert!(b_foo.is_some(), "B::foo should exist");

        // They should be different bindings (different FuncDefIds)
        if let (Some(Binding::Function { id: id_a }), Some(Binding::Function { id: id_b })) =
            (a_foo, b_foo)
        {
            assert_ne!(id_a, id_b, "A::foo and B::foo should have different IDs");
        }
    }

    #[salsa::tracked]
    fn verify_nested_module_enum_namespace(db: &dyn salsa::Database, input: TestModuleInput) {
        let env = build_env(db, input.module(db));

        // Outer::Inner should exist
        assert!(
            env.lookup_qualified(Symbol::new("Outer"), Symbol::new("Inner"))
                .is_some(),
            "Outer::Inner should exist"
        );

        // Outer::Variant should exist (enum variants are hoisted to module namespace)
        assert!(
            env.lookup_qualified(Symbol::new("Outer"), Symbol::new("Variant"))
                .is_some(),
            "Outer::Variant should exist"
        );

        // Outer::Inner::Variant should exist
        let qualified_ns = Symbol::from_dynamic("Outer::Inner");
        assert!(
            env.lookup_qualified(qualified_ns, Symbol::new("Variant"))
                .is_some(),
            "Outer::Inner::Variant should exist"
        );
    }

    #[salsa::tracked]
    fn verify_module_no_override_parent(db: &dyn salsa::Database, input: TestModuleInput) {
        let env = build_env(db, input.module(db));

        // foo should still be directly accessible
        assert!(
            env.lookup(Symbol::new("foo")).is_some(),
            "foo should be accessible"
        );

        // M::bar should be accessible
        assert!(
            env.lookup_qualified(Symbol::new("M"), Symbol::new("bar"))
                .is_some(),
            "M::bar should be accessible"
        );

        // bar should not be directly accessible
        assert!(
            env.lookup(Symbol::new("bar")).is_none(),
            "bar should not be directly accessible"
        );
    }

    /// Create a simple function declaration for testing.
    fn simple_func(name: &str) -> FuncDecl<UnresolvedName> {
        FuncDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::from_dynamic(name),
            type_params: vec![],
            params: vec![],
            return_ty: Some(TypeAnnotation {
                id: fresh_node_id(),
                kind: TypeAnnotationKind::Named(Symbol::new("Int")),
            }),
            effects: None,
            body: Expr::new(fresh_node_id(), ExprKind::NatLit(42)),
        }
    }

    /// Create a simple struct declaration for testing.
    fn simple_struct(name: &str) -> StructDecl {
        StructDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::from_dynamic(name),
            type_params: vec![],
            fields: vec![FieldDecl {
                id: fresh_node_id(),
                is_pub: false,
                name: Some(Symbol::new("value")),
                ty: TypeAnnotation {
                    id: fresh_node_id(),
                    kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                },
            }],
        }
    }

    /// Create a simple enum declaration for testing.
    fn simple_enum(name: &str, variants: &[&str]) -> EnumDecl {
        EnumDecl {
            id: fresh_node_id(),
            is_pub: false,
            name: Symbol::from_dynamic(name),
            type_params: vec![],
            variants: variants
                .iter()
                .map(|v| VariantDecl {
                    id: fresh_node_id(),
                    name: Symbol::from_dynamic(v),
                    fields: vec![],
                })
                .collect(),
        }
    }

    /// Create an inline module declaration for testing.
    fn inline_module(name: &str, decls: Vec<Decl<UnresolvedName>>) -> ModuleDecl<UnresolvedName> {
        ModuleDecl {
            id: fresh_node_id(),
            name: Symbol::from_dynamic(name),
            is_pub: false,
            body: Some(decls),
        }
    }

    // =========================================================================
    // Module namespace tests
    // =========================================================================

    #[salsa_test]
    fn test_module_function_registered_in_namespace(db: &dyn salsa::Database) {
        // mod Math { fn add() -> Int { 42 } }
        let module = Module::new(
            fresh_node_id(),
            Some(Symbol::new("test")),
            vec![Decl::Module(inline_module(
                "Math",
                vec![Decl::Function(simple_func("add"))],
            ))],
        );

        let input = TestModuleInput::new(db, module);
        verify_module_function_namespace(db, input);
    }

    #[salsa_test]
    fn test_module_struct_registered_in_namespace(db: &dyn salsa::Database) {
        // mod Types { struct Point { value: Int } }
        let module = Module::new(
            fresh_node_id(),
            Some(Symbol::new("test")),
            vec![Decl::Module(inline_module(
                "Types",
                vec![Decl::Struct(simple_struct("Point"))],
            ))],
        );

        let input = TestModuleInput::new(db, module);
        verify_module_struct_namespace(db, input);
    }

    #[salsa_test]
    fn test_module_enum_registered_in_namespace(db: &dyn salsa::Database) {
        // mod Types { enum Color { Red, Green, Blue } }
        let module = Module::new(
            fresh_node_id(),
            Some(Symbol::new("test")),
            vec![Decl::Module(inline_module(
                "Types",
                vec![Decl::Enum(simple_enum("Color", &["Red", "Green", "Blue"]))],
            ))],
        );

        let input = TestModuleInput::new(db, module);
        verify_module_enum_namespace(db, input);
    }

    #[salsa_test]
    fn test_module_multiple_definitions(db: &dyn salsa::Database) {
        // mod Utils {
        //     fn helper() -> Int { 1 }
        //     fn another() -> Int { 2 }
        //     struct Data { value: Int }
        // }
        let module = Module::new(
            fresh_node_id(),
            Some(Symbol::new("test")),
            vec![Decl::Module(inline_module(
                "Utils",
                vec![
                    Decl::Function(simple_func("helper")),
                    Decl::Function(simple_func("another")),
                    Decl::Struct(simple_struct("Data")),
                ],
            ))],
        );

        let input = TestModuleInput::new(db, module);
        verify_module_multiple_definitions(db, input);
    }

    #[salsa_test]
    fn test_multiple_modules_separate_namespaces(db: &dyn salsa::Database) {
        // mod A { fn foo() -> Int { 1 } }
        // mod B { fn foo() -> Int { 2 } }
        let module = Module::new(
            fresh_node_id(),
            Some(Symbol::new("test")),
            vec![
                Decl::Module(inline_module("A", vec![Decl::Function(simple_func("foo"))])),
                Decl::Module(inline_module("B", vec![Decl::Function(simple_func("foo"))])),
            ],
        );

        let input = TestModuleInput::new(db, module);
        verify_multiple_modules_namespaces(db, input);
    }

    #[salsa_test]
    fn test_nested_module_enum_namespace(db: &dyn salsa::Database) {
        // mod Outer { enum Inner { Variant } }
        // Should create:
        // - Outer::Inner (the enum type)
        // - Outer::Variant (the variant, also available directly under Outer)
        // - Outer::Inner::Variant (the variant via enum namespace)
        let module = Module::new(
            fresh_node_id(),
            Some(Symbol::new("test")),
            vec![Decl::Module(inline_module(
                "Outer",
                vec![Decl::Enum(simple_enum("Inner", &["Variant"]))],
            ))],
        );

        let input = TestModuleInput::new(db, module);
        verify_nested_module_enum_namespace(db, input);
    }

    #[salsa_test]
    fn test_module_does_not_override_parent_definitions(db: &dyn salsa::Database) {
        // fn foo() -> Int { 1 }
        // mod M { fn bar() -> Int { 2 } }
        let module = Module::new(
            fresh_node_id(),
            Some(Symbol::new("test")),
            vec![
                Decl::Function(simple_func("foo")),
                Decl::Module(inline_module("M", vec![Decl::Function(simple_func("bar"))])),
            ],
        );

        let input = TestModuleInput::new(db, module);
        verify_module_no_override_parent(db, input);
    }

    // =========================================================================
    // Extern function tests
    // =========================================================================

    #[salsa::tracked]
    fn verify_extern_function_registered(db: &dyn salsa::Database, input: TestModuleInput) {
        let env = build_env(db, input.module(db));

        let binding = env.lookup(Symbol::new("__bytes_len"));
        assert!(binding.is_some(), "__bytes_len should be registered");
        assert!(
            matches!(binding, Some(Binding::Function { .. })),
            "__bytes_len should be a function binding"
        );
    }

    #[salsa_test]
    fn test_extern_function_registered_as_binding(db: &dyn salsa::Database) {
        let module = Module::new(
            fresh_node_id(),
            Some(Symbol::new("test")),
            vec![Decl::ExternFunction(ExternFuncDecl {
                id: fresh_node_id(),
                is_pub: false,
                name: Symbol::new("__bytes_len"),
                abi: Symbol::new("intrinsic"),
                params: vec![ParamDecl {
                    id: fresh_node_id(),
                    name: Symbol::new("bytes"),
                    ty: Some(TypeAnnotation {
                        id: fresh_node_id(),
                        kind: TypeAnnotationKind::Named(Symbol::new("Bytes")),
                    }),
                    local_id: None,
                }],
                return_ty: TypeAnnotation {
                    id: fresh_node_id(),
                    kind: TypeAnnotationKind::Named(Symbol::new("Int")),
                },
            })],
        );

        let input = TestModuleInput::new(db, module);
        verify_extern_function_registered(db, input);
    }
}
