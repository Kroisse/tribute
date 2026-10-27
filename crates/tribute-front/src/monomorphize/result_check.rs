//! The check of the specialization output (`new-plans/generics.md`,
//! "특수화 결과"). A violation is a compiler inconsistency, not a source
//! error.

use std::ops::ControlFlow;

use rustc_hash::FxHashSet as HashSet;

use trunk_ir::SymbolPath;

use super::templates::DeclNodes;
use super::{InstanceError, InstanceErrorKind};
use crate::ast::fold::{TypeVisitable, TypeVisitor};
use crate::ast::visit::{RefSite, Refs, Visit, walk_decl};
use crate::ast::{Decl, DeclTable, NodeId, ResolvedRef, Type, TypeKind, TypeScheme, TypedRef};
use crate::typeck::{ConstructorTypeMetadata, ExpressionTypeMetadata, FieldFunctionKind};

/// The output contract a specialization result breaks.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SpecializationInvariant {
    /// A scheme `BoundVar` remains in the place `location` names.
    TypeVariable { location: String },
    /// A function reference names no output declaration.
    UnresolvedReference { path: String },
    /// Metadata is keyed by a node the output AST does not have.
    OrphanMetadata { table: &'static str },
}

/// Check the specialized `module`, its schemes, and its metadata against
/// the output contract.
pub(super) fn check<'db>(
    db: &'db dyn salsa::Database,
    module: &DeclTable<TypedRef<'db>>,
    function_types: &[(SymbolPath, TypeScheme<'db>)],
    constructor_types: &ConstructorTypeMetadata<'db>,
    expressions: &ExpressionTypeMetadata<'db>,
) -> Vec<InstanceError> {
    let mut check = Check {
        db,
        errors: Vec::new(),
    };
    for (path, scheme) in function_types {
        check.scheme(module.id, scheme, || format!("the scheme of `{path}`"));
    }
    for (ctor, scheme) in &constructor_types.schemes {
        check.scheme(module.id, scheme, || {
            format!("the scheme of `{}`", ctor.path(db))
        });
    }
    let variants = &constructor_types.specialized_enum_variants;
    check.table("node types", &expressions.node_types);
    check.table("function instances", &expressions.function_instances);
    check.table("local instances", &expressions.local_instances);
    check.table("evidence plans", &expressions.evidence_plans);
    check.table("handler operations", &expressions.handler_operations);
    check.table("perform operations", &expressions.perform_operations);
    check.table("lambda signatures", &expressions.lambda_signatures);
    check.table("enum variant schemes", variants);

    // A function of a dependency whose bodies were not merged is looked up
    // by its scheme alone.
    let mut declared: HashSet<_> = function_types
        .iter()
        .map(|(path, _)| path.clone())
        .collect();
    let mut nodes = DeclNodes::default();
    for entry in &module.entries {
        nodes.visit_decl(&entry.decl);
        match &entry.decl {
            Decl::Function(function) => {
                declared.insert(entry.module.child(function.name.clone()));
            }
            Decl::ExternFunction(function) => {
                declared.insert(entry.module.child(function.name.clone()));
            }
            Decl::Struct(declaration) => {
                let owner = entry.module.child(declaration.name.clone());
                for name in declaration
                    .fields
                    .iter()
                    .filter_map(|field| field.name.as_ref())
                {
                    for kind in [
                        FieldFunctionKind::Get,
                        FieldFunctionKind::Set,
                        FieldFunctionKind::Modify,
                    ] {
                        declared.insert(kind.path(&owner, name));
                    }
                }
            }
            _ => {}
        }
    }
    for entry in &module.entries {
        walk_decl(
            &mut Refs(|site, node, value: &TypedRef<'db>| {
                check.no_bound_var(node, &value.ty, || format!("the type of {node}"));
                if site == RefSite::Var
                    && let ResolvedRef::Function { id } = &value.resolved
                    && !declared.contains(id.path(db))
                {
                    check.fail(
                        node,
                        SpecializationInvariant::UnresolvedReference {
                            path: id.path(db).to_string(),
                        },
                    );
                }
            }),
            &entry.decl,
        );
    }

    let nodes = nodes.0;
    let keys = [
        (
            "node types",
            expressions.node_types.keys().collect::<Vec<_>>(),
        ),
        (
            "function instances",
            expressions.function_instances.keys().collect(),
        ),
        (
            "local instances",
            expressions.local_instances.keys().collect(),
        ),
        (
            "evidence plans",
            expressions.evidence_plans.keys().collect(),
        ),
        (
            "handler operations",
            expressions.handler_operations.keys().collect(),
        ),
        (
            "perform operations",
            expressions.perform_operations.keys().collect(),
        ),
        (
            "lambda signatures",
            expressions.lambda_signatures.keys().collect(),
        ),
        (
            "exhaustive cases",
            expressions.exhaustive_cases.iter().collect(),
        ),
        ("enum variant schemes", variants.keys().collect()),
    ];
    for (table, keys) in keys {
        for node in keys.into_iter().filter(|node| !nodes.contains(node)) {
            check.fail(*node, SpecializationInvariant::OrphanMetadata { table });
        }
    }

    let mut errors = check.errors;
    errors.sort_by_key(|error| error.node);
    errors
}

struct Check<'db> {
    db: &'db dyn salsa::Database,
    errors: Vec<InstanceError>,
}

impl<'db> Check<'db> {
    fn fail(&mut self, node: NodeId, kind: SpecializationInvariant) {
        self.errors.push(InstanceError {
            node,
            kind: InstanceErrorKind::SpecializationInvariant(kind),
        });
    }

    /// Fail at `node` if `value` holds a scheme `BoundVar`.
    fn no_bound_var(
        &mut self,
        node: NodeId,
        value: &impl TypeVisitable<'db>,
        location: impl FnOnce() -> String,
    ) {
        self.bound_within(node, value, 0, location);
    }

    /// Fail at `node` if `scheme` holds a `BoundVar` its own binders do not
    /// bind. Only the schemes of struct and enum templates have binders.
    fn scheme(
        &mut self,
        node: NodeId,
        scheme: &TypeScheme<'db>,
        location: impl FnOnce() -> String,
    ) {
        let binders = scheme.type_params(self.db).len();
        self.bound_within(node, scheme, binders, location);
    }

    fn bound_within(
        &mut self,
        node: NodeId,
        value: &impl TypeVisitable<'db>,
        binders: usize,
        location: impl FnOnce() -> String,
    ) {
        let mut visitor = BoundVars {
            db: self.db,
            binders,
        };
        if value.visit_with(&mut visitor).is_break() {
            let location = location();
            self.fail(node, SpecializationInvariant::TypeVariable { location });
        }
    }

    fn table<V: TypeVisitable<'db>>(
        &mut self,
        table: &'static str,
        entries: &crate::SortedMap<NodeId, V>,
    ) {
        for (node, value) in entries.iter() {
            self.no_bound_var(*node, value, || format!("the {table} of {node}"));
        }
    }
}

/// Breaks at the first `BoundVar` at or past `binders`.
struct BoundVars<'db> {
    db: &'db dyn salsa::Database,
    binders: usize,
}

impl<'db> TypeVisitor<'db> for BoundVars<'db> {
    type Break = ();

    fn db(&self) -> &'db dyn salsa::Database {
        self.db
    }

    fn visit_type(&mut self, ty: Type<'db>) -> ControlFlow<()> {
        match ty.kind(self.db) {
            TypeKind::BoundVar { index } if (*index as usize) >= self.binders => {
                ControlFlow::Break(())
            }
            _ => ty.super_visit_with(self),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typeck::DeclarationTypeMetadata;

    /// The specialization of a fixture that calls `identity` at `Nat`.
    #[salsa::tracked]
    fn specialize_fixture(
        db: &dyn salsa::Database,
        source: crate::SourceCst,
    ) -> super::super::SpecializationOutput<'_> {
        use crate::resolve::{Dependencies, build_env, resolve_with_env};
        let parsed = crate::query::parsed_ast(db, source).expect("fixture parses");
        let ast = parsed.module(db);
        let env = build_env(db, ast, Dependencies::none());
        let resolved = resolve_with_env(db, ast, &env, parsed.span_map(db));
        let checked = crate::typeck::typecheck_module(db, &resolved, parsed.span_map(db));
        super::super::monomorphize_functions(
            db,
            super::super::SpecializationInput::new(db, checked, None),
        )
        .expect("fixture specializes")
    }

    /// The parts of a specialization output the check reads, to damage.
    struct Output<'db> {
        module: DeclTable<TypedRef<'db>>,
        declarations: DeclarationTypeMetadata<'db>,
        expressions: ExpressionTypeMetadata<'db>,
    }

    fn specialized(db: &salsa::DatabaseImpl) -> Output<'_> {
        let source = crate::SourceCst::from_source_str(
            db,
            "result_check.trb",
            "fn identity(value: a) -> a { value }\nfn main() -> Nil { let _ = identity(5) }",
        );
        let output = *specialize_fixture(db, source);
        Output {
            module: output.module(db).clone(),
            declarations: output.declarations(db).clone(),
            expressions: output.expression_types(db).clone(),
        }
    }

    fn violations<'db>(
        db: &'db dyn salsa::Database,
        output: &Output<'db>,
    ) -> Vec<SpecializationInvariant> {
        check(
            db,
            &output.module,
            &output.declarations.function_types,
            &output.declarations.constructor_types,
            &output.expressions,
        )
        .into_iter()
        .map(|error| match error.kind {
            InstanceErrorKind::SpecializationInvariant(violation) => violation,
            kind => panic!("not an output violation: {kind:?}"),
        })
        .collect()
    }

    #[test]
    fn a_specialized_output_keeps_its_contract() {
        let db = salsa::DatabaseImpl::new();
        assert_eq!(violations(&db, &specialized(&db)), []);
    }

    #[test]
    fn a_bound_var_left_in_metadata_breaks_the_contract() {
        let db = salsa::DatabaseImpl::new();
        let mut output = specialized(&db);
        let bound = Type::new(&db, TypeKind::BoundVar { index: 0 });
        let ty = output.expressions.node_types.values_mut().next().unwrap();
        *ty = bound;
        assert!(matches!(
            violations(&db, &output)[..],
            [SpecializationInvariant::TypeVariable { .. }]
        ));
    }

    #[test]
    fn a_reference_to_a_missing_instance_breaks_the_contract() {
        let db = salsa::DatabaseImpl::new();
        let mut output = specialized(&db);
        output.module.entries.retain(|entry| {
            !matches!(&entry.decl, Decl::Function(function) if function.name == "identity$Nat")
        });
        output
            .declarations
            .function_types
            .retain(|(path, _)| *path != ["identity$Nat"]);
        let violations = violations(&db, &output);
        assert!(
            violations.contains(&SpecializationInvariant::UnresolvedReference {
                path: "identity$Nat".to_owned()
            }),
            "{violations:?}"
        );
    }

    #[test]
    fn metadata_of_a_missing_node_breaks_the_contract() {
        let db = salsa::DatabaseImpl::new();
        let mut output = specialized(&db);
        let nat = Type::new(&db, TypeKind::Nat);
        output
            .expressions
            .node_types
            .insert(NodeId::from_raw(usize::MAX), nat);
        assert_eq!(
            violations(&db, &output),
            [SpecializationInvariant::OrphanMetadata {
                table: "node types"
            }]
        );
    }
}
