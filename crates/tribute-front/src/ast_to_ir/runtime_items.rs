//! The frontend check of the compiler-owned runtime.
//!
//! Runtime code is the prelude module `std::runtime` (`new-plans/modules.md`,
//! "Compiler 소유 runtime 모듈"). Every runtime item lowering uses must have a
//! declaration of its kind at its canonical path, and every function of the
//! runtime module must keep to the runtime code constraints. The source
//! declarations stay the only definition of the items' signatures and fields.

use rustc_hash::FxHashMap as HashMap;

use tribute_core::diagnostic::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_ir::runtime_items::{RuntimeItem, RuntimeItemKind};
use trunk_ir::SymbolPath;

use crate::ast::visit::{Visit, walk_expr};
use crate::ast::{
    Decl, DeclTable, Expr, ExprKind, FuncDecl, NodeId, ResolvedRef, SpanMaps, TypeKind, TypeScheme,
    TypedRef,
};

/// The runtime structs lowering uses, by declaration ID, with the canonical
/// path they are published under.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RuntimeTypes(pub HashMap<NodeId, SymbolPath>);

/// A runtime declaration that breaks the runtime contract.
#[derive(Clone, Debug, PartialEq, Eq, derive_more::Display)]
#[display("runtime `{path}` {kind}")]
pub struct RuntimeItemError {
    /// The declaration, or the module when the declaration is missing.
    pub node: NodeId,
    pub path: String,
    pub kind: RuntimeItemErrorKind,
}

impl RuntimeItemError {
    /// The source diagnostic of this error, at its declaration.
    pub fn diagnostic(&self, spans: &SpanMaps<'_>) -> Diagnostic {
        Diagnostic::new(
            self.to_string(),
            spans.get_or_default(self.node),
            DiagnosticSeverity::Error,
            CompilationPhase::Lowering,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, derive_more::Display)]
pub enum RuntimeItemErrorKind {
    #[display("has no declaration")]
    Missing,
    #[display("must be declared as a {_0}")]
    WrongKind(&'static str),
    #[display("must be monomorphic")]
    Polymorphic,
    #[display("must have an empty ability row, written `->{{}}`")]
    NotDirect,
    #[display("must not call a closure value")]
    ClosureCall,
    #[display("must not install a handler")]
    Handler,
    #[display("may recurse only in tail position")]
    NonTailRecursion,
}

/// Typed declarations the runtime is checked against.
pub struct RuntimeInputs<'a, 'db> {
    pub module: &'a DeclTable<TypedRef<'db>>,
    pub function_types: &'a HashMap<SymbolPath, TypeScheme<'db>>,
}

/// Check `items` and every function of `runtime_module` in `inputs`.
///
/// Every failure is reported, so that one compilation shows them all.
pub fn check_runtime<'db>(
    db: &'db dyn salsa::Database,
    inputs: &RuntimeInputs<'_, 'db>,
    runtime_module: &[&str],
    items: &[RuntimeItem],
) -> Result<RuntimeTypes, Vec<RuntimeItemError>> {
    let mut types = RuntimeTypes::default();
    let mut errors = Vec::new();
    for item in items {
        let path = SymbolPath::new(item.path.iter().copied());
        let error = |node, kind| RuntimeItemError {
            node,
            path: path.to_string(),
            kind,
        };
        let declaration = inputs.module.entries.iter().find_map(|entry| {
            let (name, id) = declared_item(&entry.decl)?;
            (entry.module.child(name.clone()) == path).then_some((&entry.decl, id))
        });
        match (item.kind, declaration) {
            (_, None) => errors.push(error(inputs.module.id, RuntimeItemErrorKind::Missing)),
            (RuntimeItemKind::Function, Some((Decl::Function(_), _))) => {}
            (RuntimeItemKind::Struct, Some((Decl::Struct(structure), _))) => {
                if structure.type_params.is_empty() {
                    types.0.insert(structure.id, path.clone());
                } else {
                    errors.push(error(structure.id, RuntimeItemErrorKind::Polymorphic));
                }
            }
            (kind, Some((_, id))) => errors.push(error(
                id,
                RuntimeItemErrorKind::WrongKind(match kind {
                    RuntimeItemKind::Function => "function with a body",
                    RuntimeItemKind::Struct => "struct",
                }),
            )),
        }
    }
    for entry in &inputs.module.entries {
        let Decl::Function(function) = &entry.decl else {
            continue;
        };
        if !in_module(&entry.module, runtime_module) {
            continue;
        }
        let path = entry.module.child(function.name.clone());
        if let Err((node, kind)) = check_function(db, inputs, &path, function) {
            errors.push(RuntimeItemError {
                node,
                path: path.to_string(),
                kind,
            });
        }
    }
    if errors.is_empty() {
        Ok(types)
    } else {
        Err(errors)
    }
}

/// The name and ID of a declaration that a path can name.
fn declared_item<V>(decl: &Decl<V>) -> Option<(&trunk_ir::Symbol, NodeId)> {
    match decl {
        Decl::Function(function) => Some((&function.name, function.id)),
        Decl::ExternFunction(function) => Some((&function.name, function.id)),
        Decl::Struct(structure) => Some((&structure.name, structure.id)),
        Decl::Enum(enumeration) => Some((&enumeration.name, enumeration.id)),
        Decl::Ability(ability) => Some((&ability.name, ability.id)),
        Decl::Use(_) => None,
    }
}

/// The runtime code constraints on one function.
fn check_function<'db>(
    db: &'db dyn salsa::Database,
    inputs: &RuntimeInputs<'_, 'db>,
    path: &SymbolPath,
    function: &FuncDecl<TypedRef<'db>>,
) -> Result<(), (NodeId, RuntimeItemErrorKind)> {
    let at_declaration = |kind| (function.id, kind);
    let scheme = inputs
        .function_types
        .get(path)
        .expect("a typechecked function has a scheme");
    if !function.type_params.is_empty() || !scheme.type_params(db).is_empty() {
        return Err(at_declaration(RuntimeItemErrorKind::Polymorphic));
    }
    let pure = match scheme.body(db).kind(db) {
        TypeKind::Func { effect, .. } => effect.is_pure(db),
        _ => true,
    };
    if !scheme.effect_params(db).is_empty() || !pure {
        return Err(at_declaration(RuntimeItemErrorKind::NotDirect));
    }
    let mut body = BodyCheck {
        db,
        path,
        tail: true,
        lambda_depth: 0,
        violation: None,
    };
    body.visit_expr(&function.body);
    body.violation.map_or(Ok(()), Err)
}

/// Whether `module` is the module `prefix` or one nested in it.
fn in_module(module: &SymbolPath, prefix: &[&str]) -> bool {
    module.len() >= prefix.len()
        && module
            .iter()
            .zip(prefix)
            .all(|(component, &expected)| component == expected)
}

/// The body constraints of a runtime function: it calls no closure value,
/// installs no handler, and calls itself only in tail position.
///
/// Recursion through other functions is not detected.
struct BodyCheck<'a, 'db> {
    db: &'db dyn salsa::Database,
    path: &'a SymbolPath,
    tail: bool,
    lambda_depth: usize,
    violation: Option<(NodeId, RuntimeItemErrorKind)>,
}

impl BodyCheck<'_, '_> {
    fn report(&mut self, node: NodeId, kind: RuntimeItemErrorKind) {
        self.violation.get_or_insert((node, kind));
    }
}

impl<'ast, 'db: 'ast> Visit<'ast, TypedRef<'db>> for BodyCheck<'_, 'db> {
    fn visit_expr(&mut self, expr: &'ast Expr<TypedRef<'db>>) {
        let tail = std::mem::replace(&mut self.tail, false);
        match &*expr.kind {
            ExprKind::Call { callee, args, .. } => {
                match &*callee.kind {
                    ExprKind::Var(TypedRef {
                        resolved: ResolvedRef::Function { id },
                        ..
                    }) => {
                        let recursive = id.path(self.db) == self.path;
                        if recursive && (!tail || self.lambda_depth > 0) {
                            self.report(expr.id, RuntimeItemErrorKind::NonTailRecursion);
                        }
                    }
                    ExprKind::Functions { .. }
                    | ExprKind::Var(TypedRef {
                        resolved: ResolvedRef::AbilityOp { .. } | ResolvedRef::Constructor { .. },
                        ..
                    }) => {}
                    _ => {
                        self.report(expr.id, RuntimeItemErrorKind::ClosureCall);
                        self.visit_expr(callee);
                    }
                }
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            ExprKind::Block { stmts, value } => {
                for stmt in stmts {
                    self.visit_stmt(stmt);
                }
                self.tail = tail;
                self.visit_expr(value);
            }
            ExprKind::Case { scrutinee, arms } => {
                self.visit_expr(scrutinee);
                for arm in arms {
                    self.visit_pattern(&arm.pattern);
                    if let Some(guard) = &arm.guard {
                        self.visit_expr(guard);
                    }
                    self.tail = tail;
                    self.visit_expr(&arm.body);
                }
            }
            ExprKind::Become { call } => {
                self.tail = tail;
                self.visit_expr(call);
            }
            ExprKind::Lambda { body, .. } => {
                self.lambda_depth += 1;
                self.visit_expr(body);
                self.lambda_depth -= 1;
            }
            ExprKind::Handle { .. } => {
                self.report(expr.id, RuntimeItemErrorKind::Handler);
                walk_expr(self, expr);
            }
            _ => walk_expr(self, expr),
        }
        self.tail = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_prefix_matches_whole_components() {
        let runtime = ["std", "runtime"];
        assert!(in_module(&SymbolPath::new(["std", "runtime"]), &runtime));
        assert!(in_module(
            &SymbolPath::new(["std", "runtime", "evidence"]),
            &runtime
        ));
        assert!(!in_module(&SymbolPath::new(["std", "runtimes"]), &runtime));
        assert!(!in_module(&SymbolPath::new(["std"]), &runtime));
    }
}
