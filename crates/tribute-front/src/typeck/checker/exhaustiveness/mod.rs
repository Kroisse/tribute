//! Case exhaustiveness and reachability checking.
//!
//! Source patterns are lowered into a pattern matrix and checked for
//! usefulness: a case is exhaustive when a wildcard is not useful against its
//! unguarded arms, and an arm is unreachable when it is not useful against the
//! unguarded arms before it.

mod matrix;

use std::collections::HashMap;

use itertools::Itertools;
use salsa::Accumulator;
use tribute_core::{CompilationPhase, Diagnostic, DiagnosticSeverity};
use tribute_ir::ModulePathExt;
use trunk_ir::Symbol;

use self::matrix::{Analyzer, Ctor, Family, FamilyId, GiveUp, Pat, VariantInfo, Witness};
use super::TypeChecker;
use crate::ast::{
    Arm, CtorId, LiteralPattern, NodeId, Pattern, PatternKind, ResolvedRef, Type, TypeKind,
    TypedRef,
};

/// Missing patterns listed in a diagnostic before it is cut short.
const MAX_REPORTED_WITNESSES: usize = 3;

impl<'db> TypeChecker<'db> {
    /// Check that a case expression is exhaustive and warn about arms that
    /// earlier arms already cover.
    ///
    /// Returns whether the case was proved exhaustive.
    pub(super) fn check_exhaustiveness(
        &mut self,
        scrutinee_ty: Type<'db>,
        arms: &[Arm<TypedRef<'db>>],
        span_node_id: NodeId,
    ) -> bool {
        let report = self.exhaustiveness_reported.insert(span_node_id);
        if arms.is_empty() {
            if report {
                self.report_case(
                    span_node_id,
                    "non-exhaustive case expression: no patterns provided",
                    DiagnosticSeverity::Error,
                );
            }
            return false;
        }

        let mut lowering = PatternLowering::new(self);
        let patterns: Vec<Pat> = arms
            .iter()
            .map(|arm| lowering.lower(&arm.pattern))
            .collect();
        let PatternLowering {
            families,
            family_names,
            saw_error,
            unanalyzable,
            ..
        } = lowering;
        if saw_error {
            return false;
        }
        if unanalyzable {
            if report {
                self.report_unverified(span_node_id);
            }
            return false;
        }
        if let Some(scrutinee) = self.nominal_name(scrutinee_ty)
            && patterns.iter().any(|pattern| {
                matches!(pattern, Pat::Ctor(Ctor::Variant { family, .. }, _)
                    if family_names[family.0] != scrutinee)
            })
        {
            return false;
        }

        let mut analyzer = Analyzer::new(&families);
        let outcome = check_arms(&mut analyzer, arms, &patterns);
        let (unreachable, missing) = match outcome {
            Ok(outcome) => outcome,
            Err(GiveUp::Conflict) => return false,
            Err(GiveUp::Unsupported | GiveUp::Budget) => {
                if report {
                    self.report_unverified(span_node_id);
                }
                return false;
            }
        };
        if report {
            for index in unreachable {
                self.report_case(
                    arms[index].pattern.id,
                    "unreachable pattern",
                    DiagnosticSeverity::Warning,
                );
            }
            if !missing.is_empty() {
                let shown = missing
                    .iter()
                    .take(MAX_REPORTED_WITNESSES)
                    .map(|witness| analyzer.render(witness));
                let more = if missing.len() > MAX_REPORTED_WITNESSES {
                    ", ..."
                } else {
                    ""
                };
                self.report_case(
                    span_node_id,
                    &format!(
                        "non-exhaustive case expression: missing patterns: {}{more}",
                        shown.format(", ")
                    ),
                    DiagnosticSeverity::Error,
                );
            }
        }
        missing.is_empty()
    }

    /// The nominal type name of a scrutinee, if it has one.
    fn nominal_name(&self, ty: Type<'db>) -> Option<Symbol> {
        match ty.kind(self.db()) {
            TypeKind::Named { name, .. } => Some(*name),
            TypeKind::App { ctor, .. } => match ctor.kind(self.db()) {
                TypeKind::Named { name, .. } => Some(*name),
                _ => None,
            },
            _ => None,
        }
    }

    fn report_unverified(&self, node: NodeId) {
        self.report_case(
            node,
            "exhaustiveness check: unable to verify all cases are covered",
            DiagnosticSeverity::Warning,
        );
    }

    fn report_case(&self, node: NodeId, message: &str, severity: DiagnosticSeverity) {
        Diagnostic::new(
            message,
            self.get_span(node),
            severity,
            CompilationPhase::TypeChecking,
        )
        .accumulate(self.db());
    }
}

/// Indices of unreachable arms, and witnesses no unguarded arm matches.
fn check_arms<'db>(
    analyzer: &mut Analyzer<'_>,
    arms: &[Arm<TypedRef<'db>>],
    patterns: &[Pat],
) -> Result<(Vec<usize>, Vec<Witness>), GiveUp> {
    let mut covering: Vec<Vec<Pat>> = Vec::new();
    let mut unreachable = Vec::new();
    for (index, (arm, pattern)) in arms.iter().zip(patterns).enumerate() {
        let row = vec![pattern.clone()];
        if !analyzer.is_useful(&covering, &row)? {
            unreachable.push(index);
        }
        if arm.guard.is_none() {
            covering.push(row);
        }
    }
    let missing = analyzer
        .missing(&covering, 1, MAX_REPORTED_WITNESSES + 1)?
        .into_iter()
        .filter_map(|mut witness| witness.pop())
        .collect();
    Ok((unreachable, missing))
}

/// Lowers source patterns into matrix patterns, collecting the constructor
/// families they mention.
struct PatternLowering<'a, 'db> {
    checker: &'a TypeChecker<'db>,
    families: Vec<Family>,
    family_names: Vec<Symbol>,
    family_ids: HashMap<Symbol, FamilyId>,
    /// A pattern did not resolve to a constructor. The case is treated as
    /// non-exhaustive without a diagnostic here; resolution usually reports it.
    saw_error: bool,
    /// A pattern cannot be modeled by the matrix.
    unanalyzable: bool,
}

impl<'a, 'db> PatternLowering<'a, 'db> {
    fn new(checker: &'a TypeChecker<'db>) -> Self {
        Self {
            checker,
            families: Vec::new(),
            family_names: Vec::new(),
            family_ids: HashMap::new(),
            saw_error: false,
            unanalyzable: false,
        }
    }

    fn lower(&mut self, pattern: &Pattern<TypedRef<'db>>) -> Pat {
        match &*pattern.kind {
            PatternKind::Wildcard | PatternKind::Bind { .. } => Pat::Wild,
            PatternKind::As { pattern, .. } => self.lower(pattern),
            PatternKind::Literal(LiteralPattern::Bool(value)) => {
                Pat::Ctor(Ctor::Bool(*value), Vec::new())
            }
            PatternKind::Literal(LiteralPattern::Unit) => Pat::Ctor(Ctor::Unit, Vec::new()),
            PatternKind::Literal(literal) => Pat::Ctor(Ctor::Literal(literal.clone()), Vec::new()),
            PatternKind::Tuple(elements) => Pat::Ctor(
                Ctor::Tuple(elements.len()),
                elements.iter().map(|element| self.lower(element)).collect(),
            ),
            PatternKind::List(elements) => Pat::List {
                prefix: elements.iter().map(|element| self.lower(element)).collect(),
                rest: false,
            },
            PatternKind::ListRest { head, .. } => Pat::List {
                prefix: head.iter().map(|element| self.lower(element)).collect(),
                rest: true,
            },
            PatternKind::Variant { ctor, fields } => {
                let ResolvedRef::Constructor { id, variant } = ctor.resolved else {
                    self.saw_error = true;
                    return Pat::Wild;
                };
                let Some((ctor, arity)) = self.constructor(id, variant) else {
                    self.unanalyzable = true;
                    return Pat::Wild;
                };
                // Fields are positional, as in lowering: missing ones match
                // anything and extra ones are dropped.
                let mut fields: Vec<Pat> = fields.iter().map(|field| self.lower(field)).collect();
                fields.resize(arity, Pat::Wild);
                Pat::Ctor(ctor, fields)
            }
            PatternKind::Record {
                type_name: None,
                fields,
                ..
            } if fields.iter().all(|field| {
                field.pattern.as_ref().is_none_or(|pattern| {
                    matches!(
                        &*pattern.kind,
                        PatternKind::Wildcard | PatternKind::Bind { .. }
                    )
                })
            }) =>
            {
                Pat::Wild
            }
            // Record fields are matched by name, which the matrix cannot
            // model, and a named record may be one variant of an enum.
            PatternKind::Record { .. } => {
                self.unanalyzable = true;
                Pat::Wild
            }
            PatternKind::Error => {
                self.saw_error = true;
                Pat::Wild
            }
        }
    }

    /// The matrix constructor for a source constructor, with its arity.
    fn constructor(&mut self, id: CtorId<'db>, variant: Symbol) -> Option<(Ctor, usize)> {
        let db = self.checker.db();
        let (arity, result) = self.constructor_shape(id)?;
        let TypeKind::Named { name, .. } = result.kind(db) else {
            return None;
        };
        let name = *name;
        let family = match self.family_ids.get(&name) {
            Some(family) => *family,
            None => {
                let family = self.family(name, id)?;
                let family_id = FamilyId(self.families.len());
                self.families.push(family);
                self.family_names.push(name);
                self.family_ids.insert(name, family_id);
                family_id
            }
        };
        let index = self.families[family.0]
            .variants
            .iter()
            .position(|info| info.name == variant)?;
        Some((Ctor::Variant { family, index }, arity))
    }

    /// The constructors of the nominal type `name`, which `id` constructs.
    fn family(&self, name: Symbol, id: CtorId<'db>) -> Option<Family> {
        let db = self.checker.db();
        // Test for an enum first: a variant may share its enum's name.
        if let Some(variants) = self.checker.env.lookup_enum_variants(name) {
            // Variants are registered in the enum's module, not under the enum.
            let variants = variants
                .iter()
                .map(|&variant| {
                    let qualified = name
                        .parent_path()
                        .map_or(variant, |module| module.join_path(variant));
                    let arity = self
                        .constructor_shape(CtorId::new(db, qualified))
                        .map_or(0, |(arity, _)| arity);
                    VariantInfo {
                        name: variant,
                        arity,
                    }
                })
                .collect();
            return Some(Family { variants });
        }
        if id.qualified(db) != name {
            return None;
        }
        let (arity, _) = self.constructor_shape(id)?;
        Some(Family {
            variants: vec![VariantInfo {
                name: name.last_segment(),
                arity,
            }],
        })
    }

    /// A constructor's field count and the type it constructs.
    fn constructor_shape(&self, id: CtorId<'db>) -> Option<(usize, Type<'db>)> {
        let db = self.checker.db();
        let body = self.checker.env.lookup_constructor(id)?.body(db);
        Some(match body.kind(db) {
            TypeKind::Func { params, result, .. } => (params.len(), *result),
            _ => (0, body),
        })
    }
}
