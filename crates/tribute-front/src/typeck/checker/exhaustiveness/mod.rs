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
        let mut any_unanalyzable = false;
        // Each arm's pattern, and whether it counts toward coverage. An arm
        // whose pattern cannot be modeled is kept, with wildcards standing
        // for the unmodeled parts, but covers nothing, like a guarded arm.
        let rows: Vec<(Pat, bool)> = arms
            .iter()
            .map(|arm| {
                lowering.unanalyzable = false;
                let pattern = lowering.lower(&arm.pattern);
                any_unanalyzable |= lowering.unanalyzable;
                (pattern, arm.guard.is_none() && !lowering.unanalyzable)
            })
            .collect();
        let PatternLowering {
            families,
            family_names,
            saw_error,
            ..
        } = lowering;
        if saw_error {
            return false;
        }
        if let Some(scrutinee) = self.nominal_name(scrutinee_ty)
            && rows.iter().any(|(pattern, _)| {
                matches!(pattern, Pat::Ctor(Ctor::Variant { family, .. }, _)
                    if family_names[family.0] != scrutinee)
            })
        {
            return false;
        }

        let mut analyzer = Analyzer::new(&families);
        let outcome = check_arms(&mut analyzer, &rows);
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
        if !missing.is_empty() && any_unanalyzable {
            // An arm left out of coverage may match the missing values.
            if report {
                self.report_unverified(span_node_id);
            }
            return false;
        }
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

/// Indices of unreachable arms, and witnesses no covering arm matches.
///
/// Each arm is its pattern and whether it counts toward coverage.
fn check_arms(
    analyzer: &mut Analyzer<'_>,
    arms: &[(Pat, bool)],
) -> Result<(Vec<usize>, Vec<Witness>), GiveUp> {
    let mut covering: Vec<Vec<Pat>> = Vec::new();
    let mut unreachable = Vec::new();
    for (index, (pattern, covers)) in arms.iter().enumerate() {
        let row = vec![pattern.clone()];
        if !analyzer.is_useful(&covering, &row)? {
            unreachable.push(index);
        }
        if *covers {
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
    /// The pattern being lowered cannot be modeled by the matrix; the
    /// unmodeled parts are lowered as wildcards.
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
                // Type checking rejects a wrong field count; pad or drop
                // fields so that such a case is still analyzed.
                let mut fields: Vec<Pat> = fields.iter().map(|field| self.lower(field)).collect();
                fields.resize(arity, Pat::Wild);
                Pat::Ctor(ctor, fields)
            }
            PatternKind::Record {
                type_name: Some(type_name),
                fields,
                ..
            } => {
                let ResolvedRef::Constructor { id, variant } = type_name.resolved else {
                    self.saw_error = true;
                    return Pat::Wild;
                };
                let Some((ctor, arity)) = self.constructor(id, variant) else {
                    self.unanalyzable = true;
                    return Pat::Wild;
                };
                // Place each field at its declared position; omitted fields
                // match anything. Type checking reports names that do not fit.
                let mut positional = vec![Pat::Wild; arity];
                let declared = self.checker.env.lookup_constructor_field_names(id);
                for field in fields {
                    let index = declared
                        .and_then(|declared| declared.iter().position(|name| *name == field.name))
                        .filter(|index| *index < arity);
                    let (Some(index), Some(pattern)) = (index, &field.pattern) else {
                        self.saw_error = true;
                        return Pat::Wild;
                    };
                    positional[index] = self.lower(pattern);
                }
                Pat::Ctor(ctor, positional)
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
            // A record without a constructor name has no known shape.
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
                    let (arity, _) = self.constructor_shape(CtorId::new(db, qualified))?;
                    Some(VariantInfo {
                        name: variant,
                        arity,
                    })
                })
                .collect::<Option<_>>()?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn bool_(value: bool) -> Pat {
        Pat::Ctor(Ctor::Bool(value), Vec::new())
    }

    fn missing_count(arms: &[(Pat, bool)]) -> usize {
        check_arms(&mut Analyzer::new(&[]), arms).unwrap().1.len()
    }

    #[test]
    fn arms_outside_coverage_do_not_block_a_catch_all() {
        // An unanalyzable arm is lowered with wildcards but covers nothing.
        assert_eq!(missing_count(&[(Pat::Wild, false), (Pat::Wild, true)]), 0);
        assert_eq!(missing_count(&[(Pat::Wild, false), (bool_(true), true)]), 1);
    }
}
