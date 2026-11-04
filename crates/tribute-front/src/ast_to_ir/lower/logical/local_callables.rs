//! Local callables: the named-function aliases of a function body, and the
//! captures of a lambda.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use trunk_ir::context::IrContext;
use trunk_ir::refs::ValueRef;

use crate::ast::{
    AbilityRow, Expr, ExprKind, FuncDefId, LocalId, NodeId, PatternKind, ResolvedRef, RowVar, Stmt,
    TypedRef,
};
use crate::ast_to_ir::context::IrLoweringCtx;

use super::declarations::Declarations;

/// What a `let` binds a named-function alias to: the function itself, or
/// another alias.
#[derive(Clone, Copy)]
enum Target<'db> {
    Function(FuncDefId<'db>),
    Local(LocalId),
}

/// One alias: the node of the value its `let` binds, and what that value
/// names.
#[derive(Clone, Copy)]
struct Link<'db> {
    value: NodeId,
    target: Target<'db>,
}

/// The locals of one function body that a `let` binds to a named function,
/// directly or through other such locals (`new-plans/ir.md`, 지역 람다의
/// 양화와 인스턴스화). A use of one references the function again.
#[derive(Default)]
pub(super) struct Aliases<'db> {
    links: HashMap<LocalId, Link<'db>>,
}

impl<'db> Aliases<'db> {
    pub(super) fn collect(body: &Expr<TypedRef<'db>>) -> Self {
        let mut aliases = Self::default();
        body.for_each(|expr| {
            let ExprKind::Block { stmts, .. } = &*expr.kind else {
                return;
            };
            for stmt in stmts {
                let Stmt::Let { pattern, value, .. } = stmt else {
                    continue;
                };
                let (
                    PatternKind::Bind {
                        local_id: Some(local),
                        ..
                    },
                    ExprKind::Var(reference),
                ) = (&*pattern.kind, &*value.kind)
                else {
                    continue;
                };
                let target = match reference.resolved {
                    ResolvedRef::Function { id } => Target::Function(id),
                    ResolvedRef::Local { id, .. } if aliases.links.contains_key(&id) => {
                        Target::Local(id)
                    }
                    _ => continue,
                };
                aliases.links.insert(
                    *local,
                    Link {
                        value: value.id,
                        target,
                    },
                );
            }
        });
        aliases
    }

    /// Whether `local` is an alias. Its `let` lowers nothing: every use
    /// references the function again.
    pub(super) fn contains(&self, local: LocalId) -> bool {
        self.links.contains_key(&local)
    }

    /// The function the use `node` of `local` references, and the rows that
    /// reference passes, if `local` is an alias. Each alias's instance at
    /// the use substitutes the rows its scheme quantifies, from the use back
    /// to the function reference its `let` chain starts with.
    pub(super) fn reference(
        &self,
        db: &'db dyn salsa::Database,
        local: LocalId,
        node: NodeId,
        declarations: &Declarations<'db>,
    ) -> Option<(FuncDefId<'db>, Vec<AbilityRow<'db>>)> {
        let mut link = self.links.get(&local)?;
        let mut rows = instance_rows(db, declarations, node, &HashMap::default());
        loop {
            match link.target {
                Target::Function(function) => {
                    let arguments = declarations
                        .function_instances
                        .get(&link.value)
                        .map(|instance| {
                            instance
                                .row_arguments
                                .iter()
                                .map(|row| substitute(db, *row, &rows))
                                .collect()
                        })
                        .unwrap_or_default();
                    return Some((function, arguments));
                }
                Target::Local(next) => {
                    rows = instance_rows(db, declarations, link.value, &rows);
                    link = &self.links[&next];
                }
            }
        }
    }
}

/// The row each variable of the local scheme that `node` instantiates
/// takes, after `outer`.
fn instance_rows<'db>(
    db: &'db dyn salsa::Database,
    declarations: &Declarations<'db>,
    node: NodeId,
    outer: &HashMap<RowVar, AbilityRow<'db>>,
) -> HashMap<RowVar, AbilityRow<'db>> {
    let Some(instance) = declarations.local_instances.get(&node) else {
        return HashMap::default();
    };
    instance
        .scheme
        .effect_params(db)
        .iter()
        .copied()
        .zip(
            instance
                .row_arguments
                .iter()
                .map(|row| substitute(db, *row, outer)),
        )
        .collect()
}

/// `row` with its tail replaced by the row `rows` gives it.
fn substitute<'db>(
    db: &'db dyn salsa::Database,
    row: AbilityRow<'db>,
    rows: &HashMap<RowVar, AbilityRow<'db>>,
) -> AbilityRow<'db> {
    let Some(replacement) = row.rest(db).and_then(|tail| rows.get(tail)) else {
        return row;
    };
    let mut instances = row.instances(db).to_vec();
    for instance in replacement.instances(db) {
        if !instances.contains(instance) {
            instances.push(instance.clone());
        }
    }
    AbilityRow::new(db, instances, replacement.rest(db).copied())
}

/// Source order of actual external SSA uses, with duplicates removed.
pub(super) fn captures(
    ctx: &IrLoweringCtx<'_>,
    ir: &IrContext,
    region: trunk_ir::refs::RegionRef,
) -> Vec<ValueRef> {
    fn collect(
        ir: &IrContext,
        region: trunk_ir::refs::RegionRef,
        defined: &mut HashSet<ValueRef>,
        uses: &mut Vec<ValueRef>,
    ) {
        for block in &ir.region(region).blocks {
            for index in 0..ir.block(*block).args.len() {
                defined.insert(ir.block_arg(*block, index as u32));
            }
            for op in &ir.block(*block).ops {
                defined.extend(ir.op_results(*op));
                uses.extend(ir.op_operands(*op));
                for nested in ir.op_regions(*op) {
                    collect(ir, nested, defined, uses);
                }
            }
        }
    }
    let mut defined = HashSet::default();
    let mut uses = Vec::new();
    collect(ir, region, &mut defined, &mut uses);
    let mut seen = HashSet::default();
    let external: HashSet<_> = uses
        .iter()
        .copied()
        .filter(|value| !defined.contains(value))
        .collect();
    ctx.all_bindings()
        .map(|(_, _, value)| value)
        .chain(uses)
        .filter(|value| external.contains(value) && seen.insert(*value))
        .collect()
}
