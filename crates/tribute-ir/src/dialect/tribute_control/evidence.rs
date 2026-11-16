//! Evidence plans: the `evidence_plan` attribute and its checks.

use super::*;

/// Attribute that selects the evidence a call, `resume`, or handle body
/// receives from the evidence of the code around it.
pub const EVIDENCE_PLAN_ATTR: &str = "evidence_plan";

/// One element of an `evidence_plan`. `{mask = instance}`, `{dup = instance}`,
/// and `{push = instance}` name an exact `ability.decl` symbol;
/// `{select = index}` names a row tail of the surrounding callable, and
/// `{tails = [plan, ...]}` holds one plan for each row tail of the callee.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum EvidenceStep {
    /// Hide the top handler of the instance, uncovering the one beneath.
    Mask(SymbolPath),
    /// Stack one more copy of the top handler of the instance.
    Dup(SymbolPath),
    /// Take the evidence of the surrounding callable's row tail `index`.
    Select(u32),
    /// Put the surrounding evidence's top handler of the instance on top of
    /// the selected evidence.
    Push(SymbolPath),
    /// Give each row tail of the callee the surrounding evidence after its
    /// plan.
    Tails(Vec<Vec<EvidenceStep>>),
}

impl EvidenceStep {
    /// The ability instance this step changes.
    pub fn instance(&self) -> Option<&SymbolPath> {
        match self {
            Self::Mask(instance) | Self::Dup(instance) | Self::Push(instance) => Some(instance),
            Self::Select(_) | Self::Tails(_) => None,
        }
    }

    /// Decode one plan element, or say why it is malformed.
    pub fn from_attribute(ctx: &IrContext, attr: &Attribute) -> Result<Self, String> {
        let Attribute::Dict(entries) = attr else {
            return Err("evidence_plan element must be a dictionary".into());
        };
        let Ok((key, value)) = entries.iter().exactly_one() else {
            return Err(format!(
                "evidence_plan element must have exactly one entry, found {}",
                entries.len()
            ));
        };
        let instance = || match value {
            Attribute::SymbolRef(instance) => Ok(instance.clone()),
            _ => Err(format!(
                "evidence_plan {} must name an ability instance",
                ctx.str(key)
            )),
        };
        match ctx.str(key) {
            "mask" => instance().map(Self::Mask),
            "dup" => instance().map(Self::Dup),
            "push" => instance().map(Self::Push),
            "select" => match value {
                Attribute::Int(index) => u32::try_from(*index)
                    .map(Self::Select)
                    .map_err(|_| "evidence_plan select must be a row tail index".into()),
                _ => Err("evidence_plan select must be a row tail index".into()),
            },
            "tails" => {
                let Attribute::List(plans) = value else {
                    return Err("evidence_plan tails must be a list of plans".into());
                };
                plans
                    .iter()
                    .map(|plan| {
                        let Attribute::List(steps) = plan else {
                            return Err("evidence_plan tails must be a list of plans".into());
                        };
                        steps
                            .iter()
                            .map(|step| Self::from_attribute(ctx, step))
                            .collect()
                    })
                    .collect::<Result<_, _>>()
                    .map(Self::Tails)
            }
            other => Err(format!(
                "evidence_plan element must be mask, dup, select, push, or tails, found {other}"
            )),
        }
    }

    pub fn to_attribute(&self, ctx: &mut IrContext) -> Attribute {
        let (keyword, value) = match self {
            Self::Mask(instance) => ("mask", Attribute::SymbolRef(instance.clone())),
            Self::Dup(instance) => ("dup", Attribute::SymbolRef(instance.clone())),
            Self::Push(instance) => ("push", Attribute::SymbolRef(instance.clone())),
            Self::Select(index) => ("select", Attribute::Int(i128::from(*index))),
            Self::Tails(plans) => (
                "tails",
                Attribute::List(
                    plans
                        .iter()
                        .map(|plan| {
                            Attribute::List(
                                plan.iter().map(|step| step.to_attribute(ctx)).collect(),
                            )
                        })
                        .collect(),
                ),
            ),
        };
        let mut entries = AttributeMap::new();
        entries.view_mut(ctx).insert(keyword, value);
        Attribute::Dict(entries)
    }

    /// The `evidence_plan` value for these steps, or `None` for an empty
    /// plan, which is written by omitting the attribute.
    pub fn plan_attribute(
        ctx: &mut IrContext,
        steps: impl IntoIterator<Item = Self>,
    ) -> Option<Attribute> {
        let steps: Vec<_> = steps
            .into_iter()
            .map(|step| step.to_attribute(ctx))
            .collect();
        (!steps.is_empty()).then_some(Attribute::List(steps))
    }
}

impl trunk_ir::attr_kind::AttrKind for EvidenceStep {
    const KIND: trunk_ir::op_schema::AttributeKind =
        trunk_ir::op_schema::AttributeKind::Dict(&trunk_ir::op_schema::AttributeKind::Any);
    type Out<'ctx> = EvidenceStep;
    type In = EvidenceStep;

    fn read<'ctx>(ctx: &'ctx IrContext, attr: &'ctx Attribute) -> EvidenceStep {
        Self::from_attribute(ctx, attr)
            .unwrap_or_else(|error| panic!("unverified evidence_plan element: {error}"))
    }

    fn write(ctx: &mut IrContext, value: EvidenceStep) -> Attribute {
        value.to_attribute(ctx)
    }
}

/// The operation that carries an `evidence_plan`, which decides the steps it
/// may hold.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum EvidencePlanSite {
    /// A call, `resume`, or an operation built from one.
    Source,
    /// A handle: `mask` steps only.
    Handle,
}

/// Check the shape of an operation's `evidence_plan`. Whether the selection
/// matches the ability rows is typechecking's responsibility.
pub(crate) fn verify_evidence_plan(
    ctx: &IrContext,
    op: OpRef,
    site: EvidencePlanSite,
) -> Result<(), String> {
    let Some(plan) = ctx.op_attrs(op).get(EVIDENCE_PLAN_ATTR) else {
        return Ok(());
    };
    let Attribute::List(items) = plan else {
        return Err("evidence_plan must be a list".into());
    };
    if items.is_empty() {
        return Err("evidence_plan must not be empty; omit it to keep the evidence".into());
    }
    let steps: Vec<_> = items
        .iter()
        .map(|item| EvidenceStep::from_attribute(ctx, item))
        .collect::<Result<_, _>>()?;
    if site == EvidencePlanSite::Handle
        && steps
            .iter()
            .any(|step| !matches!(step, EvidenceStep::Mask(_)))
    {
        return Err("a handle's evidence_plan may only mask".into());
    }
    verify_evidence_steps(&steps, true)
}

/// Check one plan: an optional leading `select`, then the steps on ability
/// instances, each instance changed in one way, then an optional `tails`
/// whose plans follow the same rule without `tails` of their own.
fn verify_evidence_steps(steps: &[EvidenceStep], tails_allowed: bool) -> Result<(), String> {
    let mut seen = HashSet::default();
    for (index, step) in steps.iter().enumerate() {
        match step {
            EvidenceStep::Select(_) if index != 0 => {
                return Err("evidence_plan select must be the first step".into());
            }
            EvidenceStep::Select(_) => {}
            EvidenceStep::Push(_) if !matches!(steps[0], EvidenceStep::Select(_)) => {
                return Err("evidence_plan push requires a selected row tail".into());
            }
            EvidenceStep::Tails(_) if !tails_allowed => {
                return Err("an evidence_plan of a row tail may not attach tails".into());
            }
            EvidenceStep::Tails(_) if index + 1 != steps.len() => {
                return Err("evidence_plan tails must be the last step".into());
            }
            EvidenceStep::Tails(plans) => {
                for plan in plans {
                    verify_evidence_steps(plan, false)?;
                }
            }
            EvidenceStep::Mask(instance) | EvidenceStep::Dup(instance) => {
                if !seen.insert((instance, false)) {
                    return Err(format!(
                        "evidence_plan names ability instance {instance} more than once"
                    ));
                }
            }
            EvidenceStep::Push(instance) => {
                if !seen.insert((instance, true)) {
                    return Err(format!(
                        "evidence_plan names ability instance {instance} more than once"
                    ));
                }
            }
        }
    }
    Ok(())
}
