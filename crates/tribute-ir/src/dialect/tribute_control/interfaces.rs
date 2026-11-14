//! Operation interfaces of `tribute_control` and their implementations.

use super::*;

impl CallLike for Call {
    fn direct_callee<'a>(&self, ctx: &'a IrContext) -> &'a SymbolPath {
        self.callee(ctx)
    }
}
impl CallLike for TailCall {
    fn direct_callee<'a>(&self, ctx: &'a IrContext) -> &'a SymbolPath {
        self.callee(ctx)
    }
}

/// The ability operation declaration an operation names, the kind it claims
/// through the attribute `kind_attr`, and the values and result whose types
/// the declaration fixes.
pub struct DeclarationUse<'a> {
    pub ability: &'a SymbolPath,
    pub op_name: StringRef,
    pub kind: Option<StringRef>,
    pub kind_attr: &'static str,
    /// The values that take the declaration's parameter types, in order.
    pub arguments: &'a [ValueRef],
    /// The type that the declaration's result type must equal.
    pub result_type: Option<TypeRef>,
    /// The region the use belongs to when the operation names several
    /// declarations, one per region.
    pub region: Option<RegionRef>,
}

/// The local checks of a `tribute_control` or `tribute_cps` operation that
/// its definition cannot express: its placement in the enclosing callable or
/// structured operation and the contents of its regions. The local validator
/// queries it after the definition verified the operation, so
/// implementations may assume the declared shape.
#[trunk_ir::op_interface]
pub trait LocalValidation: DialectOp {
    /// Report each violation of these checks.
    fn validate(&self, ctx: &IrContext, errors: &mut Vec<ValidationError>);
}

/// A source function or lambda in either dialect: a callable whose body
/// ends in `tribute_control.return`, with the source signature it declares.
///
/// Whole-IR validation queries it on operations whose schema may not hold,
/// so implementations read the signature fallibly and leave checking that
/// it is a `tribute_control.func_sig` to the caller.
#[trunk_ir::op_interface]
pub trait SourceCallable: DialectOp {
    /// The type the callable declares as its signature; `None` when it
    /// declares none.
    fn callable_signature(&self, ctx: &IrContext) -> Option<TypeRef>;
}

impl SourceCallable for Func {
    fn callable_signature(&self, ctx: &IrContext) -> Option<TypeRef> {
        ctx.op_attrs(self.op_ref()).get_type("type")
    }
}

impl SourceCallable for Lambda {
    fn callable_signature(&self, ctx: &IrContext) -> Option<TypeRef> {
        ctx.op_result_types(self.op_ref()).first().copied()
    }
}

/// As `func.func`: a definition with a body and a foreign `abi` is
/// callable from outside the module, so it is kept.
impl trunk_ir::op_interface::SymbolDefinition for Func {
    fn is_discardable(&self, ctx: &IrContext) -> bool {
        !(ctx.op_attrs(self.op_ref()).contains_key("abi") && ctx.op_has_regions(self.op_ref()))
    }
}

impl LocalValidation for Func {
    fn validate(&self, ctx: &IrContext, errors: &mut Vec<ValidationError>) {
        validate_func(ctx, self.op_ref(), errors);
    }
}

impl LocalValidation for Lambda {
    fn validate(&self, ctx: &IrContext, errors: &mut Vec<ValidationError>) {
        validate_lambda(ctx, self.op_ref(), errors);
    }
}

impl LocalValidation for Return {
    fn validate(&self, ctx: &IrContext, errors: &mut Vec<ValidationError>) {
        validate_return(ctx, self.op_ref(), errors);
    }
}

impl LocalValidation for TailCall {
    fn validate(&self, ctx: &IrContext, errors: &mut Vec<ValidationError>) {
        validate_tail_call(ctx, self.op_ref(), errors);
    }
}

impl LocalValidation for TailCallIndirect {
    fn validate(&self, ctx: &IrContext, errors: &mut Vec<ValidationError>) {
        validate_tail_call(ctx, self.op_ref(), errors);
    }
}

impl LocalValidation for Handle {
    fn validate(&self, ctx: &IrContext, errors: &mut Vec<ValidationError>) {
        validate_handle(ctx, self.op_ref(), errors);
    }
}

impl LocalValidation for Yield {
    fn validate(&self, ctx: &IrContext, errors: &mut Vec<ValidationError>) {
        validate_yield(ctx, self.op_ref(), errors);
    }
}

impl LocalValidation for Resume {
    fn validate(&self, ctx: &IrContext, errors: &mut Vec<ValidationError>) {
        validate_resume(ctx, self.op_ref(), errors);
    }
}

/// What a region is to the `tribute_control.resume`s under it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResumptionScope {
    /// The region is an arm that owns a resumption: a `resume` under it
    /// supplies a value of type `input` and returns one of type `answer`.
    Provides { input: TypeRef, answer: TypeRef },
    /// A `resume` under the region binds to the resumption outside it.
    Transparent,
    /// No resumption reaches under the region.
    Barrier,
}

/// An operation that gives its regions a role for `tribute_control.resume`.
/// An operation that does not implement it is transparent, unless it is
/// `IsolatedFromAbove`, which is a barrier.
///
/// The validator queries it on operations whose schema may not hold, so
/// implementations read attributes and regions fallibly and answer
/// [`ResumptionScope::Barrier`] for a malformed one.
#[trunk_ir::op_interface]
pub trait ResumptionRegions: DialectOp {
    /// What `region`, one of this operation's regions, is to the resumes
    /// under it.
    fn resumption_scope(&self, ctx: &IrContext, region: RegionRef) -> ResumptionScope;
}

/// The resumption a `resume` at some point consumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resumption {
    /// The operation that owns the resumption.
    pub handle: OpRef,
    /// The arm of `handle` that receives it.
    pub arm: RegionRef,
    /// The type of the value a `resume` supplies.
    pub input: TypeRef,
    /// The type of the value a `resume` returns: the result type of the
    /// body of `handle`.
    pub answer: TypeRef,
}

impl DynResumptionRegions {
    /// The resumption of the nearest enclosing resumptive arm of `op`, or
    /// `None` when a barrier or the outermost operation comes first.
    pub fn enclosing(ctx: &IrContext, op: OpRef) -> Option<Resumption> {
        let mut current = op;
        loop {
            let region = parent_region(ctx, current)?;
            let owner = ctx.region(region).parent_op?;
            if let Some(regions) = Self::get(ctx, owner) {
                match regions.resumption_scope(ctx, region) {
                    ResumptionScope::Provides { input, answer } => {
                        return Some(Resumption {
                            handle: owner,
                            arm: region,
                            input,
                            answer,
                        });
                    }
                    ResumptionScope::Transparent => {}
                    ResumptionScope::Barrier => return None,
                }
            } else if IsolatedFromAbove::is_set(ctx, owner) {
                return None;
            }
            current = owner;
        }
    }

    /// The resumption `lambda` carries: the one enclosing it, when a
    /// `resume` under its bodies consumes that one.
    pub fn carried_by(ctx: &IrContext, lambda: OpRef) -> Option<Resumption> {
        if !SourceLambda::is_set(ctx, lambda) {
            return None;
        }
        let enclosing = Self::enclosing(ctx, lambda)?;
        let mut consumed = false;
        for body in ctx.op_regions(lambda) {
            walk_region_ops(ctx, body, &mut |op| {
                consumed |= Resume::matches(ctx, op)
                    && Self::enclosing(ctx, op).is_some_and(|r| r.arm == enclosing.arm);
            });
        }
        consumed.then_some(enclosing)
    }
}

impl ResumptionRegions for Handle {
    /// `body` and `completion` are transparent; a resumptive `op` arm
    /// provides the resumption of its operation into the rest of the body,
    /// whose result type is the completion argument's; any other arm is a
    /// barrier.
    fn resumption_scope(&self, ctx: &IrContext, region: RegionRef) -> ResumptionScope {
        let op = self.op_ref();
        let Some(index) = ctx.op_regions(op).position(|candidate| candidate == region) else {
            return ResumptionScope::Barrier;
        };
        if index < 2 {
            return ResumptionScope::Transparent;
        }
        let body_result = ctx
            .op_region(op, 1)
            .and_then(|completion| ctx.region(completion).blocks.first().copied())
            .and_then(|entry| ctx.block(entry).args.first())
            .map(|arg| arg.ty);
        let (Some(binding), Some(answer)) = (self.binding_of(ctx, region), body_result) else {
            return ResumptionScope::Barrier;
        };
        if binding.is_resumptive(ctx) {
            ResumptionScope::Provides {
                input: binding.operation_result_type,
                answer,
            }
        } else {
            ResumptionScope::Barrier
        }
    }
}

/// An operation that names ability operation declarations, whose parameter
/// and result types fix the types of values the operation supplies.
///
/// Whole-IR validation queries it on operations whose schema may not hold,
/// so implementations read attributes and regions fallibly.
#[trunk_ir::op_interface]
pub trait OperationDeclarationUse: DialectOp {
    /// Each declaration this operation names, skipping one without its
    /// ability, operation name, or argument values.
    fn declaration_uses<'a>(&self, ctx: &'a IrContext) -> Vec<DeclarationUse<'a>>;
}

impl OperationDeclarationUse for Perform {
    fn declaration_uses<'a>(&self, ctx: &'a IrContext) -> Vec<DeclarationUse<'a>> {
        let attrs = ctx.op_attrs(self.op_ref());
        let (Some(ability), Some(op_name)) = (
            attrs.get_symbol_ref("ability"),
            attrs.get_string_ref("op_name"),
        ) else {
            return Vec::new();
        };
        vec![DeclarationUse {
            ability,
            op_name,
            kind: attrs.get_string_ref("operation_kind"),
            kind_attr: "operation_kind",
            arguments: ctx.op_operands(self.op_ref()),
            result_type: ctx.op_result_types(self.op_ref()).first().copied(),
            region: None,
        }]
    }
}

/// Each arm names its binding's declaration. Its arguments are the arm's
/// entry block arguments.
impl OperationDeclarationUse for Handle {
    fn declaration_uses<'a>(&self, ctx: &'a IrContext) -> Vec<DeclarationUse<'a>> {
        let Some(Attribute::List(handlers)) = ctx.op_attrs(self.op_ref()).get("handlers") else {
            return Vec::new();
        };
        handlers
            .iter()
            .zip(ctx.op_regions(self.op_ref()).skip(2))
            .filter_map(|(binding, arm)| {
                let Attribute::Dict(binding) = binding else {
                    return None;
                };
                let binding = binding.view(ctx);
                let &block = ctx.region(arm).blocks.first()?;
                Some(DeclarationUse {
                    ability: binding.get_symbol_ref("ability")?,
                    op_name: binding.get_string_ref("op_name")?,
                    kind: binding.get_string_ref("kind"),
                    kind_attr: "kind",
                    arguments: ctx.block_args(block),
                    result_type: binding.get_type("operation_result_type"),
                    region: Some(arm),
                })
            })
            .collect()
    }
}

// A source `become` ends its callable: the callee's result is the caller's.
impl CallableExit for TailCall {}
impl CallableExit for TailCallIndirect {}

// A `return` ends its callable with the callable's result.
impl CallableExit for Return {}

// A trap ends the execution, and with it the callable.
impl CallableExit for Trap {}

impl LocalValidation for Trap {
    fn validate(&self, ctx: &IrContext, errors: &mut Vec<ValidationError>) {
        let op = self.op_ref();
        validate_return_or_yield_shape(ctx, op, errors);
        let valid_owner = parent_op(ctx, op).is_some_and(|owner| {
            DynSourceCallable::is_implemented(ctx, owner)
                || DynRegionBranch::selects_one_region(ctx, owner)
        });
        if !valid_owner {
            push_op_error(
                ctx,
                op,
                errors,
                "must end a callable body or an arm of an operation that selects one region",
            );
        }
    }
}
