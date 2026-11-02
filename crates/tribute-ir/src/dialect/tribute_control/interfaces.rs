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

/// The ability operation declaration an operation names, and the kind it
/// claims through the attribute `kind_attr`.
pub struct DeclarationUse<'a> {
    pub ability: &'a SymbolPath,
    pub op_name: StringRef,
    pub kind: Option<StringRef>,
    pub kind_attr: &'static str,
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

impl LocalValidation for Handler {
    fn validate(&self, ctx: &IrContext, errors: &mut Vec<ValidationError>) {
        validate_handler(ctx, self.op_ref(), errors);
    }
}

impl LocalValidation for Yield {
    fn validate(&self, ctx: &IrContext, errors: &mut Vec<ValidationError>) {
        validate_yield(ctx, self.op_ref(), errors);
    }
}

/// An operation that names one ability operation declaration, whose
/// parameter and result types fix the types of values the operation supplies.
///
/// Whole-IR validation queries it on operations whose schema may not hold,
/// so implementations read attributes and regions fallibly.
#[trunk_ir::op_interface]
pub trait OperationDeclarationUse: DialectOp {
    /// The declaration this operation names; `None` without its ability or
    /// operation name.
    fn declaration_use<'a>(&self, ctx: &'a IrContext) -> Option<DeclarationUse<'a>>;

    /// The values that take the declaration's parameter types, in order.
    fn declared_arguments<'a>(&self, ctx: &'a IrContext) -> Option<&'a [ValueRef]>;

    /// The type that the declaration's result type must equal.
    fn declared_result_type(&self, ctx: &IrContext) -> Option<TypeRef>;
}

impl OperationDeclarationUse for Perform {
    fn declaration_use<'a>(&self, ctx: &'a IrContext) -> Option<DeclarationUse<'a>> {
        let attrs = ctx.op_attrs(self.op_ref());
        Some(DeclarationUse {
            ability: attrs.get_symbol_ref("ability")?,
            op_name: attrs.get_string_ref("op_name")?,
            kind: attrs.get_string_ref("operation_kind"),
            kind_attr: "operation_kind",
        })
    }

    fn declared_arguments<'a>(&self, ctx: &'a IrContext) -> Option<&'a [ValueRef]> {
        Some(ctx.op_operands(self.op_ref()))
    }

    fn declared_result_type(&self, ctx: &IrContext) -> Option<TypeRef> {
        ctx.op_result_types(self.op_ref()).first().copied()
    }
}

impl OperationDeclarationUse for Handler {
    fn declaration_use<'a>(&self, ctx: &'a IrContext) -> Option<DeclarationUse<'a>> {
        let attrs = ctx.op_attrs(self.op_ref());
        Some(DeclarationUse {
            ability: attrs.get_symbol_ref("ability")?,
            op_name: attrs.get_string_ref("op_name")?,
            kind: attrs.get_string_ref("kind"),
            kind_attr: "kind",
        })
    }

    /// The clause's block arguments without its trailing resume token.
    fn declared_arguments<'a>(&self, ctx: &'a IrContext) -> Option<&'a [ValueRef]> {
        let region = ctx.op_region(self.op_ref(), 0)?;
        let &block = ctx.region(region).blocks.first()?;
        let args = ctx.block_args(block);
        Some(match args.split_last() {
            Some((token, rest)) if ResumeToken::matches(ctx, ctx.value_ty(*token)) => rest,
            _ => args,
        })
    }

    fn declared_result_type(&self, ctx: &IrContext) -> Option<TypeRef> {
        ctx.op_attrs(self.op_ref())
            .get_type("operation_result_type")
    }
}

// A source `become` ends its callable: the callee's result is the caller's.
impl CallableExit for TailCall {}
impl CallableExit for TailCallIndirect {}
