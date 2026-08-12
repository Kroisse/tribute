//! Shared target-neutral physicalization of convention-proven CPS signatures.
//!
//! The pass is intentionally not wired into the production pipeline here. It
//! consumes exact callable metadata, validates the whole transfer surface, and
//! only then maps logical CPS `core.never` results to the shared empty-result
//! marker used by target backends.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::ops::ControlFlow;

use tribute_core::{
    CALLING_CONVENTION_ATTR, CallingConvention, INDIRECT_CALL_SIGNATURE_ATTR,
    get_calling_convention, get_indirect_call_signature, get_physical_closure_convention,
};
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{core, func};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, TypeRef, ValueRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::{Attribute, TypeData};
use trunk_ir::walk::{WalkAction, walk_op};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetAbiError(String);

impl TargetAbiError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for TargetAbiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for TargetAbiError {}

#[derive(Clone, Copy)]
struct FunctionIdentity {
    signature: TypeRef,
    convention: CallingConvention,
    environment_index: Option<usize>,
}

/// Physicalize exact CPS callables without selecting target instructions.
///
/// Validation and conversion planning finish before existing IR is mutated, so
/// rejected modules remain textually unchanged.
pub fn lower_cps_signatures_to_physical(
    ctx: &mut IrContext,
    module: Module,
) -> Result<(), TargetAbiError> {
    let ops = collect_ops(ctx, module.op());
    let never = core::never(ctx).as_type_ref();
    let nil = core::nil(ctx).as_type_ref();
    let functions = collect_functions(ctx, &ops, never)?;
    validate_transfers(ctx, &ops, &functions, never)?;

    let aliases = ctx.type_aliases().to_vec();
    let mut converter = PhysicalTypeConverter::new(ctx, never, nil);
    let mut alias_updates = Vec::new();
    let mut function_types = Vec::new();
    let mut result_types = Vec::new();
    let mut attributes = Vec::new();
    let mut block_args = Vec::new();

    for (name, ty) in aliases {
        let converted = converter.convert_embedded(ty)?;
        if converted != ty {
            alias_updates.push((name, converted));
        }
    }

    for &op in &ops {
        if let Ok(function) = func::Func::from_op(converter.ctx, op) {
            let signature = function.r#type(converter.ctx);
            let converted = match exact_convention(converter.ctx, op)? {
                Some(convention) => converter.convert_callable(signature, convention)?,
                None => converter.convert_embedded(signature)?,
            };
            if converted != signature {
                function_types.push((op, converted));
            }
        }

        for (index, ty) in converter
            .ctx
            .op_result_types(op)
            .to_vec()
            .into_iter()
            .enumerate()
        {
            let converted = if let Ok(constant) = func::Constant::from_op(converter.ctx, op) {
                if let Some(identity) = function_for_symbol_optional(
                    converter.ctx,
                    op,
                    constant.func_ref(converter.ctx),
                    &functions,
                )? {
                    validate_constant(converter.ctx, constant, identity, never)?;
                    converter.convert_callable(ty, identity.convention)?
                } else {
                    converter.convert_embedded(ty)?
                }
            } else {
                converter.convert_embedded(ty)?
            };
            if converted != ty {
                result_types.push((op, index as u32, converted));
            }
        }

        let op_attributes: Vec<_> = converter
            .ctx
            .op(op)
            .attributes
            .iter()
            .map(|(name, value)| (*name, value.clone()))
            .collect();
        for (name, value) in op_attributes {
            if func::Func::matches(converter.ctx, op) && name == Symbol::new("type") {
                continue;
            }
            let converted = if name == Symbol::new(INDIRECT_CALL_SIGNATURE_ATTR) {
                let Attribute::Type(signature) = value else {
                    return Err(TargetAbiError::new(
                        "target ABI: indirect callable signature must be a type attribute",
                    ));
                };
                let convention = exact_convention(converter.ctx, op)?.ok_or_else(|| {
                    TargetAbiError::new(
                        "target ABI: indirect callable signature has no convention metadata",
                    )
                })?;
                Attribute::Type(converter.convert_callable(signature, convention)?)
            } else {
                converter.convert_attribute(value.clone())?
            };
            if converted != value {
                attributes.push((op, name, converted));
            }
        }

        let regions = converter.ctx.op(op).regions.to_vec();
        for region in regions {
            let block_count = converter.ctx.region(region).blocks.len();
            for block_index in 0..block_count {
                let block = converter.ctx.region(region).blocks[block_index];
                for (index, argument) in converter
                    .ctx
                    .block(block)
                    .args
                    .clone()
                    .into_iter()
                    .enumerate()
                {
                    let converted = converter.convert_embedded(argument.ty)?;
                    if converted != argument.ty {
                        block_args.push((block, index as u32, converted));
                    }
                }
            }
        }
    }

    drop(converter);
    for (name, ty) in alias_updates {
        ctx.register_type_alias(name, ty);
    }
    for (op, ty) in function_types {
        ctx.op_mut(op)
            .attributes
            .insert(Symbol::new("type"), Attribute::Type(ty));
    }
    for (op, index, ty) in result_types {
        ctx.set_op_result_type(op, index, ty);
    }
    for (op, name, value) in attributes {
        ctx.op_mut(op).attributes.insert(name, value);
    }
    for (block, index, ty) in block_args {
        ctx.set_block_arg_type(block, index, ty);
    }
    Ok(())
}

fn exact_convention(
    ctx: &IrContext,
    op: OpRef,
) -> Result<Option<CallingConvention>, TargetAbiError> {
    let present = ctx
        .op(op)
        .attributes
        .get(Symbol::new(CALLING_CONVENTION_ATTR))
        .is_some();
    let convention = get_calling_convention(ctx, op);
    if present && convention.is_none() {
        return Err(TargetAbiError::new(
            "target ABI: malformed calling-convention metadata",
        ));
    }
    Ok(convention)
}

fn collect_functions(
    ctx: &IrContext,
    ops: &[OpRef],
    never: TypeRef,
) -> Result<HashMap<(OpRef, Symbol), FunctionIdentity>, TargetAbiError> {
    let mut functions = HashMap::new();
    for &op in ops {
        let Ok(function) = func::Func::from_op(ctx, op) else {
            continue;
        };
        let Some(convention) = exact_convention(ctx, op)? else {
            continue;
        };
        let signature = function.r#type(ctx);
        let callable = core::Func::from_type_ref(ctx, signature).ok_or_else(|| {
            TargetAbiError::new("target ABI: tagged function must have a core.func signature")
        })?;
        if convention == CallingConvention::Cps && callable.r#return(ctx) != never {
            return Err(TargetAbiError::new(format!(
                "target ABI: Cps function `{}` must have logical core.never result",
                function.sym_name(ctx)
            )));
        }
        let key = (symbol_scope(ctx, op)?, function.sym_name(ctx));
        let identity = FunctionIdentity {
            signature,
            convention,
            environment_index: environment_index(ctx, op, callable.params(ctx))?,
        };
        if functions.insert(key, identity).is_some() {
            return Err(TargetAbiError::new(
                "target ABI: duplicate tagged function symbol",
            ));
        }
    }
    Ok(functions)
}

fn validate_transfers(
    ctx: &IrContext,
    ops: &[OpRef],
    functions: &HashMap<(OpRef, Symbol), FunctionIdentity>,
    never: TypeRef,
) -> Result<(), TargetAbiError> {
    for &op in ops {
        if func::Call::matches(ctx, op) || func::TailCall::matches(ctx, op) {
            let Some(convention) = exact_convention(ctx, op)? else {
                continue;
            };
            let callee = ctx.op(op).attributes.get_symbol("callee").ok_or_else(|| {
                TargetAbiError::new("target ABI: direct transfer lacks callee metadata")
            })?;
            let identity = function_for_symbol(ctx, op, callee, functions)?;
            if identity.convention != convention {
                return Err(TargetAbiError::new(
                    "target ABI: direct transfer convention differs from callee",
                ));
            }
            let callable = core::Func::from_type_ref(ctx, identity.signature).unwrap();
            if !operands_match(ctx, ctx.op_operands(op), callable.params(ctx)) {
                return Err(TargetAbiError::new(
                    "target ABI: direct transfer operands differ from callee signature",
                ));
            }
            if func::Call::matches(ctx, op) {
                if convention == CallingConvention::Cps {
                    return Err(TargetAbiError::new(
                        "target ABI: Cps direct transfer must use func.tail_call",
                    ));
                }
                if ctx.op_result_types(op) != [callable.r#return(ctx)] {
                    return Err(TargetAbiError::new(
                        "target ABI: direct call result differs from callee signature",
                    ));
                }
            } else if convention != CallingConvention::Cps
                || callable.r#return(ctx) != never
                || !is_cps_never_caller(ctx, op, never)?
            {
                return Err(TargetAbiError::new(
                    "target ABI: direct tail call must be a Cps core.never transfer",
                ));
            }
            continue;
        }

        if !func::CallIndirect::matches(ctx, op) && !func::TailCallIndirect::matches(ctx, op) {
            continue;
        }
        let signature = get_indirect_call_signature(ctx, op);
        let convention = exact_convention(ctx, op)?;
        if signature.is_none() && convention.is_none() {
            continue;
        }
        let convention = convention.ok_or_else(|| {
            TargetAbiError::new("target ABI: indirect signature has no convention metadata")
        })?;
        let signature = signature.ok_or_else(|| {
            TargetAbiError::new("target ABI: indirect transfer lacks exact callable signature")
        })?;
        let callable = core::Func::from_type_ref(ctx, signature).ok_or_else(|| {
            TargetAbiError::new("target ABI: indirect callable signature is not core.func")
        })?;
        let args = ctx.op_operands(op).get(1..).unwrap_or_default();
        if !operands_match(ctx, args, callable.params(ctx)) {
            return Err(TargetAbiError::new(
                "target ABI: indirect transfer operands differ from exact callable signature",
            ));
        }
        if func::CallIndirect::matches(ctx, op) {
            if convention == CallingConvention::Cps {
                return Err(TargetAbiError::new(
                    "target ABI: Cps indirect transfer must use func.tail_call_indirect",
                ));
            }
            if ctx.op_result_types(op) != [callable.r#return(ctx)] {
                return Err(TargetAbiError::new(
                    "target ABI: indirect call result differs from exact callable signature",
                ));
            }
        } else if convention != CallingConvention::Cps
            || callable.r#return(ctx) != never
            || !is_cps_never_caller(ctx, op, never)?
        {
            return Err(TargetAbiError::new(
                "target ABI: indirect tail call must be a Cps core.never transfer",
            ));
        }
    }
    Ok(())
}

fn operands_match(ctx: &IrContext, operands: &[ValueRef], params: &[TypeRef]) -> bool {
    operands.len() == params.len()
        && operands
            .iter()
            .zip(params)
            .all(|(operand, expected)| ctx.value_ty(*operand) == *expected)
}

fn is_cps_never_caller(ctx: &IrContext, op: OpRef, never: TypeRef) -> Result<bool, TargetAbiError> {
    let mut current = Some(op);
    while let Some(candidate) = current {
        if let Ok(function) = func::Func::from_op(ctx, candidate) {
            let callable =
                core::Func::from_type_ref(ctx, function.r#type(ctx)).ok_or_else(|| {
                    TargetAbiError::new("target ABI: enclosing function is not core.func")
                })?;
            return Ok(
                exact_convention(ctx, candidate)? == Some(CallingConvention::Cps)
                    && callable.r#return(ctx) == never,
            );
        }
        current = parent_op(ctx, candidate);
    }
    Err(TargetAbiError::new(
        "target ABI: tail transfer has no enclosing function",
    ))
}

fn function_for_symbol(
    ctx: &IrContext,
    op: OpRef,
    symbol: Symbol,
    functions: &HashMap<(OpRef, Symbol), FunctionIdentity>,
) -> Result<FunctionIdentity, TargetAbiError> {
    function_for_symbol_optional(ctx, op, symbol, functions)?
        .ok_or_else(|| TargetAbiError::new(format!("target ABI: unknown callable `{symbol}`")))
}

fn function_for_symbol_optional(
    ctx: &IrContext,
    op: OpRef,
    symbol: Symbol,
    functions: &HashMap<(OpRef, Symbol), FunctionIdentity>,
) -> Result<Option<FunctionIdentity>, TargetAbiError> {
    Ok(functions.get(&(symbol_scope(ctx, op)?, symbol)).copied())
}

fn validate_constant(
    ctx: &mut IrContext,
    constant: func::Constant,
    identity: FunctionIdentity,
    never: TypeRef,
) -> Result<(), TargetAbiError> {
    let target = core::Func::from_type_ref(ctx, identity.signature).unwrap();
    if identity.convention == CallingConvention::Cps && target.r#return(ctx) != never {
        return Err(TargetAbiError::new(
            "target ABI: Cps function reference must have logical core.never result",
        ));
    }
    let mut params = target.params(ctx).to_vec();
    if let Some(index) = identity.environment_index {
        if index >= params.len() {
            return Err(TargetAbiError::new(
                "target ABI: closure environment index is outside target signature",
            ));
        }
        params.remove(index);
    }
    let expected = core::func(ctx, target.r#return(ctx), params).as_type_ref();
    if ctx.op_result_types(constant.op_ref()) != [expected] {
        return Err(TargetAbiError::new(
            "target ABI: function reference differs from target signature",
        ));
    }
    Ok(())
}

fn environment_index(
    ctx: &IrContext,
    function: OpRef,
    params: &[TypeRef],
) -> Result<Option<usize>, TargetAbiError> {
    let Some(&region) = ctx.op(function).regions.first() else {
        return Ok(None);
    };
    let Some(&entry) = ctx.region(region).blocks.first() else {
        return Ok(None);
    };
    let arguments = &ctx.block(entry).args;
    if arguments.len() != params.len() {
        return Err(TargetAbiError::new(
            "target ABI: function signature and entry block arity differ",
        ));
    }
    let indices: Vec<_> = arguments
        .iter()
        .enumerate()
        .filter_map(|(index, argument)| {
            (argument.attrs.get_symbol("bind_name") == Some(Symbol::new("__env"))).then_some(index)
        })
        .collect();
    match indices.as_slice() {
        [] => Ok(None),
        [index] => {
            let data = ctx.types.get(params[*index]);
            if data.dialect != Symbol::new("tribute_rt") || data.name != Symbol::new("anyref") {
                return Err(TargetAbiError::new(
                    "target ABI: `__env` must have exact tribute_rt.anyref type",
                ));
            }
            Ok(Some(*index))
        }
        _ => Err(TargetAbiError::new(
            "target ABI: function has multiple `__env` parameters",
        )),
    }
}

fn symbol_scope(ctx: &IrContext, op: OpRef) -> Result<OpRef, TargetAbiError> {
    let mut current = Some(op);
    while let Some(candidate) = current {
        if core::Module::matches(ctx, candidate) {
            return Ok(candidate);
        }
        current = parent_op(ctx, candidate);
    }
    Err(TargetAbiError::new(
        "target ABI: operation has no enclosing module",
    ))
}

fn parent_op(ctx: &IrContext, op: OpRef) -> Option<OpRef> {
    ctx.op(op).parent_block.and_then(|block| {
        ctx.block(block)
            .parent_region
            .and_then(|region| ctx.region(region).parent_op)
    })
}

struct PhysicalTypeConverter<'a> {
    ctx: &'a mut IrContext,
    never: TypeRef,
    nil: TypeRef,
    embedded: HashMap<TypeRef, TypeRef>,
    callable: HashMap<(TypeRef, CallingConvention), TypeRef>,
}

impl<'a> PhysicalTypeConverter<'a> {
    fn new(ctx: &'a mut IrContext, never: TypeRef, nil: TypeRef) -> Self {
        Self {
            ctx,
            never,
            nil,
            embedded: HashMap::new(),
            callable: HashMap::new(),
        }
    }

    fn convert_callable(
        &mut self,
        ty: TypeRef,
        convention: CallingConvention,
    ) -> Result<TypeRef, TargetAbiError> {
        if let Some(&converted) = self.callable.get(&(ty, convention)) {
            return Ok(converted);
        }
        let mut data = self.ctx.types.get(ty).clone();
        if data.dialect != Symbol::new("core")
            || data.name != Symbol::new("func")
            || data.params.is_empty()
        {
            return Err(TargetAbiError::new(
                "target ABI: proven callable is not core.func",
            ));
        }
        if convention == CallingConvention::Cps && data.params[0] != self.never {
            return Err(TargetAbiError::new(
                "target ABI: Cps callable must have logical core.never result",
            ));
        }
        data.params[0] = if convention == CallingConvention::Cps {
            self.nil
        } else {
            self.convert_embedded(data.params[0])?
        };
        for parameter in &mut data.params[1..] {
            *parameter = self.convert_embedded(*parameter)?;
        }
        self.convert_type_attributes(&mut data)?;
        let converted = self.intern_if_changed(ty, data);
        self.callable.insert((ty, convention), converted);
        Ok(converted)
    }

    fn convert_embedded(&mut self, ty: TypeRef) -> Result<TypeRef, TargetAbiError> {
        if let Some(&converted) = self.embedded.get(&ty) {
            return Ok(converted);
        }
        let data = self.ctx.types.get(ty).clone();
        if data.dialect == Symbol::new("closure") && data.name == Symbol::new("closure") {
            let [function] = data.params.as_slice() else {
                return Err(TargetAbiError::new(
                    "target ABI: closure type must contain one callable",
                ));
            };
            let convention = get_physical_closure_convention(self.ctx, ty).ok_or_else(|| {
                TargetAbiError::new("target ABI: closure callable has no exact convention metadata")
            })?;
            let mut converted = data.clone();
            converted.params[0] = self.convert_callable(*function, convention)?;
            self.convert_type_attributes(&mut converted)?;
            let converted = self.intern_if_changed(ty, converted);
            self.embedded.insert(ty, converted);
            return Ok(converted);
        }
        if data.dialect == Symbol::new("core") && data.name == Symbol::new("func") {
            let callable = core::Func::from_type_ref(self.ctx, ty).unwrap();
            if callable.r#return(self.ctx) == self.never {
                return Err(TargetAbiError::new(
                    "target ABI: untagged nested core.func<core.never, ...>",
                ));
            }
        }
        let mut converted = data.clone();
        for parameter in &mut converted.params {
            *parameter = self.convert_embedded(*parameter)?;
        }
        self.convert_type_attributes(&mut converted)?;
        let converted = self.intern_if_changed(ty, converted);
        self.embedded.insert(ty, converted);
        Ok(converted)
    }

    fn convert_type_attributes(&mut self, data: &mut TypeData) -> Result<(), TargetAbiError> {
        let attributes: Vec<_> = data
            .attrs
            .iter()
            .map(|(name, value)| (*name, value.clone()))
            .collect();
        for (name, value) in attributes {
            data.attrs.insert(name, self.convert_attribute(value)?);
        }
        Ok(())
    }

    fn convert_attribute(&mut self, attribute: Attribute) -> Result<Attribute, TargetAbiError> {
        match attribute {
            Attribute::Type(ty) => Ok(Attribute::Type(self.convert_embedded(ty)?)),
            Attribute::List(values) => Ok(Attribute::List(
                values
                    .into_iter()
                    .map(|value| self.convert_attribute(value))
                    .collect::<Result<_, _>>()?,
            )),
            other => Ok(other),
        }
    }

    fn intern_if_changed(&mut self, original: TypeRef, data: TypeData) -> TypeRef {
        if data == *self.ctx.types.get(original) {
            original
        } else {
            self.ctx.types.intern(data)
        }
    }
}

fn collect_ops(ctx: &IrContext, root: OpRef) -> Vec<OpRef> {
    let mut operations = Vec::new();
    let _ = walk_op::<()>(ctx, root, &mut |op| {
        operations.push(op);
        ControlFlow::Continue(WalkAction::Advance)
    });
    operations
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;

    fn function(ctx: &IrContext, module: Module, name: &str) -> func::Func {
        module
            .ops(ctx)
            .into_iter()
            .find_map(|op| {
                let function = func::Func::from_op(ctx, op).ok()?;
                (function.sym_name(ctx) == Symbol::from_dynamic(name)).then_some(function)
            })
            .unwrap()
    }

    #[test]
    fn physicalizes_dispatch_aware_exact_cps_contracts_only() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !cps = closure.closure(core.func(core.never, core.i32, core.i32, core.i32, core.i32)) {tribute.calling_convention = 2}
  func.func @direct() -> core.never attributes {tribute.calling_convention = 0} { func.unreachable }
  func.func @evidence() -> core.never attributes {tribute.calling_convention = 1} { func.unreachable }
  func.func @cps() -> core.never attributes {tribute.calling_convention = 2} { func.unreachable }
  func.func @run(%callee: core.i32, %evidence: core.i32, %env: tribute_rt.anyref, %done: core.i32, %dispatch: core.i32, %value: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %callee, %evidence, %env, %done, %dispatch, %value {func.indirect_call_signature = core.func(core.never, core.i32, tribute_rt.anyref, core.i32, core.i32, core.i32), tribute.calling_convention = 2}
  }
}"#,
        );

        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();

        let never = core::never(&mut ctx).as_type_ref();
        let nil = core::nil(&mut ctx).as_type_ref();
        for (name, expected) in [
            ("direct", never),
            ("evidence", never),
            ("cps", nil),
            ("run", nil),
        ] {
            let signature = function(&ctx, module, name).r#type(&ctx);
            assert_eq!(
                core::Func::from_type_ref(&ctx, signature)
                    .unwrap()
                    .r#return(&ctx),
                expected
            );
        }
        let printed = print_module(&ctx, module.op());
        assert!(
            printed.contains("func.indirect_call_signature = core.func(core.nil"),
            "{printed}"
        );
        assert!(
            printed.contains("closure.closure(core.func(core.nil"),
            "{printed}"
        );
    }

    #[test]
    fn malformed_transfers_and_ambiguous_nested_never_leave_ir_unchanged() {
        for (input, expected) in [
            (
                r#"core.module @test {
  func.func @run(%callee: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %callee {tribute.calling_convention = 2}
  }
}"#,
                "lacks exact callable signature",
            ),
            (
                r#"core.module @test {
  func.func @callee(%value: core.i32) -> core.never attributes {tribute.calling_convention = 2} { func.unreachable }
  func.func @run(%value: core.bool) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call %value {callee = @callee, tribute.calling_convention = 2}
  }
}"#,
                "operands differ",
            ),
            (
                r#"core.module @test {
  !ambiguous = closure.closure(core.func(core.never))
}"#,
                "no exact convention metadata",
            ),
        ] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(&mut ctx, input);
            let before = print_module(&ctx, module.op());

            let error = lower_cps_signatures_to_physical(&mut ctx, module).unwrap_err();

            assert!(error.to_string().contains(expected), "{error}");
            assert_eq!(print_module(&ctx, module.op()), before);
        }
    }

    #[test]
    fn bodyless_cps_function_constant_physicalizes_without_a_body() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @external(%evidence: core.i32, %done: core.i32) -> core.never attributes {tribute.calling_convention = 2}
  func.func @holder() -> core.i32 {
    %function = func.constant {func_ref = @external} : core.func(core.never, core.i32, core.i32)
    func.unreachable
  }
}"#,
        );

        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();

        let printed = print_module(&ctx, module.op());
        assert!(
            printed.contains("func.func @external(%arg0: core.i32, %arg1: core.i32) -> core.nil")
        );
        assert!(printed.contains("func.constant {func_ref = @external} : core.func(core.nil"));
    }

    #[test]
    fn raw_constant_is_unchanged_and_malformed_tagged_constant_fails_closed() {
        let mut ctx = IrContext::new();
        let raw = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @external(%value: core.i32) -> core.i32
  func.func @holder() -> core.i32 {
    %function = func.constant {func_ref = @external} : core.func(core.i32, core.i32)
    func.unreachable
  }
}"#,
        );
        let before = print_module(&ctx, raw.op());

        lower_cps_signatures_to_physical(&mut ctx, raw).unwrap();

        assert_eq!(print_module(&ctx, raw.op()), before);

        let mut ctx = IrContext::new();
        let malformed = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @external(%value: core.i32) -> core.never attributes {tribute.calling_convention = 2}
  func.func @holder() -> core.i32 {
    %function = func.constant {func_ref = @external} : core.func(core.never, core.bool)
    func.unreachable
  }
}"#,
        );
        let before = print_module(&ctx, malformed.op());

        let error = lower_cps_signatures_to_physical(&mut ctx, malformed).unwrap_err();

        assert!(error.to_string().contains("function reference differs"));
        assert_eq!(print_module(&ctx, malformed.op()), before);
    }

    #[test]
    fn malformed_calling_convention_is_rejected_before_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @broken() -> core.never attributes {tribute.calling_convention = 9} { func.unreachable }
}"#,
        );
        let before = print_module(&ctx, module.op());

        let error = lower_cps_signatures_to_physical(&mut ctx, module).unwrap_err();

        assert!(error.to_string().contains("malformed calling-convention"));
        assert_eq!(print_module(&ctx, module.op()), before);
    }
}
