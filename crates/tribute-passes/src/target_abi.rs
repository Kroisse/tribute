//! Shared physicalization of convention-proven CPS callable signatures.
//!
//! The post-CPS IR still uses `core.never` for the logical result of a CPS
//! transfer.  This pass changes only occurrences proven by callable metadata
//! into `core.nil`, the target-neutral empty-result marker.  Native and Wasm
//! own their later representations and instruction selection.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::ops::ControlFlow;

use tribute_core::{
    CallingConvention, INDIRECT_CALL_SIGNATURE_ATTR, get_calling_convention,
    get_indirect_call_signature, get_physical_closure_convention,
};
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{core, func};
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::refs::{OpRef, TypeRef};
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

/// Convert convention-proven logical CPS results to the shared physical
/// empty-result marker.
///
/// The transformation is planned before any IR mutation.  A `core.never`
/// shape alone is never enough provenance: Direct results and untagged nested
/// callables remain logical source data and are rejected when ambiguous.
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
    let mut aliases_to_update = Vec::new();
    let mut function_types = Vec::new();
    let mut result_types = Vec::new();
    let mut attributes = Vec::new();
    let mut block_args = Vec::new();

    for (name, ty) in aliases {
        let converted = converter.convert_embedded(ty)?;
        if converted != ty {
            aliases_to_update.push((name, converted));
        }
    }

    for &op in &ops {
        if let Ok(function) = func::Func::from_op(converter.ctx, op) {
            let signature = function.r#type(converter.ctx);
            let converted = match get_calling_convention(converter.ctx, op) {
                Some(convention) => converter.convert_callable(signature, convention)?,
                None => converter.convert_embedded(signature)?,
            };
            if converted != signature {
                function_types.push((op, converted));
            }
        }

        let op_results = converter.ctx.op_result_types(op).to_vec();
        for (index, ty) in op_results.into_iter().enumerate() {
            let converted = if let Ok(constant) = func::Constant::from_op(converter.ctx, op) {
                let identity = function_for_symbol(
                    converter.ctx,
                    op,
                    constant.func_ref(converter.ctx),
                    &functions,
                )?;
                validate_constant(converter.ctx, constant, identity, never)?;
                converter.convert_callable(ty, identity.convention)?
            } else {
                converter.convert_embedded(ty)?
            };
            if converted != ty {
                result_types.push((op, index as u32, converted));
            }
        }

        let op_attrs: Vec<_> = converter
            .ctx
            .op(op)
            .attributes
            .iter()
            .map(|(name, value)| (*name, value.clone()))
            .collect();
        for (name, value) in op_attrs {
            if func::Func::matches(converter.ctx, op) && name == Symbol::new("type") {
                continue;
            }
            let converted = if name == Symbol::new(INDIRECT_CALL_SIGNATURE_ATTR) {
                let Attribute::Type(signature) = value else {
                    return Err(TargetAbiError::new(
                        "target ABI: indirect callable signature must be a type attribute",
                    ));
                };
                let convention = get_calling_convention(converter.ctx, op).ok_or_else(|| {
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
            let blocks = converter.ctx.region(region).blocks.to_vec();
            for block in blocks {
                let args = converter.ctx.block(block).args.clone();
                for (index, arg) in args.into_iter().enumerate() {
                    let converted = converter.convert_embedded(arg.ty)?;
                    if converted != arg.ty {
                        block_args.push((block, index as u32, converted));
                    }
                }
            }
        }
    }

    drop(converter);
    for (name, ty) in aliases_to_update {
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
        let Some(convention) = get_calling_convention(ctx, op) else {
            continue;
        };
        let signature = function.r#type(ctx);
        let callable = core::Func::from_type_ref(ctx, signature).ok_or_else(|| {
            TargetAbiError::new(format!(
                "target ABI: function `{}` must have a core.func signature",
                function.sym_name(ctx)
            ))
        })?;
        if convention == CallingConvention::Cps && callable.r#return(ctx) != never {
            return Err(TargetAbiError::new(format!(
                "target ABI: Cps function `{}` must have logical core.never result",
                function.sym_name(ctx)
            )));
        }
        let key = (symbol_scope(ctx, op)?, function.sym_name(ctx));
        if functions
            .insert(
                key,
                FunctionIdentity {
                    signature,
                    convention,
                    environment_index: environment_index(ctx, op, callable.params(ctx))?,
                },
            )
            .is_some()
        {
            return Err(TargetAbiError::new(
                "target ABI: duplicate function symbol in one module",
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
        let convention = get_calling_convention(ctx, op);
        if func::Call::matches(ctx, op) || func::TailCall::matches(ctx, op) {
            let Some(convention) = convention else {
                continue;
            };
            let callee = ctx.op(op).attributes.get_symbol("callee").ok_or_else(|| {
                TargetAbiError::new("target ABI: direct call lacks a callee symbol")
            })?;
            let identity = function_for_symbol(ctx, op, callee, functions)?;
            if identity.convention != convention {
                return Err(TargetAbiError::new(
                    "target ABI: direct call convention differs from callee",
                ));
            }
            let signature = core::Func::from_type_ref(ctx, identity.signature).unwrap();
            let params = signature.params(ctx);
            let operands = ctx.op_operands(op);
            if operands.len() != params.len()
                || operands
                    .iter()
                    .zip(params)
                    .any(|(value, ty)| ctx.value_ty(*value) != *ty)
            {
                return Err(TargetAbiError::new(
                    "target ABI: direct transfer operands differ from callee signature",
                ));
            }
            if func::Call::matches(ctx, op) && convention == CallingConvention::Cps {
                return Err(TargetAbiError::new(
                    "target ABI: Cps direct transfer must use func.tail_call",
                ));
            }
            if func::Call::matches(ctx, op) && ctx.op_result_types(op) != [signature.r#return(ctx)]
            {
                return Err(TargetAbiError::new(
                    "target ABI: direct call result differs from callee signature",
                ));
            }
            if func::TailCall::matches(ctx, op)
                && (convention != CallingConvention::Cps
                    || signature.r#return(ctx) != never
                    || !is_cps_never_caller(ctx, op, never)?)
            {
                return Err(TargetAbiError::new(
                    "target ABI: direct tail call must be a Cps core.never transfer",
                ));
            }
        }

        if !func::CallIndirect::matches(ctx, op) && !func::TailCallIndirect::matches(ctx, op) {
            continue;
        }
        let Some(convention) = convention else {
            continue;
        };
        let signature = get_indirect_call_signature(ctx, op).ok_or_else(|| {
            TargetAbiError::new("target ABI: indirect transfer lacks exact callable signature")
        })?;
        let callable = core::Func::from_type_ref(ctx, signature).ok_or_else(|| {
            TargetAbiError::new("target ABI: indirect callable signature is not core.func")
        })?;
        if convention == CallingConvention::Cps && callable.r#return(ctx) != never {
            return Err(TargetAbiError::new(
                "target ABI: Cps indirect signature must have logical core.never result",
            ));
        }
        let operands = ctx.op_operands(op);
        let args = operands.get(1..).unwrap_or_default();
        if args.len() != callable.params(ctx).len()
            || args
                .iter()
                .zip(callable.params(ctx))
                .any(|(value, ty)| ctx.value_ty(*value) != *ty)
        {
            return Err(TargetAbiError::new(
                "target ABI: indirect transfer operands differ from exact callable signature",
            ));
        }
        if func::CallIndirect::matches(ctx, op)
            && ctx.op_result_types(op) != [callable.r#return(ctx)]
        {
            return Err(TargetAbiError::new(
                "target ABI: indirect call result differs from exact callable signature",
            ));
        }
    }
    Ok(())
}

fn is_cps_never_caller(ctx: &IrContext, op: OpRef, never: TypeRef) -> Result<bool, TargetAbiError> {
    let mut current = Some(op);
    while let Some(candidate) = current {
        if let Ok(function) = func::Func::from_op(ctx, candidate) {
            let signature =
                core::Func::from_type_ref(ctx, function.r#type(ctx)).ok_or_else(|| {
                    TargetAbiError::new(
                        "target ABI: enclosing function must have a core.func signature",
                    )
                })?;
            return Ok(
                get_calling_convention(ctx, candidate) == Some(CallingConvention::Cps)
                    && signature.r#return(ctx) == never,
            );
        }
        current = ctx.op(candidate).parent_block.and_then(|block| {
            ctx.block(block)
                .parent_region
                .and_then(|region| ctx.region(region).parent_op)
        });
    }
    Err(TargetAbiError::new(
        "target ABI: direct tail call has no enclosing function",
    ))
}

fn function_for_symbol(
    ctx: &IrContext,
    op: OpRef,
    symbol: Symbol,
    functions: &HashMap<(OpRef, Symbol), FunctionIdentity>,
) -> Result<FunctionIdentity, TargetAbiError> {
    functions
        .get(&(symbol_scope(ctx, op)?, symbol))
        .copied()
        .ok_or_else(|| TargetAbiError::new(format!("target ABI: unknown callable `{symbol}`")))
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
    let args = &ctx.block(entry).args;
    if args.len() != params.len() {
        return Err(TargetAbiError::new(
            "target ABI: function signature and entry block arity differ",
        ));
    }
    let indices: Vec<_> = args
        .iter()
        .enumerate()
        .filter_map(|(index, arg)| {
            (arg.attrs.get_symbol("bind_name") == Some(Symbol::new("__env"))).then_some(index)
        })
        .collect();
    match indices.as_slice() {
        [] => Ok(None),
        [index] => {
            let type_data = ctx.types.get(params[*index]);
            if type_data.dialect != Symbol::new("tribute_rt")
                || type_data.name != Symbol::new("anyref")
            {
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
        current = ctx.op(candidate).parent_block.and_then(|block| {
            ctx.block(block)
                .parent_region
                .and_then(|region| ctx.region(region).parent_op)
        });
    }
    Err(TargetAbiError::new(
        "target ABI: operation has no enclosing module",
    ))
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
        for index in 1..data.params.len() {
            data.params[index] = self.convert_embedded(data.params[index])?;
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
            let [inner] = data.params.as_slice() else {
                return Err(TargetAbiError::new(
                    "target ABI: closure type must contain one callable",
                ));
            };
            let convention = get_physical_closure_convention(self.ctx, ty).ok_or_else(|| {
                TargetAbiError::new("target ABI: closure callable has no exact convention metadata")
            })?;
            let mut converted = data.clone();
            converted.params[0] = self.convert_callable(*inner, convention)?;
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
    let mut ops = Vec::new();
    let _ = walk_op::<()>(ctx, root, &mut |op| {
        ops.push(op);
        ControlFlow::Continue(WalkAction::Advance)
    });
    ops
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
    fn preserves_direct_and_evidence_source_results_but_physicalizes_cps() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @direct() -> core.never attributes {tribute.calling_convention = 0} { func.unreachable }
  func.func @evidence(%ev: core.i32) -> core.never attributes {tribute.calling_convention = 1} { func.unreachable }
  func.func @cps() -> core.never attributes {tribute.calling_convention = 2} { func.unreachable }
}"#,
        );

        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();

        let never = core::never(&mut ctx).as_type_ref();
        let nil = core::nil(&mut ctx).as_type_ref();
        for (name, expected) in [("direct", never), ("evidence", never), ("cps", nil)] {
            let signature = function(&ctx, module, name).r#type(&ctx);
            assert_eq!(
                core::Func::from_type_ref(&ctx, signature)
                    .unwrap()
                    .r#return(&ctx),
                expected
            );
        }
    }

    #[test]
    fn physicalizes_exact_indirect_cps_signature_only() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @run(%callee: core.i32, %value: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %callee, %value {tribute.calling_convention = 2, func.indirect_call_signature = core.func(core.never, core.i32)}
  }
}"#,
        );

        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();
        let ir = print_module(&ctx, module.op());
        assert!(ir.contains("core.func(core.nil, core.i32)"), "{ir}");
        assert!(!ir.contains("core.func(core.never, core.i32)"), "{ir}");
    }

    #[test]
    fn closure_lowering_retains_the_indirect_cps_signature_for_physicalization() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !evidence = core.array(adt.struct() {fields = [[@ability_id, core.i32], [@prompt_tag, core.i32], [@tr_dispatch_fn, core.ptr], [@handler_dispatch, core.ptr]], name = @_Marker})
  !cps = closure.closure(core.func(core.never, !evidence, core.i32, core.i32, core.i32)) {tribute.calling_convention = 2}
  func.func @run(%callee: !cps, %evidence: !evidence, %done: core.i32, %dispatch: core.i32, %value: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %callee, %evidence, %done, %dispatch, %value {tribute.calling_convention = 2}
  }
}"#,
        );
        let run = function(&ctx, module, "run");
        crate::closure_lower::lower_closures_in_func(&mut ctx, run);

        lower_cps_signatures_to_physical(&mut ctx, module).unwrap();
        let ir = print_module(&ctx, module.op());
        assert!(
            ir.contains("func.indirect_call_signature = core.func(core.nil"),
            "{ir}"
        );
        assert!(!ir.contains("core.func(core.never"), "{ir}");
    }

    #[test]
    fn rejects_malformed_direct_call_without_mutating() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @callee(%value: core.i32) -> core.i32 attributes {tribute.calling_convention = 0} {
    func.return %value
  }
  func.func @caller(%value: core.bool) -> core.i32 attributes {tribute.calling_convention = 0} {
    %result = func.call %value {callee = @callee, tribute.calling_convention = 0} : core.i32
    func.return %result
  }
}"#,
        );
        let before = print_module(&ctx, module.op());

        let error = lower_cps_signatures_to_physical(&mut ctx, module).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("direct transfer operands differ"),
            "{error}"
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn rejects_malformed_direct_cps_tail_without_mutating() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @callee(%value: core.i32) -> core.never attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
  func.func @caller(%value: core.bool) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call %value {callee = @callee, tribute.calling_convention = 2}
  }
}"#,
        );
        let before = print_module(&ctx, module.op());

        let error = lower_cps_signatures_to_physical(&mut ctx, module).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("direct transfer operands differ"),
            "{error}"
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn rejects_direct_tail_outside_a_cps_never_caller_without_mutating() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @callee() -> core.never attributes {tribute.calling_convention = 2} {
    func.unreachable
  }
  func.func @caller() -> core.i32 attributes {tribute.calling_convention = 0} {
    func.tail_call {callee = @callee, tribute.calling_convention = 2}
  }
}"#,
        );
        let before = print_module(&ctx, module.op());

        let error = lower_cps_signatures_to_physical(&mut ctx, module).unwrap_err();

        assert!(
            error.to_string().contains("Cps core.never transfer"),
            "{error}"
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }

    #[test]
    fn rejects_indirect_signature_without_cps_metadata() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @run(%callee: core.func(core.never)) -> core.never attributes {tribute.calling_convention = 2} {
    func.tail_call_indirect %callee {tribute.calling_convention = 2}
  }
}"#,
        );
        let before = print_module(&ctx, module.op());
        let error = lower_cps_signatures_to_physical(&mut ctx, module).unwrap_err();
        assert!(
            error.to_string().contains("lacks exact callable signature"),
            "{error}"
        );
        assert_eq!(print_module(&ctx, module.op()), before);
    }
}
