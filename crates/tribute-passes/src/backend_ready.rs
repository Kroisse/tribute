//! Final Tribute backend-boundary verification.
//!
//! The generic target validators own operation legality. This module adds the
//! Tribute contract they intentionally cannot know: all reachable types and
//! attributes must be legal for the selected target and no logical control
//! representation may cross into emission.

use std::collections::HashSet;
use std::fmt;
use std::ops::ControlFlow;

use tribute_core::{CallingConvention, get_calling_convention};
use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::func;
use trunk_ir::ops::DialectType;
use trunk_ir::refs::{OpRef, TypeRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::Attribute;
use trunk_ir::walk::{WalkAction, walk_op};

/// Backend selected for a final Tribute legality boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TributeBackend {
    Native,
    Wasm,
}

impl TributeBackend {
    fn boundary(self) -> &'static str {
        match self {
            Self::Native => "tribute-backend-ready-native",
            Self::Wasm => "tribute-backend-ready-wasm",
        }
    }

    fn dialect(self) -> &'static str {
        match self {
            Self::Native => "clif",
            Self::Wasm => "wasm",
        }
    }
}

/// A non-mutating final-boundary validation error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendReadyError(String);

impl BackendReadyError {
    fn new(backend: TributeBackend, message: impl fmt::Display) -> Self {
        Self(format!("{}: {message}", backend.boundary()))
    }
}

impl fmt::Display for BackendReadyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BackendReadyError {}

/// Verify that `module` is ready for the selected backend.
pub fn verify_tribute_backend_ready(
    ctx: &IrContext,
    module: Module,
    backend: TributeBackend,
) -> Result<(), BackendReadyError> {
    verify_target_operations(ctx, module, backend)?;

    let mut verifier = RecursiveLegalityWalker::new(ctx, backend);
    for &(symbol, ty) in ctx.type_aliases() {
        verifier.verify_symbol(symbol, "type alias")?;
        verifier.verify_type(ty, TypeSurface::Metadata, "type alias")?;
    }

    for op in collect_ops(ctx, module.op()) {
        verifier.verify_operation(op)?;
    }
    Ok(())
}

/// Verify the final native boundary.
pub fn verify_native_backend_ready(
    ctx: &IrContext,
    module: Module,
) -> Result<(), BackendReadyError> {
    verify_tribute_backend_ready(ctx, module, TributeBackend::Native)
}

/// Verify the final Wasm boundary.
pub fn verify_wasm_backend_ready(ctx: &IrContext, module: Module) -> Result<(), BackendReadyError> {
    verify_tribute_backend_ready(ctx, module, TributeBackend::Wasm)
}

fn verify_target_operations(
    ctx: &IrContext,
    module: Module,
    backend: TributeBackend,
) -> Result<(), BackendReadyError> {
    match backend {
        TributeBackend::Native => trunk_ir_cranelift_backend::validate_clif_ir(ctx, module)
            .map_err(|error| BackendReadyError::new(backend, error)),
        TributeBackend::Wasm => trunk_ir_wasm_backend::validate_wasm_ir(ctx, module)
            .map_err(|error| BackendReadyError::new(backend, error)),
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

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum TypeSurface {
    Value,
    Result,
    Metadata,
    Signature,
}

struct RecursiveLegalityWalker<'ctx> {
    ctx: &'ctx IrContext,
    backend: TributeBackend,
    visited: HashSet<(TypeRef, TypeSurface)>,
}

impl<'ctx> RecursiveLegalityWalker<'ctx> {
    fn new(ctx: &'ctx IrContext, backend: TributeBackend) -> Self {
        Self {
            ctx,
            backend,
            visited: HashSet::new(),
        }
    }

    fn verify_operation(&mut self, op: OpRef) -> Result<(), BackendReadyError> {
        let data = self.ctx.op(op);
        let surface = format!("operation {}.{}", data.dialect, data.name);

        for &ty in self.ctx.op_result_types(op) {
            self.verify_type(ty, TypeSurface::Result, &surface)?;
        }
        for &operand in self.ctx.op_operands(op) {
            self.verify_type(self.ctx.value_ty(operand), TypeSurface::Value, &surface)?;
        }
        for (key, attribute) in &data.attributes {
            self.verify_symbol(*key, &surface)?;
            if self.is_signature_attribute(op, *key) {
                let Attribute::Type(ty) = attribute else {
                    return Err(self.error("callable signature attribute is not a type"));
                };
                self.verify_type(*ty, TypeSurface::Signature, &surface)?;
            } else {
                self.verify_attribute(attribute, TypeSurface::Metadata, &surface)?;
            }
        }
        for &region in &data.regions {
            for &block in &self.ctx.region(region).blocks {
                for argument in &self.ctx.block(block).args {
                    self.verify_type(argument.ty, TypeSurface::Value, "block argument")?;
                    for (key, attribute) in &argument.attrs {
                        self.verify_symbol(*key, "block argument attribute")?;
                        self.verify_attribute(
                            attribute,
                            TypeSurface::Metadata,
                            "block argument attribute",
                        )?;
                    }
                }
            }
        }
        self.verify_abi(op)
    }

    fn verify_type(
        &mut self,
        ty: TypeRef,
        surface: TypeSurface,
        where_: &str,
    ) -> Result<(), BackendReadyError> {
        if !self.visited.insert((ty, surface)) {
            return Ok(());
        }
        let data = self.ctx.types.get(ty);
        if !type_is_legal(self.backend, surface, data) {
            return Err(self.error(format!(
                "illegal type `{}.{}` in {where_}",
                data.dialect, data.name
            )));
        }
        let is_function =
            data.dialect == Symbol::new("func") && data.name == Symbol::new("func_sig");
        if is_function {
            let Some(function) = func::FuncSig::from_type_ref(self.ctx, ty) else {
                return Err(self.error(format!("malformed `func.func_sig` type in {where_}")));
            };
            let Some(result) = function.single_result(self.ctx) else {
                return Err(self.error(format!("`func.func_sig` has no result type in {where_}")));
            };
            self.verify_type(result, TypeSurface::Result, "function result")?;
            for &parameter in function.inputs(self.ctx) {
                self.verify_type(parameter, TypeSurface::Value, "function parameter")?;
            }
        } else {
            for &parameter in &data.params {
                self.verify_type(parameter, TypeSurface::Metadata, "nested type parameter")?;
            }
        }
        for (key, attribute) in &data.attrs {
            self.verify_symbol(*key, "type attribute")?;
            self.verify_attribute(attribute, TypeSurface::Metadata, "type attribute")?;
        }

        if surface == TypeSurface::Signature && !is_function {
            return Err(self.error(format!(
                "`{}.{}` is not a callable signature in {where_}",
                data.dialect, data.name
            )));
        }
        Ok(())
    }

    fn verify_attribute(
        &mut self,
        attribute: &Attribute,
        surface: TypeSurface,
        where_: &str,
    ) -> Result<(), BackendReadyError> {
        match attribute {
            Attribute::Type(ty) => self.verify_type(*ty, surface, where_),
            Attribute::Symbol(symbol) => self.verify_symbol(*symbol, where_),
            Attribute::List(values) => {
                for value in values {
                    self.verify_attribute(value, surface, where_)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn verify_symbol(&self, symbol: Symbol, where_: &str) -> Result<(), BackendReadyError> {
        if symbol == Symbol::new("__tribute_cps_control") {
            return Err(self.error(format!(
                "private CPS control carrier `__tribute_cps_control` remains in {where_}"
            )));
        }
        Ok(())
    }

    fn verify_abi(&mut self, op: OpRef) -> Result<(), BackendReadyError> {
        let data = self.ctx.op(op);
        let convention = if data.attributes.contains_key("tribute.calling_convention") {
            Some(
                get_calling_convention(self.ctx, op)
                    .ok_or_else(|| self.error("invalid tribute.calling_convention attribute"))?,
            )
        } else {
            None
        };

        if convention == Some(CallingConvention::Cps)
            && data.dialect == Symbol::new(self.backend.dialect())
            && data.name == Symbol::new("func")
        {
            let signature = data
                .attributes
                .get_type("type")
                .ok_or_else(|| self.error("Cps function has no signature"))?;
            let signature = func::FuncSig::from_type_ref(self.ctx, signature);
            if signature
                .and_then(|function| function.single_result(self.ctx))
                .is_none_or(|result| !is_core_type(self.ctx, result, "nil"))
            {
                return Err(self.error("Cps function result is not physically empty"));
            }
        }

        if (data.name == Symbol::new("return_call")
            || data.name == Symbol::new("return_call_indirect"))
            && !self.ctx.op_result_types(op).is_empty()
        {
            return Err(self.error("result-producing CPS transfer remains"));
        }
        Ok(())
    }

    fn is_signature_attribute(&self, op: OpRef, key: Symbol) -> bool {
        let data = self.ctx.op(op);
        if data.dialect != Symbol::new(self.backend.dialect()) {
            return false;
        }
        if data.name == Symbol::new("func") {
            return key == Symbol::new("type");
        }
        if data.name != Symbol::new("call_indirect")
            && data.name != Symbol::new("return_call_indirect")
        {
            return false;
        }
        key == Symbol::new(match self.backend {
            TributeBackend::Native => "sig",
            TributeBackend::Wasm => "signature",
        })
    }

    fn error(&self, message: impl fmt::Display) -> BackendReadyError {
        BackendReadyError::new(self.backend, message)
    }
}

fn type_is_legal(
    backend: TributeBackend,
    surface: TypeSurface,
    data: &trunk_ir::types::TypeData,
) -> bool {
    match surface {
        TypeSurface::Signature => {
            data.dialect == Symbol::new("func") && data.name == Symbol::new("func_sig")
        }
        TypeSurface::Value => target_value_type(backend, data),
        TypeSurface::Result => {
            is_named_type(data.dialect, data.name, "core", "nil")
                || target_value_type(backend, data)
        }
        TypeSurface::Metadata => target_metadata_type(backend, data),
    }
}

fn target_value_type(backend: TributeBackend, data: &trunk_ir::types::TypeData) -> bool {
    match backend {
        TributeBackend::Native => core_type_is(
            data.dialect,
            data.name,
            &["nil", "i1", "i8", "i16", "i32", "i64", "f32", "f64", "ptr"],
        ),
        TributeBackend::Wasm => {
            core_type_is(
                data.dialect,
                data.name,
                &[
                    "nil", "i1", "i32", "i64", "f32", "f64", "bytes", "ptr", "array",
                ],
            ) || func_sig_type_is(data)
                || adt_type_is(data)
                || wasm_reference_type(data.dialect, data.name)
        }
    }
}

fn target_metadata_type(backend: TributeBackend, data: &trunk_ir::types::TypeData) -> bool {
    core_type_is(
        data.dialect,
        data.name,
        &[
            "nil", "i1", "i8", "i16", "i32", "i64", "f32", "f64", "ptr", "bytes", "ref", "array",
        ],
    ) || func_sig_type_is(data)
        || adt_type_is(data)
        || (backend == TributeBackend::Wasm && wasm_reference_type(data.dialect, data.name))
}

fn func_sig_type_is(data: &trunk_ir::types::TypeData) -> bool {
    data.dialect == Symbol::new("func") && data.name == Symbol::new("func_sig")
}

fn core_type_is(dialect: Symbol, name: Symbol, names: &[&str]) -> bool {
    dialect == Symbol::new("core")
        && names
            .iter()
            .any(|candidate| name == Symbol::from_dynamic(candidate))
}

fn adt_type_is(data: &trunk_ir::types::TypeData) -> bool {
    data.dialect == Symbol::new("adt")
        && (["struct", "enum", "typeref", "variant_inst"]
            .iter()
            .any(|candidate| data.name == Symbol::from_dynamic(candidate))
            || data.attrs.get_bool("is_variant") == Some(true))
}

fn wasm_reference_type(dialect: Symbol, name: Symbol) -> bool {
    dialect == Symbol::new("wasm")
        && ["anyref", "i31ref", "structref", "arrayref", "funcref"]
            .iter()
            .any(|candidate| name == Symbol::from_dynamic(candidate))
}

fn is_named_type(
    dialect: Symbol,
    name: Symbol,
    expected_dialect: &str,
    expected_name: &str,
) -> bool {
    dialect == Symbol::from_dynamic(expected_dialect) && name == Symbol::from_dynamic(expected_name)
}

fn is_core_type(ctx: &IrContext, ty: TypeRef, expected: &str) -> bool {
    let data = ctx.types.get(ty);
    is_named_type(data.dialect, data.name, "core", expected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::types::TypeDataBuilder;

    fn verify(ir: &str, backend: TributeBackend) -> Result<(), String> {
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, ir);
        match backend {
            TributeBackend::Native => verify_native_backend_ready(&ctx, module),
            TributeBackend::Wasm => verify_wasm_backend_ready(&ctx, module),
        }
        .map_err(|error| error.to_string())
    }

    #[test]
    fn malformed_function_counts_are_rejected_at_each_backend_boundary() {
        for (dialect, backend) in [
            ("clif", TributeBackend::Native),
            ("wasm", TributeBackend::Wasm),
        ] {
            let mut ctx = IrContext::new();
            let text = format!(
                "core.module @m {{ {dialect}.func @f() -> core.nil {{ {dialect}.return }} }}"
            );
            let module = parse_test_module(&mut ctx, &text);
            let op = module.ops(&ctx)[0];
            let ty = ctx.op(op).attributes.get_type("type").unwrap();
            let mut malformed = ctx.types.get(ty).clone();
            malformed.attrs.remove(func::NUM_RESULTS_ATTR);
            let malformed = ctx.types.intern(malformed);
            ctx.op_mut(op)
                .attributes
                .insert(Symbol::new("type"), Attribute::Type(malformed));
            let before = trunk_ir::printer::print_module(&ctx, module.op());
            let error = match backend {
                TributeBackend::Native => verify_native_backend_ready(&ctx, module),
                TributeBackend::Wasm => verify_wasm_backend_ready(&ctx, module),
            }
            .unwrap_err();
            assert!(
                error.to_string().contains("malformed `func.func_sig`"),
                "{error}"
            );
            assert_eq!(trunk_ir::printer::print_module(&ctx, module.op()), before);
        }
    }

    #[test]
    fn shared_func_sig_replaces_core_func_on_value_and_metadata_surfaces() {
        let mut ctx = IrContext::new();
        let shared = func::func_sig(&mut ctx, [], []).as_type_ref();
        let retired = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func")).build());

        assert!(type_is_legal(
            TributeBackend::Wasm,
            TypeSurface::Value,
            ctx.types.get(shared)
        ));
        assert!(!type_is_legal(
            TributeBackend::Wasm,
            TypeSurface::Value,
            ctx.types.get(retired)
        ));
        for backend in [TributeBackend::Native, TributeBackend::Wasm] {
            assert!(type_is_legal(
                backend,
                TypeSurface::Metadata,
                ctx.types.get(shared)
            ));
            assert!(!type_is_legal(
                backend,
                TypeSurface::Metadata,
                ctx.types.get(retired)
            ));
        }
    }

    #[test]
    fn rejects_retired_core_func_nested_in_reachable_metadata() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            "core.module @m { wasm.func @main() -> core.nil { wasm.return } }",
        );
        let retired = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("core"), Symbol::new("func")).build());
        let nested = ctx.types.intern(
            TypeDataBuilder::new(Symbol::new("core"), Symbol::new("array"))
                .params([retired])
                .build(),
        );
        ctx.op_mut(module.op())
            .attributes
            .insert(Symbol::new("metadata"), Attribute::Type(nested));

        let error = verify_wasm_backend_ready(&ctx, module)
            .expect_err("retired core.func must not cross the backend boundary")
            .to_string();
        assert!(error.contains("core.func"), "{error}");
    }

    #[test]
    fn accepts_textual_backend_functions() {
        for (ir, backend) in [
            (
                r#"core.module @test {
  clif.func @main(%unit: core.nil, %value: core.i32) -> core.nil {
    clif.return
  }
}"#,
                TributeBackend::Native,
            ),
            (
                r#"core.module @test {
  wasm.func @main(%value: core.i32) -> core.nil {
    wasm.return
  }
}"#,
                TributeBackend::Wasm,
            ),
        ] {
            verify(ir, backend).unwrap();
        }
    }

    #[test]
    fn rejects_residual_logical_operation() {
        for operation in [
            "tribute_control.perform",
            "ability.perform",
            "effect.dispatch_cps",
            "effect.legacy_dispatch_cps",
            "closure.new",
            "list.empty",
            "tribute_io.write",
            "core.unrealized_conversion_cast",
            "wasm_gc.struct_new",
            "unknown.operation",
        ] {
            let ir = format!(
                r#"core.module @test {{
  wasm.func @main() -> core.nil {{
    {operation}
    wasm.return
  }}
}}"#
            );
            let error = verify(&ir, TributeBackend::Wasm)
                .expect_err("backend boundary must reject residual and unknown operations");
            assert!(error.contains(operation), "{error}");
        }
    }

    #[test]
    fn rejects_logical_type_nested_in_textual_metadata() {
        let error = verify(
            r#"core.module @test {
  !logical = core.array(tribute_control.func_sig<() -> core.nil>)
  wasm.func @main() -> core.nil {
    wasm.return
  }
}"#,
            TributeBackend::Wasm,
        )
        .expect_err("backend boundary must reject nested logical types");
        assert!(error.contains("tribute_control.func_sig"), "{error}");
    }

    #[test]
    fn rejects_illegal_types_nested_in_callable_signatures() {
        for signature in [
            "func.func_sig<() -> core.ref(core.ptr)>",
            "func.func_sig<(core.ref(core.ptr)) -> core.nil>",
        ] {
            let ir = format!(
                r#"core.module @test {{
  !bad_callable = {signature}
  wasm.func @main() -> core.nil {{
    wasm.return
  }}
}}"#
            );
            let error = verify(&ir, TributeBackend::Wasm)
                .expect_err("callable result and parameter slots must use physical value rules");
            assert!(error.contains("core.ref"), "{error}");
        }
    }

    #[test]
    fn rejects_illegal_types_in_signatures_and_block_arguments() {
        let error = verify(
            r#"core.module @test {
  clif.func @main(%value: core.bytes) -> core.nil {
    clif.return
  }
}"#,
            TributeBackend::Native,
        )
        .expect_err("native value slots cannot carry backend metadata types");
        assert!(error.contains("core.bytes"), "{error}");
    }

    #[test]
    fn rejects_illegal_result_and_attribute_types() {
        for ir in [
            r#"core.module @test {
  clif.func @main() -> core.nil {
    %logical = clif.unknown : tribute_control.func_sig<() -> core.nil>
    clif.return
  }
}"#,
            r#"core.module @test {
  wasm.func @main() -> core.nil {
    wasm.return {metadata = [core.array(tribute_control.func_sig<() -> core.nil>)]}
  }
}"#,
        ] {
            let error = verify(
                ir,
                if ir.contains("clif.func") {
                    TributeBackend::Native
                } else {
                    TributeBackend::Wasm
                },
            )
            .expect_err("every type-bearing surface must be checked");
            assert!(error.contains("tribute_control.func_sig"), "{error}");
        }
    }

    #[test]
    fn rejects_result_producing_proper_tail_transfers() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  clif.func @main(%input: core.i32) -> core.nil {
    clif.return
  }
}"#,
        );
        let function = module.ops(&ctx)[0];
        let block = ctx.region(ctx.op(function).regions[0]).blocks[0];
        let i32 = ctx.value_ty(ctx.block_args(block)[0]);
        let transfer_data = trunk_ir::OperationDataBuilder::new(
            ctx.op(function).location,
            Symbol::new("clif"),
            Symbol::new("return_call"),
        )
        .result(i32)
        .build(&mut ctx);
        let transfer = ctx.create_op(transfer_data);
        ctx.push_op(block, transfer);
        let error = verify_tribute_backend_ready(&ctx, module, TributeBackend::Native)
            .expect_err("proper-tail transfers cannot produce a result")
            .to_string();
        assert!(error.contains("result-producing CPS transfer"), "{error}");
    }

    #[test]
    fn rejects_invalid_or_result_producing_cps_function_abis() {
        for (convention, diagnostic) in [
            (2, "Cps function result is not physically empty"),
            (9, "invalid tribute.calling_convention attribute"),
        ] {
            let ir = format!(
                r#"core.module @test {{
  clif.func {{sym_name = @main, tribute.calling_convention = {convention}, type = func.func_sig<() -> core.i32>}} {{
    ^entry:
    %zero = clif.iconst {{value = 0}} : core.i32
    clif.return %zero
  }}
}}"#
            );
            let error = verify(&ir, TributeBackend::Native)
                .expect_err("backend boundary must reject malformed CPS ABI metadata");
            assert!(error.contains(diagnostic), "{error}");
        }
    }

    #[test]
    fn rejects_private_control_carrier_symbol() {
        let error = verify(
            r#"core.module @test {
  !__tribute_cps_control = adt.enum() {name = @__tribute_cps_control}
  clif.func @main() -> core.nil {
    clif.return
  }
}"#,
            TributeBackend::Native,
        )
        .expect_err("backend boundary must reject the private control carrier");
        assert!(error.contains("__tribute_cps_control"), "{error}");
    }
}
