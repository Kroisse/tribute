//! Recursive target-type physicalization for the final Wasm boundary.
//!
//! This runs after conversion-cast materialization and before typed WasmGC
//! operations receive module-local indices.  It preserves nominal outer types
//! while converting every target-visible primitive leaf and nested type
//! attribute through the canonical Wasm type converter.

use std::collections::{HashMap, HashSet};

use trunk_ir::Symbol;
use trunk_ir::context::IrContext;
use trunk_ir::refs::{BlockRef, OpRef, TypeRef};
use trunk_ir::rewrite::Module;
use trunk_ir::types::{Attribute, TypeData};
use trunk_ir_wasm_backend::gc_types::concrete_wasm_ref_type;

use super::type_converter::wasm_type_converter;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmTypePhysicalizationError(String);

impl WasmTypePhysicalizationError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for WasmTypePhysicalizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for WasmTypePhysicalizationError {}

/// Physicalize every type surface visible to the final Wasm backend.
///
/// Conversion is planned in full before an alias, operation, or block argument
/// reference is changed.  Interning target types during planning is harmless;
/// any error leaves the module's canonical IR byte-identical.
pub(crate) fn physicalize_wasm_target_types(
    ctx: &mut IrContext,
    module: Module,
) -> Result<(), WasmTypePhysicalizationError> {
    let ops = collect_ops(ctx, module.op());
    let blocks = collect_blocks(ctx, &ops);
    let aliases = ctx.type_aliases().to_vec();
    let mut converter = RecursiveWasmTypeConverter::new(ctx);
    let mut plan = WasmTypePhysicalizationPlan::default();

    for (name, ty) in aliases {
        let converted = converter.convert_type(ty)?;
        if converted != ty {
            plan.aliases.push((name, converted));
        }
    }

    for op in ops {
        let result_types = converter.ctx.op_result_types(op).to_vec();
        for (index, ty) in result_types.into_iter().enumerate() {
            let converted = converter.convert_type(ty)?;
            if converted != ty {
                plan.op_results.push((op, index as u32, converted));
            }
        }
        let attributes = converter
            .ctx
            .op(op)
            .attributes
            .iter()
            .map(|(name, value)| (*name, value.clone()))
            .collect::<Vec<_>>();
        for (name, value) in attributes {
            let converted = converter.convert_attribute(value.clone())?;
            if converted != value {
                plan.op_attributes.push((op, name, converted));
            }
        }
    }

    for block in blocks {
        let args = converter.ctx.block(block).args.clone();
        for (index, arg) in args.into_iter().enumerate() {
            let converted = converter.convert_type(arg.ty)?;
            if converted != arg.ty {
                plan.block_arg_types.push((block, index, converted));
            }
            for (name, value) in arg.attrs {
                let converted = converter.convert_attribute(value.clone())?;
                if converted != value {
                    plan.block_arg_attributes
                        .push((block, index, name, converted));
                }
            }
        }
    }

    drop(converter);
    for (name, ty) in plan.aliases {
        ctx.register_type_alias(name, ty);
    }
    for (op, index, ty) in plan.op_results {
        ctx.set_op_result_type(op, index, ty);
    }
    for (op, name, value) in plan.op_attributes {
        ctx.op_mut(op).attributes.insert(name, value);
    }
    for (block, index, ty) in plan.block_arg_types {
        ctx.set_block_arg_type(block, index as u32, ty);
    }
    for (block, index, name, value) in plan.block_arg_attributes {
        ctx.block_mut(block).args[index].attrs.insert(name, value);
    }
    Ok(())
}

#[derive(Default)]
struct WasmTypePhysicalizationPlan {
    aliases: Vec<(Symbol, TypeRef)>,
    op_results: Vec<(OpRef, u32, TypeRef)>,
    op_attributes: Vec<(OpRef, Symbol, Attribute)>,
    block_arg_types: Vec<(BlockRef, usize, TypeRef)>,
    block_arg_attributes: Vec<(BlockRef, usize, Symbol, Attribute)>,
}

struct RecursiveWasmTypeConverter<'a> {
    ctx: &'a mut IrContext,
    converter: trunk_ir::rewrite::TypeConverter,
    cache: HashMap<TypeRef, TypeRef>,
    visiting: HashSet<TypeRef>,
}

impl<'a> RecursiveWasmTypeConverter<'a> {
    fn new(ctx: &'a mut IrContext) -> Self {
        let converter = wasm_type_converter(ctx);
        Self {
            ctx,
            converter,
            cache: HashMap::new(),
            visiting: HashSet::new(),
        }
    }

    fn convert_type(&mut self, ty: TypeRef) -> Result<TypeRef, WasmTypePhysicalizationError> {
        if let Some(&converted) = self.cache.get(&ty) {
            return Ok(converted);
        }
        if !self.visiting.insert(ty) {
            return Err(WasmTypePhysicalizationError::new(
                "wasm target type physicalization: recursive TypeRef graph is unsupported",
            ));
        }

        let data = self.ctx.types.get(ty).clone();
        self.validate_type(&data)?;
        if data.dialect == Symbol::new("core") && data.name == Symbol::new("ref") {
            let converted = self.convert_core_ref(ty, &data)?;
            self.visiting.remove(&ty);
            self.cache.insert(ty, converted);
            return Ok(converted);
        }
        let converted_params = data
            .params
            .iter()
            .copied()
            .map(|parameter| self.convert_type(parameter))
            .collect::<Result<Vec<_>, _>>()?;
        let converted_attributes = data
            .attrs
            .iter()
            .map(|(name, value)| Ok((*name, self.convert_attribute(value.clone())?)))
            .collect::<Result<Vec<_>, WasmTypePhysicalizationError>>()?;

        let converted = if preserves_nominal_identity(&data) {
            self.rebuild(&data, converted_params, converted_attributes, ty)
        } else if data.dialect == Symbol::new("adt") && data.name == Symbol::new("typeref") {
            // A regular nominal reference has no target-level layout of its
            // own.  Its source-only `name` metadata must not create a second
            // wasm.structref TypeRef; all ordinary nominal references share
            // the canonical abstract struct reference in the Wasm ABI.
            self.converter.convert_type(self.ctx, ty).ok_or_else(|| {
                WasmTypePhysicalizationError::new(
                    "wasm target type physicalization: ordinary adt.typeref has no wasm.structref conversion",
                )
            })?
        } else if let Some(target) = self.converter.convert_type(self.ctx, ty) {
            self.rebuild_target(target, converted_attributes)
        } else {
            self.rebuild(&data, converted_params, converted_attributes, ty)
        };
        self.visiting.remove(&ty);
        self.cache.insert(ty, converted);
        Ok(converted)
    }

    fn convert_attribute(
        &mut self,
        attribute: Attribute,
    ) -> Result<Attribute, WasmTypePhysicalizationError> {
        match attribute {
            Attribute::Type(ty) => Ok(Attribute::Type(self.convert_type(ty)?)),
            Attribute::List(values) => Ok(Attribute::List(
                values
                    .into_iter()
                    .map(|value| self.convert_attribute(value))
                    .collect::<Result<Vec<_>, _>>()?,
            )),
            attribute => Ok(attribute),
        }
    }

    /// Preserve the canonical source pointee for the only concrete `core.ref`
    /// form representable by the Wasm ABI.  Recursing through that pointee
    /// would turn `core.array(core.i8)` into `wasm.arrayref` and lose the
    /// concrete Bytes-array identity consumed by GC collection and emission.
    fn convert_core_ref(
        &mut self,
        original: TypeRef,
        data: &TypeData,
    ) -> Result<TypeRef, WasmTypePhysicalizationError> {
        if concrete_wasm_ref_type(self.ctx, original).is_none() {
            return Err(WasmTypePhysicalizationError::new(
                "wasm target type physicalization: unsupported core.ref pointee; expected core.array(core.i8)",
            ));
        }
        let attributes = data
            .attrs
            .iter()
            .map(|(name, value)| Ok((*name, self.convert_attribute(value.clone())?)))
            .collect::<Result<Vec<_>, WasmTypePhysicalizationError>>()?;
        Ok(self.rebuild(data, data.params.to_vec(), attributes, original))
    }

    fn validate_type(&self, data: &TypeData) -> Result<(), WasmTypePhysicalizationError> {
        if data.dialect == Symbol::new("closure") && data.name == Symbol::new("closure") {
            let Some(&signature) = data.params.first() else {
                return Err(WasmTypePhysicalizationError::new(
                    "wasm target type physicalization: closure.closure requires one core.func parameter",
                ));
            };
            if data.params.len() != 1
                || self.ctx.types.get(signature).dialect != Symbol::new("core")
                || self.ctx.types.get(signature).name != Symbol::new("func")
            {
                return Err(WasmTypePhysicalizationError::new(
                    "wasm target type physicalization: closure.closure requires one core.func parameter",
                ));
            }
        }
        Ok(())
    }

    fn rebuild(
        &mut self,
        data: &TypeData,
        params: Vec<TypeRef>,
        attributes: Vec<(Symbol, Attribute)>,
        original: TypeRef,
    ) -> TypeRef {
        if data.params.as_slice() == params.as_slice()
            && attributes
                .iter()
                .all(|(name, value)| data.attrs.get(name) == Some(value))
        {
            return original;
        }
        let mut rebuilt = data.clone();
        rebuilt.params = params.into();
        rebuilt.attrs.clear();
        rebuilt.attrs.extend(attributes);
        self.ctx.types.intern(rebuilt)
    }

    fn rebuild_target(&mut self, target: TypeRef, attributes: Vec<(Symbol, Attribute)>) -> TypeRef {
        if attributes.is_empty() {
            return target;
        }
        let mut data = self.ctx.types.get(target).clone();
        data.attrs.extend(attributes);
        self.ctx.types.intern(data)
    }
}

fn preserves_nominal_identity(data: &TypeData) -> bool {
    data.dialect == Symbol::new("adt")
        && (data.name == Symbol::new("struct")
            || data.name == Symbol::new("enum")
            || data.name == Symbol::new("variant_inst")
            // Ordinary nominal references are physicalized to wasm.structref.
            // The exact CPS Parent is the one reference whose nominal layout
            // must survive for GC type-index collection.
            || (data.name == Symbol::new("typeref")
                && data.attrs.contains_key("tribute.cps_parent_result")))
}

fn collect_ops(ctx: &IrContext, root: OpRef) -> Vec<OpRef> {
    fn visit(ctx: &IrContext, op: OpRef, ops: &mut Vec<OpRef>) {
        ops.push(op);
        for region in ctx.op(op).regions.iter().copied() {
            for block in ctx.region(region).blocks.iter().copied() {
                for child in ctx.block(block).ops.iter().copied() {
                    visit(ctx, child, ops);
                }
            }
        }
    }
    let mut ops = Vec::new();
    visit(ctx, root, &mut ops);
    ops
}

fn collect_blocks(ctx: &IrContext, ops: &[OpRef]) -> Vec<BlockRef> {
    let mut blocks = Vec::new();
    for &op in ops {
        for region in ctx.op(op).regions.iter().copied() {
            blocks.extend(ctx.region(region).blocks.iter().copied());
        }
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::types::TypeDataBuilder;

    #[test]
    fn physicalizes_nested_target_type_surfaces_without_erasing_nominal_identity() {
        let input = r#"core.module @test {
  !parent = adt.typeref() {name = @Parent, tribute.cps_parent_result = tribute_rt.anyref}
  !layout = adt.struct() {fields = [[@payload, tribute_rt.anyref], [@items, core.array(tribute_rt.bool)]], metadata = [tribute_rt.intref, [tribute_rt.float, !parent]], name = @Parent}
  func.func @outer(%parent: !parent, %payload: tribute_rt.anyref) -> tribute_rt.anyref {
    func.func @nested(%captured: tribute_rt.anyref) -> tribute_rt.anyref {
      func.unreachable
    }
    %packed = wasm_gc.struct_new %payload {type = !layout} : !parent
    func.unreachable
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        physicalize_wasm_target_types(&mut ctx, module).unwrap();

        let printed = print_module(&ctx, module.op());
        assert!(!printed.contains("tribute_rt.anyref"), "{printed}");
        assert!(!printed.contains("tribute_rt.bool"), "{printed}");
        assert!(!printed.contains("tribute_rt.intref"), "{printed}");
        assert!(!printed.contains("tribute_rt.float"), "{printed}");
        assert!(printed.contains("!parent = adt.typeref()"), "{printed}");
        assert!(printed.contains("!layout = adt.struct()"), "{printed}");
        assert!(printed.contains("wasm_gc.struct_new"), "{printed}");
    }

    #[test]
    fn physicalized_nominal_aliases_and_nested_signature_reparse() {
        let input = r#"core.module @test {
  !parent = adt.typeref() {name = @Parent, tribute.cps_parent_result = tribute_rt.anyref}
  !layout = adt.struct() {fields = [[@payload, tribute_rt.anyref]], metadata = [tribute_rt.intref, !parent], name = @Parent}
  func.func @outer(%parent: !parent, %payload: tribute_rt.anyref) -> tribute_rt.anyref {
    func.unreachable
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        physicalize_wasm_target_types(&mut ctx, module).unwrap();
        let printed = print_module(&ctx, module.op());
        let mut reparsed = IrContext::new();
        parse_test_module(&mut reparsed, &printed);
    }

    #[test]
    fn malformed_closure_type_fails_before_mutation() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @f() -> core.nil {
    func.return
  }
}"#,
        );
        let malformed = ctx
            .types
            .intern(TypeDataBuilder::new(Symbol::new("closure"), Symbol::new("closure")).build());
        ctx.register_type_alias(Symbol::new("malformed"), malformed);
        let before = print_module(&ctx, module.op());

        let error = physicalize_wasm_target_types(&mut ctx, module).unwrap_err();

        assert!(error.to_string().contains("closure.closure requires"));
        assert_eq!(print_module(&ctx, module.op()), before);
        assert_eq!(
            ctx.type_alias_by_name(Symbol::new("malformed")),
            Some(malformed)
        );
    }

    #[test]
    fn preserves_bytes_backing_reference_identity_and_nullability() {
        let input = r#"core.module @test {
  !bytes_data = core.ref(core.array(core.i8)) {nullable = false}
  !optional_bytes_data = core.ref(core.array(core.i8)) {nullable = true}
  !layout = adt.struct() {fields = [[@data, !bytes_data], [@optional_data, !optional_bytes_data]], name = @BytesFields}
  func.func @f(%data: !bytes_data, %optional_data: !optional_bytes_data) -> core.nil {
    %fields = wasm_gc.struct_new %data, %optional_data {type = !layout} : !layout
    func.unreachable
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);

        physicalize_wasm_target_types(&mut ctx, module).unwrap();

        let printed = print_module(&ctx, module.op());
        assert!(
            printed.contains("core.ref(core.array(core.i8)) {nullable = false}"),
            "{printed}"
        );
        assert!(
            printed.contains("core.ref(core.array(core.i8)) {nullable = true}"),
            "{printed}"
        );
        assert!(!printed.contains("core.ref(wasm.arrayref)"), "{printed}");
        let mut reparsed = IrContext::new();
        parse_test_module(&mut reparsed, &printed);
    }

    #[test]
    fn unsupported_core_reference_fails_before_mutation() {
        let input = r#"core.module @test {
  !unsupported = core.ref(core.i32) {nullable = false}
  func.func @f(%value: !unsupported) -> core.nil {
    func.unreachable
  }
}"#;
        let mut ctx = IrContext::new();
        let module = parse_test_module(&mut ctx, input);
        let before = print_module(&ctx, module.op());

        let error = physicalize_wasm_target_types(&mut ctx, module).unwrap_err();

        assert!(error.to_string().contains("unsupported core.ref pointee"));
        assert_eq!(print_module(&ctx, module.op()), before);
    }
}
