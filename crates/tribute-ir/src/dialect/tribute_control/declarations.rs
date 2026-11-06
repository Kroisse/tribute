//! Ability operation and compiler intrinsic declarations, as whole-IR
//! validation reads them.

use super::*;

/// One operation an `ability.decl` of the module declares, as whole-IR
/// validation reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperationDeclaration {
    pub ability: SymbolPath,
    pub op_name: StringRef,
    pub kind: StringRef,
    pub parameter_types: Vec<TypeRef>,
    pub result_type: TypeRef,
}

/// One exact compiler-intrinsic declaration supplied by the typed frontend.
///
/// This metadata is deliberately kept outside textual TrunkIR. A bodyless
/// declaration's operation attribute is accepted only when it matches the
/// identity registered for its symbol. Its signature is the declaration's own;
/// the frontend checks it against the identity before lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompilerIntrinsicDeclaration {
    pub symbol: SymbolPath,
    pub identity: Symbol,
}

impl CompilerIntrinsicDeclaration {
    pub fn new(symbol: SymbolPath, identity: Symbol) -> Self {
        Self { symbol, identity }
    }
}

impl OperationDeclaration {
    pub fn new(
        ability: SymbolPath,
        op_name: StringRef,
        kind: StringRef,
        parameter_types: impl IntoIterator<Item = TypeRef>,
        result_type: TypeRef,
    ) -> Self {
        Self {
            ability,
            op_name,
            kind,
            parameter_types: parameter_types.into_iter().collect(),
            result_type,
        }
    }
}

/// Add an `ability.decl` to `module` for each ability instance of
/// `declarations`, holding the operations given for it, in first-use order.
pub fn declare_operations(
    ctx: &mut IrContext,
    module: Module,
    declarations: &[OperationDeclaration],
) {
    let mut instances = Vec::new();
    for declaration in declarations {
        if !instances.contains(&&declaration.ability) {
            instances.push(&declaration.ability);
        }
    }
    declare_abilities(ctx, module, instances, declarations);
}

/// Add an `ability.decl` to `module` for each of `instances`, in order,
/// holding the operations `declarations` gives for it, if any.
pub fn declare_abilities<'a>(
    ctx: &mut IrContext,
    module: Module,
    instances: impl IntoIterator<Item = &'a SymbolPath>,
    declarations: &[OperationDeclaration],
) {
    use trunk_ir::context::{BlockData, RegionData};

    let location = ctx.op(module.op()).location;
    let module_block = module
        .first_block(ctx)
        .expect("a module that declares ability operations has a body block");
    for ability in instances {
        let block = ctx.create_block(BlockData {
            location,
            args: vec![],
            ops: Default::default(),
            parent_region: None,
        });
        for declaration in declarations
            .iter()
            .filter(|declaration| declaration.ability == *ability)
        {
            let operation = crate::dialect::ability::Operation::operands()
                .op_name(declaration.op_name)
                .kind(declaration.kind)
                .param_types(declaration.parameter_types.iter().copied())
                .result_type(declaration.result_type)
                .build(ctx, location);
            ctx.push_op(block, operation.op_ref());
        }
        let operations = ctx.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![block],
            parent_op: None,
        });
        let name = ability
            .as_simple()
            .expect("an ability instance is declared in the root module");
        let decl = crate::dialect::ability::Decl::operands()
            .sym_name(name.to_string())
            .regions(operations)
            .build(ctx, location);
        ctx.push_op(module_block, decl.op_ref());
    }
}

/// The operations the `ability.decl` definitions of `module` declare, in
/// module order.
pub fn operation_declarations(ctx: &IrContext, module: Module) -> Vec<OperationDeclaration> {
    declared_operations(ctx, module, &mut Vec::new())
}

pub(super) fn declared_operations(
    ctx: &IrContext,
    module: Module,
    errors: &mut Vec<ValidationError>,
) -> Vec<OperationDeclaration> {
    use crate::dialect::ability::{Decl, Operation};

    let mut declarations = Vec::new();
    for &op in module.ops(ctx) {
        let Ok(decl) = Decl::from_op(ctx, op) else {
            continue;
        };
        let Some(ability) = qualified_name(ctx, op) else {
            continue;
        };
        let blocks = &ctx.region(decl.operations(ctx)).blocks;
        for &operation in blocks.iter().flat_map(|block| &ctx.block(*block).ops) {
            match Operation::from_op(ctx, operation) {
                Ok(operation) => declarations.push(OperationDeclaration {
                    ability: ability.clone(),
                    op_name: operation.op_name_ref(ctx),
                    kind: operation.kind_ref(ctx),
                    parameter_types: operation.param_types(ctx).collect(),
                    result_type: operation.result_type(ctx),
                }),
                Err(_) => push_op_error(
                    ctx,
                    operation,
                    errors,
                    "ability.decl may hold only ability.operation",
                ),
            }
        }
    }
    declarations
}
