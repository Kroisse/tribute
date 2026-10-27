//! Declarations the logical module generates: ability operations, struct
//! accessors, extern and prelude functions.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use tribute_ir::dialect::tribute_control::{
    CompilerIntrinsicDeclaration, OperationDeclaration, Row, RowBinder,
};
use tribute_ir::dialect::{adt, tribute_control};
use trunk_ir::context::{BlockArgData, BlockData, IrContext, RegionData};
use trunk_ir::ops::DialectOp;
use trunk_ir::refs::{BlockRef, TypeRef, ValueRef};
use trunk_ir::types::{Attribute, Location};
use trunk_ir::{Symbol, SymbolPath};

use crate::SortedMap;
use crate::ast::{CtorId, Decl, DeclEntry, ExternFuncDecl, OpDeclKind, TypedRef};
use crate::ast_to_ir::context::{IrLoweringCtx, LogicalGeneratedSignature};
use crate::ast_to_ir::lower::{FuncSignature, IrBuilder};

use super::callable::operation_kind_text;
use super::{local_callables, op, result};

pub(super) struct Declarations<'db> {
    // TypeRef is an arena index and therefore has deterministic total order only
    // through its debug representation.  Preserve first source use explicitly.
    pub(super) values: Vec<OperationDeclaration>,
    /// The ability instances operations were recorded for, in first-use order.
    pub(super) instances: Vec<(
        SymbolPath,
        crate::ast::AbilityId<'db>,
        Vec<crate::ast::Type<'db>>,
    )>,
    pub(super) compiler_intrinsics: Vec<CompilerIntrinsicDeclaration>,
    pub(super) schemas: HashMap<crate::ast::AbilityId<'db>, crate::typeck::AbilityInfo<'db>>,
    pub(super) handler_operations:
        SortedMap<crate::ast::NodeId, crate::typeck::InstantiatedHandlerOperation<'db>>,
    pub(super) perform_operations:
        SortedMap<crate::ast::NodeId, crate::typeck::InstantiatedPerformOperation<'db>>,
    pub(super) lambda_signatures:
        SortedMap<crate::ast::NodeId, crate::typeck::LambdaSignature<'db>>,
    pub(super) exhaustive_cases: HashSet<crate::ast::NodeId>,
    pub(super) evidence_plans: SortedMap<crate::ast::NodeId, Vec<crate::typeck::EvidenceStep<'db>>>,
    pub(super) local_instances:
        SortedMap<crate::ast::NodeId, crate::typeck::LocalCallableInstance<'db>>,
    /// The instance each reference to a generic function selects.
    pub(super) function_instances:
        SortedMap<crate::ast::NodeId, crate::typeck::FunctionInstance<'db>>,
    pub(super) local_callables: local_callables::Plan<'db>,
    /// The plan of each function, by its declaration, until it is lowered.
    pub(super) function_local_callables: HashMap<crate::ast::NodeId, local_callables::Plan<'db>>,
    /// The uses of the let-bound lambdas of the function being lowered.
    pub(super) local_uses: Vec<crate::monomorphize::LocalInstantiation<'db>>,
}

/// Fully typed semantic inputs for one source ability-operation call.
pub(super) struct PerformMetadata<'a, 'db> {
    pub(super) ability: crate::ast::AbilityId<'db>,
    pub(super) operation: Symbol,
    pub(super) kind: OpDeclKind,
    pub(super) semantic: &'a crate::typeck::InstantiatedPerformOperation<'db>,
}

impl<'db> Declarations<'db> {
    /// Note that the module names the ability instance `symbol`.
    pub(super) fn instance(
        &mut self,
        symbol: &SymbolPath,
        ability: crate::ast::AbilityId<'db>,
        arguments: &[crate::ast::Type<'db>],
    ) {
        if !self.instances.iter().any(|(known, ..)| known == symbol) {
            self.instances
                .push((symbol.clone(), ability, arguments.to_vec()));
        }
    }

    pub(super) fn record(
        &mut self,
        ir: &IrContext,
        declaration: OperationDeclaration,
        ability: crate::ast::AbilityId<'db>,
        arguments: &[crate::ast::Type<'db>],
    ) {
        self.instance(&declaration.ability, ability, arguments);
        if let Some(existing) = self.values.iter().find(|candidate| {
            candidate.ability == declaration.ability && candidate.op_name == declaration.op_name
        }) {
            assert!(
                existing == &declaration,
                "conflicting instantiated declaration for ability operation {}",
                ir.str(declaration.op_name)
            );
            return;
        }
        self.values.push(declaration);
    }
}

/// Every operation of each ability instance the module names, with the
/// parameter and result types of that instance.
pub(super) fn instance_operations<'db>(
    ctx: &IrLoweringCtx<'db>,
    ir: &mut IrContext,
    declarations: &Declarations<'db>,
) -> Vec<OperationDeclaration> {
    let instantiate = |ty, arguments: &[crate::ast::Type<'db>]| {
        match crate::typeck::subst::substitute_bound_vars(ctx.db, ty, arguments) {
            crate::typeck::subst::SubstResult::Ok(ty) => ty,
            crate::typeck::subst::SubstResult::OutOfBounds { .. } => {
                panic!("ability operation substitution failed")
            }
        }
    };
    let mut operations = Vec::new();
    for (instance, ability, arguments) in &declarations.instances {
        let schema = &declarations.schemas[ability];
        let mut declared: Vec<_> = schema.operations.values().collect();
        declared.sort_by(|a, b| a.name.with_str(|a| b.name.with_str(|b| a.cmp(b))));
        for operation in declared {
            let parameters: Vec<_> = operation
                .param_types
                .iter()
                .map(|ty| ctx.convert_logical_type(ir, instantiate(*ty, arguments)))
                .collect();
            let result =
                ctx.convert_logical_type(ir, instantiate(operation.return_type, arguments));
            operations.push(OperationDeclaration::new(
                instance.clone(),
                ir.intern_symbol_text(&operation.name),
                ir.intern_str(operation_kind_text(operation.kind)),
                parameters,
                result,
            ));
        }
    }
    operations
}

/// The names of the setter and the modifier of the field whose getter is
/// `getter`: `T::f::set` and `T::f::modify`.
pub(super) fn field_update_names(getter: &SymbolPath) -> [SymbolPath; 2] {
    crate::ast::FIELD_LENS_FUNCTIONS.map(|name| getter.child(name))
}

pub(super) fn prescan_struct_accessor_signatures<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    entries: &[DeclEntry<TypedRef<'db>>],
) {
    for entry in entries {
        ctx.set_module(entry.module.clone());
        let Decl::Struct(declaration) = &entry.decl else {
            continue;
        };
        let prefix = ctx.module_path().clone();
        let qualified = prefix.child(declaration.name.clone());
        let type_name = super::super::qualified_type_name(ctx.db, &CtorId::new(ctx.db, qualified));
        let struct_type = ctx.adt_typeref(ir, &type_name);
        let layout = ctx
            .get_type(&type_name)
            .unwrap_or_else(|| panic!("missing logical struct layout for accessor"));
        let layout_fields = tribute_ir::dialect::adt::layout::get_struct_fields(ir, layout)
            .unwrap_or_else(|| panic!("malformed logical struct layout"));
        for (index, field) in declaration.fields.iter().enumerate() {
            let field_name = field.name.clone().unwrap_or_else(|| Symbol::new("_"));
            let getter_name = prefix
                .child(declaration.name.clone())
                .child(field_name.clone());
            let field_type = layout_fields
                .get(index)
                .map(|(_, ty)| *ty)
                .unwrap_or_else(|| panic!("missing logical struct accessor field type"));
            let closed = |param_types, return_type| LogicalGeneratedSignature {
                param_types,
                return_type,
                row: Row::default(),
                binders: vec![],
            };
            ctx.register_logical_generated_signature(
                &getter_name,
                closed(vec![struct_type], field_type),
            );
            if field.name.is_none() {
                continue;
            }
            let [setter_name, modifier_name] = field_update_names(&getter_name);
            ctx.register_logical_generated_signature(
                &setter_name,
                closed(vec![struct_type, field_type], struct_type),
            );
            // The modifier performs whatever its callback performs.
            let row = Row::new([], Some(0));
            let callback = tribute_control::FuncSig::params([field_type], field_type)
                .row(row.clone())
                .intern(ir)
                .as_type_ref();
            ctx.register_logical_generated_signature(
                &modifier_name,
                LogicalGeneratedSignature {
                    param_types: vec![struct_type, callback],
                    return_type: struct_type,
                    row: row.clone(),
                    binders: vec![RowBinder::Param],
                },
            );
        }
    }
}

pub(super) fn prescan_source_functions<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    entries: &[DeclEntry<TypedRef<'db>>],
) {
    for entry in entries {
        ctx.set_module(entry.module.clone());
        match &entry.decl {
            Decl::Function(function) => {
                ctx.register_logical_source_function(ctx.qualify_name(&function.name));
            }
            Decl::ExternFunction(function) => {
                ctx.register_logical_source_function(ctx.qualify_name(&function.name));
            }
            _ => {}
        }
    }
}

pub(super) fn lower_struct_accessors<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    top: BlockRef,
    declaration: crate::ast::StructDecl,
) {
    let location = ctx.location(declaration.id);
    let prefix = ctx.module_path().clone();
    let qualified = prefix.child(declaration.name.clone());
    let type_name = super::super::qualified_type_name(ctx.db, &CtorId::new(ctx.db, qualified));
    let struct_type = ctx.adt_typeref(ir, &type_name);
    let layout_type = ctx
        .get_type(&type_name)
        .unwrap_or_else(|| panic!("prescan did not register struct layout {type_name}"));
    for (index, field) in declaration.fields.iter().enumerate() {
        let field_name = field.name.clone().unwrap_or_else(|| Symbol::new("_"));
        let getter_name = prefix
            .child(declaration.name.clone())
            .child(field_name.clone());
        let field_type = tribute_ir::dialect::adt::layout::get_struct_fields(ir, layout_type)
            .and_then(|fields| fields.get(index).map(|(_, ty)| *ty))
            .unwrap_or_else(|| panic!("missing logical struct accessor field type"));
        let entry = ir.create_block(BlockData {
            location,
            args: vec![BlockArgData {
                ty: struct_type,
                attrs: Default::default(),
            }],
            ops: Default::default(),
            parent_region: None,
        });
        let field_value = adt::StructGet::operands(ir.block_arg(entry, 0))
            .r#type(layout_type)
            .field(index as u32)
            .results(field_type)
            .build(ir, location);
        ir.push_op(entry, field_value.op_ref());
        let field_result = field_value.result(ir);
        op(
            ir,
            entry,
            location,
            tribute_control::Return::DEF,
            |builder| builder.operand(field_result),
        );
        let body = ir.create_region(RegionData {
            location,
            blocks: trunk_ir::smallvec::smallvec![entry],
            parent_op: None,
        });
        let callable = tribute_control::FuncSig::params([struct_type], field_type)
            .intern(ir)
            .as_type_ref();
        let getter =
            tribute_control::func_declaration(ir, location, &getter_name.to_symbol(), callable);
        ir.push_op_region(getter.op_ref(), body);
        ir.push_op(top, getter.op_ref());

        if field.name.is_none() {
            continue;
        }
        let field_types: Vec<TypeRef> =
            tribute_ir::dialect::adt::layout::get_struct_fields(ir, layout_type)
                .unwrap_or_else(|| panic!("malformed logical struct layout"))
                .iter()
                .map(|(_, ty)| *ty)
                .collect();
        let [setter_name, modifier_name] = field_update_names(&getter_name);
        let callback_row = Row::new([], Some(0));
        let callback = tribute_control::FuncSig::params([field_type], field_type)
            .row(callback_row.clone())
            .intern(ir)
            .as_type_ref();
        for (name, argument_ty, modifies) in [
            (setter_name, field_type, false),
            (modifier_name, callback, true),
        ] {
            let entry = ir.create_block(BlockData {
                location,
                args: [struct_type, argument_ty]
                    .map(|ty| BlockArgData {
                        ty,
                        attrs: Default::default(),
                    })
                    .to_vec(),
                ops: Default::default(),
                parent_region: None,
            });
            let subject = ir.block_arg(entry, 0);
            let argument = ir.block_arg(entry, 1);
            // The modifier applies its callback to the field's current value.
            let replacement = if modifies {
                let current = adt::StructGet::operands(subject)
                    .r#type(layout_type)
                    .field(index as u32)
                    .results(field_type)
                    .build(ir, location);
                ir.push_op(entry, current.op_ref());
                let current = current.result(ir);
                let call = op(
                    ir,
                    entry,
                    location,
                    tribute_control::CallIndirect::DEF,
                    |builder| {
                        builder
                            .operand(argument)
                            .operand(current)
                            .result(field_type)
                    },
                );
                result(ir, call)
            } else {
                argument
            };
            let values: Vec<ValueRef> = field_types
                .iter()
                .enumerate()
                .map(|(other, ty)| {
                    if other == index {
                        return replacement;
                    }
                    let get = adt::StructGet::operands(subject)
                        .r#type(layout_type)
                        .field(other as u32)
                        .results(*ty)
                        .build(ir, location);
                    ir.push_op(entry, get.op_ref());
                    get.result(ir)
                })
                .collect();
            let updated = adt::StructNew::operands(values)
                .r#type(layout_type)
                .results(struct_type)
                .build(ir, location);
            ir.push_op(entry, updated.op_ref());
            let updated = updated.result(ir);
            op(
                ir,
                entry,
                location,
                tribute_control::Return::DEF,
                |builder| builder.operand(updated),
            );
            let body = ir.create_region(RegionData {
                location,
                blocks: trunk_ir::smallvec::smallvec![entry],
                parent_op: None,
            });
            let row = if modifies {
                callback_row.clone()
            } else {
                Row::default()
            };
            let callable =
                tribute_control::FuncSig::params([struct_type, argument_ty], struct_type)
                    .row(row)
                    .intern(ir)
                    .as_type_ref();
            let function =
                tribute_control::func_declaration(ir, location, &name.to_symbol(), callable);
            if modifies {
                let param = ir.string_attr(RowBinder::Param.keyword());
                ir.op_attrs_mut(function.op_ref()).insert(
                    tribute_control::ROW_BINDERS_ATTR,
                    Attribute::List(vec![param]),
                );
            }
            ir.push_op_region(function.op_ref(), body);
            ir.push_op(top, function.op_ref());
        }
    }
}

pub(super) fn lower_extern<'db>(
    ctx: &mut IrLoweringCtx<'db>,
    ir: &mut IrContext,
    top: BlockRef,
    decl: ExternFuncDecl,
    declarations: &mut Declarations<'db>,
) {
    let location = ctx.location(decl.id);
    let qualified = ctx.qualify_name(&decl.name);
    let signature = if qualified == [&decl.name] {
        FuncSignature::lookup_logical(ctx, ir, &qualified)
    } else {
        FuncSignature::lookup_logical(ctx, ir, &qualified)
            .or_else(|| FuncSignature::lookup_logical(ctx, ir, std::slice::from_ref(&decl.name)))
    }
    .unwrap_or_else(|| {
        panic!(
            "missing typechecked signature for extern function {}",
            decl.name
        )
    });
    let callable = signature.callable_type(ir);
    let name = ctx.function_symbol(&ctx.qualify_name(&decl.name));
    let function = tribute_control::func_declaration(ir, location, &name, callable);
    signature.attach_rows(ir, function.op_ref());
    if let Some(identity) = ctx.compiler_intrinsic(decl.id) {
        let identity_text = ir.intern_symbol_text(&identity);
        ir.op_attrs_mut(function.op_ref()).insert(
            tribute_control::COMPILER_INTRINSIC_ATTR,
            Attribute::String(identity_text),
        );
        declarations
            .compiler_intrinsics
            .push(CompilerIntrinsicDeclaration::new(
                SymbolPath::new([name]),
                identity,
            ));
        declarations
            .compiler_intrinsics
            .sort_by_key(|declaration| (declaration.symbol.clone(), declaration.identity.clone()));
    }
    let abi = decl.abi.with_str(|abi| ir.string_attr(abi));
    ir.op_attrs_mut(function.op_ref()).insert("abi", abi);
    // C linkage has one flat namespace: the declaration links under its
    // declared name, wherever it is declared.
    if decl.abi == "C" {
        let link_name = ir.intern_symbol_text(&decl.name);
        ir.op_attrs_mut(function.op_ref()).insert(
            trunk_ir::symbol_table::LINK_NAME,
            Attribute::String(link_name),
        );
    }
    ir.push_op(top, function.op_ref());
}

pub(super) fn ensure_prelude_declaration(
    builder: &mut IrBuilder<'_, '_>,
    location: Location,
    name: &SymbolPath,
    signature: &FuncSignature,
) {
    if builder.ctx.is_logical_source_function(name)
        || builder
            .ctx
            .lookup_logical_generated_signature(name)
            .is_some()
        || !builder.ctx.mark_logical_extern_emitted(name.clone())
    {
        return;
    }
    let callable = signature.callable_type(builder.ir);
    let symbol = builder.ctx.function_symbol(name);
    let declaration = tribute_control::func_declaration(builder.ir, location, &symbol, callable);
    signature.attach_rows(builder.ir, declaration.op_ref());
    let top = builder
        .ctx
        .module_block()
        .expect("logical module block must be set before lowering calls");
    builder.ir.push_op(top, declaration.op_ref());
}
