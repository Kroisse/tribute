//! Lower `heap.cell` to a GC struct with one mutable field.
//!
//! A cell's object holds its value as its only field, so each cell type
//! becomes the literal `adt.struct` of that one field, and `heap.cell_new`,
//! `heap.cell_get`, and `heap.cell_set` become `adt.struct_new`,
//! `adt.struct_get`, and `adt.struct_set` of field 0. Wasm emission declares
//! every GC struct field mutable. The target has no refcounts, so the
//! replaced value needs no release.

use tribute_ir::dialect::{adt, heap};
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::ops::{DialectOp, DialectType};
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::TypeRef;
use trunk_ir::rewrite::{Module, erase_op};
use trunk_ir::types::AttributeMap;
use trunk_ir::walk::{WalkAction, walk_op};

use std::ops::ControlFlow;

/// The literal struct that stores a value of type `contents` as its only
/// field.
fn cell_layout(ctx: &mut IrContext, contents: TypeRef) -> TypeRef {
    adt::Struct::literal(ctx, [contents], AttributeMap::new()).as_type_ref()
}

fn lower(ctx: &mut IrContext, module: Module) {
    let mut ops = Vec::new();
    let _ = walk_op::<()>(ctx, module.op(), &mut |op| {
        ops.push(op);
        ControlFlow::Continue(WalkAction::Advance)
    });
    for op in ops {
        let location = ctx.op(op).location;
        let replacement = if let Ok(new) = heap::CellNew::from_op(ctx, op) {
            let value = new.value(ctx);
            let layout = cell_layout(ctx, ctx.value_ty(value));
            adt::StructNew::operands([value])
                .results(layout)
                .build(ctx, location)
                .op_ref()
        } else if let Ok(get) = heap::CellGet::from_op(ctx, op) {
            let contents = ctx.value_ty(get.result(ctx));
            let layout = cell_layout(ctx, contents);
            adt::StructGet::operands(get.cell(ctx))
                .r#type(layout)
                .field(0)
                .results(contents)
                .build(ctx, location)
                .op_ref()
        } else if let Ok(set) = heap::CellSet::from_op(ctx, op) {
            let value = set.value(ctx);
            let layout = cell_layout(ctx, ctx.value_ty(value));
            adt::StructSet::operands(set.cell(ctx), value)
                .r#type(layout)
                .field(0)
                .build(ctx, location)
                .op_ref()
        } else {
            continue;
        };
        let block = ctx
            .op(op)
            .parent_block
            .expect("a cell operation is in a block");
        ctx.insert_op_before(block, op, replacement);
        let results = ctx.op_results(op).to_vec();
        let replaced = ctx.op_results(replacement).to_vec();
        for (old, new) in results.into_iter().zip(replaced) {
            ctx.replace_all_uses(old, new);
        }
        erase_op(ctx, op);
    }
    crate::closure_lower::substitute_module_types(ctx, module, "cell", |ctx, ty| {
        let contents = heap::Cell::from_type_ref(ctx, ty)?.contents(ctx);
        Some(cell_layout(ctx, contents))
    });
}

/// Lower every `heap.cell` type and operation to its one-field literal
/// struct, before `adt` lowering.
pub struct LowerHeapCells;

impl Pass for LowerHeapCells {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "lower-heap-cells"
    }

    fn run(
        &mut self,
        ctx: &mut IrContext,
        target: core::Module,
        _analyses: &mut AnalysisCache,
    ) -> PassRunResult {
        lower(ctx, target.into());
        Ok(())
    }
}
