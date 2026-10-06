//! Lower `tribute_rtti.descriptor` to a read of the RC header.
//!
//! A native allocation's runtime type descriptor number is the RTTI index in
//! its RC header, which precedes the payload a managed reference points to.
//!
//! The operation takes a managed reference, so it is lowered before native
//! type conversion turns its operand into a `core.ptr`.

use tribute_ir::dialect::tribute_rt::{RC_HEADER_SIZE, RTTI_IDX_OFFSET};
use tribute_ir::dialect::tribute_rtti;
use trunk_ir::analysis::AnalysisCache;
use trunk_ir::context::IrContext;
use trunk_ir::dialect::{clif, core};
use trunk_ir::ops::DialectOp;
use trunk_ir::pass::{Pass, PassRunResult};
use trunk_ir::refs::OpRef;
use trunk_ir::rewrite::{
    Module, PatternApplicator, PatternRewriter, RewritePattern, TypeConverter,
};

/// PassManager-friendly [`lower`].
pub struct DescriptorToClif;

impl Pass for DescriptorToClif {
    type Target = core::Module;

    fn name(&self) -> &'static str {
        "descriptor-to-clif"
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

/// Replace every `tribute_rtti.descriptor` in `module` with a load of the
/// RTTI index from the RC header.
fn lower(ctx: &mut IrContext, module: Module) {
    PatternApplicator::new(TypeConverter::new())
        .add_pattern(DescriptorPattern)
        .apply_partial(ctx, module);
}

struct DescriptorPattern;

impl RewritePattern for DescriptorPattern {
    fn match_and_rewrite(
        &self,
        ctx: &mut IrContext,
        op: OpRef,
        rewriter: &mut PatternRewriter<'_>,
    ) -> bool {
        let Ok(descriptor) = tribute_rtti::Descriptor::from_op(ctx, op) else {
            return false;
        };
        let loc = ctx.op(op).location;
        let header_offset = RTTI_IDX_OFFSET as i32 - RC_HEADER_SIZE as i32;
        let index_load = clif::Load::operands(descriptor.r#ref(ctx))
            .offset(header_offset)
            .results(descriptor.result_ty(ctx))
            .build(ctx, loc);
        rewriter.replace_op(index_load.op_ref());
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::printer::print_module;
    use trunk_ir::validation::validate_op_schemas;

    #[test]
    fn descriptor_loads_the_header_index_of_a_managed_reference() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @test(%choice: adt.typeref<{name = "Choice"}>) -> core.i32 {
    %0 = tribute_rtti.descriptor %choice : core.i32
    func.return %0
  }
}"#,
        );
        assert!(validate_op_schemas(&ctx, module.op()).is_ok());

        lower(&mut ctx, module);

        let printed = print_module(&ctx, module.op());
        // The RTTI index sits in the RC header, 4 bytes before the payload.
        assert!(
            printed.contains("clif.load %0 {offset = -4} : core.i32"),
            "{printed}"
        );
        assert!(!printed.contains("tribute_rtti."), "{printed}");
    }

    #[test]
    fn pass_lowers_every_descriptor_read_of_a_module() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  func.func @first(%value: tribute_rt.anyref) -> core.i32 {
    %0 = tribute_rtti.descriptor %value : core.i32
    func.return %0
  }
  func.func @second(%value: adt.typeref<{name = "Choice"}>) -> core.i32 {
    %0 = tribute_rtti.descriptor %value : core.i32
    func.return %0
  }
}"#,
        );
        let core_module = core::Module::from_op(&ctx, module.op()).expect("core.module");

        let pass = DescriptorToClif;
        assert_eq!(pass.name(), "descriptor-to-clif");
        let mut manager = trunk_ir::pass::PassManager::new();
        manager.add_pass(pass);
        manager.with_debug_verifier();
        manager
            .run(&mut ctx, core_module, &mut AnalysisCache::new())
            .expect("descriptor lowering");

        let printed = print_module(&ctx, module.op());
        assert_eq!(printed.matches("clif.load").count(), 2, "{printed}");
        assert!(!printed.contains("tribute_rtti."), "{printed}");
    }

    #[test]
    fn descriptor_of_an_unmanaged_pointer_or_scalar_is_rejected() {
        for ty in ["core.ptr", "core.i32", "core.bytes"] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    r#"core.module @test {{
  func.func @test(%value: {ty}) -> core.i32 {{
    %0 = tribute_rtti.descriptor %value : core.i32
    func.return %0
  }}
}}"#
                ),
            );
            assert!(!validate_op_schemas(&ctx, module.op()).is_ok(), "{ty}");
        }
        for ty in [
            r#"adt.typeref<{name = "Choice"}>"#,
            "adt.struct<Point(x: core.i32)>",
            "tribute_rt.anyref",
        ] {
            let mut ctx = IrContext::new();
            let module = parse_test_module(
                &mut ctx,
                &format!(
                    r#"core.module @test {{
  func.func @test(%value: {ty}) -> core.i32 {{
    %0 = tribute_rtti.descriptor %value : core.i32
    func.return %0
  }}
}}"#
                ),
            );
            assert!(validate_op_schemas(&ctx, module.op()).is_ok(), "{ty}");
        }
    }
}
