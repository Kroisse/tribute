//! Arena-based mem dialect.

// === Pure operation registrations ===
crate::register_pure_op!(Data);
crate::register_pure_op!(PtrAdd);
// mem.load is intentionally NOT pure: loads depend on mutable memory and may trap.

use crate::dialect::core::{IntegerLike, Ptr};

#[trunk_ir::dialect]
mod mem {
    fn data(bytes: Attr<_>) -> Value<_> {}

    fn load(offset: Attr<u32>, ptr: Value<_>) -> Value<_> {}

    fn store(offset: Attr<u32>, ptr: Value<_>, value: Value<_>) {}

    /// Add a pointer-width integer byte `offset` to `base`, keeping `base`'s
    /// provenance. No element-size scaling is applied.
    fn ptr_add<T: IntegerLike>(base: Value<Ptr>, offset: Value<T>) -> Value<Ptr> {}
}

#[cfg(test)]
mod tests {
    use crate::context::IrContext;
    use crate::parser::parse_test_module;
    use crate::validation::validate_op_schemas;

    fn schema_ok(body: &str) -> bool {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            &format!(
                "core.module @test {{\n  func.func @f(%p: core.ptr, %i: core.i64, %n: core.i32, %f: core.f64) -> core.nil {{\n    {body}\n    func.return\n  }}\n}}"
            ),
        );
        validate_op_schemas(&ctx, module.op()).is_ok()
    }

    #[test]
    fn ptr_add_takes_a_pointer_and_an_integer_and_yields_a_pointer() {
        assert!(schema_ok("%a = mem.ptr_add %p, %i : core.ptr"));
        assert!(schema_ok("%a = mem.ptr_add %p, %n : core.ptr"));
        assert!(!schema_ok("%a = mem.ptr_add %i, %i : core.ptr"));
        assert!(!schema_ok("%a = mem.ptr_add %p, %f : core.ptr"));
        assert!(!schema_ok("%a = mem.ptr_add %p, %i : core.i64"));
    }
}
