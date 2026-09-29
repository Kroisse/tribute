//! Arena-based mem dialect.

// === Pure operation registrations ===
crate::register_pure_op!(Data);
crate::register_pure_op!(PtrAdd);
// mem.load is intentionally NOT pure: loads depend on mutable memory and may trap.

use crate::dialect::core::{IntegerLike, Ptr, ScalarLike};

#[trunk_ir::dialect]
mod mem {
    /// The address of immutable data holding `bytes`.
    fn data(bytes: Attr<Bytes>) -> Value<Ptr> {}

    /// Load a scalar from `ptr` plus an immediate byte `offset`.
    fn load(offset: Attr<u32>, ptr: Value<Ptr>) -> Value<impl ScalarLike> {}

    /// Store a scalar `value` at `ptr` plus an immediate byte `offset`.
    fn store(offset: Attr<u32>, ptr: Value<Ptr>, value: Value<impl ScalarLike>) {}

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
                "core.module @test {{\n  func.func @f(%p: core.ptr, %i: core.i64, %n: core.i32, %f: core.f64, %b: core.bytes) -> core.nil {{\n    {body}\n    func.return\n  }}\n}}"
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

    #[test]
    fn memory_access_addresses_must_be_pointers() {
        assert!(schema_ok("%a = mem.load %p {offset = 8} : core.i64"));
        assert!(!schema_ok("%a = mem.load %i {offset = 0} : core.i64"));
        assert!(schema_ok("mem.store %p, %f {offset = 0}"));
        assert!(!schema_ok("mem.store %n, %f {offset = 0}"));
        assert!(schema_ok("%a = mem.data {bytes = b\"hi\"} : core.ptr"));
        assert!(!schema_ok("%a = mem.data {bytes = b\"hi\"} : core.i64"));
    }

    #[test]
    fn memory_access_moves_only_scalars() {
        for scalar in ["core.i8", "core.i64", "core.f64", "core.ptr"] {
            assert!(schema_ok(&format!(
                "%a = mem.load %p {{offset = 0}} : {scalar}"
            )));
        }
        assert!(!schema_ok("%a = mem.load %p {offset = 0} : core.bytes"));
        assert!(!schema_ok("%a = mem.load %p {offset = 0} : core.nil"));
        assert!(schema_ok("mem.store %p, %p {offset = 0}"));
        assert!(!schema_ok("mem.store %p, %b {offset = 0}"));
    }
}
