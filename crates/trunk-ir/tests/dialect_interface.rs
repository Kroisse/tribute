//! A dialect interface declared outside trunk-ir, implemented by a dialect
//! of this crate and by a dialect of trunk-ir, both registered with
//! `register_interfaces!`.

use trunk_ir::IrContext;
use trunk_ir::dialect::core;
use trunk_ir::ops::Dialect;

mod probe {
    #[trunk_ir::dialect]
    mod probe {
        fn nop() {}
    }
}

/// Dialects that report a version.
#[trunk_ir::dialect_interface]
pub trait Versioned: Dialect {
    fn version(&self, ctx: &IrContext) -> u32;

    fn is_stable(&self, ctx: &IrContext) -> bool {
        self.version(ctx) >= 1
    }
}

impl Versioned for probe::Dialect {
    fn version(&self, _ctx: &IrContext) -> u32 {
        0
    }
}

impl Versioned for core::Dialect {
    fn version(&self, _ctx: &IrContext) -> u32 {
        2
    }
}

trunk_ir::register_interfaces!(probe::Dialect: Versioned; core::Dialect: Versioned);

#[test]
fn queries_dispatch_to_each_dialect_by_name() {
    let ctx = IrContext::new();
    assert_eq!(<probe::Dialect as Dialect>::NAME, probe::DIALECT_NAME);

    let probe = DynVersioned::get("probe").expect("probe implements Versioned");
    assert_eq!(probe.dialect(), "probe");
    assert_eq!(probe.version(&ctx), 0);
    assert!(!probe.is_stable(&ctx));

    let core = DynVersioned::get(core::DIALECT_NAME).expect("core implements Versioned");
    assert!(core.is_stable(&ctx));

    assert!(DynVersioned::get("arith").is_none());
    assert!(!DynVersioned::is_implemented("arith"));
    assert!(DynVersioned::is_implemented("probe"));

    let mut implementors: Vec<_> = DynVersioned::implementors().collect();
    implementors.sort();
    assert_eq!(implementors, ["core", "probe"]);
}
