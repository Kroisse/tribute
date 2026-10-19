//! A type interface declared outside trunk-ir, implemented by a type of this
//! crate, registered with `#[implements(..)]`, and by a type of trunk-ir,
//! registered with `register_interfaces!`.

use trunk_ir::dialect::core;
use trunk_ir::ops::DialectType;
use trunk_ir::{IrContext, TypeDataBuilder};

mod probe {
    #[trunk_ir::dialect]
    mod probe {
        #[implements(super::Sized)]
        struct Block<Elem> {
            count: Attr<u32>,
        }

        struct Opaque;
    }
}

/// Types with a statically known size.
#[trunk_ir::type_interface]
pub trait Sized: DialectType {
    /// The size in bytes, given the size of one element.
    fn size(self, ctx: &IrContext, elem_size: u32) -> u32;

    fn is_zero_sized(&self, ctx: &IrContext) -> bool {
        self.size(ctx, 1) == 0
    }
}

impl Sized for probe::Block {
    fn size(self, ctx: &IrContext, elem_size: u32) -> u32 {
        self.count(ctx) * elem_size
    }
}

impl Sized for core::Nil {
    fn size(self, _ctx: &IrContext, _elem_size: u32) -> u32 {
        0
    }
}

trunk_ir::register_interfaces!(core::Nil: Sized);

#[test]
fn queries_dispatch_to_each_type() {
    let mut ctx = IrContext::new();
    let nil = core::Nil::params().intern(&mut ctx).as_type_ref();
    let block = probe::Block::params(nil)
        .count(3)
        .intern(&mut ctx)
        .as_type_ref();
    let opaque = TypeDataBuilder::new(probe::Opaque::NAME).intern(&mut ctx);

    let block_interface = DynSized::get(&ctx, block).expect("probe.block implements Sized");
    assert_eq!(block_interface.ty(), block);
    assert_eq!(block_interface.size(&ctx, 4), 12);
    assert!(!block_interface.is_zero_sized(&ctx));

    let nil_interface = DynSized::get(&ctx, nil).expect("core.nil implements Sized");
    assert!(nil_interface.is_zero_sized(&ctx));

    assert!(DynSized::get(&ctx, opaque).is_none());
    assert!(!DynSized::is_implemented(&ctx, opaque));
    assert!(DynSized::is_implemented(&ctx, block));

    let mut implementors: Vec<_> = DynSized::implementors()
        .map(|name| name.display(&ctx).to_string())
        .collect();
    implementors.sort();
    assert_eq!(implementors, ["core.nil", "probe.block"]);
}
