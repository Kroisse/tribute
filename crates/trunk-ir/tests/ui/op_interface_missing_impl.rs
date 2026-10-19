use trunk_ir::ops::DialectOp;

mod d {
    #[trunk_ir::dialect]
    mod d {
        fn probe() {}
    }
}

#[trunk_ir::op_interface]
pub trait Probed: DialectOp {}

trunk_ir::__private::inventory::submit! { <d::Probe as Probed>::__TRUNK_IR_INTERFACE }

fn main() {}
