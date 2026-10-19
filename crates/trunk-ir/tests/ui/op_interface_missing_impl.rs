use trunk_ir::ops::DialectOp;

mod d {
    #[trunk_ir::dialect]
    mod d {
        fn probe() {}
    }
}

#[trunk_ir::op_interface]
pub trait Probed: DialectOp {}

trunk_ir::register_interfaces!(d::Probe: Probed);

fn main() {}
