use trunk_ir::ops::DialectOp;

#[trunk_ir::op_interface]
pub trait Probed: DialectOp {}

#[trunk_ir::dialect]
mod d {
    #[implements(Probed)]
    fn probe() {}
}

fn main() {}
