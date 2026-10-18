#[trunk_ir::op_interface]
pub trait Generic: trunk_ir::ops::DialectOp {
    fn convert<T>(self, ctx: &trunk_ir::IrContext) -> T;
}

#[trunk_ir::op_interface]
pub trait NoContext: trunk_ir::ops::DialectOp {
    fn arity(self) -> usize;
}

fn main() {}
