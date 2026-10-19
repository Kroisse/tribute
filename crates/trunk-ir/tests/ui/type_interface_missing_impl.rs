use trunk_ir::ops::DialectType;

#[trunk_ir::type_interface]
pub trait Probed: DialectType {}

#[trunk_ir::dialect]
mod d {
    #[implements(Probed)]
    struct Probe;
}

fn main() {}
