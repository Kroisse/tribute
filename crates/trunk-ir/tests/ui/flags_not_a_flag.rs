pub struct NotAFlag;

#[trunk_ir::dialect]
mod d {
    #[flags(NotAFlag)]
    fn probe() {}
}

fn main() {}
