#[trunk_ir::dialect]
mod d {
    #[identified]
    struct Layout {
        name: Option<Attr<String>>,
    }
}

fn main() {}
