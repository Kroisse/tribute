#[trunk_ir::dialect]
mod d {
    #[alias_hint]
    struct Named {
        name: Attr<String>,
    }
}

fn main() {}
