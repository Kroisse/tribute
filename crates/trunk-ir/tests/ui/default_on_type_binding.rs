mod d {
    #[trunk_ir::dialect]
    mod d {
        fn probe<S>(#[default(0)] signature: Attr<TypeOf<S>>) {}
    }
}

fn main() {}
