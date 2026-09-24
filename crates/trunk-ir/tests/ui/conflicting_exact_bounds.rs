mod d {
    use trunk_ir::dialect::{clif, func};

    #[trunk_ir::dialect]
    mod d {
        fn probe<S: func::FuncSig + clif::FuncSig>(callee: Value<S>) {}
    }
}

fn main() {}
