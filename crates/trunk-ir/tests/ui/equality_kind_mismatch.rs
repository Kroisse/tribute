mod d {
    use trunk_ir::dialect::func::FuncSig;

    #[trunk_ir::dialect]
    mod d {
        // `Inputs` is a type list, but `T` is one type.
        fn probe<S: FuncSig<Inputs = T>, T>(sig: Value<S>) {}
    }
}

fn main() {}
