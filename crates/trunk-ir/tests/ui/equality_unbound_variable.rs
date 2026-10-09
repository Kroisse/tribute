mod d {
    use trunk_ir::dialect::func::FuncSig;

    #[trunk_ir::dialect]
    mod d {
        // Nothing binds `S`, so its `Inputs` can never be checked.
        fn probe<S: FuncSig<Inputs = (T,)>, T>(value: Value<T>) {}
    }
}

fn main() {}
