mod d {
    use trunk_ir::dialect::func::FuncSig;

    #[trunk_ir::dialect]
    mod d {
        fn probe<S: FuncSig>(arg: Value<S::Inputs>) {}
    }
}

fn main() {}
