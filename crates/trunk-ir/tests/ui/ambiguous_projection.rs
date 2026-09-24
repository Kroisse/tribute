mod d {
    use trunk_ir::dialect::func::FuncSig;

    #[trunk_ir::dialect]
    mod d {
        fn probe<S: FuncSig + trunk_ir::dialect::func::FuncSig>(args: Values<S::Inputs>) {}
    }
}

fn main() {}
