mod d {
    use trunk_ir::dialect::func::FuncSig;

    #[trunk_ir::dialect]
    mod d {
        fn probe<S: FuncSig>(args: Values<S::Arguments>) {}
    }
}

fn main() {}
