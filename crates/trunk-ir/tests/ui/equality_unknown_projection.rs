mod d {
    use trunk_ir::dialect::func::FuncSig;

    #[trunk_ir::dialect]
    mod d {
        fn probe<S: FuncSig<Arguments = (_,)>>(sig: Value<S>) {}
    }
}

fn main() {}
