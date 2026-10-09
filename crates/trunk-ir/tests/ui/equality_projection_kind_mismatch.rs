mod d {
    use trunk_ir::dialect::core::Ref;
    use trunk_ir::dialect::func::FuncSig;

    #[trunk_ir::dialect]
    mod d {
        // `Inputs` is a type list, but `R::Pointee` is one type.
        fn probe<S: FuncSig<Inputs = R::Pointee>, R: Ref>(sig: Value<S>, r: Value<R>) {}
    }
}

fn main() {}
