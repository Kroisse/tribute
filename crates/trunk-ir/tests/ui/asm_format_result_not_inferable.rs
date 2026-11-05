mod d {
    use trunk_ir::dialect::core::IntegerLike;

    #[trunk_ir::dialect]
    mod d {
        #[asm_format(value ":" type(value))]
        fn widen(value: Value<impl IntegerLike>) -> Value<impl IntegerLike> {}
    }
}

fn main() {}
