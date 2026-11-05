mod d {
    use trunk_ir::dialect::core::IntegerLike;

    #[trunk_ir::dialect]
    mod d {
        #[asm_format(value ":" type(value))]
        fn same<T: IntegerLike>(value: Value<T>) -> Value<T> {}
    }
}

fn main() {}
