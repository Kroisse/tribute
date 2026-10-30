mod d {
    #[trunk_ir::dialect]
    mod d {
        #[asm_format("run" args "," tail)]
        fn run(tail: Value<_>, args: Variadic<_>) {}
    }
}

fn main() {}
