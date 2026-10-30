mod d {
    #[trunk_ir::dialect]
    mod d {
        #[asm_format("ret" values)]
        fn ret(values: Variadic<_>) {}
    }
}

fn main() {}
