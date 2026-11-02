mod d {
    #[trunk_ir::dialect]
    mod d {
        #[asm_format("probe" ":" types(results))]
        fn probe() -> Variadic<_> {}
    }
}

fn main() {}
