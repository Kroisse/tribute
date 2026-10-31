mod d {
    #[trunk_ir::dialect]
    mod d {
        #[asm_format("probe" lhs "," lhs)]
        fn probe(lhs: Value<_>) {}
    }
}

fn main() {}
