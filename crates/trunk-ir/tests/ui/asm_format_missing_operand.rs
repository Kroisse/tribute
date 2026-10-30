mod d {
    #[trunk_ir::dialect]
    mod d {
        #[asm_format("probe" lhs attr_dict)]
        fn probe(lhs: Value<_>, rhs: Value<_>) {}
    }
}

fn main() {}
