mod d {
    #[trunk_ir::dialect]
    mod d {
        #[asm_format("probe" ("note" note)?)]
        fn probe(note: Option<Attr<String>>) {}
    }
}

fn main() {}
