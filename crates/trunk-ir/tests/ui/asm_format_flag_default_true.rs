mod d {
    #[trunk_ir::dialect]
    mod d {
        #[asm_format("probe" ("quick" quick^)?)]
        fn probe(#[default(!false)] quick: Attr<bool>) {}
    }
}

fn main() {}
