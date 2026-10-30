mod d {
    #[trunk_ir::dialect]
    mod d {
        #[asm_format("[" successors(targets) "]")]
        fn probe() {
            #[successors(targets)]
            {}
        }
    }
}

fn main() {}
