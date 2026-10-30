mod d {
    #[trunk_ir::dialect]
    mod d {
        #[asm_format("probe" attr_dict region(body))]
        fn probe() {
            #[region(body)]
            {}
        }
    }
}

fn main() {}
