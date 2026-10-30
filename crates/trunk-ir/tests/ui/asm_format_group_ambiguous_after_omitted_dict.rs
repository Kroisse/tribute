mod d {
    #[trunk_ir::dialect]
    mod d {
        #[asm_format("run" ("flag" left^)? attr_dict ("flag" right^)?)]
        fn run(left: Option<Attr<bool>>, right: Option<Attr<bool>>) {}
    }
}

fn main() {}
