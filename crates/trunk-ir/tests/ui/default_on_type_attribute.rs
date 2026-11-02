mod d {
    #[trunk_ir::dialect]
    mod d {
        struct Slot {
            #[default(false)]
            pinned: Attr<bool>,
        }
    }
}

fn main() {}
