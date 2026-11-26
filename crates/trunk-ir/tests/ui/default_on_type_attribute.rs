mod d {
    #[trunk_ir::dialect]
    mod d {
        struct Slot {
            #[default(false)]
            pinned: Option<Attr<bool>>,
        }
    }
}

fn main() {}
