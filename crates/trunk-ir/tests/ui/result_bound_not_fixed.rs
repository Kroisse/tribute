mod d {
    use trunk_ir::dialect::core::IntegerLike;

    #[trunk_ir::dialect]
    mod d {
        fn probe(value: Value<_>) -> Value<IntegerLike> {}

        // Checked even though the builder takes these results explicitly.
        fn probe_list(value: Value<_>) -> Values<(IntegerLike, _)> {}
    }
}

fn main() {}
