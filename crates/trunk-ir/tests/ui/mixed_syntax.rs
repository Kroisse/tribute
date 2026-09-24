mod d {
    #[trunk_ir::dialect]
    mod d {
        #[attr(predicate: Symbol)]
        fn probe(lhs: Value<_>, rhs: Value<_>) -> Value<_> {}
    }
}

fn main() {}
