mod d {
    #[trunk_ir::dialect]
    mod d {
        struct Plain;

        #[asm_format(symbol(sym_name) signature(r#type, body) region(body))]
        fn probe<S: Plain>(sym_name: Attr<String>, r#type: Attr<TypeOf<S>>) {
            #[region(body)]
            {}
        }
    }
}

fn main() {}
