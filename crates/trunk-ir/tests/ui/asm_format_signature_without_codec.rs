mod d {
    use trunk_ir::dialect::wasm::FuncSig;

    #[trunk_ir::dialect]
    mod d {
        #[asm_format(symbol(sym_name) signature(r#type, body) region(body))]
        fn probe<S: FuncSig>(sym_name: Attr<String>, r#type: Attr<TypeOf<S>>) {
            #[region(body)]
            {}
        }
    }
}

fn main() {}
