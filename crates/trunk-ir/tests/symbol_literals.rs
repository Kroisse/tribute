//! Public symbol construction from another crate's context.

use trunk_ir::{Symbol, symbol};

const FUNC: Symbol = symbol!("func");
const LONG_NAME: Symbol = symbol!("sym_name");

#[test]
fn literal_symbols_match_runtime_interning_in_const_contexts() {
    let text = String::from("sym_name");
    let runtime = Symbol::new(text.as_str());
    assert_eq!(runtime, LONG_NAME);
    assert!(LONG_NAME.is_static_or_inline());
    assert_eq!(Symbol::from(text.as_str()), runtime);

    match Symbol::new(String::from("func").as_str()) {
        name if name == symbol!("func") => assert_eq!(FUNC.as_str(), "func"),
        other => panic!("unexpected symbol: {other:?}"),
    }
}
