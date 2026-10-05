//! Decode a quoted Rust string literal at the start of source text.

pub(crate) fn value(text: &str) -> Option<String> {
    let rest = text.strip_prefix('"')?;
    let mut escaped = false;
    for (index, ch) in rest.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '"' => {
                let literal = &text[..index + 2];
                return syn::parse_str::<syn::LitStr>(literal)
                    .ok()
                    .map(|literal| literal.value());
            }
            _ => {}
        }
    }
    None
}
