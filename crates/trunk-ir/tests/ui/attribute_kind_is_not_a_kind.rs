struct NotAKind;

#[trunk_ir::dialect]
mod test {
    fn bad(value: Attr<NotAKind>) {}
}

fn main() {}
