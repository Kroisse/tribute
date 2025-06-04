use tribute::TreeSitterParser;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut parser = TreeSitterParser::new()?;
    
    let hello_source = r#"(fn (main)
    (print_line "Hello, world!"))"#;
    
    println!("Parsing hello.trb with tree-sitter:");
    let ast = parser.parse(hello_source)?;
    println!("{:#?}", ast);
    
    let calc_source = "(+ 1 2)";
    println!("\nParsing simple calc with tree-sitter:");
    let ast2 = parser.parse(calc_source)?;
    println!("{:#?}", ast2);
    
    Ok(())
}