use std::path::Path;
use tribute::parse;
use insta::assert_ron_snapshot;

pub fn parse_file(path: &Path) -> Vec<(tribute::ast::Expr, tribute::ast::SimpleSpan)> {
    let source = std::fs::read_to_string(path)
        .expect("Failed to read file");
    
    parse(&source)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_hello_example() {
        let path = Path::new("lang-examples/hello.trb");
        let ast = parse_file(path);
        
        assert_ron_snapshot!(ast);
    }
    
    #[test]
    fn test_parse_calc_example() {
        let path = Path::new("lang-examples/calc.trb");
        let ast = parse_file(path);
        
        assert_ron_snapshot!(ast);
    }
}