//! Path keywords: `pkg`, `self`, and `super`.

use derive_more::Display;
use trunk_ir::Symbol;

use crate::keywords::PATH_KEYWORDS;

/// Why a path's keywords do not denote a module.
#[derive(Clone, Debug, Display, PartialEq, Eq)]
pub(crate) enum PathKeywordError {
    #[display("`super` at the package root has no parent module")]
    SuperAtRoot,
    #[display("path keyword `{_0}` may only start a path")]
    Misplaced(Symbol),
}

fn keyword(segment: Symbol) -> Option<&'static str> {
    segment.with_str(|name| {
        PATH_KEYWORDS
            .iter()
            .copied()
            .find(|keyword| *keyword == name)
    })
}

/// The path `path` names when it starts with a path keyword, read from the
/// module at `module_path`; `None` for a path without one.
///
/// The first `package_depth` segments of `module_path` are the package root.
/// `pkg` names the package root, `self` the current module, and `super` its
/// parent, which the package root does not have. A keyword may only start a
/// path.
pub(crate) fn absolute_path(
    package_depth: usize,
    module_path: &[Symbol],
    path: &[Symbol],
) -> Result<Option<Vec<Symbol>>, PathKeywordError> {
    let Some((first, rest)) = path.split_first() else {
        return Ok(None);
    };
    let first = first.clone();
    if let Some(misplaced) = rest
        .iter()
        .find(|segment| keyword((*segment).clone()).is_some())
    {
        return Err(PathKeywordError::Misplaced(misplaced.clone()));
    }
    let mut base = match keyword(first) {
        None => return Ok(None),
        Some("pkg") => module_path[..package_depth].to_vec(),
        Some("self") => module_path.to_vec(),
        Some(_) => {
            if module_path.len() <= package_depth {
                return Err(PathKeywordError::SuperAtRoot);
            }
            module_path[..module_path.len() - 1].to_vec()
        }
    };
    base.extend_from_slice(rest);
    Ok(Some(base))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(text: &str) -> Vec<Symbol> {
        text.split("::").map(Symbol::from_dynamic).collect()
    }

    #[test]
    fn keywords_expand_from_the_current_module() {
        let module = path("a::b");
        assert_eq!(
            absolute_path(0, &module, &path("pkg::x::y")),
            Ok(Some(path("x::y")))
        );
        assert_eq!(
            absolute_path(0, &module, &path("self::x")),
            Ok(Some(path("a::b::x")))
        );
        assert_eq!(
            absolute_path(0, &module, &path("super::x")),
            Ok(Some(path("a::x")))
        );
        assert_eq!(absolute_path(0, &module, &path("x::y")), Ok(None));
    }

    #[test]
    fn keywords_stop_at_the_package_root() {
        let module = path("std::io");
        assert_eq!(
            absolute_path(1, &module, &path("pkg::x")),
            Ok(Some(path("std::x")))
        );
        assert_eq!(
            absolute_path(1, &module, &path("super::x")),
            Ok(Some(path("std::x")))
        );
        assert_eq!(
            absolute_path(1, &path("std"), &path("super::x")),
            Err(PathKeywordError::SuperAtRoot)
        );
    }

    #[test]
    fn invalid_keywords_are_errors() {
        assert_eq!(
            absolute_path(0, &[], &path("super::x")),
            Err(PathKeywordError::SuperAtRoot)
        );
        assert_eq!(
            absolute_path(0, &path("a::b"), &path("super::super::x")),
            Err(PathKeywordError::Misplaced(Symbol::new("super")))
        );
        assert_eq!(
            absolute_path(0, &[], &path("x::super::y")),
            Err(PathKeywordError::Misplaced(Symbol::new("super")))
        );
        assert_eq!(
            absolute_path(0, &[], &path("pkg::super::y")),
            Err(PathKeywordError::Misplaced(Symbol::new("super")))
        );
        assert_eq!(
            absolute_path(0, &[], &path("pkg::self")),
            Err(PathKeywordError::Misplaced(Symbol::new("self")))
        );
    }
}
