//! Arena-based func dialect.

// === Operation registrations ===
crate::register_pure_op!(func.constant);
crate::register_isolated_op!(func.func);

#[crate::dialect(crate = crate)]
mod func {
    #[attr(sym_name: Symbol, r#type: Type)]
    fn func() {
        #[region(body)]
        {}
    }

    #[attr(callee: Symbol)]
    fn call(#[rest] args: ()) -> result {}

    fn call_indirect(callee: (), #[rest] args: ()) -> result {}

    #[attr(callee: Symbol)]
    fn tail_call(#[rest] args: ()) {}

    fn r#return(#[rest] values: ()) {}

    #[attr(func_ref: Symbol)]
    fn constant() -> result {}

    fn unreachable() {}
}

// === Custom assembly format for func.func ===

/// Print func.func with decomposed signature:
/// `func.func @name(%arg: type, ...) -> return_type effects eff_type { body }`
fn print_func(
    h: &mut crate::printer::OpPrintHelper<'_, '_>,
    op: crate::OpRef,
    indent: usize,
) -> std::fmt::Result {
    use std::fmt::Write;

    let indent_str = " ".repeat(indent);

    // Extract sym_name before mutable operations
    let sym_name = {
        let data = h.ctx().op(op);
        data.attributes
            .get(&crate::Symbol::new("sym_name"))
            .and_then(|a| match a {
                crate::Attribute::Symbol(s) => Some(*s),
                _ => None,
            })
    };

    write!(h, "{indent_str}func.func")?;

    // Function name
    if let Some(name) = sym_name {
        write!(h, " @")?;
        name.with_str(|s| {
            let needs_quoting =
                s.is_empty() || !s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
            if needs_quoting {
                write!(h, "\"")?;
                crate::printer::write_escaped_string(h, s)?;
                write!(h, "\"")
            } else {
                write!(h, "{s}")
            }
        })?;
    }

    // Reset numbering for function body
    h.reset_numbering();

    // Extract region ref and func type info before mutable operations
    let region = {
        let data = h.ctx().op(op);
        assert!(
            data.regions.len() <= 1,
            "print_func: expected at most one region, found {}",
            data.regions.len(),
        );
        data.regions.first().copied()
    };

    // Extract type decomposition: return type from core.func type attribute
    let type_info = {
        let data = h.ctx().op(op);
        if let Some(crate::Attribute::Type(func_ty)) =
            data.attributes.get(&crate::Symbol::new("type"))
        {
            let ty_data = h.ctx().types.get(*func_ty);
            let is_core_func = ty_data.dialect == crate::Symbol::new("core")
                && ty_data.name == crate::Symbol::new("func");
            if is_core_func && !ty_data.params.is_empty() {
                let result_ty = ty_data.params[0];
                let param_tys: Vec<crate::TypeRef> = ty_data.params[1..].to_vec();
                Some((result_ty, param_tys))
            } else {
                // Non-standard or empty core.func type — skip signature
                None
            }
        } else {
            None
        }
    };

    if let Some(region) = region {
        // Print entry block args as function signature
        let entry_args: Vec<_> = {
            let blocks = &h.ctx().region(region).blocks;
            blocks
                .first()
                .map(|&b| h.ctx().block_args(b).to_vec())
                .unwrap_or_default()
        };

        write!(h, "(")?;
        for (i, &arg) in entry_args.iter().enumerate() {
            if i > 0 {
                write!(h, ", ")?;
            }
            let name = h.assign_value_name(arg);
            let ty = h.ctx().value_ty(arg);
            write!(h, "{name}: ")?;
            h.write_type(ty)?;
        }
        write!(h, ")")?;
    } else if let Some((_, ref param_tys)) = type_info {
        // Body-less declaration: synthesize params from type
        write!(h, "(")?;
        for (i, &ty) in param_tys.iter().enumerate() {
            if i > 0 {
                write!(h, ", ")?;
            }
            write!(h, "%arg{i}: ")?;
            h.write_type(ty)?;
        }
        write!(h, ")")?;
    } else {
        write!(h, "()")?;
    }

    if let Some((result_ty, _)) = type_info {
        write!(h, " -> ")?;
        h.write_type(result_ty)?;
    }

    // Extra attributes (everything except sym_name and type, which are
    // already encoded in the signature).  Clone to avoid borrow conflicts
    // with the mutable write helpers.
    let extra_attrs: Vec<_> = {
        let data = h.ctx().op(op);
        data.attributes
            .iter()
            .filter(|(k, _)| {
                **k != crate::Symbol::new("sym_name") && **k != crate::Symbol::new("type")
            })
            .map(|(k, v)| (*k, v.clone()))
            .collect()
    };
    if !extra_attrs.is_empty() {
        write!(h, " attributes {{")?;
        for (i, (key, val)) in extra_attrs.iter().enumerate() {
            if i > 0 {
                write!(h, ", ")?;
            }
            write!(h, "{key} = ")?;
            h.write_attribute(val)?;
        }
        write!(h, "}}")?;
    }

    if let Some(region) = region {
        // Body
        writeln!(h, " {{")?;
        h.print_region_eliding_entry(region, indent + 2)?;
        writeln!(h, "{indent_str}}}")?;
    } else {
        writeln!(h)?;
    }

    Ok(())
}

/// Parse func.func custom format back into a RawOperation.
fn parse_func<'a>(
    input: &mut &'a str,
    results: Vec<&'a str>,
    sym_name: Option<String>,
) -> winnow::ModalResult<crate::parser::raw::RawOperation<'a>> {
    use crate::parser::raw::*;
    use winnow::combinator::opt;
    use winnow::prelude::*;

    // "(%arg: type, ...)" or "()"
    ws.parse_next(input)?;
    let params = if input.starts_with('(') {
        func_params.parse_next(input)?
    } else {
        vec![]
    };

    // "-> return_type" (optional)
    let ret_ty = opt(return_type).parse_next(input)?;

    // "attributes { key = value, ... }" (optional extra attributes)
    let attributes = opt((ws, "attributes", ws, raw_attr_dict))
        .parse_next(input)?
        .map(|(_, _, _, attrs)| attrs)
        .unwrap_or_default();

    // "{ body }"
    ws.parse_next(input)?;
    let mut regions = Vec::new();
    if input.starts_with('{') {
        let region = raw_region.parse_next(input)?;
        regions.push(region);
    }

    Ok(RawOperation {
        results,
        dialect: "func",
        op_name: "func",
        sym_name,
        func_params: params,
        return_type: ret_ty,
        operands: vec![],
        attributes,
        result_types: vec![],
        regions,
        successors: vec![],
    })
}

inventory::submit! {
    crate::op_interface::OpAsmFormat {
        dialect: "func",
        op_name: "func",
        print_fn: print_func,
        parse_fn: parse_func,
    }
}
