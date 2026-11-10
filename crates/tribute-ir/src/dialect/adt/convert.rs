//! Conversion of nominal layouts, whose fields are in identified bodies
//! (`design/tribute-dialects.md` § "Nominal layout 변환").

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use trunk_ir::context::IrContext;
use trunk_ir::refs::TypeRef;

use super::{STRUCT_NAME_ATTR, bind_layout_body};

/// A type conversion that rebuilds a type from its converted parts without
/// looking into the bodies of identified types: it converts a nominal
/// layout's key by its attributes alone. [`convert_layouts`] extends it to
/// the layouts' fields.
pub trait LayoutBlindConversion {
    type Error;

    fn ctx(&self) -> &IrContext;

    fn ctx_mut(&mut self) -> &mut IrContext;

    /// Convert `ty`.
    fn convert(&mut self, ty: TypeRef) -> Result<TypeRef, Self::Error>;

    /// Make [`convert`](Self::convert) return `to` for `from` from now on.
    fn fix(&mut self, from: TypeRef, to: TypeRef);

    /// Forget every conversion made or fixed so far.
    fn reset(&mut self);
}

/// Convert the nominal layouts `roots` reach, and fix the conversion of
/// each one, so that `conversion` converts every type that mentions a layout
/// consistently afterwards.
///
/// A layout changes when its key attributes or its body's fields convert to
/// something else, or when its key or body holds the key or a row instance
/// of a layout that changes. A changed layout gets the key of its converted attributes, with
/// `$tag` appended to its name when those attributes are unchanged, and its
/// body converted after every new key is known. A layout the conversion
/// replaces with a type other than a nominal layout keeps that replacement.
/// Returns the new key of each changed layout.
pub fn convert_layouts<C: LayoutBlindConversion>(
    conversion: &mut C,
    roots: impl IntoIterator<Item = TypeRef>,
    tag: &str,
) -> Result<HashMap<TypeRef, TypeRef>, C::Error> {
    let reached = trunk_ir::walk::reachable_types_from(conversion.ctx(), roots);
    let ctx = conversion.ctx();
    let layouts: Vec<TypeRef> = reached
        .into_iter()
        .filter(|&ty| is_nominal_layout(ctx, ty) && ctx.type_body(ty).is_some())
        .collect();

    // Which layouts change: converting each one blind, then following the
    // layouts whose keys or bodies hold them.
    conversion.reset();
    let mut changed = HashSet::default();
    let mut dependents: HashMap<TypeRef, Vec<TypeRef>> = HashMap::default();
    for &layout in &layouts {
        let body = conversion
            .ctx()
            .type_body(layout)
            .expect("a reached layout has a body");
        if conversion.convert(layout)? != layout || conversion.convert(body)? != body {
            changed.insert(layout);
        }
        let ctx = conversion.ctx();
        let mut held = Vec::new();
        for attr in ctx.get_type(layout).attrs.values() {
            attr.visit_types(&mut |nested| collect_layouts(ctx, nested, &mut held));
        }
        trunk_ir::walk::visit_type_children(ctx, body, &mut |child| {
            collect_layouts(ctx, child, &mut held)
        });
        for dependency in held {
            dependents.entry(dependency).or_default().push(layout);
        }
    }
    let mut pending: Vec<_> = changed.iter().copied().collect();
    while let Some(layout) = pending.pop() {
        for &dependent in dependents.get(&layout).into_iter().flatten() {
            if changed.insert(dependent) {
                pending.push(dependent);
            }
        }
    }

    // The new keys, then the bodies. A blind conversion stays right unless
    // it holds a changed layout as a type, which only a layout holding one
    // does; otherwise the conversions made so far are kept.
    if dependents.keys().any(|layout| changed.contains(layout)) {
        conversion.reset();
        for &layout in &layouts {
            if !changed.contains(&layout) {
                conversion.fix(layout, layout);
            }
        }
    }
    let mut settler = Settler {
        changed: &changed,
        tag,
        keys: HashMap::default(),
        settled: HashSet::default(),
    };
    for &layout in &layouts {
        if changed.contains(&layout) {
            settler.settle(conversion, layout)?;
        }
    }
    let keys = settler.keys;
    for (&old, &new) in &keys {
        let body = conversion
            .ctx()
            .type_body(old)
            .expect("a changed layout has a body");
        let body = conversion.convert(body)?;
        bind_layout_body(conversion.ctx_mut(), new, body).unwrap_or_else(|error| {
            panic!(
                "converted layout {:?}: {error}",
                trunk_ir::type_def::identifier(conversion.ctx(), new).unwrap_or_default()
            )
        });
    }
    Ok(keys)
}

/// Fixes the conversion of each changed layout, after the changed layouts
/// its key attributes hold.
struct Settler<'a> {
    changed: &'a HashSet<TypeRef>,
    tag: &'a str,
    keys: HashMap<TypeRef, TypeRef>,
    settled: HashSet<TypeRef>,
}

impl Settler<'_> {
    fn settle<C: LayoutBlindConversion>(
        &mut self,
        conversion: &mut C,
        layout: TypeRef,
    ) -> Result<(), C::Error> {
        if !self.settled.insert(layout) {
            return Ok(());
        }
        let ctx = conversion.ctx();
        let mut held = Vec::new();
        for attr in ctx.get_type(layout).attrs.values() {
            attr.visit_types(&mut |nested| collect_layouts(ctx, nested, &mut held));
        }
        for nested in held {
            assert_ne!(
                nested, layout,
                "a nominal layout's key attributes hold the layout itself"
            );
            if self.changed.contains(&nested) {
                self.settle(conversion, nested)?;
            }
        }
        let converted = conversion.convert(layout)?;
        let ctx = conversion.ctx_mut();
        let same_kind = ctx.get_type(converted).name == ctx.get_type(layout).name
            && trunk_ir::type_def::is_identified(ctx, converted);
        if !same_kind {
            // The conversion replaces the layout with another type.
            return Ok(());
        }
        let new = if ctx.get_type(converted) == ctx.get_type(layout) {
            let name = trunk_ir::type_def::identifier(ctx, layout).expect("a layout is named");
            let tagged = ctx.string_attr(&format!("{name}${}", self.tag));
            let mut data = ctx.get_type(layout).clone();
            data.attrs.view_mut(ctx).insert(STRUCT_NAME_ATTR, tagged);
            ctx.intern_type(data)
        } else {
            converted
        };
        conversion.fix(layout, new);
        self.keys.insert(layout, new);
        Ok(())
    }
}

/// Whether `ty` is a nominal layout's key, which a body is bound to.
fn is_nominal_layout(ctx: &IrContext, ty: TypeRef) -> bool {
    super::is_nominal(ctx, ty) && ctx.type_attrs(ty).contains_key(STRUCT_NAME_ATTR)
}

/// Collect the nominal layout keys `ty` is or holds, without entering
/// identified types' bodies.
fn collect_layouts(ctx: &IrContext, ty: TypeRef, found: &mut Vec<TypeRef>) {
    let mut pending = vec![ty];
    let mut seen = HashSet::default();
    while let Some(ty) = pending.pop() {
        if !seen.insert(ty) {
            continue;
        }
        if is_nominal_layout(ctx, ty) {
            found.push(ty);
            continue;
        }
        let data = ctx.get_type(ty);
        pending.extend(data.params.iter().copied());
        for attr in data.attrs.values() {
            attr.visit_types(&mut |nested| pending.push(nested));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::types::TypeDataBuilder;

    /// Replaces `core.i32` with `core.i64`, rebuilding every other type from
    /// its converted parameters and attributes.
    struct Widen<'a> {
        ctx: &'a mut IrContext,
        memo: HashMap<TypeRef, TypeRef>,
    }

    impl LayoutBlindConversion for Widen<'_> {
        type Error = std::convert::Infallible;

        fn ctx(&self) -> &IrContext {
            self.ctx
        }

        fn ctx_mut(&mut self) -> &mut IrContext {
            self.ctx
        }

        fn convert(&mut self, ty: TypeRef) -> Result<TypeRef, Self::Error> {
            if let Some(&converted) = self.memo.get(&ty) {
                return Ok(converted);
            }
            let mut data = self.ctx.get_type(ty).clone();
            let converted = if data.name.is(self.ctx, "core.i32") {
                TypeDataBuilder::new("core.i64").intern(self.ctx)
            } else {
                for param in data.params.iter_mut() {
                    *param = self.convert(*param)?;
                }
                let attrs: Vec<_> = data.attrs.iter().map(|(k, v)| (k, v.clone())).collect();
                for (key, value) in attrs {
                    let value = value.map_types(|nested| self.convert(nested).unwrap());
                    data.attrs.view_mut(self.ctx).insert(key, value);
                }
                self.ctx.intern_type(data)
            };
            self.memo.insert(ty, converted);
            Ok(converted)
        }

        fn fix(&mut self, from: TypeRef, to: TypeRef) {
            self.memo.insert(from, to);
        }

        fn reset(&mut self) {
            self.memo.clear();
        }
    }

    #[test]
    fn changed_layouts_get_new_keys_and_the_layouts_holding_them_follow() {
        let mut ctx = IrContext::new();
        parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !P = adt.struct<"P", (core.i32)>
  !Q = adt.struct<"Q", (core.f64)>
  !H = adt.struct<"H", (!P)>
  !K = adt.struct<"K", (core.f64), {k = core.i32}>
  !E = adt.enum<"E", [(), (core.i32)]>
}"#,
        );
        let alias = |ctx: &IrContext, name| ctx.type_alias_by_text(name).unwrap();
        let [p, q, h, k, e] = ["P", "Q", "H", "K", "E"].map(|name| alias(&ctx, name));
        let roots = [p, q, h, k, e];
        let mut widen = Widen {
            ctx: &mut ctx,
            memo: HashMap::default(),
        };
        let keys = convert_layouts(&mut widen, roots, "w").unwrap();
        let printed = |ctx: &IrContext, ty: TypeRef| {
            let name = trunk_ir::type_def::identifier(ctx, ty).unwrap().to_owned();
            let body = ctx.type_body(ty).unwrap();
            (name, body.display(ctx).to_string())
        };

        // A field that converts gives the layout a tagged key.
        assert_eq!(
            printed(widen.ctx, keys[&p]),
            ("P$w".to_owned(), "adt.struct<(core.i64)>".to_owned())
        );
        assert_eq!(
            printed(widen.ctx, keys[&e]),
            ("E$w".to_owned(), "adt.enum<[(), (core.i64)]>".to_owned())
        );
        // A layout holding a changed one's key follows it; an unrelated one
        // keeps its key.
        assert_eq!(
            widen
                .ctx
                .type_body(keys[&h])
                .map(|body| { widen.ctx.get_type(body).params[0] }),
            Some(keys[&p])
        );
        assert!(!keys.contains_key(&q));
        assert_eq!(widen.convert(q).unwrap(), q);
        // Converted key attributes make the new key without a tag.
        assert_eq!(printed(widen.ctx, keys[&k]).0, "K");
        assert_ne!(keys[&k], k);
    }
}
