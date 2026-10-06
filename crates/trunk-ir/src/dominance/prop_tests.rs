//! Property tests comparing [`DominatorTree`] with the definition of
//! dominance on random control-flow graphs.
//!
//! Each case builds a `func.func` body whose blocks end in real terminators
//! (`func.return`, `cf.br`, `cf.cond_br`, `cf.switch`) with random successor
//! lists, so the graphs include loops, self-loops, repeated edges, and
//! blocks unreachable from the entry. The oracle uses no dataflow: a block is
//! reachable when a search from the entry finds it, and `d` dominates a
//! reachable `n` when `d == n` or `n` is unreachable once `d` is removed.

use proptest::prelude::*;
use smallvec::smallvec;

use super::*;
use crate::dialect::{arith, cf, core, func};
use crate::location::Span;
use crate::{Attribute, BlockData, Location, RegionData};

/// The successor list of each block, by block index; block 0 is the entry.
fn cfg() -> impl Strategy<Value = Vec<Vec<usize>>> {
    (0usize..=12).prop_flat_map(|blocks| {
        // Successor counts favor jumps and two-way branches but include
        // returns and multi-way switches.
        let successors = prop_oneof![
            2 => Just(0usize),
            4 => Just(1usize),
            4 => Just(2usize),
            1 => 3usize..=4,
        ]
        .prop_flat_map(move |count| prop::collection::vec(0..blocks.max(1), count));
        prop::collection::vec(successors, blocks)
    })
}

fn location(ctx: &mut IrContext) -> Location {
    let path = ctx.intern_path("file:///prop.trb");
    Location::new(path, Span::new(0, 0))
}

/// Build a function whose body has one block per entry of `cfg`, and return
/// its body region and blocks.
fn build_function(ctx: &mut IrContext, cfg: &[Vec<usize>]) -> (RegionRef, Vec<BlockRef>) {
    let loc = location(ctx);
    let blocks: Vec<BlockRef> = cfg
        .iter()
        .map(|_| {
            ctx.create_block(BlockData {
                location: loc,
                args: vec![],
                ops: smallvec![],
                parent_region: None,
            })
        })
        .collect();
    let i1 = core::I1::type_ref(ctx);
    let i32_ty = core::I32::type_ref(ctx);
    for (&block, successors) in blocks.iter().zip(cfg) {
        let targets: Vec<BlockRef> = successors.iter().map(|&i| blocks[i]).collect();
        let terminator = match targets.as_slice() {
            [] => func::Return::operands(std::iter::empty())
                .build(ctx, loc)
                .op_ref(),
            &[dest] => cf::Br::operands(std::iter::empty::<crate::ValueRef>())
                .successors(dest)
                .build(ctx, loc)
                .op_ref(),
            &[then_dest, else_dest] => {
                let cond = arith::Const::operands()
                    .value(Attribute::Int(1))
                    .results(i1)
                    .build(ctx, loc);
                ctx.push_op(block, cond.op_ref());
                let cond = cond.result(ctx);
                cf::CondBr::operands(cond)
                    .successors(then_dest, else_dest)
                    .build(ctx, loc)
                    .op_ref()
            }
            [default, cases @ ..] => {
                let discriminant = arith::Const::operands()
                    .value(Attribute::Int(0))
                    .results(i32_ty)
                    .build(ctx, loc);
                ctx.push_op(block, discriminant.op_ref());
                let discriminant = discriminant.result(ctx);
                cf::Switch::operands(discriminant)
                    .cases(0..cases.len() as i64)
                    .successors(*default, cases.iter().copied())
                    .build(ctx, loc)
                    .op_ref()
            }
        };
        ctx.push_op(block, terminator);
    }
    let body = ctx.create_region(RegionData {
        location: loc,
        blocks: blocks.iter().copied().collect(),
        parent_op: None,
    });
    let nil = core::nil(ctx).as_type_ref();
    let sig = func::func_sig(ctx, [], [nil]).as_type_ref();
    func::Func::operands()
        .sym_name("f")
        .r#type(sig)
        .regions(body)
        .build(ctx, loc);
    (body, blocks)
}

/// Blocks reachable from the entry without passing through `removed`.
fn reachable_avoiding(cfg: &[Vec<usize>], removed: Option<usize>) -> Vec<bool> {
    let mut seen = vec![false; cfg.len()];
    if cfg.is_empty() || removed == Some(0) {
        return seen;
    }
    let mut pending = vec![0];
    while let Some(block) = pending.pop() {
        if std::mem::replace(&mut seen[block], true) {
            continue;
        }
        pending.extend(
            cfg[block]
                .iter()
                .copied()
                .filter(|&successor| Some(successor) != removed),
        );
    }
    seen
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn dominance_matches_its_definition(cfg in cfg()) {
        let mut ctx = IrContext::new();
        let (region, blocks) = build_function(&mut ctx, &cfg);
        let tree = DominatorTree::compute(&ctx, region);

        prop_assert_eq!(tree.region(), region);
        prop_assert_eq!(tree.entry(), blocks.first().copied());
        prop_assert!(tree.is_valid());

        for (index, &block) in blocks.iter().enumerate() {
            let successors: Vec<BlockRef> = cfg[index].iter().map(|&i| blocks[i]).collect();
            prop_assert_eq!(tree.successors(block), successors.as_slice());
            // One predecessor entry per edge, in block order.
            let predecessors: Vec<BlockRef> = cfg
                .iter()
                .enumerate()
                .flat_map(|(from, successors)| {
                    successors.iter().filter(move |&&to| to == index).map(move |_| from)
                })
                .map(|from| blocks[from])
                .collect();
            prop_assert_eq!(tree.predecessors(block), predecessors.as_slice());
        }

        let reachable = reachable_avoiding(&cfg, None);
        for (n, &block) in blocks.iter().enumerate() {
            prop_assert_eq!(tree.is_reachable(block), reachable[n], "block {}", n);
        }
        for (d, &dominator) in blocks.iter().enumerate() {
            let without_d = reachable_avoiding(&cfg, Some(d));
            for (n, &block) in blocks.iter().enumerate() {
                let expected = reachable[n] && (d == n || !without_d[n]);
                prop_assert_eq!(
                    tree.dominates(dominator, block),
                    expected,
                    "dominates({}, {}) in {:?}",
                    d,
                    n,
                    cfg
                );
            }
        }
    }
}
