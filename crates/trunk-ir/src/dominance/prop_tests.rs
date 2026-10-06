//! Property tests comparing [`DominatorTree`] with the definition of
//! dominance on random control-flow graphs.
//!
//! Each case is a [`CfgSpec`](crate::prop::CfgSpec) built as the body of one function, so the
//! graphs include loops, self-loops, repeated edges, and blocks unreachable
//! from the entry. The oracle uses no dataflow: a block is reachable when a
//! search from the entry finds it, and `d` dominates a reachable `n` when
//! `d == n` or `n` is unreachable once `d` is removed.

use proptest::prelude::*;

use super::*;
use crate::prop::{Built, built, cfg_spec, reachable_from};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn dominance_matches_its_definition(built in built(cfg_spec(0..=12))) {
        let Built { spec, ctx, module } = &built;
        let cfg = &spec.successors;
        let function = module.ops(ctx)[0];
        let body = ctx.op_region(function, 0).expect("function body");
        let blocks: Vec<BlockRef> = ctx.region(body).blocks.iter().copied().collect();
        let tree = DominatorTree::compute(ctx, body);

        prop_assert_eq!(tree.region(), body);
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

        let entry = (!cfg.is_empty()).then_some(0);
        let reachable = reachable_from(cfg, entry, None);
        for (n, &block) in blocks.iter().enumerate() {
            prop_assert_eq!(tree.is_reachable(block), reachable[n], "block {}", n);
        }
        for (d, &dominator) in blocks.iter().enumerate() {
            let without_d = reachable_from(cfg, entry, Some(d));
            for (n, &block) in blocks.iter().enumerate() {
                let expected = reachable[n] && (d == n || !without_d[n]);
                prop_assert_eq!(
                    tree.dominates(dominator, block),
                    expected,
                    "dominates({}, {})",
                    d,
                    n
                );
            }
        }
    }
}
