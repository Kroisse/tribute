//! Names interned while compiling are released once nothing refers to them.
//!
//! The dynamic symbol set is process-wide, so this file holds a single test:
//! a second test running in the same process would change the count.

use salsa::{Database, Setter};
use tribute::database::parse_with_thread_local;
use tribute::pipeline::{
    compile_frontend_for_shared_route, compile_with_diagnostics, emit_from_boundary_exit,
    run_shared_middle_end, run_target_to_boundary_exit,
};
use tribute::{Rope, SourceCst, TributeDatabaseImpl};
use tribute_passes::abi_boundary::TargetKind;
use trunk_ir::symbol::live_dynamic_symbols;

/// A program whose names are unique to `round` and too long to be inline.
fn program(round: usize) -> String {
    format!(
        r#"ability AskRound{round}Value {{
    fn ask_round_{round}_value() -> Nat
}}

fn use_round_{round}_value() ->{{AskRound{round}Value}} Nat {{
    AskRound{round}Value::ask_round_{round}_value()
}}

fn helper_for_round_{round}(counter_in_round_{round}: Nat) -> Nat {{
    counter_in_round_{round} + 1
}}

fn main() -> Nil {{
    let result_of_round_{round} = handle use_round_{round}_value() {{
        do result {{ result }}
        fn AskRound{round}Value::ask_round_{round}_value() {{ 42 }}
    }}
    let _ = helper_for_round_{round}(result_of_round_{round})
}}
"#
    )
}

/// Compile `round`'s program to both targets on a fresh database and return
/// the number of dynamic symbols alive while its IR still exists.
fn compile_from_scratch(round: usize) -> usize {
    let rope = Rope::from_str(&program(round));
    let mut alive = 0;
    for target in [TargetKind::Native, TargetKind::Wasm] {
        let frontend = TributeDatabaseImpl::default().attach(|db| {
            let tree = parse_with_thread_local(&rope, None);
            let source = SourceCst::from_path(db, "round.trb", rope.clone(), tree);
            compile_frontend_for_shared_route(db, source).expect("the program compiles")
        });
        let (mut ctx, module) = run_shared_middle_end(frontend).expect("shared middle-end");
        run_target_to_boundary_exit(&mut ctx, module, target).expect("target boundary");
        emit_from_boundary_exit(&mut ctx, module, target).expect("emission");
        alive = alive.max(live_dynamic_symbols());
    }
    alive
}

#[test]
fn dynamic_symbols_do_not_accumulate() {
    // Repeated compilations, each on its own database.
    compile_from_scratch(0);
    let settled = live_dynamic_symbols();
    for round in 1..8 {
        let alive = compile_from_scratch(round);
        assert!(
            alive > settled,
            "round {round} interned no dynamic symbol, so the test checks nothing"
        );
        assert_eq!(
            live_dynamic_symbols(),
            settled,
            "round {round} left dynamic symbols behind"
        );
    }

    // Repeated edits of one document in one database, as a language server
    // makes them. Results of earlier revisions are replaced, but Salsa keeps
    // interned definition ids, and so their names, while the database lives.
    let mut db = TributeDatabaseImpl::default();
    let rope = Rope::from_str(&program(100));
    let tree = parse_with_thread_local(&rope, None);
    let source = SourceCst::from_path(&db, "edited.trb", rope, tree);
    let mut after_edit = Vec::new();
    for round in 100..116 {
        let rope = Rope::from_str(&program(round));
        let tree = parse_with_thread_local(&rope, None);
        source.set_text(&mut db).to(rope);
        source.set_tree(&mut db).to(tree);
        let result = compile_with_diagnostics(&db, source);
        assert!(result.diagnostics.is_empty(), "{:?}", result.diagnostics);
        drop(result);
        after_edit.push(live_dynamic_symbols());
    }
    let first = after_edit[0];
    let last = after_edit[after_edit.len() - 1];
    let per_edit = (last - first) / (after_edit.len() - 1);
    assert!(
        per_edit < first / 8,
        "an edit retains {per_edit} names; one revision uses {first}: {after_edit:?}"
    );
    drop(db);
    assert_eq!(
        live_dynamic_symbols(),
        settled,
        "dropping the database left dynamic symbols behind"
    );
}
