//! Value delimiters: CPS control inside a callable that is not Cps.

use super::*;

impl Converter<'_> {
    /// Whether `source` needs CPS control from the flow that evaluates it: a
    /// handle, a resume, or a call of a Cps callable. Pre-CPS validation
    /// rejects a `become` of a Cps callable in a callable that is not Cps.
    pub(super) fn needs_cps_control(&self, source: OpRef) -> bool {
        if tribute_control::Handle::matches(self.ctx, source)
            || tribute_control::Resume::matches(self.ctx, source)
        {
            return true;
        }
        if tribute_control::Call::matches(self.ctx, source) {
            return self
                .ctx
                .op(source)
                .attributes
                .get_symbol_ref("callee")
                .and_then(|callee| self.funcs.get(callee))
                .is_some_and(|target| target.convention == CallingConvention::Cps);
        }
        if tribute_control::CallIndirect::matches(self.ctx, source) {
            let callee = self.ctx.value_ty(self.ctx.op_operands(source)[0]);
            return tribute_control::func_sig_convention(self.ctx, callee)
                == Some(tribute_control::CallingConvention::Cps);
        }
        false
    }

    /// Run `source` under a value delimiter in a flow that is not Cps, and
    /// return its answer. The operation becomes the body of a Cps closure
    /// that exits to the delimiter's frame.
    pub(super) fn delimit(
        &mut self,
        source: OpRef,
        block: BlockRef,
        mapping: &HashMap<ValueRef, ValueRef>,
        flow: &Flow,
    ) -> Result<ValueRef, TributeControlToCpsError> {
        let location = self.ctx.op(source).location;
        let &[result] = self.ctx.op_results(source) else {
            return Err(self.malformed_source(
                source,
                "an operation under a value delimiter has exactly one result",
            ));
        };
        let result_type = self.ctx.value_ty(result);
        let exit = tribute_control::Return::operands(result).build(self.ctx, location);
        let ops = OpList::from_slice(&[source, exit.op_ref()]);
        let answer = self.convert_type(result_type);
        let evidence_type = self.evidence_type();
        let frame_type = self.frame_type(answer);
        let body = make_block(self.ctx, location, &[evidence_type, frame_type]);
        let body_flow = Flow {
            convention: CallingConvention::Cps,
            evidence: Some(self.ctx.block_args(body)[0]),
            exit_k: Some(self.ctx.block_args(body)[1]),
            root_exit_k: Some(self.ctx.block_args(body)[1]),
            void_exit_k: None,
            answer_type: answer,
            preserve_scf_yield: false,
            arm: None,
            tail_join: None,
        };
        // The exit is conversion input only; it must not keep a use of the
        // source result once the body is built, whether or not that succeeds.
        let converted = self.convert_sequence(ops, 0, body, &mut mapping.clone(), &body_flow);
        self.ctx.remove_op(exit.op_ref());
        converted?;
        let region = single_block_region(self.ctx, location, body);
        let never = self.never_type();
        let function = func::func_sig(self.ctx, [evidence_type, frame_type], [never]).as_type_ref();
        let closure_type = self.generated_continuation_type(function);
        let closure = closure_over(
            self.ctx,
            location,
            region,
            closure_type,
            CallingConvention::Cps,
        );
        self.ctx.push_op(block, closure.op_ref());
        // A Direct flow receives no evidence; its delimiter starts on the
        // initial one.
        let evidence = match flow.evidence {
            Some(evidence) => evidence,
            None => {
                let initial = effect::InitialEvidence::operands()
                    .results(evidence_type)
                    .build(self.ctx, location);
                self.ctx.push_op(block, initial.op_ref());
                initial.result(self.ctx)
            }
        };
        let delimit = ability::Delimit::operands(closure.result(self.ctx), evidence)
            .build(self.ctx, location);
        self.ctx.push_op(block, delimit.op_ref());
        Ok(delimit.result(self.ctx))
    }
}
