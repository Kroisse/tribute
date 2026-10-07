//! Conversion of source-logical types, callable signatures, and attributes to
//! their physical forms, and the continuation types the conversion builds.

use super::*;

impl Converter<'_> {
    pub(super) fn never_type(&mut self) -> TypeRef {
        core::never(self.ctx).as_type_ref()
    }

    pub(super) fn evidence_type(&mut self) -> TypeRef {
        ability::evidence_adt_type_ref(self.ctx)
    }

    pub(super) fn anyref_type(&mut self) -> TypeRef {
        tribute_rt::anyref(self.ctx).as_type_ref()
    }

    pub(super) fn resumption_type(&mut self, input: TypeRef, answer: TypeRef) -> TypeRef {
        let evidence = self.evidence_type();
        let frame = self.frames.frame_type(self.ctx, answer);
        cps_resume_exact_type(self.ctx, evidence, input, frame)
    }

    pub(super) fn completion_type(&mut self, value: TypeRef, answer: TypeRef) -> TypeRef {
        let evidence = self.evidence_type();
        let frame = self.frames.frame_type(self.ctx, answer);
        cps_completion_type(self.ctx, evidence, value, frame)
    }

    pub(super) fn generated_continuation_type(&mut self, function: TypeRef) -> TypeRef {
        physical_closure_type_with_environment_index(self.ctx, function, CallingConvention::Cps, 0)
    }

    pub(super) fn convert_attribute(&mut self, attribute: &Attribute) -> Attribute {
        attribute.map_types(|ty| self.convert_type(ty))
    }

    pub(super) fn convert_attrs(&mut self, attrs: &AttributeMap) -> Vec<(Symbol, Attribute)> {
        attrs
            .iter()
            .map(|(key, value)| (key.clone(), self.convert_attribute(value)))
            .collect()
    }

    pub(super) fn convert_type(&mut self, ty: TypeRef) -> TypeRef {
        if let Some(converted) = self.converted_types.get(&ty) {
            return *converted;
        }
        if tribute_control::FuncSig::from_type_ref(self.ctx, ty).is_some() {
            let (function, convention) = self.lower_callable_signature(ty);
            let converted = physical_closure_type(self.ctx, function, convention);
            self.converted_types.insert(ty, converted);
            return converted;
        }
        let data = self.ctx.get_type(ty).clone();
        if data.dialect == func::DIALECT_NAME() && data.name == func::FUNC_SIG() {
            let function = func::FuncSig::from_type_ref(self.ctx, ty)
                .expect("pre-CPS validation must reject malformed func.func_sig types");
            let source_inputs = function.inputs(self.ctx).to_vec();
            let source_results = function.results(self.ctx).to_vec();
            let inputs = source_inputs
                .into_iter()
                .map(|input| self.convert_type(input))
                .collect::<Vec<_>>();
            let results = source_results
                .into_iter()
                .map(|result| self.convert_type(result))
                .collect::<Vec<_>>();
            let mut attrs = data.attrs;
            func::FuncSig::remove_reserved_attrs(&mut attrs);
            for value in attrs.values_mut() {
                *value = self.convert_attribute(value);
            }
            let converted =
                func::func_sig_with_attrs(self.ctx, inputs, results, attrs).as_type_ref();
            self.converted_types.insert(ty, converted);
            return converted;
        }
        if data.dialect == "tribute_control"
            && data.name == "resume_token"
            && data.params.len() == 2
        {
            let input = self.convert_type(data.params[0]);
            let answer = self.convert_type(data.params[1]);
            let converted = self.resumption_type(input, answer);
            self.converted_types.insert(ty, converted);
            return converted;
        }

        let params: Vec<_> = data
            .params
            .iter()
            .copied()
            .map(|param| self.convert_type(param))
            .collect();
        let attrs: Vec<_> = data
            .attrs
            .iter()
            .map(|(key, value)| (key.clone(), self.convert_attribute(value)))
            .collect();
        if params == data.params.as_slice()
            && attrs
                .iter()
                .all(|(key, value)| data.attrs.get(key) == Some(value))
        {
            self.converted_types.insert(ty, ty);
            return ty;
        }
        let mut builder = TypeDataBuilder::new(data.dialect, data.name).params(params);
        for (key, value) in attrs {
            builder = builder.attr(key, value);
        }
        let converted = self.ctx.intern_type(builder.build());
        self.converted_types.insert(ty, converted);
        converted
    }

    pub(super) fn physical_function_type(&mut self, logical: TypeRef) -> TypeRef {
        self.lower_callable_signature(logical).0
    }

    /// Lower a source signature to its logical `func.func_sig` in convention
    /// order. Parameter attributes follow the parameters they describe: hidden
    /// parameters have none, and a Cps source result, replaced by
    /// `core.never`, drops its own.
    pub(super) fn lower_callable_signature(
        &mut self,
        logical: TypeRef,
    ) -> (TypeRef, CallingConvention) {
        let callable = tribute_control::FuncSig::from_type_ref(self.ctx, logical)
            .expect("pre-CPS validation checked function type");
        let convention = convert_convention(
            tribute_control::func_sig_convention(self.ctx, logical)
                .expect("pre-CPS validation checked function convention"),
        );
        let (source_result, result_attrs) = callable.result_with_attrs(self.ctx);
        let result_attrs = result_attrs.clone();
        let result = self.convert_type(source_result);
        let result_attrs = self.convert_attr_map(&result_attrs);
        let source_params: Vec<_> = callable
            .inputs_with_attrs(self.ctx)
            .map(|(param, attrs)| (param, attrs.clone()))
            .collect();
        let params: Vec<_> = source_params
            .into_iter()
            .map(|(param, attrs)| (self.convert_type(param), self.convert_attr_map(&attrs)))
            .collect();
        let evidence = (self.evidence_type(), AttributeMap::new());
        let frame = (
            self.frames.frame_type(self.ctx, result),
            AttributeMap::new(),
        );
        let abi = CallableAbi::new(convention, params, (result, result_attrs));
        let params = abi.lowered_params(evidence, frame);
        let result = if convention == CallingConvention::Cps {
            (self.never_type(), AttributeMap::new())
        } else {
            abi.source_result
        };
        let mut attrs = self.ctx.get_type(logical).attrs.clone();
        tribute_control::FuncSig::remove_reserved_attrs(&mut attrs);
        for value in attrs.values_mut() {
            *value = self.convert_attribute(value);
        }
        let function =
            func::func_sig_with_param_attrs(self.ctx, params, [result], attrs).as_type_ref();
        (function, convention)
    }

    pub(super) fn convert_attr_map(&mut self, attrs: &AttributeMap) -> AttributeMap {
        self.convert_attrs(attrs).into_iter().collect()
    }

    pub(super) fn copy_extra_attrs(&mut self, source: OpRef, target: OpRef, excluded: &[&str]) {
        let excluded: HashSet<Symbol> = excluded.iter().map(|name| Symbol::new(name)).collect();
        let attrs = self.convert_attrs(&self.ctx.op(source).attributes.clone());
        for (key, value) in attrs {
            if !excluded.contains(&key) {
                self.ctx.op_mut(target).attributes.insert(key, value);
            }
        }
    }
}
