//! Native runtime type descriptor records.
//!
//! Native RTTI generation emits `__tribute_rtti`, a read-only array holding
//! one fixed-size record per RTTI index: the index's release function and its
//! descriptor, in the format `new-plans/rc.md` fixes. Records are built while
//! nominal layouts still carry their names, so lower passes see only the
//! index.

use std::collections::HashMap;

use tribute_ir::dialect::adt;
use tribute_ir::dialect::adt::layout::get_enum_variants;
use tribute_ir::dialect::tribute_rtti::FieldKind;
use trunk_ir::context::{BlockData, IrContext, RegionData};
use trunk_ir::dialect::clif;
use trunk_ir::ops::DialectType;
use trunk_ir::smallvec::smallvec;
use trunk_ir::types::Location;
use trunk_ir::{BlockRef, OpRef, StringRef, Symbol, TypeRef};

use super::rtti::{RTTI_BOOL, RTTI_FLOAT, RTTI_INT, RTTI_NAT, RTTI_NIL};

/// Name of the RTTI table: one record per RTTI index.
pub const RTTI_TABLE: &str = "__tribute_rtti";

const POINTER_SIZE: usize = 8;
/// Size of one RTTI record.
pub const RECORD_SIZE: usize = 48;
/// Offset of the release function pointer in a record.
pub const RELEASE_FN_OFFSET: usize = 0;
const FIELD_SIZE: usize = 16;

/// The `kind` word of a descriptor record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordKind {
    Struct = 0,
    Variant = 1,
    Enum = 2,
    Builtin = 3,
}

/// The contents of one descriptor record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescriptorRecord {
    pub kind: RecordKind,
    pub name: String,
    /// The enum layout a variant belongs to, and the variant's position in it.
    pub owner: Option<(TypeRef, u32)>,
    pub fields: Vec<(String, FieldKind)>,
}

impl DescriptorRecord {
    /// The record of the descriptor `(ty, tag)`, read from its nominal layout.
    pub fn of_layout(
        ctx: &IrContext,
        ty: TypeRef,
        tag: Option<StringRef>,
        fields: Vec<FieldKind>,
    ) -> Self {
        match tag {
            None => {
                let layout = adt::Struct::from_type_ref(ctx, ty)
                    .expect("a struct descriptor has an adt.struct layout");
                let names = layout.fields(ctx).map(|(name, _)| name.to_owned());
                Self {
                    kind: RecordKind::Struct,
                    name: layout.name(ctx).to_owned(),
                    owner: None,
                    fields: names.zip(fields).collect(),
                }
            }
            Some(tag) => {
                let position = get_enum_variants(ctx, ty)
                    .and_then(|variants| variants.iter().position(|(name, _)| *name == tag))
                    .expect("a variant descriptor names a variant of its adt.enum layout");
                Self {
                    kind: RecordKind::Variant,
                    name: ctx.str(tag).to_owned(),
                    owner: Some((ty, position as u32)),
                    fields: positional(fields),
                }
            }
        }
    }

    fn builtin(name: &str, fields: Vec<FieldKind>) -> Self {
        Self {
            kind: RecordKind::Builtin,
            name: name.to_owned(),
            owner: None,
            fields: positional(fields),
        }
    }
}

fn positional(fields: Vec<FieldKind>) -> Vec<(String, FieldKind)> {
    fields
        .into_iter()
        .enumerate()
        .map(|(position, kind)| (position.to_string(), kind))
        .collect()
}

/// The records of the reserved RTTI indices.
fn reserved_records() -> [(u32, DescriptorRecord); 5] {
    [
        (RTTI_NIL, DescriptorRecord::builtin("Bytes", vec![])),
        (
            RTTI_BOOL,
            DescriptorRecord::builtin("Bool", vec![FieldKind::Bool]),
        ),
        (
            RTTI_NAT,
            DescriptorRecord::builtin(
                "Nat",
                vec![FieldKind::Int {
                    width: 32,
                    signed: false,
                }],
            ),
        ),
        (
            RTTI_INT,
            DescriptorRecord::builtin(
                "Int",
                vec![FieldKind::Int {
                    width: 32,
                    signed: true,
                }],
            ),
        ),
        (
            RTTI_FLOAT,
            DescriptorRecord::builtin("Float", vec![FieldKind::Float { width: 64 }]),
        ),
    ]
}

/// Emit `__tribute_rtti`: the record of every reserved index and of each of
/// `records`, with the release function `release_fns` names for its index.
pub fn generate(
    ctx: &mut IrContext,
    module_block: BlockRef,
    records: Vec<(u32, DescriptorRecord)>,
    release_fns: &HashMap<u32, Symbol>,
    loc: Location,
) {
    let mut emitter = Emitter {
        ctx,
        module_block,
        loc,
        names: HashMap::new(),
        enums: HashMap::new(),
    };
    let records = reserved_records()
        .into_iter()
        .chain(records)
        .collect::<Vec<_>>();
    let entries = records.iter().map(|&(index, _)| index).max().unwrap_or(0) as usize + 1;
    let mut table = vec![0; entries * RECORD_SIZE];
    let mut relocs = Vec::new();
    for (index, record) in &records {
        let base = *index as usize * RECORD_SIZE;
        if let Some(&release) = release_fns.get(index) {
            relocs.push((base + RELEASE_FN_OFFSET, clif::RelocTarget::Func(release)));
        }
        emitter.write_record(&mut table, &mut relocs, base, *index, record);
    }
    emitter.data(RTTI_TABLE, table, relocs);
}

type Relocs = Vec<(usize, clif::RelocTarget)>;

struct Emitter<'a> {
    ctx: &'a mut IrContext,
    module_block: BlockRef,
    loc: Location,
    names: HashMap<String, Symbol>,
    enums: HashMap<TypeRef, Symbol>,
}

impl Emitter<'_> {
    /// Write the descriptor part of a record at `bytes[base..]`, emitting its
    /// field array and enum record as separate data. `id` names the field
    /// array.
    fn write_record(
        &mut self,
        bytes: &mut [u8],
        relocs: &mut Relocs,
        base: usize,
        id: impl std::fmt::Display,
        record: &DescriptorRecord,
    ) {
        put_u32(bytes, base + 8, record.kind as u32);
        put_u32(bytes, base + 12, record.fields.len() as u32);
        self.put_name(bytes, relocs, base + 16, base + 24, &record.name);
        if let Some((owner, position)) = record.owner {
            put_u32(bytes, base + 28, position);
            let owner = self.enum_record(owner);
            relocs.push((base + 32, clif::RelocTarget::Data(owner)));
        }
        if !record.fields.is_empty() {
            let mut fields = vec![0; record.fields.len() * FIELD_SIZE];
            let mut field_relocs = Vec::new();
            for (position, (name, kind)) in record.fields.iter().enumerate() {
                let at = position * FIELD_SIZE;
                self.put_name(&mut fields, &mut field_relocs, at, at + 8, name);
                put_u32(&mut fields, at + 12, kind.record_code());
            }
            let fields = self.data(&format!("__tribute_rtti_fields_{id}"), fields, field_relocs);
            relocs.push((base + 40, clif::RelocTarget::Data(fields)));
        }
    }

    /// The record of an enum layout, emitted once outside the table.
    fn enum_record(&mut self, ty: TypeRef) -> Symbol {
        if let Some(&symbol) = self.enums.get(&ty) {
            return symbol;
        }
        let record = DescriptorRecord {
            kind: RecordKind::Enum,
            name: adt::nominal_name(self.ctx, ty)
                .expect("an enum layout has a nominal name")
                .to_owned(),
            owner: None,
            fields: vec![],
        };
        let id = format!("enum_{}", self.enums.len());
        let mut bytes = vec![0; RECORD_SIZE];
        let mut relocs = Vec::new();
        self.write_record(&mut bytes, &mut relocs, 0, &id, &record);
        let symbol = self.data(&format!("__tribute_rtti_{id}"), bytes, relocs);
        self.enums.insert(ty, symbol);
        symbol
    }

    /// Point `bytes[pointer]` at the UTF-8 bytes of `name` and store its
    /// length at `bytes[length]`. An empty name stays a null pointer.
    fn put_name(
        &mut self,
        bytes: &mut [u8],
        relocs: &mut Relocs,
        pointer: usize,
        length: usize,
        name: &str,
    ) {
        put_u32(bytes, length, name.len() as u32);
        if name.is_empty() {
            return;
        }
        let symbol = match self.names.get(name) {
            Some(&symbol) => symbol,
            None => {
                let symbol_name = format!("__tribute_rtti_name_{}", self.names.len());
                let symbol =
                    self.data_with_align(&symbol_name, name.as_bytes().to_vec(), 1, vec![]);
                self.names.insert(name.to_owned(), symbol);
                symbol
            }
        };
        relocs.push((pointer, clif::RelocTarget::Data(symbol)));
    }

    fn data(&mut self, name: &str, bytes: Vec<u8>, relocs: Relocs) -> Symbol {
        self.data_with_align(name, bytes, POINTER_SIZE as u32, relocs)
    }

    fn data_with_align(
        &mut self,
        name: &str,
        bytes: Vec<u8>,
        align: u32,
        relocs: Relocs,
    ) -> Symbol {
        let symbol = Symbol::from_dynamic(name);
        let regions = (!relocs.is_empty()).then(|| self.reloc_region(relocs));
        // Zero bytes rather than zero-initialized data, so the object lives in
        // a data section: macOS linkers reject relocations in zero-fill
        // sections.
        let data = clif::Data::operands()
            .sym_name(symbol)
            .bytes(bytes.into())
            .align(align)
            .regions(regions)
            .build(self.ctx, self.loc);
        self.ctx.push_op(self.module_block, data.op_ref());
        symbol
    }

    fn reloc_region(&mut self, relocs: Relocs) -> trunk_ir::RegionRef {
        let block = self.ctx.create_block(BlockData {
            location: self.loc,
            args: vec![],
            ops: smallvec![],
            parent_region: None,
        });
        for (offset, target) in relocs {
            let offset = u32::try_from(offset).expect("record offset fits u32");
            let reloc: OpRef = match target {
                clif::RelocTarget::Func(func) => clif::FuncReloc::operands()
                    .offset(offset)
                    .func(func)
                    .build(self.ctx, self.loc)
                    .op_ref(),
                clif::RelocTarget::Data(data) => clif::DataReloc::operands()
                    .offset(offset)
                    .data(data)
                    .build(self.ctx, self.loc)
                    .op_ref(),
            };
            self.ctx.push_op(block, reloc);
        }
        self.ctx.create_region(RegionData {
            location: self.loc,
            blocks: smallvec![block],
            parent_op: None,
        })
    }
}

fn put_u32(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use trunk_ir::Symbol;
    use trunk_ir::ops::DialectOp;
    use trunk_ir::parser::parse_test_module;
    use trunk_ir::rewrite::Module;

    fn data_ops(ctx: &IrContext, module: Module) -> HashMap<String, clif::Data> {
        module
            .ops(ctx)
            .iter()
            .filter_map(|&op| clif::Data::from_op(ctx, op).ok())
            .map(|data| (data.sym_name(ctx).to_string(), data))
            .collect()
    }

    fn u32_at(ctx: &IrContext, data: clif::Data, at: usize) -> u32 {
        u32::from_le_bytes(data.bytes(ctx)[at..at + 4].try_into().unwrap())
    }

    fn reloc_at(ctx: &IrContext, data: clif::Data, at: u32) -> Option<Symbol> {
        data.relocations(ctx)
            .into_iter()
            .find_map(|(offset, target)| match target {
                clif::RelocTarget::Data(symbol) if offset == at => Some(symbol),
                _ => None,
            })
    }

    #[test]
    fn variant_records_point_at_their_enum_record() {
        let mut ctx = IrContext::new();
        let module = parse_test_module(
            &mut ctx,
            r#"core.module @test {
  !Choice = adt.enum<{name = "Choice", variants = [["None", []], ["Some", [tribute_rt.anyref]]]}>
  func.func @f() -> core.nil {
    func.return
  }
}"#,
        );
        let choice = ctx.type_alias_by_text("Choice").expect("Choice alias");
        let some = ctx.intern_str("Some");
        let record =
            DescriptorRecord::of_layout(&ctx, choice, Some(some), vec![FieldKind::Dynamic]);
        assert_eq!(record.kind, RecordKind::Variant);
        assert_eq!(record.name, "Some");
        assert_eq!(record.owner, Some((choice, 1)));
        assert_eq!(record.fields, [("0".to_owned(), FieldKind::Dynamic)]);

        let block = module.first_block(&ctx).unwrap();
        let loc = ctx.op(module.op()).location;
        let release = Symbol::new("__tribute_release_5");
        generate(
            &mut ctx,
            block,
            vec![(5, record)],
            &HashMap::from([(5, release)]),
            loc,
        );

        let data = data_ops(&ctx, module);
        let table = data[RTTI_TABLE];
        assert_eq!(table.bytes(&ctx).len(), 6 * RECORD_SIZE);
        let base = 5 * RECORD_SIZE;
        let at = |offset: usize| (base + offset) as u32;
        assert!(
            table
                .relocations(&ctx)
                .contains(&(at(RELEASE_FN_OFFSET), clif::RelocTarget::Func(release)))
        );
        assert_eq!(u32_at(&ctx, table, base + 8), RecordKind::Variant as u32);
        assert_eq!(u32_at(&ctx, table, base + 12), 1);
        assert_eq!(u32_at(&ctx, table, base + 24), 4);
        assert_eq!(u32_at(&ctx, table, base + 28), 1);
        let owner = reloc_at(&ctx, table, at(32)).expect("enum record");
        let owner = data[&owner.to_string()];
        assert_eq!(u32_at(&ctx, owner, 8), RecordKind::Enum as u32);
        assert_eq!(u32_at(&ctx, owner, 12), 0);
        let name = reloc_at(&ctx, owner, 16).expect("enum name");
        assert_eq!(data[&name.to_string()].bytes(&ctx).as_ref(), b"Choice");
        let fields = reloc_at(&ctx, table, at(40)).expect("variant fields");
        assert_eq!(
            u32_at(&ctx, data[&fields.to_string()], 12),
            FieldKind::Dynamic.record_code()
        );

        // Every reserved index has a record, and none has a release function.
        for index in 0..5 {
            let base = index * RECORD_SIZE;
            assert_eq!(u32_at(&ctx, table, base + 8), RecordKind::Builtin as u32);
            assert!(reloc_at(&ctx, table, (base + 16) as u32).is_some());
        }
        assert_eq!(
            table
                .relocations(&ctx)
                .iter()
                .filter(|(_, target)| matches!(target, clif::RelocTarget::Func(_)))
                .count(),
            1
        );
    }
}
