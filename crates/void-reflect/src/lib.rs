//! Offline extraction of the engine's reflection metadata (docs/renderer-replacement.md).
//!
//! The executable fills class and member tables with static initializers made of
//! a small, regular instruction set (O, IDB of exe `b671f824...6180`):
//! `lea rdx, "string"; lea rcx, slot; call <string setter>`, immediate stores
//! (`mov dword [slot], imm`), address stores (`lea rax, X; mov [slot], rax`), and
//! 16-byte zero/constant stores. [`Evaluator`] interprets that subset over the
//! initializers and records the resulting writes to the data section; [`read_types`]
//! then reads class records (0x40 bytes) and member records (0x48 bytes) back.
//! Instructions outside the subset only clobber their destination registers.

use std::collections::{BTreeMap, HashMap};

use iced_x86::{Decoder, DecoderOptions, Instruction, Mnemonic, OpKind, Register};

/// A section of the loaded image (virtual address, bytes as in the file).
#[derive(Clone, Debug)]
pub struct Section {
    pub name: String,
    pub va: u64,
    pub vsize: u64,
    pub data: Vec<u8>,
    pub executable: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Image {
    pub base: u64,
    pub sections: Vec<Section>,
}

impl Image {
    pub fn section_of(&self, va: u64) -> Option<&Section> {
        self.sections
            .iter()
            .find(|s| va >= s.va && va < s.va + s.vsize.max(s.data.len() as u64))
    }

    pub fn bytes(&self, va: u64, n: usize) -> Option<&[u8]> {
        let s = self.section_of(va)?;
        let off = (va - s.va) as usize;
        s.data.get(off..off + n)
    }

    pub fn u64_at(&self, va: u64) -> Option<u64> {
        Some(u64::from_le_bytes(self.bytes(va, 8)?.try_into().ok()?))
    }

    pub fn is_code(&self, va: u64) -> bool {
        self.section_of(va).is_some_and(|s| s.executable)
    }

    /// NUL-terminated printable string at `va` (bounded).
    pub fn c_string(&self, va: u64) -> Option<String> {
        let s = self.section_of(va)?;
        let off = (va - s.va) as usize;
        let tail = s.data.get(off..)?;
        let end = tail.iter().take(1024).position(|&b| b == 0)?;
        let b = &tail[..end];
        if b.iter().all(|&c| (0x20..0x7F).contains(&c)) {
            Some(String::from_utf8_lossy(b).into_owned())
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Val {
    Imm(u64),
    Addr(u64),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Write {
    /// Little-endian value of `size` bytes.
    Value { size: u8, val: Val },
    /// 16 raw bytes (vector store).
    Bytes([u8; 16]),
    /// String object assigned through a setter call.
    Str(String),
}

#[derive(Default)]
pub struct Evaluator {
    /// Data address -> write (last one wins).
    pub writes: BTreeMap<u64, Write>,
    /// Functions evaluated and instructions that were outside the subset (counts).
    pub functions: usize,
    pub unhandled: usize,
}

#[derive(Clone, Copy)]
enum Xmm {
    Zero,
    Const([u8; 16]),
}

const MAX_INSTRUCTIONS: usize = 2_000_000;

impl Evaluator {
    /// Evaluates one initializer from `start` until `ret`/`int3`/tail `jmp`.
    pub fn run(&mut self, img: &Image, start: u64) {
        let Some(sec) = img.section_of(start).filter(|s| s.executable) else {
            return;
        };
        let off = (start - sec.va) as usize;
        let code = &sec.data[off..];
        let mut dec = Decoder::with_ip(64, code, start, DecoderOptions::NONE);
        let mut regs: HashMap<Register, Val> = HashMap::new();
        let mut xmm: HashMap<Register, Xmm> = HashMap::new();
        let mut ins = Instruction::default();
        self.functions += 1;
        for _ in 0..MAX_INSTRUCTIONS {
            if !dec.can_decode() {
                break;
            }
            dec.decode_out(&mut ins);
            match ins.mnemonic() {
                Mnemonic::Ret | Mnemonic::Int3 | Mnemonic::Jmp => break,
                Mnemonic::Lea
                    if ins.op0_kind() == OpKind::Register && ins.is_ip_rel_memory_operand() =>
                {
                    regs.insert(
                        ins.op0_register().full_register(),
                        Val::Addr(ins.ip_rel_memory_address()),
                    );
                }
                Mnemonic::Mov => self.mov(img, &ins, &mut regs),
                Mnemonic::Xor
                    if ins.op0_kind() == OpKind::Register
                        && ins.op1_kind() == OpKind::Register
                        && ins.op0_register() == ins.op1_register() =>
                {
                    regs.insert(ins.op0_register().full_register(), Val::Imm(0));
                }
                Mnemonic::Xorps | Mnemonic::Pxor | Mnemonic::Vxorps | Mnemonic::Vpxor
                    if ins.op0_kind() == OpKind::Register
                        && ins.op1_kind() == OpKind::Register
                        && ins.op0_register() == ins.op1_register() =>
                {
                    xmm.insert(ins.op0_register(), Xmm::Zero);
                }
                Mnemonic::Movups | Mnemonic::Movaps | Mnemonic::Movdqa | Mnemonic::Movdqu => {
                    if ins.op0_kind() == OpKind::Register && ins.is_ip_rel_memory_operand() {
                        let c = img
                            .bytes(ins.ip_rel_memory_address(), 16)
                            .and_then(|b| b.try_into().ok())
                            .map(Xmm::Const);
                        match c {
                            Some(c) => xmm.insert(ins.op0_register(), c),
                            None => xmm.remove(&ins.op0_register()),
                        };
                    } else if ins.op0_kind() == OpKind::Memory
                        && ins.is_ip_rel_memory_operand()
                        && ins.op1_kind() == OpKind::Register
                    {
                        if let Some(v) = xmm.get(&ins.op1_register()) {
                            let b = match v {
                                Xmm::Zero => [0; 16],
                                Xmm::Const(c) => *c,
                            };
                            self.writes
                                .insert(ins.ip_rel_memory_address(), Write::Bytes(b));
                        } else {
                            self.unhandled += 1;
                        }
                    } else if ins.op0_kind() == OpKind::Register {
                        xmm.remove(&ins.op0_register());
                    }
                }
                Mnemonic::Call => {
                    let rcx = regs.get(&Register::RCX).copied();
                    let rdx = regs.get(&Register::RDX).copied();
                    if let (Some(Val::Addr(dst)), Some(Val::Addr(src))) = (rcx, rdx)
                        && !img.is_code(dst)
                        && let Some(s) = img.c_string(src)
                    {
                        self.writes.insert(dst, Write::Str(s));
                    }
                    for r in [
                        Register::RAX,
                        Register::RCX,
                        Register::RDX,
                        Register::R8,
                        Register::R9,
                        Register::R10,
                        Register::R11,
                    ] {
                        regs.remove(&r);
                    }
                    xmm.clear();
                }
                _ => {
                    if ins.op_count() > 0 && ins.op0_kind() == OpKind::Register {
                        regs.remove(&ins.op0_register().full_register());
                        xmm.remove(&ins.op0_register());
                    }
                    if ins.op_count() > 0
                        && ins.op0_kind() == OpKind::Memory
                        && ins.is_ip_rel_memory_operand()
                    {
                        self.unhandled += 1;
                    }
                }
            }
        }
    }

    fn mov(&mut self, _img: &Image, ins: &Instruction, regs: &mut HashMap<Register, Val>) {
        match (ins.op0_kind(), ins.op1_kind()) {
            (OpKind::Register, OpKind::Register) => {
                let v = regs.get(&ins.op1_register().full_register()).copied();
                let d = ins.op0_register().full_register();
                match v {
                    Some(v) => regs.insert(d, v),
                    None => regs.remove(&d),
                };
            }
            (OpKind::Register, k) if is_imm(k) => {
                regs.insert(
                    ins.op0_register().full_register(),
                    Val::Imm(ins.immediate(1)),
                );
            }
            (OpKind::Register, _) => {
                regs.remove(&ins.op0_register().full_register());
            }
            (OpKind::Memory, k) if ins.is_ip_rel_memory_operand() => {
                let size = ins.memory_size().size() as u8;
                let val = if is_imm(k) {
                    Some(Val::Imm(ins.immediate(1)))
                } else if k == OpKind::Register {
                    regs.get(&ins.op1_register().full_register()).copied()
                } else {
                    None
                };
                match val {
                    Some(val) => {
                        self.writes
                            .insert(ins.ip_rel_memory_address(), Write::Value { size, val });
                    }
                    None => self.unhandled += 1,
                }
            }
            _ => {}
        }
    }

    pub fn string(&self, va: u64) -> Option<&str> {
        match self.writes.get(&va) {
            Some(Write::Str(s)) => Some(s),
            _ => None,
        }
    }

    pub fn value(&self, va: u64) -> Option<Val> {
        match self.writes.get(&va) {
            Some(Write::Value { val, .. }) => Some(*val),
            Some(Write::Bytes(b)) => Some(Val::Imm(u64::from_le_bytes(b[..8].try_into().ok()?))),
            _ => None,
        }
    }

    /// 32-bit field: a direct dword store, or the low/high half of an earlier wider store.
    pub fn u32_at(&self, va: u64) -> Option<u32> {
        match self.writes.get(&va) {
            Some(Write::Value {
                val: Val::Imm(v), ..
            }) => return Some(*v as u32),
            Some(Write::Bytes(b)) => return Some(u32::from_le_bytes(b[..4].try_into().ok()?)),
            _ => {}
        }
        for base in [va.checked_sub(4)?, va.checked_sub(8)?, va.checked_sub(12)?] {
            if let Some(Write::Bytes(b)) = self.writes.get(&base) {
                let o = (va - base) as usize;
                return Some(u32::from_le_bytes(b[o..o + 4].try_into().ok()?));
            }
            if let Some(Write::Value {
                size: 8,
                val: Val::Imm(v),
            }) = self.writes.get(&base)
                && va - base == 4
            {
                return Some((*v >> 32) as u32);
            }
        }
        None
    }
}

fn is_imm(k: OpKind) -> bool {
    matches!(
        k,
        OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Member {
    pub type_name: String,
    /// `*`, `[N]`, `&`, ... or empty.
    pub modifier: String,
    pub name: String,
    pub offset: u32,
    /// Bytes; bitfield members report 0 offset and a getter (size field -1/-3 observed).
    pub size: i32,
    pub record: u64,
    /// Bitfield members: (byte offset, bit index) decoded from the getter (record +0x30).
    pub bit: Option<(u32, u32)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Class {
    pub name: String,
    pub base: String,
    pub size: u32,
    pub record: u64,
    pub members_at: u64,
    pub members: Vec<Member>,
}

/// Reads class records: a string at `A`, an optional string at `A+8`, a size at
/// `A+16`, and a member-table address at `A+40`. Member records are 0x48 bytes
/// and end at the first record without a member name.
/// Decodes a bitfield getter of the form `movzx/mov eax, [rcx+disp]` followed by
/// optional `shr eax, n` and `and eax, mask` (or `test [rcx+disp], mask; setnz al`):
/// returns (byte offset, bit index). O: e.g. `sub_1409904E0` = `movzx eax, byte
/// [rcx+1C30h]; and eax, 1; ret`.
pub fn decode_bit_getter(img: &Image, va: u64) -> Option<(u32, u32)> {
    let sec = img.section_of(va).filter(|s| s.executable)?;
    let off = (va - sec.va) as usize;
    let code = sec.data.get(off..off + 64)?;
    let mut dec = Decoder::with_ip(64, code, va, DecoderOptions::NONE);
    let mut ins = Instruction::default();
    let mut disp: Option<u64> = None;
    let mut shift = 0u32;
    for _ in 0..8 {
        if !dec.can_decode() {
            return None;
        }
        dec.decode_out(&mut ins);
        match ins.mnemonic() {
            Mnemonic::Movzx | Mnemonic::Mov
                if ins.op1_kind() == OpKind::Memory && ins.memory_base() == Register::RCX =>
            {
                disp = Some(ins.memory_displacement64());
            }
            Mnemonic::Test
                if ins.op0_kind() == OpKind::Memory && ins.memory_base() == Register::RCX =>
            {
                let mask = ins.immediate(1);
                if mask.count_ones() != 1 {
                    return None;
                }
                let d = ins.memory_displacement64();
                let bit = mask.trailing_zeros();
                return Some(((d + u64::from(bit / 8)) as u32, bit % 8));
            }
            Mnemonic::Shr if is_imm(ins.op1_kind()) => shift += ins.immediate(1) as u32,
            Mnemonic::And if is_imm(ins.op1_kind()) => {
                let mask = ins.immediate(1);
                if mask.count_ones() != 1 {
                    return None;
                }
                let bit = shift + mask.trailing_zeros();
                let d = disp?;
                return Some(((d + u64::from(bit / 8)) as u32, bit % 8));
            }
            Mnemonic::Ret => {
                // `shr al, 7` style getters end without a mask.
                let d = disp?;
                return Some(((d + u64::from(shift / 8)) as u32, shift % 8));
            }
            _ => {}
        }
    }
    None
}

pub fn read_types(ev: &Evaluator, img: &Image) -> Vec<Class> {
    let mut out = Vec::new();
    for (&a, w) in &ev.writes {
        let Write::Str(name) = w else { continue };
        let Some(Val::Addr(members_at)) = ev.value(a + 40) else {
            continue;
        };
        let Some(size) = ev.u32_at(a + 16) else {
            continue;
        };
        // A class record's member table must itself start with a member record.
        if ev.string(members_at + 16).is_none() {
            continue;
        }
        let mut members = Vec::new();
        let mut m = members_at;
        while let Some(mname) = ev.string(m + 16) {
            members.push(Member {
                type_name: ev.string(m).unwrap_or("").to_owned(),
                modifier: ev.string(m + 8).unwrap_or("").to_owned(),
                name: mname.to_owned(),
                offset: ev.u32_at(m + 24).unwrap_or(u32::MAX),
                size: ev.u32_at(m + 28).map_or(i32::MIN, |v| v as i32),
                record: m,
                bit: match (ev.u32_at(m + 28), ev.value(m + 0x30)) {
                    (Some(sz), Some(Val::Addr(getter))) if (sz as i32) < 0 => {
                        decode_bit_getter(img, getter)
                    }
                    _ => None,
                },
            });
            m += 0x48;
            if members.len() > 100_000 {
                break;
            }
        }
        out.push(Class {
            name: name.clone(),
            base: ev.string(a + 8).unwrap_or("").to_owned(),
            size,
            record: a,
            members_at,
            members,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic code: string setter calls, dword stores, an address store, and a
    /// zero vector store, laid out like the engine's initializers.
    #[test]
    fn evaluates_initializer_subset() {
        let code_va = 0x1000u64;
        let rdata_va = 0x2000u64;
        let data_va = 0x3000u64;
        let mut rdata = vec![0u8; 0x100];
        rdata[..6].copy_from_slice(b"idFoo\0");
        rdata[0x10..0x14].copy_from_slice(b"int\0");
        rdata[0x20..0x26].copy_from_slice(b"m_bar\0");
        // Assemble by hand (RIP-relative operands computed per instruction).
        let mut code: Vec<u8> = Vec::new();
        let rel = |code: &Vec<u8>, len: usize, target: u64| -> [u8; 4] {
            let next = code_va + code.len() as u64 + len as u64;
            ((target as i64 - next as i64) as i32).to_le_bytes()
        };
        // lea rdx,[rip+s] (48 8D 15 rel32); lea rcx,[rip+d] (48 8D 0D rel32); call rel32 (E8)
        let set = |code: &mut Vec<u8>, dst: u64, src: u64| {
            let r = rel(code, 7, src);
            code.extend_from_slice(&[0x48, 0x8D, 0x15]);
            code.extend_from_slice(&r);
            let r = rel(code, 7, dst);
            code.extend_from_slice(&[0x48, 0x8D, 0x0D]);
            code.extend_from_slice(&r);
            code.extend_from_slice(&[0xE8, 0, 0, 0, 0]);
        };
        // Class record at data+0, member table at data+0x100.
        set(&mut code, data_va, rdata_va);
        // mov dword [rip+X], imm32 (C7 05 rel32 imm32)
        let movd = |code: &mut Vec<u8>, dst: u64, v: u32| {
            let r = rel(code, 10, dst);
            code.extend_from_slice(&[0xC7, 0x05]);
            code.extend_from_slice(&r);
            code.extend_from_slice(&v.to_le_bytes());
        };
        movd(&mut code, data_va + 16, 0x40);
        // lea rax,[rip+members]; mov [rip+class+40], rax
        let r = rel(&code, 7, data_va + 0x100);
        code.extend_from_slice(&[0x48, 0x8D, 0x05]);
        code.extend_from_slice(&r);
        let r = rel(&code, 7, data_va + 40);
        code.extend_from_slice(&[0x48, 0x89, 0x05]);
        code.extend_from_slice(&r);
        set(&mut code, data_va + 0x100, rdata_va + 0x10);
        set(&mut code, data_va + 0x110, rdata_va + 0x20);
        movd(&mut code, data_va + 0x118, 12);
        movd(&mut code, data_va + 0x11C, 4);
        // xorps xmm0,xmm0 (0F 57 C0); movups [rip+X], xmm0 (0F 11 05 rel32)
        code.extend_from_slice(&[0x0F, 0x57, 0xC0]);
        let r = rel(&code, 7, data_va + 0x200);
        code.extend_from_slice(&[0x0F, 0x11, 0x05]);
        code.extend_from_slice(&r);
        code.push(0xC3);
        let img = Image {
            base: 0,
            sections: vec![
                Section {
                    name: ".text".into(),
                    va: code_va,
                    vsize: 0x1000,
                    data: code,
                    executable: true,
                },
                Section {
                    name: ".rdata".into(),
                    va: rdata_va,
                    vsize: 0x100,
                    data: rdata,
                    executable: false,
                },
                Section {
                    name: ".data".into(),
                    va: data_va,
                    vsize: 0x1000,
                    data: vec![0; 0x1000],
                    executable: false,
                },
            ],
        };
        let mut ev = Evaluator::default();
        ev.run(&img, code_va);
        assert_eq!(
            ev.writes.get(&(data_va + 0x200)),
            Some(&Write::Bytes([0; 16]))
        );
        let types = read_types(&ev, &img);
        assert_eq!(types.len(), 1);
        let c = &types[0];
        assert_eq!((c.name.as_str(), c.size), ("idFoo", 0x40));
        assert_eq!(c.members.len(), 1);
        let m = &c.members[0];
        assert_eq!(
            (m.type_name.as_str(), m.name.as_str(), m.offset, m.size),
            ("int", "m_bar", 12, 4)
        );
    }
}
