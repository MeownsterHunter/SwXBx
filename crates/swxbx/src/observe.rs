//! Records the shape of the values a program passes to and gets back from
//! each function, while the original implementation still does the work.
//!
//! Nothing is known about any function's signature at the start. Watching
//! what the argument registers hold across many calls shows which of them
//! carry pointers, small numbers, constants or floating-point values, and
//! that is the evidence a signature is worked out from.
//!
//! Only the shape is kept: ranges, a handful of distinct values, and whether
//! a value was a readable address. For an address argument, the first 64
//! bytes behind it are classified word by word in the same way, and compared
//! after the call to see what the function wrote. No memory is stored or
//! reported verbatim: a word that is not a small integer or a plausible
//! float is reported only as "other".

use crate::api::{GpuAddressError, Objects};
use crate::instance::{Host, Registers};
use std::collections::BTreeMap;
use std::fmt::{self, Write};

/// Calls sampled per function. Shapes settle quickly, and the busiest
/// functions are called a million times a minute.
pub const SAMPLE_LIMIT: u64 = 2048;

const DISTINCT_LIMIT: usize = 6;
/// Values below this are never addresses in a program's address space.
const LOWEST_ADDRESS: u64 = 0x1_0000;
const ADDRESS_LIMIT: u64 = 1 << 48;
/// Largest value printed as it is.
const WIDE: u64 = u32::MAX as u64;

/// What one register held across the sampled calls.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RegisterShape {
    samples: u64,
    min: u64,
    max: u64,
    zeros: u64,
    /// How many sampled values were addresses the host could read.
    readable: u64,
    /// The first few distinct values, in order of appearance.
    distinct: Vec<u64>,
    more_distinct: bool,
}

impl RegisterShape {
    fn record(&mut self, value: u64, readable: bool) {
        if self.samples == 0 {
            self.min = value;
            self.max = value;
        } else {
            self.min = self.min.min(value);
            self.max = self.max.max(value);
        }
        self.samples += 1;
        self.zeros += u64::from(value == 0);
        self.readable += u64::from(readable);
        if !self.distinct.contains(&value) {
            if self.distinct.len() < DISTINCT_LIMIT {
                self.distinct.push(value);
            } else {
                self.more_distinct = true;
            }
        }
    }

    /// A one-word reading of the register, for the report.
    fn kind(&self) -> &'static str {
        if self.samples == 0 {
            "unseen"
        } else if self.distinct.len() == 1 && !self.more_distinct {
            if self.readable == self.samples {
                "constant-address"
            } else {
                "constant"
            }
        } else if self.readable == self.samples {
            "address"
        } else if self.readable + self.zeros == self.samples && self.readable > 0 {
            "address-or-null"
        } else if self.max < LOWEST_ADDRESS {
            "small"
        } else if self.readable == 0 {
            "number"
        } else {
            "mixed"
        }
    }
}

/// A value read as a float, when that reading looks like a number a program
/// would pass on purpose. Integers and addresses read as floats are tiny,
/// huge or not numbers at all, and printing those would bury the real ones.
fn plausible_float(value: f64) -> Option<f64> {
    let magnitude = value.abs();
    (value == 0.0 || (1e-6..1e9).contains(&magnitude)).then_some(value)
}

fn float_readings(values: &[u64]) -> Option<(String, &'static str)> {
    let join = |readings: Vec<Option<f64>>| {
        readings
            .into_iter()
            .map(|reading| reading.map(|value| format!("{value}")))
            .collect::<Option<Vec<_>>>()
            .map(|list| list.join(" "))
    };
    // A single-precision value occupies the low 32 bits with the rest zero.
    let singles = values
        .iter()
        .map(|value| {
            (*value <= u64::from(u32::MAX))
                .then(|| plausible_float(f64::from(f32::from_bits(*value as u32))))
                .flatten()
        })
        .collect();
    if let Some(list) = join(singles) {
        return Some((list, "f32"));
    }
    let doubles = values
        .iter()
        .map(|value| plausible_float(f64::from_bits(*value)))
        .collect();
    join(doubles).map(|list| (list, "f64"))
}

impl fmt::Display for RegisterShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind())?;
        if self.samples == 0 {
            return Ok(());
        }
        // Addresses identify objects in one run and mean nothing outside it,
        // so they are counted and never printed.
        if self.readable > 0 {
            let count = self.distinct.len();
            let more = if self.more_distinct { "+" } else { "" };
            write!(f, " distinct={count}{more}")?;
            if self.readable < self.samples {
                write!(f, " readable={}/{}", self.readable, self.samples)?;
            }
            if self.zeros > 0 {
                write!(f, " zero={}/{}", self.zeros, self.samples)?;
            }
            return Ok(());
        }
        // A value too large for 32 bits that the host could not read may
        // still be an address (of memory the host guards), so it is shown
        // as "wide" and only its float reading, if plausible, is printed.
        if self.max <= WIDE {
            write!(f, " min={:#x} max={:#x}", self.min, self.max)?;
        }
        f.write_str(" values=[")?;
        for (index, value) in self.distinct.iter().enumerate() {
            if index > 0 {
                f.write_char(' ')?;
            }
            if *value <= WIDE {
                write!(f, "{value:#x}")?;
            } else {
                f.write_str("wide")?;
            }
        }
        f.write_str(if self.more_distinct { " ...]" } else { "]" })?;
        let all_zero = self.max == 0;
        if !all_zero {
            if let Some((list, kind)) = float_readings(&self.distinct) {
                write!(f, " {kind}=[{list}]")?;
            }
        }
        Ok(())
    }
}

/// Bytes looked at behind an address argument.
pub const POINTEE_BYTES: usize = 64;
const POINTEE_WORDS: usize = POINTEE_BYTES / 4;
const SHADER_BYTES: usize = 0x80;
const SHADER_WORDS: usize = SHADER_BYTES / 4;
const WORD_DISTINCT_LIMIT: usize = 4;

/// What one 32-bit word behind an address argument held across the sampled
/// calls, and whether the function changed it.
#[derive(Debug, Clone, Default, PartialEq)]
struct WordShape {
    min: u32,
    max: u32,
    zeros: u64,
    /// Samples where the word read as a plausible float and not as a small
    /// integer.
    floats: u64,
    distinct: Vec<u32>,
    more_distinct: bool,
    /// Samples where the word differed after the function returned.
    changed: u64,
}

impl WordShape {
    fn record(&mut self, value: u32, first: bool) {
        if first {
            self.min = value;
            self.max = value;
        } else {
            self.min = self.min.min(value);
            self.max = self.max.max(value);
        }
        self.zeros += u64::from(value == 0);
        let small = u64::from(value) < LOWEST_ADDRESS;
        // Stricter than for registers: memory holds arbitrary data, and a
        // random word reads as a tiny or huge float far more often than as
        // one in the range programs use for colours, sizes and factors.
        let reading = f64::from(f32::from_bits(value)).abs();
        let float = !small && (1e-3..1e7).contains(&reading);
        self.floats += u64::from(float);
        if !self.distinct.contains(&value) {
            if self.distinct.len() < WORD_DISTINCT_LIMIT {
                self.distinct.push(value);
            } else {
                self.more_distinct = true;
            }
        }
    }
}

/// What the memory behind one address argument looked like: the first 64
/// bytes, as sixteen 32-bit words and eight 64-bit words.
///
/// This is the one place the library looks at memory an argument points to.
/// It keeps per-word ranges and a few small or float values. Words that are
/// neither are reported only as "other", and nothing is kept verbatim.
#[derive(Debug, Clone, Default, PartialEq)]
struct PointeeShape {
    samples: u64,
    words: [WordShape; POINTEE_WORDS],
    /// Samples where a 64-bit word was itself a readable address.
    addresses: [u64; POINTEE_WORDS / 2],
    /// Samples where a 64-bit word was wider than 32 bits but within the
    /// range addresses come from, and not readable. It may be the address
    /// of memory the host guards, so neither half is ever printed.
    wides: [u64; POINTEE_WORDS / 2],
    /// Samples compared before and after the call.
    compared: u64,
}

fn read_pointee(host: Option<&Host>, address: u64) -> Option<[u8; POINTEE_BYTES]> {
    let host = host?;
    let read = host.read_memory?;
    if !(LOWEST_ADDRESS..ADDRESS_LIMIT).contains(&address) {
        return None;
    }
    let mut bytes = [0u8; POINTEE_BYTES];
    // SAFETY: the buffer is valid for the bytes requested; the host's
    // callback reports an inaccessible range instead of faulting.
    (unsafe { read(host.user, address, bytes.as_mut_ptr(), POINTEE_BYTES as u64) } == 0)
        .then_some(bytes)
}

fn word(bytes: &[u8; POINTEE_BYTES], index: usize) -> u32 {
    u32::from_le_bytes(
        bytes[index * 4..index * 4 + 4]
            .try_into()
            .expect("four bytes"),
    )
}

fn read_shader(host: Option<&Host>, address: u64) -> Option<[u8; SHADER_BYTES]> {
    let host = host?;
    let read = host.read_memory?;
    if !(LOWEST_ADDRESS..ADDRESS_LIMIT).contains(&address) {
        return None;
    }
    let mut bytes = [0u8; SHADER_BYTES];
    (unsafe { read(host.user, address, bytes.as_mut_ptr(), SHADER_BYTES as u64) } == 0)
        .then_some(bytes)
}

fn shader_word(bytes: &[u8; SHADER_BYTES], index: usize) -> u32 {
    u32::from_le_bytes(
        bytes[index * 4..index * 4 + 4]
            .try_into()
            .expect("four bytes"),
    )
}

#[derive(Debug, Clone, Default)]
struct ShaderShape {
    samples: u64,
    unreadable: u64,
    words: [WordShape; SHADER_WORDS],
    leading_nonzero: u64,
    repeating_header: u64,
    code_lengths: BTreeMap<(usize, &'static str), u64>,
    resolved_offsets: Vec<u64>,
    unresolved: Vec<(GpuAddressError, u64)>,
}

fn unresolved_reason(reason: GpuAddressError) -> &'static str {
    match reason {
        GpuAddressError::UnknownAddress => "no registered pool",
        GpuAddressError::OutsidePool { .. } => "outside registered pool",
    }
}

impl ShaderShape {
    fn record(&mut self, host: Option<&Host>, objects: &Objects, address: u64) {
        let resolution = match objects.resolve_gpu_address(address) {
            Ok(resolution) => resolution,
            Err(reason) => {
                self.unresolved.push((reason, address));
                return;
            }
        };
        self.resolved_offsets.push(resolution.offset);
        let Some(bytes) = read_shader(host, resolution.program_address) else {
            self.unreadable += 1;
            return;
        };
        self.samples += 1;
        let first = self.samples == 1;
        for (index, shape) in self.words.iter_mut().enumerate() {
            shape.record(shader_word(&bytes, index), first);
        }
        let leading = (0..SHADER_WORDS)
            .take_while(|&index| shader_word(&bytes, index) != 0)
            .count();
        self.leading_nonzero += leading as u64;
        self.repeating_header += u64::from((1..=8).any(|period| {
            (period..SHADER_WORDS)
                .all(|index| shader_word(&bytes, index) == shader_word(&bytes, index % period))
        }));
        let read = host.and_then(|host| host.read_memory);
        let pool_end = usize::try_from(resolution.remaining).unwrap_or(usize::MAX);
        let limit = pool_end.min(64 * 1024);
        let mut length = limit;
        let mut reason = if pool_end <= 64 * 1024 {
            "pool end"
        } else {
            "cap"
        };
        let mut chunk = [0u8; 256];
        'scan: for offset in (0x80..limit).step_by(chunk.len()) {
            let Some(read) = read else {
                reason = "unreadable";
                length = offset;
                break;
            };
            let Some(at) = resolution.program_address.checked_add(offset as u64) else {
                reason = "unreadable";
                length = offset;
                break;
            };
            let read_length = chunk.len().min(limit - offset);
            if unsafe {
                read(
                    host.unwrap().user,
                    at,
                    chunk.as_mut_ptr(),
                    read_length as u64,
                )
            } != 0
            {
                reason = "unreadable";
                length = offset;
                break;
            }
            for start in (0..read_length.saturating_sub(63)).step_by(4) {
                if start + 64 <= read_length
                    && chunk[start..start + 64].iter().all(|&byte| byte == 0)
                {
                    length = offset + start;
                    reason = "zero run";
                    break 'scan;
                }
            }
        }
        *self.code_lengths.entry((length, reason)).or_default() += 1;
    }

    fn write(&self, out: &mut String, offset: usize) {
        let _ = writeln!(
            out,
            "  shader +{offset:#04x} resolved={} unreadable={} unresolved={}",
            self.resolved_offsets.len(),
            self.unreadable,
            self.unresolved.len()
        );
        if !self.resolved_offsets.is_empty() {
            let _ = writeln!(
                out,
                "    pool-offsets=[{}]",
                self.resolved_offsets
                    .iter()
                    .map(|offset| format!("{offset:#x}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        if !self.code_lengths.is_empty() {
            let _ = writeln!(out, "    code-lengths:");
            for ((length, reason), count) in &self.code_lengths {
                let _ = writeln!(out, "      length={length} stop={reason} count={count}");
            }
        }
        for (reason, _) in &self.unresolved {
            let _ = writeln!(out, "    unresolved reason={}", unresolved_reason(*reason));
        }
        if self.samples == 0 {
            return;
        }
        let _ = writeln!(
            out,
            "    leading-nonzero-total={} repeating-header={}/{}",
            self.leading_nonzero, self.repeating_header, self.samples,
        );
        for (index, shape) in self.words.iter().enumerate() {
            let mut line = format!("    +{:#04x} ", index * 4);
            if shape.max == 0 {
                line.push_str("zero");
            } else if u64::from(shape.max) < LOWEST_ADDRESS {
                let _ = write!(
                    line,
                    "small min={:#x} max={:#x} values=[",
                    shape.min, shape.max
                );
                push_words(&mut line, shape, |value| format!("{value:#x}"));
            } else if shape.floats + shape.zeros == self.samples {
                line.push_str("f32");
            } else {
                line.push_str("other");
            }
            let _ = writeln!(out, "{line}");
        }
    }
}

impl PointeeShape {
    fn record(&mut self, host: Option<&Host>, bytes: &[u8; POINTEE_BYTES]) {
        let first = self.samples == 0;
        self.samples += 1;
        for (index, shape) in self.words.iter_mut().enumerate() {
            shape.record(word(bytes, index), first);
        }
        for index in 0..POINTEE_WORDS / 2 {
            let value =
                u64::from(word(bytes, index * 2)) | u64::from(word(bytes, index * 2 + 1)) << 32;
            if is_readable_address(host, value) {
                self.addresses[index] += 1;
            } else if (WIDE + 1..ADDRESS_LIMIT).contains(&value) {
                self.wides[index] += 1;
            }
        }
    }

    fn record_change(&mut self, before: &[u8; POINTEE_BYTES], after: &[u8; POINTEE_BYTES]) {
        self.compared += 1;
        for (index, shape) in self.words.iter_mut().enumerate() {
            shape.changed += u64::from(word(before, index) != word(after, index));
        }
    }

    fn write(&self, out: &mut String, register: usize) {
        let _ = writeln!(
            out,
            "    x{register} points to ({} sampled, {} compared after the call):",
            self.samples, self.compared
        );
        let mut index = 0;
        while index < POINTEE_WORDS {
            let quad = index / 2;
            let offset = index * 4;
            // A 64-bit word that was an address every time is shown as one.
            if index % 2 == 0 && self.addresses[quad] == self.samples {
                let changed = self.words[index].changed.max(self.words[index + 1].changed);
                let _ = writeln!(out, "      +{offset:#04x} address{}", changed_note(changed));
                index += 2;
                continue;
            }
            // A 64-bit word that was an address in some samples, or could
            // have been an unreadable one, is shown without values. The cost: a float followed
            // by a small non-zero integer is hidden the same way.
            if index % 2 == 0 && (self.wides[quad] > 0 || self.addresses[quad] > 0) {
                let changed = self.words[index].changed.max(self.words[index + 1].changed);
                let _ = writeln!(
                    out,
                    "      +{offset:#04x} wide in {}/{} (address in {}/{}){}",
                    self.wides[quad],
                    self.samples,
                    self.addresses[quad],
                    self.samples,
                    changed_note(changed)
                );
                index += 2;
                continue;
            }
            let shape = &self.words[index];
            let mut line = format!("      +{offset:#04x} ");
            if shape.max == 0 {
                line.push_str("zero");
            } else if u64::from(shape.max) < LOWEST_ADDRESS {
                let _ = write!(
                    line,
                    "small min={:#x} max={:#x} values=[",
                    shape.min, shape.max
                );
                push_words(&mut line, shape, |value| format!("{value:#x}"));
            } else if shape.floats + shape.zeros == self.samples {
                line.push_str("f32 values=[");
                push_words(&mut line, shape, |value| {
                    format!("{}", f32::from_bits(value))
                });
            } else {
                line.push_str("other");
            }
            line.push_str(&changed_note(shape.changed));
            let _ = writeln!(out, "{line}");
            index += 1;
        }
    }
}

fn changed_note(changed: u64) -> String {
    if changed > 0 {
        format!(" changed-by-call={changed}")
    } else {
        String::new()
    }
}

fn push_words(line: &mut String, shape: &WordShape, show: impl Fn(u32) -> String) {
    let shown: Vec<String> = shape.distinct.iter().map(|value| show(*value)).collect();
    line.push_str(&shown.join(" "));
    line.push_str(if shape.more_distinct { " ...]" } else { "]" });
}

/// The memory behind each address argument as it was when a call started,
/// kept until the same thread reports the return so the two can be compared.
#[derive(Clone, Copy)]
pub(crate) struct CallSnapshot {
    registers: Registers,
    pointees: [Option<(u64, [u8; POINTEE_BYTES])>; 8],
}

/// Distinct keys kept per function for keyed outputs.
const KEYED_OUTPUT_LIMIT: usize = 96;
/// Distinct answers kept under one key before the key stops being recorded.
const KEYED_ANSWER_LIMIT: usize = 3;

/// What one function's calls looked like.
#[derive(Debug, Clone, Default)]
pub struct FunctionShape {
    sampled_calls: u64,
    sampled_returns: u64,
    pointees: [PointeeShape; 8],
    shader_values: [ShaderShape; 2],
    /// For a call shaped like `f(object, selector, out)`: the selector in x1
    /// and the 32-bit value the call wrote through x2, one entry per
    /// distinct selector. This is how a query function's answers are
    /// learned.
    keyed_outputs: Vec<(u64, u32)>,
    keyed_outputs_full: bool,
    x: [RegisterShape; 8],
    d: [RegisterShape; 8],
    result_x0: RegisterShape,
    result_x1: RegisterShape,
    result_d0: RegisterShape,
}

fn is_readable_address(host: Option<&Host>, value: u64) -> bool {
    if !(LOWEST_ADDRESS..ADDRESS_LIMIT).contains(&value) {
        return false;
    }
    let Some(host) = host else {
        return false;
    };
    let Some(read) = host.read_memory else {
        return false;
    };
    let mut probe = [0u8; 8];
    // SAFETY: the buffer is valid for the eight bytes requested. The host
    // promised a callback that reports an inaccessible range with a non-zero
    // result instead of faulting.
    unsafe { read(host.user, value, probe.as_mut_ptr(), probe.len() as u64) == 0 }
}

impl FunctionShape {
    /// Sample a call. Returns what the address arguments pointed to, for
    /// comparison when the call returns, or `None` once sampling has stopped.
    pub(crate) fn record_call(
        &mut self,
        host: Option<&Host>,
        objects: &Objects,
        registers: &Registers,
    ) -> Option<CallSnapshot> {
        if self.sampled_calls >= SAMPLE_LIMIT {
            return None;
        }
        self.sampled_calls += 1;
        let mut snapshot = CallSnapshot {
            registers: *registers,
            pointees: [None; 8],
        };
        for (index, value) in registers.x.into_iter().enumerate() {
            let bytes = read_pointee(host, value);
            // A full read can fail near the end of a mapping where a short
            // one succeeds, so readability is still asked separately.
            let readable = bytes.is_some() || is_readable_address(host, value);
            self.x[index].record(value, readable);
            if let Some(bytes) = bytes {
                self.pointees[index].record(host, &bytes);
                if index == 2 {
                    for (slot, start) in [0usize, 12].into_iter().enumerate() {
                        let address = u64::from(word(&bytes, start / 4))
                            | u64::from(word(&bytes, start / 4 + 1)) << 32;
                        self.shader_values[slot].record(host, objects, address);
                    }
                }
                snapshot.pointees[index] = Some((value, bytes));
            }
        }
        for (shape, value) in self.d.iter_mut().zip(registers.d) {
            shape.record(value, false);
        }
        Some(snapshot)
    }

    pub(crate) fn record_return(
        &mut self,
        host: Option<&Host>,
        registers: &Registers,
        snapshot: Option<&CallSnapshot>,
    ) {
        if let Some(snapshot) = snapshot {
            for (index, pointee) in snapshot.pointees.iter().enumerate() {
                let Some((address, before)) = pointee else {
                    continue;
                };
                if let Some(after) = read_pointee(host, *address) {
                    self.pointees[index].record_change(before, &after);
                    if index == 2 && word(before, 0) != word(&after, 0) {
                        // An answer that is itself an address, or could be
                        // the low half of one, is not kept.
                        let written = u64::from(word(&after, 0));
                        if !is_readable_address(host, written) && written < 1 << 24 {
                            self.record_keyed_output(snapshot.registers.x[1], word(&after, 0));
                        }
                    }
                }
            }
        }
        if self.sampled_returns >= SAMPLE_LIMIT {
            return;
        }
        self.sampled_returns += 1;
        let x0 = registers.x[0];
        self.result_x0.record(x0, is_readable_address(host, x0));
        let x1 = registers.x[1];
        self.result_x1.record(x1, is_readable_address(host, x1));
        self.result_d0.record(registers.d[0], false);
    }

    fn record_keyed_output(&mut self, key: u64, value: u32) {
        if key >= LOWEST_ADDRESS {
            return;
        }
        let same_key = self.keyed_outputs.iter().filter(|(k, _)| *k == key).count();
        if same_key > 0 {
            // A differing answer under the same key is worth a note, up to a
            // point: a key that answers differently every time is not a
            // query, and its answers are not kept.
            let seen = self
                .keyed_outputs
                .iter()
                .any(|(k, v)| *k == key && *v == value);
            if !seen && same_key < KEYED_ANSWER_LIMIT && !self.keyed_outputs_full {
                self.keyed_outputs.push((key, value));
            }
            return;
        }
        if self.keyed_outputs.len() < KEYED_OUTPUT_LIMIT {
            self.keyed_outputs.push((key, value));
        } else {
            self.keyed_outputs_full = true;
        }
    }

    /// Number of calls sampled for this function, capped by the observation limit.
    pub fn sampled_calls(&self) -> u64 {
        self.sampled_calls
    }

    /// Write the report for one function. Registers that never changed from
    /// call to call are still listed: a register a function does not use
    /// holds whatever the caller left in it, and only the reader can tell a
    /// constant argument from a leftover.
    pub(crate) fn write(&self, out: &mut String, name: &str, calls: u64) {
        let _ = writeln!(
            out,
            "{name} calls={calls} sampled={} returns_sampled={}",
            self.sampled_calls, self.sampled_returns
        );
        for (index, shape) in self.x.iter().enumerate() {
            let _ = writeln!(out, "  x{index} {shape}");
            if self.pointees[index].samples > 0 {
                self.pointees[index].write(out, index);
            }
        }
        for (index, shape) in self.d.iter().enumerate() {
            let _ = writeln!(out, "  d{index} {shape}");
        }
        if name == "ProgramSetShaders" || name == "nvnProgramSetShaders" {
            self.shader_values[0].write(out, 0);
            self.shader_values[1].write(out, 0x30);
        }
        if !self.keyed_outputs.is_empty() {
            let mut line = String::from("  x1 -> *x2 after call:");
            for (key, value) in &self.keyed_outputs {
                let _ = write!(line, " {key:#x}={value:#x}");
            }
            if self.keyed_outputs_full {
                line.push_str(" ...");
            }
            let _ = writeln!(out, "{line}");
        }
        if self.sampled_returns > 0 {
            let _ = writeln!(out, "  result x0 {}", self.result_x0);
            let _ = writeln!(out, "  result x1 {}", self.result_x1);
            let _ = writeln!(out, "  result d0 {}", self.result_d0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;

    /// A host whose readable memory is one range.
    unsafe extern "C" fn read(_user: *mut c_void, address: u64, out: *mut u8, size: u64) -> i32 {
        if (0x10_0000..0x20_0000).contains(&address) {
            // SAFETY: the caller passes a buffer of `size` bytes.
            unsafe { std::ptr::write_bytes(out, 0, size as usize) };
            0
        } else {
            1
        }
    }

    fn host() -> Host {
        Host {
            user: std::ptr::null_mut(),
            read_memory: Some(read),
            write_memory: None,
            present: None,
            render_scale: 1.0,
            wait_vblank: None,
            vulkan: std::ptr::null(),
        }
    }

    #[test]
    fn registers_are_classified_by_what_they_held() {
        let host = host();
        let mut shape = FunctionShape::default();
        for call in 0..10u64 {
            let mut registers = Registers::default();
            registers.x[0] = 0x10_0000 + call * 0x100; // different objects
            registers.x[1] = call % 3; // a small number
            registers.x[2] = 7; // never changes
            registers.x[3] = if call % 2 == 0 { 0 } else { 0x10_8000 }; // optional pointer
            registers.x[4] = 0xdead_beef_0000 + call; // wide, not readable
            registers.d[0] = u64::from(1.5f32.to_bits());
            shape.record_call(Some(&host), &Objects::new(), &registers);
        }
        assert_eq!(shape.x[0].kind(), "address");
        assert_eq!(shape.x[1].kind(), "small");
        assert_eq!(shape.x[2].kind(), "constant");
        assert_eq!(shape.x[3].kind(), "address-or-null");
        assert_eq!(shape.x[4].kind(), "number");
        assert_eq!(shape.d[0].kind(), "constant");

        let mut text = String::new();
        shape.write(&mut text, "someFunction", 10);
        assert!(
            text.starts_with("someFunction calls=10 sampled=10 returns_sampled=0\n"),
            "{text}"
        );
        assert!(
            text.contains("  x1 small min=0x0 max=0x2 values=[0x0 0x1 0x2]"),
            "{text}"
        );
        assert!(
            text.contains(
                "  d0 constant min=0x3fc00000 max=0x3fc00000 values=[0x3fc00000] f32=[1.5]"
            ),
            "{text}"
        );
        // Small integers are not offered as floats.
        assert!(
            text.contains("  x2 constant min=0x7 max=0x7 values=[0x7]\n"),
            "{text}"
        );
        // Addresses are counted, never printed.
        assert!(text.contains("  x0 address distinct=6+\n"), "{text}");
        assert!(
            text.contains("  x3 address-or-null distinct=2 readable=5/10 zero=5/10\n"),
            "{text}"
        );
        assert!(!text.contains("0x100"), "{text}");
        assert!(
            text.contains("  x4 number values=[wide wide wide wide wide wide ...]\n"),
            "{text}"
        );
        assert!(!text.contains("dead"), "{text}");
        assert!(!text.contains("result"), "{text}");
    }

    /// A host with one readable page whose contents the test controls.
    static PAGE: std::sync::Mutex<[u8; 256]> = std::sync::Mutex::new([0; 256]);
    /// Held for the whole of each test that uses PAGE, since tests run in
    /// parallel.
    static PAGE_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());
    const PAGE_BASE: u64 = 0x40_0000;

    unsafe extern "C" fn read_page(
        _user: *mut c_void,
        address: u64,
        out: *mut u8,
        size: u64,
    ) -> i32 {
        let page = PAGE.lock().unwrap();
        let Some(offset) = address.checked_sub(PAGE_BASE) else {
            return 1;
        };
        let (offset, size) = (offset as usize, size as usize);
        if offset + size > page.len() {
            return 1;
        }
        // SAFETY: the caller passes a buffer of `size` bytes.
        unsafe { std::ptr::copy_nonoverlapping(page[offset..].as_ptr(), out, size) };
        0
    }

    #[test]
    fn memory_behind_an_address_is_classified_and_changes_are_noticed() {
        let _serial = PAGE_TEST.lock().unwrap_or_else(|p| p.into_inner());
        let host = Host {
            user: std::ptr::null_mut(),
            read_memory: Some(read_page),
            write_memory: None,
            present: None,
            render_scale: 1.0,
            wait_vblank: None,
            vulkan: std::ptr::null(),
        };
        let mut shape = FunctionShape::default();
        for call in 0..3u32 {
            {
                let mut page = PAGE.lock().unwrap();
                page.fill(0);
                page[0..4].copy_from_slice(&0.5f32.to_bits().to_le_bytes());
                page[4..8].copy_from_slice(&0.25f32.to_bits().to_le_bytes());
                // A pointer back into the page, and two words that are
                // neither small nor floats.
                page[8..16].copy_from_slice(&(PAGE_BASE + 0x80).to_le_bytes());
                page[16..20].copy_from_slice(&0xdead_beefu32.to_le_bytes());
                page[20..24].copy_from_slice(&0x7fc0_cafeu32.to_le_bytes());
                // Something address-shaped that cannot be read: its low half
                // would read as a float and must not be printed.
                page[32..40].copy_from_slice(&0x16_c0d2_5e8bu64.to_le_bytes());
                page[40..44].copy_from_slice(&(call + 1).to_le_bytes());
                // An optional address: present in two calls, null in one.
                if call < 2 {
                    page[48..56].copy_from_slice(&(PAGE_BASE + 0x40).to_le_bytes());
                }
            }
            let mut registers = Registers::default();
            registers.x[1] = PAGE_BASE;
            registers.x[2] = PAGE_BASE + 0x40;
            let snapshot = shape.record_call(Some(&host), &Objects::new(), &registers);
            // The function writes an output into the fourth word, and the
            // answer to a query (x1 as selector) through x2.
            let mut page = PAGE.lock().unwrap();
            page[12 + 12..12 + 16].copy_from_slice(&7u32.to_le_bytes());
            page[0x40..0x44].copy_from_slice(&(0x1000 + call).to_le_bytes());
            drop(page);
            shape.record_return(Some(&host), &Registers::default(), snapshot.as_ref());
        }
        let mut text = String::new();
        shape.write(&mut text, "someFunction", 3);
        for expected in [
            "    x1 points to (3 sampled, 3 compared after the call):\n",
            "      +0x00 f32 values=[0.5]\n",
            "      +0x04 f32 values=[0.25]\n",
            "      +0x08 address\n",
            "      +0x10 other\n",
            "      +0x14 other\n",
            "      +0x18 zero changed-by-call=3\n",
            "      +0x20 wide in 3/3 (address in 0/3)\n",
            "      +0x28 small min=0x1 max=0x3 values=[0x1 0x2 0x3]\n",
            "      +0x30 wide in 0/3 (address in 2/3)\n",
        ] {
            assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
        }
        // x1 was an address, so no keyed output is recorded for it.
        assert!(!text.contains("x1 -> *x2"), "{text}");
        for hidden in ["dead", "cafe", "-6.5", "c0d2"] {
            assert!(!text.contains(hidden), "{hidden} leaked in:\n{text}");
        }
    }

    #[test]
    fn shader_observation_resolves_pool_gpu_addresses_shape_only() {
        let _serial = PAGE_TEST.lock().unwrap_or_else(|p| p.into_inner());
        {
            let mut page = PAGE.lock().unwrap();
            page.fill(0);
            page[0..8].copy_from_slice(&(PAGE_BASE + 0x80).to_le_bytes());
            page[48..56].copy_from_slice(&(PAGE_BASE + 0x90).to_le_bytes());
            page[0x80..0x84].copy_from_slice(&1u32.to_le_bytes());
            page[0x84..0x88].copy_from_slice(&2u32.to_le_bytes());
        }
        let objects = Objects::new();
        objects.put(
            7,
            crate::api::Object::MemoryPool {
                device: 0,
                flags: 0,
                storage: PAGE_BASE,
                size: 0x100,
                gpu_address: None,
                observed_gpu_address: None,
            },
        );
        let host = Host {
            user: std::ptr::null_mut(),
            read_memory: Some(read_page),
            write_memory: None,
            present: None,
            render_scale: 1.0,
            wait_vblank: None,
            vulkan: std::ptr::null(),
        };
        let mut shape = FunctionShape::default();
        let mut registers = Registers::default();
        registers.x[2] = PAGE_BASE;
        shape.record_call(Some(&host), &objects, &registers);
        let mut text = String::new();
        shape.write(&mut text, "ProgramSetShaders", 1);
        assert!(
            text.contains("shader +0x00 resolved=1 unreadable=0 unresolved=0"),
            "{text}"
        );
        assert!(text.contains("pool-offsets=[0x80]"), "{text}");
        assert!(text.contains("leading-nonzero-total=2"), "{text}");
        assert!(!text.contains("400080"), "{text}");
    }

    #[test]
    fn query_answers_are_kept_per_selector() {
        let _serial = PAGE_TEST.lock().unwrap_or_else(|p| p.into_inner());
        let host = Host {
            user: std::ptr::null_mut(),
            read_memory: Some(read_page),
            write_memory: None,
            present: None,
            render_scale: 1.0,
            wait_vblank: None,
            vulkan: std::ptr::null(),
        };
        let mut shape = FunctionShape::default();
        for selector in [3u64, 9, 3] {
            PAGE.lock().unwrap().fill(0);
            let mut registers = Registers::default();
            registers.x[1] = selector;
            registers.x[2] = PAGE_BASE;
            let snapshot = shape.record_call(Some(&host), &Objects::new(), &registers);
            PAGE.lock().unwrap()[0..4].copy_from_slice(&(selector as u32 * 100).to_le_bytes());
            shape.record_return(Some(&host), &Registers::default(), snapshot.as_ref());
        }
        let mut text = String::new();
        shape.write(&mut text, "query", 3);
        assert!(
            text.contains("  x1 -> *x2 after call: 0x3=0x12c 0x9=0x384\n"),
            "{text}"
        );
    }

    #[test]
    fn sampling_stops_at_the_limit() {
        let mut shape = FunctionShape::default();
        for _ in 0..SAMPLE_LIMIT + 10 {
            shape.record_call(None, &Objects::new(), &Registers::default());
            shape.record_return(None, &Registers::default(), None);
        }
        assert_eq!(shape.sampled_calls(), SAMPLE_LIMIT);
        assert_eq!(shape.sampled_returns, SAMPLE_LIMIT);
    }

    #[test]
    fn without_a_host_nothing_counts_as_an_address() {
        let mut shape = FunctionShape::default();
        let mut registers = Registers::default();
        registers.x[0] = 0x10_0000;
        shape.record_call(None, &Objects::new(), &registers);
        assert_eq!(shape.x[0].kind(), "constant");
    }
}
