//! What a submitted batch does at a command swxbx cannot run yet.
//!
//! The default stops the batch at that command and reports its status.
//! `SWXBX_UNSUPPORTED_COMMANDS=skip` selects a
//! diagnostic mode instead: the command is logged, counted and skipped, and
//! the batch carries on so that later syncs still signal.

use crate::{
    api::{RecordedCommand, StateCommand},
    functions, Status,
};
use std::{collections::BTreeMap, fmt, sync::Mutex};

/// Environment variable read once when an instance is created.
const VARIABLE: &str = "SWXBX_UNSUPPORTED_COMMANDS";

/// A summary is printed after this many submissions while skips exist.
const SUMMARY_INTERVAL: u64 = 1000;

/// The instance's choice for commands it cannot run.
pub(crate) enum UnsupportedCommands {
    /// The batch ends at the command and the status is reported.
    Stop,
    /// The command is recorded and execution continues with the next one.
    Skip(Mutex<Ledger>),
}

/// The skippable ways a command can fail. Every other status means state
/// can no longer be trusted and always stops.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Outcome {
    /// swxbx has no execution for this command or argument case yet.
    Unimplemented,
    /// swxbx rejected arguments the program really sent. This may be a
    /// swxbx defect rather than a missing feature, so it is kept apart.
    BadArgument,
}

impl Outcome {
    fn of(status: Status) -> Option<Self> {
        match status {
            Status::Unimplemented => Some(Self::Unimplemented),
            Status::BadArgument => Some(Self::BadArgument),
            Status::Ok | Status::BadFunction | Status::InternalError => None,
        }
    }
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Unimplemented => "unimplemented",
            Self::BadArgument => "bad-argument",
        })
    }
}

/// Skipped commands by kind and outcome, and the submissions seen.
#[derive(Default)]
pub(crate) struct Ledger {
    skipped: BTreeMap<(&'static str, Outcome), u64>,
    submissions: u64,
}

impl fmt::Display for Ledger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "swxbx: skipped commands after {} submissions, {} kinds",
            self.submissions,
            self.skipped.len()
        )?;
        writeln!(f, "swxbx: {:>10}  {:<13}  command", "count", "outcome")?;
        for ((command, outcome), count) in &self.skipped {
            writeln!(f, "swxbx: {count:>10}  {outcome:<13}  {command}")?;
        }
        Ok(())
    }
}

impl UnsupportedCommands {
    /// Read the choice from the environment. Unset means `Stop`.
    ///
    /// # Panics
    /// On any value other than `stop` or `skip`. Instance creation reports
    /// the message through `swxbx_last_error`.
    pub fn from_env() -> Self {
        match std::env::var(VARIABLE) {
            Err(std::env::VarError::NotPresent) => Self::Stop,
            Ok(value) if value == "stop" => Self::Stop,
            Ok(value) if value == "skip" => {
                eprintln!("swxbx: {VARIABLE}=skip, unsupported commands are skipped");
                Self::Skip(Mutex::default())
            }
            Ok(value) => panic!("{VARIABLE} must be stop or skip, not {value:?}"),
            Err(std::env::VarError::NotUnicode(value)) => {
                panic!("{VARIABLE} must be stop or skip, not {value:?}")
            }
        }
    }

    /// Settle the status of one executed command. `Err` ends the batch with
    /// that status; `Ok` continues with the next command.
    pub fn settle(&self, command: &RecordedCommand, status: Status) -> Result<(), Status> {
        if status == Status::Ok {
            return Ok(());
        }
        let name = command_name(command);
        let (Self::Skip(ledger), Some(outcome)) = (self, Outcome::of(status)) else {
            // The batch's status belongs to the batch, which already
            // returned to the program, so it is reported here.
            eprintln!("swxbx: batch stopped at {name} ({status:?})");
            return Err(status);
        };
        let mut ledger = ledger.lock().unwrap_or_else(|p| p.into_inner());
        let count = ledger.skipped.entry((name, outcome)).or_default();
        if *count == 0 {
            eprintln!("swxbx: skipped {name} ({outcome}), first seen: {command:?}");
        }
        *count += 1;
        Ok(())
    }

    /// Count one submission and print the summary at each interval.
    pub fn submitted(&self) {
        if let Self::Skip(ledger) = self {
            let mut ledger = ledger.lock().unwrap_or_else(|p| p.into_inner());
            ledger.submissions += 1;
            if ledger.submissions % SUMMARY_INTERVAL == 0 && !ledger.skipped.is_empty() {
                eprint!("{ledger}");
            }
        }
    }

    /// Print the final summary when the instance is destroyed.
    pub fn finish(&self) {
        if let Self::Skip(ledger) = self {
            eprint!("{}", ledger.lock().unwrap_or_else(|p| p.into_inner()));
        }
    }
}

fn command_name(command: &RecordedCommand) -> &'static str {
    match command {
        RecordedCommand::Operation { kind, .. } => kind,
        RecordedCommand::ReportCounter { .. } => "ReportCounter",
        RecordedCommand::ResetCounter(_) => "ResetCounter",
        RecordedCommand::SetRenderTargets { .. } => "SetRenderTargets",
        RecordedCommand::BindProgram(_) => "BindProgram",
        RecordedCommand::BindState { kind, .. } => kind,
        RecordedCommand::ClearColor { .. } => "ClearColor",
        RecordedCommand::ClearDepthStencil { .. } => "ClearDepthStencil",
        RecordedCommand::CopyBufferToTexture { .. } => "CopyBufferToTexture",
        RecordedCommand::CopyTextureToTexture { .. } => "CopyTextureToTexture",
        RecordedCommand::SetViewport(_) => "SetViewport",
        RecordedCommand::SetScissor(_) => "SetScissor",
        RecordedCommand::SetDepthRange(_) => "SetDepthRange",
        RecordedCommand::DrawArrays { .. } => "DrawArrays",
        RecordedCommand::DrawArraysInstanced { .. } => "DrawArraysInstanced",
        RecordedCommand::DrawElementsBaseVertex { .. } => "DrawElementsBaseVertex",
        RecordedCommand::State(state) => match state {
            StateCommand::BindState { kind, .. } => kind,
            StateCommand::BindStates { .. } => "BindStates",
            StateCommand::SetDescriptorPool { .. } => "SetDescriptorPool",
            StateCommand::BindSamplerReference { .. } => "BindSamplerReference",
            StateCommand::FenceSync { .. } => "FenceSync",
            StateCommand::SaveZCullData { .. } => "SaveZCullData",
            StateCommand::RestoreZCullData { .. } => "RestoreZCullData",
            StateCommand::Stencil { .. } => "Stencil",
            StateCommand::PolygonOffset(_) => "PolygonOffset",
            StateCommand::Barrier(_) => "Barrier",
            StateCommand::TiledCacheAction(_) => "TiledCacheAction",
            StateCommand::BindUniformBuffer { .. } => "BindUniformBuffer",
            StateCommand::BindVertexBuffer { .. } => "BindVertexBuffer",
            StateCommand::BindHandle { .. } => "BindHandle",
            StateCommand::ClearBuffer { .. } => "ClearBuffer",
            StateCommand::DispatchCompute(_) => "DispatchCompute",
        },
        RecordedCommand::Raw { function, .. } => {
            functions::name(functions::FunctionId(*function)).unwrap_or("Raw")
        }
    }
}
