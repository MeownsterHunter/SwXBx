//! Integrator misuse refused by the recording call. Source: the public contract.
//!
//! A command buffer call whose fault follows from its own arguments and the
//! commands recorded before it in the same buffer is refused by that call.
//! The call returns the status, the reason goes to `swxbx_last_error` on
//! the caller's thread, and nothing is recorded, so no device work starts for
//! a batch already known to be wrong.
//!
//! The facts kept here are the ones execution keeps for the same buffer:
//! execution starts every command buffer with fresh draw state, and so does
//! recording. Faults that depend on memory contents or on what swxbx itself
//! can do stay with execution.

use super::objects::{RecordedCommand, StateCommand, StateSettings as Settings};
use crate::{instance::InstanceState, keymap::KeyHasher, unsupported::refuse, Status};
use std::{collections::HashSet, fmt, hash::BuildHasherDefault};

/// The settings a vertex stream state may carry for execution to run it.
pub(super) const STREAM_SETTINGS: &[&str] = &["Stride", "Divisor"];
/// The settings a vertex attribute state may carry for execution to run it.
pub(super) const ATTRIBUTE_SETTINGS: &[&str] = &["Format", "StreamIndex"];

/// The last value of a named setting.
pub(super) fn value(settings: &Settings, name: &str) -> Result<[u64; 6], Status> {
    settings
        .iter()
        .rev()
        .find(|(key, _)| *key == name)
        .map(|(_, value)| *value)
        .ok_or(Status::Unimplemented)
}

/// The states a counted state binding supplies, when they are known: the
/// host-spaced snapshots, or else nothing for a count of zero and the first
/// object for a count of one.
pub(super) fn listed<S>(count: u64, first: Option<S>, spaced: Option<Vec<S>>) -> Option<Vec<S>> {
    spaced.or_else(|| match count {
        0 => Some(Vec::new()),
        1 => first.map(|settings| vec![settings]),
        _ => None,
    })
}

/// Whether every setting of every state is one execution can run.
pub(super) fn admits<'a>(states: impl IntoIterator<Item = &'a Settings>, allowed: &[&str]) -> bool {
    states
        .into_iter()
        .flatten()
        .all(|(name, _)| allowed.contains(name))
}

/// Why an attribute cannot fetch: the stream it reads.
pub(super) enum StreamFault {
    /// The bound stream state describes fewer streams.
    Undescribed { stream: u64, streams: usize },
    /// No vertex buffer is bound to the stream.
    Unbound { stream: u64 },
}

impl fmt::Display for StreamFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Undescribed { stream, streams } => write!(
                f,
                "draw: an attribute reads vertex stream {stream}, but the bound vertex stream \
                 state describes only {streams} streams"
            ),
            Self::Unbound { stream } => write!(
                f,
                "draw: an attribute reads vertex stream {stream}, which has no vertex buffer \
                 bound"
            ),
        }
    }
}

/// What the recorded commands of one buffer established so far.
#[derive(Default, Debug)]
pub(super) struct Facts {
    /// Streams with a vertex buffer bound.
    bound: HashSet<u64, BuildHasherDefault<KeyHasher>>,
    /// How many streams the bound stream state describes, when execution
    /// could run it.
    streams: Option<usize>,
    /// The stream each attribute of the bound attribute state reads, when
    /// execution could run it.
    attributes: Option<Vec<u64>>,
}

impl Facts {
    /// Take in a command about to be recorded, or refuse it.
    pub(super) fn admit(
        &mut self,
        instance: &InstanceState,
        command: &RecordedCommand,
    ) -> Result<(), Status> {
        match command {
            RecordedCommand::State(StateCommand::BindVertexBuffer { index, size, .. }) => {
                // A zero range unbinds the stream.
                if *size == 0 {
                    self.bound.remove(index);
                } else {
                    self.bound.insert(*index);
                }
            }
            RecordedCommand::State(StateCommand::BindStates {
                kind,
                count,
                first_settings,
                experiment_settings,
                ..
            }) if matches!(*kind, "VertexStreamState" | "VertexAttribState") => {
                let states = listed(
                    *count,
                    first_settings.as_ref(),
                    experiment_settings.as_ref().map(|s| s.iter().collect()),
                );
                if *kind == "VertexStreamState" {
                    self.streams = states
                        .filter(|s| admits(s.iter().copied(), STREAM_SETTINGS))
                        .map(|s| s.len());
                } else {
                    self.attributes = states
                        .filter(|s| admits(s.iter().copied(), ATTRIBUTE_SETTINGS))
                        .and_then(|s| {
                            s.iter()
                                .map(|a| value(a, "StreamIndex").ok().map(|v| v[0]))
                                .collect()
                        });
                }
            }
            RecordedCommand::DrawArrays { .. } | RecordedCommand::DrawElementsBaseVertex { .. } => {
                self.fetch()?
            }
            RecordedCommand::DrawArraysInstanced {
                count, instances, ..
            } if *count != 0 && *instances != 0 => self.fetch()?,
            RecordedCommand::Operation { kind, arguments } => {
                let (count, instances) = match *kind {
                    "DrawArraysInstanced" => (arguments[2], arguments[4]),
                    "DrawElementsInstanced" => (arguments[2], arguments[6]),
                    _ => return Ok(()),
                };
                if count != 0 && instances != 0 {
                    self.fetch()?
                }
            }
            #[cfg(feature = "vulkan")]
            RecordedCommand::CopyTextureToTexture { arguments, .. } => copy(instance, *arguments)?,
            _ => {}
        }
        #[cfg(not(feature = "vulkan"))]
        let _ = instance;
        Ok(())
    }

    /// Refuse a draw whose attributes read a stream nothing supplies.
    fn fetch(&self) -> Result<(), Status> {
        let (Some(streams), Some(attributes)) = (self.streams, &self.attributes) else {
            return Ok(());
        };
        for &stream in attributes {
            let fault = if stream >= streams as u64 {
                StreamFault::Undescribed { stream, streams }
            } else if !self.bound.contains(&stream) {
                StreamFault::Unbound { stream }
            } else {
                continue;
            };
            return Err(refuse(Status::BadArgument, fault));
        }
        Ok(())
    }
}

/// What the recording call needs to judge a texture copy. It mirrors the
/// backend's copy decoder and image rules so the caller's thread never takes
/// the device lock.
#[cfg(feature = "vulkan")]
#[derive(Default)]
pub(crate) struct CopyRules {
    pub decoder: Option<crate::gpu::image_layout::CopyDecoder>,
    pub images: crate::gpu::image_layout::ImageContract,
}

/// Refuse a texture copy whose arguments, textures or regions are wrong.
/// Blits, and images no image rule describes, are left to execution: those
/// are swxbx's own limits.
#[cfg(feature = "vulkan")]
fn copy(instance: &InstanceState, arguments: [u64; 8]) -> Result<(), Status> {
    use crate::gpu::{image_layout::CopyOperation, images};
    let rules = instance.copy_rules.load();
    let Some(decoder) = &rules.decoder else {
        return Ok(());
    };
    let Some(operation) = decoder(arguments) else {
        return Err(refuse(
            Status::BadArgument,
            "copy texture to texture: the installed copy decoder does not describe these \
             arguments",
        ));
    };
    let (source, destination) = match operation {
        CopyOperation::Copy {
            source,
            destination,
            ..
        }
        | CopyOperation::Blit {
            source,
            destination,
            ..
        } => (source, destination),
    };
    let descriptions = [source, destination]
        .map(|texture| instance.objects.texture_description(texture).ok_or(texture));
    for description in &descriptions {
        if let Err(texture) = description {
            return Err(refuse(
                Status::BadArgument,
                format_args!("copy texture to texture: {texture:#x} is not a texture"),
            ));
        }
    }
    let CopyOperation::Copy { from, to, .. } = operation else {
        return Ok(());
    };
    let fault = images::regions_fault(destination, source, from, to).or_else(|| {
        let [Ok(s), Ok(d)] = &descriptions else {
            return None;
        };
        let (s, d) = (rules.images.resolve(s)?, rules.images.resolve(d)?);
        images::footprint_fault(
            images::Footprint::of(&s.descriptor),
            images::Footprint::of(&d.descriptor),
            from,
            to,
        )
    });
    match fault {
        Some(reason) => Err(refuse(
            Status::BadArgument,
            format_args!("copy texture {source:#x} to {destination:#x}: {reason}"),
        )),
        None => Ok(()),
    }
}
