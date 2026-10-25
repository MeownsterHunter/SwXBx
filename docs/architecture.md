# Architecture

SwXBx separates host integration, graphics state, shader translation, and GPU
execution. Each boundary has one owner and one direction of data flow.

## Host adapter

The host adapter is the only layer that knows how the embedding application
represents memory, windows, threads, and function entry points. It converts a
host call into `swxbx_registers`, invokes the runtime, and copies result
registers back. Guest addresses remain opaque integers outside memory
callbacks.

## API state

The API layer validates calls and converts untyped call arguments into typed
objects and immutable recorded commands. Object handles identify state owned
by an instance. A command buffer has one recording owner at a time and becomes
immutable when submitted.

## Shader boundary

Shader translation is injected through a callback. The callback receives a
stage token and bounded program bytes, and returns borrowed SPIR-V words.
SwXBx copies those words before releasing the callback result. Compiler caches,
source formats, and compiler-specific diagnostics remain outside the runtime.

## Execution

Submitted recordings enter an ordered execution queue. Worker-owned resources
perform pipeline construction without sharing mutable command state. Resource
updates and draw state are resolved before Vulkan submission. Unsupported or
invalid states produce explicit outcomes at the boundary where they are
detected.

## Memory

The host supplies memory callbacks. SwXBx mirrors only the ranges required by
live GPU resources and tracks dirty ranges explicitly. GPU resources own their
Vulkan allocation and synchronization state; guest addresses do not escape
into Vulkan objects.

## Presentation

Presentation has two adapters:

- Native Vulkan presentation uses a host-created surface and host-reported
  drawable extent.
- Callback presentation delivers an immutable RGBA snapshot when the host
  polls its presentation queue.

Both paths use the same completed render image. Presentation policy controls
queue depth and FIFO or mailbox delivery without changing draw semantics.

## Ownership summary

| Resource | Owner | Release boundary |
|---|---|---|
| Host callback context | Host, then instance after installation | Callback replacement or instance destruction |
| Guest memory | Host | Host-defined |
| Recorded command | Command buffer, then execution queue | Submission completion |
| SPIR-V module | Instance | Instance destruction |
| Vulkan resources | Instance backend | Instance destruction |
| Callback frame bytes | Presentation queue | Return from frame callback |
