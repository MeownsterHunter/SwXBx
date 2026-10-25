# Host contract

This document defines the responsibilities of an embedding host. The C ABI in
`include/swxbx.h` is authoritative for field layout and function signatures.

## Instance lifetime

The host creates an instance with callbacks that remain valid until
`swxbx_instance_destroy` returns. Destruction drains queued work and releases
installed callback contexts. The host must prevent new calls once destruction
starts.

## Function dispatch

Resolve names with `swxbx_function_lookup` or `swxbx_instance_request`.
Forward calls with `swxbx_instance_call`. Function identifiers are stable for
one library version; do not persist them across version changes.

Arguments use the register layout in `swxbx_registers`. The host forwards the
calling convention as observed and does not reinterpret individual function
signatures.

## Memory callbacks

All program-memory access crosses `read_memory` and `write_memory`. Callbacks
must validate the full range and report failure without partial access. They
may run from the caller thread or an execution worker. They must not reenter
the same SwXBx instance or wait for that instance to finish.

Memory referenced by submitted commands must remain valid and unchanged until
the relevant completion boundary. A host with shared address space may
implement callbacks as checked copies, but it must still preserve these
lifetime rules.

## Shader translator

Install one translator with `swxbx_instance_set_shader_translator`.

- The translate callback may run concurrently.
- A successful call returns a non-null, nonempty SPIR-V word range.
- Returned words are borrowed only for the callback transaction; SwXBx copies
  them before calling `release`.
- `release` runs once for each successful borrowed result.
- `destroy_user` runs once after replacement or instance destruction, after
  outstanding translations release their references.
- Clearing or replacing a translator does not reinterpret programs already
  registered with the instance.

Compiler file paths, cache formats, and compiler-specific object types must
not cross this interface.

## Presentation

For native presentation, the host supplies required Vulkan instance extension
names, a surface-creation callback, and current drawable dimensions. The host
owns the window; SwXBx owns the returned surface. Callbacks must not reenter
the instance.

For callback presentation, the host receives immutable RGBA bytes. Those bytes
remain valid only until the callback returns. The event loop should call
`swxbx_instance_poll_presentations` at its display cadence.

## Errors

Every exported C function contains failures at the ABI boundary.
`swxbx_last_error` returns the current thread's most recent diagnostic until
the next library call on that thread. Hosts must check returned status codes;
they must not infer success from unchanged output registers.

Unsupported operations and invalid arguments are distinct outcomes. Diagnostic
skip mode is intended for coverage analysis and must not be used when exact
rendering is required.
