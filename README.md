<p align="center">
  <img src="assets/swxbx-mark.png" width="180" alt="SwXBx logo">
</p>

# SwXBx

SwXBx is a host-integrated graphics runtime. It accepts an explicit stream of
graphics API calls, translates host-supplied shader programs to SPIR-V, owns
the Vulkan resources needed to execute those calls, and presents completed
frames through either a native surface or a host callback.

The library is designed as a replaceable module. It does not load programs,
manage a CPU, provide operating-system services, or bundle a shader compiler.
The embedding host owns those responsibilities and connects them through
stable callback contracts.

## Integration boundaries

- **Function dispatch:** Resolve exported function names once, then forward
  calls by numeric identifier with a fixed register block.
- **Guest memory:** The runtime reads and writes program memory only through
  host callbacks. It never assumes that guest pointers are host pointers.
- **Shader translation:** The host provides a thread-safe callback that turns
  bounded program bytes into SPIR-V words. SwXBx copies the returned words.
- **Presentation:** The host chooses a Vulkan surface or an RGBA frame
  callback. The native window and event loop remain host-owned.
- **Lifetime:** An instance owns its queues, pipelines, GPU resources, and
  background workers. Destroying the instance drains outstanding work.

The complete ABI is in [`include/swxbx.h`](include/swxbx.h). See
[`docs/host-contract.md`](docs/host-contract.md) before implementing an
adapter and [`docs/architecture.md`](docs/architecture.md) for internal
ownership boundaries.

## Build

```sh
cargo build -p swxbx --features vulkan
cargo test -p swxbx --all-features
```

The crate builds as a Rust library, shared library, and static library. The
repository also provides a C header, CMake package file, and pkg-config file.

## Minimal host sequence

1. Compare `swxbx_host_interface_version()` with
   `SWXBX_HOST_INTERFACE_VERSION`.
2. Fill `swxbx_host` with memory and optional presentation callbacks.
3. Create one `swxbx_instance`.
4. Install a shader translator if the workload submits shaders.
5. Resolve API functions by name and forward calls through
   `swxbx_instance_call`.
6. Poll callback presentation from the host event loop when required.
7. Destroy the instance after all guest threads stop using it.

## Stability

The host-interface version changes when an existing C declaration, layout, or
function identifier changes meaning. Additive Rust APIs may evolve before a
stable release. Unsupported operations fail explicitly; they are never
silently approximated unless the host enables diagnostic skip mode.

## License

SwXBx is available under the licenses described in
[`LICENSE.md`](LICENSE.md).
