# Integration guide

## Rust

Add the crate as a dependency and enable `vulkan` for GPU execution:

```toml
[dependencies]
swxbx = { path = "crates/swxbx", features = ["vulkan"] }
```

Construct the host callbacks, create an `Instance`, install a shader
translator, then forward API calls. Keep compiler and window-system adapters
in the embedding project.

## C and C++

Include `swxbx.h`, link `libswxbx`, and compare the runtime interface version
before creating an instance.

```c
if (swxbx_host_interface_version() != SWXBX_HOST_INTERFACE_VERSION) {
    return 1;
}

swxbx_host host = {0};
host.user = context;
host.read_memory = read_program_memory;
host.write_memory = write_program_memory;

swxbx_instance *instance = swxbx_instance_create(&host);
if (instance == NULL) {
    return 1;
}
```

With CMake:

```cmake
find_package(swxbx CONFIG REQUIRED)
target_link_libraries(my_host PRIVATE swxbx::swxbx)
```

With pkg-config:

```sh
pkg-config --cflags --libs swxbx
```

## Adapter checklist

- Check the ABI version at startup.
- Validate every program-memory range before copying.
- Keep callback state alive through instance destruction.
- Make the shader translator safe for concurrent calls.
- Keep guest addresses opaque outside memory callbacks.
- Pump callback presentation from the host event loop.
- Treat every non-success status as actionable.
- Drain or destroy the instance before unloading the library.
