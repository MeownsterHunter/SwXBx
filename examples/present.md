# Cycling clear colours

This Linux example opens an X11 window and presents a cycling clear colour
through a console graphics API. It uses three textures in one flat arena
pool. Vulkan owns the surface and swapchain after the host creates the surface.

Install a Rust toolchain, a C compiler, X11 development headers, Vulkan
development headers, and a Vulkan driver. From the repository root:

```sh
cargo build --features vulkan
cc -std=c11 -Wall -Wextra -Werror -Iinclude examples/present.c \
  -Ltarget/debug -lswxbx -lX11 -lvulkan -lm \
  -Wl,-rpath,'$ORIGIN/debug' -o target/present
target/present
```

Press any key or close the window to exit. Resize the window to exercise
swapchain recreation. For a finite smoke test, run
`target/present --frames 120 --resize`. Add `--unpaced` to request immediate
presentation where the surface supports it.

For pixel verification without a display:

```sh
python3 scripts/check-presentation.py
```

These tests require a Vulkan device, including a software Vulkan driver.
They fail if the device is absent. They check every presented pixel, arena
aliases, masked clears, texture copies, depth clears, scaling, BGRA order,
sRGB encoding, and repeated frame submission. CI runs them with a software
driver. The same Vulkan blit and barriers drive offscreen and window output.

With a display, add `--window` to check both FIFO and immediate presentation
with resizes. Enable `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` when
validation layers are installed. The script fails on validation errors and
saves full command output to `target/presentation.log`.
