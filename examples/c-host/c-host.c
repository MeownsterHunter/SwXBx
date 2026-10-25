#include "swxbx.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { MEMORY_SIZE = 65536, WIDTH = 4, HEIGHT = 3 };

struct host_state {
    uint8_t memory[MEMORY_SIZE];
    const char *frame_path;
    unsigned presents;
};

static int32_t read_memory(void *user, uint64_t address, uint8_t *out, uint64_t size)
{
    struct host_state *state = user;
    if (address > MEMORY_SIZE || size > MEMORY_SIZE - address) {
        return 1;
    }
    memcpy(out, state->memory + address, (size_t)size);
    return 0;
}

static int32_t write_memory(void *user, uint64_t address, const uint8_t *data, uint64_t size)
{
    struct host_state *state = user;
    if (address > MEMORY_SIZE || size > MEMORY_SIZE - address) {
        return 1;
    }
    memcpy(state->memory + address, data, (size_t)size);
    return 0;
}

static void present(void *user, uint64_t window, uint32_t width, uint32_t height,
                   const uint8_t *rgba, uint64_t stride)
{
    struct host_state *state = user;
    FILE *file = fopen(state->frame_path, "wb");
    (void)window;
    if (file == NULL) {
        return;
    }
    fprintf(file, "P6\n%u %u\n255\n", width, height);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            fwrite(rgba + y * stride + x * 4, 1, 3, file);
        }
    }
    fclose(file);
    state->presents += 1;
}

static uint32_t function(const char *name, swxbx_instance *instance)
{
    uint32_t id = swxbx_function_lookup(name);
    if (id == SWXBX_FUNCTION_NONE || swxbx_instance_request(instance, name) != id) {
        fprintf(stderr, "could not resolve %s\n", name);
    }
    return id;
}

static swxbx_status call(swxbx_instance *instance, uint32_t id, swxbx_registers *r)
{
    return swxbx_instance_call(instance, id, r);
}

static int expect(swxbx_status actual, swxbx_status wanted, const char *what)
{
    if (actual != wanted) {
        fprintf(stderr, "%s returned status %d\n", what, actual);
        return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    struct host_state state = {0};
    uint64_t texture_list = 0x100;
    uint32_t clear_values = 0x200;
    uint64_t command_list = 0x300;
    const char *census = argc > 1 ? argv[1] : "census.txt";
    const char *shapes = argc > 2 ? argv[2] : "shapes.txt";
    state.frame_path = argc > 3 ? argv[3] : "frame.ppm";
    state.memory[clear_values + 0] = 0;
    state.memory[clear_values + 4] = 0;
    state.memory[clear_values + 8] = 0;
    state.memory[clear_values + 12] = 0;

    swxbx_host host = {&state, read_memory, write_memory, present, 1.0f, NULL, NULL};
    if (swxbx_host_interface_version() != SWXBX_HOST_INTERFACE_VERSION) {
        return 1;
    }
    swxbx_instance *instance = swxbx_instance_create(&host);
    if (instance == NULL) {
        return 1;
    }

    uint32_t device_defaults = function("nvnDeviceBuilderSetDefaults", instance);
    uint32_t device_init = function("nvnDeviceInitialize", instance);
    uint32_t texture_defaults = function("nvnTextureBuilderSetDefaults", instance);
    uint32_t texture_size = function("nvnTextureBuilderSetSize2D", instance);
    uint32_t texture_init = function("nvnTextureInitialize", instance);
    uint32_t window_defaults = function("nvnWindowBuilderSetDefaults", instance);
    uint32_t window_textures = function("nvnWindowBuilderSetTextures", instance);
    uint32_t window_init = function("nvnWindowInitialize", instance);
    uint32_t command_init = function("nvnCommandBufferInitialize", instance);
    uint32_t begin = function("nvnCommandBufferBeginRecording", instance);
    uint32_t set_targets = function("nvnCommandBufferSetRenderTargets", instance);
    uint32_t clear = function("nvnCommandBufferClearColor", instance);
    uint32_t end = function("nvnCommandBufferEndRecording", instance);
    uint32_t submit = function("nvnQueueSubmitCommands", instance);
    uint32_t present_id = function("nvnQueuePresentTexture", instance);
    if (device_defaults == SWXBX_FUNCTION_NONE || present_id == SWXBX_FUNCTION_NONE) {
        swxbx_instance_destroy(instance);
        return 1;
    }

    swxbx_registers r = {0};
    r.x[0] = 1;
    if (!expect(call(instance, device_defaults, &r), SWXBX_OK, "device defaults")) return 1;
    r = (swxbx_registers){0}; r.x[0] = 2; r.x[1] = 1;
    if (!expect(call(instance, device_init, &r), SWXBX_OK, "device initialize")) return 1;
    r = (swxbx_registers){0}; r.x[0] = 3;
    call(instance, texture_defaults, &r);
    r.x[1] = WIDTH; r.x[2] = HEIGHT; call(instance, texture_size, &r);
    r = (swxbx_registers){0}; r.x[0] = 4; r.x[1] = 3;
    call(instance, texture_init, &r);
    memcpy(state.memory + texture_list, &(uint64_t){4}, sizeof(uint64_t));
    r = (swxbx_registers){0}; r.x[0] = 5; call(instance, window_defaults, &r);
    r.x[1] = 1; r.x[2] = texture_list; call(instance, window_textures, &r);
    r = (swxbx_registers){0}; r.x[0] = 6; r.x[1] = 5;
    call(instance, window_init, &r);
    r = (swxbx_registers){0}; r.x[0] = 7; r.x[1] = 1; call(instance, command_init, &r);
    r = (swxbx_registers){0}; r.x[0] = 7; call(instance, begin, &r);
    r.x[1] = 1; r.x[2] = texture_list; r.x[3] = 0; r.x[4] = 0; call(instance, set_targets, &r);
    r = (swxbx_registers){0}; r.x[0] = 7; r.x[1] = 0; r.x[2] = clear_values; r.x[3] = 15;
    call(instance, clear, &r);
    r = (swxbx_registers){0}; r.x[0] = 7; call(instance, end, &r);
    uint64_t recording = r.x[0];
    memcpy(state.memory + command_list, &recording, sizeof(recording));
    r = (swxbx_registers){0}; r.x[0] = 1; r.x[1] = 1; r.x[2] = command_list;
    call(instance, submit, &r);
    r = (swxbx_registers){0}; r.x[0] = 6; r.x[1] = 6; r.x[2] = 0;
    if (!expect(call(instance, present_id, &r), SWXBX_OK, "present")) return 1;

    r = (swxbx_registers){0};
    if (!expect(swxbx_instance_call(instance, 999999, &r), SWXBX_BAD_FUNCTION,
                "deliberate error") || swxbx_last_error() != NULL) return 1;
    swxbx_instance_write_census(instance, census);
    swxbx_instance_write_shapes(instance, shapes);
    swxbx_instance_destroy(instance);
    return 0;
}
