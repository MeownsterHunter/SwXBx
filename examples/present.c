/* Cycling clear colors through the host graphics API. */
#define VK_USE_PLATFORM_XLIB_KHR
#include <vulkan/vulkan.h>
#include <X11/Xlib.h>
#include "swxbx.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { WIDTH = 640, HEIGHT = 360, IMAGE_BYTES = WIDTH * HEIGHT * 4,
       STORAGE = 0x1000, MEMORY_BYTES = STORAGE + IMAGE_BYTES * 3 };

struct state {
    Display *display;
    Window window;
    uint8_t *memory;
};

static int32_t read_memory(void *user, uint64_t address, uint8_t *out, uint64_t size)
{
    struct state *state = user;
    if (address > MEMORY_BYTES || size > MEMORY_BYTES - address) return 1;
    memcpy(out, state->memory + address, (size_t)size);
    return 0;
}

static int32_t write_memory(void *user, uint64_t address, const uint8_t *data, uint64_t size)
{
    struct state *state = user;
    if (address > MEMORY_BYTES || size > MEMORY_BYTES - address) return 1;
    memcpy(state->memory + address, data, (size_t)size);
    return 0;
}

static uint64_t create_surface(void *user, uint64_t instance, uint64_t object, uint64_t native_window)
{
    struct state *state = user;
    (void)object;
    if (native_window != state->window) return 0;
    VkInstance vk_instance = (VkInstance)(uintptr_t)instance;
    PFN_vkCreateXlibSurfaceKHR create = (PFN_vkCreateXlibSurfaceKHR)
        vkGetInstanceProcAddr(vk_instance, "vkCreateXlibSurfaceKHR");
    if (create == NULL) return 0;
    VkXlibSurfaceCreateInfoKHR info = {
        .sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
        .dpy = state->display, .window = state->window
    };
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkResult result = create(vk_instance, &info, NULL, &surface);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "surface creation failed: %d\n", result);
        return 0;
    }
    return (uint64_t)(uintptr_t)surface;
}

static void drawable_size(void *user, uint64_t object, uint32_t *width, uint32_t *height)
{
    struct state *state = user;
    XWindowAttributes attributes;
    (void)object;
    *width = *height = 0;
    if (XGetWindowAttributes(state->display, state->window, &attributes) &&
        attributes.map_state == IsViewable) {
        *width = (uint32_t)attributes.width;
        *height = (uint32_t)attributes.height;
    }
}

static int call(swxbx_instance *instance, const char *name, swxbx_registers *registers)
{
    uint32_t function = swxbx_instance_request(instance, name);
    swxbx_status status = swxbx_instance_call(instance, function, registers);
    if (status != SWXBX_OK) {
        fprintf(stderr, "%s failed: status %d\n", name, status);
        const char *error = swxbx_last_error();
        if (error != NULL) fprintf(stderr, "%s\n", error);
        return 0;
    }
    return 1;
}

#define CALL(name, ...) do { \
    swxbx_registers registers = {.x = {__VA_ARGS__}}; \
    if (!call(instance, name, &registers)) goto cleanup; \
} while (0)

int main(int argc, char **argv)
{
    unsigned limit = 0;
    int resize = 0, unpaced = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--frames") && i + 1 < argc) limit = (unsigned)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--resize")) resize = 1;
        else if (!strcmp(argv[i], "--unpaced")) unpaced = 1;
        else {
            fprintf(stderr, "usage: %s [--frames N] [--resize] [--unpaced]\n", argv[0]);
            return 2;
        }
    }
    if (swxbx_host_interface_version() != SWXBX_HOST_INTERFACE_VERSION) return 1;
    XInitThreads();
    struct state state = {0};
    state.display = XOpenDisplay(NULL);
    if (state.display == NULL) {
        fprintf(stderr, "An X11 display is required.\n");
        return 1;
    }
    state.window = XCreateSimpleWindow(state.display, DefaultRootWindow(state.display),
                                      0, 0, WIDTH, HEIGHT, 0, 0, 0);
    XStoreName(state.display, state.window, "swxbx cycling clear");
    XSelectInput(state.display, state.window, StructureNotifyMask | KeyPressMask);
    Atom close = XInternAtom(state.display, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(state.display, state.window, &close, 1);
    XMapWindow(state.display, state.window);
    XSync(state.display, False);
    state.memory = calloc(MEMORY_BYTES, 1);
    if (state.memory == NULL) {
        XDestroyWindow(state.display, state.window);
        XCloseDisplay(state.display);
        return 1;
    }
    const char *extensions[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_XLIB_SURFACE_EXTENSION_NAME};
    swxbx_host_vulkan vulkan = {
        .extension_count = 2, .extensions = extensions,
        .create_surface = create_surface, .drawable_size = drawable_size,
    };
    swxbx_host host = {
        .user = &state, .read_memory = read_memory, .write_memory = write_memory,
        .render_scale = 1.0f, .vulkan = &vulkan,
    };
    swxbx_instance *instance = swxbx_instance_create(&host);
    int result = 1;
    if (instance == NULL) goto cleanup;

    CALL("nvnDeviceBuilderSetDefaults", 2);
    CALL("nvnDeviceInitialize", 3, 2);
    CALL("nvnMemoryPoolBuilderSetDefaults", 4);
    CALL("nvnMemoryPoolBuilderSetStorage", 4, STORAGE, IMAGE_BYTES * 3);
    CALL("nvnMemoryPoolInitialize", 5, 4);
    uint64_t textures[] = {20, 21, 22};
    for (unsigned i = 0; i < 3; ++i) {
        CALL("nvnTextureBuilderSetDefaults", 6);
        CALL("nvnTextureBuilderSetSize2D", 6, WIDTH, HEIGHT);
        CALL("nvnTextureBuilderSetStorage", 6, 5, i * IMAGE_BYTES);
        CALL("nvnTextureInitialize", textures[i], 6);
    }
    memcpy(state.memory + 0x100, textures, sizeof(textures));
    CALL("nvnWindowBuilderSetDefaults", 7);
    CALL("nvnWindowBuilderSetNativeWindow", 7, state.window);
    CALL("nvnWindowBuilderSetTextures", 7, 3, 0x100);
    CALL("nvnWindowInitialize", 8, 7);
    CALL("nvnWindowSetPresentInterval", 8, unpaced ? 0 : 1);
    CALL("nvnCommandBufferInitialize", 9, 3);
    CALL("nvnQueueBuilderSetDefaults", 10);
    CALL("nvnQueueBuilderSetDevice", 10, 3);
    CALL("nvnQueueInitialize", 11, 10);

    for (unsigned frame = 0; limit == 0 || frame < limit; ++frame) {
        int quit = 0;
        while (XPending(state.display)) {
            XEvent event;
            XNextEvent(state.display, &event);
            if (event.type == KeyPress ||
                (event.type == ClientMessage && (Atom)event.xclient.data.l[0] == close)) quit = 1;
        }
        if (quit) break;
        if (resize && (frame == 10 || frame == 30)) {
            XResizeWindow(state.display, state.window, frame == 10 ? 480 : WIDTH, frame == 10 ? 480 : HEIGHT);
            XSync(state.display, False);
        }
        CALL("nvnWindowAcquireTexture", 8, 0, 0x400);
        uint32_t index;
        memcpy(&index, state.memory + 0x400, sizeof(index));
        if (index >= 3) goto cleanup;
        memcpy(state.memory + 0x500, &textures[index], sizeof(uint64_t));
        float phase = (float)frame * 0.03f;
        float color[] = {
            0.5f + 0.5f * sinf(phase),
            0.5f + 0.5f * sinf(phase + 2.094395f),
            0.5f + 0.5f * sinf(phase + 4.188790f), 1.0f,
        };
        memcpy(state.memory + 0x200, color, sizeof(color));
        CALL("nvnCommandBufferBeginRecording", 9);
        CALL("nvnCommandBufferSetRenderTargets", 9, 1, 0x500, 0, 0);
        CALL("nvnCommandBufferClearColor", 9, 0, 0x200, 15);
        swxbx_registers end = {.x = {9}};
        if (!call(instance, "nvnCommandBufferEndRecording", &end)) goto cleanup;
        memcpy(state.memory + 0x300, &end.x[0], sizeof(uint64_t));
        CALL("nvnQueueSubmitCommands", 11, 1, 0x300);
        CALL("nvnQueuePresentTexture", 11, 8, index);
    }
    CALL("nvnQueueFinish", 11);
    CALL("nvnWindowFinalize", 8);
    result = 0;
cleanup:
    swxbx_instance_destroy(instance);
    XDestroyWindow(state.display, state.window);
    XCloseDisplay(state.display);
    free(state.memory);
    return result;
}
