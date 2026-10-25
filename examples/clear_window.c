/* CPU clear and presentation example. */
#include "swxbx.h"
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { WIDTH = 320, HEIGHT = 180 };
struct state {
    unsigned char memory[1024];
    unsigned char pixels[WIDTH * HEIGHT * 4];
    int presented;
};

static int32_t read_memory(void *user, uint64_t address, uint8_t *out, uint64_t size)
{
    struct state *s = user;
    if (address > sizeof(s->memory) || size > sizeof(s->memory) - address) return 1;
    memcpy(out, s->memory + address, (size_t)size);
    return 0;
}

static void present(void *user, uint64_t window, uint32_t width, uint32_t height,
                    const uint8_t *rgba, uint64_t stride)
{
    struct state *s = user;
    (void)window;
    if (width != WIDTH || height != HEIGHT || stride != WIDTH * 4) return;
    memcpy(s->pixels, rgba, sizeof(s->pixels));
    s->presented = 1;
}

static uint64_t call(swxbx_instance *instance, const char *suffix,
                     const uint64_t *args, size_t count)
{
    swxbx_registers registers = {0};
    memcpy(registers.x, args, count * sizeof(*args));
    for (uint32_t id = 0; id < swxbx_function_count(); ++id) {
        const char *name = swxbx_function_name(id);
        if (strlen(name) < 3 || strcmp(name + 3, suffix)) continue;
        if (swxbx_instance_call(instance, id, &registers) == SWXBX_OK) return registers.x[0];
        break;
    }
    fprintf(stderr, "Call failed: %s\n", suffix);
    exit(1);
}

#define CALL(name, ...) call(instance, name, (uint64_t[]){__VA_ARGS__}, \
                            sizeof((uint64_t[]){__VA_ARGS__}) / sizeof(uint64_t))

static unsigned long channel(unsigned char value, unsigned long mask)
{
    unsigned shift = 0;
    if (!mask) return 0;
    while (!(mask & 1)) { mask >>= 1; ++shift; }
    return (((unsigned long)value * mask + 127) / 255) << shift;
}

int main(int argc, char **argv)
{
    int check = argc == 2 && !strcmp(argv[1], "--check");
    if (argc > 1 && !check) {
        fprintf(stderr, "usage: %s [--check]\n", argv[0]);
        return 2;
    }
    struct state state = {0};
    if (swxbx_host_interface_version() != SWXBX_HOST_INTERFACE_VERSION) return 1;
    swxbx_host host = {.user = &state, .read_memory = read_memory, .present = present};
    swxbx_instance *instance = swxbx_instance_create(&host);
    if (!instance) return 1;
    CALL("TextureBuilderSetDefaults", 1);
    CALL("TextureBuilderSetSize2D", 1, WIDTH, HEIGHT);
    CALL("TextureInitialize", 2, 1);
    uint64_t texture = 2;
    float color[] = {0.0f, 0.5f, 1.0f, 1.0f};
    memcpy(state.memory + 128, &texture, sizeof(texture));
    memcpy(state.memory + 256, color, sizeof(color));
    CALL("WindowBuilderSetDefaults", 3);
    CALL("WindowBuilderSetTextures", 3, 1, 128);
    CALL("WindowInitialize", 4, 3);
    CALL("CommandBufferInitialize", 5, 0);
    CALL("CommandBufferBeginRecording", 5);
    CALL("CommandBufferSetRenderTargets", 5, 1, 128, 0, 0);
    CALL("CommandBufferClearColor", 5, 0, 256, 15);
    uint64_t handle = CALL("CommandBufferEndRecording", 5);
    memcpy(state.memory + 512, &handle, sizeof(handle));
    CALL("QueueSubmitCommands", 0, 1, 512);
    CALL("QueuePresentTexture", 0, 4, 0);
    swxbx_instance_destroy(instance);
    if (!state.presented) return 1;
    for (size_t p = 0; p < sizeof(state.pixels); p += 4) {
        const unsigned char expected[] = {0, 128, 255, 255};
        if (memcmp(state.pixels + p, expected, 4)) return 1;
    }
    if (check) { puts("Every cleared pixel matches."); return 0; }

    Display *display = XOpenDisplay(NULL);
    if (!display) { fputs("An X11 display is required.\n", stderr); return 1; }
    int screen = DefaultScreen(display);
    Visual *visual = DefaultVisual(display, screen);
    Window window = XCreateSimpleWindow(display, RootWindow(display, screen),
                                        0, 0, WIDTH, HEIGHT, 0, 0, 0);
    XStoreName(display, window, "swxbx cleared window");
    XSelectInput(display, window, ExposureMask | KeyPressMask);
    Atom close = XInternAtom(display, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(display, window, &close, 1);
    XImage *image = XCreateImage(display, visual, (unsigned)DefaultDepth(display, screen),
                                 ZPixmap, 0, NULL, WIDTH, HEIGHT, 32, 0);
    if (!image) { XDestroyWindow(display, window); XCloseDisplay(display); return 1; }
    image->data = calloc((size_t)image->bytes_per_line, HEIGHT);
    if (!image->data) {
        XDestroyImage(image); XDestroyWindow(display, window); XCloseDisplay(display); return 1;
    }
    for (int y = 0; y < HEIGHT; ++y)
        for (int x = 0; x < WIDTH; ++x) {
            const unsigned char *p = state.pixels + (y * WIDTH + x) * 4;
            XPutPixel(image, x, y, channel(p[0], visual->red_mask) |
                      channel(p[1], visual->green_mask) | channel(p[2], visual->blue_mask));
        }
    XMapWindow(display, window);
    for (;;) {
        XEvent event;
        XNextEvent(display, &event);
        if (event.type == KeyPress ||
            (event.type == ClientMessage && (Atom)event.xclient.data.l[0] == close)) break;
        if (event.type == Expose)
            XPutImage(display, window, DefaultGC(display, screen), image,
                      0, 0, 0, 0, WIDTH, HEIGHT);
    }
    XDestroyImage(image);
    XDestroyWindow(display, window);
    XCloseDisplay(display);
    return 0;
}
