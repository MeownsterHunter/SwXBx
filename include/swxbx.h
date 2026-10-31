/* swxbx host interface, version 6. See docs/host-contract.md. */
#ifndef SWXBX_H
#define SWXBX_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SWXBX_HOST_INTERFACE_VERSION 6u
#define SWXBX_CAPABILITY_BC67_COMPRESSION (UINT64_C(1) << 0)

/* Returned by lookups for a name the library does not know. */
#define SWXBX_FUNCTION_NONE UINT32_MAX
#define SWXBX_SHADER_STAGE_UNKNOWN 0u

typedef enum swxbx_status {
    SWXBX_OK = 0,
    /* The function has no behaviour yet. The call was counted and the result
     * registers were set to zero. */
    SWXBX_UNIMPLEMENTED = 1,
    /* The function id is outside the table. */
    SWXBX_BAD_FUNCTION = 2,
    /* A pointer argument was null, or a file could not be written. */
    SWXBX_BAD_ARGUMENT = 3,
    /* The library panicked while servicing the call. */
    SWXBX_INTERNAL_ERROR = 4
} swxbx_status;

/* How a presented image fills the window's drawable (version 6).
 * SWXBX_SCALING_STRETCH fills it. SWXBX_SCALING_FIT keeps the ratio
 * aspect_width : aspect_height (both positive and finite), centred, with black
 * bars on the remaining sides; the aspect fields are read only for a fit. */
#define SWXBX_SCALING_STRETCH 0u
#define SWXBX_SCALING_FIT 1u
typedef struct swxbx_scaling {
    uint32_t kind;
    float aspect_width;
    float aspect_height;
} swxbx_scaling;

/* Optional native presentation contract, introduced in version 5; scaling
 * was added in version 6.
 * The extension names remain valid until instance creation returns.
 * create_surface returns a VkSurfaceKHR as uint64_t, or 0 on failure.
 * swxbx owns and destroys that surface. The native window stays live until
 * WindowFinalize or instance destruction. The struct and callbacks stay live
 * until instance destruction. All callbacks must be non-null and must not
 * reenter swxbx. drawable_size writes pixel dimensions, or zeros if hidden.
 * scaling runs at every present and must write the window's scaling: there is
 * no default. swxbx initializes kind to UINT32_MAX, and a present whose
 * scaling is unwritten or invalid returns SWXBX_BAD_ARGUMENT and names the
 * reason through swxbx_last_error. A present blits from the presented
 * texture's own format, reading an sRGB texture's bytes unchanged. A depth,
 * integer or compressed texture, or one the device cannot blit into the
 * window's images, returns SWXBX_UNIMPLEMENTED and names the reason through
 * swxbx_last_error. */
typedef struct swxbx_host_vulkan {
    uint32_t extension_count;
    const char *const *extensions;
    uint64_t (*create_surface)(void *user, uint64_t instance,
                               uint64_t window_object, uint64_t native_window);
    void (*drawable_size)(void *user, uint64_t window_object,
                          uint32_t *width, uint32_t *height);
    void (*scaling)(void *user, uint64_t window_object, swxbx_scaling *scaling);
} swxbx_host_vulkan;

/* What the host provides. Both callbacks return 0 on success and any other
 * value when the range is not accessible.
 *
 * The host guarantees that `user` and both callbacks stay usable until the
 * instance is destroyed, and that they tolerate being used from several
 * threads at once, including library execution threads. Queued work drains
 * before instance destruction returns. Submitted memory remains valid and
 * unchanged until completion. Callbacks must not wait for completion
 * on their own instance; a call that would drain its own execution thread
 * returns an internal error. */
typedef struct swxbx_host {
    void *user;
    int32_t (*read_memory)(void *user, uint64_t address, uint8_t *out, uint64_t size);
    int32_t (*write_memory)(void *user, uint64_t address, const uint8_t *data, uint64_t size);
    void (*present)(void *user, uint64_t window_object, uint32_t width, uint32_t height,
                    const uint8_t *rgba, uint64_t stride_bytes);
    float render_scale;
    void (*wait_vblank)(void *user);
    const swxbx_host_vulkan *vulkan;
} swxbx_host;

/* Argument and result registers of one call under the program's standard
 * calling convention: eight integer registers, the low 64 bits of eight
 * floating-point registers, and the stack pointer for arguments passed on
 * the stack. On return x[0], x[1] and d[0] hold the results. */
typedef struct swxbx_registers {
    uint64_t x[8];
    uint64_t d[8];
    uint64_t sp;
} swxbx_registers;

/* Host-owned SPIR-V returned by a successful
 * swxbx_host_shader_translator.translate call. swxbx copies the words before
 * calling release exactly once. */
typedef struct swxbx_translated_shader {
    const uint32_t *words;
    uint64_t word_count;
} swxbx_translated_shader;

/* Optional host translator for persistent embedded instances.
 * All callbacks must tolerate concurrent calls. A successful install transfers
 * ownership of user to swxbx. destroy_user runs once after the final request.
 * translate returns zero on success and writes a nonempty borrowed output of
 * at most 16,777,216 words. */
typedef struct swxbx_host_shader_translator {
    void *user;
    int32_t (*translate)(void *user, uint32_t stage,
                         const uint8_t *code, uint64_t code_size,
                         swxbx_translated_shader *output);
    void (*release)(void *user, const uint32_t *words, uint64_t word_count);
    void (*destroy_user)(void *user);
} swxbx_host_shader_translator;

typedef struct swxbx_instance swxbx_instance;

/* Library version as a NUL-terminated string owned by the library. */
const char *swxbx_version(void);

/* Host interface version the library was built with. A host compares it with
 * SWXBX_HOST_INTERFACE_VERSION and refuses to continue when they differ. */
uint32_t swxbx_host_interface_version(void);

/* Stable module capabilities available before instance creation. */
uint64_t swxbx_capabilities(void);

/* The function table. Valid ids are 0 .. count-1. */
uint32_t swxbx_function_count(void);
const char *swxbx_function_name(uint32_t function);
uint32_t swxbx_function_lookup(const char *name);

/* Instances.
 *
 * Every function below that takes an instance needs the pointer returned by
 * swxbx_instance_create, not yet destroyed. Apart from destroy, they may be
 * called on one instance from several threads at once.
 *
 * host may be null; the structure is copied. */
swxbx_instance *swxbx_instance_create(const swxbx_host *host);

/* Null is accepted. No other call on the instance may be running, and none
 * may be made afterwards. */
void swxbx_instance_destroy(swxbx_instance *instance);

/* Enable notifications for a registered pool and its aliases. Report every
 * later host storage write before submission. Legacy hosts remain supported. */
swxbx_status swxbx_instance_track_pool_writes(const swxbx_instance *instance, uint64_t pool);
swxbx_status swxbx_instance_notify_memory_write(const swxbx_instance *instance,
                                                  uint64_t address, uint64_t size);

/* Adopt coherent arena storage, preserving contents. The pointer lives until
 * the last alias is finalized. Host callbacks must route storage through it.
 * Mapping and waiting first run every batch submitted on this instance before
 * the call; batches submitted concurrently from another thread are not covered.
 * Wait before each access and report writes before the next submission.
 * Mapping returns null and waiting returns a status on failure; both name the
 * reason through swxbx_last_error. */
uint8_t *swxbx_instance_map_pool(const swxbx_instance *instance, uint64_t pool);
swxbx_status swxbx_instance_wait_pool(const swxbx_instance *instance, uint64_t pool,
                                       uint64_t offset, uint64_t size, uint32_t write);

/* Host-selected policy. Capacity is 1..16, default 2. Mode is 0 for FIFO,
 * 1 for mailbox. With a native-surface host, mailbox needs an open window
 * whose surface offers it, and is refused by name otherwise. A mode that
 * cannot show an open window's present interval is refused by name (native
 * FIFO cannot present interval 0 without waiting; native mailbox cannot hold
 * an image for interval 2 or more). Changing policy drains pending callbacks
 * first.
 * Callback bytes are borrowed and valid until the callback returns.
 * Presentation callbacks must not reenter the library. */
swxbx_status swxbx_instance_set_presentation(const swxbx_instance *instance,
                                               uint32_t frames_in_flight, uint32_t mode);

/* wait=0 delivers one ready frame without waiting for GPU completion.
 * wait=1 drains all pending frames. Before each delivery wait_vblank runs
 * once per vertical blank of the window's present interval (none for 0).
 * QueueFinish and WindowFinalize also drain pending callbacks. */
swxbx_status swxbx_instance_poll_presentations(const swxbx_instance *instance, uint32_t wait);

typedef struct swxbx_frame_statistics {
    uint64_t submitted, delivered, dropped, skipped, failed;
    uint32_t pending, peak_pending;
    uint64_t last_latency_ns;
    double mean_latency_ns, variance_latency_ns2;
} swxbx_frame_statistics;

/* Cumulative latency runs from guest present entry to callback entry, or native
 * presentation handoff. Variance is the population variance. Counts distinguish
 * mailbox drops from delivery and failures. Display scanout is not measured. */
swxbx_status swxbx_instance_frame_statistics(const swxbx_instance *instance,
                                               swxbx_frame_statistics *out);

/* Set or clear (with null) the directory for local debugging dumps of
 * translated shaders. Files contain translated output derived from the
 * observed program's shaders, never the original shader bytes. */
swxbx_status swxbx_instance_set_shader_dump_directory(const swxbx_instance *instance,
                                                         const char *directory);

/* Install or clear (with null) the translator used by future shader
 * registrations. translate, release and destroy_user are all required.
 * A successful install enables translation. */
swxbx_status swxbx_instance_set_shader_translator(
    const swxbx_instance *instance,
    const swxbx_host_shader_translator *translator);

/* The program asked its bootstrap function for a name. Records the request
 * and returns the function id, or SWXBX_FUNCTION_NONE. */
uint32_t swxbx_instance_request(const swxbx_instance *instance, const char *name);

/* The program called a function. `registers` must not be read or written by
 * anything else until the call returns; each thread uses its own. */
swxbx_status swxbx_instance_call(const swxbx_instance *instance, uint32_t function,
                                   swxbx_registers *registers);

/* The original implementation of a function returned; `registers` holds its
 * result registers. Only for hosts that let the original implementation run,
 * so that results are sampled along with arguments. */
swxbx_status swxbx_instance_returned(const swxbx_instance *instance, uint32_t function,
                                       const swxbx_registers *registers);

/* What was requested and called so far. */
uint64_t swxbx_instance_call_count(const swxbx_instance *instance, uint32_t function);
swxbx_status swxbx_instance_write_census(const swxbx_instance *instance, const char *path);

/* What the argument and result registers held in the first calls of each
 * function: ranges, a few distinct values, and which were readable addresses.
 * Never the memory behind an address. */
swxbx_status swxbx_instance_write_shapes(const swxbx_instance *instance, const char *path);

/* The most recent panic message on this thread, or null when there was none.
 * The returned string is owned by the library and remains valid until the
 * next call on this thread. */
const char *swxbx_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* SWXBX_H */
