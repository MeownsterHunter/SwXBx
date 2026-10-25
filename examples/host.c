/* The smallest possible host: asks for two functions by name, calls one,
 * and prints the census. Build and run with examples/run-host.sh. */
#include "swxbx.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (swxbx_host_interface_version() != SWXBX_HOST_INTERFACE_VERSION) {
        fprintf(stderr, "swxbx was built for another host interface version\n");
        return 1;
    }

    printf("swxbx %s, %u functions\n", swxbx_version(), swxbx_function_count());

    swxbx_instance *instance = swxbx_instance_create(NULL);

    /* A real host does this when the program calls its bootstrap function. */
    const char *first = swxbx_function_name(0);
    uint32_t function = swxbx_instance_request(instance, first);
    if (function == SWXBX_FUNCTION_NONE) {
        fprintf(stderr, "%s is missing from the table\n", first);
        return 1;
    }
    swxbx_instance_request(instance, "aNameTheLibraryDoesNotKnow");

    /* And this when the program calls the pointer it was given. */
    swxbx_registers registers = { 0 };
    registers.x[0] = 1234;
    swxbx_status status = swxbx_instance_call(instance, function, &registers);
    if (status != SWXBX_UNIMPLEMENTED || registers.x[0] != 0) {
        fprintf(stderr, "unexpected result from an unimplemented function\n");
        return 1;
    }
    printf("%s called %llu time(s)\n", first,
           (unsigned long long)swxbx_instance_call_count(instance, function));

    if (argc > 1 && swxbx_instance_write_census(instance, argv[1]) != SWXBX_OK) {
        fprintf(stderr, "could not write %s\n", argv[1]);
        return 1;
    }

    swxbx_instance_destroy(instance);
    return 0;
}
