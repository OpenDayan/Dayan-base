#include "service_internal.h"

#include <errno.h>
#include <stdlib.h>

struct parallel_call {
    int (*run)(void *, size_t);
    void *argument;
    size_t index;
    pthread_t thread;
    int started;
    int status;
};

static void *run_parallel_call(void *argument)
{
    struct parallel_call *call = argument;

    call->status = call->run(call->argument, call->index);
    return 0;
}

/* Each callback owns one Core's configuration. Join every started callback,
 * even on failure, before callers inspect results or roll configuration back. */
int npu_service_parallel(size_t count, int (*run)(void *, size_t), void *argument)
{
    struct parallel_call *calls;
    size_t index;
    int status = 0;

    if (count <= 1)
        return count ? run(argument, 0) : 0;
    calls = calloc(count, sizeof(*calls));
    if (!calls)
        return -ENOMEM;
    for (index = 0; index < count; ++index) {
        int error;

        calls[index].run = run;
        calls[index].argument = argument;
        calls[index].index = index;
        error = pthread_create(&calls[index].thread, 0, run_parallel_call,
                                &calls[index]);

        calls[index].started = !error;
        if (error)
            calls[index].status = -error;
    }
    for (index = 0; index < count; ++index) {
        if (calls[index].started)
            (void)pthread_join(calls[index].thread, 0);
        if (!status && calls[index].status < 0)
            status = calls[index].status;
    }
    free(calls);
    return status;
}
