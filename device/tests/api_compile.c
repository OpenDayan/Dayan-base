#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <dayan_sys.h>

static void compile_child(void *argument)
{
    (void)argument;
}

int operator_api_compile(dayan_resource_t *resource,
                         char *buffer, size_t size)
{
    dayan_shared_resource_t *shared_resource =
        (dayan_shared_resource_t *)shared;
    dayan_cluster_barrier_t *barrier =
        (dayan_cluster_barrier_t *)shared;
    void *memory = malloc(64);
    void *uram;
    int status;

    memset(memory, 0, 64);
    status = dayan_resource_init(resource);
    status |= dayan_resource_alloc(resource, 1, 1);
    status |= dayan_resource_produce(resource);
    status |= dayan_resource_wait_ready(resource);
    status |= dayan_resource_consume(resource);
    status |= dayan_resource_wait_free(resource);
    status |= dayan_shared_resource_init(shared_resource);
    status |= dayan_shared_resource_alloc(shared_resource, 1, 1);
    status |= dayan_shared_resource_produce(shared_resource);
    status |= dayan_shared_resource_wait_ready(shared_resource);
    status |= dayan_shared_resource_consume(shared_resource);
    status |= dayan_shared_resource_wait_free(shared_resource);
    status |= dayan_create_proc(compile_child, memory);
    uram = dayan_uram_base();
    if (uram && !dayan_uram_size())
        status = -1;
    status |= dayan_cluster_barrier(4, barrier);
    dayan_yield();
    dayan_wait_dma();
    dayan_wait_lsu();
    dayan_wait_mat();
    dayan_wait_vec();
    dayan_wait_macro();
    (void)printf("status=%d", status);
    (void)fflush(stdout);
    free(memory);

    return snprintf(buffer, size, "%d %x", status,
                    (unsigned)fabsf((float)CHAR_BIT));
}
