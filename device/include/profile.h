#ifndef NPU_PROFILE_H
#define NPU_PROFILE_H

#include <stdint.h>

struct thread;
struct rpc_profile_config;

int profile_init(void);
int profile_configure(struct rpc_profile_config *config);
void profile_record_thread_start(struct thread *thread);
void profile_record_thread_state(struct thread *thread);
struct thread *profile_thread(void);

#endif
