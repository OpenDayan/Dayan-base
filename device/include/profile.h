#ifndef NPU_PROFILE_H
#define NPU_PROFILE_H

#include <stdint.h>

struct proc;
struct rpc_profile_config;

int profile_init(void);
int profile_configure(struct rpc_profile_config *config);
void profile_record_proc_start(struct proc *proc);
void profile_record_proc_state(struct proc *proc);
struct proc *profile_proc(void);

#endif
