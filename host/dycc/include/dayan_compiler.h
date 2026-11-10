/***************************************************************************************
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Dayan-base is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *     http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
***************************************************************************************/

#ifndef DAYAN_COMPILER_ATTRIBUTES_H
#define DAYAN_COMPILER_ATTRIBUTES_H

/*
 * dycc force-includes this file while it builds the split AST.  The final
 * Host and Device compilations replace these macros with their code-generation
 * definitions, so the annotations never become part of the public ABI.
 */
#ifndef __host__
#define __host__ __attribute__((annotate("dayan.host")))
#endif

#ifndef __device__
#define __device__ __attribute__((annotate("dayan.device")))
#endif

#ifndef __kernel__
#define __kernel__ __attribute__((annotate("dayan.kernel")))
#endif

#endif
