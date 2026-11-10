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

#ifndef NPU_PLATFORM_REGS_H
#define NPU_PLATFORM_REGS_H

/* Offsets relative to CONFIG_CONTROL_BASE, from the Dayan register list. */
#define DAYAN_BOOTROM_RESET_VECTOR       0x00010000U
#define DAYAN_BOOTROM_BUILD_ID_ADDRESS   (DAYAN_BOOTROM_RESET_VECTOR + 256U)
#define DAYAN_BOOTROM_BUILD_ID_SIZE      256U
#define DAYAN_DEVICE_MAILBOX_BASE       0x001000U
#define DAYAN_IRQ_STATUS                0x002000U
#define DAYAN_IRQ_STATUS_SET            0x002008U
#define DAYAN_IRQ_STATUS_CLR            0x00200cU
#define DAYAN_IRQ_ENABLE_SET            0x002010U
#define DAYAN_IRQ_ENABLE_CLR            0x002014U
#define DAYAN_IRQ_MAILBOX_BASE          0x002020U
#define DAYAN_ATOMIC_CACHE_CONTROL_BASE 0x004000U
#define DAYAN_ATOMIC_CACHE_STATUS       0x004000U
#define DAYAN_ATOMIC_CACHE_BACKING_BASE 0x004008U
#define DAYAN_ATOMIC_CACHE_WINDOW_SIZE  0x400000U
#define DAYAN_RESOURCE_GLOBAL_INIT     0x006000U
#define DAYAN_CLINT_MSIP_BASE            0x020000U

#define DAYAN_PCU_BASE                  0x030000U
#define DAYAN_PCU_CORE_STRIDE           0x002000U
#define DAYAN_PCU_COUNTER(hart, event) \
    (DAYAN_PCU_BASE + (hart) * DAYAN_PCU_CORE_STRIDE + (event) * 8U)
#define DAYAN_PCU_CONTROL(cores) \
    (DAYAN_PCU_BASE + (cores) * DAYAN_PCU_CORE_STRIDE - 8U)
/* Reserved pending RTL support: rdcycle, latched with this core's counters. */
#define DAYAN_PCU_CYCLE(hart)           DAYAN_PCU_COUNTER(hart, 1022U)

#define DAYAN_DEVICE_MAILBOX(hart) \
    (DAYAN_DEVICE_MAILBOX_BASE + (hart) * 8U)
#define DAYAN_IRQ_MAILBOX(hart) \
    (DAYAN_IRQ_MAILBOX_BASE + (hart) * 8U)
#define DAYAN_CLINT_MSIP(hart) \
    (DAYAN_CLINT_MSIP_BASE + (hart) * 4U)

/*
 * Cluster IP aperture: the fixed module layout below CONFIG_CONTROL_BASE +
 * DAYAN_CLUSTER_APERTURE_SIZE, as the NPU sees it and as a Host reaches it
 * through the Cluster slave port.  A Host that maps the aperture at some
 * base B reaches IP offset X at B + X.  Values follow the rvtng QEMU model
 * and the FPGA register list; the SoC integration has not confirmed them.
 *
 * DAYAN_DRAM_BASE is not part of the aperture: it is where the NPU master
 * port sees external memory, and what the firmware is linked against
 * (device/Makefile RAM_BASE, which must stay equal).
 */
#define DAYAN_IRQ_ENABLE                0x002004U
#define DAYAN_IRQ_CONTROL_SIZE          0x000018U
#define DAYAN_ATOMIC_CACHE_CONTROL_SIZE 0x000010U
#define DAYAN_COHERENT_WINDOW_BASE      0x00400000U
#define DAYAN_URAM_BASE                 0x00800000U
#define DAYAN_URAM_SIZE_768K            0x000c0000U
#define DAYAN_URAM_SIZE_1536K           0x00180000U
/* Default until the platform provides a capacity/version register. */
#define DAYAN_URAM_SIZE                 DAYAN_URAM_SIZE_1536K
#define DAYAN_URAM_STRIDE               0x00200000U
#define DAYAN_URAM(hart) \
    (DAYAN_URAM_BASE + (hart) * DAYAN_URAM_STRIDE)
/* Harts the fixed layout has room for: mailboxes, MSIP, PCU and URAM. */
#define DAYAN_CLUSTER_MAX_HARTS         4U
#define DAYAN_CLUSTER_APERTURE_SIZE     0x01000000U
#define DAYAN_DRAM_BASE                 0x80000000U

/*
 * ClusterMMU APB4 register page.
 *
 * The integration manual (section 6.1 address map, section 6.6) places
 * mmuconfig at 0x005000-0x006000.  Keep the base overridable for older
 * simulators that mapped the page elsewhere, but default to the hardware
 * address used by FPGA and the RTL cluster simulator.
 */
#ifndef DAYAN_MMU_APB_BASE
#define DAYAN_MMU_APB_BASE              0x005000U
#endif
#define DAYAN_MMU_APB_SIZE              0x001000U
#define DAYAN_MMU_HW_APB_BASE           0x005000U
#define DAYAN_MMU_LOW_2G_PAGES          0x00080000U

/*
 * 56-bit physical addresses are split across a LO/HI register pair: LO
 * takes bits [31:0] and HI bits [55:32], so only the low 24 bits of a HI
 * write are kept.  Every base bound to a translation window (FixMap,
 * redirect, error-slave) must be 4 KiB aligned; the hardware silently
 * ignores a misaligned LO write and does not raise an APB error
 * (manual section 2.2.8 clause 4), so software has to check first.
 */
#define DAYAN_MMU_PTW_BASE_LO           0x000U
#define DAYAN_MMU_PTW_BASE_HI           0x004U
#define DAYAN_MMU_FIX_MAP_BASE_LO       0x008U
#define DAYAN_MMU_FIX_MAP_BASE_HI       0x00cU
#define DAYAN_MMU_N_FIX_MAP_EN          0x010U
#define DAYAN_MMU_TLB_FLUSH             0x014U
#define DAYAN_MMU_FLUSH_ACK             0x018U
#define DAYAN_MMU_FLUSH_DONE            0x01cU
#define DAYAN_MMU_FLUSH_BUSY            0x020U
#define DAYAN_MMU_FLUSH_ERR_CODE        0x024U
#define DAYAN_MMU_GLOBAL_ASID           0x028U
#define DAYAN_MMU_ERROR_SLV_MASK_LO     0x048U
#define DAYAN_MMU_ERROR_SLV_MASK_HI     0x04cU
#define DAYAN_MMU_ERROR_SLV_OFF_MODE    0x050U
#define DAYAN_MMU_ERROR_SLV_CFG_ERR     0x070U
#define DAYAN_MMU_ERROR_SLV_CFG_CLR     0x074U
#define DAYAN_MMU_FIX_MAP_SIZE_LO       0x0b0U
#define DAYAN_MMU_FIX_MAP_SIZE_HI       0x0b4U

/*
 * Redirect bases, one per fault class, in the order the hardware fault
 * codes select them (manual section 6.6.4).  A fault does not abort the
 * access: the MMU replaces the whole master AR/AW address with the base
 * for that class, so the access completes somewhere harmless instead of
 * reporting anything.
 */
#define DAYAN_MMU_REDIR_PAGE_FAULT_LO   0x080U
#define DAYAN_MMU_REDIR_READ_PERM_LO    0x088U
#define DAYAN_MMU_REDIR_WRITE_PERM_LO   0x090U
#define DAYAN_MMU_REDIR_EXEC_PERM_LO    0x098U
#define DAYAN_MMU_REDIR_PTW_ERR_LO      0x0a0U
#define DAYAN_MMU_REDIR_DEFAULT_LO      0x0a8U
#define DAYAN_MMU_REDIR_CLASS_COUNT     6U
#define DAYAN_MMU_REDIR_CLASS_STRIDE    0x008U
#define DAYAN_MMU_REDIR_LO(index) \
    (DAYAN_MMU_REDIR_PAGE_FAULT_LO + (index) * DAYAN_MMU_REDIR_CLASS_STRIDE)

/* TLB_FLUSH: bit0 triggers, bit1 selects ASID mode, bits [9:2] carry the
 * ASID.  Mode 0 is a full flush. */
#define DAYAN_MMU_TLB_FLUSH_GO          0x1U
#define DAYAN_MMU_TLB_FLUSH_ASID_SEL    0x2U
#define DAYAN_MMU_TLB_FLUSH_ASID_SHIFT  2U
#define DAYAN_MMU_TLB_FLUSH_ASID_MASK   0xffU

#define DAYAN_MMU_FLUSH_BUSY_BIT        0x1U
#define DAYAN_MMU_FLUSH_DONE_BIT        0x1U
#define DAYAN_MMU_FLUSH_ACK_BIT         0x1U
#define DAYAN_MMU_FLUSH_ERR_MASK        0xfU
#define DAYAN_MMU_FLUSH_ERR_BUSY_REJECT 0x1U

#define DAYAN_MMU_N_FIX_MAP_EN_BIT      0x1U
/* bit0: 0 substitutes the whole address, 1 adds VA[11:0] to the base. */
#define DAYAN_MMU_ERROR_SLV_OFF_VA      0x1U
#define DAYAN_MMU_ERROR_SLV_CFG_ERR_BIT 0x1U

/* Only 8 ASIDs exist and all ports share GLOBAL_ASID, so Contexts cannot
 * be isolated by ASID (manual section 2.2.8 clause 3). */
#define DAYAN_MMU_ASID_COUNT            8U

#define DAYAN_MMU_PA_BITS               56U
#define DAYAN_MMU_PA_HI_MASK            0x00ffffffU
#define DAYAN_MMU_PAGE_SHIFT            12U
#define DAYAN_MMU_PAGE_SIZE             0x1000U
#define DAYAN_MMU_PAGE_MASK             0xfffU

#endif
