#ifndef NPU_PLATFORM_REGS_H
#define NPU_PLATFORM_REGS_H

/* Offsets relative to CONFIG_CONTROL_BASE, from the Dayan register list. */
#define DAYAN_DEVICE_MAILBOX_BASE       0x001000U
#define DAYAN_IRQ_STATUS                0x002000U
#define DAYAN_IRQ_STATUS_SET            0x002008U
#define DAYAN_IRQ_STATUS_CLR            0x00200cU
#define DAYAN_IRQ_ENABLE_SET            0x002010U
#define DAYAN_IRQ_MAILBOX_BASE          0x002020U
#define DAYAN_CLINT_MSIP_BASE            0x020000U

#define DAYAN_DEVICE_MAILBOX(hart) \
    (DAYAN_DEVICE_MAILBOX_BASE + (hart) * 8U)
#define DAYAN_IRQ_MAILBOX(hart) \
    (DAYAN_IRQ_MAILBOX_BASE + (hart) * 8U)
#define DAYAN_CLINT_MSIP(hart) \
    (DAYAN_CLINT_MSIP_BASE + (hart) * 4U)

#endif
