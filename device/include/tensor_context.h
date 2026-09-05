#ifndef DAYAN_TENSOR_CONTEXT_H
#define DAYAN_TENSOR_CONTEXT_H

#include <stdint.h>

#define DAYAN_CONTEXT_URAM_GRANULE 32768ULL
#define DAYAN_CONTEXT_URAM_WINDOW  0x200000ULL
#define DAYAN_CONTEXT_TCSR_MAX_COUNT 8U
#define DAYAN_CONTEXT_TCSR_MAX_BASE  63U
#define DAYAN_CONTEXT_RESOURCE_COUNT 32U

struct dayan_tcsr_range {
    uint8_t base;
    uint8_t count;
};

struct dayan_context_fields {
    uint8_t uram_lo;
    uint8_t uram_hi;
    struct dayan_tcsr_range lsu;
    struct dayan_tcsr_range mat;
    struct dayan_tcsr_range vec;
    struct dayan_tcsr_range macro;
};

struct dayan_resource_range {
    uint8_t base;
    uint8_t count;
};

struct dayan_resource_context_fields {
    struct dayan_resource_range local;
    struct dayan_resource_range shared;
};

static inline int dayan_tcsr_range_valid(struct dayan_tcsr_range range)
{
    return range.count <= DAYAN_CONTEXT_TCSR_MAX_COUNT &&
           range.base <= DAYAN_CONTEXT_TCSR_MAX_BASE &&
           (!range.count || range.base <=
            DAYAN_CONTEXT_TCSR_MAX_BASE + 1U - range.count);
}

static inline int dayan_context_fields_valid(
    const struct dayan_context_fields *fields)
{
    return fields && fields->uram_lo <= fields->uram_hi &&
           dayan_tcsr_range_valid(fields->lsu) &&
           dayan_tcsr_range_valid(fields->mat) &&
           dayan_tcsr_range_valid(fields->vec) &&
           dayan_tcsr_range_valid(fields->macro);
}

static inline uint64_t dayan_context_pack(
    const struct dayan_context_fields *fields)
{
    if (!dayan_context_fields_valid(fields))
        return UINT64_MAX;
    return (uint64_t)fields->uram_lo |
           ((uint64_t)fields->uram_hi << 8) |
           ((uint64_t)fields->lsu.count << 16) |
           ((uint64_t)fields->lsu.base << 20) |
           ((uint64_t)fields->mat.count << 26) |
           ((uint64_t)fields->mat.base << 30) |
           ((uint64_t)fields->vec.count << 36) |
           ((uint64_t)fields->vec.base << 40) |
           ((uint64_t)fields->macro.count << 46) |
           ((uint64_t)fields->macro.base << 50);
}

static inline struct dayan_context_fields dayan_context_unpack(uint64_t value)
{
    struct dayan_context_fields fields;

    fields.uram_lo = (uint8_t)value;
    fields.uram_hi = (uint8_t)(value >> 8);
    fields.lsu.count = (uint8_t)((value >> 16) & 0xfU);
    fields.lsu.base = (uint8_t)((value >> 20) & 0x3fU);
    fields.mat.count = (uint8_t)((value >> 26) & 0xfU);
    fields.mat.base = (uint8_t)((value >> 30) & 0x3fU);
    fields.vec.count = (uint8_t)((value >> 36) & 0xfU);
    fields.vec.base = (uint8_t)((value >> 40) & 0x3fU);
    fields.macro.count = (uint8_t)((value >> 46) & 0xfU);
    fields.macro.base = (uint8_t)((value >> 50) & 0x3fU);
    return fields;
}

static inline int dayan_resource_range_valid(struct dayan_resource_range range)
{
    return range.count <= DAYAN_CONTEXT_RESOURCE_COUNT &&
           range.base < DAYAN_CONTEXT_RESOURCE_COUNT &&
           (!range.count || range.base <=
            DAYAN_CONTEXT_RESOURCE_COUNT - range.count);
}

static inline int dayan_resource_context_fields_valid(
    const struct dayan_resource_context_fields *fields)
{
    return fields && dayan_resource_range_valid(fields->local) &&
           dayan_resource_range_valid(fields->shared);
}

static inline uint32_t dayan_resource_context_pack(
    const struct dayan_resource_context_fields *fields)
{
    if (!dayan_resource_context_fields_valid(fields))
        return UINT32_MAX;
    return (uint32_t)fields->local.base |
           ((uint32_t)fields->local.count << 5) |
           ((uint32_t)fields->shared.base << 11) |
           ((uint32_t)fields->shared.count << 16);
}

static inline struct dayan_resource_context_fields
dayan_resource_context_unpack(uint32_t value)
{
    struct dayan_resource_context_fields fields;

    fields.local.base = (uint8_t)(value & 0x1fU);
    fields.local.count = (uint8_t)((value >> 5) & 0x3fU);
    fields.shared.base = (uint8_t)((value >> 11) & 0x1fU);
    fields.shared.count = (uint8_t)((value >> 16) & 0x3fU);
    return fields;
}

#endif
