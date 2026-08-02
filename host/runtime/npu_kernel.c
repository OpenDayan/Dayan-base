#include "runtime_internal.h"

#include <elf.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dayan_kernel.h"

struct dayan_kernel_descriptor {
    const void *object;
    size_t object_size;
    char *entry_symbol;
};

struct elf_view {
    const unsigned char *data;
    size_t size;
    Elf64_Ehdr header;
};

struct load_section {
    uint64_t offset;
    uint64_t size;
    uint64_t address;
    uint64_t alignment;
    uint64_t flags;
    uint32_t type;
    int loaded;
};

struct relocation_context {
    const struct elf_view *elf;
    const struct load_section *sections;
    size_t section_count;
    Elf64_Shdr symbol_section;
    Elf64_Shdr string_section;
    const uint64_t *symbol_values;
    const unsigned char *symbol_valid;
    size_t symbol_count;
    unsigned char *image;
    uint64_t image_size;
    uint64_t base;
};

static int range_valid(size_t total, uint64_t offset, uint64_t size)
{
    return offset <= total && size <= (uint64_t)total - offset;
}

static int elf_view_init(struct elf_view *view, const void *image, size_t size,
                         uint16_t expected_type)
{
    if (!view || !image || size < sizeof(Elf64_Ehdr))
        return -ENOEXEC;
    memset(view, 0, sizeof(*view));
    view->data = image;
    view->size = size;
    memcpy(&view->header, image, sizeof(view->header));
    if (memcmp(view->header.e_ident, ELFMAG, SELFMAG) != 0 ||
        view->header.e_ident[EI_CLASS] != ELFCLASS64 ||
        view->header.e_ident[EI_DATA] != ELFDATA2LSB ||
        view->header.e_ident[EI_VERSION] != EV_CURRENT ||
        view->header.e_ident[EI_OSABI] != ELFOSABI_NONE ||
        view->header.e_type != expected_type ||
        view->header.e_machine != EM_RISCV ||
        view->header.e_version != EV_CURRENT ||
        view->header.e_ehsize != sizeof(Elf64_Ehdr) ||
        view->header.e_shentsize != sizeof(Elf64_Shdr) ||
        !view->header.e_shnum || view->header.e_shnum == SHN_UNDEF ||
        view->header.e_shstrndx == SHN_XINDEX ||
        !range_valid(size, view->header.e_shoff,
                     (uint64_t)view->header.e_shnum * sizeof(Elf64_Shdr)))
        return -ENOEXEC;
    return 0;
}

static int elf_section(const struct elf_view *view, size_t index,
                       Elf64_Shdr *section)
{
    uint64_t offset;

    if (!view || !section || index >= view->header.e_shnum)
        return -ENOEXEC;
    offset = view->header.e_shoff + index * sizeof(*section);
    memcpy(section, view->data + offset, sizeof(*section));
    return 0;
}

static int elf_symbol(const struct elf_view *view,
                      const Elf64_Shdr *symbol_section, size_t index,
                      Elf64_Sym *symbol)
{
    uint64_t offset;
    size_t count;

    if (!view || !symbol_section || !symbol ||
        symbol_section->sh_entsize != sizeof(*symbol) ||
        symbol_section->sh_size % sizeof(*symbol) != 0 ||
        !range_valid(view->size, symbol_section->sh_offset,
                     symbol_section->sh_size))
        return -ENOEXEC;
    count = symbol_section->sh_size / sizeof(*symbol);
    if (index >= count)
        return -ENOEXEC;
    offset = symbol_section->sh_offset + index * sizeof(*symbol);
    memcpy(symbol, view->data + offset, sizeof(*symbol));
    return 0;
}

static int elf_rela(const struct elf_view *view,
                    const Elf64_Shdr *rela_section, size_t index,
                    Elf64_Rela *relocation)
{
    uint64_t offset;
    size_t count;

    if (!view || !rela_section || !relocation ||
        rela_section->sh_entsize != sizeof(*relocation) ||
        rela_section->sh_size % sizeof(*relocation) != 0 ||
        !range_valid(view->size, rela_section->sh_offset,
                     rela_section->sh_size))
        return -ENOEXEC;
    count = rela_section->sh_size / sizeof(*relocation);
    if (index >= count)
        return -ENOEXEC;
    offset = rela_section->sh_offset + index * sizeof(*relocation);
    memcpy(relocation, view->data + offset, sizeof(*relocation));
    return 0;
}

static const char *elf_string(const struct elf_view *view,
                              const Elf64_Shdr *strings, uint32_t offset)
{
    const char *value;
    size_t available;

    if (!view || !strings || strings->sh_type != SHT_STRTAB ||
        offset >= strings->sh_size ||
        !range_valid(view->size, strings->sh_offset, strings->sh_size))
        return 0;
    value = (const char *)(view->data + strings->sh_offset + offset);
    available = strings->sh_size - offset;
    return memchr(value, 0, available) ? value : 0;
}

static int api_symbol_compare(const void *left, const void *right)
{
    const struct npu_api_symbol *a = left;
    const struct npu_api_symbol *b = right;

    return strcmp(a->name, b->name);
}

static void free_api_symbols(struct npu_api_symbol *symbols, size_t count)
{
    size_t index;

    for (index = 0; index < count; ++index)
        free(symbols[index].name);
    free(symbols);
}

static int append_api_symbol(struct npu_api_symbol **symbols, size_t *count,
                             const char *name, uint64_t value)
{
    struct npu_api_symbol *resized;
    size_t index;

    for (index = 0; index < *count; ++index) {
        if (strcmp((*symbols)[index].name, name) == 0)
            return (*symbols)[index].value == value ? 0 : -ENOEXEC;
    }
    if (*count == SIZE_MAX / sizeof(**symbols))
        return -EOVERFLOW;
    resized = realloc(*symbols, (*count + 1) * sizeof(**symbols));
    if (!resized)
        return -ENOMEM;
    *symbols = resized;
    resized[*count].name = strdup(name);
    if (!resized[*count].name)
        return -ENOMEM;
    resized[*count].value = value;
    ++*count;
    return 0;
}

static int parse_api_symbols(const void *image, size_t size,
                             struct npu_api_symbol **symbols_out,
                             size_t *count_out)
{
    struct npu_api_symbol *symbols = 0;
    struct elf_view view;
    size_t count = 0;
    size_t section_index;
    int status;

    if (!symbols_out || !count_out)
        return -EINVAL;
    *symbols_out = 0;
    *count_out = 0;
    status = elf_view_init(&view, image, size, ET_EXEC);
    if (status < 0)
        return status;
    for (section_index = 0; section_index < view.header.e_shnum;
         ++section_index) {
        Elf64_Shdr symbol_section;
        Elf64_Shdr string_section;
        size_t symbol_count;
        size_t symbol_index;

        if (elf_section(&view, section_index, &symbol_section) < 0)
            goto malformed;
        if (symbol_section.sh_type != SHT_SYMTAB &&
            symbol_section.sh_type != SHT_DYNSYM)
            continue;
        if (symbol_section.sh_link >= view.header.e_shnum ||
            symbol_section.sh_entsize != sizeof(Elf64_Sym) ||
            symbol_section.sh_size % sizeof(Elf64_Sym) != 0 ||
            elf_section(&view, symbol_section.sh_link, &string_section) < 0 ||
            string_section.sh_type != SHT_STRTAB)
            goto malformed;
        symbol_count = symbol_section.sh_size / sizeof(Elf64_Sym);
        for (symbol_index = 1; symbol_index < symbol_count; ++symbol_index) {
            Elf64_Sym symbol;
            unsigned binding;
            unsigned type;
            const char *name;

            if (elf_symbol(&view, &symbol_section, symbol_index, &symbol) < 0)
                goto malformed;
            binding = ELF64_ST_BIND(symbol.st_info);
            type = ELF64_ST_TYPE(symbol.st_info);
            if (symbol.st_shndx == SHN_UNDEF ||
                (binding != STB_GLOBAL && binding != STB_WEAK) ||
                (type != STT_FUNC && type != STT_OBJECT &&
                 type != STT_NOTYPE))
                continue;
            name = elf_string(&view, &string_section, symbol.st_name);
            if (!name || !*name)
                continue;
            status = append_api_symbol(&symbols, &count, name,
                                       symbol.st_value);
            if (status < 0)
                goto fail;
        }
    }
    if (!count) {
        status = -ENOEXEC;
        goto fail;
    }
    qsort(symbols, count, sizeof(*symbols), api_symbol_compare);
    *symbols_out = symbols;
    *count_out = count;
    return 0;

malformed:
    status = -ENOEXEC;
fail:
    free_api_symbols(symbols, count);
    return status;
}

static int lookup_api_symbol(const npu_runtime_t *runtime, const char *name,
                             uint64_t *value_out)
{
    size_t low = 0;
    size_t high = runtime->api_symbol_count;

    while (low < high) {
        size_t middle = low + (high - low) / 2;
        int order = strcmp(name, runtime->api_symbols[middle].name);

        if (order < 0)
            high = middle;
        else if (order > 0)
            low = middle + 1;
        else {
            *value_out = runtime->api_symbols[middle].value;
            return 0;
        }
    }
    return -ENOENT;
}

int dayan_runtime_set_api_elf(npu_runtime_t *runtime,
                              const void *elf_image, size_t elf_size)
{
    struct npu_api_symbol *symbols;
    size_t count;
    int status;

    if (!runtime || !elf_image || !elf_size)
        return -EINVAL;
    status = parse_api_symbols(elf_image, elf_size, &symbols, &count);
    if (status < 0)
        return status;
    pthread_mutex_lock(&runtime->kernel_mutex);
    if (runtime->kernel_modules) {
        pthread_mutex_unlock(&runtime->kernel_mutex);
        free_api_symbols(symbols, count);
        return -EBUSY;
    }
    free_api_symbols(runtime->api_symbols, runtime->api_symbol_count);
    runtime->api_symbols = symbols;
    runtime->api_symbol_count = count;
    pthread_mutex_unlock(&runtime->kernel_mutex);
    return 0;
}

int dayan_runtime_set_api_elf_file(npu_runtime_t *runtime, const char *path)
{
    unsigned char *image = 0;
    long length;
    FILE *file;
    int status;

    if (!runtime || !path)
        return -EINVAL;
    file = fopen(path, "rb");
    if (!file)
        return -errno;
    if (fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) <= 0 || fseek(file, 0, SEEK_SET) != 0 ||
        (uint64_t)length > SIZE_MAX) {
        status = -EIO;
        goto out;
    }
    image = malloc((size_t)length);
    if (!image) {
        status = -ENOMEM;
        goto out;
    }
    if (fread(image, 1, (size_t)length, file) != (size_t)length) {
        status = -EIO;
        goto out;
    }
    status = dayan_runtime_set_api_elf(runtime, image, (size_t)length);
out:
    free(image);
    fclose(file);
    return status;
}

static int is_power_of_two_u64(uint64_t value)
{
    return value && !(value & (value - 1));
}

static int align_up_u64(uint64_t value, uint64_t alignment,
                        uint64_t *result)
{
    uint64_t mask;

    if (!is_power_of_two_u64(alignment))
        return -ENOEXEC;
    mask = alignment - 1;
    if (value > UINT64_MAX - mask)
        return -EOVERFLOW;
    *result = (value + mask) & ~mask;
    return 0;
}

static int layout_sections(const struct elf_view *elf,
                           struct load_section **sections_out,
                           uint64_t *image_size_out,
                           uint64_t *alignment_out)
{
    struct load_section *sections;
    uint64_t cursor = 0;
    uint64_t maximum_alignment = 16;
    size_t index;
    int status;

    sections = calloc(elf->header.e_shnum, sizeof(*sections));
    if (!sections)
        return -ENOMEM;
    for (index = 1; index < elf->header.e_shnum; ++index) {
        Elf64_Shdr section;
        uint64_t alignment;

        if (elf_section(elf, index, &section) < 0) {
            status = -ENOEXEC;
            goto fail;
        }
        if (!(section.sh_flags & SHF_ALLOC))
            continue;
        if ((section.sh_flags & SHF_TLS) || section.sh_type == SHT_INIT_ARRAY ||
            section.sh_type == SHT_FINI_ARRAY ||
            (section.sh_type != SHT_PROGBITS &&
             section.sh_type != SHT_NOBITS)) {
            status = -ENOEXEC;
            goto fail;
        }
        alignment = section.sh_addralign ? section.sh_addralign : 1;
        if (!is_power_of_two_u64(alignment) || alignment > SIZE_MAX) {
            status = -ENOEXEC;
            goto fail;
        }
        if (section.sh_type != SHT_NOBITS &&
            !range_valid(elf->size, section.sh_offset, section.sh_size)) {
            status = -ENOEXEC;
            goto fail;
        }
        status = align_up_u64(cursor, alignment, &cursor);
        if (status < 0 || section.sh_size > UINT64_MAX - cursor) {
            status = -EOVERFLOW;
            goto fail;
        }
        sections[index].offset = cursor;
        sections[index].size = section.sh_size;
        sections[index].alignment = alignment;
        sections[index].flags = section.sh_flags;
        sections[index].type = section.sh_type;
        sections[index].loaded = 1;
        cursor += section.sh_size;
        if (alignment > maximum_alignment)
            maximum_alignment = alignment;
    }
    if (!cursor || cursor > SIZE_MAX) {
        status = -ENOEXEC;
        goto fail;
    }
    *sections_out = sections;
    *image_size_out = cursor;
    *alignment_out = maximum_alignment;
    return 0;
fail:
    free(sections);
    return status;
}

static int find_symbol_table(const struct elf_view *elf,
                             Elf64_Shdr *symbols, Elf64_Shdr *strings,
                             size_t *symbol_count)
{
    size_t index;

    for (index = 1; index < elf->header.e_shnum; ++index) {
        if (elf_section(elf, index, symbols) < 0)
            return -ENOEXEC;
        if (symbols->sh_type != SHT_SYMTAB)
            continue;
        if (symbols->sh_link >= elf->header.e_shnum ||
            symbols->sh_entsize != sizeof(Elf64_Sym) ||
            symbols->sh_size % sizeof(Elf64_Sym) != 0 ||
            !range_valid(elf->size, symbols->sh_offset, symbols->sh_size) ||
            elf_section(elf, symbols->sh_link, strings) < 0 ||
            strings->sh_type != SHT_STRTAB ||
            !range_valid(elf->size, strings->sh_offset, strings->sh_size))
            return -ENOEXEC;
        *symbol_count = symbols->sh_size / sizeof(Elf64_Sym);
        return *symbol_count ? 0 : -ENOEXEC;
    }
    return -ENOEXEC;
}

static void free_kernel_exports(struct npu_kernel_export *exports,
                                size_t count)
{
    size_t index;

    for (index = 0; index < count; ++index)
        free(exports[index].name);
    free(exports);
}

static int append_kernel_export(struct npu_kernel_export **exports,
                                size_t *count, const char *name,
                                uint64_t value)
{
    struct npu_kernel_export *resized;

    if (*count == SIZE_MAX / sizeof(**exports))
        return -EOVERFLOW;
    resized = realloc(*exports, (*count + 1) * sizeof(**exports));
    if (!resized)
        return -ENOMEM;
    *exports = resized;
    resized[*count].name = strdup(name);
    if (!resized[*count].name)
        return -ENOMEM;
    resized[*count].value = value;
    ++*count;
    return 0;
}

static int resolve_symbols(npu_runtime_t *runtime,
                           const struct elf_view *elf,
                           const struct load_section *sections,
                           uint64_t base, const Elf64_Shdr *symbol_section,
                           const Elf64_Shdr *string_section,
                           size_t symbol_count, uint64_t **values_out,
                           unsigned char **valid_out,
                           struct npu_kernel_export **exports_out,
                           size_t *export_count_out)
{
    uint64_t *values;
    unsigned char *valid;
    struct npu_kernel_export *exports = 0;
    size_t export_count = 0;
    size_t index;
    int status = 0;

    values = calloc(symbol_count, sizeof(*values));
    valid = calloc(symbol_count, sizeof(*valid));
    if (!values || !valid) {
        status = -ENOMEM;
        goto fail;
    }
    valid[0] = 1;
    for (index = 1; index < symbol_count; ++index) {
        Elf64_Sym symbol;
        const char *name;

        if (elf_symbol(elf, symbol_section, index, &symbol) < 0) {
            status = -ENOEXEC;
            goto fail;
        }
        name = elf_string(elf, string_section, symbol.st_name);
        if (!name) {
            status = -ENOEXEC;
            goto fail;
        }
        if (symbol.st_shndx == SHN_UNDEF) {
            status = lookup_api_symbol(runtime, name, &values[index]);
            if (status == -ENOENT && ELF64_ST_BIND(symbol.st_info) == STB_WEAK) {
                values[index] = 0;
                status = 0;
            }
            if (status < 0)
                goto fail;
            valid[index] = 1;
        } else if (symbol.st_shndx == SHN_ABS) {
            values[index] = symbol.st_value;
            valid[index] = 1;
        } else if (symbol.st_shndx == SHN_COMMON ||
                   symbol.st_shndx >= elf->header.e_shnum) {
            status = -ENOEXEC;
            goto fail;
        } else if (sections[symbol.st_shndx].loaded) {
            if (symbol.st_value > sections[symbol.st_shndx].size) {
                status = -ENOEXEC;
                goto fail;
            }
            if (symbol.st_size >
                sections[symbol.st_shndx].size - symbol.st_value ||
                (ELF64_ST_TYPE(symbol.st_info) == STT_FUNC &&
                 symbol.st_value == sections[symbol.st_shndx].size)) {
                status = -ENOEXEC;
                goto fail;
            }
            values[index] = base + sections[symbol.st_shndx].offset +
                            symbol.st_value;
            valid[index] = 1;
            if (*name && ELF64_ST_TYPE(symbol.st_info) == STT_FUNC) {
                status = append_kernel_export(&exports, &export_count, name,
                                              values[index]);
                if (status < 0)
                    goto fail;
            }
        }
    }
    *values_out = values;
    *valid_out = valid;
    *exports_out = exports;
    *export_count_out = export_count;
    return 0;
fail:
    free(values);
    free(valid);
    free_kernel_exports(exports, export_count);
    return status;
}

static uint16_t read_u16(const void *address)
{
    uint16_t value;

    memcpy(&value, address, sizeof(value));
    return value;
}

static uint32_t read_u32(const void *address)
{
    uint32_t value;

    memcpy(&value, address, sizeof(value));
    return value;
}

static uint64_t read_u64(const void *address)
{
    uint64_t value;

    memcpy(&value, address, sizeof(value));
    return value;
}

static void write_u16(void *address, uint16_t value)
{
    memcpy(address, &value, sizeof(value));
}

static void write_u32(void *address, uint32_t value)
{
    memcpy(address, &value, sizeof(value));
}

static void write_u64(void *address, uint64_t value)
{
    memcpy(address, &value, sizeof(value));
}

static int signed_fits(__int128 value, unsigned bits)
{
    __int128 minimum = -((__int128)1 << (bits - 1));
    __int128 maximum = ((__int128)1 << (bits - 1)) - 1;

    return value >= minimum && value <= maximum;
}

static int patch_upper(void *where, __int128 value)
{
    uint32_t instruction;
    __int128 high;

    high = (value + 0x800) >> 12;
    if (!signed_fits(high, 20))
        return -ERANGE;
    instruction = read_u32(where);
    instruction = (instruction & 0xfffU) |
                  ((uint32_t)((uint64_t)high & 0xfffffU) << 12);
    write_u32(where, instruction);
    return 0;
}

static void patch_lower_i(void *where, __int128 value)
{
    uint32_t instruction = read_u32(where);

    instruction = (instruction & 0x000fffffU) |
                  ((uint32_t)((uint64_t)value & 0xfffU) << 20);
    write_u32(where, instruction);
}

static void patch_lower_s(void *where, __int128 value)
{
    uint32_t instruction = read_u32(where);
    uint32_t low = (uint32_t)((uint64_t)value & 0xfffU);

    instruction = (instruction & 0x01fff07fU) |
                  ((low & 0xfe0U) << 20) | ((low & 0x1fU) << 7);
    write_u32(where, instruction);
}

static int patch_branch(void *where, __int128 value)
{
    uint32_t instruction;
    uint32_t immediate;

    if (((uint64_t)value & 1U) || !signed_fits(value, 13))
        return -ERANGE;
    immediate = (uint32_t)((uint64_t)value & 0x1fffU);
    instruction = read_u32(where) & ~UINT32_C(0xfe000f80);
    instruction |= ((immediate >> 12) & 1U) << 31;
    instruction |= ((immediate >> 5) & 0x3fU) << 25;
    instruction |= ((immediate >> 1) & 0xfU) << 8;
    instruction |= ((immediate >> 11) & 1U) << 7;
    write_u32(where, instruction);
    return 0;
}

static int patch_jal(void *where, __int128 value)
{
    uint32_t instruction;
    uint32_t immediate;

    if (((uint64_t)value & 1U) || !signed_fits(value, 21))
        return -ERANGE;
    immediate = (uint32_t)((uint64_t)value & 0x1fffffU);
    instruction = read_u32(where) & 0xfffU;
    instruction |= ((immediate >> 20) & 1U) << 31;
    instruction |= ((immediate >> 1) & 0x3ffU) << 21;
    instruction |= ((immediate >> 11) & 1U) << 20;
    instruction |= ((immediate >> 12) & 0xffU) << 12;
    write_u32(where, instruction);
    return 0;
}

static int patch_call(void *where, __int128 value)
{
    int status;

    if (!signed_fits(value, 32))
        return -ERANGE;
    status = patch_upper(where, value);
    if (status < 0)
        return status;
    patch_lower_i((unsigned char *)where + 4, value);
    return 0;
}

static int relocation_symbol(const struct relocation_context *context,
                             const Elf64_Rela *relocation,
                             uint64_t *value_out)
{
    size_t index = ELF64_R_SYM(relocation->r_info);

    if (index >= context->symbol_count || !context->symbol_valid[index])
        return -ENOEXEC;
    *value_out = context->symbol_values[index];
    return 0;
}

static int find_pcrel_high(const struct relocation_context *context,
                           const Elf64_Shdr *rela_section,
                           size_t target_index, uint64_t anchor,
                           __int128 low_addend, __int128 *value_out)
{
    size_t count = rela_section->sh_size / sizeof(Elf64_Rela);
    size_t index;

    for (index = 0; index < count; ++index) {
        Elf64_Rela candidate;
        uint64_t symbol;
        uint64_t place;
        unsigned type;

        if (elf_rela(context->elf, rela_section, index, &candidate) < 0)
            return -ENOEXEC;
        type = ELF64_R_TYPE(candidate.r_info);
        if (type != R_RISCV_PCREL_HI20)
            continue;
        place = context->base + context->sections[target_index].offset +
                candidate.r_offset;
        if (place != anchor)
            continue;
        if (relocation_symbol(context, &candidate, &symbol) < 0)
            return -ENOEXEC;
        *value_out = (__int128)symbol + candidate.r_addend + low_addend -
                     place;
        return 0;
    }
    return -ENOEXEC;
}

static int apply_one_relocation(const struct relocation_context *context,
                                const Elf64_Shdr *rela_section,
                                size_t target_index,
                                const Elf64_Rela *relocation)
{
    const struct load_section *target = &context->sections[target_index];
    unsigned type = ELF64_R_TYPE(relocation->r_info);
    uint64_t symbol;
    uint64_t place;
    uint64_t width;
    unsigned char *where;
    __int128 value;
    int status;

    switch (type) {
    case R_RISCV_NONE:
    case R_RISCV_RELAX:
        return 0;
    case R_RISCV_64:
    case R_RISCV_ADD64:
    case R_RISCV_SUB64:
        width = 8;
        break;
    case R_RISCV_CALL:
    case R_RISCV_CALL_PLT:
        width = 8;
        break;
    case R_RISCV_32:
    case R_RISCV_BRANCH:
    case R_RISCV_JAL:
    case R_RISCV_PCREL_HI20:
    case R_RISCV_PCREL_LO12_I:
    case R_RISCV_PCREL_LO12_S:
    case R_RISCV_HI20:
    case R_RISCV_LO12_I:
    case R_RISCV_LO12_S:
    case R_RISCV_ADD32:
    case R_RISCV_SUB32:
    case R_RISCV_SET32:
    case R_RISCV_32_PCREL:
        width = 4;
        break;
    case R_RISCV_ADD16:
    case R_RISCV_SUB16:
    case R_RISCV_SET16:
        width = 2;
        break;
    case R_RISCV_ADD8:
    case R_RISCV_SUB8:
    case R_RISCV_SET8:
    case R_RISCV_SUB6:
    case R_RISCV_SET6:
        width = 1;
        break;
    default:
        return -ENOTSUP;
    }
    if (relocation->r_offset > target->size ||
        width > target->size - relocation->r_offset ||
        target->offset > context->image_size ||
        relocation->r_offset > context->image_size - target->offset ||
        width > context->image_size - target->offset - relocation->r_offset)
        return -ENOEXEC;
    where = context->image + target->offset + relocation->r_offset;
    place = context->base + target->offset + relocation->r_offset;
    status = relocation_symbol(context, relocation, &symbol);
    if (status < 0)
        return status;
    value = (__int128)symbol + relocation->r_addend;

    switch (type) {
    case R_RISCV_NONE:
    case R_RISCV_RELAX:
        return 0;
    case R_RISCV_64:
        if (value < 0 || value > UINT64_MAX)
            return -ERANGE;
        write_u64(where, (uint64_t)value);
        return 0;
    case R_RISCV_32:
        if (value < INT32_MIN || value > UINT32_MAX)
            return -ERANGE;
        write_u32(where, (uint32_t)value);
        return 0;
    case R_RISCV_BRANCH:
        return patch_branch(where, value - place);
    case R_RISCV_JAL:
        return patch_jal(where, value - place);
    case R_RISCV_CALL:
    case R_RISCV_CALL_PLT:
        return patch_call(where, value - place);
    case R_RISCV_PCREL_HI20:
        return patch_upper(where, value - place);
    case R_RISCV_PCREL_LO12_I:
    case R_RISCV_PCREL_LO12_S: {
        __int128 low_value;

        status = find_pcrel_high(context, rela_section, target_index, symbol,
                                 relocation->r_addend, &low_value);
        if (status < 0)
            return status;
        if (type == R_RISCV_PCREL_LO12_I)
            patch_lower_i(where, low_value);
        else
            patch_lower_s(where, low_value);
        return 0;
    }
    case R_RISCV_HI20:
        return patch_upper(where, value);
    case R_RISCV_LO12_I:
        patch_lower_i(where, value);
        return 0;
    case R_RISCV_LO12_S:
        patch_lower_s(where, value);
        return 0;
    case R_RISCV_ADD8:
        *where = (unsigned char)(*where + (uint8_t)value);
        return 0;
    case R_RISCV_ADD16:
        write_u16(where, (uint16_t)(read_u16(where) + (uint16_t)value));
        return 0;
    case R_RISCV_ADD32:
        write_u32(where, read_u32(where) + (uint32_t)value);
        return 0;
    case R_RISCV_ADD64:
        write_u64(where, read_u64(where) + (uint64_t)value);
        return 0;
    case R_RISCV_SUB8:
        *where = (unsigned char)(*where - (uint8_t)value);
        return 0;
    case R_RISCV_SUB16:
        write_u16(where, (uint16_t)(read_u16(where) - (uint16_t)value));
        return 0;
    case R_RISCV_SUB32:
        write_u32(where, read_u32(where) - (uint32_t)value);
        return 0;
    case R_RISCV_SUB64:
        write_u64(where, read_u64(where) - (uint64_t)value);
        return 0;
    case R_RISCV_SET6:
        *where = (unsigned char)((*where & 0xc0U) | ((uint8_t)value & 0x3fU));
        return 0;
    case R_RISCV_SUB6:
        *where = (unsigned char)((*where & 0xc0U) |
                 (((*where & 0x3fU) - (uint8_t)value) & 0x3fU));
        return 0;
    case R_RISCV_SET8:
        *where = (uint8_t)value;
        return 0;
    case R_RISCV_SET16:
        write_u16(where, (uint16_t)value);
        return 0;
    case R_RISCV_SET32:
        write_u32(where, (uint32_t)value);
        return 0;
    case R_RISCV_32_PCREL:
        value -= place;
        if (!signed_fits(value, 32))
            return -ERANGE;
        write_u32(where, (uint32_t)value);
        return 0;
    default:
        return -ENOTSUP;
    }
}

static int apply_relocations(const struct relocation_context *context,
                             size_t symbol_section_index)
{
    size_t section_index;

    for (section_index = 1; section_index < context->section_count;
         ++section_index) {
        Elf64_Shdr relocation_section;
        size_t target_index;
        size_t count;
        size_t relocation_index;

        if (elf_section(context->elf, section_index, &relocation_section) < 0)
            return -ENOEXEC;
        if (relocation_section.sh_type == SHT_REL)
            return -ENOTSUP;
        if (relocation_section.sh_type != SHT_RELA)
            continue;
        if (relocation_section.sh_info >= context->section_count ||
            relocation_section.sh_link != symbol_section_index)
            return -ENOEXEC;
        target_index = relocation_section.sh_info;
        if (!context->sections[target_index].loaded)
            continue;
        if (relocation_section.sh_entsize != sizeof(Elf64_Rela) ||
            relocation_section.sh_size % sizeof(Elf64_Rela) != 0 ||
            !range_valid(context->elf->size, relocation_section.sh_offset,
                         relocation_section.sh_size))
            return -ENOEXEC;
        count = relocation_section.sh_size / sizeof(Elf64_Rela);
        for (relocation_index = 0; relocation_index < count;
             ++relocation_index) {
            Elf64_Rela relocation;
            int status;

            if (elf_rela(context->elf, &relocation_section,
                         relocation_index, &relocation) < 0)
                return -ENOEXEC;
            status = apply_one_relocation(context, &relocation_section,
                                          target_index, &relocation);
            if (status < 0)
                return status;
        }
    }
    return 0;
}

static int load_kernel_module(npu_runtime_t *runtime, const void *object,
                              size_t object_size,
                              struct npu_kernel_module **module_out)
{
    struct npu_kernel_module *module = 0;
    struct npu_kernel_export *exports = 0;
    struct load_section *sections = 0;
    struct elf_view elf;
    struct relocation_context relocations;
    Elf64_Shdr symbol_section;
    Elf64_Shdr string_section;
    uint64_t *symbol_values = 0;
    unsigned char *symbol_valid = 0;
    unsigned char *image = 0;
    uint64_t image_size;
    uint64_t alignment;
    uint64_t base = 0;
    size_t symbol_count;
    size_t export_count = 0;
    size_t symbol_section_index = 0;
    size_t index;
    int allocated = 0;
    int status;

    status = elf_view_init(&elf, object, object_size, ET_REL);
    if (status < 0)
        return status;
    status = layout_sections(&elf, &sections, &image_size, &alignment);
    if (status < 0)
        goto fail;
    status = find_symbol_table(&elf, &symbol_section, &string_section,
                               &symbol_count);
    if (status < 0)
        goto fail;
    for (index = 1; index < elf.header.e_shnum; ++index) {
        Elf64_Shdr candidate;

        if (elf_section(&elf, index, &candidate) < 0) {
            status = -ENOEXEC;
            goto fail;
        }
        if (candidate.sh_offset == symbol_section.sh_offset &&
            candidate.sh_type == SHT_SYMTAB) {
            symbol_section_index = index;
            break;
        }
    }
    if (!symbol_section_index) {
        status = -ENOEXEC;
        goto fail;
    }

    pthread_mutex_lock(&runtime->mutex);
    if (!runtime->firmware_loaded) {
        status = -ENODEV;
    } else {
        status = npu_arena_alloc(&runtime->normal, image_size, alignment,
                                 NPU_ALLOC_KERNEL, UINT32_MAX, &base);
    }
    pthread_mutex_unlock(&runtime->mutex);
    if (status < 0)
        goto fail;
    allocated = 1;
    for (index = 1; index < elf.header.e_shnum; ++index) {
        if (sections[index].loaded)
            sections[index].address = base + sections[index].offset;
    }
    image = calloc(1, (size_t)image_size);
    if (!image) {
        status = -ENOMEM;
        goto fail;
    }
    for (index = 1; index < elf.header.e_shnum; ++index) {
        Elf64_Shdr section;

        if (!sections[index].loaded || sections[index].type == SHT_NOBITS)
            continue;
        if (elf_section(&elf, index, &section) < 0) {
            status = -ENOEXEC;
            goto fail;
        }
        memcpy(image + sections[index].offset, elf.data + section.sh_offset,
               (size_t)section.sh_size);
    }
    status = resolve_symbols(runtime, &elf, sections, base, &symbol_section,
                             &string_section, symbol_count, &symbol_values,
                             &symbol_valid, &exports, &export_count);
    if (status < 0)
        goto fail;
    memset(&relocations, 0, sizeof(relocations));
    relocations.elf = &elf;
    relocations.sections = sections;
    relocations.section_count = elf.header.e_shnum;
    relocations.symbol_section = symbol_section;
    relocations.string_section = string_section;
    relocations.symbol_values = symbol_values;
    relocations.symbol_valid = symbol_valid;
    relocations.symbol_count = symbol_count;
    relocations.image = image;
    relocations.image_size = image_size;
    relocations.base = base;
    status = apply_relocations(&relocations, symbol_section_index);
    if (status < 0)
        goto fail;
    status = npu_device_write(&runtime->device, base, image,
                              (size_t)image_size);
    if (status < 0)
        goto fail;
    module = calloc(1, sizeof(*module));
    if (!module) {
        status = -ENOMEM;
        goto fail;
    }
    module->object_identity = object;
    module->object_size = object_size;
    module->base = base;
    module->size = image_size;
    module->exports = exports;
    module->export_count = export_count;
    exports = 0;
    export_count = 0;
    *module_out = module;
    status = 0;
fail:
    if (status < 0 && allocated) {
        pthread_mutex_lock(&runtime->mutex);
        (void)npu_arena_free(&runtime->normal, base, NPU_ALLOC_KERNEL,
                             UINT32_MAX);
        pthread_mutex_unlock(&runtime->mutex);
    }
    free(image);
    free(symbol_values);
    free(symbol_valid);
    free(sections);
    free_kernel_exports(exports, export_count);
    if (status < 0)
        free(module);
    return status;
}

static int module_entry(const struct npu_kernel_module *module,
                        const char *name, uint64_t *entry_out)
{
    size_t index;

    for (index = 0; index < module->export_count; ++index) {
        if (strcmp(module->exports[index].name, name) == 0) {
            *entry_out = module->exports[index].value;
            return 0;
        }
    }
    return -ENOENT;
}

int npu_kernel_resolve(npu_runtime_t *runtime,
                       const void *object, size_t object_size,
                       const char *entry_symbol, uint64_t *entry_out)
{
    struct npu_kernel_module *module;
    int status;

    if (!runtime || !object || !object_size || !entry_symbol ||
        !*entry_symbol || !entry_out)
        return -EINVAL;
    pthread_mutex_lock(&runtime->kernel_mutex);
    for (module = runtime->kernel_modules; module; module = module->next) {
        if (module->object_identity == object &&
            module->object_size == object_size)
            break;
    }
    if (!module) {
        if (!runtime->api_symbol_count) {
            status = -ENXIO;
            goto out;
        }
        status = load_kernel_module(runtime, object, object_size, &module);
        if (status < 0)
            goto out;
        module->next = runtime->kernel_modules;
        runtime->kernel_modules = module;
    }
    status = module_entry(module, entry_symbol, entry_out);
out:
    pthread_mutex_unlock(&runtime->kernel_mutex);
    return status;
}

void npu_kernel_state_destroy(npu_runtime_t *runtime)
{
    struct npu_kernel_module *module;

    if (!runtime)
        return;
    module = runtime->kernel_modules;
    while (module) {
        struct npu_kernel_module *next = module->next;

        free_kernel_exports(module->exports, module->export_count);
        free(module);
        module = next;
    }
    runtime->kernel_modules = 0;
    free_api_symbols(runtime->api_symbols, runtime->api_symbol_count);
    runtime->api_symbols = 0;
    runtime->api_symbol_count = 0;
}

static int object_has_entry(const void *object, size_t object_size,
                            const char *entry_symbol)
{
    struct elf_view elf;
    Elf64_Shdr symbol_section;
    Elf64_Shdr string_section;
    size_t symbol_count;
    size_t index;
    int status;

    status = elf_view_init(&elf, object, object_size, ET_REL);
    if (status < 0)
        return status;
    status = find_symbol_table(&elf, &symbol_section, &string_section,
                               &symbol_count);
    if (status < 0)
        return status;
    for (index = 1; index < symbol_count; ++index) {
        Elf64_Sym symbol;
        const char *name;

        if (elf_symbol(&elf, &symbol_section, index, &symbol) < 0)
            return -ENOEXEC;
        name = elf_string(&elf, &string_section, symbol.st_name);
        if (!name)
            return -ENOEXEC;
        if (symbol.st_shndx != SHN_UNDEF &&
            ELF64_ST_TYPE(symbol.st_info) == STT_FUNC &&
            strcmp(name, entry_symbol) == 0)
            return 0;
    }
    return -ENOENT;
}

int dayan_kernel_descriptor_create(const void *object, size_t object_size,
                                   const char *entry_symbol,
                                   dayan_kernel_descriptor_t **descriptor_out)
{
    dayan_kernel_descriptor_t *descriptor;
    int status;

    if (!object || !object_size || !entry_symbol || !*entry_symbol ||
        !descriptor_out)
        return -EINVAL;
    *descriptor_out = 0;
    status = object_has_entry(object, object_size, entry_symbol);
    if (status < 0)
        return status;
    descriptor = calloc(1, sizeof(*descriptor));
    if (!descriptor)
        return -ENOMEM;
    descriptor->entry_symbol = strdup(entry_symbol);
    if (!descriptor->entry_symbol) {
        free(descriptor);
        return -ENOMEM;
    }
    descriptor->object = object;
    descriptor->object_size = object_size;
    *descriptor_out = descriptor;
    return 0;
}

void dayan_kernel_descriptor_destroy(dayan_kernel_descriptor_t *descriptor)
{
    if (!descriptor)
        return;
    free(descriptor->entry_symbol);
    free(descriptor);
}

void dayan_kernel_launch_attrs_init(dayan_kernel_launch_attrs_t *attrs,
                                    npu_runtime_t *runtime)
{
    if (!attrs)
        return;
    memset(attrs, 0, sizeof(*attrs));
    attrs->version = DAYAN_KERNEL_LAUNCH_ATTRS_VERSION;
    attrs->runtime = runtime;
    attrs->launch_count = 1;
    attrs->timeout_ms = NPU_TIMEOUT_INFINITE;
}

static int launch_attributes_valid(const dayan_kernel_launch_attrs_t *attrs)
{
    uint32_t count;

    if (!attrs || attrs->version != DAYAN_KERNEL_LAUNCH_ATTRS_VERSION ||
        !attrs->runtime || attrs->reserved ||
        (attrs->flags & ~DAYAN_KERNEL_LAUNCH_ASYNC))
        return 0;
    count = attrs->launch_count ? attrs->launch_count : 1;
    if (attrs->first_hart >= attrs->runtime->options.hart_count ||
        count > attrs->runtime->options.hart_count - attrs->first_hart)
        return 0;
    if (attrs->flags & DAYAN_KERNEL_LAUNCH_ASYNC) {
        if (!attrs->async_launches || !attrs->async_launch_count ||
            attrs->async_launch_capacity < count)
            return 0;
    }
    return 1;
}

static int launch_one(dayan_kernel_descriptor_t *descriptor,
                      uint64_t entry, const uint64_t *args, size_t argc,
                      const dayan_kernel_launch_attrs_t *attrs,
                      uint32_t index, npu_launch_t **launch_out)
{
    npu_launch_desc_t launch_desc;

    (void)descriptor;
    npu_launch_desc_init(&launch_desc);
    launch_desc.hart = attrs->first_hart + index;
    launch_desc.entry = entry;
    launch_desc.argc = (uint32_t)argc;
    if (argc)
        memcpy(launch_desc.args, args, argc * sizeof(*args));
    if (attrs->stack_size)
        launch_desc.stack_size = attrs->stack_size;
    launch_desc.shared_size = attrs->shared_size;
    if (attrs->printf_buffer_size)
        launch_desc.printf_buffer_size = attrs->printf_buffer_size;
    launch_desc.cookie = attrs->cookie + index;
    return npu_runtime_launch(attrs->runtime, &launch_desc, launch_out);
}

int dayan_kernel_launch(dayan_kernel_descriptor_t *descriptor,
                        const uint64_t *args, size_t argc,
                        const dayan_kernel_launch_attrs_t *attrs)
{
    uint64_t entry;
    uint32_t count;
    uint32_t index;
    int status;

    if (!descriptor || argc > DAYAN_KERNEL_MAX_ARGS ||
        (argc && !args) || !launch_attributes_valid(attrs))
        return -EINVAL;
    count = attrs->launch_count ? attrs->launch_count : 1;
    if (attrs->flags & DAYAN_KERNEL_LAUNCH_ASYNC)
        *attrs->async_launch_count = 0;
    status = npu_kernel_resolve(attrs->runtime, descriptor->object,
                                descriptor->object_size,
                                descriptor->entry_symbol, &entry);
    if (status < 0)
        return status;
    for (index = 0; index < count; ++index) {
        npu_launch_t *launch = 0;

        status = launch_one(descriptor, entry, args, argc, attrs, index,
                            &launch);
        if (status < 0) {
            if ((attrs->flags & DAYAN_KERNEL_LAUNCH_ASYNC) && launch) {
                attrs->async_launches[*attrs->async_launch_count] = launch;
                ++*attrs->async_launch_count;
            } else if (launch) {
                npu_launch_info_t ignored;

                (void)npu_launch_wait(launch, NPU_TIMEOUT_INFINITE, &ignored);
                (void)npu_launch_release(launch);
            }
            return status;
        }
        if (attrs->flags & DAYAN_KERNEL_LAUNCH_ASYNC) {
            attrs->async_launches[*attrs->async_launch_count] = launch;
            ++*attrs->async_launch_count;
        } else {
            npu_launch_info_t info;
            int wait_status;
            int release_status;

            wait_status = npu_launch_wait(launch, attrs->timeout_ms, &info);
            if (wait_status < 0) {
                /* The low-level API has no cancellation primitive.  Finish
                 * ownership cleanup before reporting the original timeout. */
                (void)npu_launch_wait(launch, NPU_TIMEOUT_INFINITE, &info);
            }
            release_status = npu_launch_release(launch);
            if (wait_status < 0)
                return wait_status;
            if (release_status < 0)
                return release_status;
            if (info.status < 0 && info.status >= INT_MIN)
                return (int)info.status;
        }
    }
    return 0;
}
