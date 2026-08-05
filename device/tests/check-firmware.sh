#!/usr/bin/env bash
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
device_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
repo_root=$(CDPATH= cd -- "$device_root/.." && pwd)
build=${BUILD_DIR:-"$repo_root/build/device"}
elf="$build/npu-os.elf"
api_elf="$build/npu-os-api.elf"
api_compile="$build/tests/api_compile.o"
api_cxx_compile="$build/tests/api_compile.cxx.o"
api_symbols="$device_root/dayan-api.symbols"

nm=${NM:-llvm-nm}
objdump=${OBJDUMP:-llvm-objdump}
readelf=${READELF:-llvm-readelf}
harts=${HARTS:-4}
stack_size=${STACK_SIZE:-0x10000}
rpc_stack_size=${RPC_STACK_SIZE:-0x1000}
profile_stack_size=${PROFILE_STACK_SIZE:-0x1000}
trap_stack_size=${TRAP_STACK_SIZE:-0x1000}
proc_slot_count=${PROC_SLOT_COUNT:-16}
proc_stack_size=${PROC_STACK_SIZE:-0x10000}
proc_printf_size=${PROC_PRINTF_SIZE:-0x1000}

fail()
{
    echo "firmware-check: error: $*" >&2
    exit 1
}

for file in "$elf" "$api_elf" "$api_compile" "$api_cxx_compile"; do
    [ -f "$file" ] || fail "missing build artifact: $file"
done

objects=()
for directory in boot lib mem platform rpc sche bl1; do
    while IFS= read -r -d '' object; do
        objects+=("$object")
    done < <(find "$build/$directory" -type f -name '*.o' -print0)
done
[ "${#objects[@]}" -gt 0 ] || fail "no firmware objects found"
relocation_objects=("${objects[@]}" "$api_compile" "$api_cxx_compile")

undefined=$($nm --undefined-only "$elf")
[ -z "$undefined" ] || fail "firmware contains undefined symbols: $undefined"
firmware_dump=$($objdump --disassemble --no-show-raw-insn "$elf")
if grep -Eq '(^|[[:space:]])(lr\.[wd]|sc\.[wd])([.[:space:]]|$)' \
        <<<"$firmware_dump"; then
    fail "firmware must not use unsupported LR/SC instructions"
fi
if $readelf --relocations "${relocation_objects[@]}" | \
        grep -q 'R_RISCV_GPREL'; then
    fail "target objects must not contain gp-relative relocations"
fi
if $readelf --relocations "${relocation_objects[@]}" | \
        grep -Eq 'R_RISCV_(TPREL|TLS_)'; then
    fail "target objects must not use tp-relative TLS"
fi
icache_fence_calls=$($readelf --relocations --wide "$build/rpc/proc.o" | \
    grep -c 'platform_instruction_fence' || true)
[ "$icache_fence_calls" -eq 1 ] ||
    fail "icache fence must exist only in the explicit flush RPC"
if $nm "$elf" | grep -Fq '__global_pointer$'; then
    fail 'firmware must not define __global_pointer$'
fi

context_dump=$($objdump --disassemble --no-show-raw-insn \
    "$build/sche/context.o")
[ "$(grep -Ec 'sd[[:space:]]+gp,' <<<"$context_dump")" -eq 1 ] &&
    [ "$(grep -Ec 'ld[[:space:]]+gp,' <<<"$context_dump")" -eq 2 ] ||
    fail "context switch must save and restore per-proc gp"

if ! $objdump --disassemble --no-show-raw-insn \
        "$api_compile" "$api_cxx_compile" | \
        grep -Eq '(^|[[:space:],(])gp([[:space:],)]|$)'; then
    fail "operator shared pointer must read gp"
fi

bl1_dump=$($objdump --disassemble --no-show-raw-insn "$build/bl1/bl1.o")
grep -Eq 'csrw[[:space:]]+mie, zero' <<<"$bl1_dump" ||
    fail "BL1 must mask functional interrupts"
mie_writes=$($objdump --disassemble --no-show-raw-insn "$elf" | \
    grep -Ec '[[:space:]](csrw|csrs|csrc|csrrw|csrrs|csrrc)[[:space:]].*mie')
[ "$mie_writes" -eq 1 ] ||
    fail "firmware must never reconfigure mie after BL1"
grep -Eq 'csrc[[:space:]]+mstatus,.*' <<<"$bl1_dump" &&
    grep -Eq 'csrw[[:space:]]+mscratch,.*' <<<"$bl1_dump" &&
    grep -Eq 'csrw[[:space:]]+mtvec,.*' <<<"$bl1_dump" ||
    fail "BL1 trap and interrupt state is incomplete"

trap_dump=$($objdump --disassemble --no-show-raw-insn \
    "$build/platform/trap_entry.o")
grep -Eq 'csrrw[[:space:]]+sp, mscratch, sp' <<<"$trap_dump" ||
    fail "trap entry must switch to the per-hart trap stack"
for csr in mstatus mepc mcause mtval; do
    grep -Eq "csrr[[:space:]]+t0, $csr" <<<"$trap_dump" ||
        fail "trap entry does not save $csr"
done
grep -Eq 'frcsr[[:space:]]+t0' <<<"$trap_dump" ||
    fail "trap entry does not save fcsr"

stack_sections=$($readelf --sections --wide "$elf" | \
    grep -Ec '\.(stack|rpc_stack|profile_stack|trap_stack|proc_stack)[[:space:]]+NOBITS')
[ "$stack_sections" -eq 5 ] ||
    fail "per-hart stacks must be NOBITS sections"
for section in proc_slots proc_printf; do
    $readelf --sections --wide "$elf" | \
        grep -Eq "\\.${section}[[:space:]]+NOBITS" ||
        fail ".${section} must be a NOBITS section"
done

symbol_address()
{
    $nm --defined-only --format=posix "$elf" | \
        awk -v symbol="$1" '$1 == symbol {print "0x" $3; exit}'
}

stack_start=$(symbol_address __stack_start)
stack_end=$(symbol_address __stack_end)
rpc_start=$(symbol_address __rpc_stack_start)
rpc_end=$(symbol_address __rpc_stack_end)
profile_start=$(symbol_address __profile_stack_start)
profile_end=$(symbol_address __profile_stack_end)
trap_start=$(symbol_address __trap_stack_start)
trap_end=$(symbol_address __trap_stack_end)
proc_stack_start=$(symbol_address __proc_stack_start)
proc_stack_end=$(symbol_address __proc_stack_end)
proc_printf_start=$(symbol_address __proc_printf_start)
proc_printf_end=$(symbol_address __proc_printf_end)
[ $((stack_end - stack_start)) -eq $((harts * stack_size)) ] &&
    [ $((rpc_end - rpc_start)) -eq $((harts * 2 * rpc_stack_size)) ] &&
    [ $((profile_end - profile_start)) -eq \
        $((harts * profile_stack_size)) ] &&
    [ $((trap_end - trap_start)) -eq $((harts * trap_stack_size)) ] &&
    [ $((proc_stack_end - proc_stack_start)) -eq \
        $((harts * proc_slot_count * proc_stack_size)) ] &&
    [ $((proc_printf_end - proc_printf_start)) -eq \
        $((harts * proc_slot_count * proc_printf_size)) ] ||
    fail "linked per-hart stack reservation has the wrong size"

dapi_raw=$($objdump --disassemble "$build/sche/dapi.o")
dapi_dump=$($objdump --disassemble --no-show-raw-insn \
    "$build/sche/dapi.o")
for encoding in 0000220b 0000420b 0000620b 0000820b; do
    grep -Fq "$encoding" <<<"$dapi_raw" ||
        fail "Produce instruction $encoding is missing"
done
grep -Fq '0000150b' <<<"$dapi_raw" ||
    fail "global Produce pop instruction encoding changed"
if grep -Eq '(^|[[:space:]])fence([[:space:]]|$)' <<<"$dapi_dump" ||
        grep -Eq '(^|[[:space:]])amo[^[:space:]]*\.(aq|rl|aqrl)([[:space:]]|$)' \
            <<<"$dapi_dump"; then
    fail "Dayan synchronization API must use relaxed coherent accesses"
fi

local_resource_apis=dayan_resource_init,dayan_resource_alloc,dayan_resource_produce,dayan_resource_consume,dayan_resource_wait_ready,dayan_resource_wait_free
if $objdump --no-show-raw-insn \
        --disassemble-symbols="$local_resource_apis" \
        "$build/sche/dapi.o" | \
        grep -Eq '(^|[[:space:]])(amo[^[:space:]]*|lr\.[wd][^[:space:]]*|sc\.[wd][^[:space:]]*|fence)([[:space:]]|$)'; then
    fail "per-hart resource API must not use atomic instructions"
fi
shared_alloc_dump=$($objdump --no-show-raw-insn \
    --disassemble-symbols=dayan_shared_resource_alloc "$build/sche/dapi.o")
shared_produce_dump=$($objdump --no-show-raw-insn \
    --disassemble-symbols=dayan_shared_resource_produce "$build/sche/dapi.o")
shared_consume_dump=$($objdump --no-show-raw-insn \
    --disassemble-symbols=dayan_shared_resource_consume "$build/sche/dapi.o")
grep -Eq 'amocas\.d([[:space:]]|$)' <<<"$shared_alloc_dump" ||
    fail "shared resource allocation must use native 64-bit AMOCAS"
grep -Eq 'amoadd\.d([[:space:]]|$)' <<<"$shared_produce_dump" ||
    fail "shared resource produce must use native 64-bit AMOADD"
grep -Eq 'amoadd\.d([[:space:]]|$)' <<<"$shared_consume_dump" ||
    fail "shared resource consume must use native 64-bit AMOADD"
shared_resource_dump="$shared_alloc_dump$shared_produce_dump$shared_consume_dump"
if grep -Eq 'amoswap\.[wd]([[:space:]]|$)' <<<"$shared_resource_dump"; then
    fail "shared resource API must not use an atomic spin lock"
fi
barrier_dump=$($objdump --no-show-raw-insn \
    --disassemble-symbols=dayan_cluster_barrier "$build/sche/dapi.o")
grep -Eq 'amoswap\.w([[:space:]]|$)' <<<"$barrier_dump" ||
    fail "cluster barrier must use a coherent atomic lock"
if grep -Eq '(^|[[:space:]])(lr\.[wd]|sc\.[wd])' <<<"$barrier_dump"; then
    fail "cluster barrier must not use per-core LR/SC reservations"
fi

LC_ALL=C sort -u "$api_symbols" > "$build/.api-symbols.expected"
$nm --defined-only --extern-only --format=posix "$api_elf" | \
    awk '{print $1}' | LC_ALL=C sort -u > "$build/.api-symbols.actual"
diff -u "$build/.api-symbols.expected" "$build/.api-symbols.actual"
$objdump --syms "$elf" | \
    awk '$4 == ".text.api.libc" || $4 == ".text.api.dapi" {print $NF}' | \
    LC_ALL=C sort -u > "$build/.api-symbols.marked"
diff -u "$build/.api-symbols.expected" "$build/.api-symbols.marked"
$nm --undefined-only --format=posix "$api_compile" | \
    awk '{print $1}' | LC_ALL=C sort -u > "$build/.api-symbols.used"
comm -23 "$build/.api-symbols.used" "$build/.api-symbols.expected" > \
    "$build/.api-symbols.invalid"
[ ! -s "$build/.api-symbols.invalid" ] || {
    echo "firmware-check: error: operator API test references non-public symbols" >&2
    cat "$build/.api-symbols.invalid" >&2
    exit 1
}

echo "firmware-check: PASS"
