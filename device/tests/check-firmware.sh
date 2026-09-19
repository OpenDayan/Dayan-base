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

has_tcsr()
{
    local expected=$1 word

    # The builtin selects the GPR in bits 11:7; all other fields must match.
    while read -r word; do
        if (( (16#$word & 0xfffff07f) == expected )); then
            return 0
        fi
    done < <(awk 'length($2) == 8 && $2 ~ /^[[:xdigit:]]+$/ {print $2}')
    return 1
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
    fail "icache fence must occur only in the explicit fencei RPC"
if $nm "$elf" | grep -Fq '__global_pointer$'; then
    fail 'firmware must not define __global_pointer$'
fi

context_dump=$($objdump --disassemble --no-show-raw-insn \
    "$build/sche/context.o")
[ "$(grep -Ec 'sd[[:space:]]+gp,' <<<"$context_dump")" -eq 1 ] &&
    [ "$(grep -Ec 'ld[[:space:]]+gp,' <<<"$context_dump")" -eq 2 ] ||
    fail "context switch must save and restore per-proc gp"

switch_dump=$($objdump --disassemble --no-show-raw-insn \
    --disassemble-symbols=switch_context "$elf")
switch_raw=$($objdump --disassemble \
    --disassemble-symbols=switch_context "$elf")
if $nm --defined-only --format=posix "$elf" | \
        grep -Eq '^(scheduler|thread_wait_poll) '; then
    fail "firmware must not have a scheduler thread or polling wait API"
fi
grep -Eq 'beqz[[:space:]]+tp,' <<<"$switch_dump" ||
    fail "first dispatch must skip saving a boot thread"
for encoding in 0000130b 0006d38b 00020e0b 00022e8b; do
    grep -Fq "$encoding" <<<"$switch_raw" ||
        fail "switch_context hardware instruction $encoding is missing"
done
mip_line=$(grep -n -m1 'csrr[[:space:]]\+t0, mip' <<<"$switch_dump" |
    cut -d: -f1)
pop_line=$(grep -n -m1 'tx\.sb\.get\.any' <<<"$switch_dump" |
    cut -d: -f1)
same_line=$(grep -n -m1 'beq[[:space:]]\+t1, tp' <<<"$switch_dump" |
    cut -d: -f1)
spill_line=$(grep -n -m1 'sd[[:space:]]\+s0,' <<<"$switch_dump" |
    cut -d: -f1)
[ -n "$mip_line" ] && [ -n "$pop_line" ] && [ -n "$same_line" ] &&
    [ -n "$spill_line" ] && [ "$mip_line" -lt "$pop_line" ] &&
    [ "$pop_line" -lt "$same_line" ] && [ "$same_line" -lt "$spill_line" ] ||
    fail "switch_context must select mip/token/same-tp before spilling s0"
grep -Eq 'li[[:space:]]+t6, 0x80' <<<"$switch_dump" &&
    grep -Eq 'addi[[:space:]]+t6, t6, -0x1' <<<"$switch_dump" &&
    grep -Eq 'beqz[[:space:]]+t6,' <<<"$switch_dump" ||
    fail "empty scheduler pops must batch 128 attempts between MSIP samples"
switch_selection=$(sed -n "1,$((spill_line - 1))p" <<<"$switch_dump")
if grep -Eq '(^|[[:space:],(])s(0|1|2|3|4|5|6|7|8|9|10|11)([[:space:],)]|$)' \
        <<<"$switch_selection"; then
    fail "switch_context selection path must use only caller-saved registers"
fi
for load in \
    'ld[[:space:]]+t2, 0x8\(t1\)' \
    'ld[[:space:]]+t3, 0x18\(t1\)' \
    'lwu[[:space:]]+t4, 0x20\(t1\)' \
    'ld[[:space:]]+a3, 0x10\(t1\)'; do
    grep -Eq "$load" <<<"$switch_dump" ||
        fail "switch_context must interleave target context loads with spills"
done
profile_dump=$(sed -n '/ld[[:space:]]\+t0, 0x0(a1)/,$p' <<<"$switch_dump")
profile_stack_line=$(grep -n -m1 'addi[[:space:]]\+sp, sp,' \
    <<<"$profile_dump" | cut -d: -f1)
[ -n "$profile_stack_line" ] || fail "profile full path must preserve its caller"
profile_fast=$(sed -n "1,$((profile_stack_line - 1))p" <<<"$profile_dump")
if grep -Eq '(^|[[:space:],(])(sp|s[0-9]+)([[:space:],)]|$)|[[:space:]]jalr[[:space:]]' \
        <<<"$profile_fast"; then
    fail "profile recording must not use a stack, saved registers or C calls"
fi
if grep -Eq '[[:space:]]mul[[:space:]]' <<<"$context_dump"; then
    fail "profile CPU-local addressing must use a constant shift"
fi
if grep -Eq '(jal|jalr).*<profile_record_thread_state' <<<"$switch_dump"; then
    fail "context switch must branch to profile recording without a function call"
fi
grep -Eq 'bnez[[:space:]]+a1,' <<<"$switch_dump" &&
    ! grep -Eq 'beqz[[:space:]]+a1,' <<<"$switch_dump" ||
    fail "disabled profiling must fall through without taking a branch"
if grep -Eq 'lbu[[:space:]]+t0, 0x1\(t1\)|<scheduler_kill_token>' \
        <<<"$switch_dump"; then
    fail "kill handling must use the redirected PC, not a dispatch state check"
fi
if grep -Eq 'lbu[[:space:]]+[^,]+, 0x1\(tp\)|<scheduler_reap_thread>' \
        <<<"$switch_dump"; then
    fail "context switch must not inspect the outgoing state or reclaim slots"
fi
grep -Eq 'lhu[[:space:]]+t4, 0x0\(t1\)' <<<"$profile_fast" ||
    fail "profile slot/state must use one aligned halfword load"

if ! $objdump --disassemble --no-show-raw-insn \
        "$api_compile" "$api_cxx_compile" | \
        grep -Eq '(^|[[:space:],(])gp([[:space:],)]|$)'; then
    fail "operator shared pointer must read gp"
fi

bl1_dump=$($objdump --disassemble --no-show-raw-insn "$build/bl1/bl1.o")
grep -Eq 'csrw[[:space:]]+mie, t0' <<<"$bl1_dump" ||
    fail "BL1 must replace mie with the tensor MEIE mask"
mie_writes=$($objdump --disassemble --no-show-raw-insn "$elf" | \
    grep -Ec '[[:space:]](csrw|csrs|csrc|csrrw|csrrs|csrrc)[[:space:]].*mie')
[ "$mie_writes" -eq 1 ] ||
    fail "firmware must never reconfigure mie after BL1"
grep -Eq 'csrs[[:space:]]+mstatus,.*' <<<"$bl1_dump" &&
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

tensor_interrupt_raw=$($objdump --disassemble \
    --disassemble-symbols=tensor_interrupt_handle \
    "$build/platform/tensor_interrupt.o")
for encoding in 0x0000300b 0x0000500b 0x0000700b 0x0000900b; do
    if has_tcsr "$encoding" <<<"$tensor_interrupt_raw"; then
        fail "tensor interrupt handler must not wait for unit Barrier $encoding"
    fi
done

rpc_halt_dump=$($objdump --disassemble --no-show-raw-insn \
    --disassemble-symbols=platform_rpc_halt \
    "$build/platform/platform.o")
rpc_halt_raw=$($objdump --disassemble \
    --disassemble-symbols=platform_rpc_halt \
    "$build/platform/platform.o")
has_tcsr 0x0000100b <<<"$rpc_halt_raw" ||
    fail "RPC halt must drain stale hardware scheduler tokens"
has_tcsr 0x0006a00b <<<"$rpc_halt_raw" ||
    fail "RPC halt must initialize Local Resource state"
halt_pop_line=$(grep -n -m1 'tx\.sb\.get\.any' <<<"$rpc_halt_dump" |
    cut -d: -f1)
halt_init_line=$(grep -n -m1 '0x35' <<<"$rpc_halt_dump" | cut -d: -f1)
[ -n "$halt_pop_line" ] && [ -n "$halt_init_line" ] &&
    [ "$halt_pop_line" -lt "$halt_init_line" ] ||
    fail "RPC halt must drain scheduler tokens before Local Resource init"

stack_sections=$($readelf --sections --wide "$elf" | \
    grep -Ec '\.(stack|rpc_stack|profile_stack|trap_stack|proc_stack)[[:space:]]+NOBITS')
[ "$stack_sections" -eq 5 ] ||
    fail "per-hart stacks must be NOBITS sections"
for section in cpu proc_printf; do
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
cpu_start=$(symbol_address __cpu_start)
cpu_end=$(symbol_address __cpu_end)
cpu_size=$(symbol_address __cpu_local_size)
[ $((cpu_end - cpu_start)) -eq $((harts * cpu_size)) ] &&
    [ $((cpu_start % 64)) -eq 0 ] &&
    [ $((cpu_size % 64)) -eq 0 ] ||
    fail "CPU Local must have isolated per-Hart cache lines"
local_segment=$($readelf --segments --wide "$elf" | \
    awk '$1 == "LOAD" && $7 ~ /W/ {print; exit}')
[ -n "$local_segment" ] ||
    fail "CPU Local and backing pools must be in a writable ELF segment"
local_segment_start=$(awk '{print $3}' <<<"$local_segment")
local_segment_size=$(awk '{print $6}' <<<"$local_segment")
local_start=$((16#${local_segment_start#0x}))
local_end=$((local_start + 16#${local_segment_size#0x}))
cpu_start_value=$((16#${cpu_start#0x}))
[ "$cpu_start_value" -ge "$local_start" ] &&
    [ "$cpu_start_value" -lt "$local_end" ] ||
    fail "CPU Local and backing pools must be in a writable ELF segment"
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
for encoding in 0x0000200b 0x0000400b 0x0000600b 0x0000800b; do
    has_tcsr "$encoding" <<<"$dapi_raw" ||
        fail "Produce instruction $encoding is missing"
done
for encoding in 0x0004200b 0x0004400b 0x0004600b 0x0004800b; do
    has_tcsr "$encoding" <<<"$dapi_raw" ||
        fail "Resource Produce instruction $encoding is missing"
done
if grep -Eq '(^|[[:space:]])fence([[:space:]]|$)' <<<"$dapi_dump" ||
        grep -Eq '(^|[[:space:]])amo[^[:space:]]*\.(aq|rl|aqrl)([[:space:]]|$)' \
            <<<"$dapi_dump"; then
    fail "Dayan synchronization API must use relaxed coherent accesses"
fi

local_resource_apis=dayan_resource_wait,dayan_resource_wait_ready,dayan_resource_wait_free,dayan_resource_produce_lsu,dayan_resource_produce_mat,dayan_resource_produce_vec,dayan_resource_produce_macro,dayan_resource_consume_lsu,dayan_resource_consume_mat,dayan_resource_consume_vec,dayan_resource_consume_macro
if $objdump --no-show-raw-insn \
        --disassemble-symbols="$local_resource_apis" \
        "$build/sche/dapi.o" | \
        grep -Eq '(^|[[:space:]])(amo[^[:space:]]*|lr\.[wd][^[:space:]]*|sc\.[wd][^[:space:]]*|fence)([[:space:]]|$)'; then
    fail "Named Resource API must not use atomic instructions"
fi
barrier_dump=$($objdump --no-show-raw-insn \
    --disassemble-symbols=dayan_cluster_barrier "$build/sche/dapi.o")
grep -Eq 'amoadd\.d([[:space:]]|$)' <<<"$barrier_dump" ||
    fail "cluster barrier must use a native 64-bit AMOADD ticket"
if grep -Eq 'thread_wait|thread_yield|switch_context' <<<"$barrier_dump"; then
    fail "cluster barrier must busy-wait without scheduling"
fi
if grep -Eq 'amoswap\.[wd]([[:space:]]|$)' <<<"$barrier_dump"; then
    fail "cluster barrier must not use an atomic spin lock"
fi
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
