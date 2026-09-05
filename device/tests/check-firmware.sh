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

switch_dump=$($objdump --disassemble --no-show-raw-insn \
    --disassemble-symbols=switch_context "$elf")
switch_raw=$($objdump --disassemble \
    --disassemble-symbols=switch_context "$elf")
scheduler_dump=$($objdump --disassemble --no-show-raw-insn \
    --disassemble-symbols=scheduler "$elf")
[ "$(grep -Ec '^[[:space:]]*[[:xdigit:]]+:' <<<"$scheduler_dump")" -eq 3 ] &&
    grep -Eq 'jalr.*<switch_context>' <<<"$scheduler_dump" ||
    fail "scheduler must contain only call switch_context and its loop branch"
if grep -Eq '(^|[[:space:],(])s(0|1|2|3|4|5|6|7|8|9|10|11)([[:space:],)]|$)' \
        <<<"$scheduler_dump"; then
    fail "scheduler must not use callee-saved registers"
fi
for encoding in 0000128b 0006d38b 0002028b 0002228b; do
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
switch_selection=$(sed -n "1,$((spill_line - 1))p" <<<"$switch_dump")
if grep -Eq '(^|[[:space:],(])s(0|1|2|3|4|5|6|7|8|9|10|11)([[:space:],)]|$)' \
        <<<"$switch_selection"; then
    fail "switch_context selection path must use only caller-saved registers"
fi
for load in \
    'ld[[:space:]]+t2, 0x8\(t1\)' \
    'ld[[:space:]]+t3, 0x18\(t1\)' \
    'lwu[[:space:]]+t4, 0x20\(t1\)' \
    'ld[[:space:]]+t5, 0x10\(t1\)'; do
    grep -Eq "$load" <<<"$switch_dump" ||
        fail "switch_context must interleave target context loads with spills"
done
kill_ra_save_line=$(grep -n -m1 'sd[[:space:]]\+ra, 0x0(sp)' \
    <<<"$switch_dump" | cut -d: -f1)
kill_call_line=$(grep -n -m1 'jalr.*<scheduler_kill_token>' \
    <<<"$switch_dump" | cut -d: -f1)
kill_ra_restore_line=$(grep -n -m1 'ld[[:space:]]\+ra, 0x0(sp)' \
    <<<"$switch_dump" | cut -d: -f1)
[ -n "$kill_ra_save_line" ] && [ -n "$kill_call_line" ] &&
    [ -n "$kill_ra_restore_line" ] &&
    [ "$kill_ra_save_line" -lt "$kill_call_line" ] &&
    [ "$kill_call_line" -lt "$kill_ra_restore_line" ] ||
    fail "kill-token handling must preserve the interrupted thread return address"

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
for encoding in 0000350b 0000550b 0000750b 0000950b; do
    grep -Fq "$encoding" <<<"$tensor_interrupt_raw" ||
        fail "tensor interrupt Barrier $encoding is missing"
done

rpc_receive_pending_dump=$($objdump --disassemble --no-show-raw-insn \
    --disassemble-symbols=platform_rpc_receive_pending \
    "$build/platform/platform.o")
grep -Eq 'csrr[[:space:]]+[^,]+,[[:space:]]*mip' \
    <<<"$rpc_receive_pending_dump" ||
    fail "RPC receive polling must read mip.MSIP through the CSR"
if grep -Eq '(^|[[:space:]])(lb|lbu|lh|lhu|lw|lwu|ld)([[:space:]]|$)' \
        <<<"$rpc_receive_pending_dump"; then
    fail "RPC receive polling must not read CLINT through MMIO"
fi

rpc_halt_dump=$($objdump --disassemble --no-show-raw-insn \
    --disassemble-symbols=platform_rpc_halt \
    "$build/platform/platform.o")
rpc_halt_raw=$($objdump --disassemble \
    --disassemble-symbols=platform_rpc_halt \
    "$build/platform/platform.o")
grep -Fq '0000150b' <<<"$rpc_halt_raw" ||
    fail "RPC halt must drain stale hardware scheduler tokens"
grep -Fq '0006a50b' <<<"$rpc_halt_raw" ||
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
for encoding in 0004250b 0004450b 0004650b 0004850b; do
    grep -Fq "$encoding" <<<"$dapi_raw" ||
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
