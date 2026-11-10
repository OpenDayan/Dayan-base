#!/usr/bin/env python3
#***************************************************************************************
# Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
#
# Dayan-base is licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
#
# See the Mulan PSL v2 for more details.
#***************************************************************************************

"""Capture a marker-aligned TestCluster waveform for one Dayan operator."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import os
from pathlib import Path
import pty
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time

import run_testcluster_operator as runner


class OperatorDrain:
    def __init__(
        self,
        process: subprocess.Popen,
        log_path: Path,
        marker: str,
        expected: list[str],
    ) -> None:
        self.process = process
        self.log_path = log_path
        self.marker = marker.encode()
        self.expected = [item.encode() for item in expected]
        self.found = [False] * len(self.expected)
        self.marker_seen = threading.Event()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        carry = b""
        carry_limit = max(
            [len(self.marker), *(len(item) for item in self.expected)],
            default=1,
        ) - 1

        assert self.process.stdout is not None
        with self.log_path.open("wb") as log:
            while True:
                try:
                    chunk = os.read(self.process.stdout.fileno(), 64 * 1024)
                except OSError:
                    break
                if not chunk:
                    break
                log.write(chunk)
                log.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                searchable = carry + chunk
                if self.marker in searchable:
                    self.marker_seen.set()
                for index, pattern in enumerate(self.expected):
                    if not self.found[index] and pattern in searchable:
                        self.found[index] = True
                carry = searchable[-carry_limit:] if carry_limit else b""
        self.done.set()

    def validate(self) -> None:
        self.thread.join(timeout=1.0)
        status = self.process.poll()
        if status is None:
            raise runner.RunnerError("operator is still running")
        if status != 0:
            raise runner.RunnerError(f"operator exited with status {status}")
        missing = [
            pattern.decode(errors="replace")
            for pattern, present in zip(self.expected, self.found)
            if not present
        ]
        if missing:
            raise runner.RunnerError(
                "operator output is missing: "
                + ", ".join(repr(item) for item in missing)
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an operator in small TestCluster steps, then capture only the "
            "cycles following a device marker."
        )
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--sim-dir",
        type=Path,
        default=runner.default_sim_dir(),
    )
    parser.add_argument("--sim-bin", type=Path)
    parser.add_argument("--dram-config", type=Path)
    parser.add_argument(
        "--device-prefix",
        type=Path,
        default=Path(
            os.environ.get(
                "NPU_DEVICE_PREFIX",
                f"/tmp/virtual_npu-wave-{os.getpid()}",
            )
        ),
    )
    parser.add_argument("--server", type=Path)
    parser.add_argument("--firmware", type=Path)
    parser.add_argument("--api-elf", type=Path)
    parser.add_argument("--cluster-count", type=int, default=1)
    parser.add_argument("--core-mask", type=runner.parse_int, default=1)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--operator-timeout", type=float, default=3600.0)
    parser.add_argument("--fuse-settle", type=float, default=1.0)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--step-cycles", type=runner.parse_int, default=20_000)
    parser.add_argument("--trace-cycles", type=runner.parse_int, required=True)
    parser.add_argument("--max-marker-cycles", type=runner.parse_int, default=20_000_000)
    parser.add_argument("--expect", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if arguments.command and arguments.command[0] == "--":
        arguments.command = arguments.command[1:]
    return arguments


def send_cli(
    master_fd: int,
    drain: runner.PtyDrain,
    command: str,
    timeout: float,
    processes: tuple[tuple[str, subprocess.Popen], ...],
) -> None:
    drain.clear_scan()
    os.write(master_fd, (command + "\n").encode())
    runner.wait_for(
        lambda: drain.contains("➜ " + command),
        timeout,
        f"TestCluster to accept {command!r}",
        processes,
    )
    drain.clear_scan()
    runner.wait_for(
        lambda: drain.contains("等待命令"),
        timeout,
        f"TestCluster command {command!r} to complete",
        processes,
    )


def start_continuous(
    master_fd: int,
    drain: runner.PtyDrain,
    timeout: float,
    processes: tuple[tuple[str, subprocess.Popen], ...],
) -> None:
    drain.clear_scan()
    os.write(master_fd, b"c\n")
    runner.wait_for(
        lambda: drain.contains("➜ c"),
        timeout,
        "TestCluster to accept continuous simulation",
        processes,
    )
    drain.clear_scan()
    runner.wait_for(
        lambda: drain.contains("命令执行中"),
        timeout,
        "TestCluster to enter continuous simulation",
        processes,
    )


def stop_continuous(
    master_fd: int,
    drain: runner.PtyDrain,
    timeout: float,
    processes: tuple[tuple[str, subprocess.Popen], ...],
) -> None:
    drain.clear_scan()
    os.write(master_fd, b"\x03")
    runner.wait_for(
        lambda: drain.contains("等待命令"),
        timeout,
        "TestCluster continuous simulation to stop",
        processes,
    )


def main() -> int:
    arguments = parse_arguments()
    if not arguments.command:
        raise runner.RunnerError("provide an operator command after --")
    if arguments.cluster_count < 1 or arguments.cluster_count > 4:
        raise runner.RunnerError("--cluster-count must be between 1 and 4")
    if arguments.core_mask <= 0 or arguments.core_mask > 0xF:
        raise runner.RunnerError("--core-mask must be between 1 and 0xf")
    if (
        arguments.startup_timeout <= 0
        or arguments.operator_timeout <= 0
        or arguments.step_cycles <= 0
        or arguments.trace_cycles <= 0
        or arguments.max_marker_cycles < arguments.step_cycles
    ):
        raise runner.RunnerError("timeouts and cycle counts must be positive")

    if arguments.sim_dir is None:
        raise runner.RunnerError(
            "no TestCluster directory: pass --sim-dir or set NPU_TC_ROOT"
        )
    repo = arguments.repo.resolve()
    sim_dir = arguments.sim_dir.resolve()
    sim_bin = runner.rooted(arguments.sim_bin, sim_dir, "TestCluster")
    dram_config = runner.rooted(
        arguments.dram_config,
        sim_dir,
        f"configs/{runner.DEFAULT_DRAM_CONFIG}",
    )
    server = runner.rooted(
        arguments.server, repo, "build/host/driver/dayan_server"
    )
    firmware = runner.rooted(arguments.firmware, repo, "build/device/npu-os.elf")
    api_elf = runner.rooted(arguments.api_elf, repo, "build/device/npu-os-api.elf")
    device_prefix = arguments.device_prefix.resolve()
    socket_path = (
        arguments.socket.resolve()
        if arguments.socket
        else Path(tempfile.gettempdir()) / f"dayan-wave-{os.getpid()}.sock"
    )
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = (
        arguments.log_dir.resolve()
        if arguments.log_dir
        else repo
        / "build"
        / "testcluster-runs"
        / f"wave-{timestamp}-{os.getpid()}"
    )
    sim_work_dir = log_dir / "sim"

    runner.require_file(sim_bin, "TestCluster", executable=True)
    runner.require_file(dram_config, "DRAM configuration")
    runner.require_file(server, "dayan_server", executable=True)
    runner.require_file(firmware, "firmware ELF")
    runner.require_file(api_elf, "API ELF")
    if runner.is_mountpoint(device_prefix):
        raise runner.RunnerError(
            f"device prefix is already mounted: {device_prefix}"
        )
    if runner.disconnected_mount(device_prefix):
        runner.unmount(device_prefix, force=True)
    if socket_path.exists():
        raise runner.RunnerError(f"socket path already exists: {socket_path}")

    device_prefix.mkdir(parents=True, exist_ok=True)
    sim_work_dir.mkdir(parents=True, exist_ok=False)
    print(f"[wave-capture] logs: {log_dir}", flush=True)

    simulator: subprocess.Popen | None = None
    server_process: subprocess.Popen | None = None
    operator_process: subprocess.Popen | None = None
    operator_drain: OperatorDrain | None = None
    server_log = None
    master_fd = -1
    drain: runner.PtyDrain | None = None
    success = False

    try:
        master_fd, slave_fd = pty.openpty()
        fcntl.ioctl(
            slave_fd,
            termios.TIOCSWINSZ,
            struct.pack("HHHH", 24, 120, 0, 0),
        )

        def simulator_child_setup() -> None:
            os.setsid()
            fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)

        environment = os.environ.copy()
        environment["SIM_DRAM_CONFIG"] = str(dram_config)
        environment["SIM_MOUNT_POINT"] = str(device_prefix)
        simulator = subprocess.Popen(
            [str(sim_bin)],
            cwd=sim_work_dir,
            env=environment,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            preexec_fn=simulator_child_setup,
            close_fds=True,
        )
        os.close(slave_fd)
        drain = runner.PtyDrain(master_fd)
        drain.start()

        endpoints = tuple(
            device_prefix / name
            for name in ("xdma0_h2c_0", "xdma0_c2h_0", "xdma0_events_0")
        )
        runner.wait_for(
            lambda: drain.cli_ready.is_set()
            and drain.input_ready.is_set()
            and all(path.exists() for path in endpoints),
            arguments.startup_timeout,
            "TestCluster CLI and FUSE endpoints",
            (("TestCluster", simulator),),
        )

        server_log = (log_dir / "dayan_server.log").open("wb")
        server_process = subprocess.Popen(
            [
                str(server),
                "--socket",
                str(socket_path),
                "--firmware",
                str(firmware),
                "--api-elf",
                str(api_elf),
                "--device-prefix",
                str(device_prefix),
                "--cluster-count",
                str(arguments.cluster_count),
                "--core-mask",
                hex(arguments.core_mask),
            ],
            cwd=repo,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        runner.wait_for_initial_fuse_request(
            server_process,
            simulator,
            arguments.startup_timeout,
            arguments.fuse_settle,
        )
        sim_and_server = (
            ("TestCluster", simulator),
            ("dayan_server", server_process),
        )
        send_cli(master_fd, drain, "n 1", arguments.startup_timeout, sim_and_server)
        start_continuous(
            master_fd, drain, arguments.startup_timeout, sim_and_server
        )
        runner.wait_for(
            lambda: socket_path.is_socket(),
            arguments.startup_timeout,
            "dayan_server socket",
            sim_and_server,
        )
        stop_continuous(
            master_fd, drain, arguments.startup_timeout, sim_and_server
        )
        print(f"[wave-capture] Dayan socket is ready: {socket_path}", flush=True)

        operator_environment = os.environ.copy()
        operator_environment["DAYAN_SERVER_SOCKET"] = str(socket_path)
        operator_process = subprocess.Popen(
            arguments.command,
            cwd=repo,
            env=operator_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        operator_drain = OperatorDrain(
            operator_process,
            log_dir / "operator.log",
            arguments.marker,
            arguments.expect,
        )
        operator_drain.start()

        stepped_cycles = 0
        while not operator_drain.marker_seen.is_set():
            if operator_process.poll() is not None:
                operator_drain.validate()
                raise runner.RunnerError(
                    f"operator exited before marker {arguments.marker!r}"
                )
            if stepped_cycles >= arguments.max_marker_cycles:
                raise runner.RunnerError(
                    f"marker was not seen within {stepped_cycles} cycles"
                )
            send_cli(
                master_fd,
                drain,
                f"n {arguments.step_cycles}",
                arguments.operator_timeout,
                sim_and_server,
            )
            stepped_cycles += arguments.step_cycles
            print(
                f"[wave-capture] searching for marker: stepped "
                f"{stepped_cycles} cycles",
                flush=True,
            )

        print(
            f"[wave-capture] marker observed after <= {stepped_cycles} "
            "stepped cycles",
            flush=True,
        )
        send_cli(master_fd, drain, "t", arguments.startup_timeout, sim_and_server)
        send_cli(
            master_fd,
            drain,
            f"n {arguments.trace_cycles}",
            arguments.operator_timeout,
            sim_and_server,
        )
        send_cli(master_fd, drain, "t", arguments.startup_timeout, sim_and_server)
        print(
            f"[wave-capture] recorded {arguments.trace_cycles} cycles",
            flush=True,
        )

        if not operator_drain.done.wait(timeout=1.0):
            start_continuous(
                master_fd, drain, arguments.startup_timeout, sim_and_server
            )
            if not operator_drain.done.wait(timeout=arguments.operator_timeout):
                raise runner.RunnerError("timed out waiting for operator completion")
            stop_continuous(
                master_fd, drain, arguments.startup_timeout, sim_and_server
            )
        operator_drain.validate()
        success = True
    finally:
        runner.stop_process(operator_process)
        if simulator is not None and simulator.poll() is None and master_fd >= 0:
            try:
                os.write(master_fd, b"\x03")
                time.sleep(0.5)
                os.write(master_fd, b"quit\n")
                simulator.wait(timeout=15.0)
            except (OSError, subprocess.TimeoutExpired):
                runner.stop_process(simulator)
        runner.stop_process(server_process)
        if server_log is not None:
            server_log.close()
        if drain is not None:
            drain.thread.join(timeout=1.0)
            (log_dir / "testcluster-tail.raw").write_bytes(drain.snapshot())
        if master_fd >= 0:
            try:
                os.close(master_fd)
            except OSError:
                pass
        runner.unmount(
            device_prefix, force=runner.disconnected_mount(device_prefix)
        )
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass

    wave_path = sim_work_dir / "logs" / "TestCluster.fst"
    if not success:
        raise runner.RunnerError("wave capture did not complete")
    runner.require_file(wave_path, "captured waveform")
    if wave_path.stat().st_size == 0:
        raise runner.RunnerError(f"captured waveform is empty: {wave_path}")
    print(
        f"[wave-capture] waveform: {wave_path} "
        f"({wave_path.stat().st_size} bytes)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except runner.RunnerError as error:
        print(f"[wave-capture] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("[wave-capture] interrupted", file=sys.stderr)
        raise SystemExit(130)
