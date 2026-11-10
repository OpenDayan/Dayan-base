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

"""Run one Dayan operator against npu-cluster-sim/TestCluster."""

from __future__ import annotations

import argparse
import datetime as dt
import errno
import fcntl
import os
from pathlib import Path
import pty
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time


DEFAULT_SIM_DIR = Path("/home/wangzhe/npu-cluster-sim")
DEFAULT_DRAM_CONFIG = "LPDDR5_8Gb_x16_6400_4ch.ini"
RING_LIMIT = 512 * 1024
SCAN_LIMIT = 256 * 1024


class RunnerError(RuntimeError):
    pass


class PtyDrain:
    def __init__(self, fd: int) -> None:
        self.fd = fd
        self._tail = bytearray()
        self._scan = bytearray()
        self._lock = threading.Lock()
        self.cli_ready = threading.Event()
        self.input_ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while True:
            try:
                data = os.read(self.fd, 64 * 1024)
            except OSError:
                return
            if not data:
                return
            with self._lock:
                self._tail.extend(data)
                if len(self._tail) > RING_LIMIT:
                    del self._tail[:-RING_LIMIT]
                self._scan.extend(data)
                if len(self._scan) > SCAN_LIMIT:
                    del self._scan[:-SCAN_LIMIT]
                if b"Starting main CLI loop" in self._scan:
                    self.cli_ready.set()
                if "等待命令".encode() in self._scan:
                    self.input_ready.set()

    def snapshot(self) -> bytes:
        with self._lock:
            return bytes(self._tail)

    def clear_scan(self) -> None:
        with self._lock:
            self._scan.clear()

    def contains(self, marker: str) -> bool:
        encoded = marker.encode()
        with self._lock:
            return encoded in self._scan


def parse_int(value: str) -> int:
    try:
        return int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid integer: {value}") from error


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run one host operator through TestCluster and dayan_server."
    )
    result.add_argument("--repo", type=Path, default=Path.cwd())
    result.add_argument(
        "--sim-dir",
        type=Path,
        default=Path(os.environ.get("NPU_CLUSTER_SIM_DIR", DEFAULT_SIM_DIR)),
    )
    result.add_argument("--sim-bin", type=Path)
    result.add_argument("--dram-config", type=Path)
    result.add_argument(
        "--device-prefix",
        type=Path,
        default=Path(os.environ.get("NPU_DEVICE_PREFIX", "/tmp/virtual_npu")),
    )
    result.add_argument("--server", type=Path)
    result.add_argument("--firmware", type=Path)
    result.add_argument("--api-elf", type=Path)
    result.add_argument("--cluster-count", type=int, default=1)
    result.add_argument("--core-mask", type=parse_int, default=1)
    result.add_argument(
        "--cores-per-cluster",
        type=int,
        help="Cores the simulated hardware has (1 for a 1c build); default: the server's 4",
    )
    result.add_argument("--socket", type=Path)
    result.add_argument("--log-dir", type=Path)
    result.add_argument("--startup-timeout", type=float, default=30.0)
    result.add_argument("--server-timeout-ms", type=int, default=60000)
    result.add_argument(
        "--fuse-settle",
        type=float,
        default=1.0,
        help="seconds to leave the initial FUSE request queued before starting simulation",
    )
    result.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="TEXT",
        help="require TEXT in operator output; repeat as needed",
    )
    result.add_argument("command", nargs=argparse.REMAINDER)
    return result


def rooted(path: Path | None, root: Path, default: str) -> Path:
    selected = path if path is not None else Path(default)
    if not selected.is_absolute():
        selected = root / selected
    return selected.resolve()


def require_file(path: Path, label: str, executable: bool = False) -> None:
    if not path.is_file():
        raise RunnerError(f"missing {label}: {path}")
    if executable and not os.access(path, os.X_OK):
        raise RunnerError(f"{label} is not executable: {path}")


def is_mountpoint(path: Path) -> bool:
    completed = subprocess.run(
        ["mountpoint", "-q", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def wait_for(predicate, timeout: float, description: str, processes=()) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        for name, process in processes:
            status = process.poll()
            if status is not None:
                raise RunnerError(f"{name} exited early with status {status}")
        time.sleep(0.05)
    raise RunnerError(f"timed out waiting for {description}")


def fuse_request_is_queued(process: subprocess.Popen) -> bool:
    task_root = Path(f"/proc/{process.pid}/task")
    try:
        tasks = tuple(task_root.iterdir())
    except OSError:
        return False
    for task in tasks:
        try:
            wait_channel = (task / "wchan").read_text().strip()
        except OSError:
            continue
        if wait_channel == "request_wait_answer":
            return True
    return False


def wait_for_initial_fuse_request(
    server: subprocess.Popen,
    simulator: subprocess.Popen,
    timeout: float,
    settle: float,
) -> None:
    deadline = time.monotonic() + min(timeout, 3.0)
    while time.monotonic() < deadline:
        if fuse_request_is_queued(server):
            print("[testcluster-runner] initial FUSE request is queued", flush=True)
            time.sleep(settle)
            return
        if server.poll() is not None:
            raise RunnerError(f"dayan_server exited early with status {server.returncode}")
        if simulator.poll() is not None:
            raise RunnerError(f"TestCluster exited early with status {simulator.returncode}")
        time.sleep(0.05)
    print(
        "[testcluster-runner] FUSE wait state was not observable; continuing after settle",
        flush=True,
    )


def stop_process(process: subprocess.Popen | None, timeout: float = 5.0) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=timeout)


def disconnected_mount(path: Path) -> bool:
    try:
        path.stat()
    except OSError as error:
        return error.errno == errno.ENOTCONN
    return False


def unmount(path: Path, force: bool = False) -> None:
    if not force and not is_mountpoint(path):
        return
    attempts = (["fusermount3", "-u", str(path)], ["fusermount3", "-uz", str(path)])
    for command in attempts:
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode == 0:
            return
        if not force and not is_mountpoint(path):
            return


def run_operator(
    command: list[str], repo: Path, socket_path: Path, log_path: Path, expected: list[str]
) -> int:
    environment = os.environ.copy()
    environment["DAYAN_SERVER_SOCKET"] = str(socket_path)
    patterns = [item.encode() for item in expected]
    found = [False] * len(patterns)
    carry = b""
    carry_limit = max((len(item) for item in patterns), default=1) - 1

    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        assert process.stdout is not None
        try:
            while True:
                chunk = process.stdout.read(64 * 1024)
                if not chunk:
                    break
                log.write(chunk)
                log.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                searchable = carry + chunk
                for index, pattern in enumerate(patterns):
                    if not found[index] and pattern in searchable:
                        found[index] = True
                carry = searchable[-carry_limit:] if carry_limit else b""
            status = process.wait()
        except BaseException:
            stop_process(process)
            raise

    if status != 0:
        raise RunnerError(f"operator exited with status {status}")
    missing = [expected[index] for index, present in enumerate(found) if not present]
    if missing:
        raise RunnerError("operator output is missing: " + ", ".join(repr(item) for item in missing))
    return status


def main() -> int:
    arguments = parser().parse_args()
    if not arguments.command:
        raise RunnerError("provide an operator command after --")
    if arguments.command[0] == "--":
        arguments.command = arguments.command[1:]
    if not arguments.command:
        raise RunnerError("provide an operator command after --")
    if arguments.cluster_count < 1 or arguments.cluster_count > 4:
        raise RunnerError("--cluster-count must be between 1 and 4")
    if arguments.core_mask <= 0 or arguments.core_mask > 0xF:
        raise RunnerError("--core-mask must be between 1 and 0xf")
    if arguments.cores_per_cluster is not None and not 1 <= arguments.cores_per_cluster <= 4:
        raise RunnerError("--cores-per-cluster must be between 1 and 4")
    if arguments.startup_timeout <= 0:
        raise RunnerError("--startup-timeout must be positive")
    if arguments.server_timeout_ms <= 0:
        raise RunnerError("--server-timeout-ms must be positive")
    if arguments.fuse_settle < 0:
        raise RunnerError("--fuse-settle cannot be negative")

    repo = arguments.repo.resolve()
    sim_dir = arguments.sim_dir.resolve()
    sim_bin = rooted(arguments.sim_bin, sim_dir, "TestCluster")
    dram_config = rooted(
        arguments.dram_config,
        sim_dir,
        f"configs/{DEFAULT_DRAM_CONFIG}",
    )
    server = rooted(arguments.server, repo, "build/host/driver/dayan_server")
    firmware = rooted(arguments.firmware, repo, "build/device/npu-os.elf")
    api_elf = rooted(arguments.api_elf, repo, "build/device/npu-os-api.elf")
    device_prefix = arguments.device_prefix.resolve()
    socket_path = (
        arguments.socket.resolve()
        if arguments.socket
        else Path(tempfile.gettempdir()) / f"dayan-testcluster-{os.getpid()}.sock"
    )
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = (
        arguments.log_dir.resolve()
        if arguments.log_dir
        else repo / "build" / "testcluster-runs" / f"{timestamp}-{os.getpid()}"
    )

    require_file(sim_bin, "TestCluster", executable=True)
    require_file(dram_config, "DRAM configuration")
    require_file(server, "dayan_server", executable=True)
    require_file(firmware, "firmware ELF")
    require_file(api_elf, "API ELF")
    if not repo.is_dir():
        raise RunnerError(f"missing repository directory: {repo}")
    if is_mountpoint(device_prefix):
        raise RunnerError(f"device prefix is already mounted: {device_prefix}")
    if disconnected_mount(device_prefix):
        print(
            f"[testcluster-runner] recovering stale FUSE endpoint: {device_prefix}",
            flush=True,
        )
        unmount(device_prefix, force=True)
        if disconnected_mount(device_prefix):
            raise RunnerError(f"cannot recover stale FUSE endpoint: {device_prefix}")
    if socket_path.exists():
        raise RunnerError(f"socket path already exists: {socket_path}")

    device_prefix.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=False)
    print(f"[testcluster-runner] logs: {log_dir}", flush=True)

    simulator: subprocess.Popen | None = None
    server_process: subprocess.Popen | None = None
    server_log = None
    master_fd = -1
    drain: PtyDrain | None = None

    try:
        master_fd, slave_fd = pty.openpty()
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 120, 0, 0))

        def simulator_child_setup() -> None:
            os.setsid()
            fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)

        environment = os.environ.copy()
        environment["SIM_DRAM_CONFIG"] = str(dram_config)
        environment["SIM_MOUNT_POINT"] = str(device_prefix)
        simulator = subprocess.Popen(
            [str(sim_bin)],
            cwd=sim_dir,
            env=environment,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            preexec_fn=simulator_child_setup,
            close_fds=True,
        )
        os.close(slave_fd)
        drain = PtyDrain(master_fd)
        drain.start()

        endpoints = (
            device_prefix / "xdma0_h2c_0",
            device_prefix / "xdma0_c2h_0",
            device_prefix / "xdma0_events_0",
        )
        wait_for(
            lambda: drain.cli_ready.is_set()
            and drain.input_ready.is_set()
            and all(path.exists() for path in endpoints),
            arguments.startup_timeout,
            "TestCluster CLI and FUSE endpoints",
            (("TestCluster", simulator),),
        )
        print("[testcluster-runner] TestCluster CLI and FUSE endpoints are ready", flush=True)

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
                *(
                    ["--cores-per-cluster", str(arguments.cores_per_cluster)]
                    if arguments.cores_per_cluster is not None
                    else []
                ),
                "--timeout-ms",
                str(arguments.server_timeout_ms),
            ],
            cwd=repo,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        wait_for_initial_fuse_request(
            server_process,
            simulator,
            arguments.startup_timeout,
            arguments.fuse_settle,
        )
        drain.clear_scan()
        os.write(master_fd, b"n 1\n")
        wait_for(
            lambda: drain.contains("➜ n 1"),
            arguments.startup_timeout,
            "TestCluster to accept the one-cycle prime",
            (("TestCluster", simulator), ("dayan_server", server_process)),
        )
        drain.clear_scan()
        wait_for(
            lambda: drain.contains("等待命令"),
            arguments.startup_timeout,
            "the one-cycle prime to complete",
            (("TestCluster", simulator), ("dayan_server", server_process)),
        )
        time.sleep(0.5)
        drain.clear_scan()
        os.write(master_fd, b"c\n")
        wait_for(
            lambda: drain.contains("➜ c") or drain.contains("命令执行中"),
            arguments.startup_timeout,
            "TestCluster to enter continuous simulation",
            (("TestCluster", simulator), ("dayan_server", server_process)),
        )
        wait_for(
            lambda: socket_path.is_socket(),
            arguments.startup_timeout,
            "dayan_server socket",
            (("TestCluster", simulator), ("dayan_server", server_process)),
        )
        print(f"[testcluster-runner] Dayan socket is ready: {socket_path}", flush=True)

        status = run_operator(
            arguments.command,
            repo,
            socket_path,
            log_dir / "operator.log",
            arguments.expect,
        )
        if simulator.poll() is not None:
            raise RunnerError(f"TestCluster exited with status {simulator.returncode}")
        if server_process.poll() is not None:
            raise RunnerError(f"dayan_server exited with status {server_process.returncode}")
        print(f"[testcluster-runner] operator passed with status {status}", flush=True)
        return 0
    finally:
        stop_process(server_process)
        if server_log is not None:
            server_log.close()
        if simulator is not None and simulator.poll() is None and master_fd >= 0:
            try:
                os.write(master_fd, b"\x03")
                time.sleep(0.5)
                os.write(master_fd, b"quit\n")
                simulator.wait(timeout=10.0)
            except (OSError, subprocess.TimeoutExpired):
                stop_process(simulator)
        if drain is not None:
            drain.thread.join(timeout=1.0)
            (log_dir / "testcluster-tail.raw").write_bytes(drain.snapshot())
        if master_fd >= 0:
            try:
                os.close(master_fd)
            except OSError:
                pass
        unmount(device_prefix, force=disconnected_mount(device_prefix))
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunnerError as error:
        print(f"[testcluster-runner] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("[testcluster-runner] interrupted", file=sys.stderr)
        raise SystemExit(130)
