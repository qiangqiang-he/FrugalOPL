"""POSIX ownership of a Worker and the subprocesses it creates."""
from __future__ import annotations
import os
import signal


def isolate_worker_group() -> None:
    if os.name == "posix" and os.getpgrp() != os.getpid():
        os.setsid()


def signal_owned_worker_group(process: object, signum: int) -> bool:
    """Signal an explicitly isolated Worker group; never its Controller group."""
    if os.name != "posix" or getattr(process, "_frugalopl_owns_process_group", False) is not True:
        return False
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        if os.getpgid(pid) != pid:
            return False
    except ProcessLookupError:
        # Its own group can still contain a vLLM EngineCore after the Worker
        # has exited; the group ID remains the original, manager-owned PID.
        pass
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass
    return True


def cleanup_exited_worker_groups(processes: tuple[object, ...]) -> None:
    for process in processes:
        if not process.is_alive():
            signal_owned_worker_group(process, signal.SIGTERM)
            signal_owned_worker_group(process, signal.SIGKILL)
