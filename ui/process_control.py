"""Start and stop a task together with the subprocesses it owns."""
import os
import signal
import subprocess
import time


def process_group_options():
    """Return Popen options that give each managed task its own process group."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {"start_new_session": True}


def terminate_process_tree(process, *, grace=1.0):
    """Stop a process created with process_group_options, including descendants.

    POSIX groups retain their original ID when the parent exits. Wait for the
    group, rather than only the parent, before escalating to SIGKILL. This also
    closes inherited output pipes so the task-monitor thread can finish.
    """
    if os.name == "nt":
        if process.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                timeout=max(grace, 1.0) + 5,
            )
        try:
            process.wait(timeout=max(grace, 1.0) + 1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=2)
        return
    deadline = time.monotonic() + max(grace, 0)
    while time.monotonic() < deadline:
        process.poll()  # Reap the direct child if it has exited.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(min(0.02, max(0, deadline - time.monotonic())))
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=2)
