"""Run model-written Python in an isolated child process.

Isolation, strongest available first (what was applied is returned with every result):
  * no credentials: the child gets a minimal environment (no HF / API keys) and, when the
    parent is root (RunPod), runs as `nobody`, so it cannot read the repo's mode-600 .env;
  * no network: a fresh network namespace (`unshare -n`) when the kernel allows it,
    otherwise a one-line prelude that disables the socket module (defence in depth only);
  * limits: wall-clock timeout (process group killed), CPU seconds, address space (Linux),
    file size, no core dumps; a throwaway working directory under /tmp.
`selfcheck()` reports which of these work on this host (scripts/pod/doctor.py runs it).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

NOBODY = 65534
PRELUDE = ("import socket as _s; _s.socket = _s.create_connection = _s.getaddrinfo = "
           "(lambda *a, **k: (_ for _ in ()).throw(OSError('network disabled in sandbox'))); del _s\n")
OUTPUT_CAP = 20_000          # characters kept per stream


@dataclass
class ExecResult:
    success: bool
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    mode: str                # e.g. "netns+nobody", "prelude+nobody", "prelude"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cap(s: str, n: int = OUTPUT_CAP) -> str:
    return s if len(s) <= n else s[: n // 2] + "\n...[truncated]...\n" + s[-n // 2:]


def python_executable() -> str:
    """The system interpreter on Linux: venvs may live where `nobody` cannot traverse (e.g.
    /root/ef_venv, mode 700 /root), and test programs only need the standard library."""
    if os.environ.get("EF_SANDBOX_PYTHON"):
        return os.environ["EF_SANDBOX_PYTHON"]
    if sys.platform.startswith("linux") and os.path.exists("/usr/bin/python3"):
        return "/usr/bin/python3"
    return sys.executable


def _is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


DROP_CMD = ["setpriv", f"--reuid={NOBODY}", f"--regid={NOBODY}", "--clear-groups", "--"]


@lru_cache(maxsize=1)
def netns_available() -> bool:
    """`unshare -n` + `setpriv` work here (root + CAP_SYS_ADMIN; RunPod containers may refuse).
    The namespace must be created as root, so privileges are dropped INSIDE it by setpriv."""
    if not sys.platform.startswith("linux") or not _is_root():
        return False
    if not (shutil.which("unshare") and shutil.which("setpriv")):
        return False
    try:
        return subprocess.run(["unshare", "-n", "--", *DROP_CMD, "true"], capture_output=True,
                              timeout=10).returncode == 0
    except Exception:
        return False


def _preexec_limits(timeout_s: int, mem_mb: int):
    """Fallback when `prlimit` is missing (e.g. macOS). preexec_fn is unsafe in threaded
    parents (vLLM), so on the pod limits come from `prlimit` and the session / uid change
    from Popen's C-level start_new_session / user / group arguments instead."""
    import resource

    cpu, fsize = (timeout_s + 5, timeout_s + 5), (50 * 2**20, 50 * 2**20)

    def fn():
        resource.setrlimit(resource.RLIMIT_CPU, cpu)
        resource.setrlimit(resource.RLIMIT_FSIZE, fsize)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return fn


def run_python(code: str, timeout_s: int = 30, mem_mb: int = 4096, extra_args: tuple[str, ...] = ("-X", "dev"),
               filename: str = "solution_test.py") -> ExecResult:
    """Execute `code` as a script: `python -I <extra_args> <file>`; success = exit status 0."""
    drop = _is_root()
    netns = netns_available()
    prlimit = sys.platform.startswith("linux") and shutil.which("prlimit")
    mode = ("netns" if netns else "prelude") + ("+nobody" if drop else "") + ("+prlimit" if prlimit else "")
    d = tempfile.mkdtemp(prefix="ef_sbx_")
    try:
        os.chmod(d, 0o777)
        path = os.path.join(d, filename)
        with open(path, "w") as f:
            f.write(code if netns else PRELUDE + code)
        os.chmod(path, 0o644)
        cmd = [python_executable(), "-I", *extra_args, path]
        if prlimit:
            cmd = ["prlimit", f"--cpu={timeout_s + 5}", f"--as={mem_mb * 2**20}", f"--fsize={50 * 2**20}",
                   "--core=0", "--", *cmd]
        kw: dict[str, Any] = {}
        if netns:   # namespace made as root, then setpriv drops to nobody before python starts
            cmd = ["unshare", "-n", "--", *DROP_CMD, *cmd]
        elif drop:
            kw.update(user=NOBODY, group=NOBODY, extra_groups=[])
        if not prlimit:
            kw["preexec_fn"] = _preexec_limits(timeout_s, mem_mb)
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": d, "LANG": "C.UTF-8", "PYTHONIOENCODING": "utf-8"}
        p = subprocess.Popen(cmd, cwd=d, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             start_new_session=True, **kw)
        try:
            out, err = p.communicate(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                p.kill()
            out, err = p.communicate()
            timed_out = True
        rc = None if timed_out else p.returncode
        return ExecResult(success=(rc == 0), returncode=rc, stdout=_cap(out.decode("utf-8", "replace")),
                          stderr=_cap(err.decode("utf-8", "replace")), timed_out=timed_out, mode=mode)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def selfcheck(secret_path: str | None = None) -> dict[str, Any]:
    """Which isolation actually holds on this host: can the child read `secret_path`, open a
    socket, see the parent's environment variables? Run before any model-written code."""
    probe = f"""
import os, socket, sys
res = {{}}
try:
    open({secret_path!r}).read(); res['secret_readable'] = True
except Exception as e:
    res['secret_readable'] = False if {secret_path!r} else None
try:
    s = socket.create_connection(('1.1.1.1', 53), timeout=3); s.close(); res['network'] = True
except Exception as e:
    res['network'] = False
res['env_keys'] = sorted(k for k in os.environ if any(t in k for t in ('TOKEN', 'KEY', 'SECRET')))
res['uid'] = os.getuid()
print(res)
"""
    r = run_python(probe, timeout_s=20)
    out = {"mode": r.mode, "ok": r.success, "stderr": r.stderr[-500:]}
    try:
        import ast

        out.update(ast.literal_eval(r.stdout.strip().splitlines()[-1]))
    except Exception:
        out["parse_error"] = r.stdout[-500:]
    return out
