#!/usr/bin/env python
"""Verify the PV-MEPCG / PulseVision environment (Phase 02, task T02.6).

Checks, in order:
  1. Python is 3.11.9 and running from the project ``.venv``.
  2. Every package pinned in the four ``requirements*.txt`` files is installed
     AT THAT EXACT VERSION -- a mismatch is a failure, not a warning, because
     package versions are recorded in every run manifest (rule 5).
  3. Every package actually IMPORTS. Installing and importing are different
     things on Windows: numba/llvmlite against the wrong numpy installs fine
     and then dies at import.
  4. Node LTS and npm are on PATH (needed from Part X to build the frontend).
  5. Advisories -- conditions that are not failures but will bite later.

Exits 0 only if every check passes. Any failure exits 1.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQ_FILES = [
    "requirements.txt",
    "requirements-extra.txt",
    "requirements-api.txt",
    "requirements-report.txt",
]

EXPECTED_PYTHON = "3.11.9"

# Distribution name -> importable module name, where the two differ.
IMPORT_NAME = {
    "scikit-learn": "sklearn",
    "scikit-optimize": "skopt",
    "PyWavelets": "pywt",
    "PyYAML": "yaml",
    "python-multipart": "multipart",
    "python-docx": "docx",
    "uvicorn": "uvicorn",
}

_TTY = sys.stdout.isatty()
GREEN = "\033[32m" if _TTY else ""
RED = "\033[31m" if _TTY else ""
YELLOW = "\033[33m" if _TTY else ""
DIM = "\033[2m" if _TTY else ""
RESET = "\033[0m" if _TTY else ""

failures: list[str] = []
advisories: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)
    print("  " + RED + "FAIL" + RESET + "  " + msg)


def advise(msg: str) -> None:
    advisories.append(msg)
    print("  " + YELLOW + "WARN" + RESET + "  " + msg)


def ok(msg: str) -> None:
    print("  " + GREEN + "ok" + RESET + "    " + msg)


PIN_RE = re.compile(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]+\])?==(\S+)$")


def parse_requirements(path: Path) -> list[tuple[str, str]]:
    """Return [(distribution, pinned_version)] parsed from a requirements file."""
    pins: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = PIN_RE.match(line)
        if match is None:
            fail(
                path.name + ": cannot parse pin " + repr(line)
                + " (every line must be name==version)"
            )
            continue
        pins.append((match.group(1), match.group(2)))
    return pins


def all_pins() -> list[tuple[str, str]]:
    pins: list[tuple[str, str]] = []
    for name in REQ_FILES:
        path = ROOT / name
        if path.is_file():
            pins += parse_requirements(path)
    return pins


def check_python() -> None:
    print()
    print(DIM + "[1/5] Python interpreter" + RESET)
    actual = platform.python_version()
    if actual != EXPECTED_PYTHON:
        fail("Python " + actual + " -- this project is locked to " + EXPECTED_PYTHON)
    else:
        ok("Python " + actual)

    exe = Path(sys.executable).resolve()
    venv = (ROOT / ".venv").resolve()
    if venv in exe.parents:
        ok("running from the project venv (" + exe.relative_to(ROOT).as_posix() + ")")
    else:
        fail(
            "not running from " + str(venv) + " -- activate it first "
            "(PowerShell: .venv\\Scripts\\Activate.ps1  |  bash: source .venv/Scripts/activate). "
            "Current interpreter: " + str(exe)
        )


def check_packages() -> None:
    print()
    print(DIM + "[2/5] Pinned packages installed at the pinned version" + RESET)
    for name in REQ_FILES:
        path = ROOT / name
        if not path.is_file():
            fail(name + " is missing")
            continue
        pins = parse_requirements(path)
        print("  " + DIM + "-- " + name + " (" + str(len(pins)) + " pins)" + RESET)
        for dist, pinned in pins:
            try:
                installed = version(dist)
            except PackageNotFoundError:
                fail(dist + " is pinned to " + pinned + " but is not installed")
                continue
            if installed != pinned:
                fail(dist + ": installed " + installed + ", pinned " + pinned)
            else:
                ok(dist + "==" + installed)


def check_imports() -> None:
    print()
    print(DIM + "[3/5] Every pinned package imports" + RESET)
    warnings.filterwarnings("ignore")
    for dist, _ in all_pins():
        module = IMPORT_NAME.get(dist, dist.replace("-", "_"))
        try:
            __import__(module)
        except Exception as exc:  # noqa: BLE001 - any import error is a failure here
            fail(
                "import " + module + " (" + dist + ") raised "
                + type(exc).__name__ + ": " + str(exc)
            )
        else:
            ok("import " + module)


def run_tool(args: list[str]) -> str | None:
    """Run a PATH executable, returning trimmed stdout, or None if it is unusable."""
    exe = shutil.which(args[0])
    if exe is None:
        return None
    try:
        result = subprocess.run(
            [exe, *args[1:]], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def check_node() -> None:
    print()
    print(DIM + "[4/5] Node LTS and npm on PATH (needed from Part X)" + RESET)
    node = run_tool(["node", "--version"])
    if node is None:
        fail("node is not on PATH -- Node LTS is required to build the frontend")
    else:
        match = re.match(r"^v(\d+)\.", node)
        if match is None:
            fail("cannot parse node version " + repr(node))
        else:
            major = int(match.group(1))
            # Node promotes only even-numbered major versions to an LTS line.
            if major % 2 != 0:
                fail("node " + node + " is an odd major version -- not an LTS line")
            elif major < 18:
                fail("node " + node + " is below the Node 18 floor required by Next.js 14")
            else:
                ok("node " + node + " (LTS line)")

    npm = run_tool(["npm", "--version"])
    if npm is None:
        fail("npm is not on PATH")
    else:
        ok("npm " + npm)


def check_advisories() -> None:
    print()
    print(DIM + "[5/5] Advisories (not failures, but they bite later)" + RESET)

    # kaleido 1.x drives an external Chrome through choreographer; 0.2.1 bundled
    # its own Chromium. Nothing in the pipeline renders Plotly from Python yet,
    # so a missing browser is not a failure -- but it must not be a surprise.
    try:
        version("kaleido")
    except PackageNotFoundError:
        pass
    else:
        candidates = [
            Path(os.environ.get("PROGRAMFILES", "")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
        ]
        found = next((str(c) for c in candidates if c.exists()), None) or shutil.which("chrome")
        if found:
            ok("kaleido: Chrome found at " + found)
        else:
            advise(
                "kaleido 1.x needs an external Chrome/Chromium and none was found. "
                "Static Plotly export from Python will fail until kaleido_get_chrome is run. "
                "Not used by the current pipeline -- all figures are matplotlib/seaborn."
            )

    # CPU only, by hardware. Any task assuming CUDA is wrong for this machine.
    ok("CPU-only build assumed -- no GPU check performed (this machine has no CUDA device)")

    if (ROOT / "dataset").is_dir():
        ok("dataset/ present")
    else:
        advise("dataset/ is not present -- every data-dependent task will fail")


def main() -> int:
    print(DIM + "PV-MEPCG / PulseVision -- environment verification" + RESET)
    print(DIM + "project root: " + str(ROOT) + RESET)

    check_python()
    check_packages()
    check_imports()
    check_node()
    check_advisories()

    print()
    if failures:
        print(RED + "FAILED" + RESET + " -- " + str(len(failures)) + " check(s) did not pass:")
        for item in failures:
            print("  - " + item)
        return 1

    suffix = " (" + str(len(advisories)) + " advisory)" if advisories else ""
    print(GREEN + "PASSED" + RESET + " -- environment verified" + suffix)
    for item in advisories:
        print("  " + YELLOW + "advisory" + RESET + ": " + item)
    return 0


if __name__ == "__main__":
    sys.exit(main())
