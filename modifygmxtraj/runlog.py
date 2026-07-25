"""Run logging.

Mirrors everything the pipeline prints into a log file, and adds structured
sections recording the verbatim input file, the fully resolved parameters,
and every file written.

Note on scope: GROMACS's own stdout/stderr streams straight to the terminal
(or the SLURM .out file) rather than through Python, so it is not captured
here by default. Set LOG_GMX_OUTPUT = yes in the input file to route it
through Python instead, at the cost of losing live streaming -- each
command's output then appears only once it finishes.
"""

from __future__ import annotations

import datetime
import getpass
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Optional, TextIO


class Tee:
    """Write to a real stream and to the log file at the same time."""

    def __init__(self, stream: TextIO, handle: TextIO) -> None:
        self._stream = stream
        self._handle = handle

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._handle.write(text)
        self._handle.flush()
        return len(text)

    def flush(self) -> None:
        self._stream.flush()
        self._handle.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._stream, "isatty", lambda: False)())

    @property
    def encoding(self) -> str:
        return getattr(self._stream, "encoding", "utf-8")


class RunLog:
    """Context manager that tees stdout/stderr into a log file."""

    def __init__(self, path: Optional[Path]) -> None:
        self.path = Path(path) if path is not None else None
        self.handle: Optional[TextIO] = None
        self.start_time: float = 0.0
        self.files_written: list[tuple[str, str]] = []
        self._saved_stdout: Optional[TextIO] = None
        self._saved_stderr: Optional[TextIO] = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "RunLog":
        self.start_time = time.time()
        if self.path is None:
            return self
        self.handle = self.path.open("w", encoding="utf-8")
        self._saved_stdout = sys.stdout
        self._saved_stderr = sys.stderr
        sys.stdout = Tee(self._saved_stdout, self.handle)  # type: ignore[assignment]
        sys.stderr = Tee(self._saved_stderr, self.handle)  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        elapsed = time.time() - self.start_time
        if exc_type is not None:
            print(f"\nRun FAILED after {elapsed:.1f} s: "
                  f"{exc_type.__name__}: {exc_value}")
        else:
            print(f"\nRun completed in {elapsed:.1f} s "
                  f"({elapsed / 60.0:.1f} min)")
        if self.path is not None:
            print(f"Log written to: {self.path}")
        if self.handle is not None:
            if self._saved_stdout is not None:
                sys.stdout = self._saved_stdout
            if self._saved_stderr is not None:
                sys.stderr = self._saved_stderr
            self.handle.close()
            self.handle = None
        return False  # never suppress the exception

    # -- structured sections ----------------------------------------------

    def write_header(self, version: str, config_path: Path) -> None:
        rule = "=" * 78
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(rule)
        print(f"modifyGMXtraj {version}")
        print(rule)
        print(f"Started            : {now}")
        try:
            print(f"User               : {getpass.getuser()}")
        except Exception:
            pass
        print(f"Host               : {socket.gethostname()}")
        print(f"Platform           : {platform.platform()}")
        print(f"Python             : {sys.version.split()[0]} ({sys.executable})")
        print(f"Working directory  : {Path.cwd()}")
        print(f"Command line       : {' '.join(sys.argv)}")
        print(f"Configuration file : {config_path}")
        if self.path is not None:
            print(f"Log file           : {self.path}")

    def write_input_file(self, config_path: Path) -> None:
        rule = "-" * 78
        print(f"\n{rule}\nINPUT FILE (verbatim)\n{rule}")
        try:
            for number, line in enumerate(
                config_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                print(f"{number:4d} | {line}")
        except OSError as exc:
            print(f"(could not re-read input file: {exc})")

    def write_parameters(self, parameters: list[tuple[str, object]]) -> None:
        rule = "-" * 78
        print(f"\n{rule}\nRESOLVED PARAMETERS\n{rule}")
        width = max((len(k) for k, _ in parameters), default=0)
        for key, value in parameters:
            print(f"  {key:<{width}s} : {value}")

    # -- output inventory --------------------------------------------------

    def record(self, description: str, path: Path | str) -> None:
        self.files_written.append((description, str(path)))

    def write_file_inventory(self) -> None:
        rule = "-" * 78
        print(f"\n{rule}\nFILES WRITTEN\n{rule}")
        if not self.files_written:
            print("  none")
            return
        width = max(len(d) for d, _ in self.files_written)
        for description, path in self.files_written:
            candidate = Path(path)
            if candidate.is_file():
                size_mb = candidate.stat().st_size / (1024.0 * 1024.0)
                size = f"{size_mb:10.2f} MB"
            else:
                size = "         -- "
            print(f"  {description:<{width}s} : {path}  ({size.strip()})")
