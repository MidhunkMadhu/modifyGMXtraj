"""Thin wrappers around GROMACS command-line invocation."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .config import CommandError, ConfigError

DRY_RUN = False
CAPTURE = False


def set_dry_run(enabled: bool) -> None:
    global DRY_RUN
    DRY_RUN = enabled


def set_capture(enabled: bool) -> None:
    """When enabled, GROMACS stdout/stderr is routed through Python so the
    run log captures it. Costs live streaming: each command's output appears
    only once it finishes."""
    global CAPTURE
    CAPTURE = enabled


def DRY_RUN_ACTIVE() -> bool:
    return DRY_RUN


def header(message: str) -> None:
    rule = "=" * 78
    print(f"\n{rule}\n{message}\n{rule}", flush=True)


def check_executable(command: str, description: str) -> None:
    fields = shlex.split(command)
    if not fields:
        raise ConfigError(f"{description} is empty")
    if shutil.which(fields[0]) is None:
        raise ConfigError(f"{description} not found in PATH: {fields[0]}")


def gmx_command(gmx: str, subcommand: str, *arguments: str) -> list[str]:
    return shlex.split(gmx) + [subcommand, *arguments]


def run(
    command: list[str],
    selections: Optional[list[str]] = None,
    capture: Optional[bool] = None,
) -> Optional[subprocess.CompletedProcess]:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"\nRunning: {printable}", flush=True)

    input_text: Optional[str] = None
    if selections is not None:
        input_text = "\n".join(str(item) for item in selections) + "\n"
        print("Input: " + " | ".join(str(item) for item in selections), flush=True)

    if DRY_RUN:
        print("[dry-run] not executed")
        return None

    effective_capture = CAPTURE if capture is None else capture

    result = subprocess.run(
        command,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE if effective_capture else None,
        stderr=subprocess.STDOUT if effective_capture else None,
        check=False,
    )
    if effective_capture and result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        raise CommandError(
            f"Command failed with exit code {result.returncode}: {printable}"
        )
    return result


def tpr_atom_count(gmx: str, tpr: Path) -> Optional[int]:
    """Read the atom count from a .tpr via `gmx dump`, or None if unreadable.

    Kept as a best-effort secondary signal, but xtc_atom_count() below is the
    check that actually gates the pipeline, since it needs no text parsing.
    """
    if DRY_RUN:
        return None
    try:
        result = subprocess.run(
            gmx_command(gmx, "dump", "-s", str(tpr)),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("natoms"):
            try:
                return int(stripped.split("=")[1].strip())
            except (IndexError, ValueError):
                return None
    return None


def xtc_atom_count(xtc: Path) -> int:
    """Atom count read straight from the xtc header via MDTraj's low-level
    reader. No topology needed, no subprocess, no text parsing -- this is the
    check that should have caught mismatches like STRUCTURE_FILE carrying LP
    dummy atoms that the production trajectory doesn't (or vice versa).
    """
    from mdtraj.formats import XTCTrajectoryFile

    with XTCTrajectoryFile(str(xtc)) as handle:
        xyz, _time, _step, _box = handle.read(n_frames=1)
    if xyz.shape[0] == 0:
        raise ConfigError(f"{xtc} contains no frames")
    return int(xyz.shape[1])


def xtc_frame_interval_ps(xtc: Path) -> Optional[float]:
    """Time between the first two frames of an xtc, in ps (None if < 2 frames)."""
    from mdtraj.formats import XTCTrajectoryFile

    with XTCTrajectoryFile(str(xtc)) as handle:
        _xyz, time, _step, _box = handle.read(n_frames=2)
    if len(time) < 2:
        return None
    return float(time[1] - time[0])
