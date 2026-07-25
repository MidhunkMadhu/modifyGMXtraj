"""Thin wrappers around GROMACS command-line invocation."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .config import CommandError, ConfigError

DRY_RUN = False


def set_dry_run(enabled: bool) -> None:
    global DRY_RUN
    DRY_RUN = enabled


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
    capture: bool = False,
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

    result = subprocess.run(
        command,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=False,
    )
    if capture and result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        raise CommandError(
            f"Command failed with exit code {result.returncode}: {printable}"
        )
    return result


def tpr_atom_count(gmx: str, tpr: Path) -> Optional[int]:
    """Read the atom count from a .tpr via `gmx dump`, or None if unreadable."""
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
