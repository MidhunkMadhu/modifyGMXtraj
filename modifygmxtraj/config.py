"""Shared configuration parsing for modifyGMXtraj.

In the original two-script layout, modify_traj.py and make_ndx.py each carried
their own copy of the config parser, the TRAJOUT expression canonicaliser and
the numeric validators. They could silently drift apart. Here there is exactly
one parser and both the pipeline and the indexer consume its output.
"""

from __future__ import annotations

import math
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

NONE_WORDS = {"", "NONE", "NO", "FALSE", "OFF", "NULL", "NA", "N/A", "-"}


class ConfigError(RuntimeError):
    """Invalid user configuration."""


class CommandError(RuntimeError):
    """External command failure."""


@dataclass(frozen=True)
class TrajectoryEntry:
    trajectory: Path
    topology: Path
    length_ns: float


@dataclass(frozen=True)
class OutputRequest:
    expression: str
    cutdown_ps: float


# ---------------------------------------------------------------------------
# Scalar validators
# ---------------------------------------------------------------------------

def parse_nonnegative_float(value: str, key: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ConfigError(f"{key} must be numeric; received: {value}") from exc
    if not math.isfinite(number) or number < 0:
        raise ConfigError(f"{key} must be a finite non-negative number: {value}")
    return number


def parse_positive_float(value: str, key: str) -> float:
    number = parse_nonnegative_float(value, key)
    if number <= 0:
        raise ConfigError(f"{key} must be greater than zero; received: {value}")
    return number


def parse_positive_int(value: str, key: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer; received: {value}") from exc
    if number < 1:
        raise ConfigError(f"{key} must be at least 1; received: {number}")
    return number


def parse_bool(value: str, key: str) -> bool:
    normalized = value.strip().upper()
    if normalized in {"YES", "TRUE", "ON", "1"}:
        return True
    if normalized in {"NO", "FALSE", "OFF", "0", "NONE"}:
        return False
    raise ConfigError(f"{key} must be yes or no; received: {value}")


def parse_name_list(value: Optional[str]) -> list[str]:
    if value is None or value.strip().upper() in NONE_WORDS:
        return []
    names: list[str] = []
    for token in shlex.split(value.replace(",", " ")):
        name = token.strip().upper()
        if name and name not in names:
            names.append(name)
    return names


def parse_optional_range(
    value: Optional[str], key: str
) -> Optional[tuple[int, int]]:
    if value is None or value.strip().upper() in NONE_WORDS:
        return None
    numbers = re.findall(r"\d+", value)
    if len(numbers) != 2:
        raise ConfigError(
            f"{key} must contain two positional residue numbers, for example "
            f"'{key} = 329 to 560'"
        )
    start, end = int(numbers[0]), int(numbers[1])
    if start < 1 or end < start:
        raise ConfigError(f"Invalid {key} range: {start} to {end}")
    return start, end


# ---------------------------------------------------------------------------
# TRAJOUT expressions
# ---------------------------------------------------------------------------

def canonical_expression(value: str) -> str:
    compact = re.sub(r"\s+", "", value).upper()
    components = [component for component in compact.split("+") if component]
    if not components:
        raise ConfigError(f"Invalid TRAJOUT expression: {value}")
    unique: list[str] = []
    for component in components:
        if component not in unique:
            unique.append(component)
    return "+".join(unique)


def parse_output_request(value: str, default_cutdown_ps: float) -> OutputRequest:
    text = value.strip()
    if not text:
        raise ConfigError("TRAJOUT cannot be empty")

    parts = text.rsplit(maxsplit=1)
    if len(parts) == 2:
        try:
            interval = float(parts[1])
            expression_text = parts[0]
        except ValueError:
            interval = default_cutdown_ps
            expression_text = text
    else:
        interval = default_cutdown_ps
        expression_text = text

    if not math.isfinite(interval) or interval <= 0:
        raise ConfigError(
            f"TRAJOUT cutdown interval must be greater than zero: {value}"
        )
    return OutputRequest(canonical_expression(expression_text), interval)


def output_slug(expression: str) -> str:
    slug = re.sub(r"[^A-Z0-9]+", "_", expression.upper()).strip("_").lower()
    if not slug:
        raise ConfigError(f"Could not create an output name from: {expression}")
    return slug


# ---------------------------------------------------------------------------
# Config file
# ---------------------------------------------------------------------------

def read_config(
    path: Path,
) -> tuple[dict[str, str], list[TrajectoryEntry], list[OutputRequest]]:
    """Parse the input file into settings, TRAJIN entries and TRAJOUT requests.

    TRAJIN takes two fields: the XTC and the segment length in ns. The TPR is
    always the XTC path with a .tpr suffix.
    """
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")

    settings: dict[str, str] = {}
    trajectories: list[TrajectoryEntry] = []
    raw_outputs: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.split(";", 1)[0].strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ConfigError(
                    f"Invalid configuration line {line_number}: {raw_line.rstrip()}"
                )

            key, value = line.split("=", 1)
            key = key.strip().upper()
            value = value.strip()

            if key == "TRAJIN":
                fields = shlex.split(value)
                if len(fields) != 2:
                    raise ConfigError(
                        f"TRAJIN on line {line_number} must contain exactly "
                        "'XTC LENGTH_NS'; the .tpr is inferred from the XTC name"
                    )
                xtc = Path(fields[0])
                trajectories.append(
                    TrajectoryEntry(
                        trajectory=xtc,
                        topology=xtc.with_suffix(".tpr"),
                        length_ns=parse_nonnegative_float(
                            fields[1], f"TRAJIN length on line {line_number}"
                        ),
                    )
                )
            elif key == "TRAJOUT":
                raw_outputs.append(value)
            else:
                settings[key] = value

    if not trajectories:
        raise ConfigError("At least one TRAJIN entry is required")

    for position, entry in enumerate(trajectories[:-1], start=1):
        if entry.length_ns <= 0:
            raise ConfigError(
                f"TRAJIN #{position} must have a positive length in ns because "
                f"it determines the start time of TRAJIN #{position + 1}"
            )
    if not math.isclose(trajectories[-1].length_ns, 0.0, abs_tol=1e-12):
        raise ConfigError(
            "The final TRAJIN length must be 0 because it is not used to shift "
            "another trajectory"
        )

    default_cutdown = parse_positive_float(
        setting(settings, "DEFAULT_TRAJOUT_CUTDOWN_PS", "200"),
        "DEFAULT_TRAJOUT_CUTDOWN_PS",
    )
    outputs = [parse_output_request(v, default_cutdown) for v in raw_outputs]

    seen_expressions: set[str] = set()
    seen_slugs: set[str] = set()
    for request in outputs:
        slug = output_slug(request.expression)
        if request.expression in seen_expressions:
            raise ConfigError(f"Duplicate TRAJOUT expression: {request.expression}")
        if slug in seen_slugs:
            raise ConfigError(
                f"TRAJOUT expressions produce the same filename slug: {slug}"
            )
        seen_expressions.add(request.expression)
        seen_slugs.add(slug)

    return settings, trajectories, outputs


def setting(
    settings: dict[str, str],
    key: str,
    default: Optional[str] = None,
    required: bool = False,
) -> str:
    if key in settings:
        return settings[key]
    if required:
        raise ConfigError(f"Required setting is missing: {key}")
    if default is None:
        raise ConfigError(f"No value is available for setting: {key}")
    return default


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise ConfigError(f"{description} not found: {path}")


# ---------------------------------------------------------------------------
# .mdp discovery for REMOVE_INITIAL_PS
# ---------------------------------------------------------------------------

def find_mdp_file(trajectory: Path) -> Path:
    candidate = trajectory.with_suffix(".mdp")
    if candidate.is_file():
        return candidate

    directory = trajectory.parent if str(trajectory.parent) else Path(".")
    matches = sorted(directory.glob("*.mdp"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ConfigError(
            f"No .mdp file found for {trajectory} (looked for {candidate.name} "
            f"and any *.mdp in {directory}). Set REMOVE_INITIAL_PS explicitly."
        )
    raise ConfigError(
        f"Ambiguous .mdp for {trajectory}; {len(matches)} candidates in "
        f"{directory}: {', '.join(m.name for m in matches)}. "
        "Set REMOVE_INITIAL_PS explicitly."
    )


def parse_mdp_value(mdp_path: Path, key: str) -> Optional[str]:
    with mdp_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split(";", 1)[0].strip()
            if not line or "=" not in line:
                continue
            found_key, value = line.split("=", 1)
            if found_key.strip().lower() == key.lower():
                return value.strip()
    return None


def remove_initial_ps_from_mdp(mdp_path: Path, verbose: bool = True) -> float:
    """dt * nstxout-compressed  ->  the wall time of one written frame."""
    dt_text = parse_mdp_value(mdp_path, "dt")
    if dt_text is None:
        raise ConfigError(f"'dt' not found in {mdp_path}")

    source_key = "nstxout-compressed"
    nst_text = parse_mdp_value(mdp_path, source_key)
    if nst_text is None or float(nst_text) == 0:
        source_key = "nstxout"
        nst_text = parse_mdp_value(mdp_path, source_key)
    if nst_text is None or float(nst_text) == 0:
        raise ConfigError(
            f"Neither a nonzero nstxout-compressed nor nstxout found in {mdp_path}"
        )

    dt_ps, nst = float(dt_text), float(nst_text)
    interval_ps = dt_ps * nst
    if verbose:
        print(
            f"  Auto-detected removal window from {mdp_path.name}: "
            f"dt={dt_ps:g} ps x {source_key}={nst:g} -> {interval_ps:g} ps"
        )
    return interval_ps


def trajectory_start_times(entries: list[TrajectoryEntry]) -> list[float]:
    starts: list[float] = []
    cumulative_ps = 0.0
    for entry in entries:
        starts.append(cumulative_ps)
        cumulative_ps += entry.length_ns * 1000.0
    return starts


def format_interval(interval_ps: float) -> str:
    return f"{interval_ps:g}"
