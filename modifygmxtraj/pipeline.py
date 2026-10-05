"""Trajectory concatenation, PBC correction and per-TRAJOUT extraction."""

from __future__ import annotations

import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from .config import (
    ConfigError,
    TrajectoryEntry,
    find_mdp_file,
    format_interval,
    parse_bool,
    parse_nonnegative_float,
    parse_positive_float,
    parse_positive_int,
    read_config,
    remove_initial_ps_from_mdp,
    require_file,
    setting,
    trajectory_start_times,
)
from . import gmx as gmx_module
from .gmx import check_executable, gmx_command, header, run, set_capture, xtc_atom_count
from .indexer import build_index
from .mindist import write_mindist_script
from .runlog import RunLog

# ---------------------------------------------------------------------------
# Group selections for WHOLE-trajectory operations.
#
# These deliberately use GROMACS's own inherent default groups, which gmx
# builds automatically from the topology whenever -n is NOT supplied:
#     0 = System, 1 = Protein
#
# Supplying -n REPLACES that automatic classification rather than adding to
# it, so passing the custom gpcr_only.ndx here makes `-pbc cluster` operate on
# a different atom set than gmx's own bookkeeping expects -- which is exactly
# what broke on LP/dummy-atom-containing systems. The custom index is used
# ONLY for per-TRAJOUT extraction and the receptor-helix fit, where the
# required selections (MAIN, MAIN+LIPIDS, RECEPTOR_HELICES_CA, ...) simply do
# not exist among gmx's defaults.
# ---------------------------------------------------------------------------
PBC_WHOLE_GROUP = "0"          # System
PBC_CLUSTER_PIVOT_GROUP = "1"  # Protein
PBC_OUTPUT_GROUP = "0"         # System
PBC_CENTER_GROUP = "1"         # Protein
PBC_FIT_GROUP = "1"            # Protein


@dataclass(frozen=True)
class OutputProduct:
    expression: str
    structure: Path
    trajectory: Path
    fitted_trajectory: Optional[Path]
    reduced_trajectory: Path


T = TypeVar("T")


def temporary_path(source: Path, label: str, directory: Optional[Path] = None) -> Path:
    name = f"{source.stem}_{label}{source.suffix}"
    return directory / name if directory is not None else source.with_name(name)


class _ThreadBufferedStdout:
    """sys.stdout stand-in used while independent steps run in parallel.

    Each worker thread's prints go to its own buffer, which is written out as
    one block when the worker finishes, so the console and the run log show
    every output set's commands together instead of interleaved line by line.
    Threads without a buffer (the main thread) write straight through.
    """

    def __init__(self, stream) -> None:
        self._stream = stream
        self._local = threading.local()

    def start(self) -> None:
        self._local.buffer = []

    def stop(self) -> str:
        text = "".join(getattr(self._local, "buffer", None) or [])
        self._local.buffer = None
        return text

    def write(self, text: str) -> int:
        buffer = getattr(self._local, "buffer", None)
        if buffer is not None:
            buffer.append(text)
            return len(text)
        return self._stream.write(text)

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def run_tasks(tasks: list[Callable[[], T]], n_jobs: int) -> list[T]:
    """Run independent pipeline steps, in parallel when n_jobs > 1.

    Each step is a sequence of GROMACS subprocesses, so threads are enough:
    the work happens in the child processes. GROMACS output is captured while
    steps run concurrently (it would otherwise interleave on the terminal) and
    printed with the rest of that step's block. Results come back in task
    order; the first failure is raised after the other steps have finished.
    """
    if n_jobs <= 1 or len(tasks) <= 1:
        return [task() for task in tasks]

    saved_stdout = sys.stdout
    saved_capture = gmx_module.CAPTURE
    wrapper = _ThreadBufferedStdout(saved_stdout)
    lock = threading.Lock()

    def call(task: Callable[[], T]) -> T:
        wrapper.start()
        try:
            return task()
        finally:
            text = wrapper.stop()
            with lock:
                saved_stdout.write(text)
                saved_stdout.flush()

    print(f"\nRunning {len(tasks)} independent steps with up to {n_jobs} in parallel", flush=True)
    sys.stdout = wrapper  # type: ignore[assignment]
    set_capture(True)
    try:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(call, task) for task in tasks]
            return [future.result() for future in futures]
    finally:
        sys.stdout = saved_stdout
        set_capture(saved_capture)


def check_multiple_of(value: float, base: float, key: str) -> None:
    """Downsampling an already downsampled trajectory only works on multiples."""
    ratio = value / base
    if round(ratio) < 1 or abs(ratio - round(ratio)) > 1e-6:
        raise ConfigError(
            f"{key} = {value:g} ps is not a whole multiple of EARLY_CUTDOWN_PS = "
            f"{base:g} ps; frames at that interval would not exist after the early "
            "cutdown. Use a multiple, or lower EARLY_CUTDOWN_PS."
        )


def default_interval_filename(base: str, interval_ps: float) -> str:
    return f"{base}_{format_interval(interval_ps)}ps.xtc"


def product_filenames(slug: str, cutdown_ps: float) -> tuple[str, str]:
    """Fitted and reduced trajectory names for a TRAJOUT slug.

    MAIN is the primary product of the pipeline, so it drops the redundant
    'main' infix: traj_fit.xtc / traj_fit_200ps.xtc rather than
    traj_main_fit.xtc / traj_main_fit_200ps.xtc. Every other selection keeps
    its slug so the files stay distinguishable.
    """
    interval = format_interval(cutdown_ps)
    if slug == "main":
        return "traj_fit.xtc", f"traj_fit_{interval}ps.xtc"
    return f"traj_{slug}_fit.xtc", f"traj_{slug}_fit_{interval}ps.xtc"


def prepare_sequential_segments(
    gmx: str,
    entries: list[TrajectoryEntry],
    remove_initial_ps_override: Optional[float],
    temporary_dir: Optional[Path] = None,
) -> tuple[list[Path], list[Path]]:
    header("Preparing sequential trajectory segments with manual time shifts")

    starts = trajectory_start_times(entries)
    concat_inputs: list[Path] = []
    temporary_files: list[Path] = []

    for position, (entry, assigned_start_ps) in enumerate(zip(entries, starts)):
        require_file(entry.trajectory, "Trajectory")
        require_file(entry.topology, "Trajectory topology (inferred from XTC name)")

        print(
            f"\n{position + 1}: {entry.trajectory}\n"
            f"   TPR (inferred): {entry.topology}\n"
            f"   declared length: {entry.length_ns:g} ns\n"
            f"   assigned segment origin: {assigned_start_ps:g} ps"
        )

        if position == 0:
            print("   first segment defines time zero; used unchanged")
            concat_inputs.append(entry.trajectory)
            continue

        if remove_initial_ps_override is not None:
            trim_begin = remove_initial_ps_override
            print(
                f"   removal window: {trim_begin:g} ps (explicit REMOVE_INITIAL_PS)"
            )
        else:
            trim_begin = remove_initial_ps_from_mdp(find_mdp_file(entry.trajectory))

        trimmed = temporary_path(entry.trajectory, "trimmed", temporary_dir)
        shifted = temporary_path(entry.trajectory, "shifted", temporary_dir)

        run(
            gmx_command(
                gmx, "trjconv",
                "-f", str(entry.trajectory),
                "-s", str(entry.topology),
                "-o", str(trimmed),
                "-b", f"{trim_begin:g}",
                "-quiet",
            ),
            selections=["0"],
        )
        run(
            gmx_command(
                gmx, "trjconv",
                "-f", str(trimmed),
                "-s", str(entry.topology),
                "-o", str(shifted),
                "-t0", f"{assigned_start_ps:g}",
                "-quiet",
            ),
            selections=["0"],
        )
        concat_inputs.append(shifted)
        temporary_files.extend([trimmed, shifted])

    return concat_inputs, temporary_files


def concatenate(gmx: str, inputs: list[Path], output: Path, log: RunLog) -> None:
    header("Concatenating production trajectories")
    for number, path in enumerate(inputs, start=1):
        print(f"{number:3d}: {path}")

    if len(inputs) == 1:
        print(f"Single segment; copying to {output}")
        shutil.copy2(inputs[0], output)
    else:
        command = gmx_command(gmx, "trjcat", "-f")
        command.extend(str(path) for path in inputs)
        command.extend(["-o", str(output), "-cat"])
        run(command)
    require_file(output, "Combined trajectory")
    log.record("Combined trajectory", output)


def process_pbc(
    gmx: str,
    combined: Path,
    reference_tpr: Path,
    output: Path,
    log: RunLog,
    early_cutdown_ps: float = 0.0,
    temporary_dir: Optional[Path] = None,
) -> list[Path]:
    """PBC correction on the WHOLE trajectory using gmx's inherent default
    groups. No -n here by design; see the comment above PBC_WHOLE_GROUP.

    With early_cutdown_ps > 0 the first pass keeps only frames at that
    interval (-dt), so the three later passes and every TRAJOUT step handle
    fewer frames. The PBC treatment of each kept frame is unchanged."""
    header("Applying PBC correction, centring and whole-system fitting")
    print("Using GROMACS inherent default groups (no -n): 0 = System, 1 = Protein")
    if early_cutdown_ps > 0:
        print(f"  early cutdown: only frames every {early_cutdown_ps:g} ps are kept from step 1 on")
    print("  step 1: -pbc whole      output System")
    print("  step 2: -pbc cluster    pivot Protein, output System")
    print("  step 3: -pbc mol        output System")
    print("  step 4: -center Protein, -fit rot+trans on Protein, output System")

    directory = temporary_dir if temporary_dir is not None else Path(".")
    whole = directory / "temporary_pbc_whole.xtc"
    clustered = directory / "temporary_pbc_cluster.xtc"
    molecule = directory / "temporary_pbc_molecule.xtc"

    first_pass = gmx_command(
        gmx, "trjconv", "-f", str(combined), "-s", str(reference_tpr),
        "-o", str(whole), "-pbc", "whole",
    )
    if early_cutdown_ps > 0:
        first_pass.extend(["-dt", f"{early_cutdown_ps:g}"])
    run(first_pass, selections=[PBC_WHOLE_GROUP])
    run(
        gmx_command(
            gmx, "trjconv", "-f", str(whole), "-s", str(reference_tpr),
            "-o", str(clustered), "-pbc", "cluster",
        ),
        selections=[PBC_CLUSTER_PIVOT_GROUP, PBC_OUTPUT_GROUP],
    )
    run(
        gmx_command(
            gmx, "trjconv", "-f", str(clustered), "-s", str(reference_tpr),
            "-o", str(molecule), "-pbc", "mol",
        ),
        selections=[PBC_OUTPUT_GROUP],
    )
    run(
        gmx_command(
            gmx, "trjconv", "-f", str(molecule), "-s", str(reference_tpr),
            "-o", str(output), "-center", "-fit", "rot+trans",
        ),
        selections=[PBC_CENTER_GROUP, PBC_FIT_GROUP, PBC_OUTPUT_GROUP],
    )

    require_file(output, "PBC-corrected full trajectory")
    log.record("Full PBC-fitted trajectory", output)
    return [whole, clustered, molecule]


def downsample(
    gmx: str,
    trajectory: Path,
    topology: Path,
    output: Path,
    interval_ps: float,
    output_group: str = "0",
    index_file: Optional[Path] = None,
) -> None:
    command = gmx_command(
        gmx, "trjconv", "-f", str(trajectory), "-s", str(topology)
    )
    if index_file is not None:
        command.extend(["-n", str(index_file)])
    command.extend(["-o", str(output), "-dt", f"{interval_ps:g}", "-quiet"])
    run(command, selections=[output_group])
    require_file(output, "Downsampled trajectory")


def _make_output_product(
    gmx: str,
    item: dict[str, Any],
    full_trajectory: Path,
    reference_tpr: Path,
    structure_file: Path,
    index_file: Path,
    fit_group: str,
) -> tuple[OutputProduct, list[tuple[str, Path]]]:
    """Build one TRAJOUT set. Returns the product and its log records, so the
    caller can record them in input order even when sets run in parallel."""
    expression = str(item["expression"])
    group_name = str(item["group"])
    slug = str(item["slug"])
    contains_receptor = bool(item["contains_receptor"])
    cutdown_ps = float(item["cutdown_ps"])
    records: list[tuple[str, Path]] = []

    header(f"Creating output set: {expression}")
    print(f"Index group      : {group_name} ({item.get('atom_count', '?')} atoms)")
    print(f"Contains receptor: {contains_receptor}")
    print(f"Cutdown interval : {cutdown_ps:g} ps")

    structure_output = Path(f"{slug}.pdb")
    trajectory_output = Path(f"traj_{slug}.xtc")

    run(
        gmx_command(
            gmx, "trjconv", "-f", str(full_trajectory),
            "-s", str(reference_tpr), "-n", str(index_file),
            "-o", str(trajectory_output), "-quiet",
        ),
        selections=[group_name],
    )
    run(
        gmx_command(
            gmx, "trjconv", "-f", str(structure_file),
            "-s", str(structure_file), "-n", str(index_file),
            "-o", str(structure_output), "-quiet",
        ),
        selections=[group_name],
    )

    require_file(trajectory_output, f"{expression} trajectory")
    require_file(structure_output, f"{expression} structure")
    records.append((f"{expression} reference PDB", structure_output))
    records.append((f"{expression} extracted XTC", trajectory_output))

    fitted_name, reduced_name = product_filenames(slug, cutdown_ps)

    if contains_receptor:
        fitted_output: Optional[Path] = Path(fitted_name)
        reduced_output = Path(reduced_name)
        print(f"Fitting on {fit_group}; writing {fitted_output}")
        run(
            gmx_command(
                gmx, "trjconv", "-f", str(full_trajectory),
                "-s", str(reference_tpr), "-n", str(index_file),
                "-o", str(fitted_output), "-fit", "rot+trans", "-quiet",
            ),
            selections=[fit_group, group_name],
        )
        require_file(fitted_output, f"Fitted {expression} trajectory")
        downsample(
            gmx, fitted_output, structure_output, reduced_output, cutdown_ps
        )
        records.append((f"{expression} helix-CA-fitted XTC", fitted_output))
    else:
        fitted_output = None
        reduced_output = Path(
            f"traj_{slug}_{format_interval(cutdown_ps)}ps.xtc"
        )
        print(f"{expression} has no receptor atoms; fitting skipped.")
        downsample(
            gmx, trajectory_output, structure_output, reduced_output, cutdown_ps
        )

    records.append((f"{expression} reduced XTC", reduced_output))
    product = OutputProduct(
        expression=expression,
        structure=structure_output,
        trajectory=trajectory_output,
        fitted_trajectory=fitted_output,
        reduced_trajectory=reduced_output,
    )
    return product, records


def output_product_tasks(
    gmx: str,
    full_trajectory: Path,
    reference_tpr: Path,
    structure_file: Path,
    index_file: Path,
    manifest: dict[str, Any],
) -> list[Callable[[], tuple[OutputProduct, list[tuple[str, Path]]]]]:
    """One independent task per TRAJOUT set; they share only read-only inputs
    and write distinct files, so they can run concurrently."""
    fit_group = str(manifest.get("receptor_fit_group", "RECEPTOR_HELICES_CA"))
    return [
        (lambda item=item: _make_output_product(
            gmx, item, full_trajectory, reference_tpr, structure_file,
            index_file, fit_group,
        ))
        for item in manifest.get("outputs", [])
    ]


def remove_files(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            print(f"Removing temporary file: {path}")
            path.unlink()


def run_pipeline(
    config_path: Path,
    index_only: bool = False,
    log_override: Optional[Path] = None,
    no_log: bool = False,
    version: str = "",
) -> None:
    settings, trajectories, output_requests = read_config(config_path)

    if no_log:
        log_path: Optional[Path] = None
    elif log_override is not None:
        log_path = log_override
    else:
        log_path = Path(setting(settings, "LOG_FILE", "modifyGMXtraj.log"))

    with RunLog(log_path) as log:
        log.write_header(version, config_path)
        log.write_input_file(config_path)
        _run(settings, trajectories, output_requests, index_only, log)
        log.write_file_inventory()


def _run(
    settings: dict[str, str],
    trajectories: list[TrajectoryEntry],
    output_requests: list,
    index_only: bool,
    log: RunLog,
) -> None:
    gmx = setting(settings, "GMX", "gmx_mpi")
    check_executable(gmx, "GROMACS executable")

    log_gmx_output = parse_bool(
        setting(settings, "LOG_GMX_OUTPUT", "no"), "LOG_GMX_OUTPUT"
    )
    set_capture(log_gmx_output)

    for entry in trajectories:
        require_file(entry.trajectory, "TRAJIN trajectory")
        require_file(entry.topology, "TRAJIN topology (inferred from XTC name)")

    reference_tpr = Path(
        setting(settings, "REFERENCE_TPR", str(trajectories[0].topology))
    )
    structure_file = Path(setting(settings, "STRUCTURE_FILE", required=True))
    require_file(reference_tpr, "Reference TPR")
    require_file(structure_file, "Structure file")

    # Hard check before anything else runs: STRUCTURE_FILE must agree with the
    # actual production trajectory. Catches e.g. a reference structure that
    # still carries CGenFF lone-pair/dummy atoms the production run does not.
    import mdtraj as _md

    structure_atoms = _md.load(str(structure_file)).n_atoms
    trajectory_atoms = xtc_atom_count(trajectories[0].trajectory)
    if structure_atoms != trajectory_atoms:
        raise ConfigError(
            f"Atom count mismatch: STRUCTURE_FILE ({structure_file}) has "
            f"{structure_atoms} atoms but {trajectories[0].trajectory} has "
            f"{trajectory_atoms}. The index groups built from STRUCTURE_FILE "
            "would silently reference atoms past the end of the trajectory. "
            "This commonly happens when STRUCTURE_FILE predates or postdates "
            "a topology change (e.g. added/removed CGenFF lone-pair/dummy "
            "atoms). Regenerate STRUCTURE_FILE from the actual production "
            "TPR, e.g.:\n"
            f"  echo 0 | {gmx} trjconv -s {reference_tpr} -f {reference_tpr} "
            "-o structure_from_tpr.gro -pbc none"
        )

    remove_initial_ps_override: Optional[float] = None
    if "REMOVE_INITIAL_PS" in settings:
        remove_initial_ps_override = parse_nonnegative_float(
            settings["REMOVE_INITIAL_PS"], "REMOVE_INITIAL_PS"
        )

    # Output names are deliberately generic. They must not encode which
    # input produced them: a run on step6.6_equilibration.xtc previously
    # wrote files called step7_production_*, which both mislabels the
    # provenance and silently overwrites the equilibration results when the
    # real production trajectory is later processed in the same directory.
    combined_output = Path(setting(settings, "COMBINED_OUTPUT", "combined.xtc"))
    full_output = Path(setting(settings, "FULL_OUTPUT", "pbc_fit.xtc"))
    full_cutdown_ps = parse_nonnegative_float(
        setting(settings, "FULL_CUTDOWN_PS", "1000"), "FULL_CUTDOWN_PS"
    )
    full_cutdown_output = Path(
        setting(
            settings, "FULL_CUTDOWN_OUTPUT",
            default_interval_filename(full_output.stem, full_cutdown_ps or 1000),
        )
    )

    index_file = Path(setting(settings, "INDEX_FILE", "gpcr_only.ndx"))
    manifest_file = Path(
        setting(settings, "INDEX_MANIFEST", "trajectory_groups.json")
    )

    generate_mindist = parse_bool(
        setting(settings, "GENERATE_MINDIST_SCRIPT", "yes"),
        "GENERATE_MINDIST_SCRIPT",
    )
    mindist_cutdown_ps = parse_positive_float(
        setting(settings, "MINDIST_CUTDOWN_PS", "1000"), "MINDIST_CUTDOWN_PS"
    )
    mindist_script = Path(setting(settings, "MINDIST_SCRIPT", "run_mindist.sh"))
    slurm_account = setting(settings, "SLURM_ACCOUNT", "naiss2025-3-21")
    slurm_partition = setting(settings, "SLURM_PARTITION", "shared")
    slurm_mem = setting(settings, "SLURM_MEM", "8G")

    remove_temporary = parse_bool(
        setting(settings, "REMOVE_TEMPORARY", "yes"), "REMOVE_TEMPORARY"
    )

    # ---- speed settings (all default to the original sequential behaviour)
    early_cutdown_ps = parse_nonnegative_float(
        setting(settings, "EARLY_CUTDOWN_PS", "0"), "EARLY_CUTDOWN_PS"
    )
    if early_cutdown_ps > 0:
        for request in output_requests:
            check_multiple_of(
                request.cutdown_ps, early_cutdown_ps, f"TRAJOUT {request.expression}"
            )
        if full_cutdown_ps > 0:
            check_multiple_of(full_cutdown_ps, early_cutdown_ps, "FULL_CUTDOWN_PS")
    parallel_jobs = parse_positive_int(
        setting(settings, "PARALLEL_JOBS", "1"), "PARALLEL_JOBS"
    )
    temporary_dir: Optional[Path] = None
    if "TEMPORARY_DIR" in settings:
        temporary_dir = Path(settings["TEMPORARY_DIR"])
        temporary_dir.mkdir(parents=True, exist_ok=True)

    # ---- resolved parameter block for the log ----------------------------
    parameters: list[tuple[str, object]] = [
        ("GMX", gmx),
        ("STRUCTURE_FILE", f"{structure_file}  ({structure_atoms} atoms)"),
        ("REFERENCE_TPR", reference_tpr),
        ("First TRAJIN atom count", trajectory_atoms),
        ("MAIN_RESIDUES", settings.get("MAIN_RESIDUES", "(required)")),
        ("LIGANDS", settings.get("LIGANDS", "NONE")),
        ("GPROTEIN", settings.get("GPROTEIN", "NONE")),
        ("LIPIDS", settings.get("LIPIDS", "NONE")),
        ("MIN_HELIX_LENGTH", settings.get("MIN_HELIX_LENGTH", "3")),
        ("EXTRA_VIRTUAL_NAMES", settings.get("EXTRA_VIRTUAL_NAMES", "NONE")),
        (
            "REMOVE_INITIAL_PS",
            f"{remove_initial_ps_override:g} (explicit)"
            if remove_initial_ps_override is not None
            else "auto-detected per segment from .mdp",
        ),
        ("COMBINED_OUTPUT", combined_output),
        ("FULL_OUTPUT", full_output),
        (
            "FULL_CUTDOWN_PS",
            f"{full_cutdown_ps:g}" if full_cutdown_ps else "0 (disabled)",
        ),
        ("INDEX_FILE", index_file),
        ("INDEX_MANIFEST", manifest_file),
        ("PBC chain groups", "gmx inherent defaults (no -n): 0=System, 1=Protein"),
        ("TRAJOUT index groups", f"custom {index_file}"),
        ("GENERATE_MINDIST_SCRIPT", generate_mindist),
        ("MINDIST_CUTDOWN_PS", f"{mindist_cutdown_ps:g}"),
        ("MINDIST_SCRIPT", mindist_script),
        ("SLURM_ACCOUNT", slurm_account),
        ("REMOVE_TEMPORARY", remove_temporary),
        ("LOG_GMX_OUTPUT", log_gmx_output),
        (
            "EARLY_CUTDOWN_PS",
            f"{early_cutdown_ps:g} (FULL_OUTPUT and extracted XTCs at this interval)"
            if early_cutdown_ps > 0 else "0 (disabled; every frame kept)",
        ),
        ("PARALLEL_JOBS", parallel_jobs),
        ("TEMPORARY_DIR", temporary_dir if temporary_dir is not None else "(next to inputs / working dir)"),
        ("SLURM_PARTITION (mindist)", slurm_partition),
        ("SLURM_MEM (mindist)", slurm_mem),
    ]
    for number, entry in enumerate(trajectories, start=1):
        parameters.append(
            (
                f"TRAJIN {number}",
                f"{entry.trajectory} -> {entry.topology} | {entry.length_ns:g} ns",
            )
        )
    for number, request in enumerate(output_requests, start=1):
        parameters.append(
            (
                f"TRAJOUT {number}",
                f"{request.expression} | {request.cutdown_ps:g} ps",
            )
        )
    log.write_parameters(parameters)

    # ---- index generation -------------------------------------------------
    header("Generating structural and requested output groups")
    manifest = build_index(
        settings, output_requests, structure_file, index_file, manifest_file
    )
    log.record("Index file", index_file)
    log.record("Index manifest", manifest_file)

    if index_only:
        print("\n--index-only requested; stopping after index generation.")
        return

    # ---- trajectory processing -------------------------------------------
    concat_inputs, segment_temporary = prepare_sequential_segments(
        gmx, trajectories, remove_initial_ps_override, temporary_dir
    )
    concatenate(gmx, concat_inputs, combined_output, log)

    pbc_temporary = process_pbc(
        gmx, combined_output, reference_tpr, full_output, log,
        early_cutdown_ps, temporary_dir,
    )

    # The reduced full trajectory and every TRAJOUT set only read FULL_OUTPUT,
    # so they are independent and may run concurrently (PARALLEL_JOBS).
    def reduced_full() -> tuple[None, list[tuple[str, Path]]]:
        header("Creating reduced full trajectory")
        downsample(
            gmx, full_output, reference_tpr, full_cutdown_output,
            full_cutdown_ps, PBC_OUTPUT_GROUP,
        )
        return None, [("Reduced full trajectory", full_cutdown_output)]

    tasks: list[Callable[[], tuple[Any, list[tuple[str, Path]]]]] = []
    if full_cutdown_ps > 0:
        tasks.append(reduced_full)
    if output_requests:
        tasks.extend(
            output_product_tasks(
                gmx, full_output, reference_tpr, structure_file, index_file,
                manifest,
            )
        )
    products: list[OutputProduct] = []
    for result, records in run_tasks(tasks, parallel_jobs):
        for description, path in records:
            log.record(description, path)
        if result is not None:
            products.append(result)

    if generate_mindist:
        write_mindist_script(
            gmx, trajectories, reference_tpr, mindist_cutdown_ps,
            mindist_script, account=slurm_account,
            partition=slurm_partition, mem=slurm_mem,
        )
        log.record("Mindist submission script", mindist_script)

    if remove_temporary:
        header("Removing temporary files")
        remove_files(segment_temporary + pbc_temporary)

    header("Completed")
    print(f"Combined trajectory: {combined_output}")
    print(f"Full PBC-fitted trajectory: {full_output}")
    if full_cutdown_ps > 0:
        print(f"Reduced full trajectory: {full_cutdown_output}")
    for product in products:
        print(f"\n{product.expression}")
        print(f"  PDB: {product.structure}")
        print(f"  extracted XTC: {product.trajectory}")
        if product.fitted_trajectory is not None:
            print(f"  helix-CA-fitted XTC: {product.fitted_trajectory}")
        else:
            print("  helix-CA-fitted XTC: skipped; receptor absent")
        print(f"  reduced XTC: {product.reduced_trajectory}")
    if generate_mindist:
        print(f"\nMindist script: {mindist_script} (submit with sbatch)")
