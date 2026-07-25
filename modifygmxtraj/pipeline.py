"""Trajectory concatenation, PBC correction and per-TRAJOUT extraction."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .config import (
    ConfigError,
    TrajectoryEntry,
    find_mdp_file,
    format_interval,
    parse_bool,
    parse_nonnegative_float,
    parse_positive_float,
    read_config,
    remove_initial_ps_from_mdp,
    require_file,
    setting,
    trajectory_start_times,
)
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


def temporary_path(source: Path, label: str) -> Path:
    return source.with_name(f"{source.stem}_{label}{source.suffix}")


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

        trimmed = temporary_path(entry.trajectory, "trimmed")
        shifted = temporary_path(entry.trajectory, "shifted")

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
    gmx: str, combined: Path, reference_tpr: Path, output: Path, log: RunLog
) -> list[Path]:
    """PBC correction on the WHOLE trajectory using gmx's inherent default
    groups. No -n here by design; see the comment above PBC_WHOLE_GROUP."""
    header("Applying PBC correction, centring and whole-system fitting")
    print("Using GROMACS inherent default groups (no -n): 0 = System, 1 = Protein")
    print("  step 1: -pbc whole      output System")
    print("  step 2: -pbc cluster    pivot Protein, output System")
    print("  step 3: -pbc mol        output System")
    print("  step 4: -center Protein, -fit rot+trans on Protein, output System")

    whole = Path("temporary_pbc_whole.xtc")
    clustered = Path("temporary_pbc_cluster.xtc")
    molecule = Path("temporary_pbc_molecule.xtc")

    run(
        gmx_command(
            gmx, "trjconv", "-f", str(combined), "-s", str(reference_tpr),
            "-o", str(whole), "-pbc", "whole",
        ),
        selections=[PBC_WHOLE_GROUP],
    )
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


def create_output_products(
    gmx: str,
    full_trajectory: Path,
    reference_tpr: Path,
    structure_file: Path,
    index_file: Path,
    manifest: dict[str, Any],
    log: RunLog,
) -> list[OutputProduct]:
    outputs = manifest.get("outputs", [])
    if not outputs:
        return []

    fit_group = str(manifest.get("receptor_fit_group", "RECEPTOR_HELICES_CA"))
    products: list[OutputProduct] = []

    for item in outputs:
        expression = str(item["expression"])
        group_name = str(item["group"])
        slug = str(item["slug"])
        contains_receptor = bool(item["contains_receptor"])
        cutdown_ps = float(item["cutdown_ps"])

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
        log.record(f"{expression} reference PDB", structure_output)
        log.record(f"{expression} extracted XTC", trajectory_output)

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
            log.record(f"{expression} helix-CA-fitted XTC", fitted_output)
        else:
            fitted_output = None
            reduced_output = Path(
                f"traj_{slug}_{format_interval(cutdown_ps)}ps.xtc"
            )
            print(f"{expression} has no receptor atoms; fitting skipped.")
            downsample(
                gmx, trajectory_output, structure_output, reduced_output, cutdown_ps
            )

        log.record(f"{expression} reduced XTC", reduced_output)

        products.append(
            OutputProduct(
                expression=expression,
                structure=structure_output,
                trajectory=trajectory_output,
                fitted_trajectory=fitted_output,
                reduced_trajectory=reduced_output,
            )
        )
    return products


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

    combined_output = Path(
        setting(settings, "COMBINED_OUTPUT", "step7_production_combined.xtc")
    )
    full_output = Path(
        setting(settings, "FULL_OUTPUT", "step7_production_pbc_fit.xtc")
    )
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

    remove_temporary = parse_bool(
        setting(settings, "REMOVE_TEMPORARY", "yes"), "REMOVE_TEMPORARY"
    )

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
        gmx, trajectories, remove_initial_ps_override
    )
    concatenate(gmx, concat_inputs, combined_output, log)

    pbc_temporary = process_pbc(
        gmx, combined_output, reference_tpr, full_output, log
    )
    if full_cutdown_ps > 0:
        header("Creating reduced full trajectory")
        downsample(
            gmx, full_output, reference_tpr, full_cutdown_output,
            full_cutdown_ps, PBC_OUTPUT_GROUP,
        )
        log.record("Reduced full trajectory", full_cutdown_output)

    products = (
        create_output_products(
            gmx, full_output, reference_tpr, structure_file, index_file,
            manifest, log,
        )
        if output_requests
        else []
    )

    if generate_mindist:
        write_mindist_script(
            gmx, trajectories, reference_tpr, mindist_cutdown_ps,
            mindist_script, account=slurm_account,
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
