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
from .gmx import check_executable, gmx_command, header, run, tpr_atom_count
from .indexer import build_index
from .mindist import write_mindist_script

# Named index groups used by the PBC chain. Selecting by name rather than by
# gmx's default numbering removes seven config keys and the assumption that
# group 0/1 are System/Protein.
PBC_WHOLE_GROUP = "SYSTEM"
PBC_CLUSTER_PIVOT_GROUP = "PROTEIN"
PBC_OUTPUT_GROUP = "SYSTEM"
PBC_CENTER_GROUP = "PROTEIN"
PBC_FIT_GROUP = "PROTEIN"


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
            concat_inputs.append(entry.trajectory)
            continue

        if remove_initial_ps_override is not None:
            trim_begin = remove_initial_ps_override
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


def concatenate(gmx: str, inputs: list[Path], output: Path) -> None:
    header("Concatenating production trajectories")
    for number, path in enumerate(inputs, start=1):
        print(f"{number:3d}: {path}")

    if len(inputs) == 1:
        shutil.copy2(inputs[0], output)
    else:
        command = gmx_command(gmx, "trjcat", "-f")
        command.extend(str(path) for path in inputs)
        command.extend(["-o", str(output), "-cat"])
        run(command)


def process_pbc(
    gmx: str,
    combined: Path,
    reference_tpr: Path,
    index_file: Path,
    output: Path,
) -> list[Path]:
    header("Applying PBC correction, centring and whole-system fitting")

    whole = Path("temporary_pbc_whole.xtc")
    clustered = Path("temporary_pbc_cluster.xtc")
    molecule = Path("temporary_pbc_molecule.xtc")

    run(
        gmx_command(
            gmx, "trjconv", "-f", str(combined), "-s", str(reference_tpr),
            "-n", str(index_file), "-o", str(whole), "-pbc", "whole",
        ),
        selections=[PBC_WHOLE_GROUP],
    )
    run(
        gmx_command(
            gmx, "trjconv", "-f", str(whole), "-s", str(reference_tpr),
            "-n", str(index_file), "-o", str(clustered), "-pbc", "cluster",
        ),
        selections=[PBC_CLUSTER_PIVOT_GROUP, PBC_OUTPUT_GROUP],
    )
    run(
        gmx_command(
            gmx, "trjconv", "-f", str(clustered), "-s", str(reference_tpr),
            "-n", str(index_file), "-o", str(molecule), "-pbc", "mol",
        ),
        selections=[PBC_OUTPUT_GROUP],
    )
    run(
        gmx_command(
            gmx, "trjconv", "-f", str(molecule), "-s", str(reference_tpr),
            "-n", str(index_file), "-o", str(output),
            "-center", "-fit", "rot+trans",
        ),
        selections=[PBC_CENTER_GROUP, PBC_FIT_GROUP, PBC_OUTPUT_GROUP],
    )
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


def create_output_products(
    gmx: str,
    full_trajectory: Path,
    reference_tpr: Path,
    structure_file: Path,
    index_file: Path,
    manifest: dict[str, Any],
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

        if contains_receptor:
            fitted_output = Path(f"traj_{slug}_fit.xtc")
            reduced_output = Path(
                f"traj_{slug}_fit_{format_interval(cutdown_ps)}ps.xtc"
            )
            print(f"{expression} contains the receptor; fitting with {fit_group}.")
            run(
                gmx_command(
                    gmx, "trjconv", "-f", str(full_trajectory),
                    "-s", str(reference_tpr), "-n", str(index_file),
                    "-o", str(fitted_output), "-fit", "rot+trans", "-quiet",
                ),
                selections=[fit_group, group_name],
            )
            downsample(
                gmx, fitted_output, structure_output, reduced_output, cutdown_ps
            )
        else:
            fitted_output = None
            reduced_output = Path(
                f"traj_{slug}_{format_interval(cutdown_ps)}ps.xtc"
            )
            print(f"{expression} does not contain the receptor; fitting skipped.")
            downsample(
                gmx, trajectory_output, structure_output, reduced_output, cutdown_ps
            )

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


def run_pipeline(config_path: Path, index_only: bool = False) -> None:
    settings, trajectories, output_requests = read_config(config_path)

    gmx = setting(settings, "GMX", "gmx_mpi")
    check_executable(gmx, "GROMACS executable")

    for entry in trajectories:
        require_file(entry.trajectory, "TRAJIN trajectory")
        require_file(entry.topology, "TRAJIN topology (inferred from XTC name)")

    reference_tpr = Path(
        setting(settings, "REFERENCE_TPR", str(trajectories[0].topology))
    )
    structure_file = Path(setting(settings, "STRUCTURE_FILE", required=True))
    require_file(reference_tpr, "Reference TPR")
    require_file(structure_file, "Structure file")

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

    header("Workflow configuration")
    print(f"Configuration: {config_path}")
    print(f"Structure: {structure_file}")
    print(f"Reference TPR: {reference_tpr}")
    print("TRAJIN (TPR inferred from XTC name):")
    for number, entry in enumerate(trajectories, start=1):
        print(
            f"  {number:3d}: {entry.trajectory} -> {entry.topology} | "
            f"{entry.length_ns:g} ns"
        )
    print("TRAJOUT:")
    for number, request in enumerate(output_requests, start=1):
        print(f"  {number:3d}: {request.expression} | {request.cutdown_ps:g} ps")
    if not output_requests:
        print("  none")

    header("Generating structural and requested output groups")
    manifest = build_index(
        settings,
        output_requests,
        structure_file,
        index_file,
        manifest_file,
        expected_atoms=tpr_atom_count(gmx, reference_tpr),
    )

    if index_only:
        print("\n--index-only requested; stopping after index generation.")
        return

    concat_inputs, segment_temporary = prepare_sequential_segments(
        gmx, trajectories, remove_initial_ps_override
    )
    concatenate(gmx, concat_inputs, combined_output)

    pbc_temporary = process_pbc(
        gmx, combined_output, reference_tpr, index_file, full_output
    )
    if full_cutdown_ps > 0:
        header("Creating reduced full trajectory")
        downsample(
            gmx, full_output, reference_tpr, full_cutdown_output,
            full_cutdown_ps, PBC_OUTPUT_GROUP, index_file=index_file,
        )

    products = create_output_products(
        gmx, full_output, reference_tpr, structure_file, index_file, manifest
    ) if output_requests else []

    if generate_mindist:
        write_mindist_script(
            gmx, trajectories, reference_tpr, index_file,
            mindist_cutdown_ps, mindist_script, account=slurm_account,
        )

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
        print(f"  reduced XTC: {product.reduced_trajectory}")
    if generate_mindist:
        print(f"\nMindist script: {mindist_script} (submit with sbatch)")
