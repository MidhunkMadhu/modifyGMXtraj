"""Index-group generation (the former make_ndx.py), as an importable helper.

Residue ranges are positional, 1-based ranges in the order residues occur in
the structure. MAIN is RECEPTOR plus declared ligands. COMPLEX is MAIN plus the
optional explicitly declared GPROTEIN range.

Receptor helices are identified from the first frame of STRUCTURE_FILE only.
DSSP is never calculated for a trajectory. RECEPTOR_HELICES_CA is created from
the reference structure and used for receptor-containing trajectory fits.

This module is called directly by the pipeline (no subprocess, no PYTHON
setting) and is also reachable from the CLI via --index-only.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

import mdtraj as md

from .config import (
    ConfigError,
    OutputRequest,
    output_slug,
    parse_name_list,
    parse_optional_range,
    parse_positive_int,
    setting,
)

AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
    "HSD", "HSE", "HSP", "HID", "HIE", "HIP",
    "ASH", "GLH", "LYN", "CYX", "CYM", "SEC", "MSE",
}

DEFAULT_LIPIDS = {
    "CHL", "CHL1", "CHOL", "CLR", "ERG", "SITO",
    "POPC", "DOPC", "DPPC", "DMPC", "DLPC", "DSPC", "PLPC", "SOPC",
    "SAPC", "MYPC", "PMPC", "PAPC", "SDPC", "DEPC", "DAPC",
    "POPE", "DOPE", "DPPE", "DMPE", "DLPE", "DSPE", "PLPE", "SAPE",
    "PAPE", "SOPE", "DEPE",
    "POPG", "DOPG", "DPPG", "DMPG", "DLPG", "DSPG", "SOPG",
    "POPS", "DOPS", "DPPS", "DMPS", "DLPS", "DSPS", "PLPS", "SAPS",
    "POPA", "DOPA", "DPPA", "DMPA", "DLPA", "DSPA", "PLPA",
    "POPI", "DOPI", "DPPI", "SAPI", "SOPI",
    "POPI13", "POPI14", "POPI15", "POPI24", "POPI25",
    "POPI33", "POPI34", "POPI35",
    "PIP", "PIP2", "PIP3", "PI13", "PI14", "PI15", "PI24", "PI25",
    "PI33", "PI34", "PI35",
    "PSM", "SSM", "PVSM", "PGSM", "DPSM", "DBSM",
    "CER", "CER160", "CER180", "CER181", "CER200", "CER220", "CER240",
    "LPC", "LPE", "LPG", "LPS", "LPA",
    "TRIO", "TAG", "DAG", "MAG", "TOCL", "CDL2",
    "GM1", "GM3", "GD1A", "GD1B", "GT1B",
}

WATER_IONS = {
    "SOL", "HOH", "WAT", "TIP3", "TIP3P", "TIP4P", "TIP5P", "SPC", "SPCE",
    "NA", "CL", "K", "MG", "ZN", "CAL", "CA", "CS", "LI", "RB",
    "SOD", "CLA", "POT", "NA+", "CL-", "MG2", "ZN2", "CA2",
}

VIRTUAL_SITE_RE = re.compile(r"^(LP\d*|EP\d*|MW|DUM\d*)$", re.IGNORECASE)

COMPONENT_ALIASES = {
    "GPCR": "RECEPTOR",
    "G-PROTEIN": "GPROTEIN",
    "G_PROTEIN": "GPROTEIN",
    "ALL_LIGANDS": "LIGANDS",
    "WATER": "SOLVENT",
    "WATERS": "SOLVENT",
    "WATER_IONS": "SOLVENT",
}

RECEPTOR_FIT_GROUP = "RECEPTOR_HELICES_CA"


def positional_indices(start: int, end: int) -> list[int]:
    return list(range(start - 1, end))


def is_virtual_site(atom_name: str, extra_virtual_names: set[str]) -> bool:
    return (
        bool(VIRTUAL_SITE_RE.match(atom_name))
        or atom_name.upper() in extra_virtual_names
    )


def atoms_for_residues(
    topology: md.Topology,
    residue_indices: Iterable[int],
    extra_virtual_names: set[str],
) -> list[int]:
    residue_set = set(residue_indices)
    return [
        atom.index
        for atom in topology.atoms
        if atom.residue.index in residue_set
        and not is_virtual_site(atom.name, extra_virtual_names)
    ]


def ca_for_residues(
    topology: md.Topology, residue_indices: Iterable[int]
) -> list[int]:
    residue_set = set(residue_indices)
    return [
        atom.index
        for atom in topology.atoms
        if atom.residue.index in residue_set and atom.name.upper() == "CA"
    ]


def format_group(name: str, indices: Iterable[int]) -> tuple[str, int]:
    unique = sorted(set(indices))
    lines = [f"[ {name} ]\n"]
    for start in range(0, len(unique), 15):
        lines.append(
            " ".join(str(index + 1) for index in unique[start:start + 15]) + "\n"
        )
    lines.append("\n")
    return "".join(lines), len(unique)


def residue_is_lipid_like(residue) -> bool:
    """Conservative fallback for an unlisted lipid species."""
    carbon_count = heavy_count = phosphorus_count = 0
    for atom in residue.atoms:
        symbol = atom.element.symbol.upper() if atom.element is not None else ""
        if symbol != "H":
            heavy_count += 1
        if symbol == "C":
            carbon_count += 1
        elif symbol == "P":
            phosphorus_count += 1
    return (carbon_count >= 12 and heavy_count >= 15) or (
        phosphorus_count >= 1 and carbon_count >= 8 and heavy_count >= 12
    )


def detect_receptor_helices(
    trajectory: md.Trajectory,
    receptor_residues: list[int],
    minimum_length: int,
) -> list[dict[str, int]]:
    atom_indices: list[int] = []
    for residue_index in receptor_residues:
        atom_indices.extend(
            atom.index for atom in trajectory.topology.residue(residue_index).atoms
        )

    sub_trajectory = trajectory.atom_slice(atom_indices)
    dssp = md.compute_dssp(sub_trajectory, simplified=True)[0]
    if len(dssp) != len(receptor_residues):
        raise RuntimeError("DSSP residue count does not match receptor selection")

    segments: list[tuple[int, int]] = []
    start: Optional[int] = None
    for position, code in enumerate(dssp):
        if code == "H" and start is None:
            start = position
        elif code != "H" and start is not None:
            if position - start >= minimum_length:
                segments.append((start, position - 1))
            start = None
    if start is not None and len(dssp) - start >= minimum_length:
        segments.append((start, len(dssp) - 1))

    helices: list[dict[str, int]] = []
    for start_position, end_position in segments:
        full_start = receptor_residues[start_position]
        full_end = receptor_residues[end_position]
        helices.append(
            {
                "start_index": full_start,
                "end_index": full_end,
                "start_resseq": trajectory.topology.residue(full_start).resSeq,
                "end_resseq": trajectory.topology.residue(full_end).resSeq,
                "length": end_position - start_position + 1,
            }
        )
    return helices


def resolve_component_name(name: str) -> str:
    return COMPONENT_ALIASES.get(name.upper(), name.upper())


def build_index(
    settings: dict[str, str],
    output_requests: list[OutputRequest],
    structure: Path,
    index_path: Path,
    manifest_path: Path,
    expected_atoms: Optional[int] = None,
) -> dict:
    """Write the .ndx and the TRAJOUT manifest; return the manifest dict."""
    if not structure.is_file():
        raise ConfigError(f"Structure file not found: {structure}")

    receptor_count = parse_positive_int(
        setting(settings, "MAIN_RESIDUES", required=True), "MAIN_RESIDUES"
    )
    requested_ligands = parse_name_list(settings.get("LIGANDS"))
    declared_lipids = parse_name_list(settings.get("LIPIDS"))
    gprotein_range = parse_optional_range(settings.get("GPROTEIN"), "GPROTEIN")
    extra_virtual_names = set(parse_name_list(settings.get("EXTRA_VIRTUAL_NAMES")))
    minimum_helix_length = parse_positive_int(
        setting(settings, "MIN_HELIX_LENGTH", "3"), "MIN_HELIX_LENGTH"
    )

    loaded_structure = md.load(str(structure))
    if loaded_structure.n_frames < 1:
        raise ConfigError(f"No coordinate frame was found in {structure}")

    trajectory = loaded_structure[0]
    topology = trajectory.topology
    residue_count = topology.n_residues

    if expected_atoms is not None and topology.n_atoms != expected_atoms:
        raise ConfigError(
            f"STRUCTURE_FILE has {topology.n_atoms} atoms but the reference TPR "
            f"has {expected_atoms}. The index would be silently wrong."
        )

    if receptor_count > residue_count:
        raise ConfigError(
            f"MAIN_RESIDUES={receptor_count} exceeds the {residue_count} "
            f"residues in {structure}"
        )

    receptor_residues = positional_indices(1, receptor_count)
    receptor_set = set(receptor_residues)

    if gprotein_range is None:
        gprotein_residues: list[int] = []
    else:
        start, end = gprotein_range
        if end > residue_count:
            raise ConfigError(
                f"GPROTEIN={start} to {end} exceeds the {residue_count} residues "
                f"in {structure}"
            )
        gprotein_residues = positional_indices(start, end)
        overlap = receptor_set.intersection(gprotein_residues)
        if overlap:
            raise ConfigError(
                "GPROTEIN overlaps the receptor at positional residues "
                f"{min(overlap) + 1}-{max(overlap) + 1}"
            )
    gprotein_set = set(gprotein_residues)

    ligand_residues_by_name: dict[str, list[int]] = defaultdict(list)
    requested_ligand_set = set(requested_ligands)
    for residue in topology.residues:
        if residue.name.upper() in requested_ligand_set:
            ligand_residues_by_name[residue.name.upper()].append(residue.index)

    ligand_residues = sorted(
        {
            index
            for indices in ligand_residues_by_name.values()
            for index in indices
            if index not in receptor_set and index not in gprotein_set
        }
    )
    ligand_set = set(ligand_residues)

    for ligand_name in requested_ligands:
        if not [
            i for i in ligand_residues_by_name.get(ligand_name, []) if i in ligand_set
        ]:
            print(f"WARNING: declared ligand {ligand_name} was not found")

    declared_lipid_set = set(declared_lipids)
    lipid_catalogue = set(DEFAULT_LIPIDS) | declared_lipid_set
    detected_lipids_by_name: dict[str, list[int]] = defaultdict(list)
    heuristic_lipid_names: set[str] = set()

    for residue in topology.residues:
        name = residue.name.upper()
        if (
            residue.index in receptor_set
            or residue.index in gprotein_set
            or residue.index in ligand_set
            or name in WATER_IONS
            or name in AMINO_ACIDS
        ):
            continue
        if name in lipid_catalogue:
            detected_lipids_by_name[name].append(residue.index)
        elif residue_is_lipid_like(residue):
            detected_lipids_by_name[name].append(residue.index)
            heuristic_lipid_names.add(name)

    detected_lipid_names = set(detected_lipids_by_name)
    omitted_lipids = sorted(detected_lipid_names - declared_lipid_set)
    if omitted_lipids:
        print(
            "WARNING: lipid species detected but not listed in LIPIDS: "
            + ", ".join(omitted_lipids)
        )
        print("They will still be incorporated into the combined LIPIDS group.")

    if heuristic_lipid_names:
        print(
            "WARNING: these unlisted species were classified as lipids by their "
            "large carbon-rich residue composition: "
            + ", ".join(sorted(heuristic_lipid_names))
        )
        print("Verify these if any is actually an undeclared ligand/cofactor.")

    absent_lipids = sorted(declared_lipid_set - detected_lipid_names)
    if absent_lipids:
        print(
            "WARNING: species listed in LIPIDS but absent from the structure: "
            + ", ".join(absent_lipids)
        )

    lipid_residues = sorted(
        index for indices in detected_lipids_by_name.values() for index in indices
    )
    lipid_set = set(lipid_residues)

    solvent_residues = sorted(
        residue.index
        for residue in topology.residues
        if residue.name.upper() in WATER_IONS
    )
    solvent_set = set(solvent_residues)

    classified = receptor_set | gprotein_set | ligand_set | lipid_set | solvent_set
    unclassified = [r for r in topology.residues if r.index not in classified]
    if unclassified:
        counts = Counter(r.name.upper() for r in unclassified)
        print(
            "WARNING: unclassified residue species remain: "
            + ", ".join(f"{n}x{c}" for n, c in sorted(counts.items()))
        )

    main_residues = sorted(receptor_set | ligand_set)
    complex_residues = sorted(set(main_residues) | gprotein_set)
    protein_residues = sorted(receptor_set | gprotein_set)

    component_residues: dict[str, set[int]] = {
        "SYSTEM": set(range(residue_count)),
        "RECEPTOR": receptor_set,
        "MAIN": set(main_residues),
        "COMPLEX": set(complex_residues),
        "PROTEIN": set(protein_residues),
        "GPROTEIN": gprotein_set,
        "LIGANDS": ligand_set,
        "LIPIDS": lipid_set,
        "SOLVENT": solvent_set,
    }
    for ligand_name in requested_ligands:
        component_residues[ligand_name] = {
            i for i in ligand_residues_by_name.get(ligand_name, []) if i in ligand_set
        }
    for lipid_name, indices in detected_lipids_by_name.items():
        component_residues[lipid_name] = set(indices)

    resolved_outputs: list[dict[str, object]] = []
    for request in output_requests:
        tokens = [
            resolve_component_name(t) for t in request.expression.split("+")
        ]
        selected: set[int] = set()
        empty_components: list[str] = []
        for token in tokens:
            if token not in component_residues:
                available = ", ".join(sorted(component_residues))
                raise ConfigError(
                    f"Unknown TRAJOUT component '{token}' in {request.expression}. "
                    f"Available components: {available}"
                )
            component = component_residues[token]
            if not component:
                empty_components.append(token)
            selected.update(component)

        if empty_components:
            print(
                f"WARNING: {request.expression} contains empty component(s): "
                + ", ".join(empty_components)
            )
        if not selected:
            raise ConfigError(
                f"TRAJOUT expression {request.expression} selects no residues"
            )

        slug = output_slug(request.expression)
        resolved_outputs.append(
            {
                "expression": request.expression,
                "group": "OUT_" + slug.upper(),
                "slug": slug,
                "cutdown_ps": request.cutdown_ps,
                "contains_receptor": bool(selected & receptor_set),
                "residues": sorted(selected),
            }
        )

    # Warn when two different expressions resolve to identical atom sets, e.g.
    # MAIN+LIPIDS and MAIN+GPROTEIN+LIPIDS when GPROTEIN = NONE.
    by_atoms: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for item in resolved_outputs:
        by_atoms[tuple(item["residues"])].append(str(item["expression"]))
    for expressions in by_atoms.values():
        if len(expressions) > 1:
            print(
                "WARNING: these TRAJOUT expressions select identical residues "
                "and will produce duplicate files: " + ", ".join(expressions)
            )

    try:
        receptor_helices = detect_receptor_helices(
            trajectory, receptor_residues, minimum_helix_length
        )
    except Exception as exc:
        raise ConfigError(
            f"Receptor helix detection from STRUCTURE_FILE failed: {exc}"
        ) from exc

    if not receptor_helices:
        raise ConfigError(
            "No receptor helices were detected in STRUCTURE_FILE; "
            f"{RECEPTOR_FIT_GROUP} cannot be created"
        )

    receptor_helix_residues: list[int] = []
    for helix in receptor_helices:
        receptor_helix_residues.extend(
            range(helix["start_index"], helix["end_index"] + 1)
        )

    virtual_atoms = [
        atom for atom in topology.atoms
        if is_virtual_site(atom.name, extra_virtual_names)
    ]
    if virtual_atoms:
        counts = Counter(
            (a.residue.name, a.residue.resSeq) for a in virtual_atoms
        )
        print(f"Excluding {len(virtual_atoms)} virtual/lone-pair atom(s):")
        for (resname, resseq), count in sorted(counts.items()):
            print(f"  {resname}{resseq}: {count}")

    print(f"RECEPTOR: {len(receptor_residues)} residues")
    print(f"MAIN: {len(main_residues)} residues")
    print(f"GPROTEIN: {len(gprotein_residues)} residues")
    print(f"COMPLEX: {len(complex_residues)} residues")
    print(f"LIPIDS: {len(lipid_residues)} residues")
    print(f"RECEPTOR HELICES: {len(receptor_helices)} segments")

    group_summary: list[tuple[str, int]] = []
    output_atom_counts: dict[str, int] = {}

    with index_path.open("w", encoding="utf-8") as handle:
        def add_group(name: str, atom_indices: Iterable[int]) -> int:
            text, count = format_group(name, atom_indices)
            if count == 0:
                return 0
            handle.write(text)
            group_summary.append((name, count))
            return count

        def residue_atoms(residues: Iterable[int]) -> list[int]:
            return atoms_for_residues(topology, residues, extra_virtual_names)

        # SYSTEM and PROTEIN must come first: the PBC chain selects them by name.
        add_group("SYSTEM", residue_atoms(range(residue_count)))
        add_group("PROTEIN", residue_atoms(protein_residues))
        add_group("RECEPTOR", residue_atoms(receptor_residues))
        add_group("RECEPTOR_CA", ca_for_residues(topology, receptor_residues))
        add_group("MAIN", residue_atoms(main_residues))
        add_group("MAIN_CA", ca_for_residues(topology, receptor_residues))
        add_group("COMPLEX", residue_atoms(complex_residues))
        add_group("COMPLEX_CA", ca_for_residues(topology, protein_residues))

        if gprotein_residues:
            add_group("GPROTEIN", residue_atoms(gprotein_residues))
            add_group("GPROTEIN_CA", ca_for_residues(topology, gprotein_residues))
        if ligand_residues:
            add_group("LIGANDS", residue_atoms(ligand_residues))
        if lipid_residues:
            add_group("LIPIDS", residue_atoms(lipid_residues))
        if solvent_residues:
            add_group("SOLVENT", residue_atoms(solvent_residues))

        for ligand_name in requested_ligands:
            residues = component_residues.get(ligand_name, set())
            if residues:
                add_group(ligand_name, residue_atoms(residues))
        for lipid_name, residues in sorted(detected_lipids_by_name.items()):
            add_group(lipid_name, residue_atoms(residues))

        add_group("RECEPTOR_HELICES", residue_atoms(receptor_helix_residues))
        add_group(
            RECEPTOR_FIT_GROUP,
            ca_for_residues(topology, receptor_helix_residues),
        )
        for number, helix in enumerate(receptor_helices, start=1):
            residues = list(range(helix["start_index"], helix["end_index"] + 1))
            add_group(f"RECEPTOR_HELIX_{number}", residue_atoms(residues))
            add_group(
                f"RECEPTOR_HELIX_{number}_CA",
                ca_for_residues(topology, residues),
            )

        for item in resolved_outputs:
            output_atom_counts[str(item["group"])] = add_group(
                str(item["group"]), residue_atoms(item["residues"])
            )

    manifest_data = {
        "index_file": str(index_path),
        "structure_file": str(structure),
        "receptor_fit_group": RECEPTOR_FIT_GROUP,
        "outputs": [
            {
                "expression": item["expression"],
                "group": item["group"],
                "slug": item["slug"],
                "cutdown_ps": item["cutdown_ps"],
                "contains_receptor": item["contains_receptor"],
                "atom_count": output_atom_counts[str(item["group"])],
            }
            for item in resolved_outputs
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nWrote index file: {index_path}")
    print(f"Wrote output manifest: {manifest_path}")
    print("Group order:")
    for number, (name, atom_count) in enumerate(group_summary):
        print(f"  {number:3d}: {name:<30s} {atom_count} atoms")

    return manifest_data
