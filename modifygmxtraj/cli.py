"""Command-line interface for modifyGMXtraj."""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from . import __version__
from .config import CommandError, ConfigError
from .gmx import set_dry_run

EXAMPLE_INPUT = """\
# ---------------------------------------------------------------------------
# modifyGMXtraj input file
# ---------------------------------------------------------------------------
GMX = gmx_mpi

TRAJIN = step7_production_01.xtc 100
TRAJIN = step7_production_02.xtc 100
TRAJIN = step7_production_03.xtc 0

STRUCTURE_FILE = step6.7_equilibration.gro

MAIN_RESIDUES = 298
LIGANDS = LIG
GPROTEIN = NONE
LIPIDS = POPC CHL1

TRAJOUT = MAIN 200
TRAJOUT = COMPLEX 200
TRAJOUT = MAIN+LIPIDS 200

MINDIST_CUTDOWN_PS = 1000
"""

INPUT_DOC = """\
INPUT FILE FORMAT
=================
Plain 'KEY = VALUE' lines. Blank lines, '#' comments and trailing ';' comments
are ignored. Keys are case-insensitive. TRAJIN and TRAJOUT may repeat; every
other key appears at most once.

TRAJECTORY INPUT
----------------
  TRAJIN = <xtc> <length_ns>       (repeatable, order defines the time axis)

      The topology is NOT listed. It is always the XTC path with the suffix
      replaced by .tpr, so step7_production_02.xtc implies
      step7_production_02.tpr in the same directory.

      <length_ns> is the segment length used to place the NEXT segment on the
      combined time axis. Every entry except the last must be positive; the
      last must be 0 because nothing follows it. No duration scan is performed
      -- the value you give is trusted.

  REMOVE_INITIAL_PS = <ps>         (optional; auto-detected when absent)

      The overlap window trimmed from the start of every segment after the
      first. When absent, it is read per segment from the matching .mdp file
      (<xtc stem>.mdp, else the single *.mdp in that directory) as

          dt x nstxout-compressed        e.g. 0.002 ps x 5000 = 10 ps

      falling back to nstxout if nstxout-compressed is absent or zero. If no
      .mdp is found, or several ambiguous ones are, the run stops and asks you
      to set this key explicitly.

  REFERENCE_TPR = <tpr>            (default: the first TRAJIN's inferred .tpr)

SYSTEM COMPOSITION
------------------
  STRUCTURE_FILE = <gro|pdb>       (required)

      Reference coordinates. All index groups and the DSSP helix assignment
      are derived from its FIRST FRAME only; no trajectory frame ever reaches
      DSSP. Its atom count is checked against REFERENCE_TPR.

  MAIN_RESIDUES = <N>              (required)

      The receptor is positional residues 1..N in the order they occur in
      STRUCTURE_FILE. This is a count, not a resid.

  LIGANDS  = LIG [UNK ...]         Residue names treated as ligands.
  GPROTEIN = <first> to <last>     Positional range, or NONE.
  LIPIDS   = POPC CHL1 [...]       Added to a built-in lipid catalogue.
  EXTRA_VIRTUAL_NAMES = <names>    Extra atom names excluded as virtual sites
                                   (LP*, EP*, MW, DUM* are always excluded).
  MIN_HELIX_LENGTH = 3             Minimum consecutive DSSP 'H' residues.

TRAJECTORY OUTPUT
-----------------
  TRAJOUT = <EXPRESSION> [cutdown_ps]      (repeatable)

      EXPRESSION is components joined by '+'. Available components:

        SYSTEM     everything
        RECEPTOR   residues 1..MAIN_RESIDUES        (alias: GPCR)
        LIGANDS    all declared ligand residues
        MAIN       RECEPTOR + LIGANDS
        GPROTEIN   the declared G-protein range     (alias: G-PROTEIN)
        COMPLEX    MAIN + GPROTEIN
        PROTEIN    RECEPTOR + GPROTEIN
        LIPIDS     all detected lipid residues
        SOLVENT    water and ions                   (alias: WATER)
        <resname>  any declared ligand or detected lipid, e.g. CHL1, POPC

      Each TRAJOUT produces a reference PDB, an extracted XTC, and a
      downsampled XTC at cutdown_ps. If the selection contains receptor atoms
      it is additionally superposed on RECEPTOR_HELICES_CA and the downsample
      is taken from the fitted trajectory.

  DEFAULT_TRAJOUT_CUTDOWN_PS = 200   Used when a TRAJOUT omits its interval.

WHOLE-SYSTEM OUTPUT
-------------------
  COMBINED_OUTPUT    = step7_production_combined.xtc
  FULL_OUTPUT        = step7_production_pbc_fit.xtc
  FULL_CUTDOWN_PS    = 1000        0 disables the reduced full trajectory
  FULL_CUTDOWN_OUTPUT = <derived from FULL_OUTPUT and the interval>

      The PBC chain is fixed: -pbc whole (SYSTEM), -pbc cluster (PROTEIN
      pivot), -pbc mol, then -center on PROTEIN with -fit rot+trans. Groups
      are selected by NAME from the generated index, so no group-number keys
      exist and the default gmx numbering is never assumed.

INDEX FILES
-----------
  INDEX_FILE     = gpcr_only.ndx
  INDEX_MANIFEST = trajectory_groups.json

MINDIST
-------
  GENERATE_MINDIST_SCRIPT = yes
  MINDIST_SCRIPT          = run_mindist.sh
  MINDIST_CUTDOWN_PS      = 1000
  SLURM_ACCOUNT           = naiss2025-3-21

      Mindist is never run inline: it needs raw, unfitted coordinates, which
      the pipeline does not keep. A standalone sbatch script is written that
      re-concatenates the original XTCs, downsamples, and runs
      'gmx mindist -pi -d 1.2' on PROTEIN. Review it, then submit it.

MISC
----
  GMX              = gmx_mpi
  REMOVE_TEMPORARY = yes
"""

EPILOG = f"""\
{INPUT_DOC}

EXAMPLES
========
  modifyGMXtraj -i input_modify_traj.txt
  modifyGMXtraj -i input.txt --index-only     # write the .ndx, stop
  modifyGMXtraj -i input.txt --dry-run        # print every gmx command
  modifyGMXtraj --write-example > input_modify_traj.txt

OUTPUT FILES
============
  gpcr_only.ndx                        all index groups
  trajectory_groups.json               manifest linking TRAJOUT to groups
  step7_production_combined.xtc        concatenated, time-shifted
  step7_production_pbc_fit.xtc         PBC-corrected, centred, fitted
  <slug>.pdb / traj_<slug>*.xtc        one set per TRAJOUT
  run_mindist.sh                       submit separately with sbatch
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modifyGMXtraj",
        description=textwrap.dedent(
            """\
            Post-process GROMACS trajectories of GPCR simulations: concatenate
            production segments onto one time axis, correct periodic boundary
            artefacts, build named index groups from a reference structure, and
            extract per-selection trajectories fitted on the receptor helices.

            Everything is driven by one plain-text input file given with -i.
            The full key reference is printed below.
            """
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        metavar="FILE",
        default="input_modify_traj.txt",
        help="input configuration file (default: %(default)s)",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="generate the index file and manifest, then stop",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print every GROMACS command without executing it",
    )
    parser.add_argument(
        "--write-example",
        action="store_true",
        help="print a template input file to stdout and exit",
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.write_example:
        print(EXAMPLE_INPUT, end="")
        return 0

    set_dry_run(args.dry_run)

    from .pipeline import run_pipeline

    try:
        run_pipeline(Path(args.input), index_only=args.index_only)
    except (ConfigError, CommandError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
