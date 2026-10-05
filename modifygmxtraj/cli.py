"""Command-line interface for modifyGMXtraj."""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from . import __version__
from .config import CommandError, ConfigError
from .gmx import set_dry_run

EXAMPLE_HEADER = """\
# ---------------------------------------------------------------------------
# modifyGMXtraj input file
#
# TRAJIN = <xtc> <length_ns>   -- the .tpr is inferred as <xtc stem>.tpr
#                                 the LAST entry must use 0
# With several segments, the overlapping first frame of every segment after
# the first is trimmed automatically (dt x nstxout-compressed from its .mdp).
# ---------------------------------------------------------------------------
GMX = gmx_mpi
"""

EXAMPLE_TAIL = """\
STRUCTURE_FILE = step6.7_equilibration.gro

MAIN_RESIDUES = 298
LIGANDS = NONE
GPROTEIN = NONE
LIPIDS = POPC CHL1

TRAJOUT = MAIN 200
TRAJOUT = MAIN+LIPIDS 200

MINDIST_CUTDOWN_PS = 1000

# Speed (optional). auto skips frames no output needs; note that pbc_fit.xtc and
# the unreduced traj_*.xtc are then at that coarser interval.
# EARLY_CUTDOWN_PS = auto
# PARALLEL_JOBS    = 4        ; request the same number of SLURM cores
"""


def detect_trajin_lines(directory: Path = Path(".")) -> tuple[list[str], str]:
    """Scan for step7_production_??.xtc and build TRAJIN lines in order.

    Each detected segment gets a placeholder length of 100 ns except the last,
    which must be 0. If nothing matches the numbered pattern, fall back to a
    single unnumbered step7_production.xtc entry.
    """
    matches = sorted(directory.glob("step7_production_??.xtc"))

    if not matches:
        return (
            ["TRAJIN = step7_production.xtc 0"],
            "# No step7_production_??.xtc found; wrote a single default entry.",
        )

    lines: list[str] = []
    for position, path in enumerate(matches):
        length = 0 if position == len(matches) - 1 else 100
        lines.append(f"TRAJIN = {path.name} {length}")

    note = (
        f"# Auto-detected {len(matches)} segment(s) matching "
        "step7_production_??.xtc.\n"
        "# Lengths are PLACEHOLDERS (100 ns each, last = 0) -- verify them "
        "against\n# your actual segment durations before running."
    )
    return lines, note


def build_example(directory: Path = Path(".")) -> str:
    trajin_lines, note = detect_trajin_lines(directory)
    return (
        EXAMPLE_HEADER
        + "\n"
        + note
        + "\n"
        + "\n".join(trajin_lines)
        + "\n\n"
        + EXAMPLE_TAIL
    )


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
      last must be 0 because nothing follows it. No duration scan is
      performed -- the value you give is trusted.

  REMOVE_INITIAL_PS = <ps>         (optional; auto-detected when absent)

      The overlap window trimmed from the start of every segment after the
      first. When absent, it is read per segment from the matching .mdp file
      (<xtc stem>.mdp, else the single *.mdp in that directory) as

          dt x nstxout-compressed        e.g. 0.002 ps x 5000 = 10 ps

      falling back to nstxout if nstxout-compressed is absent or zero.

  REFERENCE_TPR = <tpr>            (default: the first TRAJIN's inferred .tpr)

SYSTEM COMPOSITION
------------------
  STRUCTURE_FILE = <gro|pdb>       (required)

      Reference coordinates. All index groups and the DSSP helix assignment
      are derived from its FIRST FRAME only. Its atom count is checked
      against the first TRAJIN trajectory before anything runs; a mismatch
      (e.g. lone-pair/dummy atoms present in one but not the other) aborts
      immediately rather than failing later inside trjconv.

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

      Naming: MAIN drops its slug from the fitted products, giving
      traj_fit.xtc and traj_fit_<N>ps.xtc. Every other selection keeps it,
      e.g. traj_main_lipids_fit.xtc.

  DEFAULT_TRAJOUT_CUTDOWN_PS = 200   Used when a TRAJOUT omits its interval.

WHOLE-SYSTEM OUTPUT
-------------------
  COMBINED_OUTPUT     = combined.xtc
  FULL_OUTPUT         = pbc_fit.xtc
  FULL_CUTDOWN_PS     = 1000       0 disables the reduced full trajectory
  FULL_CUTDOWN_OUTPUT = <derived from FULL_OUTPUT and the interval>

      The PBC chain is fixed: -pbc whole, -pbc cluster (Protein pivot),
      -pbc mol, then -center and -fit rot+trans on Protein. These whole-system
      steps deliberately use GROMACS's own inherent default groups (0=System,
      1=Protein) with NO -n, because supplying an index file replaces gmx's
      automatic classification instead of adding to it. The custom index is
      used only for the TRAJOUT selections and the helix fit.

INDEX FILES
-----------
  INDEX_FILE     = gpcr_only.ndx
  INDEX_MANIFEST = trajectory_groups.json

LOGGING
-------
  LOG_FILE       = modifyGMXtraj.log
  LOG_GMX_OUTPUT = no

      Every run writes a log containing the verbatim input file, the fully
      resolved parameters, every GROMACS command and selection, and an
      inventory of files written with their sizes. GROMACS's own output
      streams to the terminal by default; LOG_GMX_OUTPUT = yes routes it
      through Python into the log instead, at the cost of live streaming.

SPEED (all optional; the defaults reproduce the original sequential run)
-----
  EARLY_CUTDOWN_PS = 0 | auto      0 = off. When > 0, the first PBC pass keeps
                                   only frames every EARLY_CUTDOWN_PS ps (-dt),
                                   so every later pass and every TRAJOUT step
                                   handles fewer frames. The PBC treatment of each
                                   kept frame is unchanged, but FULL_OUTPUT and the
                                   extracted/fitted TRAJOUT XTCs are then at this
                                   interval, not the native one. Every TRAJOUT
                                   cutdown and FULL_CUTDOWN_PS must be a whole
                                   multiple of it (checked before anything runs).
                                   E.g. native 100 ps frames, outputs at 200/1000
                                   ps: EARLY_CUTDOWN_PS = 200 halves the work.
                                   'auto' reads the frame interval from the xtc
                                   files and picks the largest interval every
                                   output is a multiple of, or stays off when
                                   nothing can be skipped; the choice and the
                                   reason are written to the run log.

  PARALLEL_JOBS = 1                Run the reduced full trajectory and the TRAJOUT
                                   sets (which only read FULL_OUTPUT) this many at a
                                   time. Each set is a chain of single-threaded
                                   trjconv calls, so request as many SLURM cores as
                                   jobs (e.g. --cpus-per-task=4 for 4). GROMACS
                                   output is captured and printed per set while
                                   they run concurrently.

  TEMPORARY_DIR = <dir>            Where the trimmed/shifted segments and the three
                                   whole-system PBC intermediates are written
                                   (default: next to the inputs / working dir).

MINDIST
-------
  GENERATE_MINDIST_SCRIPT = yes
  MINDIST_SCRIPT          = run_mindist.sh
  MINDIST_CUTDOWN_PS      = 1000
  SLURM_ACCOUNT           = naiss2025-3-21
  SLURM_PARTITION         = shared     mindist is one serial process; a shared
  SLURM_MEM               = 8G         partition avoids billing a whole node

      Mindist is never run inline: it needs raw, unfitted coordinates, which
      the pipeline does not keep. A standalone sbatch script is written that
      re-concatenates the original XTCs, downsamples, and runs
      'gmx mindist -pi -d 1.2'. Review it, then submit it.

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
  modifyGMXtraj -i input.txt --log run2.log   # override the log path
  modifyGMXtraj --write-example > input_modify_traj.txt

  --write-example scans the current directory for step7_production_??.xtc and
  writes one TRAJIN line per segment in order, with placeholder lengths of
  100 ns (last = 0). If none are found it writes a single
  'TRAJIN = step7_production.xtc 0'. Always verify the lengths.

OUTPUT FILES
============
  modifyGMXtraj.log                    full run log
  gpcr_only.ndx                        all index groups
  trajectory_groups.json               manifest linking TRAJOUT to groups
  combined.xtc                         concatenated, time-shifted
  pbc_fit.xtc                          PBC-corrected, centred, fitted
  pbc_fit_1000ps.xtc                   reduced full trajectory
  traj_fit.xtc / traj_fit_200ps.xtc    the MAIN products
  <slug>.pdb / traj_<slug>*.xtc        one set per other TRAJOUT
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
        "-i", "--input", metavar="FILE", default="input_modify_traj.txt",
        help="input configuration file (default: %(default)s)",
    )
    parser.add_argument(
        "--index-only", action="store_true",
        help="generate the index file and manifest, then stop",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print every GROMACS command without executing it",
    )
    parser.add_argument(
        "--log", metavar="FILE", default=None,
        help="log file path (overrides LOG_FILE in the input file)",
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="disable log file writing entirely",
    )
    parser.add_argument(
        "--write-example", action="store_true",
        help="print a template input file to stdout and exit; auto-detects "
             "step7_production_??.xtc in the current directory",
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.write_example:
        print(build_example(), end="")
        return 0

    set_dry_run(args.dry_run)

    from .pipeline import run_pipeline

    try:
        run_pipeline(
            Path(args.input),
            index_only=args.index_only,
            log_override=Path(args.log) if args.log else None,
            no_log=args.no_log,
            version=__version__,
        )
    except (ConfigError, CommandError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
