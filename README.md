# modifyGMXtraj

Post-processing pipeline for GROMACS trajectories of GPCR simulations.

Concatenates production segments onto a single time axis, corrects periodic
boundary artefacts, builds named index groups from a reference structure, and
extracts per-selection trajectories superposed on the receptor helices. All
behaviour is driven by one plain-text input file.

## Installation

```bash
git clone https://github.com/<your-user>/modifyGMXtraj.git
cd modifyGMXtraj
conda env create -f environment.yml
conda activate modifygmxtraj
modifyGMXtraj --help
```

`environment.yml` installs MDTraj from conda-forge and then runs
`pip install -e .`, so edits to the source take effect without reinstalling.

If you prefer to add it to an existing environment:

```bash
conda activate <your-env>
conda install -c conda-forge mdtraj numpy
pip install -e /path/to/modifyGMXtraj
```

GROMACS is **not** installed by this package. It is invoked as an external
command (`gmx_mpi` by default, set `GMX` in the input file to change it), so
load your cluster's module first:

```bash
module load GROMACS/2023.3-cpeGNU  # Dardel; adjust for your site
```

## Usage

```bash
modifyGMXtraj --write-example > input_modify_traj.txt   # template
modifyGMXtraj -i input_modify_traj.txt --index-only     # groups only
modifyGMXtraj -i input_modify_traj.txt --dry-run        # print gmx commands
modifyGMXtraj -i input_modify_traj.txt                  # run
sbatch run_mindist.sh                                   # separately
```

`modifyGMXtraj --help` prints the complete input-file key reference.


## Running faster

Every step is a single-threaded `trjconv` pass over the trajectory, so the
run time is dominated by reading and writing whole-system frames, not by
CPU. Three optional keys (defaults keep the original behaviour):

| key | effect |
|---|---|
| `EARLY_CUTDOWN_PS = 200` | keep only every 200 ps from the first PBC pass on; with 100 ps input and outputs at 200/1000 ps this halves the work. Output intervals must be multiples of it. |
| `PARALLEL_JOBS = 4` | build the reduced full trajectory and the TRAJOUT sets concurrently; request the same number of cores. |
| `TEMPORARY_DIR = /path` | where the large intermediate files go. |

On clusters with whole-node partitions, use a shared/small partition: the
tool needs one core per parallel job, e.g. on Dardel

```bash
#SBATCH -p shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4      # = PARALLEL_JOBS
#SBATCH --mem=16gb
```

The biggest gain for many replicas is to submit one such job per run
directory so they all run at the same time.

## Minimal input file

```
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
```

Three conventions remove most of the boilerplate:

- **The `.tpr` is never listed.** It is `<xtc stem>.tpr`.
- **`REMOVE_INITIAL_PS` is auto-detected** from the matching `.mdp` as
  `dt x nstxout-compressed` (e.g. `0.002 x 5000 = 10 ps`). Set the key
  explicitly to override.
- **`--write-example` auto-detects** `step7_production_??.xtc` in the current
  directory and writes one `TRAJIN` line per segment in order, with placeholder
  lengths of 100 ns (last = 0). Verify the lengths before running.

## Index-file scoping

Whole-system steps (concatenation, the `-pbc whole` / `cluster` / `mol` chain,
`-center -fit`, and the full-trajectory downsample) use GROMACS's own inherent
default groups with **no** `-n`: `0 = System`, `1 = Protein`. Supplying an index
file *replaces* gmx's automatic classification rather than adding to it, which
breaks `-pbc cluster` on systems containing lone-pair/dummy atoms.

The custom `gpcr_only.ndx` is used **only** for per-`TRAJOUT` extraction and the
`RECEPTOR_HELICES_CA` fit, where the required selections don't exist as gmx
defaults.

## What gets written

| File | Contents |
|---|---|
| `gpcr_only.ndx` | all index groups |
| `trajectory_groups.json` | manifest linking each `TRAJOUT` to its group |
| `combined.xtc` | concatenated, time-shifted |
| `pbc_fit.xtc` | PBC-corrected, centred, fitted |
| `pbc_fit_1000ps.xtc` | reduced full trajectory |
| `modifyGMXtraj.log` | full run log |
| `traj_fit.xtc`, `traj_fit_200ps.xtc` | the `MAIN` products |
| `<slug>.pdb`, `traj_<slug>*.xtc` | one set per other `TRAJOUT` |
| `run_mindist.sh` | submit separately |

`MAIN` drops its slug from the fitted products (`traj_fit.xtc` rather than
`traj_main_fit.xtc`) since it's the primary output; every other selection keeps
its slug so files stay distinguishable.

Output names are deliberately generic and never encode the input filename. An
earlier version defaulted to `step7_production_*`, which mislabelled runs on
equilibration trajectories and — worse — meant that later processing the real
production trajectory in the same directory silently overwrote the earlier
results. Keep separate runs in separate directories, or set `COMBINED_OUTPUT`
and `FULL_OUTPUT` explicitly.

## Logging

Every run writes `modifyGMXtraj.log` (override with `LOG_FILE` or `--log`,
disable with `--no-log`) containing the verbatim input file, every resolved
parameter, each GROMACS command with its selections, and an inventory of files
written with sizes. GROMACS's own output streams to the terminal by default;
`LOG_GMX_OUTPUT = yes` routes it into the log instead.

## Why mindist is a separate script

`gmx mindist -pi` measures the minimum distance between periodic images of the
protein. It must therefore see raw, unfitted, un-PBC-corrected coordinates —
exactly what the pipeline does not keep. Rather than carry a parallel raw
concatenation through the main run, the pipeline writes a standalone script
that redoes the concatenation from the original XTCs, downsamples, and runs
mindist. Review the SBATCH header, then submit it.

## Scope and known limits

This is deliberately a GPCR-only tool.

- The receptor is **positional residues 1..`MAIN_RESIDUES`**, and only one
  additional protein entity (`GPROTEIN`, a contiguous positional range) is
  supported. Peptide ligands, nanobodies and dimer partners are not.
- `RECEPTOR_HELICES_CA` comes from DSSP on the reference structure and is a
  hard requirement: non-helical systems and coarse-grained models will not run.
- Every `TRAJOUT` derives from the rotationally fitted trajectory, so there is
  no lab-frame output. Area-per-lipid, lateral diffusion and MSD need a
  separate `-pbc nojump` pass.
- Segment lengths in `TRAJIN` are trusted, not verified against the actual
  trajectory duration (including the placeholder lengths `--write-example`
  emits).
- The lipid heuristic (>=12 C and >=15 heavy atoms) will absorb undeclared
  cofactors such as GTP or heme into `LIPIDS`. Read the warnings.

## Layout

```
modifygmxtraj/
├── config.py    input parsing, validators, .mdp discovery
├── indexer.py   index-group generation and DSSP helix detection
├── gmx.py       GROMACS invocation helpers
├── pipeline.py  concatenation, PBC correction, TRAJOUT extraction
├── mindist.py   mindist script generation
├── runlog.py    run logging and file inventory
└── cli.py       argument parsing and the -h reference
```

`config.py` exists because the original two-script layout carried two
independent copies of the parser that could drift apart. There is now one.
