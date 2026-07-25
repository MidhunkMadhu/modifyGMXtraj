# modifyGMXtraj

Post-processing pipeline for GROMACS trajectories of GPCR simulations.

Concatenates production segments onto a single time axis, corrects periodic
boundary artefacts, builds named index groups from a reference structure, and
extracts per-selection trajectories superposed on the receptor helices. All
behaviour is driven by one plain-text input file.

## Installation

```bash
git clone https://github.com/midhunkmadhu/modifyGMXtraj.git
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
- **PBC groups are selected by name** (`SYSTEM`, `PROTEIN`) from the generated
  index, so no group-number settings exist and the default GROMACS numbering
  is never assumed.

## What gets written

| File | Contents |
|---|---|
| `gpcr_only.ndx` | all index groups |
| `trajectory_groups.json` | manifest linking each `TRAJOUT` to its group |
| `step7_production_combined.xtc` | concatenated, time-shifted |
| `step7_production_pbc_fit.xtc` | PBC-corrected, centred, fitted |
| `<slug>.pdb`, `traj_<slug>*.xtc` | one set per `TRAJOUT` |
| `run_mindist.sh` | submit separately |

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
  trajectory duration.
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
└── cli.py       argument parsing and the -h reference
```

`config.py` exists because the original two-script layout carried two
independent copies of the parser that could drift apart. There is now one.
