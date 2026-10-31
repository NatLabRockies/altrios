# ALTRIOS

![Altrios Logo](https://raw.githubusercontent.com/NatLabRockies/altrios/main/.github/images/ALTRIOS-logo-web.png)

[![homepage](https://img.shields.io/badge/homepage-altrios-blue)](https://www.nlr.gov/transportation/altrios.html) [![rust-tests](https://github.com/NatLabRockies/altrios/actions/workflows/rust-tests.yaml/badge.svg)](https://github.com/NatLabRockies/altrios/actions/workflows/rust-tests.yaml) [![py-tests](https://github.com/NatLabRockies/altrios/actions/workflows/py-tests.yaml/badge.svg)](https://github.com/NatLabRockies/altrios/actions/workflows/py-tests.yaml) [![wheels](https://github.com/NatLabRockies/altrios/actions/workflows/wheels.yaml/badge.svg)](https://github.com/NatLabRockies/altrios/actions/workflows/wheels.yaml?event=release) [![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://pypi.org/project/altrios/) [![Documentation](https://img.shields.io/badge/doc-mdBook-blue.svg)](https://natlabrockies.github.io/altrios/) [![GitHub](https://img.shields.io/badge/GitHub-altrios-blue.svg)](https://github.com/NatLabRockies/altrios)


![Model Framework Schematic](https://raw.githubusercontent.com/NatLabRockies/altrios/main/.github/images/ALTRIOS_schematic.png)

The Advanced Locomotive Technology and Rail Infrastructure Optimization System ([ALTRIOS](https://www.nlr.gov/transportation/altrios.html)) is a unique, fully integrated, open-source software tool to evaluate strategies for deploying advanced locomotive technologies and associated infrastructure. ALTRIOS simulates freight-demand driven train scheduling, mainline meet-pass planning, locomotive dynamics, train dynamics, energy conversion efficiencies, and energy storage dynamics of line-haul train operations. Because new locomotives represent a significant long-term capital investment and new technologies must be thoroughly demonstrated before deployment, this tool provides guidance on the risk/reward tradeoffs of different technology rollout strategies. An open, integrated simulation tool is invaluable for identifying future research needs and making decisions on technology development, routes, and train selection. ALTRIOS was developed as part of a collaborative effort by a team comprising the National Laboratory of the Rockies (NLR), the University of Texas at Austin (UT Austin), Southwest Research Institute (SwRI), and BNSF Railway.

Much of the core code in ALTRIOS is written in the [Rust Programming Language](https://www.rust-lang.org/) to ensure excellent computational performance and robustness, but we've built ALTRIOS with the intent of users interacting with the code through our feature-rich [Python](https://www.python.org/) interface.  

## Installation

ALTRIOS is a Python package with a compiled Rust core. Installing a release from PyPI needs no Rust toolchain; the Rust core ships pre-compiled. If you want to modify ALTRIOS itself, see the [Developer Documentation](https://NatLabRockies.github.io/altrios/developers.html) instead.

### Prerequisites

Python 3.10 through 3.14. Install into a virtual environment so ALTRIOS and its dependencies stay separate from other Python projects:

```
python -m venv altrios-venv
altrios-venv\Scripts\activate       # Windows
source altrios-venv/bin/activate    # macOS, Linux
```

conda users: `conda create -n altrios python=3.13` then `conda activate altrios`.

### Install

With the environment active:

```
pip install altrios
python -c "import altrios; print(altrios.__version__)"
```

The second line prints the installed version and confirms the install worked. If `pip` reports `No matching distribution found`, the newest release on PyPI does not yet provide wheels for your Python version: use an older supported version, or build from source as described in the [Developer Documentation](https://NatLabRockies.github.io/altrios/developers.html).

## Running ALTRIOS

ALTRIOS is used from Python scripts and notebooks, and the package ships a set of demo scripts that exercise most major capabilities. The quickest way to see it work:

```
python -c "import altrios as alt; alt.copy_demo_files()"
python demos/speed_limit_train_sim_demo.py
```

The first line copies the demos into a `demos/` folder in your current directory. The second simulates one manifest train over the Taconite network with speed limits enforced; it runs for a few seconds to half a minute, depending on your machine, and then opens plots of speed, power and energy. Close the plot windows to let the script finish. To run without plot windows (for example on a server), set `SHOW_PLOTS=false` in the environment first: `$env:SHOW_PLOTS="false"` in PowerShell, `set SHOW_PLOTS=false` in Command Prompt, `export SHOW_PLOTS=false` elsewhere.

### The demo scripts

Each demo runs in a couple of minutes or less (seconds for single-train demos). Outputs are written next to the script, in `plots/` and `results/`.

| Script | What it shows |
|---|---|
| `speed_limit_train_sim_demo.py` | One train over the Taconite network, obeying speed limits; the core single-train simulation |
| `speed_limit_train_sim_demo_with_derating.py` | The same train with locomotive power derating applied |
| `set_speed_train_sim_demo.py` | One train following a prescribed speed trace |
| `conv_demo.py`, `bel_demo.py`, `hel_demo.py` | A single conventional, battery-electric or hybrid locomotive driven by a power trace |
| `catenary_demo.py` | A dual-mode locomotive with and without overhead power, and the energization optimizer choosing where to put wire under a budget |
| `set_speed_simple_corr_demo.py`, `speed_limit_simple_corr_demo.py` | The two train simulators on a small synthetic corridor |
| `manual_train_planner_demo.py` | Driving the simulation from a hand-written consist plan |
| `sim_manager_demo.py` | A multi-week fleet simulation from freight demand: train planning, dispatch, meet-pass and energy for a whole network |
| `rollout_demo.py` | A multi-year technology rollout with cost and emissions metrics |

### Working interactively

The demos are divided into cells marked `# %%`, so you can step through them in [VS Code](https://code.visualstudio.com/Download) with the Python extension, or in [Spyder](https://www.spyder-ide.org/): put the cursor in a cell and press `Shift+Enter` to run it in an interactive window and move to the next cell. This is the easiest way to inspect intermediate objects and adapt a demo to your own case.

### Using your own data

The demos load their inputs from the files bundled with the package, via `alt.resources_root()`. To change an input, copy the file next to your script, edit it, and load it from there. For example

```python
res = alt.ReversibleEnergyStorage.from_file(
   alt.resources_root() / "powertrains/reversible_energy_storages/Kokam_NMC_75Ah_flx_drive.yaml"
)
```

becomes

```python
res = alt.ReversibleEnergyStorage.from_file("./custom_battery.yaml")
```

Nearly every object in ALTRIOS can be read from and written to YAML, JSON and MessagePack with `from_file` and `to_file`; see [the SerdeAPI trait](https://docs.rs/altrios-core/latest/altrios_core/traits/serde_api/trait.SerdeAPI.html) for the full set of methods, all of which are available from Python.


## Acknowledgements

The ALTRIOS Team would like to thank ARPA-E for financially supporting the research through the LOCOMOTIVES program and Dr. Robert Ledoux for his vision and support. We would also like to thank the ARPA-E team for their support and guidance: Dr. Apoorv Agarwal, Mirjana Marden, Alexis Amos, and Catherine Good.  We would also like to thank BNSF for their cost share financial support, guidance, and deep understanding of the rail industry’s needs.  Additionally, we would like to thank Jinghu Hu for his contributions to the core ALTRIOS code.  We would like to thank Chris Hennessy at SwRI for his support. Thank you to Michael Cleveland for his help with developing and kicking off this project.  
