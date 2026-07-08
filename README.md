# ALTRIOS

![Altrios Logo](.github/images/ALTRIOS-logo-web.png)

[![homepage](https://img.shields.io/badge/homepage-altrios-blue)](https://www.nlr.gov/transportation/altrios.html) [![rust-tests](https://github.com/NatLabRockies/altrios/actions/workflows/rust-tests.yaml/badge.svg)](https://github.com/NatLabRockies/altrios/actions/workflows/rust-tests.yaml) [![py-tests](https://github.com/NatLabRockies/altrios/actions/workflows/py-tests.yaml/badge.svg)](https://github.com/NatLabRockies/altrios/actions/workflows/py-tests.yaml) [![wheels](https://github.com/NatLabRockies/altrios/actions/workflows/wheels.yaml/badge.svg)](https://github.com/NatLabRockies/altrios/actions/workflows/wheels.yaml?event=release) [![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://pypi.org/project/altrios/) [![Documentation](https://img.shields.io/badge/doc-mdBook-blue.svg)](https://natlabrockies.github.io/altrios/) [![GitHub](https://img.shields.io/badge/GitHub-altrios-blue.svg)](https://github.com/NatLabRockies/altrios)


![Model Framework Schematic](.github/images/ALTRIOS_schematic.png)

The Advanced Locomotive Technology and Rail Infrastructure Optimization System ([ALTRIOS](https://www.nlr.gov/transportation/altrios.html)) is a unique, fully integrated, open-source software tool to evaluate strategies for deploying advanced locomotive technologies and associated infrastructure. ALTRIOS simulates freight-demand driven train scheduling, mainline meet-pass planning, locomotive dynamics, train dynamics, energy conversion efficiencies, and energy storage dynamics of line-haul train operations. Because new locomotives represent a significant long-term capital investment and new technologies must be thoroughly demonstrated before deployment, this tool provides guidance on the risk/reward tradeoffs of different technology rollout strategies. An open, integrated simulation tool is invaluable for identifying future research needs and making decisions on technology development, routes, and train selection. ALTRIOS was developed as part of a collaborative effort by a team comprising the National Laboratory of the Rockies (NLR), the University of Texas at Austin (UT Austin), Southwest Research Institute (SwRI), and BNSF Railway.

Much of the core code in ALTRIOS is written in the [Rust Programming Language](https://www.rust-lang.org/) to ensure excellent computational performance and robustness, but we've built ALTRIOS with the intent of users interacting with the code through our feature-rich [Python](https://www.python.org/) interface.  

## Installation

If you are an ALTRIOS developer, see [Developer Documentation](https://NatLabRockies.github.io/altrios/developers.html).  Otherwise, read on.  

### Python Setup

1. Python installation options:
   - Option 1 -- Python: https://www.python.org/downloads/. We recommend Python 3.10. Be sure to check the `Add to PATH` option during installation.
   - Option 2 -- Anaconda: we recommend https://docs.conda.io/en/latest/miniconda.html.
1. Setup a python environment. ALTRIOS can work with Python 3.9, or 3.10, but we recommend 3.10 for better performance and user experience. Create a python environment for ALTRIOS with either of two methods:
   - Option 1 -- [Python Venv](https://docs.python.org/3/library/venv.html)
     1. Navigate to the ALTRIOS folder you just cloned or any folder you'd like for using ALTRIOS. Remember the folder you use!
     1. Assuming you have Python 3.10 installed, run 
         - `(path to your python3.10 e.g. ~/AppData/Local/Programs/Python/Python310/python.exe) -m venv altrios-venv` in Windows
         - `python3.10 -m venv altrios-venv` in Mac/Unix/Linux
        in your terminal enviroment (we recommend PowerShell in Windows, which comes pre-installed). 
      
        This tells Python 3.10 to use the `venv` module to create a virtual environment (which will be ignored by git if named `altrios-venv`) in the `ALTRIOS/altrios-venv/`.
     1. Activate the environment you just created to install packages or anytime you're running ALTRIOS:
        - Mac and Linux: `source altrios-venv/bin/activate`
        - Windows: `altrios-venv/Scripts/activate.bat` in a windows command prompt or power shell or `source altrios-venv/Scripts/activate` in git bash terminal
        - When the environment is activated, your terminal session will have a decorator that looks like `(altrios-venv)`.
   - Option 2 -- Anaconda:
     1. Open an Anaconda prompt (in Windows, we recommend _Anaconda_ Powershell Prompt) and run the command `conda create -n altrios python=3.10` to create an Anaconda environment named `altrios`.
     1. Activate the environment to install packages or anytime you're running ALTRIOS: run `conda activate altrios`.

### ALTRIOS Setup

With your Python environment activated, run `pip install altrios`.

Congratulations, you've completed installation! Whenever you need to use ALTRIOS, be sure to activate your python environment created above.

## How to run ALTRIOS

With your activated Python environment with ALTRIOS fully installed, you can download the demo scripts to the current working directory inside of a `demos/` folder with:
```python
import altrios as alt  
alt.copy_demo_files()
```
You can run the Simulation Manager through a multi-week simulation of train operations in by running `python sim_manager_demo.py` in `demos/`. This will create a `plots/` subfolder in which the plots will be saved. To run interactively, fire up a Python IDE (e.g. [VS Code](https://code.visualstudio.com/Download), [Spyder](https://www.spyder-ide.org/)), and run the file. If you're in VS Code, you can run the file as a virtual jupyter notebook because of the "cells" that are marked with the `# %%` annotation. You can click on line 2, for example, and hit `<Shift> + <Enter>` to run the current cell in an interactive terminal (which will take several seconds to launch) and advance to the next cell. Alternatively, you can hit `<Ctrl> + <Shift> + p` to enable interactive commands and type "run current cell".  There are several other python files in the `demos/` folder to demonstrate various capabilities of ALTRIOS.  

If you plan to modify the data used in the demo files, copy the data files to your local directory and load them  from there, e.g.  
```python
res = alt.ReversibleEnergyStorage.from_file(
   alt.resources_root() / "powertrains/reversible_energy_storages/Kokam_NMC_75Ah_flx_drive.yaml"
)
```
would become  
```python
res = alt.ReversibleEnergyStorage.from_file(
   "./custom_battery.yaml"
)
```

Nearly every code object in ALTRIOS can be read from or written to common data formats.  For more details, see [the SerdeAPI trait](https://docs.rs/altrios-core/latest/altrios_core/traits/trait.SerdeAPI.html) documentation.  All of the functions in the SerdeAPI are available through the python interface.  



## Acknowledgements

The ALTRIOS Team would like to thank ARPA-E for financially supporting the research through the LOCOMOTIVES program and Dr. Robert Ledoux for his vision and support. We would also like to thank the ARPA-E team for their support and guidance: Dr. Apoorv Agarwal, Mirjana Marden, Alexis Amos, and Catherine Good.  We would also like to thank BNSF for their cost share financial support, guidance, and deep understanding of the rail industry’s needs.  Additionally, we would like to thank Jinghu Hu for his contributions to the core ALTRIOS code.  We would like to thank Chris Hennessy at SwRI for his support. Thank you to Michael Cleveland for his help with developing and kicking off this project.  
