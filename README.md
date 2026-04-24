# AstroSkyFlow

AstroSkyFlow is a modular photometric image simulator designed to generate high-fidelity, multi-epoch astronomical image sequences for pipeline validation, signal-injection experiments, and machine-learning applications.

## Repository Structure

This repository is organized into four main parts:

```text
AstroSkyFlow/
├── modular_core/
├── tests/
├── reproduce_paper/
├── README.md
└── LICENSE
```

- `modular_core/` contains the main code and the modular directory structure of the simulator.
- `tests/` contains minimal public test files for lightweight software validation.
- `reproduce_paper/` contains the materials needed to reproduce the scientific results presented in the manuscript.
- `README.md` describes the repository structure and usage.
- `LICENSE` provides the software license information.

## 1. Main simulator structure: `modular_core/`

`modular_core/` corresponds to the “Modular directory structure” described in the manuscript.

A typical layout is:

```text
modular_core/
├── simulator.py
├── local_catalog_screening.py
├── config.json
├── schedule.csv
├── sim_events/
│   ├── transit.csv
│   ├── binary.csv
│   ├── flare.csv
│   ├── occultation.csv
│   ├── supernova_erupt.csv
│   ├── satellite.txt
│   └── user-added_scatter_light.fits / .npy
├── reference_data/
│   ├── star_catalog.fits
│   ├── galaxy_catalog.csv
│   └── filter_transmission.csv
└── out/
```

The main execution logic is implemented in `simulator.py`, which comprises several cooperating classes, while `local_catalog_screening.py` handles local catalog filtering for specific FOVs in different observation tasks and is called by `simulator.py` during execution.

AstroSkyFlow requires users to supply reference photometric data specifying the observational system: a stellar catalog, a galaxy catalog, and filter transmission curves. These files should be placed in the `reference_data` directory.

## 2. Consistent internal directory organization

All runnable folders in this repository follow the same internal organization as `modular_core/`, including:

- `config.json`
- `schedule.csv`
- `sim_events/`
- `reference_data/`
- output directory structure

This consistent layout is used to simplify software testing, manuscript reproducibility, and user customization.

## 3. Minimal public tests: `tests/`

The `tests/` directory contains the **minimal executable files** used for lightweight software validation.

All files in this directory are intentionally reduced to the minimum needed to confirm that the code can run successfully. In particular, the `star_catalog` and `galaxy_catalog` included in `tests/` contain only the objects located in the sky region covered by the corresponding `schedule.csv`. These reduced catalogs are provided only to demonstrate code execution and basic workflow validation. They are **not** intended to be used as general-purpose reference catalogs for other simulations.

Thus, the role of `tests/` is to provide minimal runnable examples for software testing, rather than scientific validation or large-scale simulation.

## 4. Manuscript reproducibility: `reproduce_paper/`

The `reproduce_paper/` directory contains the materials needed to reproduce the validation workflows and example results described in the paper.

This directory has already been configured according to the manuscript settings for the **Muguang-transit case** and the **Xinglong-binary case**. The generated images can be used to reproduce the corresponding results presented in the paper.

Some paper-reproduction examples require large reference catalogs that are distributed separately through Zenodo rather than duplicated in the GitHub repository.

- For the **transit** reproduction workflow, the required `star_catalog` and `galaxy_catalog` in `reference_data/` should be downloaded from the Zenodo dataset.
- For the **binary** reproduction workflow, the required `galaxy_catalog` in `reference_data/` should also be downloaded from the Zenodo dataset.

After downloading, please place these files in the corresponding `reference_data/` directory before running the reproduction workflows.

### Zenodo reference dataset

The reference catalog dataset is archived separately on Zenodo because of file size. It contains the large star and galaxy catalogs used by AstroSkyFlow.

Zenodo dataset DOI: `10.5281/zenodo.18830766`

Users should download the required files from the Zenodo record and place them into the appropriate `reference_data/` folder of the corresponding working directory.

## 5. Quick start

AstroSkyFlow is organized into a modular directory structure to separate input configurations, reference data, and simulation outputs. The main components are the core scripts, configuration files, events directory, reference photometric data directory, and output directory.

The main execution logic is implemented in `simulator.py`, which comprises several cooperating classes. `local_catalog_screening.py` handles local catalog filtering for specific FOVs in different observation tasks and is called by `simulator.py` during execution. The simulator parameters of different classes are specified in `config.json`.

### Step 1. Prepare reference photometric data

AstroSkyFlow requires users to supply reference photometric data specifying the observational system: a stellar catalog, a galaxy catalog, and filter transmission curves. These files should be placed in the `reference_data/` directory.

If the simulated observation system lacks a proprietary stellar catalog, users can set the corresponding parameter `star_catalog` in `config.json` to `"online"`. AstroSkyFlow will then query the Gaia archive over the network and use the Gaia DR3 catalog and its photometry as the reference stellar catalog. Note that this mode requires an internet connection and may be subject to archive-query limits and latency.

### Step 2. Prepare the observation schedule

Modify `schedule.csv` to define the observing sequence. The file includes the observation order, target names, celestial coordinates, number of frames, exposure time, and start and end times. Individual observations must be arranged in strict chronological order.

### Step 3. Prepare event files

The `sim_events/` directory contains the schedule file and other variable-input files. These include `transit`, `binary`, `flare`, `occultation`, `supernova_erupt`, and `satellite` files. Users can also provide a user-defined scatter-light file.

If users need to inject specific variable sources, they can modify the parameters in the corresponding event files. To successfully inject and simulate a variable source, it is essential that this variable exists within the stellar catalog used for the current observation. If users do not require the source to be truly present, they may assign the desired optical variability properties to any star in the fixed stellar catalog by entering the Gaia DR3 ID of that fixed star into the corresponding column of the event files.

### Step 4. Edit configuration

Final setup is completed by adjusting parameters in `config.json`. Users may modify only the key parameters and remove the others; the code will automatically apply default values for unspecified entries.

Two essential operational constraints must be satisfied:

1. the initial simulator time set in `config.json` must be earlier than the start time of the first scheduled observation;  
2. all scheduled targets must be at an altitude greater than 25 degrees at their respective observation times.

This constraint mirrors real-world telescope safety protocols, and AstroSkyFlow will generate a warning if it is violated.

### Step 5. Run AstroSkyFlow

Run the simulator from the corresponding working directory:

```bash
python simulator.py
```

### Step 6. Output products

In the output directory, each scheduled target has a dedicated subdirectory that contains the simulated FITS images and corresponding injected-variable files for subsequent comparison and validation.
