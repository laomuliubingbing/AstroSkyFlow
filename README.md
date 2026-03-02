# Astronomical Photometric Simulator

## Overview

This simulator generates synthetic astronomical observation images with configurable parameters, variable sources, and realistic effects. It is designed with a modular architecture to facilitate customization and reproducibility.

## Directory Structure

The simulator follows a modular organization separating core scripts, input configurations, reference data, and simulation outputs:

```
simulator/
├── simulator.py                            # Main execution script
├── local_catalog_screening.py              # FOV-specific catalog filtering
├── config.json                             # Simulator parameters configuration
├── events/                                 # Observation schedules and variable sources
│   ├── schedule.csv                        # Observation targets and timeline
│   ├── transit.csv                         # Transit event parameters
│   ├── binary.csv                          # Binary star parameters
│   ├── flare.csv                           # Stellar flare parameters
│   ├── occultation.csv                     # Occultation event parameters
│   ├── supernova.csv                       # Supernova parameters
│   ├── satellite.txt                       # Lateset satellite TLE data (from CelesTrak)
│   └── stray_field.fits or stray_field.npy # Artificially defined stray light field
├── reference_data/                         # Photometric reference data
│   ├── stellar_catalog.fits                # Stellar photometry
│   ├── galaxy_catalog.csv                  # Galaxy photometry
│   └── filter_curves.csv                   # Filter transmission curves
└── output/                                 # Simulation results
    └── [order_target_name]/                # Per-target output directory
        ├── *.fits                          # Simulated images
        └── injected_*.csv                  # Injected variable source parameters
```

## Key Components

### Core Scripts

- **simulator.py**: Main execution logic for the simulation pipeline.
- **local_catalog_screening.py**: Filters catalogs for specific fields of view (FOV) based on observation tasks.


### Configuration Files

- **config.json**: Defines simulator parameters for different classes (see Appendix Table in paper for full parameter list and descriptions)


### Events Directory

- **schedule.csv**: Observation schedule containing:
  - Observation order
  - Target name
  - Celestial coordinates 
  - Number of frames
  - Exposure time
  - Delay between frame
  - Start and end times
  - flat level in photon (if simulate flat image)
  
- **Variable source files**: Define parameters for different types of variable phenomena (see Table in paper of target parameters):
  - Transit events
  - Binary star systems
  - Stellar flares
  - Occultation events
  - Supernova eruptions
  
- **satellite.txt**: Satellite Two-Line Element (TLE) data downloaded from [CelesTrak](https://celestrak.org/).

- **Artificially defined stray light field**: The file format is either FITS or NPy.


### Reference Data

Stores photometric reference data including:
- Stellar catalog
- Galaxy catalog
- Filter transmission curve for different observation systems

**warn**: If stellar catalog is not provided, we can set it as "online" in the configuration file. And then simulator defaults to Gaia DR3 catalog and associated photometry.


### Output Directory

Each scheduled target creates a dedicated subdirectory containing:
- Simulated FITS images
- Corresponding input variable files with injected source information for validation


## Quick Start Guide

### 1. Prepare Reference Data

Provide the following photometric reference data for your observational system:
- Stellar catalog
- Galaxy catalog
- Filter transmission curve

**warn**: Tianyu stellar catalog and gaia galaxy catalog are avilable on [Zenodo](https://zenodo.org/records/18830766). Tianyu filter transmission curve is located in the `reference_data/` directory of this repository.  

### 2. Configure Observation Schedule

Edit `events/schedule.csv` directory of this repository to define your observation sequence:
- Targets must be arranged in **strict chronological order**
- Include target coordinates, exposure times, and observation windows

**warn**: You can directly download the `events/schedule.csv` file from this repository and edit it as needed.

### 3. Configure Variable Sources (Optional)

Edit `events/[specific_variable].csv` directory of this repository to inject specific variable sources:
- Modify parameters in the corresponding variable files (`transit.csv`, `binary.csv`, etc.)
- **Important**: The variable star must exist in your stellar catalog
- **Workaround**: To inject synthetic variables not in the catalog, assign the desired properties to any existing star by using its Gaia DR3 ID in the variable file

**warn**: You can directly download the `events/` files from this repository and edit them as needed.

### 4. Adjust Simulator Parameters

Edit `config.json`:
- Modify only the key parameters required for your use case
- All other parameters can remain at their validated default values
- See Appendix Table for detailed parameter descriptions

**warn**: You can directly download the `config.json` file from this repository and edit it as needed.

### 5. Run the Simulator

```bash
python simulator.py
```

## Important Constraints

⚠️ **Operational Requirements**:

1. **Reference Time**: The simulation reference time (`T_0`) in `config.json` must be **earlier than** the start time of the first scheduled observation

2. **Altitude Constraint**: All scheduled targets must be at an altitude **greater than 25 degrees** at their respective observation times
   - This mirrors real-world telescope safety protocols
   - The simulator will generate a warning if this constraint is violated

## Output

For each target in the schedule, the simulator generates:
- **FITS images**: Synthetic observations with realistic noise and effects
- **Injection logs**: CSV files documenting injected variable sources for validation and comparison

## Documentation

- **Full Parameter List**: See Appendix Table (referenced in config.json)
- **Variable Source Parameters**: See Table of target parameters (referenced in events directory)
- **Directory Structure**: See Figure (folder organization diagram)

## Support

For issues, questions, or contributions, please open an issue on the GitHub repository.

---

**Note**: Ensure all schedule times are properly ordered and altitude constraints are satisfied before running the simulation to avoid warnings and ensure physically realistic outputs.
