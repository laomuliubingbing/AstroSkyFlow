# Astronomical Observation Simulator

## Overview

This simulator generates synthetic astronomical observations with configurable parameters, variable sources, and realistic imaging effects. It is designed with a modular architecture to facilitate customization and reproducibility.

## Directory Structure

The simulator follows a modular organization separating input configurations, reference data, and simulation outputs:

```
simulator/
├── simulator.py                    # Main execution script
├── local_catalog_screening.py      # FOV-specific catalog filtering
├── config.json                     # Simulator parameters configuration
├── events/                         # Observation schedules and variable sources
│   ├── schedule.csv               # Observation targets and timeline
│   ├── transit.csv                # Transit event parameters
│   ├── binary.csv                 # Binary star parameters
│   ├── flare.csv                  # Stellar flare parameters
│   ├── occultation.csv            # Occultation event parameters
│   ├── supernova.csv              # Supernova parameters
│   └── satellite.txt              # Satellite TLE data (from CelesTrak)
├── reference_data/                 # Photometric reference data
│   ├── stellar_catalog/           # Stellar photometry
│   ├── galaxy_catalog/            # Galaxy photometry
│   └── filter_curves/             # Filter transmission curves
└── output/                         # Simulation results
    └── [target_name]/             # Per-target output directory
        ├── *.fits                 # Simulated images
        └── injected_*.csv         # Injected variable source parameters
```

## Key Components

### Core Scripts

- **simulator.py**: Main execution logic for the simulation pipeline
- **local_catalog_screening.py**: Filters catalogs for specific fields of view (FOV) based on observation tasks

### Configuration Files

- **config.json**: Defines simulator parameters for different classes (see Appendix Table for full parameter list and descriptions)

### Events Directory

- **schedule.csv**: Observation schedule containing:
  - Target names
  - Celestial coordinates (RA, Dec)
  - Number of frames
  - Exposure time
  - Start and end times
  
- **Variable source files**: Define parameters for different types of variable phenomena (see Table of target parameters):
  - Transit events
  - Binary star systems
  - Stellar flares
  - Occultation events
  - Supernova eruptions
  
- **satellite.txt**: Satellite Two-Line Element (TLE) data downloaded from [CelesTrak](https://celestrak.org/)

### Reference Data Directory

Stores photometric reference data including:
- Stellar catalogs
- Galaxy catalogs
- Filter transmission curves for different observation systems

**Default**: If reference data is not provided, the simulator defaults to Gaia DR3 catalog and associated photometry.

### Output Directory

Each scheduled target creates a dedicated subdirectory containing:
- Simulated FITS images
- Corresponding input variable files with injected source information for validation

## Quick Start Guide

### 1. Prepare Reference Data

Provide the following photometric reference data for your observational system:
- Stellar catalog
- Galaxy catalog
- Filter transmission curves

If these files are unavailable, the simulator will use Gaia DR3 data by default.

### 2. Configure Observation Schedule

Edit `events/schedule.csv` to define your observation sequence:
- Targets must be arranged in **strict chronological order**
- Include target coordinates, exposure times, and observation windows

### 3. Configure Variable Sources (Optional)

To inject specific variable sources:
- Modify parameters in the corresponding variable files (`transit.csv`, `binary.csv`, etc.)
- **Important**: The variable star must exist in your stellar catalog
- **Workaround**: To inject synthetic variables not in the catalog, assign the desired properties to any existing star by using its Gaia DR3 ID in the variable file

### 4. Adjust Simulator Parameters

Edit `config.json`:
- Modify only the key parameters required for your use case
- All other parameters can remain at their validated default values
- See Appendix Table for detailed parameter descriptions

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
