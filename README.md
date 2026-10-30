# Average Cell Mechanical Mapping (avg_cell_mech_mapping)

[![Python 3.10](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-enabled-orange.svg)](https://git-lfs.github.com/)

This repository provides Python pipelines and analysis utilities for processing microrheology datasets and average cell template, and constructing spatial maps of mechanical properties.

---

## Features & Modules

* **Active & Passive Microrheology:** Extract mechanical fingerprint from .tdms and particle-tracking datasets.
* **Average Cell:** Providing average cell template of intracellular structures of micropatterned cells, using optimal transport framework. 
* **Spatial Mapping:** GAM-based analysis of the spatial dependence of the mechanical fingerprint on average-cell coordinates, with nested cross-validation (`Spatial_mapping/`, developed by Sascha Lambert).
* **Workflows:** Interactive Jupyter notebooks (Microrheology.ipynb, Average_cell.ipynb, Spatial_mapping/) demonstrating end-to-end data processing.

---

## Project Structure

> **Dataset Note:** `Example_Data/` contains minimal sample data for testing. The full dataset is available on GRO.data at [https://doi.org/10.25625/I1XIFI](https://doi.org/10.25625/I1XIFI).

```text
avg_cell_mech_mapping/
├── Example_Data/                   # Minimal sample datasets
│   ├── Average_cell/
│   └── Microrheology/
├── src/                            # Modular source code
│   ├── microrheology_analysis_functions.py  # Microrheology data processing
│   ├── average_cell_functions.py            # Average cell calculations
│   ├── gaussian_process_functions.py        # Coordinate transformation & GP regression
│   ├── image_analysis.py                    # Segmentation & image processing
│   ├── image_visualization.py               # Visualizing image data & masks
│   ├── plot_functions.py                    # Plotting microrheology results
│   ├── tdms_functions.py                    # Loader for .tdms files
│   └── utility_functions.py                 # General helper tools
├── Spatial_mapping/                # GAM analysis of the mechanical fingerprint
│   ├── analysis/                   # Notebooks fitting the GAMs (nested CV)
│   ├── plotting/                   # Notebooks plotting the results
│   ├── data/                       # mechanical_fingerprint.npz, actin_nucleus_borders.npz
│   └── output/                     # Precomputed results (.pkl)
├── Average_cell.ipynb              # Workflow notebook for average cell mapping
├── Microrheology.ipynb             # Workflow notebook for microrheology analysis
├── env.yml                         # Micromamba/Conda environment specification
├── requirements.txt                # Pip dependency requirements
└── README.md                       # Project documentation
```
---


## Installation & Setup

### 1. Prerequisites
Ensure **Git LFS** is installed on your system before cloning so that large datasets (`.npz`, `.h5`, `.tdms`) are downloaded properly:
```bash
git lfs install
```

### 2. Clone the Repository
```bash
git clone https://github.com/m-a-eskandari/avg_cell_mech_mapping.git
cd avg_cell_mech_mapping
git lfs pull
```

### 3. Environment Setup (Micromamba)

Create and activate the Conda environment using env.yml, then install additional requirements:

```bash
# Create the conda environment:
micromamba env create -f env.yml

# Activate the conda environment:
micromamba activate avg_cell_mech_mapping

# Install all the requirements:
pip install -r requirements.txt
```

---

## Usage

1. **Microrheology Pipeline:** Open `Microrheology.ipynb` to process microrheology datasets and extract local mechanical fingerprints. This notebook handles probe localization relative to the micropattern, calculates viscoelastic moduli, and maps localized cellular coordinates onto normalized average-cell coordinates.
2. **Average Cell Pipeline:** Open `Average_cell.ipynb` to construct standardized average-cell templates via optimal transport algorithms. This workflow utilizes [MultiScaleOT](https://bernhard-schmitzer.github.io/MultiScaleOT/build/html/index.html) and [LinOT](https://gitlab.gwdg.de/bernhard.schmitzer/linot) to perform density registration and morphological averaging.
3. **Spatial Mapping (GAM analysis):** The notebooks in `Spatial_mapping/` read `data/mechanical_fingerprint.npz` and write their results to `output/` (precomputed results are included). Run the notebooks in `analysis/` (about 30-40 min each on 4 CPU cores; `logA x partitions` is quick and must run before `plotting/partition_plots.ipynb`), then the notebooks in `plotting/`.
---
