# Average Cell Mechanical Mapping (avg_cell_mech_mapping)

[![Python 3.10](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-enabled-orange.svg)](https://git-lfs.github.com/)

This repository provides Python pipelines and analysis utilities for processing microrheology datasets and average cell template, and constructing spatial maps of mechanical properties.

---

## Features & Modules

* **Active & Passive Microrheology:** Extract mechanical fingerprint from .tdms and particle-tracking datasets.
* **Average Cell:** Providing average cell template of intracellular structures of micropatterned cells, using optimal transport framework. 
* **Spatial Mapping:** GAM-based analysis of the spatial dependence of the mechanical fingerprint on average-cell coordinates, with nested cross-validation (`Spatial_mapping/`).
* **Workflows:** Interactive Jupyter notebooks (Microrheology.ipynb, Average_cell.ipynb, Spatial_mapping/) demonstrating end-to-end data processing.

---

## Project Structure
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

### Dataset Note 
The `Example_Data/` directory contains minimal sample data for testing code functionality. The complete dataset is hosted on GRO.data at [doi.org/10.25625/I1XIFI](https://doi.org/10.25625/I1XIFI).

### Data Mapping to Paper Figures

The mapping below links each figure in our publication, **["Structural order emerges without mechanical order inside living cells"](https://doi.org/10.64898/2026.10.15.759452)**, to its corresponding dataset files and indices in the full repository:

#### Main Text Figures

<table>
  <thead>
    <tr>
      <th align="left">Figure</th>
      <th align="left">Source File / Path</th>
      <th align="left">Notes / Specific Indices</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td nowrap="nowrap"><strong>Fig.&nbsp;1b</strong></td>
      <td><code>Average_cell/data/datasets.npz</code></td>
      <td><code>["dataset4"][10]</code></td>
    </tr>
    <tr>
      <td nowrap="nowrap"><strong>Fig.&nbsp;2a</strong></td>
      <td><code>Average_cell/data/structures_images.npz</code></td>
      <td>N=10 (sampled with <code>random.seed(10)</code>); N=437 (all actin images)</td>
    </tr>
    <tr>
      <td nowrap="nowrap"><strong>Fig.&nbsp;2d</strong></td>
      <td><code>Average_cell/data_processed/Average_cell.tif</code></td>
      <td></td>
    </tr>
    <tr>
      <td nowrap="nowrap"><strong>Fig.&nbsp;2e</strong></td>
      <td><code>Average_cell/data_processed/Actin/actin_pca_dist.npy</code></td>
      <td></td>
    </tr>
    <tr>
      <td nowrap="nowrap"><strong>Fig.&nbsp;3a,&nbsp;d</strong></td>
      <td><code>Mechanical_fingerprint/Mechanical_fingerprint.npz</code></td>
      <td></td>
    </tr>
    <tr>
      <td nowrap="nowrap"><strong>Fig.&nbsp;3b,&nbsp;c</strong></td>
      <td><code>Microrheology/RPE1/2026-02-25_eLabID_2264/Cell 3/particle 1/Active_microrheology_0003/results.h5</code></td>
      <td></td>
    </tr>
    <tr>
      <td nowrap="nowrap"><strong>Fig.&nbsp;5a</strong></td>
      <td><code>Average_cell/data/structures_images.npz</code></td>
      <td></td>
    </tr>
    <tr>
      <td nowrap="nowrap"><strong>Fig.&nbsp;5b</strong></td>
      <td><code>Microrheology/Actin/</code></td>
      <td></td>
    </tr>
    <tr>
      <td nowrap="nowrap"><strong>Fig.&nbsp;6a,&nbsp;b</strong></td>
      <td><code>Mechanical_fingerprint/Mechanical_fingerprint.npz</code></td>
      <td></td>
    </tr>
    <tr>
      <td nowrap="nowrap"><strong>Fig.&nbsp;6c</strong></td>
      <td><code>Average_cell/data/structures_images.npz</code></td>
      <td><code>["actin"][0, 51, 65, 66]</code>, <code>["microtubules"][0, 1, 2, 3]</code>, <code>["endoplasmic reticulum"][0, 1, 2, 3]</code></td>
    </tr>
  </tbody>
</table>

---

#### SI Figures

<table>
  <thead>
    <tr>
      <th align="left">Figure</th>
      <th align="left">Source File / Path</th>
      <th align="left">Notes / Specific Indices</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3" valign="top" nowrap="nowrap"><strong>SI&nbsp;Fig.&nbsp;1</strong></td>
      <td><strong>Golgi &amp; ER:</strong> <code>Average_cell/data/datasets.npz</code></td>
      <td><code>["dataset5"][14]</code>, <code>["dataset4"][14]</code></td>
    </tr>
    <tr>
      <td><strong>Others:</strong> <code>Average_cell/data/structures_images.npz</code></td>
      <td><code>["nucleus"][0]</code>, <code>["actin"][100]</code>, <code>["microtubules"][9]</code></td>
    </tr>
    <tr>
      <td><strong>Average Cell:</strong> <code>Average_cell/data_processed/Average_cell.tif</code></td>
      <td></td>
    </tr>
    <tr>
      <td nowrap="nowrap"><strong>SI&nbsp;Fig.&nbsp;2</strong></td>
      <td><code>Average_cell/data_processed/[structure_name]/[structure_name]_pca_dist.npy</code></td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2" valign="top" nowrap="nowrap"><strong>SI&nbsp;Fig.&nbsp;3</strong></td>
      <td><strong>Active:</strong> <code>Microrheology/RPE1/2026-02-23_eLabID_2253/Cell 0/particle 0/Active_microrheology_0004/</code></td>
      <td> <code>Active_microrheology.tdms</code> &amp; <code>results.h5</code></td>
    </tr>
    <tr>
      <td><strong>Passive:</strong> <code>Microrheology/RPE1/2026-02-23_eLabID_2253/Cell 0/particle 0/Passive_microrheology_0004/</code></td>
      <td><code>Passive_microrheology.tdms</code></td>
    </tr>
    <tr>
      <td nowrap="nowrap"><strong>SI&nbsp;Fig.&nbsp;4</strong></td>
      <td><code>Microrheology/Hydrogel/</code></td>
      <td></td>
    </tr>
    <tr>
      <td nowrap="nowrap"><strong>SI&nbsp;Fig.&nbsp;5</strong></td>
      <td><code>Mechanical_fingerprint/Mechanical_fingerprint.npz</code></td>
      <td></td>
    </tr>
    <tr>
      <td nowrap="nowrap"><strong>SI&nbsp;Fig.&nbsp;6</strong></td>
      <td><code>Mechanical_fingerprint/Mechanical_fingerprint.npz</code></td>
      <td></td>
    </tr>
    <tr>
      <td nowrap="nowrap"><strong>SI&nbsp;Fig.&nbsp;10</strong></td>
      <td><code>Mechanical_fingerprint/Mechanical_fingerprint.npz</code></td>
      <td></td>
    </tr>
    <tr>
      <td nowrap="nowrap"><strong>SI&nbsp;Fig.&nbsp;11</strong></td>
      <td><code>Mechanical_fingerprint/Mechanical_fingerprint.npz</code></td>
      <td></td>
    </tr>
  </tbody>
</table>



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
