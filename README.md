# MICrONs Dataset — Structural Connectivity Analysis

Exploratory analysis of the [MICrONs public dataset](https://www.microns-explorer.org/) (`minnie65_public`) to study synaptic connectivity and laminar organization in mouse visual cortex (V1).

## Project Structure

```
MICrONs-dataset/
├── scripts/
│   ├── 01_setup/          # CAVE client initialization & authentication
│   ├── 02_download/       # Download proofread neurons & synapse data (parallel)
│   ├── 03_matrix/         # Build raw N×N synaptic connectivity matrix W
│   ├── 04_layers/         # Classify neurons by cortical layer (L2/3–L6), sort W
│   └── 05_analysis/       # Coupling alignment (W·Wᵀ) & shared input (Wᵀ·W) analysis
├── outputs/               # Generated figures (PNG)
├── data/                  # Exported data files (CSV)
└── cache/                 # Local cache for downloaded data (pkl / npy, git-ignored)
```

## Pipeline

1. **Setup** (`scripts/01_setup/`) — Initialize CAVEclient with `minnie65_public` dataset, materialization version 1621.
2. **Download** (`scripts/02_download/`) — Fetch morphologically proofread neurons and functionally co-registered neurons; parallel-download all synapse records.
3. **Matrix** (`scripts/03_matrix/`) — Build the recurrent connectivity weight matrix W (N×N synapse counts).
4. **Layer classification** (`scripts/04_layers/`) — Map neurons to cortical layers (L2/3, L4, L5, L6) using the AIBS deep-learning metamodel; sort and export W.
5. **Analysis** (`scripts/05_analysis/`) — Compute coupling alignment and shared input matrices; generate final figures.

## Key Technologies

- `caveclient` — Access to MICrONs CAVE database
- `numpy` / `pandas` — Data processing
- `matplotlib` / `seaborn` — Visualization
- `scipy` — Statistical analysis
- `concurrent.futures` — Parallel data download
