# Seam Carving Algorithm Repository

This repository contains the CS240 project implementation for content-aware
image resizing with seam carving. It includes the basic backward-energy and
forward-energy implementations, three acceleration extensions, and the scripts
used to generate the experimental figures and timing data in the report.

The implementation is written in Python with NumPy and Pillow. The seam-search
dynamic programs, seam backtracking, seam removal, seam insertion, masking, and
extension methods are implemented directly in this repository.

## Running Examples

Install dependencies:

Use Python 3.9.7
```bash
pip install -r requirements.txt
```

Run backward-energy seam carving:

```bash
python backward_energy_seam_carving.py picture/input/castle.jpg results/backward/castle.jpg --width 500
```

Run forward-energy seam carving:

```bash
python forward_energy_seam_carving.py picture/input/bench.png results/forward/bench.jpg --width 400
```

Run the combined scalability experiment:

```bash
python combined_methods_experiment.py
```

Run scalability experiments:

```bash
# Main backward-vs-forward runtime scalability experiment as shown in report.
python runtime_scalability_experiment.py

# Extension comparison: runnning time comparion of backward, forward, batch, local DP, and multiscale as shown in report.
python combined_methods_experiment.py
```

The generated CSV files and plots are saved under `results/timing/`.


## Directory Structure

```text
.
├── picture/                         # Testable input images and masks
├── results/                         # Generated images, plots, and experiment data
├── backward_energy_seam_carving.py
├── forward_energy_seam_carving.py
├── batch_seam_carving.py
├── local_dp_update_carving.py
├── multiscale_carving.py
├── *_experiment.py                  # Experiment scripts
├── baselines.py
├── visualization.py
├── energy_visualization.py
├── requirements.txt
└── README.md
```

### `picture/`

`picture/` stores images that can be used for testing the algorithms. In
particular:

- `picture/input/` contains input images such as `bench.png`, `castle.jpg`,
  `museum.jpg`, `shore.jpg`, `ratatouille.jpg`, and `eiffel.jpg`.
- `picture/masks/` contains optional masks for protection or object removal
  experiments.

### `results/`

`results/` stores all generated image outputs, comparison panels, plots, and
experiment data. Important subdirectories include:

- `results/poster_demos/`: visual examples used for qualitative comparison and
  poster figures.
- `results/energy/`: backward/forward energy-map visualizations.
- `results/failure/`: failure-case comparison images.
- `results/timing/`: all scalability-experiment inputs, CSV files, runtime
  plots, and extension experiment results.

The `results/timing/` directory is especially important. It contains:

- prepared scalability inputs under `results/timing/inputs/`;
- main backward/forward timing data and plots;
- batch seam extraction timing data and plots;
- local update / local DP timing data and plots;
- multiscale timing data, quality comparisons, and plots;
- the combined five-method runtime comparison.

## Code Organization

### Basic Implementation

- `baselines.py`  
  Implements ordinary resizing baselines such as scaling and center cropping.

- `backward_energy_seam_carving.py`  
  Implements the standard backward-energy seam carving pipeline. It includes
  Sobel energy computation, vertical seam DP, seam removal, seam insertion,
  protective masks, removal masks, width/height resizing, and object removal.

- `forward_energy_seam_carving.py`  
  Implements forward-energy seam carving. It uses transition-dependent costs to
  estimate the disruption introduced after seam removal, while keeping the same
  DP/backtracking structure.

### Extension Methods

- `batch_seam_carving.py`  
  Implements batch seam extraction. It computes one energy map for the current
  image, repeatedly extracts several non-overlapping seams from a penalized
  working energy map, and removes the batch together.

- `local_dp_update_carving.py`  
  Implements local DP update. It reuses the previous cumulative DP table and
  parent table, recomputing only a conservative affected region around the
  removed seam. It also includes correctness validation utilities.

- `local_update_carving.py`  
  Implements local energy-map update. It recomputes Sobel energy only in a band
  around the removed seam while still running global DP.

- `multiscale_carving.py`  
  Implements coarse-to-fine multiscale seam carving. It finds a coarse seam on a
  downsampled image, maps the seam back to full resolution, and refines it inside
  a narrow band.

### Experiment Scripts

- `runtime_scalability_experiment.py`  
  Runs the main backward-energy vs forward-energy scalability experiment.

- `combined_methods_experiment.py`  
  Runs the unified five-method scalability comparison: backward, forward, batch
  seam extraction, local DP update, and multiscale seam carving.

- `batch_scalability_experiment.py`  
  Runs scalability experiments for batch seam extraction with different batch
  sizes.

- `local_update_experiment.py`  
  Runs scalability experiments for local energy-map update.

- `validate_local_dp.py`  
  Validates local DP correctness and speedup under different affected-region
  bandwidths.

- `multiscale_experiment.py`  
  Runs scalability and quality-comparison experiments for multiscale seam
  carving.

- `poster_baseline_comparisons.py`  
  Generates baseline comparison figures for poster/report demos.

- `poster_application_demos.py`  
  Generates application-level demo figures such as protection, insertion, and
  object removal.

- `bench_pipeline_visualization.py`  
  Generates pipeline-style visualizations for the `bench.png` example.

### Other Utility Code

- `experiments.py`  
  Visual-comparison script for running scaling, cropping, backward seam
  carving, and forward seam carving on one image.

- `timing.py`  
  Timing script for simple runtime measurements.

- `visualization.py`  
  Helper functions for side-by-side comparison images.

- `energy_visualization.py`  
  Exports normalized backward/forward energy-map visualizations.
