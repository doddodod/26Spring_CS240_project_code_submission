# Seam Carving Algorithm Repo Submission

This repository contains the CS240 project implementation for content-aware
image resizing with backward-energy and forward-energy seam carving.

The core dynamic programming seam search, seam backtracking, and seam removal
procedures are implemented directly in Python. The code uses standard numerical
and image I/O libraries, but it does not use a third-party seam carving library,
Numba JIT compilation, or automatic image downsizing for the main experiments.

## Structure

```text
.
├── backward_energy_seam_carving.py   # Backward-energy DP seam carving
├── forward_energy_seam_carving.py    # Forward-energy DP seam carving
├── baselines.py                      # Standard scaling and center-crop baselines
├── experiments.py                    # Batch visual comparison experiment
├── timing.py                         # Runtime and scalability experiment
├── visualization.py                  # Side-by-side comparison images
├── energy_visualization.py           # Export backward/forward energy maps
├── picture/
│   ├── input/                        # Input test images
│   └── masks/                        # Optional masks for extensions
├── results/
│   ├── backward/
│   ├── forward/
│   ├── baseline_resize/
│   ├── baseline_crop/
│   ├── comparisons/
│   └── timing/
├── requirements.txt
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

## Test Images

Input images under `picture/input/` come from the
[andrewdcampbell/seam-carving](https://github.com/andrewdcampbell/seam-carving)
demo set and match the examples in that repository's README:

| File | Typical use |
|------|-------------|
| `castle.jpg` | Vertical seam removal |
| `museum.jpg` | Horizontal seam removal |
| `ratatouille.jpg` | Resize with protective mask |
| `shore.jpg` | Seam insertion / expansion |
| `gotcast.jpg` | Object removal |
| `eiffel.jpg` | Object removal with seam insertion |
| `plane.jpg` | Backward vs forward energy comparison |
| `bench.png` | Backward vs forward energy comparison |

Optional masks for extension experiments live in `picture/masks/`:
`ratatouille_mask.jpg`, `eiffel_mask.jpg`.

## Run Single Algorithms

Backward-energy seam carving:

```bash
python backward_energy_seam_carving.py picture/input/castle.jpg results/backward/castle.jpg --width 500
```

Forward-energy seam carving:

```bash
python forward_energy_seam_carving.py picture/input/plane.jpg results/forward/plane.jpg --width 400
```

Both scripts also support height reduction:

```bash
python backward_energy_seam_carving.py picture/input/museum.jpg results/backward/museum.jpg --width 400 --height 300
```

Targets smaller than the original dimensions trigger seam removal. Targets
larger than the original dimensions trigger seam insertion.

## Run Mask, Insertion, and Object Removal Examples

Protective mask during resizing:

```bash
python forward_energy_seam_carving.py picture/input/ratatouille.jpg results/forward/ratatouille_protected.jpg --width 1500 --mask picture/masks/ratatouille_mask.jpg
```

Seam insertion for image expansion:

```bash
python forward_energy_seam_carving.py picture/input/shore.jpg results/forward/shore_inserted.jpg --width 1900
```

Object removal with a removal mask:

```bash
python forward_energy_seam_carving.py picture/input/eiffel.jpg results/forward/eiffel_removed.jpg --remove-mask picture/masks/eiffel_mask.jpg
```

## Run Visual Comparison Experiment

This command generates standard scaling, center cropping, backward seam carving,
forward seam carving, and one side-by-side comparison image.

```bash
python experiments.py picture/input/bench.png --width 400 --height 300
python experiments.py picture/input/plane.jpg --width 350 --height 250
```

Outputs are saved under `results/`.


## Visualize Energy Maps

Export normalized energy maps (grayscale PNG) for one input image:

```bash
python energy_visualization.py picture/input/bench.png --method both --out-prefix results/energy/bench
```

This generates:

```text
results/energy/bench_backward.png
results/energy/bench_forward.png
```

`forward` visualization uses the cumulative forward-energy DP map (log-compressed)
so high-cost regions are easier to inspect.

## Run Runtime and Scalability Experiment

This command creates multiple square versions of the same input image and
removes a fixed percentage of vertical seams.

```bash
python timing.py picture/input/shore.jpg --sizes 100 200 300 400 --removal-ratio 0.10 --repeats 3
```

The timing table is saved to:

```text
results/timing/runtime.csv
```
