# FusionNet-PD-Handwriting

Dual-branch Bidirectional Cross-Attention Transformer (FusionNet) for early Parkinson's disease screening from handwriting images (spatial + FFT frequency-domain representations), evaluated on the NewHandPD dataset (Circle, Meander, and Spiral tracing tasks).


## Overview

FusionNet combines two Transformer-based branches — a spatial-domain branch (SpatialNet) and a frequency-domain branch (FreqNet, operating on 2D-FFT amplitude images) — via bidirectional cross-attention. Training follows a three-stage progressive procedure: 
(1) SpatialNet trained independently,
(2) FreqNet trained independently, 
(3) both branches frozen and fused via cross-attention and a final Transformer block.

## Requirements


## Environment configuration

Data and output paths are configured via environment variables (see `Opt` class in `kfold_combine.py`), so no hardcoded local paths need to be edited:

```bash
export PD_DATA_ROOT=/path/to/your/data       # where your dataset CSVs live
export PD_OUTPUT_ROOT=/path/to/your/outputs  # where checkpoints/logs are written
python kfold_combine.py
```

If unset, these default to `./data` and `./outputs` relative to the
current working directory.

## Dataset

Experiments use the **NewHandPD** dataset (Circle, Meander, Spiral tracing tasks). Due to participant privacy, raw handwriting images are **not** redistributed in this repository.

### Expected data layout

Place your dataset CSVs under `$PD_DATA_ROOT/csv/`. `kfold_combine.py` expects one CSV per data source (original + augmented images), for both the spatial-domain (normal) and frequency-domain (FFT) branches:

```
$PD_DATA_ROOT/
└── csv/
    ├── Original_Circle.csv              # spatial, original (non-augmented) images
    ├── PatientCircle_Aug24_Normal.csv   # spatial, augmented (PD patients)
    ├── HealthyCircle_Aug24_Normal.csv   # spatial, augmented (healthy controls)
    ├── Original_Circle_fft.csv          # FFT, original (non-augmented) images
    ├── PatientCircle_Aug24_Amplitude.csv
    └── HealthyCircle_Aug24_Amplitude.csv
```

Update the `train_csv_normal_list` / `train_csv_fft_list` lists in `Opt` (inside `kfold_combine.py`) to point to your own CSVs if the filenames/task differ.

### CSV format

Each CSV (read by `ImageListIter` in `image_iterator.py`) must have these columns:

```
ID,Image,ClassID
1,/absolute/or/relative/path/to/image.jpg,0
```

- `ID`: integer row identifier
- `Image`: path to the image file (spatial or FFT-amplitude, matching
  which CSV it's listed in)
- `ClassID`: `0` for healthy control, `1` for PD patient

Filenames are also used for two things internally, so naming matters:
- **Subject/drawing-instance grouping** (`kfold_combine.py` parses the substring before the first underscore as the group ID for GroupKFold)
- **Augmentation detection** (any filename containing `ang`, `aug`, or `flip` is treated as an augmented sample; training folds use augmented images only, validation folds use original images only)


