# 3D Point Cloud Classification with Spatial Hashing

This project implements a memory-efficient 3D point cloud classifier utilizing single-resolution spatial hashing. It compresses a 32³ binary voxel grid into a fixed-size 1D hash table, followed by classification using a lightweight Multi-Layer Perceptron (MLP). The methodology is evaluated on a 10-class ModelNet subset against a dense 3D CNN baseline.

## Repository Structure
## Repository Structure

- `data/`
  - `processed/`
  - `raw/ModelNet40/header_fix/`
- `experiment_results/`
- `src/`
  - `__init__.py`
  - `data/`
    - `__init__.py`
    - `dataset.py`
    - `dense_dataset.py`
    - `off_to_pointcloudv1.py`
    - `off_to_pointcloudv2.py`
    - `voxelization.py`
  - `experiments/`
    - `__init__.py`
    - `plot_results.py`
    - `run_hash_sweep.py`
  - `models/`
    - `__init__.py`
    - `cnn_3d_baseline.py`
    - `hash_encoder.py`
    - `hash_model.py`
    - `mlp_classifier.py`
  - `training/`
    - `__init__.py`
    - `metrics_logger.py`
    - `train_cnn.py`
    - `train_hash.py`
  - `utils/`
    - `__init__.py`
    - `config.py`
    - `hash_function.py`
- `.gitignore`
- `requirements.txt`
- `run_cnn.py`
- `test_cnn.py`
- `test_collision.py`
- `test_dataset.py`
- `test_models.py`
- `voxel_visual.py`
- `voxel_visualv2.py`

## Dataset

We use a 10-class subset of ModelNet40 (airplane, car, chair, table, sofa, bed, bookshelf, desk, dresser, monitor). The dataset can be downloaded from [Princeton ModelNet](https://modelnet.cs.princeton.edu/).

> Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang and J. Xiao.  
> *3D ShapeNets: A Deep Representation for Volumetric Shapes*.  
> CVPR 2015.

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

## Prepare the Data

1. Download and extract ModelNet40.zip.
2. Place the extracted folder as `data/raw/ModelNet40/` so that each category appears as a subdirectory, e.g. `data/raw/ModelNet40/airplane/`, `data/raw/ModelNet40/car/`, etc.
3. Run the preprocessing script:

```bash
python src/data/off_to_pointcloudv2.py
```

This samples 2048 points per mesh, voxelizes into a 32³ binary grid, extracts occupied coordinates, and precomputes hash indices. Processed data is saved in `data/processed/`.

## Run Experiments

After configuring the desired parameters in the source files, execute the following:
Run hash table size sweep (T = 256, 4096, 65536, 1048576):

```bash
python src/experiments/run_hash_sweep.py
```

Run CNN baseline:

```bash
python run_cnn.py
```

## Plot Results

Generate accuracy, memory, and collision rate plots:

```bash
python src/experiments/plot_results.py
```

Results are saved in `experiment_results/`.

## Results Summary

| Method               | Memory (MB) | Accuracy (%) | Collision Rate (%) |
|----------------------|-------------|--------------|--------------------|
| 3D CNN Baseline      | 4.52        | 95.7         | 0.0                |
| Hash T = 1048576     | 32.00       | 85.8         | 0.0                |
| Hash T = 65536       | 2.00        | 82.8         | 0.5                |
| Hash T = 4096        | 0.12        | 85.3         | 11.8               |
| Hash T = 256         | 0.01        | 44.2         | 74.3               |

The spatial hash model at T=4096 reduces memory by 37.6x compared to the dense CNN baseline, with only a 10.4% drop in accuracy. Aggressive compression to T=256 leads to a 74.3% collision rate and collapses accuracy to near-random (44.2%). Moderate collisions (12%) are well tolerated, confirming that spatial hashing is a viable compression strategy for memory-constrained edge devices.
```
