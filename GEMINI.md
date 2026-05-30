# Project Overview
This repository contains a PyTorch-based machine learning pipeline designed for **3D semantic segmentation of seismic faults**. It leverages advanced 3D network architectures and incorporates a robust synthetic data generation engine to simulate complex geological features (stratigraphy, folding, shearing, faulting, and seismic wavelets) for training models when real data is scarce.

### Key Technologies
- **Python 3**
- **PyTorch** & **TorchVision**
- **MONAI** (Medical Open Network for AI, utilized for its powerful 3D architectures like SegResNet, VNet, UNet3D)
- **TorchMetrics**
- **Jupyter Notebooks** & **Papermill** (for task orchestration and pipeline execution)
- **Scipy**, **Numpy**, **Pandas**, **Albumentations**

### Architecture & Directory Structure
- `Dataset/`: Stores raw dataset files (e.g., `.dat` formats) and data formatting pipelines (`Format.ipynb`) to prepare the data for training. Currently, the workflow typically uses `dataset2` (as per the README) or orchestrated datasets.
- `Model/`: Contains the core logic for neural network training and inference.
  - `Analysis.ipynb`, `Predict.ipynb`, `PostProc.ipynb`: Main notebooks for model training, prediction, and post-processing evaluation.
  - `Network/`: Defines 3D model architectures, including standard UNet3D, MONAI networks, and specialized variants like `ResACEUnet`, `ResACEUnetWu`, and `Unet3D_V2`.
  - `Losses/`: Custom loss functions (e.g., Jaccard/IoU) for binary and multiclass segmentation.
  - `utils/`: Auxiliary functions and plotting tools.
- `Synthetic/`: A procedural 3D seismic data generator. It simulates geological formations by applying reflectivity, folding, shearing, and faulting, alongside wavelet convolutions to output realistic 3D seismic volumes (`images`) and corresponding fault masks (`masks`).
- `Searcher/`: Contains notebooks (`Analysis.ipynb`, `Table.ipynb`) geared towards hyperparameter search, experimental tracking, and result tabulation.
- `Task/`: Pipeline orchestrator. Uses Python and `papermill` (`Task/index.py`) to programmatically run a sequence of Jupyter notebooks (e.g., Data Formatting followed by Model Analysis) based on a configured task list (e.g., `task.json` / `info.json`).

## Building and Running

### Environment Setup
The project requires a specific conda environment equipped with GPU-accelerated PyTorch and Jupyter kernel support.
```bash
# Example setup
conda create -n torch-gpu python=3.9
conda activate torch-gpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install monai torchmetrics albumentations papermill scipy jupyter pandas tqdm
# Install the kernel for papermill and Jupyter
python -m ipykernel install --user --name=torch-gpu
```
*(Note: The environment name `torch-gpu` is expected by the papermill orchestration script in `Task/index.py`)*

### Executing the Pipeline
The preferred method of execution is through the automated task orchestrator rather than manually running notebooks.

1. Configure your tasks in a JSON file (e.g., `task.json`).
2. Run the orchestrator:
   ```bash
   cd Task
   conda activate torch-gpu
   python index.py
   ```
   This script will sequentially execute notebooks like `../Dataset/dataset0/Format.ipynb` and `../Model/Analysis.ipynb`, saving the execution logs and evaluated output notebooks into `Task/logs/`.

### Synthetic Data Generation
To generate synthetic data without running the full pipeline:
1. Open or run `Synthetic/Generate.ipynb`.
2. Alternatively, use the `SyntheticGenerator` class in `Synthetic/index.py` to programmatically generate structural volumes.

## Development Conventions
- **Modular Notebooks**: The workflow is heavily modularized into distinct Jupyter Notebooks for data formatting, model training, and analysis. Always ensure notebooks are linearly executable from top to bottom so they can be seamlessly run via `papermill`.
- **Object-Oriented Utilities**: Auxiliary scripts, model structures, and data augmenters are encapsulated within Python classes (e.g., `ModelNetwork`, `SyntheticGenerator`) following OOP principles.
- **Seeding and Reproducibility**: When generating synthetic data or setting up data loaders, deterministic seeds are strictly used (as seen in the multiprocessing wrappers) to ensure reproducible datasets and experiments.
- **Data Pipeline**: The raw input files are processed and output as `.npy` arrays, which are then consumed by the DataLoader for optimized reading during 3D model training.
