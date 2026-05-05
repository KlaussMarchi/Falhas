# Antigravity Project Context

## Project Overview

The Antigravity project is a Data Science and Machine Learning codebase focused on the processing, formatting, and visualization of 3D synthetic seismic data and fault masks. 

The project manages multiple datasets (e.g., `dataset0`, `dataset1`), working primarily with `.npy` files representing 3D arrays. It includes tools for loading volumes, extracting 2D slices (inline, crossline, depth), and generating comparative visual reports in PDF format. 

**Main Technologies:**
* Python
* NumPy
* Matplotlib
* Jupyter Notebooks

**Architecture:**
* `Dataset/`: The core directory containing datasets and processing logic.
    * `dataset0/`, `dataset1/`: Individual dataset folders containing original images, masks, and partitioned tiles, as well as `Format.ipynb` notebooks for localized formatting.
    * `index.py`: The primary script for volume loading, 2D slicing, and report generation.
    * `DataBase.csv`: A catalog mapping image paths to mask paths and shapes for tile datasets.
* `Synthetic/`: Contains additional tools and Jupyter notebooks like `Format.ipynb` for synthetic data manipulation.

## Building and Running

The primary executable artifact found so far is the report generator.

**Generate PDF Report:**
To generate a comparative PDF report of the datasets (`output.pdf`), run the index script within the `Dataset` directory:
```bash
python Dataset/index.py
```

**Data Formatting:**
Jupyter notebooks (`Format.ipynb`) are provided in the `Synthetic` and individual dataset directories. These can be run via a Jupyter environment to inspect or re-format the datasets:
```bash
jupyter notebook
```
*(Open `Synthetic/Format.ipynb` or `Dataset/dataset0/Format.ipynb`)*

## Development Conventions

* **Object-Oriented Design:** The Python code in `index.py` follows an object-oriented paradigm, utilizing classes like `VolumeLoader`, `Slicer`, and `PDFReportGenerator` to encapsulate specific responsibilities.
* **Data Storage:** Uses NumPy binary files (`.npy`) for storing 3D volumetric data, which is standard for performance and efficiency in numerical computation.
* **Data Organization:** Datasets are logically separated into subdirectories containing images and masks (e.g., `dataset0/images/`, `dataset0/masks/`). Tile-based data is tracked using a central `DataBase.csv` file.