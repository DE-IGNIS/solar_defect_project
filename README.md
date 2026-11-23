## Getting Started

### 1. Fork and Clone the Repository

1. Open this repository on GitHub and click **Fork**.
2. Clone your fork and navigate into the project directory:

```bash
git clone [https://github.com/TEAM-USERNAME/solar_defect_project.git](https://github.com/TEAM-USERNAME/solar_defect_project.git)
cd solar_defect_project
```

> Replace `TEAM-USERNAME` with your GitHub username.

---

### 2. Create a Python Environment

**Ubuntu / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

---

### 3. Install Dependencies

Ensure `requirements.txt` contains the project's non-PyTorch dependencies:

```text
numpy
pandas
matplotlib
opencv-python
scikit-learn
pillow
pvlib
```

Install them with:

```bash
python -m pip install -r requirements.txt
```

Install **PyTorch** and **torchvision** separately using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) based on your operating system and GPU/CUDA setup. This avoids forcing a specific CUDA build onto incompatible hardware.

---

### 4. Download the Dataset

Download the dataset from [Raptor Maps – InfraredSolarModules](https://github.com/RaptorMaps/InfraredSolarModules) and extract it into the directory configured in `src/config.py`.

Verify the following before running any scripts:
- [ ] `module_metadata.json` exists.
- [ ] The `images` directory exists.
- [ ] Image paths in the CSV splits resolve correctly.
- [ ] The dataset contains **20,000 images** across **12 classes**.

> **Note:** Use the committed CSV splits rather than generating new ones so that all experiments remain directly comparable.

---

### 5. Evaluate the Trained Model

Once the dataset and model checkpoints are in place, run evaluation on the test split to generate classification reports and confusion matrices:

```bash
python -m src.evaluate
```

To regenerate the training plots:

```bash
python -m src.plot_training
```

---

### 6. Train the Models (Optional)

To reproduce training from scratch:

```bash
python -m src.train
```

This trains both the **RAW** and **CLAHE** models. Training time depends on your hardware and will be considerably slower on CPU-only machines.

---

### 7. Run the Energy-Loss Estimator

Generate the energy-loss CSV and visualizations using the project's assumptions and `pvlib` calculations:

```bash
python -m src.loss_estimator
```