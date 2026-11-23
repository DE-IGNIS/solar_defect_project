Step 1: Fork and clone
On GitHub, teammates should open your repository and click Fork.

Then:
git clone https://github.com/TEAM-USERNAME/solar_defect_project.git

cd solar_defect_project


Replace TEAM-USERNAME with the fork owner's GitHub username.
Step 2: Create a Python environment
Ubuntu/Linux:
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip


Step 3: Install dependencies
Create requirements.txt with the project's non-PyTorch dependencies, for example:
numpy
pandas
matplotlib
opencv-python
scikit-learn
pillow
pvlib


Install them:
python -m pip install -r requirements.txt


Install PyTorch and torchvision separately using the official PyTorch installation selector, according to the teammate's operating system and GPU/CUDA requirements:
PyTorch installation instructionsThis avoids forcing your particular CUDA build onto teammates with different hardware.
Step 4: Download the dataset
The dataset is publicly available from:
Raptor Maps – InfraredSolarModulesTeammates should download and extract it into the directory expected by src/config.py.

Then verify:
- module_metadata.json exists.
- The images directory exists.
- The image paths in the CSV splits resolve correctly.
- There are 20,000 images and 12 classes.
They should use your committed CSV splits rather than generating new ones, so their experiments remain comparable.
Step 5: Test the trained model
Once the dataset and checkpoints are available:
python -m src.evaluate


This runs evaluation on the test split and generates classification reports and confusion matrices.

To regenerate training plots:
python -m src.plot_training


Step 6: Train models themselves (optional)
If teammates want to reproduce the training:
python -m src.train


This trains both RAW and CLAHE models. They should expect different training times depending on their hardware.

For a CPU-only machine, training may be considerably slower.
Step 7: Run the energy-loss estimator
python -m src.loss_estimator


This generates the energy-loss CSV and visualization using the project's assumptions and pvlib calculations.