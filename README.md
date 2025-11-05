# 🧠 MNTB Principal Neuron Model – NEURON Simulation (Python)

This repository contains a NEURON-based simulation of principal neurons (PN) in the 
Medial Nucleus of the Trapezoid Body (MNTB), developed for understanding intrinsic properties 
and responses after blocking the spontaneous activity. The model is built using Python and `.mod` files.
The optimizers used from SciPy library were differential_evolution and minimize. The folders contain only
python and mod files necessary to do the simulation. 
####
PN file: ~/optimization/MNTB_PN_fit.py.
####
Steady-state fitting file: ~/optimization/fit_passive_v2.py
####
AP fitting files: ~/optimization/fit_AP_v2_iMNTB.py & fit_AP_v2_TeNT.py
####
Simulation of the current clamp simulation: ~/optimization/fit_simulation.py
####
Other files were used to analyze and plot routine. 

---
## 🛠 Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:NikollasBenites/MNTB_NEURON_model.git
cd MNTB_NEURON_model
```
### 2. Create the Conda Environment

There are two envs files: one for mac and other for windows.

``` bash
conda env create -f environment_(YOUR_OS).yml
conda activate neuron_env
```
### 3. Compile NEURON Mechanisms

Make sure you're in the root project directory (Mac and Windows) using Terminal:

```bash
nrnivmodl mod/
```
This will generate the arm64/ folder with compiled special (in Mac).
In Windows OS you will generate a file nrnmech.dll in the Root.


# 👥 Collaboration Workflow
🧪 Recommended Git Practice
## Before working
```bash
git pull origin main
```
## After making changes
```bash
git add .
git commit -m "Describe your change"
git push origin main
```
Use branches for feature development or testing:
```bash
git checkout -b feature/new-analysis
```

📦 Reproducing the Environment
If the environment ever changes:
``` bash
conda env export --no-builds | grep -v "prefix:" > environment.yml
git commit -am "Update environment with new packages"
git push
```

# 📚 References
NEURON Simulation Environment – https://neuron.yale.edu/neuron
Heller et al., 202X (in preparation)

# 👤 Code adapted by
Nikollas Benites, University of South Florida

Daniel Heller, University of South Florida

# 📝 License

This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
https://creativecommons.org/licenses/by-nc-nd/4.0/

---

