# 🧠 MNTB Principal Neuron Model – NEURON Simulation (Python)

This repository contains a NEURON-based simulation of principal neurons (PN) in the 
Medial Nucleus of the Trapezoid Body (MNTB), developed for understanding intrinsic properties 
and responses after blocking the spontaneous activity. The model is built using Python and `.mod` files.
The optimizers used from SciPy library were differential_evolution and minimize. The CSV folder contains the averaged
params from iMNTB and TeNT cells (avg_iMNTB_transposed.csv & avg_TeNT_transposed.csv) and also the last params used
on the simulation for each cell. The data folder contains action potential sweeps used to be optimized and fitted. Also
contains the VI used to fit the "passive" conductance. The figures folder contains the 3D graphs showing the behavior 
of simulated 2500 neurons when the algorithm changes the conductance values. Also contains the bar plots comparing the 
optimized values including the stats. The optimization folder contains all the core codes for the paper: 
####
PN file: MNTB_PN_fit.py.
Essential functions: MNTB_PN_myFunctions.py
####
Steady-state fitting single files: fit_passive_v2_TeNT.py & fit_passive_v2_iMNTB.py
Steady-state fitting in a batch: batch_fit_passive_v2_TeNT.py $ batch_fit_passive_v2_iMNTB.py
####
AP fitting files: fit_AP_v2_iMNTB.py & fit_AP_v2_TeNT.py
####
Simulation of the current clamp simulation:fit_simulation.py
####
Plot voltage traces as you desire: plotting_exp_data_traces.py

The results folder contains all the results for the paper and also some figures and csv files to check if the values of
the optimization.

---
## 🛠 Setup Instructions

### 1. Clone the Repository
To clone the repo is necessary that you have installed git on your computer.
```bash
git clone git@github.com:NikollasBenites/PN_MNTB_optimizations
cd PN_MNTB_optimizations
```
### 2. Create the Conda Environment

There are two envs files: one for mac and other for windows. Use the file that match YOUR_OS

``` bash
conda env create -f environment_(YOUR_OS).yml
conda activate neuron_env
```
### 3. Compile NEURON Mechanisms

Make sure you're in the root project directory (Mac and Windows) using Terminal. After you clone the repo, the mod
folders are on ~/optimization/mod and ~/optimization/3D_and_bar_graphs/mod. Set those as your root directory and use the
command bellow on the Terminal. 

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

# 👤 Code adapted by
Nikollas Benites, University of South Florida

Daniel Heller, University of South Florida

# 📝 License

This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
https://creativecommons.org/licenses/by-nc-nd/4.0/

---

