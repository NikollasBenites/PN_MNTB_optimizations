# 🧠 MNTB Principal Neuron Model – NEURON Simulation (Python)

This repository contains a NEURON-based simulation of principal neurons (PN) in the 
Medial Nucleus of the Trapezoid Body (MNTB), developed for understanding intrinsic properties 
and responses after blocking the pre-sensory spontaneous activity (psSA). The model is built using Python and `.mod` files.
The directory tree is:
```
PN_MNTB_modeling/
├── CSV/
│
├── data/
│
├── figures/
│
├── optimization/
│
├── results/
│
├── README.md
│
├── environment_windows.yml
├── environment_mac.yml
│
└── .gitignore
```
---

The optimizers used from SciPy library were differential_evolution and minimize. The CSV folder contains the averaged
params from iMNTB and TeNT cells (avg_iMNTB_transposed.csv & avg_TeNT_transposed.csv) and also the last params used
on the simulation for each cell. The data folder contains action potential sweeps used to be optimized and fitted. Also
contains the VI used to fit the "passive" conductance. The figures folder contains the 3D graphs showing the behavior 
of simulated 2500 neurons with slightly changes in conductance values. Also contains the bar plots comparing the 
optimized values including the stats. The optimization folder contains all the core codes for the paper: 

---
#### PN file: 

```MNTB_PN_fit.py.```

Essential functions: 

```MNTB_PN_myFunctions.py```

#### Steady-state fitting (FIT FIRST STAGE): 
Steady-state fitting in a batch: 

```batch_fit_passive_v2_TeNT.py``` 

```batch_fit_passive_v2_iMNTB.py```
RECOMMENDED TO USE.
Those files work calling the function described bellow and fitting all the files proposed. You just need to run the 
script. All the files will be fitted and save at ~/PN_MNTB_modeling/results/test/passive_fits separated by folders.
The script last about ```20-30 minutes```.
A csv file summary will be generated. The mechanisms optimized on this stage are:
``` gKLT, gH, ELeak, gLeak ```

To fit one value at time, use:
```fit_passive_v2_TeNT.py```  

```fit_passive_v2_iMNTB.py```

To use those, you need to run the code from the terminal inside the optimization folder:
```python fit_passive_tent.py --data YOUR_DESIRED_FILE.cvs``` 

```python fit_passive_imntb.py --data YOUR_DESIRED_FILE.cvs```

The csv files are in ~/PN_MNTB_modeling/data/fit_passive/iMNTB AND ~/TeNT.
Each fitting last about ```2-5 minutes```.
After fitting, they could be found at ~/PN_MNTB_modeling/results/test/passive_fits in each respective folder

#### AP fitting files (FIT SECOND STAGE): 
```fit_AP_v2_iMNTB.py```

```fit_AP_v2_TeNT.py```

The script open the prior fitting files derived from fit_passive scripts and optimizes the conductance:

```gNa, gKHT, gKA```

To improving fitting, a small variation is set up on the passive conductance.
Those files fit the action potentials traces using a mixed approach. The function extract features from the trace and 
also point-by-point voltage comparison. 

#### ✔ Point-by-point voltage comparison

MSE between experimental and simulated AP, but only in the AP window.

#### ✔ Feature matching

Differences in:

latency

AP amplitude

peak

half-width

threshold

AHP

resting potential
Weighted by biological importance.

#### ✔ Temporal alignment

Difference in peak timing → big penalty.

#### ✔ Firing-pattern correctness

Ensures simulated neuron fires:

“phasic” vs “tonic” at +20 pA

adds penalties if spike count is wrong

#### ✔ Physiological constraints

Penalties for:

unrealistic RMP

bad AP peak

too many spikes

unstable resting potential

#### ✔ Final refinement loop

Run multiple cycles of differential_evolution inside shrinking local bounds.

#### Simulation of the current clamp simulation: 
```fit_simulation.py```

#### Plot voltage traces as you desire: 
```plotting_exp_data_traces.py```
---
The results folder contains all the results for the paper and also some figures and csv files to check if the values of
the optimization.

---
## 🛠 Setup Instructions

### 1. Clone the Repository
To clone the repo is necessary that you have installed git on your computer <https://github.com/git-guides/install-git>.
```bash
git clone git@github.com:NikollasBenites/PN_MNTB_optimizations
cd PN_MNTB_optimizations
```
### 2. Create the Conda Environment

There are two envs files: one for mac and other for windows. Use the file that match YOUR_OS. Open a terminal on the
directory the .yml files are. We tested the simulations exhaustively on MacOS. But, we also tested on Windows.
If anything don't work, you can contact us anytime.

``` bash
conda env create -f environment_(YOUR_OS).yml
conda activate neuron_env
```
### 3. Compile NEURON (8.2.6) Mechanisms

FOR WINDOWS USERS IS NECESSARY TO INSTALL NEURON DIRECTLY FROM THE SOURCE. 
Several updates occurred on NEURON simulation environment regarding the API. We strongly recommend to use the specific 
version because the recent version (NEURON 9.0) was not tested.
The detailed documentation and how to install to the version used is found at 
https://nrn.readthedocs.io/en/8.2.6/install/install.html

Make sure you're in the root project directory (Mac and Windows) using Terminal. After you clone the repo, the mod
folders are on <~/optimization/mod> and <~/optimization/3D_and_bar_graphs/mod>. Open a Terminal from those folders and 
use the command bellow on the Terminal. 

```bash
nrnivmodl mod/
```
This will generate the arm64/ folder with compiled special (in Mac).
In Windows OS you will generate a file nrnmech.dll in the Root.

#

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

