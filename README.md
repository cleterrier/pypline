# pypline
Spectral demixing 3D-SMLM pipeline in python
code by Christopher Parperis

## Third-party code
- Globloc (python DLLs, included in this repository):  
https://github.com/Li-Lab-SUSTech/GlobLoc/tree/master/GlobLoc_python/source  
Global fitting for high-accuracy multi-channel single-molecule localization. Yiming Li, Wei Shi, Sheng Liu, Ivana Cavka, Yu-Le Wu, Ulf Matti, Decheng Wu, Simone Koehler, Jonas Ries. Nat. Commun. 2022; 13, 3133.  
https://www.nature.com/articles/s41467-022-30719-4
- Optional: uiPSF for PSF calibration (to be done separately, upstream of pypline)  
https://github.com/ries-lab/uiPSF  
Sheng Liu, Jianwei Chen, Jonas Hellgoth, Lucas-Raphael Müller, Boris Ferdman, Christian Karras, Dafei Xiao, Keith A Lidke, Rainer Heintzmann, Yoav Shechtman, Yiming, Jonas Ries. Universal inverse modeling of point spread functions for SMLM localization and microscope characterization. Nat Methods 2024 Jun;21(6):1082-1093.
- Optional: COMET for drift correction (to be added manually):  
https://github.com/gpufit/Comet  
Cost-function Optimized Maximal Overlap Drift Estimation for Single Molecule Localization Microscopy. Lenny Reinkensmeier, Sarah Aufmkolk, Irene Farabella, Alexander Egner, Mark Bates. bioRxiv March 31, 2026.  
https://www.biorxiv.org/content/10.64898/2026.03.27.714864v1

## What is pypline
A pipeline for spectral demixing 3D-SMLM adapted for two-camera setups (like the Abbelight SAFe360).

You need to have PSF calibration files prepared (either from SMAP or ui-PSF).  Then for every pair of image stacks (R and T, obtained from each camera) inside a folder, it automates the following steps:
1. Fitting of each side (R and T) independently using spline fitting from the PSF calibrations.
2. Calculation of the transform between channels from step 1 data.
3. Global fitting using GlobLoc (with the individual fitting from step 1 as seed), with either fixed ratios (performs channel assignment) or free ratios (channel assignment has to be done downstream of the pipeline using the calculated ratios).
4. RCC and COMET drift correction (COMET has to be installed separately)
5. Grouping of blinking events accross frames.
6. Compensation of the deformation from the cylindrical lens (requires calibrations stacks with and without the lens).
6. Export as ThunderSTORM-compatible csv files for downstream visualization.

## Install procedure
### Installing miniforge
Download and install Miniforge 3: https://conda-forge.org/download/

### Setting up the environment
In the Miniforge prompt, run:
```
mamba create -n pypline-py310 python=3.10 -y
```
this will create the environment with python 3.10
once "transaction finished" is displayed, run:
```
conda activate pyglobloc-py310
```
(mamba activate can work but conda activate is the standard)
you will see the prompt change from base to pyglobloc-py310

You can check the python version using:
```
python --version
```
and the install directory with:
```
where python
```

To check that you have only conda forge as the channel:
```
conda config --show channels
```
should just return:
```
-conda-forge
```

If there are more (for example 'defaults'), remove them using:
```
conda config --remove channels defaults
```

### Install packages
Install packages
```
mamba install -c conda-forge -y numpy scipy pandas scikit-learn matplotlib h5py numba tifffile opencv sympy tqdm
```

Check if CUDA is installed
```
python -c "from numba import cuda; print('CUDA available:',cuda.is_available())"
```

### Get the pypline code (includes GlobLoc)
Get the pypline repository as a .zip file and put in in a 'pypline' folder locally
Go into the folder
```
cd C:\Users\chris\christo\Processing\pypline
```

## Using pypline
### Editing the main pypline file
Open run_pipeline.py with a code editor, check and edit the paths and options.

### Running pypline:
In the Miniforge prompt:
```
python run_pipeline.py
```

## Steps upstream of pypline
### Preparing the PSF
Can be done with SMAP (in Matlab) or ui-PSF (within a different env). Both use multiple folders the two beads stacks (R and T) in each.

SMAP outputs the file to use in run_pipeline.py: Axcal_inputZStack_cam_R_3dcal.mat

ui-PSF uses the uiPSF_prepare_bead_stacks.py script to stitch the R and T into a single image (currently exported as a .mat file). ui-PSF uses the Zernicke_vector mode and outputs the following file to use in run_pipeline.py: xySwap_PSFmodel_zernike_vector_multi.h5

### Preparing the cylindrical lens transform
Use the cylindrical_lens_correction_calibration.py script. The script has to use the Matlab PSF calibration as of now (.mat file, not ui-PSF .h5). The script outputs the file to use in run_pipeline.py: H_cyl_to_nocyl.txt

## Adding COMET drift correction
download the COMET repository as a .zip file: https://github.com/gpufit/Comet

put Comet-master in the pypline folder:
```
cd Comet-master\Python_interface
pip install -e .
```
Installing COMET will downgrade numpy and matplotlib compared to the pypline install but it still runs.  

Test COMET by using:
```
comet_self_test --plot 
```
