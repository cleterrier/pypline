# pypline
Spectral demixing SMLM pipeline in python

## Pypline uses the following third-party code
- Globloc (python DLLs, included in this repository): https://github.com/Li-Lab-SUSTech/GlobLoc/tree/master/GlobLoc_python/source
Yiming Li, Wei Shi, Sheng Liu, Ivana Cavka, Yu-Le Wu, Ulf Matti, Decheng Wu, Simone Koehler, Jonas Ries. Global fitting for high-accuracy multi-channel single-molecule localization. Nat. Commun. 13, 3133 (2022).
https://www.nature.com/articles/s41467-022-30719-4
- COMET for drift correction (has to be added separately): https://github.com/gpufit/Comet
Cost-function Optimized Maximal Overlap Drift Estimation for Single Molecule Localization Microscopy. Lenny Reinkensmeier, Sarah Aufmkolk, Irene Farabella, Alexander Egner, Mark Bates. bioRxiv March 31, 2026.
https://www.biorxiv.org/content/10.64898/2026.03.27.714864v1

## Install procedure
### Installing miniforge
Download and install Miniforge 3

## Setting up the environment
In the Miniforge prompt, run:
mamba create -n pyglobloc-py310 python=3.10 -y
mamba create -n pypline-py310 python=3.10 -y
this will create the environment with python 3.10
once "transaction finished" is displayed, run:
conda activate pyglobloc-py310
(mamba activate can work but conda activate is the standard)
you will see the prompt change from base to pyglobloc-py310

You can check the python version using:
python --version
and the install directory with:
where python

To check that you have only conda forge as the channel:
conda config --show channels
should just return:
-conda-forge

If there are more (XXX for example defaults), remove them using:
conda config --remove channels XXX
conda config --remove channels defaults

## Install packages
Install packages
mamba install -c conda-forge -y numpy scipy pandas scikit-learn matplotlib h5py numba tifffile opencv sympy tqdm

Check if CUDA is installed
python -c "from numba import cuda; print('CUDA available:',cuda.is_available())"

## Get the pypline code (includes GlobLoc)
Get the pipeline folder (Globloc_pypline) from the server and copy it locally (foldr called "pypline")
Go into the folder
cd "C:\Users\chris\christo\Processing\pypline

## Running pypline:
python run_pipeline.py

## Preparing the PSF
Can be done with SMAP (in Matlab) or ui-PSF (within a different env)
Uses multiple folders with one R, one T Z-stack in each
SMAP outputs the file to use: Axcal_inputZStack_cam_R_3dcal.mat
ui-PSF uses the uiPSF_prepare_bead_stacks.py script to stitch the R and T into a single image (currently as a mat file)
ui-PSF uses the Zernicke_vector mode and outputs the file to use: xySwap_PSFmodel_zernike_vector_multi.h5

## Preparing the cylindrical lens transform
Uses the cylindrical_lens_correction_calibration.py
Has to use the Matlab PSF calibration as of now (.mat file, not ui-PSF .h5)
Outputs the file to use in cylindrical_lens_correction_calibration.py: as H_cyl_to_nocyl.txt

## Adding COMET drift correction
https://github.com/gpufit/Comet
download the COMET repository as a .zip file
put Comet-master in the pypline folder
cd Comet-master\Python_interface
pip install -e .
Installing COMET will downgrade numpy and matplotlib compared to the pypline install but it still runs
Test by using:
comet_self_test --plot 





