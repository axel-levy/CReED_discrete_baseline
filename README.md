# Introduction

This repository contains the necessary materials to reproduce the results I obtained in my report.

# Package requirements

The codes should be run in `Python 3`. The following packages will be required to run the codes (latest versions should work):
- `numpy`
- `matplotlib`
- `tqdm`
- `tensorflow`
- `scikit-image`

All packages can be installed with `$pip intall name_of_the_package`.

# User manual

- `data` contains the file downloaded from http://rpg.ifi.uzh.ch/davis_data.html (under `Text (zip)`). The files must be downloaded before running the codes. Once you have downloaded a folder, put it in `data`.
- `events` will contain the stream of events under `.npy` format.
- `dataio.py` manages the data loading.
- `txt_to_npy` translates `.txt` files to `.npy` files.
- `gradient descent.py` contains experiments with gradient descent.
- `ADMM.py` contains experiments with ADMM.

1. Download the folder called `slider_depth` (http://rpg.ifi.uzh.ch/datasets/davis/slider_depth.zip)
2. Run `$python txt_to_npy.py` to convert `.txt` file in `data` to `.npy` files, stored in `events`.
3. Run `$python gradient_descent.py` to perform gradient descent (on slider_depth by default).
4. Run `$python ADMM.py` to perform ADMM (on slider_depth by default).

- To use another stream of events than slider_depth, just uncomment the indicated lines in `gradient_descent.py` or `ADMM.py`
```
# Uncomment to use shapes_6dof
# ************
Uncomment this
#************
```
- The parameters of the reconstruction can be modified under the line
```
### RUN EXPERIMENT ###
```

Have fun!
