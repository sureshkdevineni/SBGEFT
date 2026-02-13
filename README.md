# SBGEFT
Strain-Based Effective Field Theory of Gravity: cosmology inference pipelines, EHT shadow analysis, and MCMC chains.

Data and Resources

The repository contains the cosmological inference code for the strain model, EHT shadow analysis modules, and posterior chains for the dataset combinations reported in Tables I–III.

All datasets used in this work are publicly available from the corresponding collaborations:

DESI DR2 BAO measurements (mean vectors and covariance):
https://github.com/CobayaSampler/bao_data/tree/master/desi_bao_dr2

Pantheon+ Type Ia Supernova dataset and covariance:
https://github.com/PantheonPlusSH0ES/DataRelease

DES-SN5YR Supernova dataset and covariance:
https://github.com/des-science/DES-SN5YR/releases/tag/1.3
https://zenodo.org/records/12720778

PArthENoPE nucleosynthesis tables (used by CAMB):
https://github.com/cmbant/CAMB/blob/master/camb/PArthENoPE_880.2_standard.dat

The analysis makes use of the publicly available CAMB Boltzmann solver for early-universe calculations and associated nucleosynthesis tables. No proprietary data were used, and all results can be reproduced using the code and chains provided in this repository.
