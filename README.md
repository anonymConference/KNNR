# KNNR
K-Nearest-Neighbor Resampling for Off-Policy Evaluation in Stochastic Control

This repository contains the code to the experiments conducted in the paper "K-Nearest-Neighbor Resampling for Off-Policy Evaluation in Stochastic Control". The files in this repository are mostly self-contained. The exception is the BP environment where an environmnent and a util file is used additionally. The reward_stored variable contains the value estimates with which the MSEs can be calculated. The data set sizes can be adapted by modifying the "Nsample_list" variable. The target values for the MSE calculation can be retrieved by setting get_target=True for LQR and LOB. For the BP, one needs to  run the get_target file.
