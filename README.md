# DynoGP

This repository contains the Python code for a novel approach to system identification in dynamical systems using a specialised class of Deep Gaussian Processes. Our method integrates linear dynamic GPs, which represent stochastic Linear Time-Invariant  systems, with static GPs to model nonlinearities. This framework not only captures nonlinear system dynamics but also provides uncertainty quantification.

![cover](https://github.com/benavoli/dynoDeepGP/tree/master/notebooks/gp_linear_cut_Wiener.png)

For more details, see the below paper. 

The repository includes a Jupyter Notebook that runs dynoGP and dynoNet on a simulated problem (folder notebook). Additionally, the simulations folder contains scripts for applying dynoGP to real datasets, as described in the paper.

## How to cite
If you are using `SkewGP` for your research, consider citing the following papers: 
```
@article{dynoDeepgp,
title = {dynoGP: Deep Gaussian Processes for dynamic system
identification},
author = {Alessio Benavoli and Dario Piga and Marco Forgione and Marzo Zaffalon},
url = {https://arxiv.org/abs/},
year = {2025},
}
