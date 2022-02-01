# Inductive Matrix Completion (IMC)
In the IMC problem, the goal is to recover a low rank matrix X* from few observed entries while incorporating prior knowledge about
its row and column subspaces. The prior knowledge is given in the form of two matrices A, B.

This repository contains `Matlab` and `Python` implementations for three IMC algrithms:
* `GNIMC`: Gauss-Newton IMC, as described in [P. Zilber and B. Nadler (preprint, 2022)](https://arxiv.org/abs/2201.13052)
* `AltMin`: alternating minimization
* `RGD`: (balance-regularized) gradient descent.

## Usage
Simple demos demonstrating the usage of the algorithms are available, see `IMC_demo.m` and `IMC_demo.py`.
In these demos, the expected configurations of the matrix to be recovered and the side information matrices are:
- n1: number of rows in X*
- n2: number of columns in X*
- d1: number of columns in A
- d2: number of columns in B
- rank of X*
- condition number of X*
- oversampling ratio: number of observed entries, normalized by the factor _(d1 + d2 - rank)*rank_. Oversampling ratio of one corresponds to the information limit.

Given these configurations, the demo generates a matrix using `generate_matrix.m`/`generate_matrix.py`, and a mask (sampling pattern) using `generate_mask.m`/`generate_mask.py`, and then run one of the algorithms: `GNIMC`, `AltMin` or `RGD`.

## Citation
If you refer to the `GNIMC` method or the paper, please cite them as:
```
@article{zilber2022inductive,
      title={Inductive Matrix Completion: No Bad Local Minima and a Fast Algorithm}, 
      author={Pini Zilber and Boaz Nadler},
      year={2022},
      eprint={2201.13052},
      archivePrefix={arXiv},
}
```
