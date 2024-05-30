# Sparse-UOT
Code for our ICML '24 paper, "Submodular framework for structured-sparse optimal transport".

To install the 'sparse_ot' package, please clone this repository & run `pip install .`. The list of packages specific to the experiments has been shared separately in their respective folders (under examples/).

Notations:
- $\gamma$: Optimal Transport plan
- $K$: Cardinality constraint.

## Implementation of Algorithms
1. *GenSparse UOT*, general sparsity constraint:
    - [Implementation](https://github.com/Piyushi-0/Sparse-UOT/blob/main/sparse_ot/sparse_repr_autok.py) when $K$ unspecified.
    - [Implementation](https://github.com/Piyushi-0/Sparse-UOT/blob/main/sparse_ot/sparse_repr.py) when $K$ specified.
> [!NOTE]
> While our experiments use a sparse vectorial representation of $\gamma$, we also provide implementation with $\gamma$ as a matrix: (i) [code](https://github.com/Piyushi-0/Sparse-UOT/blob/main/sparse_ot/full_repr_autok.py) when $K$ unspecified, (ii) [code](https://github.com/Piyushi-0/Sparse-UOT/blob/main/sparse_ot/full_repr.py) when $K$ specified.

2. *ColSparse UOT*, column-wise sparsity constraint: [Implementation](https://github.com/Piyushi-0/Sparse-UOT/blob/main/sparse_ot/matroid_col_k.py).

> [!TIP]
> 'ws' in the function names signifies warm start, where we use the previous outer iteration's $\gamma$. We found that a warm start results in faster optimization.

*Toy example with Gaussians*:
- [GenSparse UOT Supplementary](https://github.com/Piyushi-0/Sparse-UOT/blob/main/examples/Gaussian/Gaussian_Supplementary.ipynb).
- [CS-UOT](https://github.com/Piyushi-0/Sparse-UOT/blob/main/examples/Gaussian/matroid_col_k.ipynb).

Codes for the experiments are under 'examples/'.

*If you find this useful, consider giving a* ⭐ *to this repository.*
