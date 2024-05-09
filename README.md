# Sparse-UOT
Code for our ICML '24 paper, "Submodular framework for structured-sparse optimal transport".

*Notations:*
- $\gamma$: Optimal Transport plan
- $K$: Cardinality constraint.

## Implementation of Algorithms
1. *GenSparse UOT* (general sparsity constraint):
    - [Implementation](./sparse_ot/sparse_repr_autok.py) when $K$ unspecified.
    - [Implementation](./sparse_ot/sparse_repr.py) when $K$ specified.
> [!NOTE]
> While our experiments use the vector representation of $\gamma$, we also provide implementation with $\gamma$ as a matrix: (i) [code](./sparse_ot/full_repr_autok.py) when $K$ unspecified, (ii) [code](./sparse_ot/full_repr.py) when $K$ specified.

2. *ColSparse UOT* (column-wise sparsity constraint): [Implementation](./sparse_ot/matroid_col_k.py).
#### Demo Usage
- [Gen-Sparse UOT with Gaussians](./examples/Gaussian/sparse_repr.ipynb) (vector representation).

- [Gen-Sparse UOT with Gaussians](./examples/Gaussian/full_repr.ipynb) (matrix representation).

- [CS-UOT with Gaussians](./examples/Gaussian/matroid_col_k.ipynb).
> [!TIP]
> 'ws' in the function names signify warm start, where we use the last outer iterate's $\gamma$. We found that warm start results in faster optimization.

*If you find this useful, consider giving a* ⭐ *to this repository.*
