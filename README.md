# Sparse-UOT
Code for our ICML '24 paper, "Submodular framework for structured-sparse optimal transport".

*Notations:*
- $\gamma$: Optimal Transport plan
- $K$: Cardinality constraint.

## Implementation of Algorithms
- **Gen-Sparse UOT** (General sparsity constraint)
    - Case: $K$ unspecified. [Code](./sparse_ot/sparse_repr_autok.py)
    - Case: $K$ specified. [Code](./sparse_ot/sparse_repr.py)
> [!NOTE]
> While our experiments use the vector representation of $\gamma$, we have also provided codes with $\gamma$ as matrix: (i) [code](./sparse_ot/full_repr_autok.py) with $K$ unspecified, (ii) [code](./sparse_ot/full_repr.py) with $K$ specified.

- [**CS-UOT**](./sparse_ot/matroid_col_k.py) (Column-wise sparsity constraint)

### Demo Usage
[Gen-Sparse UOT with Gaussians](./examples/Gaussian/sparse_repr.ipynb) (vector representation).
[Gen-Sparse UOT with Gaussians](./examples/Gaussian/full_repr.ipynb) (matrix representation).
[CS-UOT with Gaussians](./examples/Gaussian/matroid_col_k.ipynb)
> [!TIP]
> 'ws' in the function names signifies warm_start, where we use the last outer iterate's $\gamma$. We found warm-start results in faster optimization.

*If you find this useful, consider giving a* :star *to this repository.*
