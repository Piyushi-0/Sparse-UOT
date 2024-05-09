# Sparse-UOT
Code for our ICML '24 paper, "Submodular framework for structured-sparse optimal transport".

*Notations:*
- $\gamma$: Optimal Transport plan
- $K$: Cardinality constraint.

## Implementation of Algorithms
- **Gen-Sparse UOT** (General sparsity constraint)
    - [$K$ unspecified](./sparse_ot/sparse_repr_autok.py)
    - [$K$ specified](./sparse_ot/sparse_repr.py)
> > [!NOTE]
> > While our experiments use the vector representation of $\gamma$, we have also provided codes with $\gamma$ as matrix: [$K$ unspecified](./sparse_ot/full_repr_autok.py), [$K$ specified](./sparse_ot/full_repr.py).

- [**CS-UOT**](./sparse_ot/matroid_col_k.py) (Column-wise sparsity constraint)

### Demo Usage
    - [Gen-Sparse UOT with Gaussians (vector representation)](./examples/Gaussian/sparse_repr.ipynb)
    - [Gen-Sparse UOT with Gaussians (matrix representation)](./examples/Gaussian/full_repr.ipynb)
    - [CS-UOT with Gaussians](./examples/Gaussian/matroid_col_k.ipynb)
> [!TIP]
> 'ws' in the function names signifies warm_start, where we use the last outer iterate's $\gamma$. We found warm-start results in faster optimization.

*If you find this useful, consider giving a :star to this repository.*
