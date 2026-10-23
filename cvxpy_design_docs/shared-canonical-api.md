ChatGPT generated:

# Shared canonical parameter API

**Status:** Draft · **Updated:** October 23, 2026  
**Original authors:** Claude Opus 4.6, with prompting from Parth Nobel  
**Revision:** GPT-6.1 Sol, with prompting from Parth Nobel · **Reviewer:** Max Schaller

## Proposal

Add `Problem.get_canonical_maps()` to expose affine parameter-to-solver-data maps
and primal/dual recovery matrices for CVXPYgen and CVXPYlayers.

```python
canon = problem.get_canonical_maps(
    solver=cp.CLARABEL,
    variables=[x, constraint.dual_variables[0]],
)

M, v = canon.maps["q"]
q = M @ p + v
primal_flat = canon.R_primal @ solver_x
dual_flat = canon.R_dual @ solver_y
```

## Map construction

Export forward maps from the final compiled program. Cone conversion, attribute
lowering, and row formatting are already reflected in its coefficients. Earlier
reductions supply parameter-packing and solution-recovery operators; the chain
composes them using the metadata recorded during compilation.

| Owner | Maps or metadata to expose |
| --- | --- |
| `CvxAttr2Constr` | Parameter packing; symmetric, diagonal, and sparse primal expansion. |
| `Dcp2Cone` and cone conversions | Original-output recovery from canonical variables and constraint duals, including auxiliary constraints. `ExactCone2Cone` supplies PSD/svec dual recovery using `SolverInfo`. |
| `ConeMatrixStuffing` / compiled `ParamProb` | Objective, constraint, and bound affine maps; matrix structures; variable and constraint slices. |
| `ConeFormat` | Its signed row permutation and the dual-coordinate recovery operator. |
| `ExtractDirectCones` | Correspondence between removed constraint rows and native direct-cone dual outputs. |
| Solver interface | Input names, row splits/stacks, final signs, triangular storage, CSC/CSR ordering, and native output blocks. |
| `SolvingChain` / `Problem` | Composition, requested-output selection, and caching with the compilation. |

Trace IDs with the existing chain helpers and compose sparse operators for
packing, scaling, and scattering. Recovery from several auxiliary constraints
selects from their combined dual blocks. Reuse these operators in `invert()` so
numerical and exported recovery share an implementation.

Solver interfaces specify their final layout operations using shared utilities
for selection, permutation, signs, stacking, and constant insertion. These
utilities transform values maps and sparse structures together. They operate on
the program produced by the resolved chain: conic interfaces receive formatted
rows; QP interfaces receive unformatted rows. PSD scaling and cone interleaving
remain in the reductions that perform them.

## Contract

- **Affine maps:** `maps[name] = (M, v)` gives `M @ p + v`. `M` contains finite
  coefficients. Constant infinite bounds appear only in `v`, with zero rows in
  `M`; parameter adjoints use `M.T`.
- **Parameter storage:** `p` concatenates compact blocks with no constant
  sentinel. Sparse parameters contribute stored entries; symmetric parameters
  use the compiled triangular packing. Each block records its original ID,
  logical shape, packed size, offset, and entry order. Packing and its adjoint
  operate on stored entries without densification. Dense blocks use Fortran order.
- **Matrix storage:** Declare CSC/CSR structure and triangle for each matrix
  input. Preserve structural slots across parameter values, including evaluated
  zeros. Export native bounds and fixed cone data consumed by the solver.
- **Recovery:** Record original variable IDs, shapes, and output offsets. Each
  `constraint.dual_variables` object gets an entry carrying its parent constraint
  ID. Reshape slices in Fortran order. Specify the native solver output blocks
  concatenated into `solver_x` and `solver_y`.
- **Selection:** `variables=None` selects all original primal variables and
  explicit constraint duals; a list selects outputs in caller order within each
  recovery vector. An empty list requests forward maps only. Construct only
  requested recovery rows. Raise an error for unsupported requested recovery.

Parameter value changes reuse the compiled maps. Exported arrays must not alias
mutable cached tensors.

## Decisions for review

1. **Recovery interface:** Keep separate primal and dual maps, or use one block
   operator on `[solver_x; solver_y]` to accommodate reductions that exchange
   primal and dual roles?
2. **Coverage:** Is SCS/Clarabel with the stuffed-tensor representation sufficient
   initially, or must the first version include OSQP/ECOS, direct-cone recovery,
   and other `ParamProb` representations? For DGP, expose logs and exponentials
   as separate parameter/recovery transformations or defer support?

## Appendix: contract sketch

```python
@dataclass
class MatrixStructure:
    format: Literal["csc", "csr"]
    shape: tuple[int, int]
    indices: np.ndarray
    indptr: np.ndarray
    triangle: Literal["full", "upper", "lower"]


@dataclass
class VarInfo:
    var_id: int
    shape: tuple[int, ...]
    offset: int
    size: int


@dataclass
class DualInfo:
    dual_var_id: int
    constraint_id: int
    shape: tuple[int, ...]
    offset: int
    size: int


@dataclass
class ParameterInfo:
    param_id: int
    shape: tuple[int, ...]  # Logical shape, not packed storage size.
    offset: int
    size: int              # Number of packed entries.
    # Ordered logical indices for compact storage; None for dense F-order.
    indices: tuple[np.ndarray, ...] | None


@dataclass
class CanonicalMaps:
    maps: dict[str, tuple[scipy.sparse.csr_array, np.ndarray]]
    structures: dict[str, MatrixStructure]
    R_primal: scipy.sparse.csr_array
    R_dual: scipy.sparse.csr_array
    primal_vars: list[VarInfo]
    dual_vars: list[DualInfo]
    parameters: list[Parameter]
    parameter_layouts: list[ParameterInfo]
    param_id_to_col: dict[int, int]
    param_id_to_size: dict[int, int]
    total_param_size: int
    solver_metadata: dict[str, object]
```

`M.shape == (number_of_output_values, total_param_size)`; its CSR format is
independent of the output matrix format. `solver_metadata` supplies cone order,
fixed cone parameters, vector lengths, and native output block order. The `d` map
contains the compiled objective offset; original-objective recovery also needs
the maximization/minimization sign.

`param_id_to_size` and `total_param_size` count packed entries.
`ParameterInfo.indices` gives their logical indices in packing order. The packing
adjoint scatters cotangents into those coordinates without duplicating
off-diagonal cotangents into a full symmetric matrix.

### Example adapter layouts

For the base conic adapter's formatted program, let its exported constraint
coefficients be `A0, b0`. The adapter uses `A = -A0`, `b = b0`. SCS names the
linear objective `c`; Clarabel names it `q`. PSD scaling and cone interleaving
have already happened upstream.

For OSQP's unformatted program, let its coefficients be `AF, bg` with equality
rows first:

```text
A_eq =  AF[:n_eq]       b_eq = -bg[:n_eq]
F    = -AF[n_eq:]       g    =  bg[n_eq:]

A = [A_eq; F]
l = [b_eq; -inf]
u = [b_eq; g]
```

Triangular extraction of `P`, CSC/CSR conversion, and duplicate consolidation
must transform both the values map and sparse structure. Bounds
lowered into constraints are already represented in the constraint maps.

### Source snapshots

Implementation references checked against upstream sources on October 23, 2026:

- **CVXPY, `8280648`:**
  [chain construction](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/solvers/solving_chain.py),
  [ID composition](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/chain.py),
  [attribute lowering](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/cvx_attr2constr.py),
  [exact cone conversions](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/cone2cone/exact.py),
  [compiled cone program](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/dcp2cone/cone_matrix_stuffing.py),
  [cone formatting](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/cone_format.py),
  [direct-cone extraction](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/cone2cone/extract_direct_cones.py),
  [QP adapter](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/solvers/qp_solvers/qp_solver.py),
  [OSQP assembly](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/solvers/qp_solvers/osqp_qpif.py).
- **CVXPYgen, `79c0255`:**
  [canonicalizer](https://github.com/cvxgrp/cvxpygen/blob/79c0255236b78d271285c3689a9ff0cd8079ae30/cvxpygen/canonicalizer.py),
  [map extraction](https://github.com/cvxgrp/cvxpygen/blob/79c0255236b78d271285c3689a9ff0cd8079ae30/cvxpygen/solvers/_interface.py).
- **CVXPYlayers, `bd9caf9`:**
  [solver contexts](https://github.com/cvxpy/cvxpylayers/blob/bd9caf9826d758722a5c42b8a51775fcbc037ffe/src/cvxpylayers/interfaces/__init__.py),
  [parameter and recovery bookkeeping](https://github.com/cvxpy/cvxpylayers/blob/bd9caf9826d758722a5c42b8a51775fcbc037ffe/src/cvxpylayers/utils/parse_args.py).
