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

Export forward maps from the final compiled program. Add sparse parameter-packing
and recovery operators alongside the reductions' `invert()` implementations;
compose them in `SolvingChain`. Share the recovery operators with numerical
inversion, including recovery from multiple auxiliary constraint duals.

Solver interfaces specify the remaining signs, row splits/stacks, triangular
extraction, and CSC/CSR ordering using shared layout utilities. Apply each
operation to both the values map and sparse structure.

## Contract

- **Affine maps:** `maps[name] = (M, v)` gives `M @ p + v`. `M` contains finite
  coefficients. Constant infinite bounds appear only in `v`, with zero rows in
  `M`.
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

`param_id_to_size` and `total_param_size` count packed entries.
`ParameterInfo.indices` records the gather order; its adjoint scatters into those
coordinates without duplicating symmetric off-diagonal cotangents.
`solver_metadata` supplies cone order, fixed cone parameters, vector lengths,
and native output block order. The `d` map contains the compiled objective offset;
original-objective recovery also needs the maximization/minimization sign.

### Source references

CVXPY snapshot `8280648`:
[compiled program](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/dcp2cone/cone_matrix_stuffing.py),
[chain composition](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/chain.py),
[cone recovery](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/cone2cone/exact.py).
