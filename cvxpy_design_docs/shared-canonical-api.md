ChatGPT generated:

# Shared canonical parameter API

**Status:** Draft · **Updated:** October 23, 2026  
**Original authors:** Claude Opus 4.6, with prompting from Parth Nobel  
**Revision:** GPT-6.1 Sol, with prompting from Parth Nobel · **Reviewer:** Max Schaller

## Proposal

`Problem.get_canonical_maps()` returns affine parameter-to-solver-data maps and
primal/dual recovery matrices for CVXPYgen and CVXPYlayers.

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

The proposal adds affine-map export to the compiled `ParamProb` representation.
Each reduction exposes sparse operators for the parameter packing and solution
recovery it performs. `SolvingChain` composes those operators and the exported
maps. The reductions' `invert()` methods use the same recovery operators,
including those that combine duals from multiple auxiliary constraints.

Each solver interface specifies its final signs, row splits/stacks, triangular
extraction, CSC/CSR ordering, and native output blocks. Shared layout utilities
apply these operations to both the values maps and sparse structures.
`Problem.get_canonical_maps()` compiles the problem or reuses its cached
compilation, then packages the chain's result as `CanonicalMaps`.

## Contract

- **Affine maps:** The program exporter produces `(M, v)` pairs with finite
  coefficients in `M`. The layout utilities insert constant infinite bounds
  into `v` and zero rows into `M`. Callers evaluate `maps[name]` as `M @ p + v`.
- **Parameter storage:** `SolvingChain` records each original parameter's ID,
  logical shape, packed size, offset, and entry order in `parameter_layouts`.
  Callers assemble `p` by concatenating these compact blocks without a constant
  sentinel. Sparse blocks contain stored entries; symmetric blocks use the
  compiled triangular packing; dense blocks use Fortran order. The reductions'
  packing operators and their adjoints work on stored entries without densification.
- **Matrix storage:** The program exporter preserves structural slots across
  parameter values, including evaluated zeros. The layout utilities produce
  `MatrixStructure` entries with the solver's CSC/CSR ordering and triangle.
  The solver interface identifies native bound inputs and fixed cone data;
  `SolvingChain` includes their maps and metadata in the result.
- **Recovery:** `SolvingChain` constructs the requested recovery rows and records
  original IDs, shapes, and offsets in `primal_vars` and `dual_vars`. It creates
  one dual entry per `constraint.dual_variables` object, with its parent constraint
  ID. The solver interface defines the native output blocks that callers
  concatenate into `solver_x` and `solver_y`. Callers reshape recovered slices
  in Fortran order.
- **Selection:** `Problem.get_canonical_maps()` passes the requested outputs to
  `SolvingChain`: `None` requests all original primal variables and explicit
  constraint duals, a list requests those outputs in caller order within each
  recovery vector, and an empty list requests forward maps only.
  `Problem.get_canonical_maps()` raises an error if the chain cannot recover a
  requested output.

## Decisions for review

1. **Recovery interface:** Should `SolvingChain` return separate primal and dual
   maps, or one block operator on `[solver_x; solver_y]` for reductions that
   exchange primal and dual roles?
2. **Coverage:** Can the first implementation target SCS/Clarabel and the
   stuffed-tensor representation, or must it also export OSQP/ECOS inputs,
   direct-cone recovery, and other `ParamProb` representations? Should
   `Dgp2Dcp` expose its logs and exponentials as separate parameter/recovery
   transformations in that implementation?

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

`SolvingChain` sets `param_id_to_size` and `total_param_size` from packed block
sizes and records the packing operators' gather order in `ParameterInfo.indices`.
The packing adjoints scatter cotangents into those coordinates without duplicating
symmetric off-diagonal cotangents.

The solver interface supplies cone order, fixed cone parameters, vector lengths,
and native output block order for `solver_metadata`. The program exporter supplies
the compiled objective offset as the `d` map; `SolvingChain` records the
maximization/minimization sign in `solver_metadata` for objective recovery.

### Source references

CVXPY snapshot `8280648`:
[compiled program](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/dcp2cone/cone_matrix_stuffing.py),
[chain composition](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/chain.py),
[cone recovery](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/cone2cone/exact.py).
