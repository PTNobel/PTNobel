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

## Parameter coordinates and maps

The compiled `ParamProb` defines the columns of every forward map. Its
`param_id_to_col`, `param_id_to_size`, and `parameters` refer to **compiled
parameter IDs**. For a compiled parameter `r`, callers place its Fortran-flattened
value in `p[col:col + r.size]`, where `col = param_id_to_col[r.id]`.
The exporter splits the internal constant column into `v`:

```text
solver_input_values = M @ p + v
```

`Chain.compose_param_id_map()` supplies the separate correspondence
`original_id -> [compiled_id, ...]`; an omitted ID is unchanged. A reduction can
change a parameter's ID and shape, or split it into several parameters.
`CanonicalMaps.parameter_id_map` exposes this existing correspondence.
It does not assign vector offsets or transform values.

For example, `CvxAttr2Constr` replaces a sparse parameter with a reduced parameter
whose size is its number of stored entries. The exported maps use that reduced
parameter's ID and size. Callers never construct a dense vector of the original
parameter's logical entries.

The reductions already implement `update_parameters()`, `param_forward()`, and
`param_backward()`. These hooks define value updates, parameter tangents, and
user-visible parameter gradients. The proposal adds framework-independent exports
of these transformations. `M.T` produces compiled-coordinate cotangents; the
reductions' `param_backward()` transformations convert them to original-parameter
gradients. The exports preserve the existing symmetric-parameter gradient
conventions.

## Construction and returned data

The compiled program exports `(M, v)` pairs, matrix structures, and native bound
maps. Each solver interface supplies its remaining signs, row splits/stacks,
triangular extraction, CSC/CSR ordering, and native output blocks. Shared layout
utilities apply each operation to the values map and matrix structure together.
The exporter and layout utilities preserve structural slots across parameter
values. Constant infinite bounds belong in `v`, with zero corresponding rows in
`M`.

The reductions export sparse recovery operators from the transformations used by
`var_forward()`, `var_backward()`, and `invert()`. `SolvingChain` composes these
operators with the solver interface's native output layout to produce `R_primal`
and `R_dual`. The chain uses `compose_var_id_map()` and `compose_constr_id_map()`
for ID tracing; the numerical operators supply expansion, scaling, and recovery
from auxiliary constraints.

`Problem.get_canonical_maps()` compiles or reuses its cached compilation. It asks
`SolvingChain` for the requested recovery rows and packages the maps and metadata.
`variables=None` requests original primal variables and explicit constraint duals;
a list requests those outputs in caller order within each recovery vector; an
empty list requests forward maps only. The method raises an error when the chain
cannot recover a requested output.

`primal_vars` and `dual_vars` identify original outputs by ID, shape, and offset.
Each dual entry identifies an object in `constraint.dual_variables` and its parent
constraint. The solver interface records how native output blocks form `solver_x`
and `solver_y`; callers reshape recovered slices in Fortran order.

## Decisions for review

1. **Parameter transformations:** How should reductions export the existing
   update/tangent/gradient transformations for CVXPYlayers and CVXPYgen? Linear
   transformations admit sparse operators; DGP's logs and exponentials require
   a nonlinear representation. The ID correspondence and compiled-vector layout
   already exist.
2. **Recovery interface:** Should `SolvingChain` return separate primal and dual
   maps, or one block operator on `[solver_x; solver_y]` for reductions that
   exchange primal and dual roles?
3. **Coverage:** Can the first implementation target SCS/Clarabel and the
   stuffed-tensor representation, or must it also export OSQP/ECOS inputs,
   direct-cone recovery, and other `ParamProb` representations?

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
class CanonicalMaps:
    maps: dict[str, tuple[scipy.sparse.csr_array, np.ndarray]]
    structures: dict[str, MatrixStructure]
    R_primal: scipy.sparse.csr_array
    R_dual: scipy.sparse.csr_array
    primal_vars: list[VarInfo]
    dual_vars: list[DualInfo]
    parameters: list[Parameter]  # Compiled parameters.
    parameter_id_map: dict[int, list[int]]  # Original -> compiled IDs.
    param_id_to_col: dict[int, int]
    param_id_to_size: dict[int, int]
    total_param_size: int
    solver_metadata: dict[str, object]
```

The program exporter copies `param_id_to_col`, `param_id_to_size`, and
`total_param_size` from the compiled program, excluding the constant sentinel.
`SolvingChain` obtains `parameter_id_map` from `compose_param_id_map()`.
The parameter-transformation export is the first review decision above.

The solver interface supplies cone order, fixed cone parameters, vector lengths,
and native output block order for `solver_metadata`. The program exporter supplies
the compiled objective offset as the `d` map; `SolvingChain` records the
maximization/minimization sign in `solver_metadata` for objective recovery.

### Source references

CVXPY snapshot `8280648`:
[compiled program](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/dcp2cone/cone_matrix_stuffing.py),
[ID composition](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/chain.py),
[parameter hooks](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/reduction.py),
[attribute lowering](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/cvx_attr2constr.py),
[cone recovery](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/cone2cone/exact.py).
