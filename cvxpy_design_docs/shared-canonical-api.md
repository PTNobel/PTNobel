ChatGPT generated:

# Shared canonical parameter API

**Status:** Draft · **Updated:** October 23, 2026  
**Original authors:** Claude Opus 4.6, with prompting from Parth Nobel  
**Revision:** GPT-6.1 Sol, with prompting from Parth Nobel · **Reviewer:** Max Schaller

## Proposal

`Problem.get_canonical_maps()` exposes the sparse affine maps from compiled
parameters to solver inputs, together with matrices that recover requested
primal and dual variables. CVXPYgen generates code from these maps; CVXPYlayers
evaluates them in its tensor framework.

```python
canon = problem.get_canonical_maps(
    solver=cp.CLARABEL,
    variables=[x, constraint.dual_variables[0]],
)

M, v = canon.maps["q"]
q = M @ param_vec + v
primal_flat = canon.R_primal @ solver_x
dual_flat = canon.R_dual @ solver_y
```

The public method reuses `get_problem_data()` compilation and its reduction
chain. `variables=None` includes all original variables and explicit constraint
dual variables. A supplied list limits the recovery rows and accompanying
metadata to those objects. GP parameter logs and primal exponentials remain
separate from the affine maps.

## What the API returns

Each `canon.maps[name]` is an `(M, v)` pair. `M @ param_vec + v` produces a vector
input or the stored values of a sparse matrix input. `canon.structures[name]`
provides that matrix's shape, indices, indptr, and CSC/CSR format. The map's row
order matches the declared storage, including the solver's triangular storage
when applicable. Structural entries remain present when their values evaluate
to zero. Keys follow solver names: `c` for SCS/ECOS, `q` for Clarabel/OSQP,
with `d` for the compiled objective offset. Native bounds have their own entries
when the solver consumes them.

The map columns use `ParamConeProg`'s existing parameter coordinates.
`parameters`, `param_id_to_col`, `param_id_to_size`, and `total_param_size`
describe the **compiled** parameters. For compiled parameter `r`, its block
begins at `param_id_to_col[r.id]` and contains `r.size` entries in Fortran order.
The public `param_vec` omits the internal constant sentinel; the exporter puts
that column in `v`.

These IDs need not be the original parameter IDs. `CvxAttr2Constr` creates fresh
reduced parameters for sparse, diagonal, and symmetric attributes;
`Chain.compose_param_id_map()` traces original IDs to lists of compiled IDs.
The reduction's value transformation determines the entries of each compiled
block. In particular, sparse parameters contribute their stored entries, without
densification. The ID correspondence does not specify offsets or replace the
existing `update_parameters()`, `param_forward()`, and `param_backward()` hooks.

`R_primal` and `R_dual` map the solver's primal and dual vectors to concatenated
original-variable values. `VarInfo` and `DualInfo` identify each output's ID,
shape, and offset; callers reshape slices in Fortran order. The proposed
`DualInfo` describes each object in `constraint.dual_variables`, also recording
its parent constraint ID. Cone dimensions and constraint ordering accompany the
maps so callers can construct solver inputs and identify the dual blocks.

## Construction

The compiled program already contains the parameter-to-coefficient tensors.
CVXPY's shared extraction code splits them into objective, constraint-matrix,
right-hand-side, and bound maps. The solver's internal `get_canonical_maps()`
method applies the remaining signs, row splits, and storage conversions used by
its numerical adapter. SCS and Clarabel share the base conic extraction; ECOS
splits equality and conic rows; OSQP assembles `A`, `l`, and `u` from the QP
blocks. The base method returns `NotImplemented` until an interface supports it.

The shared reductions have already applied solver-aware PSD conversion and cone
layout before conic map extraction. The exporter reads those transformed
coefficients. It does not repeat PSD scaling or row interleaving in each solver
interface. QP extraction follows `QpSolver.apply()`'s signs on its unformatted
program. Each matrix-storage conversion transforms the values map and its
indices together.

`Problem.get_canonical_maps()` constructs original-output recovery from the
solver output layout and the reductions' recovery metadata. It uses the chain's
existing variable and constraint ID maps to trace requested objects, and composes
the sparse versions of their slicing, expansion, permutation, and scaling
operations. The reductions retain those numerical operations beside their
existing recovery implementations.

For example, `ExactCone2Cone` already owns PSD-to-svec conversion and full-matrix
dual recovery using `SolverInfo`. Its recovery supplies the corresponding block
of `R_dual`. Symmetric primal expansion comes from attribute lowering and has
unscaled off-diagonal entries; scaled PSD dual recovery uses `1/sqrt(2)` entries.
Those are different operators. Cone conversions involving auxiliary constraints
must contribute the recovery from those constraints as well as the ID mapping.

## Constant offsets

The exporter separates `v` from `M` so infinite constant bounds never enter the
linear map. OSQP's unbounded inequality lower rows, for example, have zero rows
in `M` and `-inf` entries in `v`. A cotangent application uses `M.T` without
multiplying by the infinite constants. CVXPYgen can initialize constant entries
once instead of recomputing them on parameter updates.

## Downstream adoption

CVXPYgen replaces affine-map extraction in `cvxpygen/solvers/_interface.py` and
primal/dual extraction in `canonicalizer.py`. It retains solver code generation,
settings, update adjacency, and its derivative machinery. The current
canonicalizer already uses `compose_var_id_map()` for primal IDs and traverses
inverse data for dual IDs; the proposal removes its remaining dependence on
canonical offsets and constraint layout.

CVXPYlayers replaces its direct reduced-tensor access and per-framework recovery
bookkeeping. Its current `get_solver_ctx()` already permutes non-DIFFCP maps
into CSR value order at construction; evaluated-value adapters still perform
sign changes and vector extraction. The new maps absorb those operations.
CVXPYlayers retains framework conversion, batching, solver/derivative calls,
and GP transformations.

CVXPY can keep its existing numerical solve path while downstream libraries
adopt the API. If CVXPY later evaluates these maps internally, it can cache them
with compilation and fall back to `apply()` for interfaces returning
`NotImplemented`.

## Open questions

- **`apply_param_jac()` integration:** Should it consume these maps, or should
  both paths share their construction? `ParamConeProg.apply_param_jac()` returns
  derivatives in compiled-parameter coordinates and rejects quadratic objectives;
  `Problem.backward()` separately applies the reductions' `param_backward()`
  hooks. Any shared implementation must preserve that coordinate boundary.
- **PSD output:** Should callers also be able to request packed PSD duals, in
  addition to the proposed full-matrix recovery?
- **Dual grouping:** The sketch uses one entry per dual variable. CVXPYgen
  currently groups duals by constraint, whereas CVXPYlayers selects individual
  dual variables. Should the shared API use per-dual-variable or per-constraint
  grouping?

## Appendix: API sketch

```python
class Problem:
    def get_canonical_maps(
        self, solver: str, variables: list[Variable] | None = None,
        *, gp: bool = False, **compilation_options,
    ) -> CanonicalMaps:
        ...


@dataclass
class MatrixStructure:
    format: Literal["csc", "csr"]
    shape: tuple[int, int]
    indices: np.ndarray
    indptr: np.ndarray


@dataclass
class CanonicalMaps:
    maps: dict[str, tuple[scipy.sparse.csr_array, np.ndarray]]
    structures: dict[str, MatrixStructure]
    R_primal: scipy.sparse.csr_array
    R_dual: scipy.sparse.csr_array
    primal_vars: list[VarInfo]
    dual_vars: list[DualInfo]
    cone_dims: ConeDims
    constraints: list[Constraint]
    parameters: list[Parameter]  # Compiled parameters.
    param_id_to_col: dict[int, int]  # Excludes the constant sentinel.
    param_id_to_size: dict[int, int]
    total_param_size: int


@dataclass
class VarInfo:
    var_id: int
    shape: tuple[int, ...]
    offset: int
    size: int
    is_symmetric: bool


@dataclass
class DualInfo:
    dual_var_id: int
    constraint_id: int
    shape: tuple[int, ...]
    offset: int
    size: int
    cone_type: str
    is_psd: bool
```

### Implementation references

CVXPY [`8280648`](https://github.com/cvxpy/cvxpy/tree/8280648bd1f70fce701d362de4f58120f940c2f3):
[parameter lowering and hooks](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/cvx_attr2constr.py),
[ID composition](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/chain.py),
[compiled parameter maps](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/dcp2cone/cone_matrix_stuffing.py),
[PSD recovery](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/cone2cone/exact.py),
[cone layout](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/cone_format.py).

CVXPYgen [`79c0255`](https://github.com/cvxgrp/cvxpygen/tree/79c0255236b78d271285c3689a9ff0cd8079ae30):
[map extraction](https://github.com/cvxgrp/cvxpygen/blob/79c0255236b78d271285c3689a9ff0cd8079ae30/cvxpygen/solvers/_interface.py),
[canonicalizer](https://github.com/cvxgrp/cvxpygen/blob/79c0255236b78d271285c3689a9ff0cd8079ae30/cvxpygen/canonicalizer.py).

CVXPYlayers [`bd9caf9`](https://github.com/cvxpy/cvxpylayers/tree/bd9caf9826d758722a5c42b8a51775fcbc037ffe):
[solver setup](https://github.com/cvxpy/cvxpylayers/blob/bd9caf9826d758722a5c42b8a51775fcbc037ffe/src/cvxpylayers/interfaces/__init__.py),
[parameter and recovery handling](https://github.com/cvxpy/cvxpylayers/blob/bd9caf9826d758722a5c42b8a51775fcbc037ffe/src/cvxpylayers/utils/parse_args.py).
