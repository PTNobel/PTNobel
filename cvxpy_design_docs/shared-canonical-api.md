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
`Problem.get_canonical_maps()` includes the result of
`Chain.compose_param_id_map()` as `canon.param_id_map`: original IDs map to lists
of compiled IDs, and omitted IDs are unchanged.
The reduction's value transformation determines the entries of each compiled
block. In particular, sparse parameters contribute their stored entries, without
densification. The ID correspondence is separate from vector offsets. The
reductions handle parameter updates and derivatives through `update_parameters()`,
`param_forward()`, and `param_backward()`.

CVXPYgen derives its user-parameter-to-solver-input update adjacency from
`param_id_map` and the nonzero columns of each `M`. The compiled IDs identify
column blocks through `param_id_to_col` and `param_id_to_size`; CVXPYgen combines
dependencies across all compiled blocks belonging to an original parameter.
Constant offsets `v` contribute no dependencies.

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
blocks.

The shared reductions have already applied solver-aware PSD conversion and cone
layout before conic map extraction. QP extraction follows `QpSolver.apply()`'s
signs on its unformatted program. Each matrix-storage conversion transforms the values map and its
indices together.

`Problem.get_canonical_maps()` constructs recovery of original outputs from the
solver output layout and the reductions' recovery metadata. It uses the chain's
existing variable and constraint ID maps to trace requested objects, and composes
the sparse versions of their slicing, expansion, permutation, and scaling
operations.

For example, `ExactCone2Cone` already owns PSD-to-svec conversion and full-matrix
dual recovery using `SolverInfo`. Its recovery supplies the corresponding block
of `R_dual`. Symmetric primal expansion comes from attribute lowering and has
unscaled off-diagonal entries; scaled PSD dual recovery uses `1/sqrt(2)` entries.
Cone conversions involving auxiliary constraints contribute their dual-recovery
operators as well as their ID mappings.

## Constant offsets

The exporter separates `v` from `M` so infinite constant bounds never enter the
linear map. OSQP's unbounded inequality lower rows, for example, have zero rows
in `M` and `-inf` entries in `v`. A cotangent application uses `M.T` without
multiplying by the infinite constants. CVXPYgen can initialize constant entries
once instead of recomputing them on parameter updates.

## Incremental adoption

CVXPY can migrate its numerical solve path one solver interface at a time.
The base `get_canonical_maps()` returns `NotImplemented`. CVXPY caches either
that sentinel or the interface's maps with the compilation:

- When the interface returns maps, CVXPY evaluates them to produce solver inputs.
- When the interface returns `NotImplemented`, CVXPY calls its existing `apply()`.

Each converted interface shares its map construction between CVXPY's solve path
and downstream callers.

### Migration plan

1. **CVXPY adds the API.** The base conic implementation covers SCS/Clarabel;
   OSQP and ECOS supply their splitting and storage adaptations. CVXPY tests
   compare map evaluation with the existing numerical adapter inputs and compare
   recovery with the reduction chain's recovery.
2. **CVXPYlayers adopts the maps.** It replaces direct reduced-tensor access and
   per-framework recovery bookkeeping. The returned maps incorporate the signs,
   vector extraction, and storage ordering its adapters currently perform.
   CVXPYlayers retains framework conversion, batching, solver/derivative calls,
   and GP transformations.
3. **CVXPYgen adopts the maps.** It replaces affine-map extraction in
   `cvxpygen/solvers/_interface.py` and primal/dual extraction in
   `canonicalizer.py`. It retains solver code generation, settings, update
   adjacency, and derivative machinery.
4. **CVXPY deprecates downstream reliance on internal tensors.** After both
   libraries have migrated, CVXPY documents `reduced_A.reduced_mat`, `q`, and
   related tensor/layout access as implementation details and directs downstream
   callers to `get_canonical_maps()`.

## Open questions

- **`apply_param_jac()` integration:** Should it consume these maps, or should
  both paths share their construction? `ParamConeProg.apply_param_jac()` returns
  derivatives in compiled-parameter coordinates and rejects quadratic objectives;
  `Problem.backward()` separately applies the reductions' `param_backward()`
  hooks.
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
    param_id_map: dict[int, list[int]]  # Original -> compiled IDs; omitted IDs unchanged.
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
