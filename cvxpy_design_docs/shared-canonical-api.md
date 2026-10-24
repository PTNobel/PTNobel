ChatGPT generated:

# Shared canonical parameter API

**Status:** Draft · **Updated:** October 23, 2026  
**Original authors:** Claude Opus 4.6, with prompting from Parth Nobel  
**Revision:** GPT-6.1 Sol, with prompting from Parth Nobel · **Reviewer:** Max Schaller

## Proposal

`Problem.get_canonical_maps()` exposes the sparse affine maps from compiled
parameters to solver inputs, together with indexing and linear operations that
recover requested primal and dual variables. CVXPYgen generates code from these maps; CVXPYlayers
evaluates them in its tensor framework.

```python
canon = problem.get_canonical_maps(
    solver=cp.CLARABEL,
    variables=[x, constraint.dual_variables[0]],
)

M, v = canon.maps["q"]
q = M @ param_vec + v
recovery = canon.primal_recovery[x.id]
if isinstance(recovery, Exp):
    x_flat = np.exp(solver_x[recovery.indices])
elif isinstance(recovery, np.ndarray):
    x_flat = solver_x[recovery]
else:
    x_flat = recovery @ solver_x  # Linear recovery requiring more than indexing.
x_value = x_flat.reshape(x.shape, order="F")
```

The public method reuses `get_problem_data()` compilation and its reduction
chain. `variables=None` includes all original variables and explicit constraint
dual variables. A supplied list limits recovery and accompanying metadata to those objects. GP parameter logs use the parameter-transform tags;
GP primal exponentiation uses the recovery objects described below.

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
reduced parameters for sparse, diagonal, and symmetric attributes.
`canon.param_transforms` contains the reductions' parameter transformations in
execution order. Each entry has one `source_id`, one `target_id`, and a tagged
operation: `Gather(indices)`, `Log`, `Real`, or `Imag`. Multiple entries can
read the same source; complex splitting uses separate `Real` and `Imag` entries.

Each reduction supplies its entries alongside the parameters it creates.
`SolvingChain` concatenates the entries in reduction order, retaining intermediate
IDs. A consumer starts with original parameter values, evaluates each entry,
then packs the compiled IDs using `param_id_to_col` and `param_id_to_size`.
Unchanged IDs use their original values. `Log`, `Real`, and `Imag` act
elementwise and preserve shape. `Gather` selects a vector from Fortran-flattened
source values; its indices encode the reduction's existing packing order.
Sparse sources contain their stored entries in CVXPY's sparse-index order;
their gathers address those entries without constructing a dense parameter.
A reduction can emit successive entries when packing follows another operation.

CVXPY's `update_parameters()` evaluates these same entries with NumPy/SciPy.
CVXPYlayers dispatches the tags to framework operations, retaining autodiff;
CVXPYgen emits code for the tags it supports. The reductions own the packing
indices and transformation choices. The existing `param_forward()` and
`param_backward()` hooks retain their derivative conventions.

`canon.param_id_map` also exposes `Chain.compose_param_id_map()`'s replacement
correspondence. It does not describe every dependency: a GP compilation can
retain both an original parameter and a new parameter containing its logarithm.
CVXPYgen traces `param_transforms` and unchanged compiled IDs back to original
inputs, then combines those dependencies with the nonzero column blocks of
each `M`. Constant offsets `v` contribute no dependencies.

`canon.primal_recovery[var_id]` is an index array for ordinary indexing or
`Exp(indices)` for GP recovery. An index array means `solver_x[indices]`;
`Exp(indices)` means `exp(solver_x[indices])`. Repeated indices express
expansion, including symmetric primal expansion. Callers reshape each result
to the original variable's shape in Fortran order. `Dgp2Dcp` supplies `Exp`
for the log-variables it creates.

`canon.dual_recovery[dual_var_id]` similarly indexes the solver dual vector.
`DualInfo` identifies the parent constraint of each requested dual variable.
GP duals retain CVXPY's transformed-DCP convention and receive no exponentiation.
The solver interface declares the input vectors' block ordering; Moreau's dual
vector concatenates slack and direct-cone duals, and ECOS's concatenates equality
and inequality duals.

An index array cannot express scaling, zero insertion, or sums of contributions.
For those linear cases, the recovery entry is a sparse matrix applied to the
same solver vector. Examples include scaled PSD dual unpacking, sparse or diagonal
primal expansion with explicit zeros, and cone conversions combining auxiliary
duals. Ordinary indexing stays as an index array. When a consumer needs a sparse
matrix, it can represent indexing by one unit entry per output row; the
exponentiation in `Exp` remains separate.

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

Each reduction supplies its local recovery alongside its inversion metadata:
the input blocks it reads and the indices selecting its outputs.
`SolvingChain` resolves those blocks to solver coordinates and composes indexing
by substitution: `composed = inner_indices[outer_indices]`. Pass-through blocks
reuse their indices; repeated indices preserve expansion. `Dgp2Dcp` wraps the
corresponding composed primal indices in `Exp`.

A reduction requiring scaling, zero insertion, or addition supplies the sparse
linear operation for that output. `SolvingChain` composes it with the preceding
indexing or linear recovery. `ExactCone2Cone` owns PSD dual scaling and recovery
from auxiliary constraints; attribute reduction owns primal expansion.
Recovery from multiple auxiliaries retains all contributing blocks until their
combination has been composed. `Problem.get_canonical_maps()` returns the
requested original outputs after chain composition.

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
   and evaluates the returned recovery entries. Parameter preprocessing uses the exported tags.
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


class ParameterTransformKind(Enum):
    GATHER = "gather"
    LOG = "log"
    REAL = "real"
    IMAG = "imag"


@dataclass
class ParameterTransform:
    source_id: int
    target_id: int
    kind: ParameterTransformKind
    indices: np.ndarray | None = None  # Required only for GATHER.


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
    primal_recovery: dict[int, np.ndarray | Exp | scipy.sparse.csr_array]
    dual_recovery: dict[int, np.ndarray | scipy.sparse.csr_array]
    primal_vars: list[VarInfo]
    dual_vars: list[DualInfo]
    cone_dims: ConeDims
    constraints: list[Constraint]
    parameters: list[Parameter]  # Compiled parameters.
    param_transforms: list[ParameterTransform]  # In execution order.
    param_id_map: dict[int, list[int]]  # Existing replacement correspondence.
    param_id_to_col: dict[int, int]  # Excludes the constant sentinel.
    param_id_to_size: dict[int, int]
    total_param_size: int


@dataclass
class Exp:
    indices: np.ndarray


@dataclass
class VarInfo:
    var_id: int
    shape: tuple[int, ...]
    size: int
    is_symmetric: bool


@dataclass
class DualInfo:
    dual_var_id: int
    constraint_id: int
    shape: tuple[int, ...]
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
