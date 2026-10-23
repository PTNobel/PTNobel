ChatGPT generated:

# Shared canonical parameter API

**Status:** Draft · **Updated:** October 23, 2026  
**Original authors:** Claude Opus 4.6, with prompting from Parth Nobel  
**Revision:** OpenAI GPT-6.1 (`gpt-6.1-sol`), with prompting from Parth Nobel · **Reviewer:** Max Schaller

## Proposal

Expose the affine maps from parameters to solver inputs and the linear maps from
solver outputs to requested primal and dual variables. CVXPYgen can generate code
from them; CVXPYlayers can evaluate them in its tensor framework.

The restructured solving chain lets us put most construction in shared reductions.
Each reduction should expose the numerical transformation it owns. The chain
composes those transformations; solver adapters declare only their remaining
input/output layout. We should not introduce a second hierarchy of solver-specific
map builders.

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

This is a proposed API, not existing functionality. Initial coverage is supported
continuous, real DPP compilations with linear recovery. DGP transformations,
attribute projection, and differentiation through the solver are outside this
contract.

## Construction belongs with the transformation

| Owner | Responsibility |
| --- | --- |
| `CvxAttr2Constr` | Parameter packing and primal expansion for dimension-reducing attributes. |
| `Dcp2Cone` and cone conversions | Recovery of original outputs from introduced variables and constraints, where linear. In particular, `ExactCone2Cone` owns PSD/svec conversion and dual recovery using `SolverInfo`. |
| `ConeMatrixStuffing` / compiled `ParamProb` | Export objective, constraint, and bound affine maps, sparse structures, and variable/constraint slices. Tensor extraction belongs to the program representation. |
| `ConeFormat` | Apply its existing signed row permutation to the compiled program; expose the corresponding dual-coordinate mapping. |
| `ExtractDirectCones` | Preserve removed-row/direct-cone correspondence. The adapter supplies access to the native direct-cone dual outputs. |
| Solver adapter | Declare input names, row splits/stacks, remaining signs, triangular storage, CSC/CSR ordering, and native output blocks. |
| `SolvingChain` / `Problem` | Compose recovery and parameter coordinates, select requested outputs, and package/cache the public result. |

The forward maps already exist implicitly in the compiled program. Export them
after the shared reductions have transformed it; do not propagate a second set
of forward matrices through every expression reduction. The new work in earlier
reductions is principally to expose their parameter-coordinate and recovery
operators.

For example, `PSDToSvecPSD` already constructs the solver-aware PSD transformation
and recovers its dual in `ExactCone2Cone`. Expose that local operator there,
including its transpose where appropriate. SCS and Clarabel adapters should not
independently reconstruct PSD scaling or unpacking. Likewise, `ConeFormat` should
own interleaving and its inverse, rather than each adapter implementing them.

Use the existing chain ID-composition helpers for identity tracing. Add numerical
operators for packing, scaling, scattering, and recovery from multiple auxiliary
constraints: ID maps alone cannot express these. Construct operators from the
same metadata as `invert()`, ideally sharing the operator implementation with
numerical inversion. Do not infer them by probing `invert()` on basis vectors.

## Keep the adapter declarative

A shared layout utility should apply selections, permutations, signs, stacks,
and constant insertion to both affine maps and sparse structures. An adapter
provides the recipe; it does not inspect `reduced_A` or trace upstream inverse
data. This utility is an implementation detail, not another public API.

For SCS and Clarabel, the remaining forward work is largely naming, the final
constraint-matrix sign, and matrix storage. ECOS additionally splits equality
and conic rows. OSQP stacks equality and inequality rows and inserts constant
infinite lower bounds. These operations need no separate map-extraction code.

The recipe is relative to the actual compiled program: conic adapters receive
formatted rows; QP adapters receive unformatted rows. Reuse the resolved chain,
`SolverInfo`, and compilation options. Do not normalize the problem a second
time or redo PSD conversion in the terminal layout utility.

## Public contract

- **Forward maps:** `maps[name] = (M, v)` means `values = M @ p + v`.
  `M` has finite entries. Infinite bound constants belong only in `v`, with zero
  corresponding rows in `M`. The parameter adjoint uses `M.T`, excluding `v`.
- **Coordinates:** `p` uses the compact parameter coordinates chosen by
  compilation, with no trailing constant. Sparse parameters contribute only
  stored entries; symmetric parameters use the compiled triangular packing.
  Expose each block's original parameter ID, logical shape, packed size, offset,
  and exact entry order. Packing and its adjoint must work directly on stored
  entries, without allocating dense parameters or dense packing matrices.
  Unstructured dense parameters retain Fortran order.
- **Structure:** Matrix values follow an explicitly declared CSC/CSR structure
  and triangle. Structural slots remain fixed across parameter assignments,
  including coefficients that evaluate to zero. Bound maps and fixed cone data
  must be included whenever the solver consumes them.
- **Recovery:** Return original-variable IDs, shapes, and output offsets. Use one
  dual entry per object in `constraint.dual_variables`, also carrying its parent
  constraint ID. Reshape output slices in Fortran order. Adapters specify how
  native output blocks form `solver_x` and `solver_y`.
- **Selection:** `variables=None` requests all original primal variables and
  explicit constraint duals; a list selects outputs, preserving relative order
  within each recovery vector. `variables=[]` requests forward maps only.
  Unsupported requested recovery raises an explicit error.

`Problem` is the public entry point; chain composition is internal. Cache against
the actual compilation and parameter layout, independently of parameter values.
Output selection should not require constructing every recovery row. Returned
maps must not expose mutable aliases into cached tensors.

## Implementation scope

Start by exporting the compiled-program maps, adding local recovery operators to
the shared reductions needed by SCS/Clarabel, and composing them in the chain.
Add OSQP/ECOS through layout recipes. Each adapter can opt in; the existing solve
path remains available.

Compare evaluated inputs with the numerical adapter path and structural recovery
with chain inversion before attribute projection. Cover PSD conventions,
auxiliary-constraint recovery, zero crossings, parameter packing, bounds, output
selection, and adjoint identities. These are equivalence checks for the maps,
not tests of the solver's implicit derivative.

CVXPYgen and CVXPYlayers then replace their extraction/recovery bookkeeping.
Their solver calls, code generation, framework operations, and derivative
machinery remain. Reusing the same operators in CVXPY's numerical solve path
can follow incrementally.

## Decisions for review

1. **Recovery scope:** Start with separate primal/dual maps and reject chains
   requiring cross-recovery, or expose a block operator on `[solver_x; solver_y]`
   immediately? Forward-only requests should remain usable when recovery is
   unsupported.
2. **Initial coverage:** Which cone conversions, direct-cone paths, and compiled
   program representations must support export in the first version?

## Appendix: contract sketch

These types illustrate the returned data; names and metadata types are provisional.
The main proposal does not require a particular internal layout-recipe DSL.

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

`M` has shape `(number_of_output_values, total_param_size)`. Its CSR format is
independent of the output matrix's CSC/CSR format. Metadata specifies cone order,
fixed cone parameters, primal/dual input lengths, and native output block order.
Missing objectives or bounds follow the adapter's declared input contract.
Empty recovery selections have zero rows; scalars have shape `()` and size one.
The `d` map is the compiled objective offset, not a solver input; recovering an
original maximization objective also requires its sign.

`param_id_to_size` and `total_param_size` count packed entries, not logical
parameter elements. `ParameterInfo.indices` records the compilation's actual
packing order; consumers must not infer a triangle ordering from shape alone.
Already-packed inputs can be concatenated directly. For logical inputs, packing
gathers the declared entries and its adjoint accumulates cotangents into those
input coordinates using sparse storage when applicable. It does not implicitly
duplicate off-diagonal cotangents into a full symmetric matrix. Any framework
conversion to a different gradient representation must be explicit.

### Example adapter layouts

For the base conic adapter's formatted program, let its exported constraint
coefficients be `A0, b0`. The adapter uses `A = -A0`, `b = b0`. SCS names the
linear objective `c`; Clarabel names it `q`. PSD scaling and cone interleaving
have already happened upstream.

For OSQP's unformatted program, let its coefficients be `AF, bg` with equality
rows first. Its recipe reproduces the existing QP adapter and OSQP input assembly:

```text
A_eq =  AF[:n_eq]       b_eq = -bg[:n_eq]
F    = -AF[n_eq:]       g    =  bg[n_eq:]

A = [A_eq; F]
l = [b_eq; -inf]
u = [b_eq; g]
```

Infinite entries have zero parameter-map rows. Triangular extraction of `P` and
CSC/CSR conversion must transform its values map and structure together. The
same holds for stacking, row selection, and duplicate consolidation. Native
bounds that are retained by compilation require their own map entries; bounds
lowered into constraints are already represented there.

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
