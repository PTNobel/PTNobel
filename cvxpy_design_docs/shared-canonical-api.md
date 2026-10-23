ChatGPT generated:

# Shared canonical parameter API for CVXPY

**Status:** Draft proposal; the API below is not implemented in the sources reviewed here.  
**Updated:** October 23, 2026.  
**Original authors:** Claude Opus 4.6, with prompting from Parth Nobel.  
**Revision:** ChatGPT, with prompting from Parth Nobel.  
**Reviewer:** Max Schaller.

## Proposal in brief

Expose the affine maps that CVXPY compiles between a problem's parameters and a
solver's numerical inputs, together with sparse maps for recovering requested
primal and dual variables. CVXPYgen can generate code from these maps;
CVXPYlayers can evaluate them in its tensor framework.

For a fixed compilation and a packed parameter vector `p`, the contract is:

```text
solver input k:    values[k] = M[k] @ p + v[k]
primal recovery:  primal_flat = R_primal @ solver_x
dual recovery:    dual_flat   = R_dual   @ solver_y
```

Matrix inputs also carry their sparse storage layout. Recovery outputs carry
IDs, shapes, and offsets into the flat vectors. The public entry point belongs
on `Problem`: only that layer has enough information to compose recovery through
the entire reduction chain.

This proposal covers the affine portion of compilation and solution recovery for
supported continuous, real, DPP problems. It does not claim that solving a problem,
differentiating its solution, or every CVXPY reduction is linear.

## Why CVXPY should own this

CVXPY already compiles parametric problem data, but downstream libraries must
interpret internal tensors and repeat parts of the solver adapter:

| Library | Current responsibility that this API would absorb |
| --- | --- |
| CVXPY | Evaluate parameter tensors, apply solver signs and layouts, and invert reductions. |
| CVXPYgen | Extract per-input affine maps and trace primal/dual recovery information. |
| CVXPYlayers | Adapt parameter tensors to solver storage and reconstruct requested outputs in each framework. |

The shared API should make CVXPY the source of truth for these transformations.
Code generation, tensor frameworks, batching, solver calls, and differentiation
through the solve remain downstream responsibilities.

### What has changed since the original draft

- **CVXPY:** `ConeFormat` is now an explicit reduction before conic solver
  interfaces. `ConicSolver.apply()` expects formatted rows;
  `format_constraints()` is a deprecated compatibility wrapper. QP interfaces
  consume unformatted rows and apply their own equality/inequality signs.
  PSD triangle selection and scaling must follow the compilation's solver
  conventions rather than be attributed entirely to `format_constraints()`.
- **CVXPYgen:** Solver adapters now live in `cvxpygen/solvers/`, with shared
  affine-map extraction in `_interface.py`. `Canonicalizer` in
  `canonicalizer.py` owns parameter and recovery extraction. It already uses
  `chain.compose_var_id_map()` for primal IDs and traverses inverse data for
  dual IDs; the older description of all recovery as fixed
  `inverse_data[-1..-4]` indexing is outdated. It still depends on internal
  offsets, constraint ordering, and some positional inverse-data access.
- **CVXPYlayers:** `get_solver_ctx()` pre-permutes reduced maps to CSR order
  for its built-in non-DIFFCP paths. DIFFCP retains its CSC path. Structural
  conversion is therefore already partly done at construction time, while
  interfaces still adapt evaluated values for their solver. Recovery and
  dimension-reducing parameter attributes require additional bookkeeping.

These observations refer to the source snapshots linked at the end, rather than
to a particular released version.

## Public API

The following signatures describe the proposed contract; names remain open.

```python
canon = problem.get_canonical_maps(
    solver=cp.CLARABEL,
    variables=[x, constraint.dual_variables[0]],
    # Compilation options, as applicable to get_problem_data().
)

M, v = canon.maps["q"]
q = M @ p + v

M, v = canon.maps["A"]
A_values = M @ p + v
layout = canon.structures["A"]
# Reconstruct A from A_values and layout, in the declared CSC or CSR order.
```

`variables=None` requests all original primal variables and all dual variables
of explicit constraints. An explicit list requests only those outputs. Within
each of `primal_vars` and `dual_vars`, entries retain the caller's relative
order. Each entry identifies its source object, so callers can restore a mixed
primal/dual request order. Unknown or unsupported outputs raise an error rather
than disappear silently. Attribute-implied constraints do not acquire public
dual variables merely because canonicalization introduces constraints for them.

An empty list requests forward maps only. Selection changes recovery rows, not
the compiled optimization problem. It can avoid materializing large unneeded
outputs, although the performance benefit depends on the framework and workload.

### Returned data

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

    parameters: list[Parameter]
    param_id_to_col: dict[int, int]
    param_id_to_size: dict[int, int]
    total_param_size: int

    # Solver/cone metadata, including cone order and fixed cone parameters.
    solver_metadata: dict[str, object]
```

Recovery metadata describes the result, so consumers need no `is_symmetric`,
`is_psd`, or cone-type dispatch to interpret it. All output slices are reshaped
in Fortran order, including matrix-valued duals. A scalar has shape `()` and
size one. Empty recovery requests produce matrices with zero rows and metadata
lists with no entries.

`solver_metadata` must specify primal and dual vector lengths, cone order, and
how native solver outputs form `solver_x` and `solver_y`. If a solver returns
separate equality and inequality dual vectors, their concatenation order is
part of this contract. Recovery inputs use the solver API's returned coordinates,
after any internal solver scaling has been undone.

### Parameter coordinates

The coordinate system must be public and independent of internal parameter IDs
introduced by reductions. The proposed default is the concatenation of the
original parameters in `canon.parameters` order, each flattened in Fortran order.
`param_id_to_col` and `param_id_to_size` describe those blocks;
`total_param_size` is their total length. There is no constant sentinel ID or
trailing `1.0` in this public vector.

Any linear packing introduced by attribute lowering is composed into the maps.
For example, a symmetric parameter may have a reduced internal representation,
but callers should not have to discover that representation through private
attributes. Values must satisfy the original parameter attributes; the API does
not define an extension to invalid parameter values. A compact representation
for sparse or symmetric parameters is a possible additional interface, discussed
below.

Compilation must preserve affine dependence on these coordinates. Paths that
evaluate nonlinear parameter expressions before compilation cannot advertise
the result as an affine map from the original parameters. In particular, DGP's
parameter logs and primal exponentials require a separate transformation
contract; the initial implementation should reject `gp=True` rather than hide
those operations in `M` or `R`.

## Forward maps and solver layout

Every map has shape `(number_of_output_values, total_param_size)`. Vector inputs
produce dense flat vectors. Matrix inputs produce the data array in the exact
order declared by `MatrixStructure`; the sparse format of `M` itself is
independent of the output matrix's format.

The sparsity structure is fixed across valid parameter values. A coefficient
that evaluates to zero remains in its structural slot. Compilation must not
infer sparsity from one numerical parameter assignment or drop slots needed by
later assignments. Duplicate contributions must be combined into the declared
storage layout.

Keys use solver-native names, with `d` reserved for the compiled objective's
constant offset:

| Solver | Target form | Map keys |
| --- | --- | --- |
| SCS | Minimize `0.5 x'Px + c'x`, with `Ax + s = b`, `s in K` | `P` when used, `c`, `d`, `A`, `b` |
| Clarabel | Minimize `0.5 x'Px + q'x`, with `Ax + s = b`, `s in K` | `P` when used, `q`, `d`, `A`, `b` |
| ECOS | Minimize `c'x`, with `Ax = b`, `Gx + s = h`, `s in K` | `c`, `d`, `A`, `b`, `G`, `h` |
| OSQP | Minimize `0.5 x'Px + q'x`, with `l <= Ax <= u` | `P`, `q`, `d`, `A`, `l`, `u` |

These are numerical solver inputs, not necessarily the intermediate dictionary
returned by `get_problem_data()`. OSQP, for example, currently stacks equality
and inequality blocks inside `solve_via_data()`. `d` is bookkeeping for objective
recovery and is not passed to the solver. Recovering the original objective
also requires accounting for any maximization-to-minimization sign change.

### Conic interfaces

For the formatted program used by the base conic adapter,
`apply_parameters()` returns coefficients `A0, b0` with
`A0 @ x + b0 in K`. Its solver inputs are `A = -A0`, `b = b0`.
The maps must reproduce that adapter's result, including the already-applied
cone layout and the solver's quadratic storage convention. ECOS additionally
splits equality and conic-inequality rows.

This is a specification of the result, not a requirement to extract rows from
`reduced_A` in every backend. CVXPY's `ParamProb` abstraction also allows
representations other than the traditional stuffed sparse tensors.

### OSQP

For the unformatted program consumed by `QpSolver`, write the constraint
coefficients as `AF` and the constant vector as `bg`. With equality rows first:

```text
A_eq =  AF[:n_eq]       b_eq = -bg[:n_eq]
F    = -AF[n_eq:]       g    =  bg[n_eq:]

A = [A_eq; F]
l = [b_eq; -inf]
u = [b_eq; g]
```

The infinite lower-bound entries have zero map rows and constant offsets of
`-inf`. Apply triangular extraction and storage ordering to `P` as required by
the solver's setup/update interface; copying an internal full `P` map unchanged
is not a sufficient contract.

### Constants and bounds

Each map is an `(M, v)` pair. `M` contains finite coefficients; `v` may contain
infinite entries for absent bounds. Constant infinite entries have zero rows
in `M`, so their parameter derivative is zero. Infinite constants must be
extracted without introducing `inf` or `NaN` into the linear map.

Keeping `v` separate prevents a transpose multiply from encountering
`inf * 0` through a homogeneous constant column. For a forward-map cotangent
`g`, the parameter contribution is `M.T @ g`; the constant offset is excluded.
Code generators may initialize constant entries once and update only entries
that depend on parameters.

Native variable bounds, including `lb_tensor` and `ub_tensor` when applicable,
must be included if the target adapter passes them to the solver. If an adapter
instead lowers bounds into constraints, they are represented in those constraint
maps. Bounds must not be silently omitted or duplicated.

## Recovery must compose through the chain

The solver adapter knows its output layout. It does not, by itself, know how
to reconstruct the original problem's variables. `Problem` must compose the
adapter's recovery with the applicable reductions, including variable attribute
lowering, constraint ID mappings, and cone conversions.

Build these maps structurally from reduction metadata and local recovery
operators. Do not probe `invert()` on basis vectors: that would be expensive,
would execute status-dependent code, and would not establish linearity.

Typical local operators include:

| Operation | Sparse recovery block |
| --- | --- |
| Variable or dual slicing | Row selection |
| Symmetric primal expansion | Duplicate unscaled triangular entries into both matrix positions |
| Diagonal or sparse variable expansion | Scatter into the original shape |
| PSD dual unpacking | Expand the solver's triangle, with `1/sqrt(2)` off-diagonal scaling for scaled svec conventions |
| Batched SOC or exponential-cone duals | Undo argument interleaving and select each dual-variable block |

Symmetric primal packing and scaled PSD dual svec are different conventions.
For SCS, scaled PSD dual entries follow lower-triangular column order; Clarabel
uses upper-triangular column order. Recovery must compose the convention chosen
by the actual chain and return the same full matrix representation to callers.

Use one `DualInfo` per object in `constraint.dual_variables`, identified by both
dual-variable ID and parent constraint ID. This resolves the earlier ambiguity
between a whole constraint dual and an individual dual variable. CVXPYgen can
group these entries by constraint; CVXPYlayers can select individual entries.

The initial split into `R_primal @ solver_x` and `R_dual @ solver_y` is supported
only when the selected chain admits that separation. A dualizing reduction may
exchange primal and dual roles; other reductions may not admit these recovery
maps at all. Such chains need either a more general block recovery operator or
an explicit unsupported error. Infeasibility certificates, missing solver
outputs, status handling, projection of numerical solutions onto attributes,
and nonlinear inverse transformations are outside this linear recovery contract.

## Implementation and adoption

1. **Add internal map-building hooks.** Solver adapters provide forward maps,
   solver output layouts, and local recovery operators. Relevant reductions
   expose composable recovery information. An unsupported internal hook may
   return `NotImplemented`; the public API raises a descriptive error.
2. **Add `Problem.get_canonical_maps()`.** Reuse the compilation chain, compose
   parameter coordinates and recovery, and select requested outputs. Construct
   selected recovery rows directly where practical. Returned arrays must not
   alias mutable cached tensors that a downstream library can reorder.
3. **Start with SCS and Clarabel, then OSQP and ECOS.** Support each solver only
   for the chains and cone families whose forward and recovery contracts are
   covered. Keep CVXPY's existing solve path while establishing equivalence.
4. **Migrate downstream extraction.** CVXPYgen replaces affine-map and recovery
   extraction in its canonicalizer and adapter mixins. It retains solver code
   generation, settings, update adjacency, and derivative machinery.
   CVXPYlayers replaces internal tensor/layout and recovery bookkeeping while
   retaining framework conversion, batching, warm starts, and solver derivatives.
5. **Share CVXPY's own implementation when practical.** Reusing map construction
   in the numerical solve path and `apply_param_jac()` can reduce duplication.
   Adoption should be incremental and measured; neither deleting all downstream
   solver interfaces nor changing CVXPY's solve path is required for this API.

Cache entries must be tied to the actual compilation: problem structure,
parameter coordinate layout, solver, backend, and compilation options that
affect the chain. Parameter value changes reuse the maps. Output selections can
reuse compilation without requiring a fully materialized recovery map.

### Acceptance criteria

- Compare evaluated maps with existing numerical adapter inputs across multiple
  parameter assignments, including zero crossings, constants, sparse parameters,
  and dimension-reducing attributes. Check matrix layout as well as values.
- Compare recovery with supported reduction-chain inversion for scalar, vector,
  matrix, symmetric, sparse, and batched cone outputs. Include both PSD storage
  conventions, multi-dual constraints, selection ordering, and empty selections.
  Compare the structural recovery before any numerical attribute projection.
- Check adjoint identities for forward and recovery maps. When comparing against
  `apply_param_jac()`, account for its internal coordinates, sign conventions,
  and supported objective terms; it is not automatically the transpose of every
  solver-native map.
- Check infinite bound offsets, fixed structural zeros, cache reuse, and explicit
  errors for unsupported chains or requested outputs.

These checks establish map correctness. The solver's implicit derivative remains
a separate operation: `M.T` and `R.T` alone do not differentiate the optimization
problem.

## Decisions still needed

1. **Parameter storage:** Is the default full original-parameter coordinate
   system sufficient, or should the first version also expose a compact packing
   operator for sparse and symmetric parameters? Either choice needs an explicit
   packing and adjoint contract; internal reduced IDs should not become public.
2. **Recovery generality:** Should the first version support a single operator
   on `[solver_x; solver_y]`, or reject chains that require cross-recovery and
   add block operators later? How should unavailable dual recovery be requested
   separately from otherwise supported forward maps?
3. **Coverage:** Which cone families, direct-cone paths, and compilation backends
   ship initially? DGP, complex realification, and other transformations need
   explicit contracts before expanding beyond the initial real DPP scope.
4. **Surface and metadata:** Finalize method names, compilation keyword arguments,
   solver/cone metadata types, and how the contract is versioned before making
   it public.

The earlier alternatives still do not address the shared need: returning only
evaluated data cannot support framework-native evaluation or code generation;
moving all of CVXPYgen's solver adapters into CVXPY also moves code-generation
policy; making CVXPYlayers depend on CVXPYgen leaves ownership of CVXPY's
compilation conventions downstream.

## Source snapshots

Reviewed against upstream sources on October 23, 2026:

- **CVXPY** ([`8280648`](https://github.com/cvxpy/cvxpy/tree/8280648bd1f70fce701d362de4f58120f940c2f3)):
  [cone formatting](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/cone_format.py),
  [conic adapter](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/solvers/conic_solvers/conic_solver.py),
  [QP adapter](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/solvers/qp_solvers/qp_solver.py),
  [OSQP inputs](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/solvers/qp_solvers/osqp_qpif.py),
  [parametric cone program](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/dcp2cone/cone_matrix_stuffing.py),
  [attribute lowering](https://github.com/cvxpy/cvxpy/blob/8280648bd1f70fce701d362de4f58120f940c2f3/cvxpy/reductions/cvx_attr2constr.py).
- **CVXPYgen** ([`79c0255`](https://github.com/cvxgrp/cvxpygen/tree/79c0255236b78d271285c3689a9ff0cd8079ae30)):
  [canonicalizer](https://github.com/cvxgrp/cvxpygen/blob/79c0255236b78d271285c3689a9ff0cd8079ae30/cvxpygen/canonicalizer.py),
  [shared adapter extraction](https://github.com/cvxgrp/cvxpygen/blob/79c0255236b78d271285c3689a9ff0cd8079ae30/cvxpygen/solvers/_interface.py).
- **CVXPYlayers** ([`bd9caf9`](https://github.com/cvxpy/cvxpylayers/tree/bd9caf9826d758722a5c42b8a51775fcbc037ffe)):
  [solver context construction](https://github.com/cvxpy/cvxpylayers/blob/bd9caf9826d758722a5c42b8a51775fcbc037ffe/src/cvxpylayers/interfaces/__init__.py),
  [parameter and recovery metadata](https://github.com/cvxpy/cvxpylayers/blob/bd9caf9826d758722a5c42b8a51775fcbc037ffe/src/cvxpylayers/utils/parse_args.py),
  [Moreau interface](https://github.com/cvxpy/cvxpylayers/blob/bd9caf9826d758722a5c42b8a51775fcbc037ffe/src/cvxpylayers/interfaces/moreau_if.py).
