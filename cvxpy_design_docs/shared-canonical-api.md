# Design Doc: Shared Canonical Parameter API for CVXPY

## Status: Draft

## Authors
Claude Opus 4.6, with prompting from Parth Nobel

## Summary

CVXPY should expose a public API on its solver interfaces that provides
**pre-decomposed parametric affine maps** from user parameters to
solver-specific canonical parameters (P, q, A, b, etc.). This would
upstream the core decomposition logic currently duplicated in CVXPYgen's
`solvers.py` and allow CVXPYlayers to drastically simplify its solver
interface layer.

## Background

### The Problem Both Libraries Solve

Both CVXPYgen and CVXPYlayers consume CVXPY's `ParamConeProg`, the
internal representation of a compiled DPP-compliant problem. A
`ParamConeProg` stores three sparse "tensor" matrices that encode affine
maps from a flat parameter vector to problem data:

```
reduced_P.reduced_mat @ param_vec  -->  P values (CSC-ordered)
q                     @ param_vec  -->  [c; d]  (objective + constant offset)
reduced_A.reduced_mat @ param_vec  -->  values of augmented [A | b] (CSC-ordered)
```

Both libraries then need to go from these raw data vectors to
solver-specific inputs. Every solver has a different convention:

| Solver   | Form                                     | Canonical params      |
|----------|------------------------------------------|-----------------------|
| SCS      | min c'x s.t. Ax+s=b, s in K             | P, c, d, A, b         |
| Clarabel | min (1/2)x'Px+q'x s.t. Ax+s=b, s in K   | P, q, d, A, b         |
| ECOS     | min c'x s.t. Ax=b, Gx<=h                | c, d, A, b, G, h      |
| OSQP     | min (1/2)x'Px+q'x s.t. l<=Ax<=u         | P, q, d, A, l, u      |

The conversion from the raw `ParamConeProg` tensors to these
solver-specific parameters involves:

1. **Splitting** the augmented constraint matrix `[A | b]` into matrix
   and vector parts.
2. **Sign correction** (CVXPY stores `Ax + b in K`; solvers typically
   want `Ax + s = b` or `Ax <= b`, requiring negation of A).
3. **Equality/inequality splitting** for solvers that distinguish them
   (ECOS: A/b vs G/h; OSQP: bounds l/u).
4. **Structural formatting** (PSD scaling, SOC reordering) via
   `format_constraints()`.
5. **Matrix format conversion** (CSC vs CSR) depending on the solver.

### How This Is Done Today

**In CVXPY itself**: The solver `apply()` methods do steps 1-4 for
concrete numerical values. `ConicSolver.apply()` calls
`apply_parameters()` to get `(c, d, A, b)`, then negates A.
`QpSolver.apply()` additionally splits equality/inequality rows.
`format_constraints()` handles structural formatting. But these all
operate on evaluated numbers, not on the parametric affine maps.

**In CVXPYgen**: The `SolverInterface.get_affine_map()` method
(in `solvers.py`) reimplements the same decomposition, but on the
parametric affine maps. For each canonical parameter (P, q, A, b, etc.),
it extracts the relevant rows from `reduced_P.reduced_mat`,
`param_prob.q`, and `reduced_A.reduced_mat`, applies sign corrections,
and produces a per-parameter affine map:
`p_id_to_mapping[p_id] @ param_vec = canonical_param_values`. This is
the core logic that could be upstreamed.

**In CVXPYlayers** (v1.0.3, master): CSC-to-CSR conversion has been
centralized into `solver_utils.py` (`convert_to_csr()`), and the row
permutation is applied once during `get_solver_ctx()` in
`interfaces/__init__.py`. Each solver interface still independently
reconstructs the augmented `[A | b]` matrix, splits off `b`, negates
`A`, and converts to the solver's preferred format, but the CSC-to-CSR
permutation no longer happens at forward time. Instead, `param_prob`'s
`reduced_A.reduced_mat` and `reduced_P.reduced_mat` are pre-permuted
so that `matrix @ params` produces values directly in CSR order.

Additional recent changes on master:
- **scipy sparse matmul** for CPU (80-200x faster than torch sparse CSR)
- **`solve_only_batch()`** for DIFFCP when gradients aren't needed
- **warm-start** support for Moreau solver
- **parametric `quad_form(x, P)`** via `quad_form_dpp_scope()`
- **full dual variable support** (see Dual Variable Recovery below)

### The Duplication

The sign correction and splitting logic exists in **three places**:

1. `cvxpy/reductions/solvers/conic_solvers/conic_solver.py` --
   `apply()` negates A for concrete values
2. `cvxpygen/solvers.py` -- `get_affine_map()` applies sign=-1 to the
   affine map rows for A
3. `cvxpylayers/interfaces/*.py` -- each solver reconstructs and negates
   A from CSC values at forward-pass time

## Proposal

### Core Idea

Every operation in the CVXPY solve pipeline — from user parameters to
solver inputs, and from solver outputs back to user variables — is a
**linear operation** that can be represented as a sparse matrix
multiply. CVXPY should expose the complete set of these matrices as
a public API.

The entire data flow becomes six sparse matmuls:

```
Forward (parameters → solver inputs):
    P_map  @ param_vec  →  P values
    q_map  @ param_vec  →  q/c vector
    A_map  @ param_vec  →  A values     (sign-corrected)
    b_map  @ param_vec  →  b vector

Recovery (solver outputs → user variables):
    R_primal  @ solver_x  →  [var1_flat; var2_flat; ...]
    R_dual    @ solver_y  →  [dual1_flat; dual2_flat; ...]
```

The recovery matrices encode all normalization: slicing from
`var_id_to_col`, symmetric svec-to-full expansion, PSD 1/sqrt(2)
off-diagonal scaling, Fortran-order permutation — everything baked
into constant sparse matrix entries computed once at compilation time.

### Why This Works

Both primal and dual recovery are purely linear:

| Recovery step | Operation | Matrix representation |
|---|---|---|
| Variable slicing | `x[offset:offset+size]` | Rows of identity (selection) |
| Symmetric unpacking | Upper-tri → full matrix | Sparse matrix with 0/1 entries |
| PSD dual svec unpacking | Scale off-diag by 1/sqrt(2), expand | Sparse matrix with 1 and 1/sqrt(2) entries |
| Multi-dual offset | Slice within constraint dual | Row selection |
| Fortran reshape | Column-major reordering | Permutation matrix |

Similarly, `format_constraints()` — which handles PSD sqrt(2) scaling,
SOC interleaving, ExpCone reordering — is a block-diagonal sparse
matrix left-multiplication on the A tensor. It is already implemented
as a `LinearOperator` in CVXPY.

There are **zero nonlinear operations** in either direction. (The GP
`exp()` transform is applied separately after recovery and is
explicitly outside the linear pipeline.)

### Proposed API

```python
@dataclass
class CanonicalMaps:
    """Complete set of sparse matrices for a compiled parametric problem.

    Forward maps (parameter → solver input):
        map @ param_vec = solver_input_values
        where param_vec is the flat parameter vector with trailing 1.0.

    Recovery maps (solver output → user variables):
        R @ solver_vec = user_variable_values (flat, concatenated)
    """

    # --- Forward maps (parameter → solver input) ---

    # Per-canonical-parameter affine maps (sparse CSR matrices).
    maps: dict[str, scipy.sparse.csr_array]
    # e.g. {"P": ..., "q": ..., "d": ..., "A": ..., "b": ...}

    # Sparsity structure of output matrices (CSC format).
    # For matrix params (P, A, G): (indices, indptr, shape).
    structures: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int]]]

    # --- Recovery maps (solver output → user variables) ---

    # Primal recovery: R_primal @ solver_x → flat user variables.
    # Encodes var_id_to_col slicing + symmetric unpacking.
    R_primal: scipy.sparse.csr_array

    # Dual recovery: R_dual @ solver_y → flat user duals.
    # Encodes constraint slicing + PSD svec unpacking with 1/sqrt(2).
    R_dual: scipy.sparse.csr_array

    # Per-variable metadata for interpreting the flat output.
    primal_vars: list[VarInfo]   # shape, offset into R_primal output
    dual_vars: list[DualInfo]    # shape, offset into R_dual output,
                                 # constraint_id, cone_type

    # --- Problem metadata ---

    cone_dims: ConeDims
    constraints: list[Constraint]

    # Parameter info
    param_id_to_col: dict[int, int]
    param_id_to_size: dict[int, int]
    parameters: list[Parameter]
    total_param_size: int


@dataclass
class VarInfo:
    """Metadata for one primal variable in the recovery output."""
    var_id: int
    shape: tuple[int, ...]
    offset: int              # start index in R_primal output
    size: int                # number of elements in output
    is_symmetric: bool


@dataclass
class DualInfo:
    """Metadata for one constraint's dual in the recovery output."""
    constraint_id: int
    shape: tuple[int, ...]
    offset: int              # start index in R_dual output
    size: int                # number of elements in output
    cone_type: str           # "zero", "nonneg", "soc", "psd", etc.
    is_psd: bool
```

### Where It Lives

```python
# In conic_solver.py
class ConicSolver(Solver):
    def get_canonical_maps(self, param_prog: ParamConeProg) -> CanonicalMaps:
        """Return the complete set of sparse matrices.

        Forward maps: decomposes param_prog's monolithic tensors into
        per-canonical-parameter affine maps with solver-specific sign
        conventions applied.

        Recovery maps: builds sparse matrices that map the solver's
        flat primal/dual vectors to per-variable/per-constraint values,
        with svec unpacking and scaling baked in.
        """
        ...
```

Each solver subclass (SCS, Clarabel, OSQP, ECOS) would either inherit
the base implementation or override for solver-specific behavior (e.g.,
PSD storage format, equality/inequality splitting).

### How Recovery Matrices Are Built

**R_primal** (shape: `sum(var_output_sizes) x n_solver_vars`):

For each variable `v` with `col = var_id_to_col[v.id]`:

- **Non-symmetric**: Identity block selecting `x[col:col+v.size]`.
  Rows `[offset:offset+v.size]`, columns `[col:col+v.size]`,
  all entries = 1.

- **Symmetric** (n x n): The svec-to-full expansion matrix.
  Input: `n*(n+1)/2` entries from the solver's x vector.
  Output: `n*n` entries (full symmetric matrix in Fortran order).
  Entries: 1 on diagonal, two entries per off-diagonal (both = 1,
  writing to both `(i,j)` and `(j,i)`).

**R_dual** (shape: `sum(dual_output_sizes) x m_solver_dual`):

For each constraint `c` at `dual_offset` in the solver's dual vector:

- **Non-PSD**: Identity block selecting
  `y[dual_offset:dual_offset+c.size]`. All entries = 1.

- **PSD** (n x n): The svec unpacking matrix.
  Input: `n*(n+1)/2` svec entries from the solver's dual vector.
  Output: `n*n` entries (full symmetric matrix in Fortran order).
  Entries: 1 on diagonal positions, `1/sqrt(2)` on off-diagonal
  positions (two entries per off-diagonal svec element, writing to
  both `(i,j)` and `(j,i)`).

  For SCS (lower-triangular svec): columns ordered by lower-tri.
  For Clarabel (upper-triangular svec): columns ordered by upper-tri.
  The *output* is identical; only the column ordering differs.

### Forward Maps: Base Implementation (SCS, Clarabel)

1. **P map**: `reduced_P.reduced_mat` (unchanged)
2. **q/c map**: `param_prob.q[:-1, :]` (all rows except last)
3. **d map**: `param_prog.q[[-1], :]` (last row only)
4. **A map**: Rows of `reduced_A.reduced_mat` corresponding to matrix
   columns, with sign = **-1** applied.
5. **b map**: Rows of `reduced_A.reduced_mat` corresponding to the
   last column, scattered to dense shape `(m, total_param_size + 1)`.
6. **Structures**: `reduced_P.problem_data_index` for P;
   constraint indices/indptr (excluding last column) for A.

### Forward Maps: OSQP Override

1. **P map**: Same as base.
2. **q map**: `param_prog.q[:-1, :]`
3. **d map**: `param_prog.q[[-1], :]`
4. **A map**: Matrix rows from `reduced_A`, with selective sign flip
   (negate inequality rows only).
5. **l map**: Equality vector rows, negated, augmented with zero rows
   for inequalities.
6. **u map**: All vector rows, with per-row sign vector.

### Forward Maps: ECOS Override

Splits constraints into equality (A, b) and inequality (G, h) groups.

## How This Simplifies Downstream Libraries

### CVXPYgen

CVXPYgen's `solvers.py` (~800 lines of `get_affine_map()` logic across
5 solver classes) and `get_dual_variable_info()` (~60 lines of fragile
`inverse_data` chain tracing) would be replaced by a single call:

```python
# Before (cvxpygen/cpg.py + cvxpygen/solvers.py)
solver_interface = SCSInterface(data, param_prob, ...)
for p_id in solver_interface.canon_p_ids:
    affine_map = solver_interface.get_affine_map(p_id, param_prob, ...)
    # ... process ...
dual_info = get_dual_variable_info(problem, inverse_data, ...)

# After
canon = solving_chain.solver.get_canonical_maps(param_prog)
# Forward maps -- generate C code for each sparse matmul:
for p_id, mapping in canon.maps.items():
    codegen_sparse_matmul(p_id, mapping)
# Primal recovery -- generate C code for R_primal @ x:
codegen_sparse_matmul("primal_recovery", canon.R_primal)
# Dual recovery -- generate C code for R_dual @ y:
codegen_sparse_matmul("dual_recovery", canon.R_dual)
```

The entire `SolverInterface` class hierarchy and the fragile
`inverse_data[-4]` indexing in CVXPYgen would be deleted. CVXPYgen
keeps its sparsity optimization, adjacency tracking, and C codegen.

### CVXPYlayers

CVXPYlayers' entire `_recover_results()` function — with its
`if/elif` dispatch on `unpack_fn`, `_unpack_primal_svec()`,
`_unpack_svec()`, `_reshape_fortran()`, `_svec_to_symmetric()` —
would become a single sparse matmul:

```python
# Current: ~100 lines of recovery logic per framework
for var in ctx.var_recover:
    if var.source == "primal":
        data = primal[..., var.primal]
    else:
        data = dual[..., var.dual]
    if var.unpack_fn == "svec_primal":
        result = _unpack_primal_svec(data, var.shape[0], batch)
    elif var.unpack_fn == "svec_dual":
        result = _unpack_svec(data, var.shape[0], batch)
    elif var.unpack_fn == "reshape":
        result = _reshape_fortran(data, batch + var.shape)
    ...

# Proposed: one sparse matmul + reshape
primal_flat = self.R_primal_torch @ solver_x   # all vars at once
dual_flat = self.R_dual_torch @ solver_y       # all duals at once
# Split into per-variable tensors using VarInfo/DualInfo offsets
results = [
    primal_flat[..., vi.offset:vi.offset+vi.size].reshape(vi.shape)
    for vi in canon.primal_vars
]
```

Key simplifications:

1. **Unified recovery path.** No per-cone-type dispatch. The sparse
   matrix encodes PSD unpacking, symmetric expansion, scaling — all
   in its constant entries. One matmul handles everything.
2. **Framework-agnostic.** The R matrices are scipy sparse at setup
   time, converted once to torch CSR / JAX BCSR / MLX. No need for
   per-framework `_unpack_svec()` / `_svec_to_symmetric()`.
3. **Differentiable.** Sparse matmul is differentiable in all
   frameworks. Gradients flow through `R.T` automatically —
   no custom backward needed for recovery.
4. **No CSC-to-CSR permutation.** Forward maps can be pre-arranged
   for the solver's preferred format. The negation of A is baked in.
5. **No b extraction.** The b map produces b directly.

The solver interfaces also simplify: the `torch_to_data()` methods
become just sparse matmuls, no reconstruction from CSC indices.

### CVXPY's Own Solvers

CVXPY's `apply_parameters()` + solver `apply()` could also adopt
the same matrices internally, but this is optional — the existing
path is well-tested and performant. The new API is additive.

## What Stays in CVXPYgen

Even with this API, CVXPYgen would retain:

- **User sparsity masks**: Pruning columns for known-zero parameter
  entries (the `sparsity` attribute on Parameters).
- **Adjacency tracking**: `user_p_name_to_canon_outdated` for
  selective updates.
- **C code generation**: The actual codegen for evaluating affine maps,
  reconstructing sparse matrices, and calling solver C APIs.
- **Gradient support**: The two-stage canonicalization for non-OSQP
  gradient computation.
- **Explicit solver**: The MPQP piecewise-affine solution map.

## What Stays in CVXPYlayers

CVXPYlayers would retain:

- **Framework-specific tensor operations**: Converting scipy sparse
  maps to torch CSR / JAX BCSR / MLX arrays; scipy-backed CPU matmul
  (`_ScipySparseMatmul`).
- **Batching logic**: Broadcasting unbatched parameters, batch
  dimension handling.
- **GP support**: Log-space transformations via `Dgp2Dcp`.
- **Solver-specific solve/derivative calls**: The actual solver
  invocation, backward pass, warm-start, `solve_only_batch()`.
- **Recovery matrix conversion**: Converting `R_primal` / `R_dual`
  scipy sparse matrices to framework-specific sparse tensors
  (torch CSR, JAX BCSR, MLX).
- **Parametric quad_form**: The `quad_form_dpp_scope()` monkey-patch
  for parametric `quad_form(x, P)` with PSD parameters.

## Dual Variable Recovery

### The Problem

Dual variable recovery -- mapping the solver's flat dual vector back to
per-constraint values that the user can access -- is implemented in all
three codebases with different approaches and different levels of
fragility.

### How It Works Today

**In CVXPY**: The reduction chain's `invert()` method handles dual
recovery in three layers:

1. **Solver `invert()`** splits the flat dual vector at the
   Zero/inequality boundary using `cone_dims.zero` as the split point.
   It calls `utilities.get_dual_values()` to walk the constraint list
   and slice the vector into per-constraint chunks. Solver-specific
   `extract_dual_value()` methods handle special cases (PSD cone
   storage format differences between SCS lower-triangular and
   Clarabel upper-triangular).

2. **`ConeMatrixStuffing.invert()`** remaps constraint IDs from the
   stuffed problem back to the original problem using `cons_id_map`
   and reshapes dual values to match the user's constraint shapes.

3. **Upstream reductions** (Dcp2Cone, CvxAttr2Constr, etc.) each have
   their own `cons_id_map` that traces constraint IDs backward through
   the chain.

**In CVXPYgen**: `get_dual_variable_info()` (cpg.py, line 394)
reimplements the same tracing logic, but produces a static mapping
instead of applying it to concrete values. It:

1. Manually indexes into `inverse_data` using hardcoded offsets
   (`inverse_data[-1]`, `[-2]`, `[-3]`, `[-4]`) with different
   ordering for conic vs quadratic solver types.
2. Chains the `cons_id_map` dictionaries to trace each user constraint
   ID to its canonical constraint ID.
3. Computes per-constraint offsets into the solver's dual vector(s).
4. Handles `dual_var_split` (ECOS/QOCO have separate `y`/`z` vectors;
   SCS/Clarabel/OSQP have a single dual vector).
5. Produces a `DualVariableInfo` with `name_to_offset`, `name_to_vec`,
   `name_to_indices`, `name_to_shape`.

This is some of the most fragile code in CVXPYgen. It depends on the
exact ordering of reductions in CVXPY's chain and breaks if reductions
are added, removed, or reordered. The hardcoded `inverse_data[-4]`
indexing with different semantics for conic vs quadratic solvers is
particularly brittle.

**In CVXPYlayers** (v1.0.3, master): Full dual variable support has
been implemented with a cleaner approach that avoids tracing through
`inverse_data` chains entirely. Users pass `constraint.dual_variables[N]`
objects in the `variables` list when constructing `CvxpyLayer`:

```python
layer = CvxpyLayer(
    problem,
    parameters=[c, b],
    variables=[x, eq_con.dual_variables[0]],  # mix primals and duals
)
x_opt, eq_dual = layer(c_t, b_t)
```

The implementation in `parse_args.py` uses three key functions:

1. **`_build_dual_var_map()`**: Creates a `dict[int, Constraint]`
   mapping dual variable IDs to their parent constraints by iterating
   over all constraints' `.dual_variables`.

2. **`_build_constr_id_to_slice()`**: Directly computes constraint-to-
   slice mappings from `param_prob.constraints` using CVXPY's canonical
   cone ordering (Zero -> NonNeg -> SOC -> ExpCone -> PSD -> PowCone3D).
   No `inverse_data` or `cons_id_map` tracing needed.

3. **`_build_dual_recovery()`**: Creates `VariableRecovery` with
   `source="dual"`, handling PSD duals (svec format with 1/sqrt(2)
   off-diagonal scaling), multi-dual constraints (SOC has 2 duals,
   ExpCone has 3), and per-dual-variable offsets within a constraint.

The `VariableRecovery` dataclass now has active fields:

```python
@dataclass
class VariableRecovery:
    primal: slice | None
    dual: slice | None
    shape: tuple[int, ...]
    is_symmetric: bool = False
    is_psd_dual: bool = False
    source: Literal["primal", "dual"] = "primal"
    unpack_fn: Literal["reshape", "svec_primal", "svec_dual"] = "reshape"
```

Dual support covers all cone types (Zero, NonNeg, SOC, ExpCone, PSD,
PowCone3D) across PyTorch, JAX, and MLX backends, with gradients
verified via `torch.autograd.gradcheck`. SOC duals are blocked for
Moreau (raises `ValueError`).

### Two Approaches to Dual Recovery: Comparison

| Aspect | CVXPYgen | CVXPYlayers |
|---|---|---|
| **Approach** | Traces `cons_id_map` through `inverse_data[-1..-4]` | Uses `param_prob.constraints` directly |
| **Fragility** | High -- hardcoded reduction chain indices, different for conic vs QP | Low -- only depends on `ParamConeProg.constraints` ordering |
| **Multi-dual** | Not clear if supported | Full support (SOC=2 duals, ExpCone=3) |
| **PSD handling** | Via solver-specific `extract_dual_value()` | Via `unpack_fn="svec_dual"` with 1/sqrt(2) scaling |
| **Solver split** | Explicit `dual_var_split` / `dual_var_names` per solver | Not needed -- works on the unified dual vector |
| **Output** | `DualVariableInfo` with offset/vector/shape dicts | `VariableRecovery` with slice + unpack_fn |

The CVXPYlayers approach is significantly simpler because it works
directly with the `ParamConeProg`'s constraint list, which already
has the canonical ordering baked in after `format_constraints()`.

### Proposed: `R_dual` Recovery Matrix

Rather than exposing a `constr_id_to_dual_slice` dictionary (the
intermediate step CVXPYlayers computes in `_build_constr_id_to_slice()`),
the `CanonicalMaps` object includes a single sparse matrix `R_dual`
that does the *complete* dual recovery in one matmul. This is the
same philosophy as `R_primal`: bake all the linear operations
(slicing, svec unpacking, 1/sqrt(2) scaling, reordering) into
constant sparse matrix entries computed once at compilation time.

The `DualInfo` metadata tells downstream code how to split and
reshape the output:

```python
# In CanonicalMaps (already defined in Proposal section):
R_dual: scipy.sparse.csr_array    # R_dual @ solver_y -> flat user duals
dual_vars: list[DualInfo]         # per-constraint metadata

# Usage:
dual_flat = R_dual @ solver_y     # one sparse matmul
for di in canon.dual_vars:
    dual_val = dual_flat[di.offset : di.offset + di.size]
    dual_val = dual_val.reshape(di.shape)
```

No per-cone-type dispatch, no `unpack_fn` branching, no
`_svec_to_symmetric()`. The `R_dual` matrix handles everything:

- **Non-PSD constraints**: Identity rows selecting from `solver_y`.
- **PSD constraints**: Svec-to-full expansion with 1/sqrt(2) on
  off-diagonal entries. Column ordering matches the solver's
  convention (lower-tri for SCS, upper-tri for Clarabel).
- **Multi-dual constraints** (SOC, ExpCone): Each dual variable gets
  its own `DualInfo` entry pointing to its slice of the output.

### What This Replaces

| Currently in CVXPYgen | Replaced by |
|---|---|
| `get_dual_variable_info()` (~60 lines) | `canon.R_dual` + `canon.dual_vars` |
| Hardcoded `inverse_data[-1..-4]` indexing | Not needed -- R_dual computed from `ParamConeProg.constraints` |
| `dual_id_maps` chain tracing | Not needed -- uses canonical ordering directly |
| `dual_var_split` / `dual_var_names` per solver | Not needed -- baked into R_dual column structure |
| `DualVariableInfo` dataclass | `DualInfo` |
| Solver-specific `extract_dual_value()` | Baked into R_dual entries (1/sqrt(2) scaling, tri ordering) |

| Currently in CVXPYlayers | Replaced by |
|---|---|
| `_build_constr_id_to_slice()` (~40 lines) | Baked into R_dual construction |
| `_build_dual_var_map()` (~15 lines) | Still needed (maps dual var IDs to constraints) |
| `_build_dual_recovery()` (~25 lines) | `canon.dual_vars` metadata |
| `_unpack_svec()` / `_svec_to_symmetric()` | Baked into R_dual entries |
| Per-framework svec unpacking logic | One sparse matmul (framework-agnostic after conversion) |

### Implementation Notes

The `R_dual` construction follows CVXPYlayers' approach of walking
`param_prog.constraints` in canonical cone order -- the same logic
as `_build_constr_id_to_slice()`, but instead of producing slices,
it produces rows of a sparse matrix:

```python
def _build_R_dual(param_prog, solver_name):
    rows, cols, vals = [], [], []
    out_offset = 0
    in_offset = 0
    dual_vars = []

    for c in param_prog.constraints:
        if isinstance(c, PSD):
            n = c.shape[0]
            svec_size = n * (n + 1) // 2
            full_size = n * n
            # Build svec-to-full expansion block
            _add_svec_unpack_block(
                rows, cols, vals,
                out_offset, in_offset, n,
                lower_tri=(solver_name == "SCS"),
            )
            dual_vars.append(DualInfo(
                constraint_id=c.id, shape=c.shape,
                offset=out_offset, size=full_size,
                cone_type="psd", is_psd=True,
            ))
            out_offset += full_size
            in_offset += svec_size
        else:
            # Identity block
            for i in range(c.size):
                rows.append(out_offset + i)
                cols.append(in_offset + i)
                vals.append(1.0)
            dual_vars.append(DualInfo(
                constraint_id=c.id, shape=c.shape,
                offset=out_offset, size=c.size,
                cone_type=type(c).__name__.lower(),
                is_psd=False,
            ))
            out_offset += c.size
            in_offset += c.size

    R_dual = scipy.sparse.csr_array(
        (vals, (rows, cols)),
        shape=(out_offset, in_offset),
    )
    return R_dual, dual_vars
```

## Detailed Design Questions

### 1. Should maps be in CSR or CSC format?

The maps are matrices that get multiplied by the parameter vector on
the right: `map @ param_vec = values`. CSR is the natural choice for
row-oriented operations (each row produces one output value). This is
also what CVXPYgen currently uses for `p_id_to_mapping`.

The output *structure* (for matrix canonical params like P and A) should
be in whatever format the solver prefers -- CSC for SCS/Clarabel/DIFFCP,
CSR for Moreau. This can be a parameter to `get_canonical_maps()`:

```python
def get_canonical_maps(
    self,
    param_prog: ParamConeProg,
    output_format: str = "csc",  # or "csr"
) -> CanonicalMaps:
```

### 2. Should the constant offset column be included?

Yes. Following CVXPYgen's convention, the parameter vector has a
trailing `1.0` entry, and the last column of each map encodes the
constant part of the affine mapping. This means:

```
values = map @ [user_param_1, ..., user_param_k, 1.0]
       = map[:, :-1] @ [user_param_1, ..., user_param_k] + map[:, -1]
```

This is the existing convention in both `ParamConeProg` and CVXPYgen.

### 3. Should this replace or augment `apply_parameters()`?

**Augment.** The existing `apply_parameters()` is well-tested and
performant for CVXPY's own solve path. `get_canonical_maps()` would be
a new method for downstream libraries that need the parametric maps.

### 4. How does `format_constraints()` interact with this?

`format_constraints()` modifies the `A` tensor in `ParamConeProg`
in-place (setting `formatted=True`). It handles structural formatting
(PSD scaling, SOC reordering, etc.) that is solver-specific.

`get_canonical_maps()` should require that `format_constraints()` has
already been called. Since `format_constraints()` is called inside
`_prepare_data_and_inv_data()`, and `get_problem_data()` calls
`solver.apply()` which calls `_prepare_data_and_inv_data()`, the
formatted state will always be available after `get_problem_data()`.

Alternatively, `get_canonical_maps()` could call `format_constraints()`
itself if it hasn't been called yet. It only needs to happen once and
is idempotent.

### 5. What about the `q` vs `c` naming?

The proposed API uses the canonical parameter IDs as dictionary keys.
Different solvers use different names for the linear objective:
- SCS, ECOS: `c`
- Clarabel, OSQP: `q`

The API should use whatever name the solver uses, matching CVXPYgen's
existing `canon_p_ids`. This eliminates the `getattr(param_prob, "q",
getattr(param_prob, "c", None))` pattern in CVXPYlayers.

### 6. What about bounds (lower_bounds, upper_bounds)?

For solvers that support variable bounds (OSQP, Clarabel), the bounds
may also be parametric (via `lb_tensor`, `ub_tensor`). If present,
`get_canonical_maps()` should include maps for these as well (e.g.,
`"lb"` and `"ub"` keys).

## Migration Path

### Phase 1: Add the API to CVXPY

Add `get_canonical_maps()` to the solver interface classes. The base
conic solver implementation handles SCS/Clarabel. OSQP and ECOS get
overrides. Unit tests verify that `canon.apply(param_vec)` produces
the same values as the existing `apply_parameters()` + solver
`apply()` path.

### Phase 2: Adopt in CVXPYlayers

Replace `parse_args.py`'s direct access to `reduced_P`, `q`,
`reduced_A` with a call to `get_canonical_maps()`. Simplify all
solver interfaces to consume the pre-decomposed maps. The
`problem_data_index` plumbing goes away.

### Phase 3: Adopt in CVXPYgen

Replace `cvxpygen/solvers.py`'s `SolverInterface` hierarchy with
calls to `get_canonical_maps()`. CVXPYgen's `get_affine_map()` logic
is deleted. The `process_canonical_parameters()` function simplifies
to iterating over `canon.maps.items()` and applying sparsity masks.

### Phase 4: Deprecate internal access

Once both libraries are migrated, the direct access patterns
(`param_prob.reduced_A.reduced_mat`, `param_prob.q`, etc.) can be
considered internal and documented as such.

## Alternatives Considered

### Alternative 1: Upstream CVXPYgen's SolverInterface into CVXPY

Instead of a new API on CVXPY's existing solver classes, we could move
CVXPYgen's `SolverInterface` class hierarchy into CVXPY wholesale.

**Rejected because**: CVXPYgen's solver interfaces are tightly coupled
to code generation concerns (C type names, augmentation with -inf,
constraint info for code layout). The decomposition logic is what we
want; the codegen scaffolding should stay in CVXPYgen.

### Alternative 2: Have CVXPYlayers use CVXPYgen's decomposition

CVXPYlayers could depend on CVXPYgen and call its `get_affine_map()`.

**Rejected because**: This creates an unnecessary dependency and
doesn't fix the underlying problem (the decomposition logic belongs
in CVXPY, not in a downstream library).

### Alternative 3: Only expose a parametric `apply_parameters_decomposed()`

Instead of returning affine maps, return the decomposed concrete values
from a single `apply_parameters()` call (i.e., return `P_values, q, d,
A_values, b` instead of `c, d, A, b`).

**Rejected because**: This doesn't help CVXPYgen (which needs the maps,
not the values) and doesn't fully help CVXPYlayers (which needs the
maps to do the matmul in its own tensor framework).

## Open Questions

1. **Should `get_canonical_maps()` be cached?** Probably yes, since the
   maps are static for a given solver + problem. The `ParamConeProg`
   is already cached; the decomposed maps could be cached alongside it.

2. **Should the API support the gradient/backward use case?** CVXPY's
   `backward()` uses `apply_param_jac()`, which is the transpose of
   the parameter mapping. The decomposed maps could provide this too:
   `map.T @ dvalues = dparam_vec`. This would simplify CVXPY's own
   backward path and could be useful for CVXPYlayers' derivative
   computation.

3. **Naming**: `get_canonical_maps()` vs `get_parametric_decomposition()`
   vs `decompose_param_prog()` vs something else?

4. **PSD dual storage convention**: The `R_dual` matrix bakes in the
   solver's PSD storage convention (SCS lower-triangular, Clarabel
   upper-triangular). The output is always a full symmetric matrix
   regardless. Should we also expose the intermediate svec format
   for libraries that want it? (Likely no -- the full matrix is the
   user-facing format and the overhead is minimal.)

5. **CVXPYgen dual recovery migration**: CVXPYgen currently traces
   `inverse_data[-1..-4]` for dual recovery. With `R_dual`, it would
   instead generate C code for a single sparse matmul. Does CVXPYgen
   need anything beyond `R_dual` and `DualInfo` for its codegen, or
   does it also use the `inverse_data` chain for primal variable
   recovery? (Primal recovery via `R_primal` should cover that case.)

6. **Multi-dual constraints**: SOC has 2 dual variables, ExpCone has
   3. The `R_dual` / `DualInfo` design handles this by having one
   `DualInfo` entry per dual variable (not per constraint). Does
   CVXPYgen expose multi-dual constraints, or only one dual per
   constraint?
