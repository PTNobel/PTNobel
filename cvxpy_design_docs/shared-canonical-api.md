# Design Doc: Shared Canonical Parameter API for CVXPY

## Status: Draft

## Authors
Claude Opus 4.6, with prompting from Parth Nobel

## Reviewers
Max Schaller

## Summary

CVXPY should expose a public API on its solver interfaces that provides
**compiled parametric affine maps** from user parameters to
solver-specific canonical parameters (P, q, A, b, etc.). This would
upstream the map-construction logic currently duplicated in CVXPYgen's
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

| Solver   | Form                                                       | Canonical params      |
|----------|------------------------------------------------------------|-----------------------|
| SCS      | min. $\tfrac{1}{2}x^\top Px + c^\top x$ s.t. $Ax + s = b,s \in K$ | $P, c, d, A, b$ |
| Clarabel | min. $\tfrac{1}{2}x^\top Px + q^\top x$ s.t. $Ax + s = b,s \in K$ | $P, q, d, A, b$ |
| ECOS     | min. $c^\top x$ s.t. $Ax = b,Gx \le_K h$               | $c, d, A, b, G, h$    |
| OSQP     | min. $\tfrac{1}{2}x^\top Px + q^\top x$ s.t. $l \le Ax \le u$ | $P, q, d, A, l, u$ |

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
(in `solvers.py`) reimplements the same logic, but on the
parametric affine maps. For each canonical parameter (P, q, A, b, etc.),
it extracts the relevant rows from `reduced_P.reduced_mat`,
`param_prob.q`, and `reduced_A.reduced_mat`, applies sign corrections,
and produces a per-parameter affine map:
`p_id_to_mapping[p_id] @ param_vec = canonical_param_values`. This is
the core logic that could be upstreamed.

**In CVXPYlayers**: Each solver interface independently reconstructs
the augmented `[A | b]` matrix, splits off `b`, negates `A`, and
converts to the solver's preferred format at `CvxpyLayer`
construction time.

### The Duplication

The sign correction and splitting logic exists in **three places**:

1. `cvxpy/reductions/solvers/conic_solvers/conic_solver.py` --
   `apply()` negates A for concrete values
2. `cvxpygen/solvers.py` -- `get_affine_map()` applies sign=-1 to the
   affine map rows for A
3. `cvxpylayers/interfaces/*.py` -- each solver reconstructs and negates
   A from CSC values at construction time

## Proposal

### Core Idea

Every operation in the CVXPY solve pipeline — from user parameters to
solver inputs, and from solver outputs back to user variables — is a
**linear operation** that can be represented as a sparse matrix
multiply. CVXPY should expose the complete set of these matrices as
a public API.

The entire data flow becomes six sparse matmuls plus constant offsets:

```
Forward (parameters → solver inputs):
    P_map  @ param_vec + P_offset  →  P values
    q_map  @ param_vec + q_offset  →  q/c vector
    A_map  @ param_vec + A_offset  →  A values     (sign-corrected)
    b_map  @ param_vec + b_offset  →  b vector

Recovery (solver outputs → user variables):
    R_primal  @ solver_x  →  [var1_flat; var2_flat; ...]
    R_dual    @ solver_y  →  [dual1_flat; dual2_flat; ...]
```

The recovery matrices encode all normalization: slicing from
`var_id_to_col`, symmetric svec-to-full expansion, PSD $1/\sqrt{2}$
off-diagonal scaling, Fortran-order permutation — everything baked
into constant sparse matrix entries computed once at compilation time.

There are **zero nonlinear operations** in either direction. (The GP
`exp()` transform is applied separately after recovery and is
explicitly outside the linear pipeline.)

### `CanonicalMaps` Dataclass

```python
@dataclass
class CanonicalMaps:
    """Complete set of sparse matrices for a compiled parametric problem.

    Forward maps (parameter → solver input):
        M @ param_vec + v = solver_input_values
        where param_vec is the flat parameter vector (no trailing 1.0).

    Recovery maps (solver output → user variables):
        R @ solver_vec = user_variable_values (flat, concatenated)
    """

    # --- Forward maps (parameter → solver input) ---

    # Per-canonical-parameter affine maps: (matrix, offset) pairs.
    # Usage: maps[key][0] @ param_vec + maps[key][1] = values
    maps: dict[str, tuple[scipy.sparse.csr_array, np.ndarray]]
    # e.g. {"P": (M, v), "q": (M, v), "d": (M, v), "A": (M, v), ...}

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
    """Metadata for one dual variable in the recovery output."""
    dual_var_id: int         # ID of the DualVariable object
    constraint_id: int
    shape: tuple[int, ...]
    offset: int              # start index in R_dual output
    size: int                # number of elements in output
    cone_type: str           # "zero", "nonneg", "soc", "psd", etc.
    is_psd: bool
```

### `Problem.get_canonical_maps()` — User-Facing API

The primary entry point for downstream libraries. Lives on `Problem`
because it needs the compilation cache and the reduction chain's
variable mappings:

```python
class Problem:
    def get_canonical_maps(
        self,
        solver: str,
        variables: list[Variable | DualVariable] | None = None,
        *,
        gp: bool = False,
        # ... same compilation kwargs as get_problem_data()
    ) -> CanonicalMaps:
        """Return parametric affine maps for this problem.

        Compiles the problem (or uses the cached compilation) and
        returns a CanonicalMaps with forward maps from parameters to
        solver inputs and recovery maps from solver outputs to user
        variables.

        Parameters
        ----------
        solver : str
            Target solver (e.g., cp.SCS, cp.CLARABEL).
        variables : list, optional
            Subset of Variable / DualVariable objects to include in
            recovery. When None, all variables and constraint duals
            are recovered. When specified, R_primal / R_dual contain
            only the rows for the requested variables, and
            primal_vars / dual_vars are filtered to match.
        """
        # 1. Compile / use cache to get param_prog, inverse_data
        # 2. Call solver.get_canonical_maps(param_prog) for full maps
        # 3. If variables is not None, row-slice R_primal / R_dual
        #    using VarInfo/DualInfo offsets
        ...
```

This method is the right level for variable subsetting because:

- It has access to the compilation cache (`_cache.param_prog`,
  `_cache.solving_chain`), avoiding redundant compilation.
- It can map user-level `Variable` objects to canonical variable IDs
  via the `inverse_data` chain (the reduction chain may rename or
  split variables).
- The solver's `get_canonical_maps()` stays simple — it always builds
  full recovery matrices over canonical variables, with no knowledge
  of which user variables matter.

One large sparse matmul is significantly faster than many small ones,
especially on GPU where kernel launch overhead dominates. The
`variables` parameter lets callers (e.g., MPC problems that only need
$u_0$ out of thousands of state variables) avoid recovering unneeded
variables while preserving the single-matmul pattern.

### `ConicSolver.get_canonical_maps()` — Internal API

The solver-level method that builds the maps from `ParamConeProg`.
Each solver subclass inherits the base implementation or overrides
for solver-specific behavior (equality/inequality splitting, PSD
storage format, output matrix format):

```python
# In conic_solver.py
class ConicSolver(Solver):
    def get_canonical_maps(self, param_prog: ParamConeProg) -> CanonicalMaps:
        """Return the complete set of sparse matrices.

        Forward maps: compiles param_prog's monolithic tensors into
        per-solver-parameter affine maps with solver-specific sign
        conventions applied. Output matrix structures (P, A) use the
        solver's preferred sparse format (CSC or CSR).

        Recovery maps: builds sparse matrices that map the solver's
        flat primal/dual vectors to per-variable/per-constraint values,
        with svec unpacking and scaling baked in.
        """
        ...
```

Map dictionary keys use whatever name the solver uses for each
canonical parameter (SCS/ECOS use `c`; Clarabel/OSQP use `q`),
matching CVXPYgen's existing `canon_p_ids`.

`get_canonical_maps()` requires that `format_constraints()` has
already been called on the `ParamConeProg`. Since this happens inside
`_prepare_data_and_inv_data()`, the formatted state is always
available after `get_problem_data()`. Alternatively,
`get_canonical_maps()` could call `format_constraints()` itself if
needed — it is idempotent.

For solvers that support variable bounds (e.g., HiGHS), parametric
bounds (`lb_tensor`, `ub_tensor`) are included as additional map
entries (e.g., `"lb"` and `"ub"` keys).

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

`R_dual` walks `param_prog.constraints` in canonical cone order,
building a block for each constraint:

- **Non-PSD**: Identity block selecting
  `y[dual_offset:dual_offset+c.size]`. All entries = 1.

- **PSD** (n x n): The svec unpacking matrix.
  Input: `n*(n+1)/2` svec entries from the solver's dual vector.
  Output: `n*n` entries (full symmetric matrix in Fortran order).
  Entries: 1 on diagonal positions, $1/\sqrt{2}$ on off-diagonal
  positions (two entries per off-diagonal svec element, writing to
  both `(i,j)` and `(j,i)`).

  For SCS (lower-triangular svec): columns ordered by lower-tri.
  For Clarabel (upper-triangular svec): columns ordered by upper-tri.
  The *output* is identical; only the column ordering differs.

- **Multi-dual constraints** (SOC, ExpCone): Each dual variable gets
  its own `DualInfo` entry pointing to its slice of the output.

This replaces CVXPYgen's fragile `get_dual_variable_info()` (which
traces `cons_id_map` through hardcoded `inverse_data[-1..-4]` indices)
and CVXPYlayers' per-cone-type unpacking logic.

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
5. **l map**: Equality vector rows, negated. Inequality rows have
   offset `v = -inf` (no lower bound).
6. **u map**: All vector rows, with per-row sign vector.

### Forward Maps: ECOS Override

Splits constraints into equality (A, b) and inequality (G, h) groups.

### Constant Offset Representation

Each forward map is an `(M, v)` pair: `M @ param_vec + v`.

The `ParamConeProg` internally uses a trailing-`1.0` convention where
a single matrix's last column encodes constants.
`get_canonical_maps()` splits this into `(M, v)` at construction time.
This split is necessary because consolidating all solver parameters
(including bounds) into affine maps would introduce `inf` values into
the matrix. Today, `inf` values never appear in any matrix — OSQP's
`-inf` lower bounds are hardcoded at solve time in `osqp_qpif.py`,
not stored in `ParamConeProg`. Naively putting them in the last column
of a combined map matrix would cause two problems:

1. **Differentiability.** The backward pass through
   `[M | v] @ [p; 1]` computes `[M | v].T @ d_output`. Any
   `d_output` entry that is zero for an `inf` row produces
   `inf * 0.0 = NaN`, poisoning gradients.

2. **Embedded efficiency.** CVXPYgen would recompute always-`inf`
   entries on every re-solve and would need to represent `inf` in
   generated C code.

With the `(M, v)` split, `M` contains only finite entries and `v`
(which may contain `inf`) is a constant with zero gradient — it never
participates in differentiation. The split costs zero extra flops.
CVXPYgen can trivially identify `inf` entries in `v` and construct
reduced maps if it wants to skip those rows in generated code.

**Alternative considered: exclude purely constant rows from `M`.**
This is a special case of the `(M, v)` split that additionally changes
the shape of `M`. Rejected because it complicates the API (the caller
must recombine rows) without meaningful efficiency gains.

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
canon = problem.get_canonical_maps(solver)
# Forward maps -- generate C code for each affine map:
for p_id, (M, v) in canon.maps.items():
    codegen_sparse_matmul(p_id, M, v)
# Primal recovery -- generate C code for R_primal @ x:
codegen_sparse_matmul("primal_recovery", canon.R_primal)
# Dual recovery -- generate C code for R_dual @ y:
codegen_sparse_matmul("dual_recovery", canon.R_dual)
```

The entire `SolverInterface` class hierarchy and the fragile
`inverse_data[-4]` indexing in CVXPYgen would be deleted. CVXPYgen
keeps its sparsity optimization, adjacency tracking, and C codegen.

### CVXPYlayers

CVXPYlayers' recovery logic (~100 lines of per-cone-type dispatch
and per-framework svec unpacking) becomes one sparse matmul per
direction (`R_primal @ solver_x`, `R_dual @ solver_y`). The R
matrices are scipy sparse at setup time, converted once to the
framework's sparse format. Forward maps similarly replace the
per-solver `torch_to_data()` reconstruction with sparse matmuls.


### CVXPY's Own Solvers

The existing `apply_parameters()` + solver `apply()` path is
unchanged. The new API is additive.

## What Stays in CVXPYgen

Even with this API, CVXPYgen would retain:

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

## Migration Path

### Phase 1: Add the API to CVXPY

Add `get_canonical_maps()` to the solver interface classes and
`Problem`. The base conic solver implementation handles SCS/Clarabel.
OSQP and ECOS get overrides. Unit tests verify that the maps produce
the same values as the existing `apply_parameters()` + solver
`apply()` path.

### Phase 2: Adopt in CVXPYlayers

Replace `parse_args.py`'s direct access to `reduced_P`, `q`,
`reduced_A` with a call to `problem.get_canonical_maps()`. Simplify
all solver interfaces to consume the compiled maps. The
`problem_data_index` plumbing goes away.

### Phase 3: Adopt in CVXPYgen

Replace `cvxpygen/solvers.py`'s `SolverInterface` hierarchy with
calls to `problem.get_canonical_maps()`. CVXPYgen's `get_affine_map()`
logic is deleted. The `process_canonical_parameters()` function
simplifies to iterating over `canon.maps.items()`.

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
constraint info for code layout). The map-construction logic is what
we want; the codegen scaffolding should stay in CVXPYgen.

### Alternative 2: Have CVXPYlayers use CVXPYgen's decomposition

CVXPYlayers could depend on CVXPYgen and call its `get_affine_map()`.

**Rejected because**: This creates an unnecessary dependency and
doesn't fix the underlying problem (the map-construction logic belongs
in CVXPY, not in a downstream library).

### Alternative 3: Only expose a split `apply_parameters()`

Instead of returning affine maps, return the concrete values split
per solver parameter from a single `apply_parameters()` call (i.e.,
return `P_values, q, d, A_values, b` instead of `c, d, A, b`).

**Rejected because**: This doesn't help CVXPYgen (which needs the maps,
not the values) and doesn't fully help CVXPYlayers (which needs the
maps to do the matmul in its own tensor framework).

## Open Questions

1. **Should `get_canonical_maps()` be cached?** Probably yes, since the
   maps are static for a given solver + problem. The `ParamConeProg`
   is already cached; the compiled maps could be cached alongside it.

2. **Should the API support the gradient/backward use case?** CVXPY's
   `backward()` uses `apply_param_jac()`, which is the transpose of
   the parameter mapping. The compiled maps could provide this too:
   `map.T @ dvalues = dparam_vec`. This would simplify CVXPY's own
   backward path and could be useful for CVXPYlayers' derivative
   computation.

3. **Naming**: `get_canonical_maps()` vs `get_compiled_maps()`
   vs `compile_param_maps()` vs something else?

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
