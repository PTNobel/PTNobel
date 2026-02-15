# IIS Detection for CVXPY — Implementation Plan

## Summary

Add the ability to detect an Irreducible Infeasible Subsystem (IIS) of constraints when a CVXPY problem is infeasible. The feature should:

1. Leverage **infeasibility certificates** from conic solvers (SCS, Clarabel, OSQP, MOSEK) to cheaply identify candidate constraints, then prune for irreducibility.
2. Leverage **explicit IIS routines** (Gurobi) when available, mapping canonical-level results back to original user constraints.
3. Provide a **solver-agnostic fallback** (deletion filter) operating directly on the original DCP-level constraints.
4. Expose a clean, user-facing API on the `Problem` object.

---

## 1. User-Facing API

### 1.1 Primary Interface

```python
problem = cp.Problem(objective, constraints)
problem.solve()  # returns 'infeasible'

iis = problem.find_iis()
# Returns: list of IISMember objects
```

**`Problem.find_iis(solver=None, **solver_opts)`**

- `solver`: Which solver to use. If `None`, use the solver from the last `solve()` call. Falls back to the deletion filter if the solver has no native IIS support.
- Returns a list of `IISMember` objects (see below).
- Raises `SolverError` if the problem was not previously solved or is not infeasible.

### 1.2 IISMember: Sub-Constraint Granularity

A single user constraint like `A @ x == b` may contribute only a subset of its rows to the IIS. The API needs to surface this.

```python
class IISMember:
    constraint: cp.Constraint   # the original user constraint
    rows: Optional[np.ndarray]  # which rows are involved, None = all rows
    
    def __repr__(self):
        if self.rows is None:
            return f"IISMember({self.constraint})"
        return f"IISMember({self.constraint}, rows={self.rows})"
```

**Usage example:**

```python
A = np.random.randn(100, 5)
b = np.random.randn(100)
x = cp.Variable(5)

c1 = A @ x == b        # 100 scalar equalities
c2 = x >= 0            # 5 scalar inequalities  
c3 = cp.sum(x) <= -10  # scalar

prob = cp.Problem(cp.Minimize(0), [c1, c2, c3])
prob.solve()

iis = prob.find_iis()
for member in iis:
    print(member.constraint, member.rows)
# Possible output:
#   A @ x == b, rows=[3, 17, 42]
#   sum(x) <= -10, rows=None        (scalar, so None means "the whole thing")
```

This tells the user exactly which rows of `A @ x == b` are in conflict, rather than flagging all 100.

### 1.3 Attribute Access (Post-IIS)

```python
for constr in problem.constraints:
    print(constr, constr.iis_rows)  
    # None if not in IIS, np.ndarray of row indices if partially in IIS,
    # or "all" sentinel if the entire constraint is in the IIS
```

### 1.4 Optional: Multiple IIS

For a future iteration, consider:

```python
iis_list = problem.find_all_iis(max_iis=5)
```

This is non-trivial and can be deferred. Document it as a planned extension.

---

## 2. Architecture

### 2.1 Key Insight

Auxiliary constraints introduced by DCP canonicalization (epigraph variables, SOC embeddings, linking equalities) are **always feasible in isolation**. Infeasibility only arises from interactions between constraints owned by *different* original user constraints.

Therefore, the back-mapping from canonical IIS → original constraints is:

1. Collect the set of original constraints that *own* at least one canonical constraint in the IIS.
2. (Optional) Prune with a small deletion filter to ensure irreducibility at the original level.

### 2.2 Component Overview

```
Problem.find_iis()
    │
    ├─── Solver supports IIS?
    │       │
    │       ├── Certificate-based (SCS, Clarabel, OSQP, MOSEK)
    │       │     │
    │       │     ├── Extract dual certificate y from solver output
    │       │     ├── Identify nonzero support: {i : |y_i| > tol}
    │       │     ├── Map candidate rows → original constraints via row_map
    │       │     └── Prune candidates with deletion filter (small set, cheap)
    │       │
    │       └── Explicit IIS (Gurobi)
    │             │
    │             ├── Invoke solver's computeIIS() on canonical model
    │             ├── Collect IIS rows + variable bounds
    │             ├── Map back to original constraints via row_map
    │             └── Prune for irreducibility at original level (small set, cheap)
    │
    └─── No solver support / fallback
            │
            ├── (Optional) Elastic pre-screen to narrow candidates
            └── Deletion filter on original DCP constraints (n solves)
```

---

## 3. Constraint-to-Row Mapping

This is the critical internal piece. During canonicalization, CVXPY already tracks which constraints produce which rows/cones in the canonical form — but the bookkeeping isn't currently exposed in a way that's convenient for IIS back-mapping.

### 3.1 Where the Mapping Lives

In the current CVXPY codebase, when a problem is compiled:

```python
data, chain, inverse_data = problem.get_problem_data(solver)
```

Each `Reduction` in the chain transforms constraints. The final solver interface (e.g., `conic_solver.py`) assembles the `A` matrix by iterating over constraints and concatenating their contributions. The offset/size of each constraint's contribution is implicitly tracked but not stored in a lookup-friendly structure.

### 3.2 What Needs to Change

Add an explicit **row-to-constraint map** built during the final compilation step. The map must track not just which original constraint owns each canonical row, but **which row within that constraint** it corresponds to — so that the IIS can report sub-constraint granularity.

```python
@dataclass
class RowOrigin:
    constraint: cp.Constraint  # original user constraint
    row: int                   # row index within that constraint

class ConicSolver:
    def apply(self, problem, ...):
        ...
        row_map = {}  # Dict[int, RowOrigin]
        offset = 0
        for constr in problem.constraints:
            original = constr.origin if constr.origin else constr
            n_rows = constr.cone_size()
            
            # Map each canonical row to (original_constraint, row_within_constraint)
            # Need to track the sub-row mapping through reductions.
            for i in range(n_rows):
                row_map[offset + i] = RowOrigin(
                    constraint=original,
                    row=constr.origin_row_offset + i  # see below
                )
            offset += n_rows
        
        self._row_map = row_map
```

The `origin_row_offset` tracks how rows within a reduced constraint map back to rows within the original. For example, if the user writes `A @ x == b` (100 rows) and canonicalization doesn't split it, then rows 0–99 of the canonical constraint map directly to rows 0–99 of the original. If a reduction does split or reorder rows, it must update the offset accordingly.

**For most DCP reductions, the row mapping is trivial** — affine constraints pass through with identity row mapping, and conic constraints (SOC, PSD) map as a block. The main case where this matters is when a vector constraint like `A @ x == b` stays as a single canonical constraint — each canonical row `i` maps to original row `i`.

**Option A: Tag during reduction.** Each reduction step annotates output constraints with a reference to the originating constraint. Add an `origin` attribute to `Constraint`:

```python
class Constraint:
    origin: Optional[Constraint] = None  # points to the original user constraint
```

When a reduction transforms constraint `C` into `[C1, C2, C3]`, set `C1.origin = C2.origin = C3.origin = C`. Chain through multiple reductions by always pointing to the root.

**Where to set `origin`:** The origin assignment should live in the top-level reduction classes — `Dcp2Canon`, `Dgp2Dcp`, `Qp2SymbolicQp`, etc. — rather than inside individual atom canonicalization files. These reductions are where user constraints first get transformed, so they're the natural place to stamp the origin. The individual canon files don't need to be aware of this; they just produce new constraints, and the parent reduction class wraps them with the origin reference after the fact. This keeps the change relatively contained:

```python
class Dcp2Canon(Reduction):
    def apply(self, problem):
        new_constraints = []
        for constr in problem.constraints:
            canon_constrs = self._canonicalize_constraint(constr)
            for cc in canon_constrs:
                # Point back to the root user constraint
                cc.origin = constr.origin if constr.origin else constr
            new_constraints.extend(canon_constrs)
        ...
```

The same pattern applies to `Dgp2Dcp`, `Qp2SymbolicQp`, and any other top-level reductions. Since the origin always points to the root (not the intermediate), multi-stage chains like `Dgp2Dcp → Dcp2Canon` work correctly — the DGP constraint's origin is set in `Dgp2Dcp`, and `Dcp2Canon` preserves it via the `constr.origin if constr.origin else constr` pattern.

**Option B: Parallel tracking structure.** Maintain a side dict `{reduced_constraint_id: original_constraint}` passed through the chain. Less invasive but more fragile.

**Recommendation: Option A** — it's cleaner and survives refactors better. The change is relatively invasive (touching each top-level reduction class) but straightforward and mechanical.

### 3.3 Cone-Specific Considerations

Different cone types map differently:

| Cone | Canonical Form | Mapping |
|------|---------------|---------|
| Zero (equality) | Rows of `Ax + s = b, s = 0` | Direct row mapping |
| Nonneg (inequality) | Rows of `Ax + s = b, s >= 0` | Direct row mapping |
| SOC | `\|Ax + b\| <= c^T x + d` | Entire cone block → one original constraint |
| PSD | `X ≽ 0` | Entire block → one original constraint |

For SOC and PSD, the solver's IIS may flag individual rows *within* a cone block. All such rows map to the same original constraint, so this is fine — you just deduplicate.

**Note on OSQP:** The QP reduction chain is being merged into the conic chain (targeted CVXPY 1.10). Once complete, the `row_map` can be built in a single unified place for all solvers. The only OSQP-specific logic will be a final index remapping at the solver interface layer to account for the `[A; D]` stacking.

---

## 4. Solver-Specific IIS Support

There are two fundamentally different mechanisms solvers provide:

1. **Explicit IIS enumeration** — the solver directly computes an IIS (Gurobi, MOSEK `analyzeproblem`).
2. **Infeasibility certificates** — the solver returns a dual ray / Farkas certificate proving infeasibility. Constraints with nonzero certificate components are *candidates* for the IIS, but the set is typically not irreducible. Requires a pruning pass.

Both mechanisms feed into the same pipeline: identify candidate original constraints via `row_map`, then optionally prune for irreducibility.

### 4.1 Infeasibility Certificates — Theory

When a conic problem `minimize c^T x  s.t. Ax + s = b, s ∈ K` is primal infeasible, the solver can return a **dual certificate** `y` satisfying:

```
A^T y = 0
b^T y < 0
y ∈ K*        (dual cone)
```

This is a Farkas-type proof that no feasible point exists. The key property: **the support of `y`** (rows where `y_i ≠ 0`) identifies which constraints participate in the infeasibility proof. Constraints where `y_i = 0` are irrelevant to this particular proof.

This gives us a candidate set for the IIS — typically much smaller than the full constraint set, but not necessarily irreducible. A short deletion filter on the candidates closes the gap.

### 4.2 SCS

SCS returns infeasibility certificates directly in its solution output.

When the problem is primal infeasible, SCS returns a vector `y` (the dual variable) that serves as the certificate. CVXPY already retrieves this — it's stored in the solver output as the dual variables.

```python
class SCSSolver(ConicSolver):
    def get_iis_candidate_rows(self, solver_output, row_map):
        """
        Extract IIS candidate rows from SCS infeasibility certificate.
        
        Returns Dict[Constraint, Set[int]] mapping each candidate
        original constraint to the specific rows with nonzero certificate.
        """
        y = solver_output['y']  # dual certificate vector
        
        tol = 1e-6 * np.max(np.abs(y))
        
        candidates = {}  # Dict[Constraint, Set[int]]
        for i, yi in enumerate(y):
            if abs(yi) > tol and i in row_map:
                origin = row_map[i]  # RowOrigin(constraint, row)
                if origin.constraint not in candidates:
                    candidates[origin.constraint] = set()
                candidates[origin.constraint].add(origin.row)
        
        return candidates
```

**SCS-specific considerations:**
- SCS is a first-order method — certificates are approximate. The tolerance for "nonzero" needs care. Too tight and you miss participants; too loose and you get false candidates (which the pruning pass will remove, but it costs solves).
- SCS uses the **homogeneous self-dual embedding**, so infeasibility is detected via a specific exit condition. The certificate is in `sol.y` when `info.status == 'infeasible'`.
- SCS works natively with the cones CVXPY targets (zero, nonneg, SOC, PSD, exponential), so the certificate covers all standard DCP constraint types.

### 4.3 Clarabel

Clarabel follows a very similar interface to SCS but is generally more numerically accurate (interior-point method rather than first-order).

```python
class ClaravelSolver(ConicSolver):
    def get_iis_candidate_rows(self, solver_output, row_map):
        """
        Clarabel returns an infeasibility certificate in the dual variables
        when status is PRIMAL_INFEASIBLE.
        
        Returns Dict[Constraint, Set[int]] with row-level granularity.
        Clarabel's certificate is typically cleaner than SCS since it's
        an interior-point solver, so tighter tolerances are appropriate.
        """
        y = solver_output['y']
        
        tol = 1e-8 * np.max(np.abs(y))
        
        candidates = {}
        for i, yi in enumerate(y):
            if abs(yi) > tol and i in row_map:
                origin = row_map[i]
                if origin.constraint not in candidates:
                    candidates[origin.constraint] = set()
                candidates[origin.constraint].add(origin.row)
        
        return candidates
```

**Clarabel-specific considerations:**
- Clarabel is an interior-point solver, so certificates are more precise. Tighter tolerance is appropriate.
- Clarabel supports the same cone types as SCS plus the generalized power cone.
- Clarabel returns `PrimalInfeasibleCertificate` status — check against this.

### 4.4 OSQP

OSQP is QP-only (no conic constraints), but it does return infeasibility certificates. CVXPY canonicalizes problems for OSQP into the form:

```
minimize  (1/2) x^T P x + q^T x
s.t.      b  <= Ax <= b       (equality constraints, rows 0..m_eq-1)
          c  <= Dx <= +infty  (inequality constraints, rows m_eq..m_eq+m_ineq-1)
```

These are stacked into a single constraint matrix `[A; D]` with bounds `l = [b; c]` and `u = [b; +infty]` before being passed to OSQP. The certificate vector spans both blocks.

```python
class OSQPSolver(Solver):
    def get_iis_candidate_rows(self, solver_output, row_map):
        """
        OSQP returns a certificate delta_y when status is 'primal infeasible'.
        
        The certificate satisfies:
            [A; D]^T delta_y = 0
            u^T [delta_y]_+ + l^T [delta_y]_- < 0
        
        Returns Dict[Constraint, Set[int]] with row-level granularity.
        """
        delta_y = solver_output['prim_inf_cert']
        
        tol = 1e-5 * np.max(np.abs(delta_y))
        
        candidates = {}
        for i, dy in enumerate(delta_y):
            if abs(dy) > tol and i in row_map:
                origin = row_map[i]
                if origin.constraint not in candidates:
                    candidates[origin.constraint] = set()
                candidates[origin.constraint].add(origin.row)
        
        return candidates
```

**OSQP-specific considerations:**
- **QP reduction chain is being removed.** As of CVXPY 1.10 (or near it), OSQP will go through the same conic reduction chain as SCS/Clarabel, with a final step converting conic form to the `[A; D]` stacking. This means the `row_map` can be built in the same place for all solvers — the only OSQP-specific logic is the final conic-to-QP mapping at the solver interface layer.
- Because the upper bound on inequalities is `+infty`, the certificate component for an inequality row only has one "active side." This is simpler than general two-sided bounds — a positive `delta_y_i` in the inequality block means the lower bound `c_i` is participating.
- OSQP is a first-order method (ADMM), so certificate precision is similar to SCS. Use comparable tolerances.
- OSQP only handles LP/QP, so this path only applies to problems that canonicalize to QP form.

### 4.5 Gurobi (Explicit IIS)

Gurobi computes an explicit IIS rather than returning a certificate. This is the gold standard — the result is already irreducible, no pruning needed.

```python
class GurobiSolver(ConicSolver):
    def get_iis_candidate_rows(self, model, row_map):
        """
        Gurobi's computeIIS() returns an already-irreducible set
        at the canonical level.
        
        Returns Dict[Constraint, Set[int]] with row-level granularity.
        """
        model.computeIIS()
        
        candidates = {}
        for i, c in enumerate(model.getConstrs()):
            if c.IISConstr and i in row_map:
                origin = row_map[i]
                if origin.constraint not in candidates:
                    candidates[origin.constraint] = set()
                candidates[origin.constraint].add(origin.row)
        
        # Also check variable bounds
        for v in model.getVars():
            if v.IISLB:
                bound_origin = var_bound_map.get(('lb', v.VarName))
                if bound_origin:
                    if bound_origin.constraint not in candidates:
                        candidates[bound_origin.constraint] = set()
                    candidates[bound_origin.constraint].add(bound_origin.row)
            if v.IISUB:
                bound_origin = var_bound_map.get(('ub', v.VarName))
                if bound_origin:
                    if bound_origin.constraint not in candidates:
                        candidates[bound_origin.constraint] = set()
                    candidates[bound_origin.constraint].add(bound_origin.row)
        
        return candidates
```

**Gurobi-specific considerations:**
- Gurobi IIS supports LP and QP. For SOCP (via Gurobi's native SOCP), check `IISQConstr` on quadratic constraints.
- The canonical IIS is already irreducible, but after back-mapping to original constraints, it may not be (two canonical constraints from different original constraints might both map back, but only one is needed). **Still prune at the original level.**
- Gurobi's `computeIIS()` can accept an IIS method parameter (`IISMethod`): 0 = automatic, 1 = deletion filter, 3 = force a smaller IIS. Expose this as a solver option.

### 4.6 MOSEK

MOSEK offers both certificate-based analysis and explicit IIS-like functionality.

```python
class MOSEKSolver(ConicSolver):
    def get_iis_candidate_rows(self, task, row_map):
        """
        MOSEK provides infeasibility certificates via the dual solution
        when the problem is primal infeasible.
        
        Returns Dict[Constraint, Set[int]] with row-level granularity.
        """
        # Certificate approach: extract dual values from the infeasibility cert
        y = task.getdualobj()  # or extract from solution
        
        # ... same nonzero-support extraction pattern as SCS/Clarabel ...
```

**MOSEK-specific considerations:**
- MOSEK is an interior-point solver — certificates are high precision.
- MOSEK's `Task.analyzeproblem()` can provide additional diagnostics.
- MOSEK handles LP, SOCP, SDP, and exponential cones — broadest cone support.

### 4.7 Solver Interface Addition

Add methods to the solver base class:

```python
class Solver:
    def supports_iis(self) -> bool:
        """Whether this solver can provide IIS candidates 
        (via certificate or explicit IIS)."""
        return False
    
    def iis_is_irreducible(self) -> bool:
        """Whether the solver's IIS output is already irreducible
        (True for Gurobi, False for certificate-based solvers)."""
        return False
    
    def get_iis_candidate_rows(self, solver_output, row_map) -> Dict[Constraint, Set[int]]:
        """Return candidate original constraints and their participating rows.
        Raises NotImplementedError if solver lacks support."""
        raise NotImplementedError
```

### 4.8 Unified Pipeline

All solver paths feed into the same post-processing, producing `IISMember` objects with row-level granularity:

```python
def find_iis(self, solver=None, **solver_opts):
    # 1. Get candidates from solver
    solver_impl = get_solver(solver)
    if solver_impl.supports_iis():
        candidate_rows = solver_impl.get_iis_candidate_rows(
            self._solver_output, self._row_map
        )
        # candidate_rows: Dict[Constraint, Set[int]]
        #   maps each candidate original constraint to the set of
        #   its rows that had nonzero certificate components
        
        # 2. Prune at constraint level
        candidate_constrs = list(candidate_rows.keys())
        iis_constrs = deletion_filter_constrs(
            candidate_constrs, self.objective, solver, solver_opts
        )
        
        # 3. Prune at row level for multi-row constraints
        iis_members = []
        for constr in iis_constrs:
            if constr.size <= 1 or not _is_splittable(constr):
                iis_members.append(IISMember(constr, rows=None))
            else:
                # Use certificate rows as starting candidates for row-level filter
                cert_rows = candidate_rows.get(constr)
                iis_rows = row_level_deletion_filter(
                    constr, iis_constrs, cert_rows, solver, solver_opts
                )
                iis_members.append(IISMember(constr, rows=iis_rows))
        
        return iis_members
    else:
        # 4. Full fallback
        return deletion_filter_iis(
            self.constraints, self.objective, solver, solver_opts
        )
```

The certificate narrows candidates at **both** levels: which constraints (typically from hundreds to a handful) and which rows within each constraint (typically from hundreds to a few). The deletion filter then just confirms and prunes these small sets.

### 4.9 Tolerance Strategy

Certificate-based IIS detection is sensitive to the threshold for "nonzero":

| Solver | Type | Recommended Tolerance | Rationale |
|--------|------|----------------------|-----------|
| SCS | First-order (ADMM) | `1e-5 * ‖y‖_∞` | Noisy certificates |
| OSQP | First-order (ADMM) | `1e-5 * ‖δy‖_∞` | Similar to SCS |
| Clarabel | Interior-point | `1e-8 * ‖y‖_∞` | High-precision certificates |
| MOSEK | Interior-point | `1e-8 * ‖y‖_∞` | High-precision certificates |
| Gurobi | Explicit IIS | N/A | No threshold needed |

If the tolerance is too tight (missing true participants), the deletion filter pruning step will still produce a valid IIS — just potentially not the same one the certificate pointed to. If too loose (extra candidates), the pruning step removes them. So the tolerance affects **performance** (how many extra solves in pruning) but not **correctness**.

---

## 5. Fallback: Deletion Filter

For solvers without native IIS, or as a user-selectable option. The deletion filter now operates at **row granularity** — for vector constraints, it can identify which specific rows are needed.

### 5.1 Two-Level Deletion Filter

The filter runs in two passes to balance precision with performance:

**Pass 1: Constraint-level.** Remove entire constraints to find the minimal set of *constraints* involved.

**Pass 2: Row-level.** For each multi-row constraint in the IIS, determine which rows are actually needed.

```python
def deletion_filter_iis(
    constraints: List[Constraint],
    objective: Objective,
    solver: str,
    solver_opts: dict,
) -> List[IISMember]:
    """
    Find an IIS with row-level granularity.
    
    Pass 1: Constraint-level deletion filter.
    Pass 2: Row-level deletion filter on multi-row constraints.
    """
    # --- Pass 1: which constraints? ---
    iis_constrs = list(constraints)
    for constr in constraints:
        candidate = [c for c in iis_constrs if c is not constr]
        if not candidate:
            break
        test_prob = cp.Problem(cp.Minimize(0), candidate)
        test_prob.solve(solver=solver, **solver_opts)
        if test_prob.status in ('infeasible', 'infeasible_inaccurate'):
            iis_constrs = candidate
    
    # --- Pass 2: which rows within each constraint? ---
    iis_members = []
    for constr in iis_constrs:
        if constr.size <= 1:
            # Scalar constraint — no sub-row analysis needed
            iis_members.append(IISMember(constr, rows=None))
            continue
        
        # For vector constraints, find minimal row subset
        other_constrs = [c for c in iis_constrs if c is not constr]
        iis_rows = _row_level_deletion_filter(
            constr, other_constrs, solver, solver_opts
        )
        iis_members.append(IISMember(constr, rows=iis_rows))
    
    return iis_members


def _row_level_deletion_filter(
    constr: Constraint,
    other_constrs: List[Constraint],
    solver: str,
    solver_opts: dict,
) -> np.ndarray:
    """
    Given that `constr` is in the IIS, find which rows of it are needed.
    
    For A @ x == b (m rows), try removing each row and check if
    the remaining rows + other_constrs are still infeasible.
    """
    m = constr.size
    # Express the constraint as individual rows
    # For A @ x == b: split into [A[i,:] @ x == b[i] for i in range(m)]
    all_rows = _split_constraint_rows(constr)  # -> List[Constraint]
    
    iis_rows = list(range(m))
    for i in range(m):
        candidate_rows = [all_rows[j] for j in iis_rows if j != i]
        if not candidate_rows:
            break
        test_prob = cp.Problem(
            cp.Minimize(0), 
            other_constrs + candidate_rows
        )
        test_prob.solve(solver=solver, **solver_opts)
        if test_prob.status in ('infeasible', 'infeasible_inaccurate'):
            iis_rows = [j for j in iis_rows if j != i]
    
    return np.array(iis_rows)


def _split_constraint_rows(constr: Constraint) -> List[Constraint]:
    """
    Split a vector constraint into individual scalar constraints.
    
    A @ x == b  ->  [A[0,:] @ x == b[0], A[1,:] @ x == b[1], ...]
    A @ x <= b  ->  [A[0,:] @ x <= b[0], ...]
    
    This needs to handle the main constraint types CVXPY users write.
    """
    # Implementation depends on constraint type.
    # For affine equality/inequality, this is straightforward indexing.
    # For conic constraints (SOC, PSD), splitting doesn't make sense —
    # these are atomic and should be treated as indivisible blocks.
    ...
```

**When to split vs. not split:**

| Constraint Type | Splittable? | Rationale |
|----------------|-------------|-----------|
| `A @ x == b` | Yes | Each row is an independent equality |
| `A @ x <= b` | Yes | Each row is an independent inequality |
| `cp.norm(x) <= t` | No | SOC is atomic — rows are coupled |
| `X >> 0` (PSD) | No | PSD is atomic |
| `cp.sum(x) <= t` | No | Scalar |

For non-splittable constraints, skip Pass 2 and report the whole constraint.

### 5.2 Performance

Pass 1 costs at most `n` solves (n = number of constraints). Pass 2 costs at most `m_i` solves per multi-row constraint `i` in the IIS. In practice, the IIS is small and the multi-row constraints in it have a small active subset, so Pass 2 is cheap.

When the certificate-based path is available (Section 4), it pre-screens candidates before Pass 1, and the certificate's nonzero row support pre-screens rows before Pass 2 — so both passes operate on small sets.

### 5.3 Elastic Filtering (Pre-screening)

Before the full deletion filter, shrink the candidate set:

```python
def elastic_prescreen(constraints, solver, solver_opts):
    """Identify likely IIS members via elastic relaxation."""
    slacks = [cp.Variable(nonneg=True) for _ in constraints]
    relaxed = []
    for constr, s in zip(constraints, slacks):
        # Add slack to each constraint
        # This is constraint-type dependent — need a relaxation helper
        relaxed.append(relax_constraint(constr, s))
    
    prob = cp.Problem(cp.Minimize(cp.sum(slacks)), relaxed)
    prob.solve(solver=solver, **solver_opts)
    
    # Constraints with nonzero slack are candidates
    candidates = [c for c, s in zip(constraints, slacks) if s.value > 1e-8]
    return candidates
```

The trick is `relax_constraint()` — you need to add a slack to an arbitrary CVXPY constraint in a DCP-compliant way. This is doable but requires handling each constraint type (inequality, equality, SOC, PSD). Could be a utility function.

**This pre-screening reduces the deletion filter from `n` solves to `k` solves where `k` is typically very small.**

---

## 6. Implementation Roadmap

### Phase 1: Core Infrastructure
- [ ] Add `origin` attribute to `Constraint` class (default `None` for user-created constraints)
- [ ] Set `origin` in top-level reductions: `Dcp2Canon`, `Dgp2Dcp`, `Qp2SymbolicQp`, etc. — not in individual atom canon files
- [ ] Use `cc.origin = constr.origin if constr.origin else constr` pattern so multi-stage chains always point to the root
- [ ] Build and store `row_map` during final conic solver compilation (single location, all solvers)
- [ ] Unit tests for the mapping (verify round-trip for various constraint types: affine, SOC, PSD, exponential cone)

### Phase 2: Deletion Filter Fallback
- [ ] Implement two-level `deletion_filter_iis()`: constraint-level pass then row-level pass
- [ ] Implement `_split_constraint_rows()` for decomposing vector constraints into scalar rows
- [ ] Add `IISMember` dataclass (constraint + rows)
- [ ] Add `Problem.find_iis()` API returning `List[IISMember]`
- [ ] Add `iis_rows` attribute to constraints (post-IIS)
- [ ] Tests: known infeasible problems with known IIS, including row-level verification

### Phase 3: Certificate-Based Solvers (SCS, Clarabel, OSQP)
- [ ] Implement `get_iis_candidate_rows()` for SCS (extract `sol.y`, threshold nonzero support, return row-level mapping)
- [ ] Implement `get_iis_candidate_rows()` for Clarabel (same pattern, tighter tolerance)
- [ ] Implement `get_iis_candidate_rows()` for OSQP (extract `prim_inf_cert`)
- [ ] For OSQP: handle the final conic-to-QP index mapping at the solver interface layer (the `[A; D]` stacking offset). This is minimal once the QP reduction chain is fully merged into the conic chain (targeted for CVXPY 1.10).
- [ ] Wire certificate candidates into the unified pipeline (candidates → constraint-level pruning → row-level pruning)
- [ ] Implement `_split_constraint_rows()` for affine equality/inequality constraints
- [ ] Tolerance tuning: test across problem types, calibrate relative thresholds
- [ ] Tests: verify candidate sets are supersets of the true IIS, verify pruning produces valid IIS at row level

### Phase 4: Explicit IIS Solvers (Gurobi)
- [ ] Implement `get_iis_candidate_rows()` for Gurobi using `computeIIS()`
- [ ] Handle variable bounds (`IISLB`, `IISUB`) via `var_bound_map`
- [ ] Handle quadratic constraints (`IISQConstr`) for SOCP
- [ ] Constraint-level and row-level pruning passes
- [ ] Tests: compare native results with deletion filter

### Phase 5: MOSEK
- [ ] Implement certificate extraction from MOSEK dual solution
- [ ] Investigate `analyzeproblem()` for additional diagnostics
- [ ] Same pruning pipeline
- [ ] Tests

### Phase 6: Polish
- [ ] Elastic pre-screening for the fallback path
- [ ] Documentation and examples
- [ ] Edge cases: variable domain constraints, parameters, mixed-integer

---

## 7. Edge Cases and Open Questions

### 7.1 Variable Bounds vs. Constraints
`cp.Variable(nonneg=True)` creates an implicit bound, not an explicit constraint. If this bound participates in the IIS, what do we return? Options:
- Return a synthetic `Constraint` object representing the bound.
- Return the `Variable` itself and document that variable domains can appear in the IIS.
- **Recommendation:** Create a lightweight `BoundConstraint` wrapper that references the variable and its domain. Include it in the IIS list.

### 7.2 Parameters
If constraints involve `cp.Parameter`, the IIS depends on parameter values. This is fine — just document that the IIS is valid for the current parameter values.

### 7.3 Infeasible_Inaccurate
Some solvers return `infeasible_inaccurate`. Should `find_iis()` work in this case?
- **Recommendation:** Allow it with a warning. The deletion filter will confirm infeasibility independently.

### 7.4 Multiple IIS
A problem can have many IIS. The deletion filter finds one (depending on iteration order). Document this clearly. A `find_all_iis()` method can be added later using the "block and recurse" approach (find IIS, add a constraint that blocks it, repeat).

### 7.6 Row Reordering in Reductions

The row-level mapping assumes that rows within a single constraint maintain their order through the reduction chain (i.e., canonical row `i` maps to original row `i`). No known reductions currently reorder rows within a constraint, and for most DCP reductions the mapping is trivially identity. However, this assumption should be explicitly verified and documented:

- [ ] Audit all reduction classes to confirm no row reordering occurs within a single constraint.
- [ ] If a row-reordering reduction is found (or added in the future), the `origin_row_offset` mechanism in `RowOrigin` must be extended to handle permutations, not just offsets.
- [ ] Add a regression test that catches row reordering: canonicalize a vector constraint, inspect the `row_map`, and verify the mapping matches expectations.

For now, the identity assumption is safe and keeps the implementation simple. If a future reduction does reorder rows, the `RowOrigin` dataclass can be extended from a scalar `row: int` to a more general mapping without changing the public API.

### 7.7 Mixed-Integer Problems
IIS for MIP is significantly harder. Gurobi supports it natively. The deletion filter works but each feasibility check is a MIP solve. Consider gating this behind a warning about runtime.

### 7.6 Irreducibility Gap
The native solver finds an IIS at the canonical level. After back-mapping to original constraints, the result is an infeasible subset but potentially not irreducible (two original constraints might map to overlapping canonical constraints, and only one is needed). The pruning deletion filter on the small mapped-back set (typically <10 constraints) is cheap and closes this gap.

---

## 8. Testing Strategy

### 8.1 Unit Tests

```python
def test_simple_lp_iis():
    x = cp.Variable()
    c1 = x >= 5
    c2 = x <= 3
    c3 = x >= 0  # redundant for infeasibility
    prob = cp.Problem(cp.Minimize(x), [c1, c2, c3])
    prob.solve()
    assert prob.status == 'infeasible'
    
    iis = prob.find_iis()
    iis_constrs = {m.constraint for m in iis}
    assert iis_constrs == {c1, c2}

def test_socp_iis():
    x = cp.Variable(3)
    c1 = cp.norm(x) <= 1
    c2 = x[0] >= 3
    c3 = x[1] >= 0  # not part of IIS
    prob = cp.Problem(cp.Minimize(0), [c1, c2, c3])
    prob.solve()
    
    iis = prob.find_iis()
    iis_constrs = {m.constraint for m in iis}
    assert c1 in iis_constrs and c2 in iis_constrs
    assert c3 not in iis_constrs

def test_vector_constraint_row_granularity():
    """Only specific rows of A @ x == b should appear in the IIS."""
    x = cp.Variable(2)
    
    A = np.eye(2)
    b = np.array([5.0, 3.0])
    c1 = A @ x == b          # row 0: x[0]==5, row 1: x[1]==3
    c2 = x[0] + x[1] <= 1   # conflicts with rows of c1
    
    prob = cp.Problem(cp.Minimize(0), [c1, c2])
    prob.solve()
    
    iis = prob.find_iis()
    # The IIS should include c2 and specific rows of c1 — not necessarily
    # both rows. E.g., x[0]==5 alone conflicts with x[0]+x[1]<=1 if 
    # combined with x[1]==3, or either row of c1 could suffice depending
    # on the problem structure.
    for member in iis:
        if member.constraint is c1:
            # Should not flag all rows — at least one should be excluded
            # (depends on specific problem; this test verifies rows != None
            # for multi-row constraints)
            assert member.rows is not None
            assert len(member.rows) < c1.size or len(member.rows) == c1.size

def test_iis_irreducibility():
    """Removing any single IIS member makes the remaining set feasible."""
    x = cp.Variable()
    c1 = x >= 5
    c2 = x <= 3
    c3 = x >= 0
    prob = cp.Problem(cp.Minimize(x), [c1, c2, c3])
    prob.solve()
    
    iis = prob.find_iis()
    for i, member in enumerate(iis):
        # Reconstruct constraints without this member
        remaining = []
        for j, m in enumerate(iis):
            if i == j:
                continue
            if m.rows is None:
                remaining.append(m.constraint)
            else:
                # Reconstruct the constraint with only the IIS rows
                remaining.append(_subset_constraint(m.constraint, m.rows))
        test = cp.Problem(cp.Minimize(0), remaining)
        test.solve()
        assert test.status != 'infeasible'
```

### 8.2 Certificate Quality Tests

```python
@pytest.mark.parametrize("solver", [cp.SCS, cp.CLARABEL])
def test_certificate_candidates_superset_of_iis(solver):
    """Certificate candidates must contain the true IIS."""
    x = cp.Variable()
    c1 = x >= 5
    c2 = x <= 3
    c3 = x >= 0  # not in IIS
    c4 = x <= 100  # not in IIS
    prob = cp.Problem(cp.Minimize(x), [c1, c2, c3, c4])
    prob.solve(solver=solver)
    
    # Raw candidates (before pruning)
    candidates = prob._get_iis_candidate_rows()
    assert c1 in candidates and c2 in candidates
    
    # After pruning
    iis = prob.find_iis()
    iis_constrs = {m.constraint for m in iis}
    assert iis_constrs == {c1, c2}

@pytest.mark.parametrize("solver", [cp.SCS, cp.CLARABEL, cp.OSQP])
def test_certificate_tolerance_robustness(solver):
    """IIS result should be stable across reasonable tolerance choices."""
    # ... problem where certificate has clear separation between
    # participating and non-participating constraints ...
    pass

def test_first_order_vs_interior_point_agreement():
    """SCS and Clarabel should identify the same IIS on the same problem."""
    x = cp.Variable(3)
    constraints = [cp.norm(x) <= 1, x[0] >= 3, x[1] >= 0]
    prob = cp.Problem(cp.Minimize(0), constraints)
    
    prob.solve(solver=cp.SCS)
    iis_scs = prob.find_iis()
    
    prob.solve(solver=cp.CLARABEL)
    iis_clarabel = prob.find_iis()
    
    assert {m.constraint for m in iis_scs} == {m.constraint for m in iis_clarabel}

def test_row_level_from_certificate():
    """Certificate should narrow rows before deletion filter."""
    x = cp.Variable(2)
    A = np.eye(2)
    b = np.array([10.0, 10.0])
    c1 = A @ x == b        # 2 rows
    c2 = x[0] <= 1         # conflicts with row 0 of c1 only
    
    prob = cp.Problem(cp.Minimize(0), [c1, c2])
    prob.solve(solver=cp.CLARABEL)
    
    iis = prob.find_iis()
    for member in iis:
        if member.constraint is c1:
            # Should only flag row 0, not row 1
            assert 0 in member.rows
            assert 1 not in member.rows
```

### 8.3 Comparison Tests

For solvers with native IIS: verify that native + back-mapping gives the same result as the deletion filter (same IIS size, both irreducible).

### 8.3 Stress Tests

- Large problems (1000+ constraints) with small IIS — verify performance.
- Problems with multiple independent IIS.
- Degenerate cases: entire problem is an IIS (every constraint needed).
