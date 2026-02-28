# broadcast_to sparse constant canonicalization — design notes

## Problem

`broadcast_to.numeric()` calls `np.broadcast_to(sparse_matrix, shape)` which
creates an object array of sparse matrices instead of a proper result.

This surfaces during canonicalization: `Atom.canonicalize()` (atom.py:374-376)
shortcuts constant atoms via `Constant(self.value)`, which triggers `numeric()`
on the constant `broadcast_to` expression. The resulting object array cannot be
wrapped in a `Constant`.

Concrete failure: `multiply(t_expanded, sp.eye_array(n))` in `lambda_max_canon`
creates a `broadcast_to(Constant(sparse_eye), 3d_shape)` during expression
broadcasting, which is constant and hits the shortcut.

## Proposed solutions

### A. Override `canonicalize()` in `broadcast_to`

Skip the constant shortcut; always use `graph_implementation()`:

```python
def canonicalize(self):
    arg_objs = []
    constraints = []
    for arg in self.args:
        obj, constr = arg.canonical_form
        arg_objs.append(obj)
        constraints += constr
    data = self.get_data()
    graph_obj, graph_constr = self.graph_implementation(arg_objs, self.shape, data)
    return graph_obj, constraints + graph_constr
```

**Pro**: No densification. graph_implementation → LinOp → scipy backend handles
sparse correctly via row-index broadcasting.
**Con**: Loses the constant-folding optimization for broadcast_to (minor — the
scipy backend is efficient anyway). `.value` still broken for sparse 3D broadcast.

### B. Override `canonicalize()` in `AffAtom` (broader fix)

Same as A but applied to all affine atoms. AffAtom always has a valid
`graph_implementation`, so the constant shortcut is purely an optimization.

**Pro**: Prevents the same class of bug in any AffAtom with sparse constants.
**Con**: May regress performance for simple constant-folding cases. Needs
benchmarking.

### C. Fix `numeric()` to handle sparse inputs

Add `@AffAtom.numpy_numeric` decorator (like reshape, transpose do), which
calls `const_to_matrix()` → `toarray()` on sparse inputs before broadcasting.

**Pro**: Simple one-line change. Fixes both canonicalization and `.value`.
**Con**: Densifies the sparse matrix during numeric evaluation. For large sparse
matrices broadcast to 3D, this could be expensive.

### D. Fix `numeric()` with sparse-aware broadcasting

Handle sparse inputs explicitly using `scipy.sparse.coo_array`, which supports
N-D. Build a broadcast coo_array by repeating the 2D COO data/coords across the
new dimensions.

**Pro**: Preserves sparsity fully, no densification.
**Con**: More complex. Requires CVXPY's `Constant` and interface layer to handle
N-D sparse arrays (currently they assume 2D sparse). May need updates to
`is_sparse()`, `const_to_matrix()`, and `Constant.__init__()`.

### E. Skip constant shortcut only when sparse leaves are present

Add a `has_sparse_const_leaf` check (tree walk) to the constant shortcut in
`Atom.canonicalize()`. Only skip `Constant(self.value)` and fall through to
`graph_implementation()` when the expression tree contains a sparse constant
leaf.

```python
def canonicalize(self):
    if self.is_constant() and not self.parameters():
        if not self._has_sparse_const_leaf():
            return Constant(self.value).canonical_form
    # fall through to graph_implementation path
    ...
```

```python
# On Constant:
def _has_sparse_const_leaf(self):
    return self._sparse

# On Atom (cached via @lazyprop or similar):
def _has_sparse_const_leaf(self):
    return any(a._has_sparse_const_leaf() for a in self.args)
```

**Pro**: Preserves constant folding for dense constants (the common case).
Only pays the graph_implementation cost when sparse is actually involved.
Works for all atoms, not just broadcast_to. Cacheable and trivial to implement.
**Con**: Adds a method to the expression tree base classes.

### F. Hybrid: E + D

Override `canonicalize()` in broadcast_to (option A) so the constant shortcut
is never hit during solving. Separately, fix `numeric()` with sparse-aware
broadcasting (option D) so that `.value` also works correctly.

**Pro**: Fixes both solving and value evaluation. graph_implementation handles
the solving path efficiently; sparse-aware numeric handles value evaluation.
**Con**: Two changes instead of one.
