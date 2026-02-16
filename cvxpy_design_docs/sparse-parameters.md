# Plan: Respect Dimension-Reducing Attributes on Parameters

## Context

When a user creates a `Parameter` with a dimension-reducing attribute (`sparsity`, `diag`, `symmetric`, `PSD`, `NSD`), the attribute is accepted and stored but the canonicalization pipeline ignores it. The solver allocates `prod(shape)` entries in the parameter vector instead of the reduced size. This wastes memory and computation.

For **Variables**, `CvxAttr2Constr` already handles these attributes (lines 147-193 of `cvx_attr2constr.py`): it creates a reduced-size variable and a coefficient/scatter matrix to reconstruct the full shape. But Parameters bypass this — they go straight through `canonicalize()` → `lu.create_param(self.shape, self.id)` with the full shape.

The fix: modify `Parameter.canonicalize()` to emit a reduced LinOp tree (reduced param + reconstruction matrix), and update the pipeline to use reduced sizes and values.

### Dimension-reducing attributes and their reductions

| Attribute | Original size | Canonical size | Reconstruction |
|-----------|--------------|----------------|----------------|
| `sparsity` | `prod(shape)` | `n` (nnz count) | Scatter matrix `(prod(shape), n)` |
| `diag` | `n*n` | `n` | `DIAG_VEC` LinOp |
| `symmetric`/`PSD`/`NSD` | `n*n` | `n*(n+1)/2` | `upper_tri_to_full(n)` matrix `(n*n, n*(n+1)/2)` |

## Changes

### 1. `cvxpy/expressions/constants/parameter.py`

**Add imports** at top:
```python
import numpy as np
import scipy.sparse as sp
from cvxpy.atoms.affine.upper_tri import upper_tri_to_full
```

**Add `canon_param_size` property:**
```python
@property
def canon_param_size(self) -> int:
    """Size of this parameter in the canonical (solver) representation."""
    if self.sparse_idx is not None:
        return len(self.sparse_idx[0])
    elif self.attributes['diag']:
        return self.shape[0]
    elif self.attributes['symmetric'] or self.attributes['PSD'] or self.attributes['NSD']:
        n = self.shape[0]
        return n * (n + 1) // 2
    return self.size
```

**Add `canon_param_value()` method:**
Returns the reduced-form value (analogous to `lower_value()` in `cvx_attr2constr.py:96-102`):
```python
def canon_param_value(self):
    """Returns the parameter value in canonical (reduced) form."""
    val = self.value
    if val is None:
        return None
    if self.sparse_idx is not None:
        if isinstance(self._value, sp.coo_array):
            return self._value.data
        return np.asarray(val)[self.sparse_idx]
    elif self.attributes['diag']:
        return np.diag(val)
    elif self.attributes['symmetric'] or self.attributes['PSD'] or self.attributes['NSD']:
        return val[np.triu_indices(self.shape[0])]
    return val
```

**Modify `canonicalize()`** to emit reduced LinOp trees:

- **sparsity**: `reshape(scatter_matrix @ reduced_param, shape)` — scatter matrix from `np.ravel_multi_index(sparse_idx, shape, order='F')`, exactly like `CvxAttr2Constr` lines 183-188.
- **diag**: `diag_vec(reduced_param)` — uses `lu.diag_vec()` from `lin_utils.py:496`.
- **symmetric/PSD/NSD**: `reshape(upper_tri_to_full(n) @ reduced_param, (n, n))` — uses `upper_tri_to_full()` from `cvxpy/atoms/affine/upper_tri.py:147`, exactly like `CvxAttr2Constr` lines 147-155.

LinOp utilities from `cvxpy/lin_ops/lin_utils.py`:
- `lu.create_param(shape, id)` → `LinOp(PARAM, shape, [], id)`
- `lu.create_const(value, shape, sparse=True)` → `LinOp(SPARSE_CONST, shape, [], value)`
- `lu.mul_expr(lh_const, rh_expr, shape)` → `LinOp(MUL, shape, [rh_expr], lh_const)`
- `lu.reshape(operator, shape)` → `LinOp(RESHAPE, shape, [operator], None)`
- `lu.diag_vec(operator, k=0)` → `LinOp(DIAG_VEC, (n,n), [operator], k)`

### 2. `cvxpy/reductions/inverse_data.py`

In `InverseData.__init__()` (lines 33-37), use `canon_param_size`:

```python
for param in problem.parameters():
    self.param_shapes[param.id] = param.shape
    self.param_to_size[param.id] = param.canon_param_size
    self.param_id_map[param.id] = offset
    offset += param.canon_param_size
```

### 3. `cvxpy/reductions/dcp2cone/cone_matrix_stuffing.py`

**a) `ParamConeProg.__init__()` (lines 197-198):** Use canonical sizes:
```python
self.param_id_to_size = {p.id: p.canon_param_size for p in self.parameters}
self.total_param_size = sum(self.param_id_to_size.values())
```

**b) `ParamConeProg.apply_parameters()` (lines 227-229):** Extract canonical values:
```python
def param_value(idx):
    if id_to_param_value is not None:
        return id_to_param_value[idx]
    param = self.id_to_param[idx]
    return np.array(param.canon_param_value())
```

**c) `ParamConeProg.apply_param_jac()` (lines 296-301):** Use canonical sizes and scatter delta back to full shape (analogous to `recover_value_for_variable()` in `cvx_attr2constr.py:76-93`):
```python
for param_id, col in self.param_id_to_col.items():
    if param_id in active_params:
        param = self.id_to_param[param_id]
        size = self.param_id_to_size[param_id]
        delta = del_param_vec[col:col + size]
        if param.attributes['diag']:
            param_id_to_delta_param[param_id] = np.diag(delta.flatten(order='F'))
        elif param.attributes['symmetric'] or param.attributes['PSD'] or param.attributes['NSD']:
            n = param.shape[0]
            value = np.zeros(param.shape)
            idxs = np.triu_indices(n)
            value[idxs] = delta.flatten(order='F')
            param_id_to_delta_param[param_id] = value + value.T - np.diag(value.diagonal())
        elif param.sparse_idx is not None:
            full_delta = np.zeros(param.shape)
            full_delta[param.sparse_idx] = delta
            param_id_to_delta_param[param_id] = full_delta
        else:
            param_id_to_delta_param[param_id] = np.reshape(delta, param.shape, order='F')
```

### 4. `cvxpy/cvxcore/python/canonInterface.py`

Update the TODO on line 50 to note that structured parameters are now handled:
```python
# Structured parameters (sparse, diag, symmetric, PSD, NSD) are handled
# via reduced canonical forms in Parameter.canonicalize().
```

### 5. `cvxpy/tests/test_attributes.py`

Add tests in the existing `TestMultipleAttributes` class:

- **`test_sparse_parameter_reduces_param_size`**: Create a problem with a sparse parameter, get problem data, verify `total_param_size` reflects the reduced size.
- **`test_sparse_parameter_solve`**: Solve with a sparse parameter and verify correctness.
- **`test_sparse_parameter_dpp_resolve`**: DPP re-solve with changed sparse parameter values.
- **`test_diag_parameter_solve`**: Solve with a `diag=True` parameter and verify.
- **`test_symmetric_parameter_solve`**: Solve with a `symmetric=True` parameter and verify.
- **`test_psd_parameter_solve`**: Solve with a `PSD=True` parameter and verify.

All tests use `solver=cp.CLARABEL`.

## Files Modified

| File | Change |
|------|--------|
| `cvxpy/expressions/constants/parameter.py` | Add `canon_param_size`, `canon_param_value()`, modify `canonicalize()` |
| `cvxpy/reductions/inverse_data.py` | Use `canon_param_size` in size/offset computation |
| `cvxpy/reductions/dcp2cone/cone_matrix_stuffing.py` | Use canonical sizes/values in `__init__`, `apply_parameters`, `apply_param_jac` |
| `cvxpy/cvxcore/python/canonInterface.py` | Update TODO comment |
| `cvxpy/tests/test_attributes.py` | Add tests |

## Verification

```bash
# Run existing attribute tests (should all still pass)
uv run pytest cvxpy/tests/test_attributes.py -xvs

# Run DPP tests (uses parameters heavily)
uv run pytest cvxpy/tests/test_dpp.py -xvs

# Run broader test suite for regressions
uv run pytest cvxpy/tests/test_problem.py -x
```
