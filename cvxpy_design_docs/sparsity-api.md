# Public APIs for sparse operations

Author: Parth Nobel  
Target Audience: William + CVXPY Maintainers  
Date: November 18, 2024  
Conclusion: Accepted with option 3.

This doc considers miscellaneous options around how to have more ergonomic support for sparse variables in CVXPY.

### Principles

* Taking `.value` of any subexpression (and replacing cp with an appropriate library) should always be equivalent to taking `.value` of an expression as a whole
* Things should "just work"
* Sparse variables and parameters must feel like first-class features. Other expressions need not have good sparsity support

## Scipy Data Structure

### Proposal

If a Leaf `leaf` has a non-`None` sparsity attribute, `leaf.value` is of type `scipy.sparse.coo_array` (nD COO is supported as of SciPy 1.15.0 coming out in December 2024).
Specifically, `leaf.value.coords == tuple(np.array(axis) for axis in leaf.sparsity)`.


### What to do before 1.15.0 and when the user uses older SciPy
We can have a datatype `cp.compat.scipy_coo_array` that is defined in the following way:
```python
if scipy.__version__  < 1.15.0:
    class scipy_coo_array_compat:
        def __init__(self, data_coords, dtype, shape):
            self.shape = shape
            self.dtype = dtype
            self.data, self.coords = data_coords
    scipy_coo_array = scipy_coo_array_compat
else:
    scipy_coo_array = scipy.sparse.coo_array
```

This means if a user wants to support using their library with old SciPy they should use `cp.compat.scipy_coo_array` and never do anything other than construct and destruct the type.
It is not clear if SciPy1.15 will have support for much more on the nd-array case, which is part of why I support this.
Eventually, when we drop support for SciPy 1.14, we can just make `cp.compat.scipy_coo_array` always be equal to `scipy.sparse.coo_array` and remove it from documentation.

### Pros/Cons

Pros: 

- Fairly natural API that satisfies all the principles, and will have good library integration 

Cons:

- `.value` has always been an np.ndarray.
- The hack for SciPy 1.14  support doesn't feel great.

## assparse Function
We could require that `.value` is always a np.ndarray, and add a new function that turns a CVXPY Variable of shape `(nnz,)` and a coordinate list to create a sparse CVXPY expression.

### Example

```python
x = cp.Variable(3)
cp.coo_sparse_expr(x, (np.array([0, 5, 6]), np.array([0, 2, 6])), shape=(8, 7))
```

### Pros/Cons

Pros:

- Symmetric with the API for `coo_array`
- User has to decide how to pack and unpack the data
- `.value` stays always an `np.ndarray`
- Mirrors how the backends handle this

Cons:

- May require migration of existing code
- Feels like a hack
- Leaves weird footguns around accessing `.value` creating a huge dense array

## Sparse attribute

Another alternative is to have a `.value_sparse` that can be written to by or read from as a `scipy.sparse.coo_array`.

### Pros/Cons

Pros:

- `.value` stays always an `np.ndarray`
- No weird function API
- Plays nice with ecosystem
- Clearly built into the ecosystem.

Cons:

- Leaves weird footguns around accessing `.value` creating a huge dense array
