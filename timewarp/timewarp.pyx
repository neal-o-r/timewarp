import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs, INFINITY

cdef inline double d_min(double a, double b, double c):
    if a < b and a < c:
        return a
    elif b < c:
        return b
    else:
        return c

def create_dist_mat(np.ndarray[np.float64_t, ndim=2, mode="c"] a, 
                    np.ndarray[np.float64_t, ndim=2, mode="c"] b):
    return _create_dist_mat(a, b)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double chebyshev(double[::1] a, double[::1] b):
    cdef int i
    cdef double d
    d = 0
    for i in range(a.shape[0]):
        d += abs(a[i] - b[i])
    return d


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:, ::1] _create_dist_mat(double[:, ::1] a, double[:, ::1] b):
    cdef double[:, ::1] dist_mat = np.empty((a.shape[0] + 1, b.shape[0] + 1), 
                                    dtype=np.float64)
    dist_mat[:] = INFINITY
    dist_mat[0, 0] = 0
    cdef int i, j
    for i in range(1, dist_mat.shape[0]):
        for j in range(1, dist_mat.shape[1]):
            dist_mat[i, j] = chebyshev(a[i - 1], b[j - 1]) +\
                d_min(dist_mat[i - 1, j], dist_mat[i, j - 1], dist_mat[i - 1, j - 1])

    return dist_mat[1:, 1:]
