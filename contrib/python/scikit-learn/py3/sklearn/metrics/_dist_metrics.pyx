# By Jake Vanderplas (2013) <jakevdp@cs.washington.edu>
# written for the scikit-learn project
# License: BSD

import numpy as np
cimport numpy as cnp

cnp.import_array()  # required in order to use C-API

from libc.math cimport fabs, sqrt, exp, pow, cos, sin, asin

from scipy.sparse import csr_matrix, issparse
from ..utils._typedefs cimport float64_t, float32_t, int32_t, intp_t
from ..utils import check_array
from ..utils.fixes import parse_version, sp_base_version

cdef inline double fmax(double a, double b) noexcept nogil:
    return max(a, b)


######################################################################
# newObj function
#  this is a helper function for pickling
def newObj(obj):
    return obj.__new__(obj)


BOOL_METRICS = [
    "hamming",
    "jaccard",
    "dice",
    "rogerstanimoto",
    "russellrao",
    "sokalmichener",
    "sokalsneath",
]
if sp_base_version < parse_version("1.11"):
    # Deprecated in SciPy 1.9 and removed in SciPy 1.11
    BOOL_METRICS += ["kulsinski"]
if sp_base_version < parse_version("1.9"):
    # Deprecated in SciPy 1.0 and removed in SciPy 1.9
    BOOL_METRICS += ["matching"]

def get_valid_metric_ids(L):
    """Given an iterable of metric class names or class identifiers,
    return a list of metric IDs which map to those classes.

    Example:
    >>> L = get_valid_metric_ids([EuclideanDistance, 'ManhattanDistance'])
    >>> sorted(L)
    ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan']
    """
    return [key for (key, val) in METRIC_MAPPING64.items()
            if (val.__name__ in L) or (val in L)]

cdef class DistanceMetric:
    """Uniform interface for fast distance metric functions.

    The `DistanceMetric` class provides a convenient way to compute pairwise distances
    between samples. It supports various distance metrics, such as Euclidean distance,
    Manhattan distance, and more.

    The `pairwise` method can be used to compute pairwise distances between samples in
    the input arrays. It returns a distance matrix representing the distances between
    all pairs of samples.

    The :meth:`get_metric` method allows you to retrieve a specific metric using its
    string identifier.

    Examples
    --------
    >>> from sklearn.metrics import DistanceMetric
    >>> dist = DistanceMetric.get_metric('euclidean')
    >>> X = [[1, 2], [3, 4], [5, 6]]
    >>> Y = [[7, 8], [9, 10]]
    >>> dist.pairwise(X,Y)
    array([[7.81..., 10.63...]
           [5.65...,  8.48...]
           [1.41...,  4.24...]])

    Available Metrics

    The following lists the string metric identifiers and the associated
    distance metric classes:

    **Metrics intended for real-valued vector spaces:**

    ==============  ====================  ========  ===============================
    identifier      class name            args      distance function
    --------------  --------------------  --------  -------------------------------
    "euclidean"     EuclideanDistance     -         ``sqrt(sum((x - y)^2))``
    "manhattan"     ManhattanDistance     -         ``sum(|x - y|)``
    "chebyshev"     ChebyshevDistance     -         ``max(|x - y|)``
    "minkowski"     MinkowskiDistance     p, w      ``sum(w * |x - y|^p)^(1/p)``
    "seuclidean"    SEuclideanDistance    V         ``sqrt(sum((x - y)^2 / V))``
    "mahalanobis"   MahalanobisDistance   V or VI   ``sqrt((x - y)' V^-1 (x - y))``
    ==============  ====================  ========  ===============================

    **Metrics intended for two-dimensional vector spaces:**  Note that the haversine
    distance metric requires data in the form of [latitude, longitude] and both
    inputs and outputs are in units of radians.

    ============  ==================  ===============================================================
    identifier    class name          distance function
    ------------  ------------------  ---------------------------------------------------------------
    "haversine"   HaversineDistance   ``2 arcsin(sqrt(sin^2(0.5*dx) + cos(x1)cos(x2)sin^2(0.5*dy)))``
    ============  ==================  ===============================================================


    **Metrics intended for integer-valued vector spaces:**  Though intended
    for integer-valued vectors, these are also valid metrics in the case of
    real-valued vectors.

    =============  ====================  ========================================
    identifier     class name            distance function
    -------------  --------------------  ----------------------------------------
    "hamming"      HammingDistance       ``N_unequal(x, y) / N_tot``
    "canberra"     CanberraDistance      ``sum(|x - y| / (|x| + |y|))``
    "braycurtis"   BrayCurtisDistance    ``sum(|x - y|) / (sum(|x|) + sum(|y|))``
    =============  ====================  ========================================

    **Metrics intended for boolean-valued vector spaces:**  Any nonzero entry
    is evaluated to "True".  In the listings below, the following
    abbreviations are used:

     - N  : number of dimensions
     - NTT : number of dims in which both values are True
     - NTF : number of dims in which the first value is True, second is False
     - NFT : number of dims in which the first value is False, second is True
     - NFF : number of dims in which both values are False
     - NNEQ : number of non-equal dimensions, NNEQ = NTF + NFT
     - NNZ : number of nonzero dimensions, NNZ = NTF + NFT + NTT

    =================  =======================  ===============================
    identifier         class name               distance function
    -----------------  -----------------------  -------------------------------
    "jaccard"          JaccardDistance          NNEQ / NNZ
    "matching"         MatchingDistance         NNEQ / N
    "dice"             DiceDistance             NNEQ / (NTT + NNZ)
    "kulsinski"        KulsinskiDistance        (NNEQ + N - NTT) / (NNEQ + N)
    "rogerstanimoto"   RogersTanimotoDistance   2 * NNEQ / (N + NNEQ)
    "russellrao"       RussellRaoDistance       (N - NTT) / N
    "sokalmichener"    SokalMichenerDistance    2 * NNEQ / (N + NNEQ)
    "sokalsneath"      SokalSneathDistance      NNEQ / (NNEQ + 0.5 * NTT)
    =================  =======================  ===============================

    **User-defined distance:**

    ===========    ===============    =======
    identifier     class name         args
    -----------    ---------------    -------
    "pyfunc"       PyFuncDistance     func
    ===========    ===============    =======

    Here ``func`` is a function which takes two one-dimensional numpy
    arrays, and returns a distance.  Note that in order to be used within
    the BallTree, the distance must be a true metric:
    i.e. it must satisfy the following properties

    1) Non-negativity: d(x, y) >= 0
    2) Identity: d(x, y) = 0 if and only if x == y
    3) Symmetry: d(x, y) = d(y, x)
    4) Triangle Inequality: d(x, y) + d(y, z) >= d(x, z)

    Because of the Python object overhead involved in calling the python
    function, this will be fairly slow, but it will have the same
    scaling as other distances.
    """
    @classmethod
    def get_metric(cls, metric, dtype=np.float64, **kwargs):
        """Get the given distance metric from the string identifier.

        See the docstring of DistanceMetric for a list of available metrics.

        Parameters
        ----------
        metric : str or class name
            The string identifier or class name of the desired distance metric.
            See the documentation of the `DistanceMetric` class for a list of
            available metrics.

        dtype : {np.float32, np.float64}, default=np.float64
            The data type of the input on which the metric will be applied.
            This affects the precision of the computed distances.
            By default, it is set to `np.float64`.

        **kwargs
            Additional keyword arguments that will be passed to the requested metric.
            These arguments can be used to customize the behavior of the specific
            metric.

        Returns
        -------
        metric_obj : instance of the requested metric
            An instance of the requested distance metric class.
        """
        if dtype == np.float32:
            specialized_class = DistanceMetric32
        elif dtype == np.float64:
            specialized_class = DistanceMetric64
        else:
            raise ValueError(
                f"Unexpected dtype {dtype} provided. Please select a dtype from"
                " {np.float32, np.float64}"
            )

        return specialized_class.get_metric(metric, **kwargs)

######################################################################
# metric mappings
#  These map from metric id strings to class names
METRIC_MAPPING64 = {
    'euclidean': EuclideanDistance64,
    'l2': EuclideanDistance64,
    'minkowski': MinkowskiDistance64,
    'p': MinkowskiDistance64,
    'manhattan': ManhattanDistance64,
    'cityblock': ManhattanDistance64,
    'l1': ManhattanDistance64,
    'chebyshev': ChebyshevDistance64,
    'infinity': ChebyshevDistance64,
    'seuclidean': SEuclideanDistance64,
    'mahalanobis': MahalanobisDistance64,
    'hamming': HammingDistance64,
    'canberra': CanberraDistance64,
    'braycurtis': BrayCurtisDistance64,
    'matching': MatchingDistance64,
    'jaccard': JaccardDistance64,
    'dice': DiceDistance64,
    'kulsinski': KulsinskiDistance64,
    'rogerstanimoto': RogersTanimotoDistance64,
    'russellrao': RussellRaoDistance64,
    'sokalmichener': SokalMichenerDistance64,
    'sokalsneath': SokalSneathDistance64,
    'haversine': HaversineDistance64,
    'pyfunc': PyFuncDistance64,
}

cdef inline object _buffer_to_ndarray64(const float64_t* x, intp_t n):
    # Wrap a memory buffer with an ndarray. Warning: this is not robust.
    # In particular, if x is deallocated before the returned array goes
    # out of scope, this could cause memory errors.  Since there is not
    # a possibility of this for our use-case, this should be safe.

    # Note: this Segfaults unless np.import_array() is called above
    # TODO: remove the explicit cast to cnp.intp_t* when cython min version >= 3.0
    return cnp.PyArray_SimpleNewFromData(1, <cnp.intp_t*>&n, cnp.NPY_FLOAT64, <void*>x)


cdef float64_t INF64 = np.inf


######################################################################
# Distance Metric Classes
cdef class DistanceMetric64(DistanceMetric):
    """DistanceMetric class

    This class provides a uniform interface to fast distance metric
    functions.  The various metrics can be accessed via the :meth:`get_metric`
    class method and the metric string identifier (see below).

    Examples
    --------
    >>> from sklearn.metrics import DistanceMetric
    >>> dist = DistanceMetric.get_metric('euclidean')
    >>> X = [[0, 1, 2],
             [3, 4, 5]]
    >>> dist.pairwise(X)
    array([[ 0.        ,  5.19615242],
           [ 5.19615242,  0.        ]])

    Available Metrics

    The following lists the string metric identifiers and the associated
    distance metric classes:

    **Metrics intended for real-valued vector spaces:**

    ==============  ====================  ========  ===============================
    identifier      class name            args      distance function
    --------------  --------------------  --------  -------------------------------
    "euclidean"     EuclideanDistance     -         ``sqrt(sum((x - y)^2))``
    "manhattan"     ManhattanDistance     -         ``sum(|x - y|)``
    "chebyshev"     ChebyshevDistance     -         ``max(|x - y|)``
    "minkowski"     MinkowskiDistance     p, w      ``sum(w * |x - y|^p)^(1/p)``
    "seuclidean"    SEuclideanDistance    V         ``sqrt(sum((x - y)^2 / V))``
    "mahalanobis"   MahalanobisDistance   V or VI   ``sqrt((x - y)' V^-1 (x - y))``
    ==============  ====================  ========  ===============================

    **Metrics intended for two-dimensional vector spaces:**  Note that the haversine
    distance metric requires data in the form of [latitude, longitude] and both
    inputs and outputs are in units of radians.

    ============  ==================  ===============================================================
    identifier    class name          distance function
    ------------  ------------------  ---------------------------------------------------------------
    "haversine"   HaversineDistance   ``2 arcsin(sqrt(sin^2(0.5*dx) + cos(x1)cos(x2)sin^2(0.5*dy)))``
    ============  ==================  ===============================================================


    **Metrics intended for integer-valued vector spaces:**  Though intended
    for integer-valued vectors, these are also valid metrics in the case of
    real-valued vectors.

    =============  ====================  ========================================
    identifier     class name            distance function
    -------------  --------------------  ----------------------------------------
    "hamming"      HammingDistance       ``N_unequal(x, y) / N_tot``
    "canberra"     CanberraDistance      ``sum(|x - y| / (|x| + |y|))``
    "braycurtis"   BrayCurtisDistance    ``sum(|x - y|) / (sum(|x|) + sum(|y|))``
    =============  ====================  ========================================

    **Metrics intended for boolean-valued vector spaces:**  Any nonzero entry
    is evaluated to "True".  In the listings below, the following
    abbreviations are used:

     - N  : number of dimensions
     - NTT : number of dims in which both values are True
     - NTF : number of dims in which the first value is True, second is False
     - NFT : number of dims in which the first value is False, second is True
     - NFF : number of dims in which both values are False
     - NNEQ : number of non-equal dimensions, NNEQ = NTF + NFT
     - NNZ : number of nonzero dimensions, NNZ = NTF + NFT + NTT

    =================  =======================  ===============================
    identifier         class name               distance function
    -----------------  -----------------------  -------------------------------
    "jaccard"          JaccardDistance          NNEQ / NNZ
    "matching"         MatchingDistance         NNEQ / N
    "dice"             DiceDistance             NNEQ / (NTT + NNZ)
    "kulsinski"        KulsinskiDistance        (NNEQ + N - NTT) / (NNEQ + N)
    "rogerstanimoto"   RogersTanimotoDistance   2 * NNEQ / (N + NNEQ)
    "russellrao"       RussellRaoDistance       (N - NTT) / N
    "sokalmichener"    SokalMichenerDistance    2 * NNEQ / (N + NNEQ)
    "sokalsneath"      SokalSneathDistance      NNEQ / (NNEQ + 0.5 * NTT)
    =================  =======================  ===============================

    **User-defined distance:**

    ===========    ===============    =======
    identifier     class name         args
    -----------    ---------------    -------
    "pyfunc"       PyFuncDistance     func
    ===========    ===============    =======

    Here ``func`` is a function which takes two one-dimensional numpy
    arrays, and returns a distance.  Note that in order to be used within
    the BallTree, the distance must be a true metric:
    i.e. it must satisfy the following properties

    1) Non-negativity: d(x, y) >= 0
    2) Identity: d(x, y) = 0 if and only if x == y
    3) Symmetry: d(x, y) = d(y, x)
    4) Triangle Inequality: d(x, y) + d(y, z) >= d(x, z)

    Because of the Python object overhead involved in calling the python
    function, this will be fairly slow, but it will have the same
    scaling as other distances.
    """
    def __cinit__(self):
        self.p = 2
        self.vec = np.zeros(1, dtype=np.float64, order='C')
        self.mat = np.zeros((1, 1), dtype=np.float64, order='C')
        self.size = 1

    def __reduce__(self):
        """
        reduce method used for pickling
        """
        return (newObj, (self.__class__,), self.__getstate__())

    def __getstate__(self):
        """
        get state for pickling
        """
        if self.__class__.__name__ == "PyFuncDistance64":
            return (float(self.p), np.asarray(self.vec), np.asarray(self.mat), self.func, self.kwargs)
        return (float(self.p), np.asarray(self.vec), np.asarray(self.mat))

    def __setstate__(self, state):
        """
        set state for pickling
        """
        self.p = state[0]
        self.vec = state[1]
        self.mat = state[2]
        if self.__class__.__name__ == "PyFuncDistance64":
            self.func = state[3]
            self.kwargs = state[4]
        self.size = self.vec.shape[0]

    @classmethod
    def get_metric(cls, metric, **kwargs):
        """Get the given distance metric from the string identifier.

        See the docstring of DistanceMetric for a list of available metrics.

        Parameters
        ----------
        metric : str or class name
            The distance metric to use
        **kwargs
            additional arguments will be passed to the requested metric
        """
        if isinstance(metric, DistanceMetric64):
            return metric

        if callable(metric):
            return PyFuncDistance64(metric, **kwargs)

        # Map the metric string ID to the metric class
        if isinstance(metric, type) and issubclass(metric, DistanceMetric64):
            pass
        else:
            try:
                metric = METRIC_MAPPING64[metric]
            except:
                raise ValueError("Unrecognized metric '%s'" % metric)

        # In Minkowski special cases, return more efficient methods
        if metric is MinkowskiDistance64:
            p = kwargs.pop('p', 2)
            w = kwargs.pop('w', None)
            if p == 1 and w is None:
                return ManhattanDistance64(**kwargs)
            elif p == 2 and w is None:
                return EuclideanDistance64(**kwargs)
            elif np.isinf(p) and w is None:
                return ChebyshevDistance64(**kwargs)
            else:
                return MinkowskiDistance64(p, w, **kwargs)
        else:
            return metric(**kwargs)

    def __init__(self):
        if self.__class__ is DistanceMetric64:
            raise NotImplementedError("DistanceMetric64 is an abstract class")

    def _validate_data(self, X):
        """Validate the input data.

        This should be overridden in a base class if a specific input format
        is required.
        """
        return

    cdef float64_t dist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        """Compute the distance between vectors x1 and x2

        This should be overridden in a base class.
        """
        return -999

    cdef float64_t rdist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        """Compute the rank-preserving surrogate distance between vectors x1 and x2.

        This can optionally be overridden in a base class.

        The rank-preserving surrogate distance is any measure that yields the same
        rank as the distance, but is more efficient to compute. For example, the
        rank-preserving surrogate distance of the Euclidean metric is the
        squared-euclidean distance.
        """
        return self.dist(x1, x2, size)

    cdef int pdist(
        self,
        const float64_t[:, ::1] X,
        float64_t[:, ::1] D,
    ) except -1:
        """Compute the pairwise distances between points in X"""
        cdef intp_t i1, i2
        for i1 in range(X.shape[0]):
            for i2 in range(i1, X.shape[0]):
                D[i1, i2] = self.dist(&X[i1, 0], &X[i2, 0], X.shape[1])
                D[i2, i1] = D[i1, i2]
        return 0


    cdef int cdist(
        self,
        const float64_t[:, ::1] X,
        const float64_t[:, ::1] Y,
        float64_t[:, ::1] D,
    ) except -1:
        """Compute the cross-pairwise distances between arrays X and Y"""
        cdef intp_t i1, i2
        if X.shape[1] != Y.shape[1]:
            raise ValueError('X and Y must have the same second dimension')
        for i1 in range(X.shape[0]):
            for i2 in range(Y.shape[0]):
                D[i1, i2] = self.dist(&X[i1, 0], &Y[i2, 0], X.shape[1])
        return 0

    cdef float64_t dist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:
        """Compute the distance between vectors x1 and x2 represented
        under the CSR format.

        This must be overridden in a subclass.

        Notes
        -----
        0. The implementation of this method in subclasses must be robust to the
        presence of explicit zeros in the CSR representation.

        1. The `data` arrays are passed using pointers to be able to support an
        alternative representation of the CSR data structure for supporting
        fused sparse-dense datasets pairs with minimum overhead.

        See the explanations in `SparseDenseDatasetsPair.__init__`.

        2. An alternative signature would be:

            cdef float64_t dist_csr(
                self,
                const float64_t* x1_data,
                const int32_t[:] x1_indices,
                const float64_t* x2_data,
                const int32_t[:] x2_indices,
            ) except -1 nogil:

        Where callers would use slicing on the original CSR data and indices
        memoryviews:

            x1_start = X1_csr.indices_ptr[i]
            x1_end   = X1_csr.indices_ptr[i+1]
            x2_start = X2_csr.indices_ptr[j]
            x2_end   = X2_csr.indices_ptr[j+1]

            self.dist_csr(
                &x1_data[x1_start],
                x1_indices[x1_start:x1_end],
                &x2_data[x2_start],
                x2_indices[x2_start:x2_end],
            )

        Yet, slicing on memoryview slows down execution as it takes the GIL.
        See: https://github.com/scikit-learn/scikit-learn/issues/17299

        Hence, to avoid slicing the data and indices arrays of the sparse
        matrices containing respectively x1 and x2 (namely x{1,2}_{data,indices})
        are passed as well as their indices pointers (namely x{1,2}_{start,end}).

        3. For reference about the CSR format, see section 3.4 of
        Saad, Y. (2003), Iterative Methods for Sparse Linear Systems, SIAM.
        https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf
        """
        return -999

    cdef float64_t rdist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:
        """Distance between rows of CSR matrices x1 and x2.

        This can optionally be overridden in a subclass.

        The rank-preserving surrogate distance is any measure that yields the same
        rank as the distance, but is more efficient to compute. For example, the
        rank-preserving surrogate distance of the Euclidean metric is the
        squared-euclidean distance.

        Notes
        -----
        The implementation of this method in subclasses must be robust to the
        presence of explicit zeros in the CSR representation.

        More information about the motives for this method signature is given
        in the docstring of dist_csr.
        """
        return self.dist_csr(
            x1_data,
            x1_indices,
            x2_data,
            x2_indices,
            x1_start,
            x1_end,
            x2_start,
            x2_end,
            size,
        )

    cdef int pdist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const int32_t[:] x1_indptr,
        const intp_t size,
        float64_t[:, ::1] D,
    ) except -1 nogil:
        """Pairwise distances between rows in CSR matrix X.

        Note that this implementation is twice faster than cdist_csr(X, X)
        because it leverages the symmetry of the problem.
        """
        cdef:
            intp_t i1, i2
            intp_t n_x1 = x1_indptr.shape[0] - 1
            intp_t x1_start, x1_end, x2_start, x2_end

        for i1 in range(n_x1):
            x1_start = x1_indptr[i1]
            x1_end = x1_indptr[i1 + 1]
            for i2 in range(i1, n_x1):
                x2_start = x1_indptr[i2]
                x2_end = x1_indptr[i2 + 1]
                D[i1, i2] = D[i2, i1] = self.dist_csr(
                    x1_data,
                    x1_indices,
                    x1_data,
                    x1_indices,
                    x1_start,
                    x1_end,
                    x2_start,
                    x2_end,
                    size,
                )
        return 0

    cdef int cdist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const int32_t[:] x1_indptr,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t[:] x2_indptr,
        const intp_t size,
        float64_t[:, ::1] D,
    ) except -1 nogil:
        """Compute the cross-pairwise distances between arrays X and Y
        represented in the CSR format."""
        cdef:
            intp_t i1, i2
            intp_t n_x1 = x1_indptr.shape[0] - 1
            intp_t n_x2 = x2_indptr.shape[0] - 1
            intp_t x1_start, x1_end, x2_start, x2_end

        for i1 in range(n_x1):
            x1_start = x1_indptr[i1]
            x1_end = x1_indptr[i1 + 1]
            for i2 in range(n_x2):
                x2_start = x2_indptr[i2]
                x2_end = x2_indptr[i2 + 1]

                D[i1, i2] = self.dist_csr(
                    x1_data,
                    x1_indices,
                    x2_data,
                    x2_indices,
                    x1_start,
                    x1_end,
                    x2_start,
                    x2_end,
                    size,
                )
        return 0

    cdef float64_t _rdist_to_dist(self, float64_t rdist) except -1 nogil:
        """Convert the rank-preserving surrogate distance to the distance"""
        return rdist

    cdef float64_t _dist_to_rdist(self, float64_t dist) except -1 nogil:
        """Convert the distance to the rank-preserving surrogate distance"""
        return dist

    def rdist_to_dist(self, rdist):
        """Convert the rank-preserving surrogate distance to the distance.

        The surrogate distance is any measure that yields the same rank as the
        distance, but is more efficient to compute. For example, the
        rank-preserving surrogate distance of the Euclidean metric is the
        squared-euclidean distance.

        Parameters
        ----------
        rdist : double
            Surrogate distance.

        Returns
        -------
        double
            True distance.
        """
        return rdist

    def dist_to_rdist(self, dist):
        """Convert the true distance to the rank-preserving surrogate distance.

        The surrogate distance is any measure that yields the same rank as the
        distance, but is more efficient to compute. For example, the
        rank-preserving surrogate distance of the Euclidean metric is the
        squared-euclidean distance.

        Parameters
        ----------
        dist : double
            True distance.

        Returns
        -------
        double
            Surrogate distance.
        """
        return dist

    def _pairwise_dense_dense(self, X, Y):
        cdef const float64_t[:, ::1] Xarr
        cdef const float64_t[:, ::1] Yarr
        cdef float64_t[:, ::1] Darr

        Xarr = np.asarray(X, dtype=np.float64, order='C')
        self._validate_data(Xarr)
        if X is Y:
            Darr = np.empty((Xarr.shape[0], Xarr.shape[0]), dtype=np.float64, order='C')
            self.pdist(Xarr, Darr)
        else:
            Yarr = np.asarray(Y, dtype=np.float64, order='C')
            self._validate_data(Yarr)
            Darr = np.empty((Xarr.shape[0], Yarr.shape[0]), dtype=np.float64, order='C')
            self.cdist(Xarr, Yarr, Darr)
        return np.asarray(Darr)

    def _pairwise_sparse_sparse(self, X: csr_matrix , Y: csr_matrix):
        cdef:
            intp_t n_X, n_features
            const float64_t[:] X_data
            const int32_t[:] X_indices
            const int32_t[:] X_indptr

            intp_t n_Y
            const float64_t[:] Y_data
            const int32_t[:] Y_indices
            const int32_t[:] Y_indptr

            float64_t[:, ::1] Darr

        X_csr = X.tocsr()
        n_X, n_features = X_csr.shape
        X_data = np.asarray(X_csr.data, dtype=np.float64)
        X_indices = np.asarray(X_csr.indices, dtype=np.int32)
        X_indptr = np.asarray(X_csr.indptr, dtype=np.int32)
        if X is Y:
            Darr = np.empty((n_X, n_X), dtype=np.float64, order='C')
            self.pdist_csr(
                x1_data=&X_data[0],
                x1_indices=X_indices,
                x1_indptr=X_indptr,
                size=n_features,
                D=Darr,
            )
        else:
            Y_csr = Y.tocsr()
            n_Y, _ = Y_csr.shape
            Y_data = np.asarray(Y_csr.data, dtype=np.float64)
            Y_indices = np.asarray(Y_csr.indices, dtype=np.int32)
            Y_indptr = np.asarray(Y_csr.indptr, dtype=np.int32)

            Darr = np.empty((n_X, n_Y), dtype=np.float64, order='C')
            self.cdist_csr(
                x1_data=&X_data[0],
                x1_indices=X_indices,
                x1_indptr=X_indptr,
                x2_data=&Y_data[0],
                x2_indices=Y_indices,
                x2_indptr=Y_indptr,
                size=n_features,
                D=Darr,
            )
        return np.asarray(Darr)

    def _pairwise_sparse_dense(self, X: csr_matrix, Y):
        cdef:
            intp_t n_X = X.shape[0]
            intp_t n_features = X.shape[1]
            const float64_t[:] X_data = np.asarray(
                X.data, dtype=np.float64,
            )
            const int32_t[:] X_indices = np.asarray(
                X.indices, dtype=np.int32,
            )
            const int32_t[:] X_indptr = np.asarray(
                X.indptr, dtype=np.int32,
            )

            const float64_t[:, ::1] Y_data = np.asarray(
                Y, dtype=np.float64, order="C",
            )
            intp_t n_Y = Y_data.shape[0]
            const int32_t[:] Y_indices = (
                np.arange(n_features, dtype=np.int32)
            )

            float64_t[:, ::1] Darr = np.empty((n_X, n_Y), dtype=np.float64, order='C')

            intp_t i1, i2
            intp_t x1_start, x1_end
            float64_t * x2_data

        with nogil:
            # Use the exact same adaptation for CSR than in SparseDenseDatasetsPair
            # for supporting the sparse-dense case with minimal overhead.
            # Note: at this point this method is only a convenience method
            # used in the tests via the DistanceMetric.pairwise method.
            # Therefore, there is no need to attempt parallelization of those
            # nested for-loops.
            # Efficient parallel computation of pairwise distances can be
            # achieved via the PairwiseDistances class instead. The latter
            # internally calls into vector-wise distance computation from
            # the DistanceMetric subclass while benefiting from the generic
            # Cython/OpenMP parallelization template for the generic pairwise
            # distance + reduction computational pattern.
            for i1 in range(n_X):
                x1_start = X_indptr[i1]
                x1_end = X_indptr[i1 + 1]
                for i2 in range(n_Y):
                    x2_data = &Y_data[0, 0] + i2 * n_features

                    Darr[i1, i2] = self.dist_csr(
                        x1_data=&X_data[0],
                        x1_indices=X_indices,
                        x2_data=x2_data,
                        x2_indices=Y_indices,
                        x1_start=x1_start,
                        x1_end=x1_end,
                        x2_start=0,
                        x2_end=n_features,
                        size=n_features,
                    )

        return np.asarray(Darr)

    def _pairwise_dense_sparse(self, X, Y: csr_matrix):
        # We could have implemented this method using _pairwise_dense_sparse by
        # swapping argument and by transposing the results, but this would
        # have come with an extra copy to ensure C-contiguity of the result.
        cdef:
            intp_t n_X = X.shape[0]
            intp_t n_features = X.shape[1]

            const float64_t[:, ::1] X_data = np.asarray(
                X, dtype=np.float64, order="C",
            )
            const int32_t[:] X_indices = np.arange(
                n_features, dtype=np.int32,
            )

            intp_t n_Y = Y.shape[0]
            const float64_t[:] Y_data = np.asarray(
                Y.data, dtype=np.float64,
            )
            const int32_t[:] Y_indices = np.asarray(
                Y.indices, dtype=np.int32,
            )
            const int32_t[:] Y_indptr = np.asarray(
                Y.indptr, dtype=np.int32,
            )

            float64_t[:, ::1] Darr = np.empty((n_X, n_Y), dtype=np.float64, order='C')

            intp_t i1, i2
            float64_t * x1_data

            intp_t x2_start, x2_end

        with nogil:
            # Use the exact same adaptation for CSR than in SparseDenseDatasetsPair
            # for supporting the dense-sparse case with minimal overhead.
            # Note: at this point this method is only a convenience method
            # used in the tests via the DistanceMetric.pairwise method.
            # Therefore, there is no need to attempt parallelization of those
            # nested for-loops.
            # Efficient parallel computation of pairwise distances can be
            # achieved via the PairwiseDistances class instead. The latter
            # internally calls into vector-wise distance computation from
            # the DistanceMetric subclass while benefiting from the generic
            # Cython/OpenMP parallelization template for the generic pairwise
            # distance + reduction computational pattern.
            for i1 in range(n_X):
                x1_data = &X_data[0, 0] + i1 * n_features
                for i2 in range(n_Y):
                    x2_start = Y_indptr[i2]
                    x2_end = Y_indptr[i2 + 1]

                    Darr[i1, i2] = self.dist_csr(
                        x1_data=x1_data,
                        x1_indices=X_indices,
                        x2_data=&Y_data[0],
                        x2_indices=Y_indices,
                        x1_start=0,
                        x1_end=n_features,
                        x2_start=x2_start,
                        x2_end=x2_end,
                        size=n_features,
                    )

        return np.asarray(Darr)


    def pairwise(self, X, Y=None):
        """Compute the pairwise distances between X and Y

        This is a convenience routine for the sake of testing.  For many
        metrics, the utilities in scipy.spatial.distance.cdist and
        scipy.spatial.distance.pdist will be faster.

        Parameters
        ----------
        X : ndarray or CSR matrix of shape (n_samples_X, n_features)
            Input data.
        Y : ndarray or CSR matrix of shape (n_samples_Y, n_features)
            Input data.
            If not specified, then Y=X.

        Returns
        -------
        dist : ndarray of shape  (n_samples_X, n_samples_Y)
            The distance matrix of pairwise distances between points in X and Y.
        """
        X = check_array(X, accept_sparse=['csr'])

        if Y is None:
            Y = X
        else:
            Y = check_array(Y, accept_sparse=['csr'])

        X_is_sparse = issparse(X)
        Y_is_sparse = issparse(Y)

        if not X_is_sparse and not Y_is_sparse:
            return self._pairwise_dense_dense(X, Y)

        if X_is_sparse and Y_is_sparse:
            return self._pairwise_sparse_sparse(X, Y)

        if X_is_sparse and not Y_is_sparse:
            return self._pairwise_sparse_dense(X, Y)

        return self._pairwise_dense_sparse(X, Y)

#------------------------------------------------------------
# Euclidean Distance
#  d = sqrt(sum(x_i^2 - y_i^2))
cdef class EuclideanDistance64(DistanceMetric64):
    r"""Euclidean Distance metric

    .. math::
       D(x, y) = \sqrt{ \sum_i (x_i - y_i) ^ 2 }
    """
    def __init__(self):
        self.p = 2

    cdef inline float64_t dist(self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        return euclidean_dist64(x1, x2, size)

    cdef inline float64_t rdist(self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        return euclidean_rdist64(x1, x2, size)

    cdef inline float64_t _rdist_to_dist(self, float64_t rdist) except -1 nogil:
        return sqrt(rdist)

    cdef inline float64_t _dist_to_rdist(self, float64_t dist) except -1 nogil:
        return dist * dist

    def rdist_to_dist(self, rdist):
        return np.sqrt(rdist)

    def dist_to_rdist(self, dist):
        return dist ** 2

    cdef inline float64_t rdist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            float64_t d = 0.0
            float64_t unsquared = 0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            if ix1 == ix2:
                unsquared = x1_data[i1] - x2_data[i2]
                d = d + (unsquared * unsquared)
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                unsquared = x1_data[i1]
                d = d + (unsquared * unsquared)
                i1 = i1 + 1
            else:
                unsquared = x2_data[i2]
                d = d + (unsquared * unsquared)
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                unsquared = x2_data[i2]
                d = d + (unsquared * unsquared)
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                unsquared = x1_data[i1]
                d = d + (unsquared * unsquared)
                i1 = i1 + 1

        return d

    cdef inline float64_t dist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:
        return sqrt(
            self.rdist_csr(
            x1_data,
            x1_indices,
            x2_data,
            x2_indices,
            x1_start,
            x1_end,
            x2_start,
            x2_end,
            size,
        ))

#------------------------------------------------------------
# SEuclidean Distance
#  d = sqrt(sum((x_i - y_i2)^2 / v_i))
cdef class SEuclideanDistance64(DistanceMetric64):
    r"""Standardized Euclidean Distance metric

    .. math::
       D(x, y) = \sqrt{ \sum_i \frac{ (x_i - y_i) ^ 2}{V_i} }
    """
    def __init__(self, V):
        self.vec = np.asarray(V, dtype=np.float64)
        self.size = self.vec.shape[0]
        self.p = 2

    def _validate_data(self, X):
        if X.shape[1] != self.size:
            raise ValueError('SEuclidean dist: size of V does not match')

    cdef inline float64_t rdist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef float64_t tmp, d=0
        cdef intp_t j
        for j in range(size):
            tmp = x1[j] - x2[j]
            d += (tmp * tmp / self.vec[j])
        return d

    cdef inline float64_t dist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        return sqrt(self.rdist(x1, x2, size))

    cdef inline float64_t _rdist_to_dist(self, float64_t rdist) except -1 nogil:
        return sqrt(rdist)

    cdef inline float64_t _dist_to_rdist(self, float64_t dist) except -1 nogil:
        return dist * dist

    def rdist_to_dist(self, rdist):
        return np.sqrt(rdist)

    def dist_to_rdist(self, dist):
        return dist ** 2

    cdef inline float64_t rdist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            float64_t d = 0.0
            float64_t unsquared = 0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            if ix1 == ix2:
                unsquared = x1_data[i1] - x2_data[i2]
                d = d + (unsquared * unsquared) / self.vec[ix1]
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                unsquared = x1_data[i1]
                d = d + (unsquared * unsquared) / self.vec[ix1]
                i1 = i1 + 1
            else:
                unsquared = x2_data[i2]
                d = d + (unsquared * unsquared) / self.vec[ix2]
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                ix2 = x2_indices[i2]
                unsquared = x2_data[i2]
                d = d + (unsquared * unsquared) / self.vec[ix2]
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                ix1 = x1_indices[i1]
                unsquared = x1_data[i1]
                d = d + (unsquared * unsquared) / self.vec[ix1]
                i1 = i1 + 1
        return d

    cdef inline float64_t dist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:
        return sqrt(
            self.rdist_csr(
            x1_data,
            x1_indices,
            x2_data,
            x2_indices,
            x1_start,
            x1_end,
            x2_start,
            x2_end,
            size,
        ))

#------------------------------------------------------------
# Manhattan Distance
#  d = sum(abs(x_i - y_i))
cdef class ManhattanDistance64(DistanceMetric64):
    r"""Manhattan/City-block Distance metric

    .. math::
       D(x, y) = \sum_i |x_i - y_i|
    """
    def __init__(self):
        self.p = 1

    cdef inline float64_t dist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef float64_t d = 0
        cdef intp_t j
        for j in range(size):
            d += fabs(x1[j] - x2[j])
        return d

    cdef inline float64_t dist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            float64_t d = 0.0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            if ix1 == ix2:
                d = d + fabs(x1_data[i1] - x2_data[i2])
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                d = d + fabs(x1_data[i1])
                i1 = i1 + 1
            else:
                d = d + fabs(x2_data[i2])
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                d = d + fabs(x2_data[i2])
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                d = d + fabs(x1_data[i1])
                i1 = i1 + 1

        return d


#------------------------------------------------------------
# Chebyshev Distance
#  d = max_i(abs(x_i - y_i))
cdef class ChebyshevDistance64(DistanceMetric64):
    """Chebyshev/Infinity Distance

    .. math::
       D(x, y) = max_i (|x_i - y_i|)

    Examples
    --------
    >>> from sklearn.metrics.dist_metrics import DistanceMetric
    >>> dist = DistanceMetric.get_metric('chebyshev')
    >>> X = [[0, 1, 2],
    ...      [3, 4, 5]]
    >>> Y = [[-1, 0, 1],
    ...      [3, 4, 5]]
    >>> dist.pairwise(X, Y)
    array([[1.732..., 5.196...],
           [6.928..., 0....   ]])
    """
    def __init__(self):
        self.p = INF64

    cdef inline float64_t dist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef float64_t d = 0
        cdef intp_t j
        for j in range(size):
            d = fmax(d, fabs(x1[j] - x2[j]))
        return d


    cdef inline float64_t dist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            float64_t d = 0.0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            if ix1 == ix2:
                d = fmax(d, fabs(x1_data[i1] - x2_data[i2]))
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                d = fmax(d, fabs(x1_data[i1]))
                i1 = i1 + 1
            else:
                d = fmax(d, fabs(x2_data[i2]))
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                d = fmax(d, fabs(x2_data[i2]))
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                d = fmax(d, fabs(x1_data[i1]))
                i1 = i1 + 1

        return d


#------------------------------------------------------------
# Minkowski Distance
cdef class MinkowskiDistance64(DistanceMetric64):
    r"""Minkowski Distance

    .. math::
        D(x, y) = {||u-v||}_p

    when w is None.

    Here is the more general expanded expression for the weighted case:

    .. math::
        D(x, y) = [\sum_i w_i *|x_i - y_i|^p] ^ (1/p)

    Parameters
    ----------
    p : float
        The order of the p-norm of the difference (see above).

        .. versionchanged:: 1.4.0
            Minkowski distance allows `p` to be `0<p<1`.


    w : (N,) array-like (optional)
        The weight vector.

    Minkowski Distance requires p > 0 and finite.
    When :math:`p \in (0,1)`, it isn't a true metric but is permissible when
    the triangular inequality isn't necessary.
    For p = infinity, use ChebyshevDistance.
    Note that for p=1, ManhattanDistance is more efficient, and for
    p=2, EuclideanDistance is more efficient.

    """
    def __init__(self, p, w=None):
        if p <= 0:
            raise ValueError("p must be greater than 0")
        elif np.isinf(p):
            raise ValueError("MinkowskiDistance requires finite p. "
                             "For p=inf, use ChebyshevDistance.")

        self.p = p
        if w is not None:
            w_array = check_array(
                w, ensure_2d=False, dtype=np.float64, input_name="w"
            )
            if (w_array < 0).any():
                raise ValueError("w cannot contain negative weights")
            self.vec = w_array
            self.size = self.vec.shape[0]
        else:
            self.vec = np.asarray([], dtype=np.float64)
            self.size = 0

    def _validate_data(self, X):
        if self.size > 0 and X.shape[1] != self.size:
            raise ValueError("MinkowskiDistance: the size of w must match "
                             f"the number of features ({X.shape[1]}). "
                             f"Currently len(w)={self.size}.")

    cdef inline float64_t rdist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef float64_t d=0
        cdef intp_t j
        cdef bint has_w = self.size > 0
        if has_w:
            for j in range(size):
                d += (self.vec[j] * pow(fabs(x1[j] - x2[j]), self.p))
        else:
            for j in range(size):
                d += (pow(fabs(x1[j] - x2[j]), self.p))
        return d

    cdef inline float64_t dist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        return pow(self.rdist(x1, x2, size), 1. / self.p)

    cdef inline float64_t _rdist_to_dist(self, float64_t rdist) except -1 nogil:
        return pow(rdist, 1. / self.p)

    cdef inline float64_t _dist_to_rdist(self, float64_t dist) except -1 nogil:
        return pow(dist, self.p)

    def rdist_to_dist(self, rdist):
        return rdist ** (1. / self.p)

    def dist_to_rdist(self, dist):
        return dist ** self.p

    cdef inline float64_t rdist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            float64_t d = 0.0
            bint has_w = self.size > 0

        if has_w:
            while i1 < x1_end and i2 < x2_end:
                ix1 = x1_indices[i1]
                ix2 = x2_indices[i2]

                if ix1 == ix2:
                    d = d + (self.vec[ix1] * pow(fabs(
                        x1_data[i1] - x2_data[i2]
                    ), self.p))
                    i1 = i1 + 1
                    i2 = i2 + 1
                elif ix1 < ix2:
                    d = d + (self.vec[ix1] * pow(fabs(x1_data[i1]), self.p))
                    i1 = i1 + 1
                else:
                    d = d + (self.vec[ix2] * pow(fabs(x2_data[i2]), self.p))
                    i2 = i2 + 1

            if i1 == x1_end:
                while i2 < x2_end:
                    ix2 = x2_indices[i2]
                    d = d + (self.vec[ix2] * pow(fabs(x2_data[i2]), self.p))
                    i2 = i2 + 1
            else:
                while i1 < x1_end:
                    ix1 = x1_indices[i1]
                    d = d + (self.vec[ix1] * pow(fabs(x1_data[i1]), self.p))
                    i1 = i1 + 1

            return d
        else:
            while i1 < x1_end and i2 < x2_end:
                ix1 = x1_indices[i1]
                ix2 = x2_indices[i2]

                if ix1 == ix2:
                    d = d + (pow(fabs(
                        x1_data[i1] - x2_data[i2]
                    ), self.p))
                    i1 = i1 + 1
                    i2 = i2 + 1
                elif ix1 < ix2:
                    d = d + (pow(fabs(x1_data[i1]), self.p))
                    i1 = i1 + 1
                else:
                    d = d + (pow(fabs(x2_data[i2]), self.p))
                    i2 = i2 + 1

            if i1 == x1_end:
                while i2 < x2_end:
                    d = d + (pow(fabs(x2_data[i2]), self.p))
                    i2 = i2 + 1
            else:
                while i1 < x1_end:
                    d = d + (pow(fabs(x1_data[i1]), self.p))
                    i1 = i1 + 1

            return d

    cdef inline float64_t dist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:
        return pow(
            self.rdist_csr(
                x1_data,
                x1_indices,
                x2_data,
                x2_indices,
                x1_start,
                x1_end,
                x2_start,
                x2_end,
                size,
            ),
            1 / self.p
        )

#------------------------------------------------------------
# Mahalanobis Distance
#  d = sqrt( (x - y)^T V^-1 (x - y) )
cdef class MahalanobisDistance64(DistanceMetric64):
    """Mahalanobis Distance

    .. math::
       D(x, y) = \sqrt{ (x - y)^T V^{-1} (x - y) }

    Parameters
    ----------
    V : array-like
        Symmetric positive-definite covariance matrix.
        The inverse of this matrix will be explicitly computed.
    VI : array-like
        optionally specify the inverse directly.  If VI is passed,
        then V is not referenced.
    """
    cdef float64_t[::1] buffer

    def __init__(self, V=None, VI=None):
        if VI is None:
            if V is None:
                raise ValueError("Must provide either V or VI "
                                 "for Mahalanobis distance")
            VI = np.linalg.inv(V)
        if VI.ndim != 2 or VI.shape[0] != VI.shape[1]:
            raise ValueError("V/VI must be square")

        self.mat = np.asarray(VI, dtype=np.float64, order='C')

        self.size = self.mat.shape[0]

        # We need to create a buffer to store the vectors' coordinates' differences
        self.buffer = np.zeros(self.size, dtype=np.float64)

    def __setstate__(self, state):
        super().__setstate__(state)
        self.size = self.mat.shape[0]
        self.buffer = np.zeros(self.size, dtype=np.float64)

    def _validate_data(self, X):
        if X.shape[1] != self.size:
            raise ValueError('Mahalanobis dist: size of V does not match')

    cdef inline float64_t rdist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef float64_t tmp, d = 0
        cdef intp_t i, j

        # compute (x1 - x2).T * VI * (x1 - x2)
        for i in range(size):
            self.buffer[i] = x1[i] - x2[i]

        for i in range(size):
            tmp = 0
            for j in range(size):
                tmp += self.mat[i, j] * self.buffer[j]
            d += tmp * self.buffer[i]
        return d

    cdef inline float64_t dist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        return sqrt(self.rdist(x1, x2, size))

    cdef inline float64_t _rdist_to_dist(self, float64_t rdist) except -1 nogil:
        return sqrt(rdist)

    cdef inline float64_t _dist_to_rdist(self, float64_t dist) except -1 nogil:
        return dist * dist

    def rdist_to_dist(self, rdist):
        return np.sqrt(rdist)

    def dist_to_rdist(self, dist):
        return dist ** 2

    cdef inline float64_t rdist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            float64_t tmp, d = 0.0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            if ix1 == ix2:
                self.buffer[ix1] = x1_data[i1] - x2_data[i2]
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                self.buffer[ix1] = x1_data[i1]
                i1 = i1 + 1
            else:
                self.buffer[ix2] = - x2_data[i2]
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                ix2 = x2_indices[i2]
                self.buffer[ix2] = - x2_data[i2]
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                ix1 = x1_indices[i1]
                self.buffer[ix1] = x1_data[i1]
                i1 = i1 + 1

        for i in range(size):
            tmp = 0
            for j in range(size):
                tmp += self.mat[i, j] * self.buffer[j]
            d += tmp * self.buffer[i]

        return d

    cdef inline float64_t dist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:
        return sqrt(
            self.rdist_csr(
            x1_data,
            x1_indices,
            x2_data,
            x2_indices,
            x1_start,
            x1_end,
            x2_start,
            x2_end,
            size,
        ))

#------------------------------------------------------------
# Hamming Distance
#  d = N_unequal(x, y) / N_tot
cdef class HammingDistance64(DistanceMetric64):
    r"""Hamming Distance

    Hamming distance is meant for discrete-valued vectors, though it is
    a valid metric for real-valued vectors.

    .. math::
       D(x, y) = \frac{1}{N} \sum_i \delta_{x_i, y_i}
    """
    cdef inline float64_t dist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef int n_unequal = 0
        cdef intp_t j
        for j in range(size):
            if x1[j] != x2[j]:
                n_unequal += 1
        return float(n_unequal) / size


    cdef inline float64_t dist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            float64_t d = 0.0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            if ix1 == ix2:
                d += (x1_data[i1] != x2_data[i2])
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                d += (x1_data[i1] != 0)
                i1 = i1 + 1
            else:
                d += (x2_data[i2] != 0)
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                d += (x2_data[i2] != 0)
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                d += (x1_data[i1] != 0)
                i1 = i1 + 1

        d /= size

        return d


#------------------------------------------------------------
# Canberra Distance
#  D(x, y) = sum[ abs(x_i - y_i) / (abs(x_i) + abs(y_i)) ]
cdef class CanberraDistance64(DistanceMetric64):
    r"""Canberra Distance

    Canberra distance is meant for discrete-valued vectors, though it is
    a valid metric for real-valued vectors.

    .. math::
       D(x, y) = \sum_i \frac{|x_i - y_i|}{|x_i| + |y_i|}
    """
    cdef inline float64_t dist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef float64_t denom, d = 0
        cdef intp_t j
        for j in range(size):
            denom = fabs(x1[j]) + fabs(x2[j])
            if denom > 0:
                d += fabs(x1[j] - x2[j]) / denom
        return d

    cdef inline float64_t dist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            float64_t d = 0.0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            if ix1 == ix2:
                d += (
                        fabs(x1_data[i1] - x2_data[i2]) /
                        (fabs(x1_data[i1]) + fabs(x2_data[i2]))
                )
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                d += 1.
                i1 = i1 + 1
            else:
                d += 1.
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                d += 1.
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                d += 1.
                i1 = i1 + 1

        return d

#------------------------------------------------------------
# Bray-Curtis Distance
#  D(x, y) = sum[abs(x_i - y_i)] / sum[abs(x_i) + abs(y_i)]
cdef class BrayCurtisDistance64(DistanceMetric64):
    r"""Bray-Curtis Distance

    Bray-Curtis distance is meant for discrete-valued vectors, though it is
    a valid metric for real-valued vectors.

    .. math::
       D(x, y) = \frac{\sum_i |x_i - y_i|}{\sum_i(|x_i| + |y_i|)}
    """
    cdef inline float64_t dist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef float64_t num = 0, denom = 0
        cdef intp_t j
        for j in range(size):
            num += fabs(x1[j] - x2[j])
            denom += fabs(x1[j]) + fabs(x2[j])
        if denom > 0:
            return num / denom
        else:
            return 0.0

    cdef inline float64_t dist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            float64_t num = 0.0
            float64_t denom = 0.0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            if ix1 == ix2:
                num += fabs(x1_data[i1] - x2_data[i2])
                denom += fabs(x1_data[i1]) + fabs(x2_data[i2])
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                num += fabs(x1_data[i1])
                denom += fabs(x1_data[i1])
                i1 = i1 + 1
            else:
                num += fabs(x2_data[i2])
                denom += fabs(x2_data[i2])
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                num += fabs(x1_data[i1])
                denom += fabs(x1_data[i1])
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                num += fabs(x2_data[i2])
                denom += fabs(x2_data[i2])
                i1 = i1 + 1

        return num / denom

#------------------------------------------------------------
# Jaccard Distance (boolean)
#  D(x, y) = N_unequal(x, y) / N_nonzero(x, y)
cdef class JaccardDistance64(DistanceMetric64):
    r"""Jaccard Distance

    Jaccard Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = (N_TF + N_FT) / (N_TT + N_TF + N_FT)
    """
    cdef inline float64_t dist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef int tf1, tf2, n_eq = 0, nnz = 0
        cdef intp_t j
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            nnz += (tf1 or tf2)
            n_eq += (tf1 and tf2)
        # Based on https://github.com/scipy/scipy/pull/7373
        # When comparing two all-zero vectors, scipy>=1.2.0 jaccard metric
        # was changed to return 0, instead of nan.
        if nnz == 0:
            return 0
        return (nnz - n_eq) * 1.0 / nnz

    cdef inline float64_t dist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            intp_t tf1, tf2, n_tt = 0, nnz = 0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            tf1 = x1_data[i1] != 0
            tf2 = x2_data[i2] != 0

            if ix1 == ix2:
                nnz += (tf1 or tf2)
                n_tt += (tf1 and tf2)
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                nnz += tf1
                i1 = i1 + 1
            else:
                nnz += tf2
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                tf2 = x2_data[i2] != 0
                nnz += tf2
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                tf1 = x1_data[i1] != 0
                nnz += tf1
                i1 = i1 + 1

        # Based on https://github.com/scipy/scipy/pull/7373
        # When comparing two all-zero vectors, scipy>=1.2.0 jaccard metric
        # was changed to return 0, instead of nan.
        if nnz == 0:
            return 0
        return (nnz - n_tt) * 1.0 / nnz

#------------------------------------------------------------
# Matching Distance (boolean)
#  D(x, y) = n_neq / n
cdef class MatchingDistance64(DistanceMetric64):
    r"""Matching Distance

    Matching Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = (N_TF + N_FT) / N
    """
    cdef inline float64_t dist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef int tf1, tf2, n_neq = 0
        cdef intp_t j
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            n_neq += (tf1 != tf2)
        return n_neq * 1. / size

    cdef inline float64_t dist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            intp_t tf1, tf2, n_neq = 0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            if ix1 == ix2:
                tf1 = x1_data[i1] != 0
                tf2 = x2_data[i2] != 0
                n_neq += (tf1 != tf2)
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                n_neq += (x1_data[i1] != 0)
                i1 = i1 + 1
            else:
                n_neq += (x2_data[i2] != 0)
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                n_neq += (x2_data[i2] != 0)
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                n_neq += (x1_data[i1] != 0)
                i1 = i1 + 1

        return n_neq * 1.0 / size

#------------------------------------------------------------
# Dice Distance (boolean)
#  D(x, y) = n_neq / (2 * ntt + n_neq)
cdef class DiceDistance64(DistanceMetric64):
    r"""Dice Distance

    Dice Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = (N_TF + N_FT) / (2 * N_TT + N_TF + N_FT)

    """
    cdef inline float64_t dist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef int tf1, tf2, n_neq = 0, n_tt = 0
        cdef intp_t j
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            n_tt += (tf1 and tf2)
            n_neq += (tf1 != tf2)
        return n_neq / (2.0 * n_tt + n_neq)

    cdef inline float64_t dist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            intp_t tf1, tf2, n_tt = 0, n_neq = 0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            tf1 = x1_data[i1] != 0
            tf2 = x2_data[i2] != 0

            if ix1 == ix2:
                n_tt += (tf1 and tf2)
                n_neq += (tf1 != tf2)
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                n_neq += tf1
                i1 = i1 + 1
            else:
                n_neq += tf2
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                tf2 = x2_data[i2] != 0
                n_neq += tf2
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                tf1 = x1_data[i1] != 0
                n_neq += tf1
                i1 = i1 + 1

        return n_neq / (2.0 * n_tt + n_neq)


#------------------------------------------------------------
# Kulsinski Distance (boolean)
#  D(x, y) = (ntf + nft - ntt + n) / (n_neq + n)
cdef class KulsinskiDistance64(DistanceMetric64):
    r"""Kulsinski Distance

    Kulsinski Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = 1 - N_TT / (N + N_TF + N_FT)

    """
    cdef inline float64_t dist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef int tf1, tf2, n_tt = 0, n_neq = 0
        cdef intp_t j
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            n_neq += (tf1 != tf2)
            n_tt += (tf1 and tf2)
        return (n_neq - n_tt + size) * 1.0 / (n_neq + size)

    cdef inline float64_t dist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            intp_t tf1, tf2, n_tt = 0, n_neq = 0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            tf1 = x1_data[i1] != 0
            tf2 = x2_data[i2] != 0

            if ix1 == ix2:
                n_tt += (tf1 and tf2)
                n_neq += (tf1 != tf2)
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                n_neq += tf1
                i1 = i1 + 1
            else:
                n_neq += tf2
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                tf2 = x2_data[i2] != 0
                n_neq += tf2
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                tf1 = x1_data[i1] != 0
                n_neq += tf1
                i1 = i1 + 1

        return (n_neq - n_tt + size) * 1.0 / (n_neq + size)

#------------------------------------------------------------
# Rogers-Tanimoto Distance (boolean)
#  D(x, y) = 2 * n_neq / (n + n_neq)
cdef class RogersTanimotoDistance64(DistanceMetric64):
    r"""Rogers-Tanimoto Distance

    Rogers-Tanimoto Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = 2 (N_TF + N_FT) / (N + N_TF + N_FT)
    """
    cdef inline float64_t dist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef int tf1, tf2, n_neq = 0
        cdef intp_t j
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            n_neq += (tf1 != tf2)
        return (2.0 * n_neq) / (size + n_neq)

    cdef inline float64_t dist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            intp_t tf1, tf2, n_neq = 0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            tf1 = x1_data[i1] != 0
            tf2 = x2_data[i2] != 0

            if ix1 == ix2:
                n_neq += (tf1 != tf2)
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                n_neq += tf1
                i1 = i1 + 1
            else:
                n_neq += tf2
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                tf2 = x2_data[i2] != 0
                n_neq += tf2
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                tf1 = x1_data[i1] != 0
                n_neq += tf1
                i1 = i1 + 1

        return (2.0 * n_neq) / (size + n_neq)

#------------------------------------------------------------
# Russell-Rao Distance (boolean)
#  D(x, y) = (n - ntt) / n
cdef class RussellRaoDistance64(DistanceMetric64):
    r"""Russell-Rao Distance

    Russell-Rao Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = (N - N_TT) / N
    """
    cdef inline float64_t dist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef int tf1, tf2, n_tt = 0
        cdef intp_t j
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            n_tt += (tf1 and tf2)
        return (size - n_tt) * 1. / size

    cdef inline float64_t dist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            intp_t tf1, tf2, n_tt = 0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            tf1 = x1_data[i1] != 0
            tf2 = x2_data[i2] != 0

            if ix1 == ix2:
                n_tt += (tf1 and tf2)
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                i1 = i1 + 1
            else:
                i2 = i2 + 1

        # We don't need to go through all the longest
        # vector because tf1 or tf2 will be false
        # and thus n_tt won't be increased.

        return (size - n_tt) * 1. / size



#------------------------------------------------------------
# Sokal-Michener Distance (boolean)
#  D(x, y) = 2 * n_neq / (n + n_neq)
cdef class SokalMichenerDistance64(DistanceMetric64):
    r"""Sokal-Michener Distance

    Sokal-Michener Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = 2 (N_TF + N_FT) / (N + N_TF + N_FT)
    """
    cdef inline float64_t dist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef int tf1, tf2, n_neq = 0
        cdef intp_t j
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            n_neq += (tf1 != tf2)
        return (2.0 * n_neq) / (size + n_neq)

    cdef inline float64_t dist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            intp_t tf1, tf2, n_neq = 0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            tf1 = x1_data[i1] != 0
            tf2 = x2_data[i2] != 0

            if ix1 == ix2:
                n_neq += (tf1 != tf2)
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                n_neq += tf1
                i1 = i1 + 1
            else:
                n_neq += tf2
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                tf2 = x2_data[i2] != 0
                n_neq += tf2
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                tf1 = x1_data[i1] != 0
                n_neq += tf1
                i1 = i1 + 1

        return (2.0 * n_neq) / (size + n_neq)

#------------------------------------------------------------
# Sokal-Sneath Distance (boolean)
#  D(x, y) = n_neq / (0.5 * n_tt + n_neq)
cdef class SokalSneathDistance64(DistanceMetric64):
    r"""Sokal-Sneath Distance

    Sokal-Sneath Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = (N_TF + N_FT) / (N_TT / 2 + N_FT + N_TF)
    """
    cdef inline float64_t dist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef int tf1, tf2, n_tt = 0, n_neq = 0
        cdef intp_t j
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            n_neq += (tf1 != tf2)
            n_tt += (tf1 and tf2)
        return n_neq / (0.5 * n_tt + n_neq)

    cdef inline float64_t dist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            intp_t tf1, tf2, n_tt = 0, n_neq = 0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            tf1 = x1_data[i1] != 0
            tf2 = x2_data[i2] != 0

            if ix1 == ix2:
                n_tt += (tf1 and tf2)
                n_neq += (tf1 != tf2)
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                n_neq += tf1
                i1 = i1 + 1
            else:
                n_neq += tf2
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                tf2 = x2_data[i2] != 0
                n_neq += tf2
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                tf1 = x1_data[i1] != 0
                n_neq += tf1
                i1 = i1 + 1

        return n_neq / (0.5 * n_tt + n_neq)


#------------------------------------------------------------
# Haversine Distance (2 dimensional)
#  D(x, y) = 2 arcsin{sqrt[sin^2 ((x1 - y1) / 2)
#                          + cos(x1) cos(y1) sin^2 ((x2 - y2) / 2)]}
cdef class HaversineDistance64(DistanceMetric64):
    """Haversine (Spherical) Distance

    The Haversine distance is the angular distance between two points on
    the surface of a sphere.  The first distance of each point is assumed
    to be the latitude, the second is the longitude, given in radians.
    The dimension of the points must be 2:

    D(x, y) = 2 arcsin[sqrt{sin^2((x1 - y1) / 2) + cos(x1)cos(y1)sin^2((x2 - y2) / 2)}]

    """

    def _validate_data(self, X):
        if X.shape[1] != 2:
            raise ValueError("Haversine distance only valid "
                             "in 2 dimensions")

    cdef inline float64_t rdist(self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef float64_t sin_0 = sin(0.5 * ((x1[0]) - (x2[0])))
        cdef float64_t sin_1 = sin(0.5 * ((x1[1]) - (x2[1])))
        return (sin_0 * sin_0 + cos(x1[0]) * cos(x2[0]) * sin_1 * sin_1)

    cdef inline float64_t dist(self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        return 2 * asin(sqrt(self.rdist(x1, x2, size)))

    cdef inline float64_t _rdist_to_dist(self, float64_t rdist) except -1 nogil:
        return 2 * asin(sqrt(rdist))

    cdef inline float64_t _dist_to_rdist(self, float64_t dist) except -1 nogil:
        cdef float64_t tmp = sin(0.5 *  dist)
        return tmp * tmp

    def rdist_to_dist(self, rdist):
        return 2 * np.arcsin(np.sqrt(rdist))

    def dist_to_rdist(self, dist):
        tmp = np.sin(0.5 * dist)
        return tmp * tmp

    cdef inline float64_t dist_csr(
         self,
         const float64_t* x1_data,
         const int32_t[:] x1_indices,
         const float64_t* x2_data,
         const int32_t[:] x2_indices,
         const int32_t x1_start,
         const int32_t x1_end,
         const int32_t x2_start,
         const int32_t x2_end,
         const intp_t size,
    ) except -1 nogil:
        return 2 * asin(sqrt(self.rdist_csr(
            x1_data,
            x1_indices,
            x2_data,
            x2_indices,
            x1_start,
            x1_end,
            x2_start,
            x2_end,
            size,
        )))

    cdef inline float64_t rdist_csr(
        self,
        const float64_t* x1_data,
        const int32_t[:] x1_indices,
        const float64_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            float64_t x1_0 = 0
            float64_t x1_1 = 0
            float64_t x2_0 = 0
            float64_t x2_1 = 0
            float64_t sin_0
            float64_t sin_1

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            # Find the components in the 2D vectors to work with
            x1_component = ix1 if (x1_start == 0) else ix1 % x1_start
            x2_component = ix2 if (x2_start == 0) else ix2 % x2_start

            if x1_component == 0:
                x1_0 = x1_data[i1]
            else:
                x1_1 = x1_data[i1]

            if x2_component == 0:
                x2_0 = x2_data[i2]
            else:
                x2_1 = x2_data[i2]

            i1 = i1 + 1
            i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                ix2 = x2_indices[i2]
                x2_component = ix2 if (x2_start == 0) else ix2 % x2_start
                if x2_component == 0:
                    x2_0 = x2_data[i2]
                else:
                    x2_1 = x2_data[i2]
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                ix1 = x1_indices[i1]
                x1_component = ix1 if (x1_start == 0) else ix1 % x1_start
                if x1_component == 0:
                    x1_0 = x1_data[i1]
                else:
                    x1_1 = x1_data[i1]
                i1 = i1 + 1

        sin_0 = sin(0.5 * (x1_0 - x2_0))
        sin_1 = sin(0.5 * (x1_1 - x2_1))

        return (sin_0 * sin_0 + cos(x1_0) * cos(x2_0) * sin_1 * sin_1)

#------------------------------------------------------------
# User-defined distance
#
cdef class PyFuncDistance64(DistanceMetric64):
    """PyFunc Distance

    A user-defined distance

    Parameters
    ----------
    func : function
        func should take two numpy arrays as input, and return a distance.
    """
    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs

    # in cython < 0.26, GIL was required to be acquired during definition of
    # the function and inside the body of the function. This behaviour is not
    # allowed in cython >= 0.26 since it is a redundant GIL acquisition. The
    # only way to be back compatible is to inherit `dist` from the base class
    # without GIL and called an inline `_dist` which acquire GIL.
    cdef inline float64_t dist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 nogil:
        return self._dist(x1, x2, size)

    cdef inline float64_t _dist(
        self,
        const float64_t* x1,
        const float64_t* x2,
        intp_t size,
    ) except -1 with gil:
        cdef:
            object x1arr = _buffer_to_ndarray64(x1, size)
            object x2arr = _buffer_to_ndarray64(x2, size)
        d = self.func(x1arr, x2arr, **self.kwargs)
        try:
            # Cython generates code here that results in a TypeError
            # if d is the wrong type.
            return d
        except TypeError:
            raise TypeError("Custom distance function must accept two "
                            "vectors and return a float.")

######################################################################
# metric mappings
#  These map from metric id strings to class names
METRIC_MAPPING32 = {
    'euclidean': EuclideanDistance32,
    'l2': EuclideanDistance32,
    'minkowski': MinkowskiDistance32,
    'p': MinkowskiDistance32,
    'manhattan': ManhattanDistance32,
    'cityblock': ManhattanDistance32,
    'l1': ManhattanDistance32,
    'chebyshev': ChebyshevDistance32,
    'infinity': ChebyshevDistance32,
    'seuclidean': SEuclideanDistance32,
    'mahalanobis': MahalanobisDistance32,
    'hamming': HammingDistance32,
    'canberra': CanberraDistance32,
    'braycurtis': BrayCurtisDistance32,
    'matching': MatchingDistance32,
    'jaccard': JaccardDistance32,
    'dice': DiceDistance32,
    'kulsinski': KulsinskiDistance32,
    'rogerstanimoto': RogersTanimotoDistance32,
    'russellrao': RussellRaoDistance32,
    'sokalmichener': SokalMichenerDistance32,
    'sokalsneath': SokalSneathDistance32,
    'haversine': HaversineDistance32,
    'pyfunc': PyFuncDistance32,
}

cdef inline object _buffer_to_ndarray32(const float32_t* x, intp_t n):
    # Wrap a memory buffer with an ndarray. Warning: this is not robust.
    # In particular, if x is deallocated before the returned array goes
    # out of scope, this could cause memory errors.  Since there is not
    # a possibility of this for our use-case, this should be safe.

    # Note: this Segfaults unless np.import_array() is called above
    # TODO: remove the explicit cast to cnp.intp_t* when cython min version >= 3.0
    return cnp.PyArray_SimpleNewFromData(1, <cnp.intp_t*>&n, cnp.NPY_FLOAT64, <void*>x)


cdef float32_t INF32 = np.inf


######################################################################
# Distance Metric Classes
cdef class DistanceMetric32(DistanceMetric):
    """DistanceMetric class

    This class provides a uniform interface to fast distance metric
    functions.  The various metrics can be accessed via the :meth:`get_metric`
    class method and the metric string identifier (see below).

    Examples
    --------
    >>> from sklearn.metrics import DistanceMetric
    >>> dist = DistanceMetric.get_metric('euclidean')
    >>> X = [[0, 1, 2],
             [3, 4, 5]]
    >>> dist.pairwise(X)
    array([[ 0.        ,  5.19615242],
           [ 5.19615242,  0.        ]])

    Available Metrics

    The following lists the string metric identifiers and the associated
    distance metric classes:

    **Metrics intended for real-valued vector spaces:**

    ==============  ====================  ========  ===============================
    identifier      class name            args      distance function
    --------------  --------------------  --------  -------------------------------
    "euclidean"     EuclideanDistance     -         ``sqrt(sum((x - y)^2))``
    "manhattan"     ManhattanDistance     -         ``sum(|x - y|)``
    "chebyshev"     ChebyshevDistance     -         ``max(|x - y|)``
    "minkowski"     MinkowskiDistance     p, w      ``sum(w * |x - y|^p)^(1/p)``
    "seuclidean"    SEuclideanDistance    V         ``sqrt(sum((x - y)^2 / V))``
    "mahalanobis"   MahalanobisDistance   V or VI   ``sqrt((x - y)' V^-1 (x - y))``
    ==============  ====================  ========  ===============================

    **Metrics intended for two-dimensional vector spaces:**  Note that the haversine
    distance metric requires data in the form of [latitude, longitude] and both
    inputs and outputs are in units of radians.

    ============  ==================  ===============================================================
    identifier    class name          distance function
    ------------  ------------------  ---------------------------------------------------------------
    "haversine"   HaversineDistance   ``2 arcsin(sqrt(sin^2(0.5*dx) + cos(x1)cos(x2)sin^2(0.5*dy)))``
    ============  ==================  ===============================================================


    **Metrics intended for integer-valued vector spaces:**  Though intended
    for integer-valued vectors, these are also valid metrics in the case of
    real-valued vectors.

    =============  ====================  ========================================
    identifier     class name            distance function
    -------------  --------------------  ----------------------------------------
    "hamming"      HammingDistance       ``N_unequal(x, y) / N_tot``
    "canberra"     CanberraDistance      ``sum(|x - y| / (|x| + |y|))``
    "braycurtis"   BrayCurtisDistance    ``sum(|x - y|) / (sum(|x|) + sum(|y|))``
    =============  ====================  ========================================

    **Metrics intended for boolean-valued vector spaces:**  Any nonzero entry
    is evaluated to "True".  In the listings below, the following
    abbreviations are used:

     - N  : number of dimensions
     - NTT : number of dims in which both values are True
     - NTF : number of dims in which the first value is True, second is False
     - NFT : number of dims in which the first value is False, second is True
     - NFF : number of dims in which both values are False
     - NNEQ : number of non-equal dimensions, NNEQ = NTF + NFT
     - NNZ : number of nonzero dimensions, NNZ = NTF + NFT + NTT

    =================  =======================  ===============================
    identifier         class name               distance function
    -----------------  -----------------------  -------------------------------
    "jaccard"          JaccardDistance          NNEQ / NNZ
    "matching"         MatchingDistance         NNEQ / N
    "dice"             DiceDistance             NNEQ / (NTT + NNZ)
    "kulsinski"        KulsinskiDistance        (NNEQ + N - NTT) / (NNEQ + N)
    "rogerstanimoto"   RogersTanimotoDistance   2 * NNEQ / (N + NNEQ)
    "russellrao"       RussellRaoDistance       (N - NTT) / N
    "sokalmichener"    SokalMichenerDistance    2 * NNEQ / (N + NNEQ)
    "sokalsneath"      SokalSneathDistance      NNEQ / (NNEQ + 0.5 * NTT)
    =================  =======================  ===============================

    **User-defined distance:**

    ===========    ===============    =======
    identifier     class name         args
    -----------    ---------------    -------
    "pyfunc"       PyFuncDistance     func
    ===========    ===============    =======

    Here ``func`` is a function which takes two one-dimensional numpy
    arrays, and returns a distance.  Note that in order to be used within
    the BallTree, the distance must be a true metric:
    i.e. it must satisfy the following properties

    1) Non-negativity: d(x, y) >= 0
    2) Identity: d(x, y) = 0 if and only if x == y
    3) Symmetry: d(x, y) = d(y, x)
    4) Triangle Inequality: d(x, y) + d(y, z) >= d(x, z)

    Because of the Python object overhead involved in calling the python
    function, this will be fairly slow, but it will have the same
    scaling as other distances.
    """
    def __cinit__(self):
        self.p = 2
        self.vec = np.zeros(1, dtype=np.float64, order='C')
        self.mat = np.zeros((1, 1), dtype=np.float64, order='C')
        self.size = 1

    def __reduce__(self):
        """
        reduce method used for pickling
        """
        return (newObj, (self.__class__,), self.__getstate__())

    def __getstate__(self):
        """
        get state for pickling
        """
        if self.__class__.__name__ == "PyFuncDistance32":
            return (float(self.p), np.asarray(self.vec), np.asarray(self.mat), self.func, self.kwargs)
        return (float(self.p), np.asarray(self.vec), np.asarray(self.mat))

    def __setstate__(self, state):
        """
        set state for pickling
        """
        self.p = state[0]
        self.vec = state[1]
        self.mat = state[2]
        if self.__class__.__name__ == "PyFuncDistance32":
            self.func = state[3]
            self.kwargs = state[4]
        self.size = self.vec.shape[0]

    @classmethod
    def get_metric(cls, metric, **kwargs):
        """Get the given distance metric from the string identifier.

        See the docstring of DistanceMetric for a list of available metrics.

        Parameters
        ----------
        metric : str or class name
            The distance metric to use
        **kwargs
            additional arguments will be passed to the requested metric
        """
        if isinstance(metric, DistanceMetric32):
            return metric

        if callable(metric):
            return PyFuncDistance32(metric, **kwargs)

        # Map the metric string ID to the metric class
        if isinstance(metric, type) and issubclass(metric, DistanceMetric32):
            pass
        else:
            try:
                metric = METRIC_MAPPING32[metric]
            except:
                raise ValueError("Unrecognized metric '%s'" % metric)

        # In Minkowski special cases, return more efficient methods
        if metric is MinkowskiDistance32:
            p = kwargs.pop('p', 2)
            w = kwargs.pop('w', None)
            if p == 1 and w is None:
                return ManhattanDistance32(**kwargs)
            elif p == 2 and w is None:
                return EuclideanDistance32(**kwargs)
            elif np.isinf(p) and w is None:
                return ChebyshevDistance32(**kwargs)
            else:
                return MinkowskiDistance32(p, w, **kwargs)
        else:
            return metric(**kwargs)

    def __init__(self):
        if self.__class__ is DistanceMetric32:
            raise NotImplementedError("DistanceMetric32 is an abstract class")

    def _validate_data(self, X):
        """Validate the input data.

        This should be overridden in a base class if a specific input format
        is required.
        """
        return

    cdef float64_t dist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        """Compute the distance between vectors x1 and x2

        This should be overridden in a base class.
        """
        return -999

    cdef float64_t rdist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        """Compute the rank-preserving surrogate distance between vectors x1 and x2.

        This can optionally be overridden in a base class.

        The rank-preserving surrogate distance is any measure that yields the same
        rank as the distance, but is more efficient to compute. For example, the
        rank-preserving surrogate distance of the Euclidean metric is the
        squared-euclidean distance.
        """
        return self.dist(x1, x2, size)

    cdef int pdist(
        self,
        const float32_t[:, ::1] X,
        float64_t[:, ::1] D,
    ) except -1:
        """Compute the pairwise distances between points in X"""
        cdef intp_t i1, i2
        for i1 in range(X.shape[0]):
            for i2 in range(i1, X.shape[0]):
                D[i1, i2] = self.dist(&X[i1, 0], &X[i2, 0], X.shape[1])
                D[i2, i1] = D[i1, i2]
        return 0


    cdef int cdist(
        self,
        const float32_t[:, ::1] X,
        const float32_t[:, ::1] Y,
        float64_t[:, ::1] D,
    ) except -1:
        """Compute the cross-pairwise distances between arrays X and Y"""
        cdef intp_t i1, i2
        if X.shape[1] != Y.shape[1]:
            raise ValueError('X and Y must have the same second dimension')
        for i1 in range(X.shape[0]):
            for i2 in range(Y.shape[0]):
                D[i1, i2] = self.dist(&X[i1, 0], &Y[i2, 0], X.shape[1])
        return 0

    cdef float64_t dist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:
        """Compute the distance between vectors x1 and x2 represented
        under the CSR format.

        This must be overridden in a subclass.

        Notes
        -----
        0. The implementation of this method in subclasses must be robust to the
        presence of explicit zeros in the CSR representation.

        1. The `data` arrays are passed using pointers to be able to support an
        alternative representation of the CSR data structure for supporting
        fused sparse-dense datasets pairs with minimum overhead.

        See the explanations in `SparseDenseDatasetsPair.__init__`.

        2. An alternative signature would be:

            cdef float64_t dist_csr(
                self,
                const float32_t* x1_data,
                const int32_t[:] x1_indices,
                const float32_t* x2_data,
                const int32_t[:] x2_indices,
            ) except -1 nogil:

        Where callers would use slicing on the original CSR data and indices
        memoryviews:

            x1_start = X1_csr.indices_ptr[i]
            x1_end   = X1_csr.indices_ptr[i+1]
            x2_start = X2_csr.indices_ptr[j]
            x2_end   = X2_csr.indices_ptr[j+1]

            self.dist_csr(
                &x1_data[x1_start],
                x1_indices[x1_start:x1_end],
                &x2_data[x2_start],
                x2_indices[x2_start:x2_end],
            )

        Yet, slicing on memoryview slows down execution as it takes the GIL.
        See: https://github.com/scikit-learn/scikit-learn/issues/17299

        Hence, to avoid slicing the data and indices arrays of the sparse
        matrices containing respectively x1 and x2 (namely x{1,2}_{data,indices})
        are passed as well as their indices pointers (namely x{1,2}_{start,end}).

        3. For reference about the CSR format, see section 3.4 of
        Saad, Y. (2003), Iterative Methods for Sparse Linear Systems, SIAM.
        https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf
        """
        return -999

    cdef float64_t rdist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:
        """Distance between rows of CSR matrices x1 and x2.

        This can optionally be overridden in a subclass.

        The rank-preserving surrogate distance is any measure that yields the same
        rank as the distance, but is more efficient to compute. For example, the
        rank-preserving surrogate distance of the Euclidean metric is the
        squared-euclidean distance.

        Notes
        -----
        The implementation of this method in subclasses must be robust to the
        presence of explicit zeros in the CSR representation.

        More information about the motives for this method signature is given
        in the docstring of dist_csr.
        """
        return self.dist_csr(
            x1_data,
            x1_indices,
            x2_data,
            x2_indices,
            x1_start,
            x1_end,
            x2_start,
            x2_end,
            size,
        )

    cdef int pdist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const int32_t[:] x1_indptr,
        const intp_t size,
        float64_t[:, ::1] D,
    ) except -1 nogil:
        """Pairwise distances between rows in CSR matrix X.

        Note that this implementation is twice faster than cdist_csr(X, X)
        because it leverages the symmetry of the problem.
        """
        cdef:
            intp_t i1, i2
            intp_t n_x1 = x1_indptr.shape[0] - 1
            intp_t x1_start, x1_end, x2_start, x2_end

        for i1 in range(n_x1):
            x1_start = x1_indptr[i1]
            x1_end = x1_indptr[i1 + 1]
            for i2 in range(i1, n_x1):
                x2_start = x1_indptr[i2]
                x2_end = x1_indptr[i2 + 1]
                D[i1, i2] = D[i2, i1] = self.dist_csr(
                    x1_data,
                    x1_indices,
                    x1_data,
                    x1_indices,
                    x1_start,
                    x1_end,
                    x2_start,
                    x2_end,
                    size,
                )
        return 0

    cdef int cdist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const int32_t[:] x1_indptr,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t[:] x2_indptr,
        const intp_t size,
        float64_t[:, ::1] D,
    ) except -1 nogil:
        """Compute the cross-pairwise distances between arrays X and Y
        represented in the CSR format."""
        cdef:
            intp_t i1, i2
            intp_t n_x1 = x1_indptr.shape[0] - 1
            intp_t n_x2 = x2_indptr.shape[0] - 1
            intp_t x1_start, x1_end, x2_start, x2_end

        for i1 in range(n_x1):
            x1_start = x1_indptr[i1]
            x1_end = x1_indptr[i1 + 1]
            for i2 in range(n_x2):
                x2_start = x2_indptr[i2]
                x2_end = x2_indptr[i2 + 1]

                D[i1, i2] = self.dist_csr(
                    x1_data,
                    x1_indices,
                    x2_data,
                    x2_indices,
                    x1_start,
                    x1_end,
                    x2_start,
                    x2_end,
                    size,
                )
        return 0

    cdef float64_t _rdist_to_dist(self, float32_t rdist) except -1 nogil:
        """Convert the rank-preserving surrogate distance to the distance"""
        return rdist

    cdef float64_t _dist_to_rdist(self, float32_t dist) except -1 nogil:
        """Convert the distance to the rank-preserving surrogate distance"""
        return dist

    def rdist_to_dist(self, rdist):
        """Convert the rank-preserving surrogate distance to the distance.

        The surrogate distance is any measure that yields the same rank as the
        distance, but is more efficient to compute. For example, the
        rank-preserving surrogate distance of the Euclidean metric is the
        squared-euclidean distance.

        Parameters
        ----------
        rdist : double
            Surrogate distance.

        Returns
        -------
        double
            True distance.
        """
        return rdist

    def dist_to_rdist(self, dist):
        """Convert the true distance to the rank-preserving surrogate distance.

        The surrogate distance is any measure that yields the same rank as the
        distance, but is more efficient to compute. For example, the
        rank-preserving surrogate distance of the Euclidean metric is the
        squared-euclidean distance.

        Parameters
        ----------
        dist : double
            True distance.

        Returns
        -------
        double
            Surrogate distance.
        """
        return dist

    def _pairwise_dense_dense(self, X, Y):
        cdef const float32_t[:, ::1] Xarr
        cdef const float32_t[:, ::1] Yarr
        cdef float64_t[:, ::1] Darr

        Xarr = np.asarray(X, dtype=np.float32, order='C')
        self._validate_data(Xarr)
        if X is Y:
            Darr = np.empty((Xarr.shape[0], Xarr.shape[0]), dtype=np.float64, order='C')
            self.pdist(Xarr, Darr)
        else:
            Yarr = np.asarray(Y, dtype=np.float32, order='C')
            self._validate_data(Yarr)
            Darr = np.empty((Xarr.shape[0], Yarr.shape[0]), dtype=np.float64, order='C')
            self.cdist(Xarr, Yarr, Darr)
        return np.asarray(Darr)

    def _pairwise_sparse_sparse(self, X: csr_matrix , Y: csr_matrix):
        cdef:
            intp_t n_X, n_features
            const float32_t[:] X_data
            const int32_t[:] X_indices
            const int32_t[:] X_indptr

            intp_t n_Y
            const float32_t[:] Y_data
            const int32_t[:] Y_indices
            const int32_t[:] Y_indptr

            float64_t[:, ::1] Darr

        X_csr = X.tocsr()
        n_X, n_features = X_csr.shape
        X_data = np.asarray(X_csr.data, dtype=np.float32)
        X_indices = np.asarray(X_csr.indices, dtype=np.int32)
        X_indptr = np.asarray(X_csr.indptr, dtype=np.int32)
        if X is Y:
            Darr = np.empty((n_X, n_X), dtype=np.float64, order='C')
            self.pdist_csr(
                x1_data=&X_data[0],
                x1_indices=X_indices,
                x1_indptr=X_indptr,
                size=n_features,
                D=Darr,
            )
        else:
            Y_csr = Y.tocsr()
            n_Y, _ = Y_csr.shape
            Y_data = np.asarray(Y_csr.data, dtype=np.float32)
            Y_indices = np.asarray(Y_csr.indices, dtype=np.int32)
            Y_indptr = np.asarray(Y_csr.indptr, dtype=np.int32)

            Darr = np.empty((n_X, n_Y), dtype=np.float64, order='C')
            self.cdist_csr(
                x1_data=&X_data[0],
                x1_indices=X_indices,
                x1_indptr=X_indptr,
                x2_data=&Y_data[0],
                x2_indices=Y_indices,
                x2_indptr=Y_indptr,
                size=n_features,
                D=Darr,
            )
        return np.asarray(Darr)

    def _pairwise_sparse_dense(self, X: csr_matrix, Y):
        cdef:
            intp_t n_X = X.shape[0]
            intp_t n_features = X.shape[1]
            const float32_t[:] X_data = np.asarray(
                X.data, dtype=np.float32,
            )
            const int32_t[:] X_indices = np.asarray(
                X.indices, dtype=np.int32,
            )
            const int32_t[:] X_indptr = np.asarray(
                X.indptr, dtype=np.int32,
            )

            const float32_t[:, ::1] Y_data = np.asarray(
                Y, dtype=np.float32, order="C",
            )
            intp_t n_Y = Y_data.shape[0]
            const int32_t[:] Y_indices = (
                np.arange(n_features, dtype=np.int32)
            )

            float64_t[:, ::1] Darr = np.empty((n_X, n_Y), dtype=np.float64, order='C')

            intp_t i1, i2
            intp_t x1_start, x1_end
            float32_t * x2_data

        with nogil:
            # Use the exact same adaptation for CSR than in SparseDenseDatasetsPair
            # for supporting the sparse-dense case with minimal overhead.
            # Note: at this point this method is only a convenience method
            # used in the tests via the DistanceMetric.pairwise method.
            # Therefore, there is no need to attempt parallelization of those
            # nested for-loops.
            # Efficient parallel computation of pairwise distances can be
            # achieved via the PairwiseDistances class instead. The latter
            # internally calls into vector-wise distance computation from
            # the DistanceMetric subclass while benefiting from the generic
            # Cython/OpenMP parallelization template for the generic pairwise
            # distance + reduction computational pattern.
            for i1 in range(n_X):
                x1_start = X_indptr[i1]
                x1_end = X_indptr[i1 + 1]
                for i2 in range(n_Y):
                    x2_data = &Y_data[0, 0] + i2 * n_features

                    Darr[i1, i2] = self.dist_csr(
                        x1_data=&X_data[0],
                        x1_indices=X_indices,
                        x2_data=x2_data,
                        x2_indices=Y_indices,
                        x1_start=x1_start,
                        x1_end=x1_end,
                        x2_start=0,
                        x2_end=n_features,
                        size=n_features,
                    )

        return np.asarray(Darr)

    def _pairwise_dense_sparse(self, X, Y: csr_matrix):
        # We could have implemented this method using _pairwise_dense_sparse by
        # swapping argument and by transposing the results, but this would
        # have come with an extra copy to ensure C-contiguity of the result.
        cdef:
            intp_t n_X = X.shape[0]
            intp_t n_features = X.shape[1]

            const float32_t[:, ::1] X_data = np.asarray(
                X, dtype=np.float32, order="C",
            )
            const int32_t[:] X_indices = np.arange(
                n_features, dtype=np.int32,
            )

            intp_t n_Y = Y.shape[0]
            const float32_t[:] Y_data = np.asarray(
                Y.data, dtype=np.float32,
            )
            const int32_t[:] Y_indices = np.asarray(
                Y.indices, dtype=np.int32,
            )
            const int32_t[:] Y_indptr = np.asarray(
                Y.indptr, dtype=np.int32,
            )

            float64_t[:, ::1] Darr = np.empty((n_X, n_Y), dtype=np.float64, order='C')

            intp_t i1, i2
            float32_t * x1_data

            intp_t x2_start, x2_end

        with nogil:
            # Use the exact same adaptation for CSR than in SparseDenseDatasetsPair
            # for supporting the dense-sparse case with minimal overhead.
            # Note: at this point this method is only a convenience method
            # used in the tests via the DistanceMetric.pairwise method.
            # Therefore, there is no need to attempt parallelization of those
            # nested for-loops.
            # Efficient parallel computation of pairwise distances can be
            # achieved via the PairwiseDistances class instead. The latter
            # internally calls into vector-wise distance computation from
            # the DistanceMetric subclass while benefiting from the generic
            # Cython/OpenMP parallelization template for the generic pairwise
            # distance + reduction computational pattern.
            for i1 in range(n_X):
                x1_data = &X_data[0, 0] + i1 * n_features
                for i2 in range(n_Y):
                    x2_start = Y_indptr[i2]
                    x2_end = Y_indptr[i2 + 1]

                    Darr[i1, i2] = self.dist_csr(
                        x1_data=x1_data,
                        x1_indices=X_indices,
                        x2_data=&Y_data[0],
                        x2_indices=Y_indices,
                        x1_start=0,
                        x1_end=n_features,
                        x2_start=x2_start,
                        x2_end=x2_end,
                        size=n_features,
                    )

        return np.asarray(Darr)


    def pairwise(self, X, Y=None):
        """Compute the pairwise distances between X and Y

        This is a convenience routine for the sake of testing.  For many
        metrics, the utilities in scipy.spatial.distance.cdist and
        scipy.spatial.distance.pdist will be faster.

        Parameters
        ----------
        X : ndarray or CSR matrix of shape (n_samples_X, n_features)
            Input data.
        Y : ndarray or CSR matrix of shape (n_samples_Y, n_features)
            Input data.
            If not specified, then Y=X.

        Returns
        -------
        dist : ndarray of shape  (n_samples_X, n_samples_Y)
            The distance matrix of pairwise distances between points in X and Y.
        """
        X = check_array(X, accept_sparse=['csr'])

        if Y is None:
            Y = X
        else:
            Y = check_array(Y, accept_sparse=['csr'])

        X_is_sparse = issparse(X)
        Y_is_sparse = issparse(Y)

        if not X_is_sparse and not Y_is_sparse:
            return self._pairwise_dense_dense(X, Y)

        if X_is_sparse and Y_is_sparse:
            return self._pairwise_sparse_sparse(X, Y)

        if X_is_sparse and not Y_is_sparse:
            return self._pairwise_sparse_dense(X, Y)

        return self._pairwise_dense_sparse(X, Y)

#------------------------------------------------------------
# Euclidean Distance
#  d = sqrt(sum(x_i^2 - y_i^2))
cdef class EuclideanDistance32(DistanceMetric32):
    r"""Euclidean Distance metric

    .. math::
       D(x, y) = \sqrt{ \sum_i (x_i - y_i) ^ 2 }
    """
    def __init__(self):
        self.p = 2

    cdef inline float64_t dist(self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        return euclidean_dist32(x1, x2, size)

    cdef inline float64_t rdist(self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        return euclidean_rdist32(x1, x2, size)

    cdef inline float64_t _rdist_to_dist(self, float32_t rdist) except -1 nogil:
        return sqrt(rdist)

    cdef inline float64_t _dist_to_rdist(self, float32_t dist) except -1 nogil:
        return dist * dist

    def rdist_to_dist(self, rdist):
        return np.sqrt(rdist)

    def dist_to_rdist(self, dist):
        return dist ** 2

    cdef inline float64_t rdist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            float64_t d = 0.0
            float64_t unsquared = 0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            if ix1 == ix2:
                unsquared = x1_data[i1] - x2_data[i2]
                d = d + (unsquared * unsquared)
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                unsquared = x1_data[i1]
                d = d + (unsquared * unsquared)
                i1 = i1 + 1
            else:
                unsquared = x2_data[i2]
                d = d + (unsquared * unsquared)
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                unsquared = x2_data[i2]
                d = d + (unsquared * unsquared)
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                unsquared = x1_data[i1]
                d = d + (unsquared * unsquared)
                i1 = i1 + 1

        return d

    cdef inline float64_t dist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:
        return sqrt(
            self.rdist_csr(
            x1_data,
            x1_indices,
            x2_data,
            x2_indices,
            x1_start,
            x1_end,
            x2_start,
            x2_end,
            size,
        ))

#------------------------------------------------------------
# SEuclidean Distance
#  d = sqrt(sum((x_i - y_i2)^2 / v_i))
cdef class SEuclideanDistance32(DistanceMetric32):
    r"""Standardized Euclidean Distance metric

    .. math::
       D(x, y) = \sqrt{ \sum_i \frac{ (x_i - y_i) ^ 2}{V_i} }
    """
    def __init__(self, V):
        self.vec = np.asarray(V, dtype=np.float64)
        self.size = self.vec.shape[0]
        self.p = 2

    def _validate_data(self, X):
        if X.shape[1] != self.size:
            raise ValueError('SEuclidean dist: size of V does not match')

    cdef inline float64_t rdist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef float64_t tmp, d=0
        cdef intp_t j
        for j in range(size):
            tmp = x1[j] - x2[j]
            d += (tmp * tmp / self.vec[j])
        return d

    cdef inline float64_t dist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        return sqrt(self.rdist(x1, x2, size))

    cdef inline float64_t _rdist_to_dist(self, float32_t rdist) except -1 nogil:
        return sqrt(rdist)

    cdef inline float64_t _dist_to_rdist(self, float32_t dist) except -1 nogil:
        return dist * dist

    def rdist_to_dist(self, rdist):
        return np.sqrt(rdist)

    def dist_to_rdist(self, dist):
        return dist ** 2

    cdef inline float64_t rdist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            float64_t d = 0.0
            float64_t unsquared = 0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            if ix1 == ix2:
                unsquared = x1_data[i1] - x2_data[i2]
                d = d + (unsquared * unsquared) / self.vec[ix1]
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                unsquared = x1_data[i1]
                d = d + (unsquared * unsquared) / self.vec[ix1]
                i1 = i1 + 1
            else:
                unsquared = x2_data[i2]
                d = d + (unsquared * unsquared) / self.vec[ix2]
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                ix2 = x2_indices[i2]
                unsquared = x2_data[i2]
                d = d + (unsquared * unsquared) / self.vec[ix2]
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                ix1 = x1_indices[i1]
                unsquared = x1_data[i1]
                d = d + (unsquared * unsquared) / self.vec[ix1]
                i1 = i1 + 1
        return d

    cdef inline float64_t dist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:
        return sqrt(
            self.rdist_csr(
            x1_data,
            x1_indices,
            x2_data,
            x2_indices,
            x1_start,
            x1_end,
            x2_start,
            x2_end,
            size,
        ))

#------------------------------------------------------------
# Manhattan Distance
#  d = sum(abs(x_i - y_i))
cdef class ManhattanDistance32(DistanceMetric32):
    r"""Manhattan/City-block Distance metric

    .. math::
       D(x, y) = \sum_i |x_i - y_i|
    """
    def __init__(self):
        self.p = 1

    cdef inline float64_t dist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef float64_t d = 0
        cdef intp_t j
        for j in range(size):
            d += fabs(x1[j] - x2[j])
        return d

    cdef inline float64_t dist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            float64_t d = 0.0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            if ix1 == ix2:
                d = d + fabs(x1_data[i1] - x2_data[i2])
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                d = d + fabs(x1_data[i1])
                i1 = i1 + 1
            else:
                d = d + fabs(x2_data[i2])
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                d = d + fabs(x2_data[i2])
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                d = d + fabs(x1_data[i1])
                i1 = i1 + 1

        return d


#------------------------------------------------------------
# Chebyshev Distance
#  d = max_i(abs(x_i - y_i))
cdef class ChebyshevDistance32(DistanceMetric32):
    """Chebyshev/Infinity Distance

    .. math::
       D(x, y) = max_i (|x_i - y_i|)

    Examples
    --------
    >>> from sklearn.metrics.dist_metrics import DistanceMetric
    >>> dist = DistanceMetric.get_metric('chebyshev')
    >>> X = [[0, 1, 2],
    ...      [3, 4, 5]]
    >>> Y = [[-1, 0, 1],
    ...      [3, 4, 5]]
    >>> dist.pairwise(X, Y)
    array([[1.732..., 5.196...],
           [6.928..., 0....   ]])
    """
    def __init__(self):
        self.p = INF32

    cdef inline float64_t dist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef float64_t d = 0
        cdef intp_t j
        for j in range(size):
            d = fmax(d, fabs(x1[j] - x2[j]))
        return d


    cdef inline float64_t dist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            float64_t d = 0.0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            if ix1 == ix2:
                d = fmax(d, fabs(x1_data[i1] - x2_data[i2]))
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                d = fmax(d, fabs(x1_data[i1]))
                i1 = i1 + 1
            else:
                d = fmax(d, fabs(x2_data[i2]))
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                d = fmax(d, fabs(x2_data[i2]))
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                d = fmax(d, fabs(x1_data[i1]))
                i1 = i1 + 1

        return d


#------------------------------------------------------------
# Minkowski Distance
cdef class MinkowskiDistance32(DistanceMetric32):
    r"""Minkowski Distance

    .. math::
        D(x, y) = {||u-v||}_p

    when w is None.

    Here is the more general expanded expression for the weighted case:

    .. math::
        D(x, y) = [\sum_i w_i *|x_i - y_i|^p] ^ (1/p)

    Parameters
    ----------
    p : float
        The order of the p-norm of the difference (see above).

        .. versionchanged:: 1.4.0
            Minkowski distance allows `p` to be `0<p<1`.


    w : (N,) array-like (optional)
        The weight vector.

    Minkowski Distance requires p > 0 and finite.
    When :math:`p \in (0,1)`, it isn't a true metric but is permissible when
    the triangular inequality isn't necessary.
    For p = infinity, use ChebyshevDistance.
    Note that for p=1, ManhattanDistance is more efficient, and for
    p=2, EuclideanDistance is more efficient.

    """
    def __init__(self, p, w=None):
        if p <= 0:
            raise ValueError("p must be greater than 0")
        elif np.isinf(p):
            raise ValueError("MinkowskiDistance requires finite p. "
                             "For p=inf, use ChebyshevDistance.")

        self.p = p
        if w is not None:
            w_array = check_array(
                w, ensure_2d=False, dtype=np.float64, input_name="w"
            )
            if (w_array < 0).any():
                raise ValueError("w cannot contain negative weights")
            self.vec = w_array
            self.size = self.vec.shape[0]
        else:
            self.vec = np.asarray([], dtype=np.float64)
            self.size = 0

    def _validate_data(self, X):
        if self.size > 0 and X.shape[1] != self.size:
            raise ValueError("MinkowskiDistance: the size of w must match "
                             f"the number of features ({X.shape[1]}). "
                             f"Currently len(w)={self.size}.")

    cdef inline float64_t rdist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef float64_t d=0
        cdef intp_t j
        cdef bint has_w = self.size > 0
        if has_w:
            for j in range(size):
                d += (self.vec[j] * pow(fabs(x1[j] - x2[j]), self.p))
        else:
            for j in range(size):
                d += (pow(fabs(x1[j] - x2[j]), self.p))
        return d

    cdef inline float64_t dist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        return pow(self.rdist(x1, x2, size), 1. / self.p)

    cdef inline float64_t _rdist_to_dist(self, float32_t rdist) except -1 nogil:
        return pow(rdist, 1. / self.p)

    cdef inline float64_t _dist_to_rdist(self, float32_t dist) except -1 nogil:
        return pow(dist, self.p)

    def rdist_to_dist(self, rdist):
        return rdist ** (1. / self.p)

    def dist_to_rdist(self, dist):
        return dist ** self.p

    cdef inline float64_t rdist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            float64_t d = 0.0
            bint has_w = self.size > 0

        if has_w:
            while i1 < x1_end and i2 < x2_end:
                ix1 = x1_indices[i1]
                ix2 = x2_indices[i2]

                if ix1 == ix2:
                    d = d + (self.vec[ix1] * pow(fabs(
                        x1_data[i1] - x2_data[i2]
                    ), self.p))
                    i1 = i1 + 1
                    i2 = i2 + 1
                elif ix1 < ix2:
                    d = d + (self.vec[ix1] * pow(fabs(x1_data[i1]), self.p))
                    i1 = i1 + 1
                else:
                    d = d + (self.vec[ix2] * pow(fabs(x2_data[i2]), self.p))
                    i2 = i2 + 1

            if i1 == x1_end:
                while i2 < x2_end:
                    ix2 = x2_indices[i2]
                    d = d + (self.vec[ix2] * pow(fabs(x2_data[i2]), self.p))
                    i2 = i2 + 1
            else:
                while i1 < x1_end:
                    ix1 = x1_indices[i1]
                    d = d + (self.vec[ix1] * pow(fabs(x1_data[i1]), self.p))
                    i1 = i1 + 1

            return d
        else:
            while i1 < x1_end and i2 < x2_end:
                ix1 = x1_indices[i1]
                ix2 = x2_indices[i2]

                if ix1 == ix2:
                    d = d + (pow(fabs(
                        x1_data[i1] - x2_data[i2]
                    ), self.p))
                    i1 = i1 + 1
                    i2 = i2 + 1
                elif ix1 < ix2:
                    d = d + (pow(fabs(x1_data[i1]), self.p))
                    i1 = i1 + 1
                else:
                    d = d + (pow(fabs(x2_data[i2]), self.p))
                    i2 = i2 + 1

            if i1 == x1_end:
                while i2 < x2_end:
                    d = d + (pow(fabs(x2_data[i2]), self.p))
                    i2 = i2 + 1
            else:
                while i1 < x1_end:
                    d = d + (pow(fabs(x1_data[i1]), self.p))
                    i1 = i1 + 1

            return d

    cdef inline float64_t dist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:
        return pow(
            self.rdist_csr(
                x1_data,
                x1_indices,
                x2_data,
                x2_indices,
                x1_start,
                x1_end,
                x2_start,
                x2_end,
                size,
            ),
            1 / self.p
        )

#------------------------------------------------------------
# Mahalanobis Distance
#  d = sqrt( (x - y)^T V^-1 (x - y) )
cdef class MahalanobisDistance32(DistanceMetric32):
    """Mahalanobis Distance

    .. math::
       D(x, y) = \sqrt{ (x - y)^T V^{-1} (x - y) }

    Parameters
    ----------
    V : array-like
        Symmetric positive-definite covariance matrix.
        The inverse of this matrix will be explicitly computed.
    VI : array-like
        optionally specify the inverse directly.  If VI is passed,
        then V is not referenced.
    """
    cdef float64_t[::1] buffer

    def __init__(self, V=None, VI=None):
        if VI is None:
            if V is None:
                raise ValueError("Must provide either V or VI "
                                 "for Mahalanobis distance")
            VI = np.linalg.inv(V)
        if VI.ndim != 2 or VI.shape[0] != VI.shape[1]:
            raise ValueError("V/VI must be square")

        self.mat = np.asarray(VI, dtype=np.float64, order='C')

        self.size = self.mat.shape[0]

        # We need to create a buffer to store the vectors' coordinates' differences
        self.buffer = np.zeros(self.size, dtype=np.float64)

    def __setstate__(self, state):
        super().__setstate__(state)
        self.size = self.mat.shape[0]
        self.buffer = np.zeros(self.size, dtype=np.float64)

    def _validate_data(self, X):
        if X.shape[1] != self.size:
            raise ValueError('Mahalanobis dist: size of V does not match')

    cdef inline float64_t rdist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef float64_t tmp, d = 0
        cdef intp_t i, j

        # compute (x1 - x2).T * VI * (x1 - x2)
        for i in range(size):
            self.buffer[i] = x1[i] - x2[i]

        for i in range(size):
            tmp = 0
            for j in range(size):
                tmp += self.mat[i, j] * self.buffer[j]
            d += tmp * self.buffer[i]
        return d

    cdef inline float64_t dist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        return sqrt(self.rdist(x1, x2, size))

    cdef inline float64_t _rdist_to_dist(self, float32_t rdist) except -1 nogil:
        return sqrt(rdist)

    cdef inline float64_t _dist_to_rdist(self, float32_t dist) except -1 nogil:
        return dist * dist

    def rdist_to_dist(self, rdist):
        return np.sqrt(rdist)

    def dist_to_rdist(self, dist):
        return dist ** 2

    cdef inline float64_t rdist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            float64_t tmp, d = 0.0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            if ix1 == ix2:
                self.buffer[ix1] = x1_data[i1] - x2_data[i2]
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                self.buffer[ix1] = x1_data[i1]
                i1 = i1 + 1
            else:
                self.buffer[ix2] = - x2_data[i2]
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                ix2 = x2_indices[i2]
                self.buffer[ix2] = - x2_data[i2]
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                ix1 = x1_indices[i1]
                self.buffer[ix1] = x1_data[i1]
                i1 = i1 + 1

        for i in range(size):
            tmp = 0
            for j in range(size):
                tmp += self.mat[i, j] * self.buffer[j]
            d += tmp * self.buffer[i]

        return d

    cdef inline float64_t dist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:
        return sqrt(
            self.rdist_csr(
            x1_data,
            x1_indices,
            x2_data,
            x2_indices,
            x1_start,
            x1_end,
            x2_start,
            x2_end,
            size,
        ))

#------------------------------------------------------------
# Hamming Distance
#  d = N_unequal(x, y) / N_tot
cdef class HammingDistance32(DistanceMetric32):
    r"""Hamming Distance

    Hamming distance is meant for discrete-valued vectors, though it is
    a valid metric for real-valued vectors.

    .. math::
       D(x, y) = \frac{1}{N} \sum_i \delta_{x_i, y_i}
    """
    cdef inline float64_t dist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef int n_unequal = 0
        cdef intp_t j
        for j in range(size):
            if x1[j] != x2[j]:
                n_unequal += 1
        return float(n_unequal) / size


    cdef inline float64_t dist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            float64_t d = 0.0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            if ix1 == ix2:
                d += (x1_data[i1] != x2_data[i2])
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                d += (x1_data[i1] != 0)
                i1 = i1 + 1
            else:
                d += (x2_data[i2] != 0)
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                d += (x2_data[i2] != 0)
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                d += (x1_data[i1] != 0)
                i1 = i1 + 1

        d /= size

        return d


#------------------------------------------------------------
# Canberra Distance
#  D(x, y) = sum[ abs(x_i - y_i) / (abs(x_i) + abs(y_i)) ]
cdef class CanberraDistance32(DistanceMetric32):
    r"""Canberra Distance

    Canberra distance is meant for discrete-valued vectors, though it is
    a valid metric for real-valued vectors.

    .. math::
       D(x, y) = \sum_i \frac{|x_i - y_i|}{|x_i| + |y_i|}
    """
    cdef inline float64_t dist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef float64_t denom, d = 0
        cdef intp_t j
        for j in range(size):
            denom = fabs(x1[j]) + fabs(x2[j])
            if denom > 0:
                d += fabs(x1[j] - x2[j]) / denom
        return d

    cdef inline float64_t dist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            float64_t d = 0.0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            if ix1 == ix2:
                d += (
                        fabs(x1_data[i1] - x2_data[i2]) /
                        (fabs(x1_data[i1]) + fabs(x2_data[i2]))
                )
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                d += 1.
                i1 = i1 + 1
            else:
                d += 1.
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                d += 1.
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                d += 1.
                i1 = i1 + 1

        return d

#------------------------------------------------------------
# Bray-Curtis Distance
#  D(x, y) = sum[abs(x_i - y_i)] / sum[abs(x_i) + abs(y_i)]
cdef class BrayCurtisDistance32(DistanceMetric32):
    r"""Bray-Curtis Distance

    Bray-Curtis distance is meant for discrete-valued vectors, though it is
    a valid metric for real-valued vectors.

    .. math::
       D(x, y) = \frac{\sum_i |x_i - y_i|}{\sum_i(|x_i| + |y_i|)}
    """
    cdef inline float64_t dist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef float64_t num = 0, denom = 0
        cdef intp_t j
        for j in range(size):
            num += fabs(x1[j] - x2[j])
            denom += fabs(x1[j]) + fabs(x2[j])
        if denom > 0:
            return num / denom
        else:
            return 0.0

    cdef inline float64_t dist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            float64_t num = 0.0
            float64_t denom = 0.0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            if ix1 == ix2:
                num += fabs(x1_data[i1] - x2_data[i2])
                denom += fabs(x1_data[i1]) + fabs(x2_data[i2])
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                num += fabs(x1_data[i1])
                denom += fabs(x1_data[i1])
                i1 = i1 + 1
            else:
                num += fabs(x2_data[i2])
                denom += fabs(x2_data[i2])
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                num += fabs(x1_data[i1])
                denom += fabs(x1_data[i1])
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                num += fabs(x2_data[i2])
                denom += fabs(x2_data[i2])
                i1 = i1 + 1

        return num / denom

#------------------------------------------------------------
# Jaccard Distance (boolean)
#  D(x, y) = N_unequal(x, y) / N_nonzero(x, y)
cdef class JaccardDistance32(DistanceMetric32):
    r"""Jaccard Distance

    Jaccard Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = (N_TF + N_FT) / (N_TT + N_TF + N_FT)
    """
    cdef inline float64_t dist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef int tf1, tf2, n_eq = 0, nnz = 0
        cdef intp_t j
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            nnz += (tf1 or tf2)
            n_eq += (tf1 and tf2)
        # Based on https://github.com/scipy/scipy/pull/7373
        # When comparing two all-zero vectors, scipy>=1.2.0 jaccard metric
        # was changed to return 0, instead of nan.
        if nnz == 0:
            return 0
        return (nnz - n_eq) * 1.0 / nnz

    cdef inline float64_t dist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            intp_t tf1, tf2, n_tt = 0, nnz = 0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            tf1 = x1_data[i1] != 0
            tf2 = x2_data[i2] != 0

            if ix1 == ix2:
                nnz += (tf1 or tf2)
                n_tt += (tf1 and tf2)
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                nnz += tf1
                i1 = i1 + 1
            else:
                nnz += tf2
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                tf2 = x2_data[i2] != 0
                nnz += tf2
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                tf1 = x1_data[i1] != 0
                nnz += tf1
                i1 = i1 + 1

        # Based on https://github.com/scipy/scipy/pull/7373
        # When comparing two all-zero vectors, scipy>=1.2.0 jaccard metric
        # was changed to return 0, instead of nan.
        if nnz == 0:
            return 0
        return (nnz - n_tt) * 1.0 / nnz

#------------------------------------------------------------
# Matching Distance (boolean)
#  D(x, y) = n_neq / n
cdef class MatchingDistance32(DistanceMetric32):
    r"""Matching Distance

    Matching Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = (N_TF + N_FT) / N
    """
    cdef inline float64_t dist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef int tf1, tf2, n_neq = 0
        cdef intp_t j
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            n_neq += (tf1 != tf2)
        return n_neq * 1. / size

    cdef inline float64_t dist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            intp_t tf1, tf2, n_neq = 0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            if ix1 == ix2:
                tf1 = x1_data[i1] != 0
                tf2 = x2_data[i2] != 0
                n_neq += (tf1 != tf2)
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                n_neq += (x1_data[i1] != 0)
                i1 = i1 + 1
            else:
                n_neq += (x2_data[i2] != 0)
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                n_neq += (x2_data[i2] != 0)
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                n_neq += (x1_data[i1] != 0)
                i1 = i1 + 1

        return n_neq * 1.0 / size

#------------------------------------------------------------
# Dice Distance (boolean)
#  D(x, y) = n_neq / (2 * ntt + n_neq)
cdef class DiceDistance32(DistanceMetric32):
    r"""Dice Distance

    Dice Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = (N_TF + N_FT) / (2 * N_TT + N_TF + N_FT)

    """
    cdef inline float64_t dist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef int tf1, tf2, n_neq = 0, n_tt = 0
        cdef intp_t j
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            n_tt += (tf1 and tf2)
            n_neq += (tf1 != tf2)
        return n_neq / (2.0 * n_tt + n_neq)

    cdef inline float64_t dist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            intp_t tf1, tf2, n_tt = 0, n_neq = 0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            tf1 = x1_data[i1] != 0
            tf2 = x2_data[i2] != 0

            if ix1 == ix2:
                n_tt += (tf1 and tf2)
                n_neq += (tf1 != tf2)
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                n_neq += tf1
                i1 = i1 + 1
            else:
                n_neq += tf2
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                tf2 = x2_data[i2] != 0
                n_neq += tf2
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                tf1 = x1_data[i1] != 0
                n_neq += tf1
                i1 = i1 + 1

        return n_neq / (2.0 * n_tt + n_neq)


#------------------------------------------------------------
# Kulsinski Distance (boolean)
#  D(x, y) = (ntf + nft - ntt + n) / (n_neq + n)
cdef class KulsinskiDistance32(DistanceMetric32):
    r"""Kulsinski Distance

    Kulsinski Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = 1 - N_TT / (N + N_TF + N_FT)

    """
    cdef inline float64_t dist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef int tf1, tf2, n_tt = 0, n_neq = 0
        cdef intp_t j
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            n_neq += (tf1 != tf2)
            n_tt += (tf1 and tf2)
        return (n_neq - n_tt + size) * 1.0 / (n_neq + size)

    cdef inline float64_t dist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            intp_t tf1, tf2, n_tt = 0, n_neq = 0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            tf1 = x1_data[i1] != 0
            tf2 = x2_data[i2] != 0

            if ix1 == ix2:
                n_tt += (tf1 and tf2)
                n_neq += (tf1 != tf2)
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                n_neq += tf1
                i1 = i1 + 1
            else:
                n_neq += tf2
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                tf2 = x2_data[i2] != 0
                n_neq += tf2
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                tf1 = x1_data[i1] != 0
                n_neq += tf1
                i1 = i1 + 1

        return (n_neq - n_tt + size) * 1.0 / (n_neq + size)

#------------------------------------------------------------
# Rogers-Tanimoto Distance (boolean)
#  D(x, y) = 2 * n_neq / (n + n_neq)
cdef class RogersTanimotoDistance32(DistanceMetric32):
    r"""Rogers-Tanimoto Distance

    Rogers-Tanimoto Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = 2 (N_TF + N_FT) / (N + N_TF + N_FT)
    """
    cdef inline float64_t dist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef int tf1, tf2, n_neq = 0
        cdef intp_t j
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            n_neq += (tf1 != tf2)
        return (2.0 * n_neq) / (size + n_neq)

    cdef inline float64_t dist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            intp_t tf1, tf2, n_neq = 0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            tf1 = x1_data[i1] != 0
            tf2 = x2_data[i2] != 0

            if ix1 == ix2:
                n_neq += (tf1 != tf2)
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                n_neq += tf1
                i1 = i1 + 1
            else:
                n_neq += tf2
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                tf2 = x2_data[i2] != 0
                n_neq += tf2
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                tf1 = x1_data[i1] != 0
                n_neq += tf1
                i1 = i1 + 1

        return (2.0 * n_neq) / (size + n_neq)

#------------------------------------------------------------
# Russell-Rao Distance (boolean)
#  D(x, y) = (n - ntt) / n
cdef class RussellRaoDistance32(DistanceMetric32):
    r"""Russell-Rao Distance

    Russell-Rao Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = (N - N_TT) / N
    """
    cdef inline float64_t dist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef int tf1, tf2, n_tt = 0
        cdef intp_t j
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            n_tt += (tf1 and tf2)
        return (size - n_tt) * 1. / size

    cdef inline float64_t dist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            intp_t tf1, tf2, n_tt = 0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            tf1 = x1_data[i1] != 0
            tf2 = x2_data[i2] != 0

            if ix1 == ix2:
                n_tt += (tf1 and tf2)
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                i1 = i1 + 1
            else:
                i2 = i2 + 1

        # We don't need to go through all the longest
        # vector because tf1 or tf2 will be false
        # and thus n_tt won't be increased.

        return (size - n_tt) * 1. / size



#------------------------------------------------------------
# Sokal-Michener Distance (boolean)
#  D(x, y) = 2 * n_neq / (n + n_neq)
cdef class SokalMichenerDistance32(DistanceMetric32):
    r"""Sokal-Michener Distance

    Sokal-Michener Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = 2 (N_TF + N_FT) / (N + N_TF + N_FT)
    """
    cdef inline float64_t dist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef int tf1, tf2, n_neq = 0
        cdef intp_t j
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            n_neq += (tf1 != tf2)
        return (2.0 * n_neq) / (size + n_neq)

    cdef inline float64_t dist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            intp_t tf1, tf2, n_neq = 0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            tf1 = x1_data[i1] != 0
            tf2 = x2_data[i2] != 0

            if ix1 == ix2:
                n_neq += (tf1 != tf2)
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                n_neq += tf1
                i1 = i1 + 1
            else:
                n_neq += tf2
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                tf2 = x2_data[i2] != 0
                n_neq += tf2
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                tf1 = x1_data[i1] != 0
                n_neq += tf1
                i1 = i1 + 1

        return (2.0 * n_neq) / (size + n_neq)

#------------------------------------------------------------
# Sokal-Sneath Distance (boolean)
#  D(x, y) = n_neq / (0.5 * n_tt + n_neq)
cdef class SokalSneathDistance32(DistanceMetric32):
    r"""Sokal-Sneath Distance

    Sokal-Sneath Distance is a dissimilarity measure for boolean-valued
    vectors. All nonzero entries will be treated as True, zero entries will
    be treated as False.

        D(x, y) = (N_TF + N_FT) / (N_TT / 2 + N_FT + N_TF)
    """
    cdef inline float64_t dist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef int tf1, tf2, n_tt = 0, n_neq = 0
        cdef intp_t j
        for j in range(size):
            tf1 = x1[j] != 0
            tf2 = x2[j] != 0
            n_neq += (tf1 != tf2)
            n_tt += (tf1 and tf2)
        return n_neq / (0.5 * n_tt + n_neq)

    cdef inline float64_t dist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            intp_t tf1, tf2, n_tt = 0, n_neq = 0

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            tf1 = x1_data[i1] != 0
            tf2 = x2_data[i2] != 0

            if ix1 == ix2:
                n_tt += (tf1 and tf2)
                n_neq += (tf1 != tf2)
                i1 = i1 + 1
                i2 = i2 + 1
            elif ix1 < ix2:
                n_neq += tf1
                i1 = i1 + 1
            else:
                n_neq += tf2
                i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                tf2 = x2_data[i2] != 0
                n_neq += tf2
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                tf1 = x1_data[i1] != 0
                n_neq += tf1
                i1 = i1 + 1

        return n_neq / (0.5 * n_tt + n_neq)


#------------------------------------------------------------
# Haversine Distance (2 dimensional)
#  D(x, y) = 2 arcsin{sqrt[sin^2 ((x1 - y1) / 2)
#                          + cos(x1) cos(y1) sin^2 ((x2 - y2) / 2)]}
cdef class HaversineDistance32(DistanceMetric32):
    """Haversine (Spherical) Distance

    The Haversine distance is the angular distance between two points on
    the surface of a sphere.  The first distance of each point is assumed
    to be the latitude, the second is the longitude, given in radians.
    The dimension of the points must be 2:

    D(x, y) = 2 arcsin[sqrt{sin^2((x1 - y1) / 2) + cos(x1)cos(y1)sin^2((x2 - y2) / 2)}]

    """

    def _validate_data(self, X):
        if X.shape[1] != 2:
            raise ValueError("Haversine distance only valid "
                             "in 2 dimensions")

    cdef inline float64_t rdist(self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        cdef float64_t sin_0 = sin(0.5 * ((x1[0]) - (x2[0])))
        cdef float64_t sin_1 = sin(0.5 * ((x1[1]) - (x2[1])))
        return (sin_0 * sin_0 + cos(x1[0]) * cos(x2[0]) * sin_1 * sin_1)

    cdef inline float64_t dist(self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        return 2 * asin(sqrt(self.rdist(x1, x2, size)))

    cdef inline float64_t _rdist_to_dist(self, float32_t rdist) except -1 nogil:
        return 2 * asin(sqrt(rdist))

    cdef inline float64_t _dist_to_rdist(self, float32_t dist) except -1 nogil:
        cdef float64_t tmp = sin(0.5 *  dist)
        return tmp * tmp

    def rdist_to_dist(self, rdist):
        return 2 * np.arcsin(np.sqrt(rdist))

    def dist_to_rdist(self, dist):
        tmp = np.sin(0.5 * dist)
        return tmp * tmp

    cdef inline float64_t dist_csr(
         self,
         const float32_t* x1_data,
         const int32_t[:] x1_indices,
         const float32_t* x2_data,
         const int32_t[:] x2_indices,
         const int32_t x1_start,
         const int32_t x1_end,
         const int32_t x2_start,
         const int32_t x2_end,
         const intp_t size,
    ) except -1 nogil:
        return 2 * asin(sqrt(self.rdist_csr(
            x1_data,
            x1_indices,
            x2_data,
            x2_indices,
            x1_start,
            x1_end,
            x2_start,
            x2_end,
            size,
        )))

    cdef inline float64_t rdist_csr(
        self,
        const float32_t* x1_data,
        const int32_t[:] x1_indices,
        const float32_t* x2_data,
        const int32_t[:] x2_indices,
        const int32_t x1_start,
        const int32_t x1_end,
        const int32_t x2_start,
        const int32_t x2_end,
        const intp_t size,
    ) except -1 nogil:

        cdef:
            intp_t ix1, ix2
            intp_t i1 = x1_start
            intp_t i2 = x2_start

            float64_t x1_0 = 0
            float64_t x1_1 = 0
            float64_t x2_0 = 0
            float64_t x2_1 = 0
            float64_t sin_0
            float64_t sin_1

        while i1 < x1_end and i2 < x2_end:
            ix1 = x1_indices[i1]
            ix2 = x2_indices[i2]

            # Find the components in the 2D vectors to work with
            x1_component = ix1 if (x1_start == 0) else ix1 % x1_start
            x2_component = ix2 if (x2_start == 0) else ix2 % x2_start

            if x1_component == 0:
                x1_0 = x1_data[i1]
            else:
                x1_1 = x1_data[i1]

            if x2_component == 0:
                x2_0 = x2_data[i2]
            else:
                x2_1 = x2_data[i2]

            i1 = i1 + 1
            i2 = i2 + 1

        if i1 == x1_end:
            while i2 < x2_end:
                ix2 = x2_indices[i2]
                x2_component = ix2 if (x2_start == 0) else ix2 % x2_start
                if x2_component == 0:
                    x2_0 = x2_data[i2]
                else:
                    x2_1 = x2_data[i2]
                i2 = i2 + 1
        else:
            while i1 < x1_end:
                ix1 = x1_indices[i1]
                x1_component = ix1 if (x1_start == 0) else ix1 % x1_start
                if x1_component == 0:
                    x1_0 = x1_data[i1]
                else:
                    x1_1 = x1_data[i1]
                i1 = i1 + 1

        sin_0 = sin(0.5 * (x1_0 - x2_0))
        sin_1 = sin(0.5 * (x1_1 - x2_1))

        return (sin_0 * sin_0 + cos(x1_0) * cos(x2_0) * sin_1 * sin_1)

#------------------------------------------------------------
# User-defined distance
#
cdef class PyFuncDistance32(DistanceMetric32):
    """PyFunc Distance

    A user-defined distance

    Parameters
    ----------
    func : function
        func should take two numpy arrays as input, and return a distance.
    """
    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs

    # in cython < 0.26, GIL was required to be acquired during definition of
    # the function and inside the body of the function. This behaviour is not
    # allowed in cython >= 0.26 since it is a redundant GIL acquisition. The
    # only way to be back compatible is to inherit `dist` from the base class
    # without GIL and called an inline `_dist` which acquire GIL.
    cdef inline float64_t dist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 nogil:
        return self._dist(x1, x2, size)

    cdef inline float64_t _dist(
        self,
        const float32_t* x1,
        const float32_t* x2,
        intp_t size,
    ) except -1 with gil:
        cdef:
            object x1arr = _buffer_to_ndarray32(x1, size)
            object x2arr = _buffer_to_ndarray32(x2, size)
        d = self.func(x1arr, x2arr, **self.kwargs)
        try:
            # Cython generates code here that results in a TypeError
            # if d is the wrong type.
            return d
        except TypeError:
            raise TypeError("Custom distance function must accept two "
                            "vectors and return a float.")
