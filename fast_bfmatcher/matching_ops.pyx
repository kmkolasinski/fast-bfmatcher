import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

np.import_array()


ctypedef np.float32_t Float32_t
ctypedef np.float64_t Float64_t
ctypedef np.uint8_t Uint8_t
ctypedef np.int32_t Int32_t


cdef extern from "fast_ops.h":
    void sum_square_cols(float* X, float *y, int num_rows, int num_cols)
    void sum_row_and_col_vectors(float * row, float *col, float* X, int num_rows, int num_cols)
    void _fast_cross_check_match "fast_cross_check_match"(int *irow, float *vrow, float *vcol, float * X, int num_rows, int num_cols)
    void _fast_ratio_test_match "fast_ratio_test_match"(int *irow, float *vrow, float * X, int num_rows, int num_cols, float ratio)
    void _fast_ratio_test_cross_check_match "fast_ratio_test_cross_check_match"(int *irow, float *vrow, float *vcol, float * X, int num_rows, int num_cols, float ratio);


cdef extern from "blis.h":
    ctypedef enum trans_t:
        BLIS_TRANSPOSE
        BLIS_NO_TRANSPOSE

    void _bli_sgemm "bli_sgemm"(trans_t trans_a, trans_t trans_b,
                        int m, int n, int k, float* alpha,
                        float *a, int rsa, int csa,
                        float *b, int rsb, int csb,
                        float *beta, float *c, int rsc, int csc) nogil


cpdef void blis_sgemm_transpose(
        float alpha,
        float[:, ::1] A,
        float[:, ::1] B,
        float beta,
        float[:, ::1] C
):

    """
    Computes C = α A B^T + β C
    """

    cdef float* A_ptr = &A[0, 0]
    cdef float* B_ptr = &B[0, 0]
    cdef float* C_ptr = &C[0, 0]

    _bli_sgemm(BLIS_NO_TRANSPOSE,BLIS_TRANSPOSE,
            C.shape[0], C.shape[1], A.shape[1],
            &alpha,
            A_ptr, A.shape[1], 1,
            B_ptr, B.shape[1], 1,
            &beta,
            C_ptr, C.shape[1], 1)


cpdef void l2_distance_matrix(
    float[:, ::1] A,
    float[:, ::1] B,
    float[:, ::1] C
):
    """
    Computes squared euclidean distance matrix between arrays A and B
        
    Args:
        A: array of shape [N, C] 
        B: array of shape [M, C]
        C: output array of shape [N, M]

    """
    cdef float* A_ptr = &A[0, 0]
    cdef float* B_ptr = &B[0, 0]
    cdef float* C_ptr = &C[0, 0]

    cdef int a_num_rows = A.shape[0]
    cdef int a_num_cols = A.shape[1]
    cdef int b_num_rows = B.shape[0]
    cdef int b_num_cols = B.shape[1]

    cdef float *a_sq = <float*> malloc(a_num_rows * sizeof(float))
    cdef float *b_sq = <float*> malloc(b_num_rows * sizeof(float))

    try:
        sum_square_cols(A_ptr, a_sq, a_num_rows, a_num_cols)
        sum_square_cols(B_ptr, b_sq, b_num_rows, b_num_cols)
        sum_row_and_col_vectors(a_sq, b_sq, C_ptr, a_num_rows, b_num_rows)
        blis_sgemm_transpose(-2, A, B, 1, C)

    finally:
        free(a_sq)
        free(b_sq)


cpdef find_cross_check_matches(float[:, ::1] X):
    """
    
    Args:
        X: [N, M] float32 distance matrix  

    """

    cdef:
        int num_rows = X.shape[0]
        int num_cols = X.shape[1]

        int[::1] row_indices = np.zeros((num_rows,), dtype = np.int32)
        float[::1] row_values = np.zeros((num_rows,), dtype=np.float32)
        float[::1] col_values = np.zeros((num_cols,), dtype=np.float32)

        float *X_ptr = &X[0, 0]

    _fast_cross_check_match(
        &row_indices[0], &row_values[0], &col_values[0],
        X_ptr, num_rows, num_cols
    )

    return row_indices, row_values


cpdef find_ratio_test_matches(float[:, ::1] X, float ratio = 0.7):
    """

    Args:
        X: [N, M] float32 distance matrix  
        ratio: ratio test threshold for the second nearest neighbour

    """

    cdef:
        int num_rows = X.shape[0]
        int num_cols = X.shape[1]

        int[::1] row_indices = np.zeros((num_rows,), dtype = np.int32)
        float[::1] row_values = np.zeros((num_rows,), dtype=np.float32)

        float *X_ptr = &X[0, 0]

    _fast_ratio_test_match(
        &row_indices[0], &row_values[0],
        X_ptr, num_rows, num_cols, ratio
    )

    return row_indices, row_values


cpdef find_cross_check_and_ratio_test_matches(float[:, ::1] X, float ratio):
    """

    Args:
        X: [N, M] float32 distance matrix  
        ratio: ratio test value

    """

    cdef:
        int num_rows = X.shape[0]
        int num_cols = X.shape[1]

        int[::1] row_indices = np.zeros((num_rows,), dtype = np.int32)
        float[::1] row_values = np.zeros((num_rows,), dtype=np.float32)
        float[::1] col_values = np.zeros((num_cols,), dtype=np.float32)

        float *X_ptr = &X[0, 0]

    _fast_ratio_test_cross_check_match(
        &row_indices[0], &row_values[0], &col_values[0],
        X_ptr, num_rows, num_cols, ratio
    )

    return row_indices, row_values


cpdef l2_matches(float[:, ::1] A, float[:, ::1] B, str mode, float ratio = 0.0):

    cdef:
        int num_rows = A.shape[0]
        int num_cols = B.shape[0]

        float[:,::1] D = np.zeros((num_rows, num_cols), dtype = np.float32)
        # use ratio^2 because D is squared distance matrix
        float squared_ratio = ratio ** 2

    l2_distance_matrix(A, B, D)

    if mode == "ratio_test":
        row_indices, row_values = find_ratio_test_matches(D, squared_ratio)
    elif mode == "cross_check_test":
        row_indices, row_values = find_cross_check_matches(D)
    elif mode == "ratio_and_cross_check_test":
        row_indices, row_values = find_cross_check_and_ratio_test_matches(D, squared_ratio)
    else:
        raise NotImplemented(f"Not implemented match mode: {mode}")

    row_index = np.arange(0, A.shape[0])
    valid_matches = np.array(row_indices) != -1

    rows = row_index[valid_matches]
    cols = np.array(row_indices)[valid_matches]

    distances = np.array(row_values)[valid_matches]
    indices = np.transpose(np.stack([rows, cols]))
    distances = np.sqrt(np.maximum(0, distances))
    return indices, distances
