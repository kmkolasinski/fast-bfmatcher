
int argmin_vector(float *x, int n, float* v);

void sum_square_cols(float* X, float *y, int num_rows, int num_cols);

void sum_row_and_col_vectors(float* row, float *col, float* X, int num_rows, int num_cols);

void fast_cross_check_match(int *irow, float *vrow, float *vcol, float* X, int num_rows, int num_cols);

void fast_ratio_test_match(int *irow, float *vrow, float* X, int num_rows, int num_cols, float ratio);