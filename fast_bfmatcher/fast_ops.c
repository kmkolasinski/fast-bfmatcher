#include <stdio.h>
#include <blis.h>
#include "cblas.h"

// porting to mobile information
// https://stackoverflow.com/questions/11228855/header-files-for-x86-simd-intrinsics
// https://github.com/DLTcollab/sse2neon
#include <immintrin.h>
#include "fast_ops.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


void printv(__m128 v){
    printf("fvec = [ %f, %f, %f, %f ]\n", v[0], v[1], v[2], v[3]);
}


void printvi(__m128i v){
    int v0 = _mm_extract_epi32(v, 0);
    int v1 = _mm_extract_epi32(v, 1);
    int v2 = _mm_extract_epi32(v, 2);
    int v3 = _mm_extract_epi32(v, 3);

    printf("ivec = [ %d, %d, %d, %d ]\n", v0, v1, v2, v3);
}


float find_min(float* buff, int n)
{
    // https://doc.rust-lang.org/nightly/core/arch/x86_64/index.html
    // https://www.cs.virginia.edu/~cr4bd/3330/S2018/simdref.html
    int i;
    float vmin = buff[0];
    const int K = 4;
    __m128 minval = _mm_loadu_ps(&buff[0]);

    for (i = 0; i + K < n; i += K) {
         minval = _mm_min_ps(minval,  _mm_loadu_ps(&buff[i]));
    }

    for (; i < n; ++i) {
        if(buff[i] < vmin){
            vmin = buff[i];
        }
    }

    for(i = 0; i < K; ++i){
        if(minval[i] < vmin){
            vmin = minval[i];
        }
    }

    return vmin;
}



int argmin_vector(float *x, int n, float* min_value){

    int ret_val;
    float smin;
    int i, k;

    ret_val = 0;
    smin = find_min(x, n);

    const int K = 4;
    const __m128i vIndexInc = _mm_set1_epi32(K);
    const __m128 vMinVal = _mm_set1_ps(smin);

    __m128i vMinIndex = _mm_setr_epi32(0, 1, 2, 3);
    __m128i vIndex = vMinIndex;

    for (i = 0; i + 4 < n; i+=4) {
        __m128 vcmp = _mm_cmpeq_ps(_mm_loadu_ps(&x[i]), vMinVal);
        __m128i mask = _mm_castps_si128(vcmp);
        vMinIndex = _mm_min_epi32(_mm_madd_epi16(vIndex, mask), vMinIndex);
        vIndex = _mm_add_epi32(vIndex, vIndexInc);
    }

    k = -1;
    for (; i < n; ++i) {
        k = (x[i] == smin) ? i : k;
    }

    if ( k < 0){
        k = MAX(-_mm_extract_epi32(vMinIndex, 0), k);
        k = MAX(-_mm_extract_epi32(vMinIndex, 1), k);
        k = MAX(-_mm_extract_epi32(vMinIndex, 2), k);
        k = MAX(-_mm_extract_epi32(vMinIndex, 3), k);
    }

    ret_val = k;
    *min_value = smin;
    return ret_val;
}


void sum_square_cols(float* X, float *y, int num_rows, int num_cols) {

  int i, j;
  float sum;
  float *row_ptr;

  #pragma omp parallel for private(i, j, sum)
  for (i = 0; i < num_rows; ++i)
  {
       row_ptr = (X + i * num_cols);
       sum = 0.0;
       for (j = 0; j < num_cols; ++j){
            sum += row_ptr[j] * row_ptr[j];
       }
       y[i] = sum;
  }

}


void fast_cross_check_match(int *irow, float *vrow, float *vcol, float* X, int num_rows, int num_cols) {

  int i, j;
  float min_value;
  float *row_ptr;

  #pragma omp parallel for private(i, min_value)
  for (i = 0; i < num_rows; ++i){
       irow[i] = argmin_vector((X + i * num_cols), num_cols, &min_value);
       vrow[i] = min_value;
  }

  #pragma GCC ivdep
  for (j = 0; j < num_cols; ++j){
      vcol[j] = X[j];
  }

  for (i = 0; i < num_rows; ++i){
    row_ptr = (X + i * num_cols);

    #pragma GCC ivdep
    for (j = 0; j < num_cols; ++j){
        vcol[j] = MIN(row_ptr[j], vcol[j]);
    }
  }

}


void sum_row_and_col_vectors(float* row, float *col, float* X, int num_rows, int num_cols) {

  int i, j;
  float *row_ptr;
  float row_val;

  #pragma omp parallel for private(i, j, row_val, row_ptr)
  for (i = 0; i < num_rows; ++i){

    row_ptr = (X + i * num_cols);
    row_val = row[i];

    #pragma GCC ivdep
    for (j = 0; j < num_cols; ++j){
       row_ptr[j] = row_val + col[j];
    }
  }
}


void fast_ratio_test_match(int *irow, float *vrow, float* X, int num_rows, int num_cols, float ratio) {
  // finds two nearest neighbours for Lowe's test

  int i, min_index;
  float min_value, second_min_value;

  #pragma omp parallel for private(i, min_value, min_index, second_min_value)
  for (i = 0; i < num_rows; ++i){

       min_index = argmin_vector((X + i * num_cols), num_cols, &min_value);

       float tmp_value = X[min_index + i * num_cols];
       // some large value,
       X[min_index + i * num_cols] = 1000000.0;

       // search for second min value in the row
       argmin_vector((X + i * num_cols), num_cols, &second_min_value);

       // revert change
       X[min_index + i * num_cols] = tmp_value;

       if (min_value / second_min_value > ratio){
            irow[i] = -1;
            vrow[i] = 0;
       }else{
            irow[i] = min_index;
            vrow[i] = min_value;
       }
  }
}