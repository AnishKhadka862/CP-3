#include <omp.h>

const char* dgemv_desc = "Vectorized implementation of matrix-vector multiply using OpenMP.";

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */
void my_dgemv(int n, double* A, double* x, double* y) {
   int num_threads = omp_get_max_threads();

   #pragma omp parallel for num_threads(num_threads)
   for (int i = 0; i < n; i++) {
      double sum = 0.0;
      for (int j = 0; j < n; j++) {
         sum += A[i*n+j] * x[j];
      }
      #pragma omp atomic
      y[i] += sum;
   }
}
