/** \file ccl_math.h
    \brief CCL header file for math
    \ingroup CCL_C
*/

#ifndef __CCL_MATH_H
#define __CCL_MATH_H

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_sort.h>

/** \brief This structure describes the workspace memory of calculating matrix pseudo inverse MP_INV_WS.

    This structure defines the model variables and initialises with their dimensionalities. It should always initialised
    with ccl_MP_inv_ws_alloc, and destroyed with ccl_MP_inv_ws_free.
    \ingroup CCL_C
*/
typedef struct {
    gsl_matrix *V;              /**< \brief SVD return of A */
    gsl_matrix *Sigma_pinv;     /**< \brief Pseudo inverse of sigma */
    gsl_matrix *U;              /**< \brief Pad to full matrix of A */
    gsl_matrix *A_pinv;         /**< \brief Pseodo inverse of A */
    gsl_matrix *A;              /**< \brief Copy of input matrix A */
    gsl_matrix *_tmp_mat;       /**< \brief A temporal matrix */
    gsl_vector *_tmp_vec;       /**< \brief A temporal vector */
    gsl_vector *u;              /**< \brief SVD return of A */
}MP_INV_WS;

/** \brief Matrix addition.

  \param[in] A           Input matrix, must point to an array of <em>row * col<em> doubles
  \param[in] B           Output matrix, must point to an array of <em>row * col<em> doubles
  \param[in] row         Number of rows
  \param[in] col         Number of cols
  \param[out] A           Input matrix, must point to an array of <em>row * col<em> doubles
 */
void ccl_mat_add (double *A, const double *B, int row, int col);

/** \brief Matrix subtraction.

  \param[in] A           Input matrix, must point to an array of <em>row * col<em> doubles
  \param[in] B           Output matrix, must point to an array of <em>row * col<em> doubles
  \param[in] row         Number of rows
  \param[in] col         Number of cols
  \param[out] A           Input matrix, must point to an array of <em>row * col<em> doubles
 */
void ccl_mat_sub(double *A, const double *B, int row, int col);

/** \brief Sum of a vector.

  \param[in] vec         Input vector, must point to an array of <em>size * 1<em> doubles
  \param[in] B           Output matrix, must point to an array of <em>row * col<em> doubles
  \return           The summation of input variable
 */
double ccl_vec_sum(double* vec, int size);

/** \brief Append matrix horizontally.

  \param[in] A           Input matrix, must point to an array of <em>i * j<em> doubles
  \param[in] i           Number of rows
  \param[in] j           Number of cols
  \param[in] B           Output matrix, must point to an array of <em>k * d<em> doubles
  \param[in] k           Number of rows
  \param[in] d           Number of cols
  \param[out] c          Appended matrix, must point to an array of <em>i * j+d<em> doubles
 */
void mat_hotz_app ( double *A, int i,int j, const double *B,int k,int d,double* c);

/** \brief Append matrix vertically.

  \param[in] A           Input matrix, must point to an array of <em>i * j<em> doubles
  \param[in] i           Number of rows
  \param[in] j           Number of cols
  \param[in] B           Output matrix, must point to an array of <em>k * d<em> doubles
  \param[in] k           Number of rows
  \param[in] d           Number of cols
  \param[out] c          Appended matrix, must point to an array of <em>i+k * j<em> doubles
 */
void mat_vert_app (const double *A, int i, int j, const double *B, int k, int d, double * c);

/** \brief Dot product of two matrix.

  \param[in] A           Input matrix, must point to an array of <em>i * j<em> doubles
  \param[in] i           Number of rows
  \param[in] j           Number of cols
  \param[in] B           Output matrix, must point to an array of <em>k * d<em> doubles
  \param[in] k           Number of rows
  \param[in] d           Number of cols
  \param[out] C          Returned matrix, must point to an array of <em>i * d<em> doubles
 */
void ccl_dot_product (const double *A, int i,int j, const double *B,int k,int d,double *C);

/** \brief Matrix inverse of square matrix.

  \param[in] A           Input matrix, must point to an array of <em>i * j<em> doubles
  \param[in] i           Number of rows
  \param[in] j           Number of cols
  \param[out] invA       Matrix inverse, must point to an array of <em>j * i<em> doubles
 */
void ccl_mat_inv(const double *A, int i, int j, double *invA);

/** \brief Matrix inverse of arbitrary matrix.

  \param[in] A_in        Input matrix, must point to an array of <em>row * col<em> doubles
  \param[in] row         Number of rows
  \param[in] col         Number of cols
  \param[out] invA       Matrix inverse, must point to an array of <em>row * col<em> doubles
 */
void ccl_MP_pinv(const double *A_in, int row, int col, double *invA);

/** \brief Allocates the workspace memory for MP_INV_WS.

  \param[in] ws     Must point to a valid MP_INV_WS structure
  \param[in] n      Number of rows
  \param[in] m      Number of cols
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be allocated)
 */
int ccl_MP_inv_ws_alloc(MP_INV_WS *ws, int n, int m);

/** \brief Free the workspace memory for MP_INV_WS.

  \param[in] ws     Must point to a valid MP_INV_WS structure
  \param[in] n      Number of rows
  \param[in] m      Number of cols
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be freed)
 */
int ccl_MP_inv_ws_free(MP_INV_WS *ws);
//void ccl_MP_pinv_test(const double *A_in, int row, int col, MP_INV_WS *ws,double *invA);

/** \brief Equally divided space.

  \param[in] min         Mimimum value
  \param[in] max         maximum value
  \param[in] n           Number of data
  \param[out] y          Returned vector, must point to an array of <em>n<em> doubles
 */
void linspace(double min, double max, double n, double *y);

/** \brief Replicate matrix.

  \param[in] mat         Input matrix, must point to an array of <em>row*col<em> doubles
  \param[in] row         Number of rows
  \param[in] col         Number of cols
  \param[in] rows        Number of rows for replication
  \param[in] cols        Number of cols for replication
  \param[out] A          Returned matix, must point to an array of <em>row*rows * col*cols<em> doubles
 */
void repmat(const double *mat, int row, int col, int rows, int cols, double *A);

/** \brief Replicate vector.

  \param[in] vec         Input vector
  \param[in] rows        Number of rows for replication
  \param[in] cols        Number of cols for replication
  \param[out] A          Returned matix
 */
void repvec(const gsl_vector *vec, int rows, int cols, gsl_matrix * A);

/** \brief Replicate vector vertically (to be removed).

  \param[in] vec         Input vector, must point to an array of <em> size <em> doubles
  \param[in] rows        Number of rows for replication
  \param[out] A          Returned matix, must point to an array of <em>row*rows * 1<em> doubles
 */
void repvvec(const double *vec, int size, int rows, double *A);

/** \brief Flatten matrix to vector.

  \param[in] mat         Input matrix, must point to an array of <em> row * col <em> doubles
  \param[in] row         Number of rows
  \param[in] col         Number of cols
  \param[out] vec        Returned vector, must point to an array of <em>row*col * 1<em> doubles
 */
void flt_mat(const double *mat, int row, int col, double *vec);

/** \brief Variances of a matrix.

  \param[in] data_in     Input matrix, must point to an array of <em> row * col <em> doubles
  \param[in] row         Number of rows
  \param[in] col         Number of cols
  \param[in] axis        Axis of calculation
  \param[out] var        Returned vector
 */
void ccl_mat_var(const double* data_in,int row,int col,int axis,double * var);

/** \brief Print GSL matrix.

  \param[in] mat     Input GSL matrix
 */
void print_mat(gsl_matrix *mat);

/** \brief Print double matrix.

  \param[in] mat         Input GSL matrix
  \param[in] row         Number of rows
  \param[in] col         Number of cols
 */
void print_mat_d(double *mat, int row, int col);

/** \brief Print int matrix.

  \param[in] mat         Input GSL matrix
  \param[in] row         Number of rows
  \param[in] col         Number of cols
 */
void print_mat_i(int * mat,int row,int col);

/** \brief Convert vector type to matrix type.

  \param[in] vec         Input GSL vector
  \param[out] mat        Output GSL matrix
 */
void vec_to_mat(const gsl_vector *vec, gsl_matrix *mat);

/** \brief Round double array to some decimal.

  \param[in] n           Input double array, must point to an array of <em>size<em> doubles
  \param[in] size        Size of array
  \param[in] c           Number of decimal
  \param[out] ret        Rounded double array
 */
void nround(const double *n, int size, unsigned int c, double *ret);

/** \brief K mean algorithm
  \param[in] X           Input variable, must point to an array of <em>dim_x * dim_n<em> doubles
  \param[in] dim_x       Dimensionality of input variable
  \param[in] dim_n       Number of data samples
  \param[in] dim_b       Number of rbf centers
  \param[out] centres    Centers of input data
 */
void generate_kmeans_centres(const double * X, const int dim_x,const int dim_n, const int dim_b,double * centres);

/** \brief Get sub cols of a matrix
  \param[in] mat         Input matrix, must point to an array of <em>row * col<em> doubles
  \param[in] row         Number of rows
  \param[in] col         Number of cols
  \param[in] ind         Index of the sub matrix, must point to an array of <em>size<em> doubles
  \param[in] size        Number of index
  \param[out] ret        Sub matrix
 */
void ccl_get_sub_mat_cols(const double * mat, const int row, const int col,const int * ind, int size, double * ret);

/** \brief Distance of two matrices
  \param[in] A           Input matrix, must point to an array of <em>i * j<em> doubles
  \param[in] a_i         Number of rows
  \param[in] a_j         Number of cols
  \param[in] B           Output matrix, must point to an array of <em>k * d<em> doubles
  \param[in] b_i         Number of rows
  \param[in] b_j         Number of cols
  \param[out] D          Distance matrix, must point to an array of <em>a_i * b_j<em> doubles
 */
int ccl_mat_distance(const double *A,const int a_i,const int a_j,const double *B,const int b_i,const int b_j,double * D);

/** \brief Summation of a matrix
  \param[in] mat         Input matrix, must point to an array of <em>i * j<em> doubles
  \param[in] i           Number of rows
  \param[in] j           Number of cols
  \param[in] axis        Axis of calculation
  \param[out] ret        Vector of summation
 */
void ccl_mat_sum(const double *mat, const int i, const int j, const int axis, double * ret);

/** \brief Minimum values of a matrix
  \param[in] mat         Input matrix, must point to an array of <em>i * j<em> doubles
  \param[in] i           Number of rows
  \param[in] j           Number of cols
  \param[in] axis        Axis of calculation
  \param[out] val        Vector of minimum values in the matrix
  \param[out] indx       Vector of index for minimum values
 */
void ccl_mat_min(const double * mat,const int i,const int j,const int axis,double* val,int* indx);

/** \brief Mean values of a matrix
  \param[in] mat         Input matrix, must point to an array of <em>i * j<em> doubles
  \param[in] i           Number of rows
  \param[in] j           Number of cols
  \param[in] axis        Axis of calculation
  \param[out] val        Vector of mean values in the matrix
 */
void ccl_mat_mean(const double *mat,const int i, const int j,const int axis,double*val);

/** \brief Find index of an array of integers
  \param[in] a             Input array, must point to an array of <em>num_elements<em> int
  \param[in] num_elements  Number of elements
  \param[in] operand       Operations
  \param[in] value         Tolerance value
  \param[out] idx          Index of the value
  \return
    - 1 found
    - 0 not found
 */
int ccl_find_index_int(const int *a, const int num_elements, const int operand, const int value,int* idx);

/** \brief Find index of an array of integers. operand 1: = ; 2 >=; 3<= .
  \param[in] a             Input array, must point to an array of <em>num_elements<em> integers
  \param[in] num_elements  Number of elements
  \param[in] operand       Operations
  \param[in] value         Tolerance value
  \param[out] idx          Index of the value
  \return
    - 1 found
    - 0 not found
 */
int ccl_find_index_double(const double *a, const int num_elements, const int operand, const double value,int* idx);

/** \brief Find index of an array of doubles. operand 1: = ; 2 >=; 3<= .
  \param[in] mat           Input matrix, must point to an array of <em>i * j<em> doubles
  \param[in] i             Number of rows
  \param[in] j             Number of cols
  \param[in] c             Index of the col
  \param[out] vec          Returned vector
 */
void ccl_mat_set_col(double * mat, int i, int j, int c, double* vec);

/** \brief Radial basis function
  \param[in] X             Input variables, must point to an array of <em>i * j<em> double
  \param[in] i             Number of rows
  \param[in] j             Number of cols
  \param[in] C             Centers
  \param[in] k             Number of rows of centers
  \param[in] d             Number of cols of centers
  \param[in] s             Variance
  \param[out] BX           Higher dimensionality of the input variables
 */
void ccl_gaussian_rbf(const double * X,int i, int j,const double *C,int k, int d,double s,double * BX);

/** \brief Matrix transpose
  \param[in] mat           Input matrix, must point to an array of <em>i * j<em> double
  \param[in] i             Number of rows
  \param[in] j             Number of cols
  \param[out] mat_T        Tranpose matrix
 */
void ccl_mat_transpose(const double* mat, int i, int j, double* mat_T);

/** \brief Return true if any of the element in the vector satify the tolerance: op==0 '>=', op==1 '<='
  \param[in] vec           Input vector, must point to an array of <em>size<em> doubles
  \param[in] eps           Tolerance
  \param[in] size          Number of elements
  \param[in] op            Operations
  \return
     - 0 false
     - 1 true
 */
int  ccl_any(double* vec, double epx, int size, int op);

/** \brief Reshape vector to matrix
  \param[in] vec           Input vector, must point to an array of <em>i*j<em> doubles
  \param[in] i             Number of rows for the matrix
  \param[in] j             Number of cols for the matrix
  \param[in] mat           Reshaped matrix
 */
void ccl_mat_reshape(const double *vec, int i, int j, double *mat);
#ifdef __cplusplus
}
#endif
#endif
