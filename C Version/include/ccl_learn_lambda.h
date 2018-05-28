/** \file ccl_learn_lambda.h
    \brief CCL header file for learning state dependent constraint with known jacobian
    \ingroup CCL_C
*/

#ifndef __CCL_LEARN_LAMBDA_H
#define __CCL_LEARN_LAMBDA_H

#include <../include/ccl_math.h>
#include <../include/ccl_learn_alpha.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

/** NUM_CONSTRAINT is the default full rank of the task constraint */
#define NUM_CONSTRAINT 2
#ifdef __cplusplus
extern "C" {
#endif
//typedef void (*JACOBIAN)(const double*,const int,double*);

/** \brief Computation routine for calculating jacobian for the robot inverse kinematic.

  \param[in] X       Observed state variables, Must point to an array of <em>dim_x * 1<em> of doubles
  \param[in] size    Number of links (Todo)
  \param[out] out    Jacobian matrix, must point to an array of <em>2 * 3<em> doubles (3 links arm)
 */
void Jacobian(const double* X,const int size,double* out);

/** \brief Allocates the memory for the learn_lambda model.

  \param[in] model Must point to a valid LEARN_A_MODEL structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be allocated)
 */
int ccl_learn_lambda_model_alloc(LEARN_A_MODEL *model);

/** \brief Free the memory for the learn_lambda model.

  \param[in] model Must point to a valid LEARN_A_MODEL structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be freed)
 */
int ccl_learn_lambda_model_free(LEARN_A_MODEL *model);

/** \brief Main computation routine for learning state dependent constraint lambda.

  \param[in] Un      Observed actions, Must point to an array of <em>dim_u * dim_n<em> of doubles
  \param[in] X       Observed state variables, Must point to an array of <em>dim_x * dim_n<em> of doubles
  \param[in] J_func  Functor for calculating the jacobian matrix
  \param[in] dim_b   Number of basis functions
  \param[in] dim_r   Dimensionality of the task space
  \param[in] dim_n   NUmber of data points
  \param[in] dim_x   Dimensionality of the state variables
  \param[in] dim_u   Dimensionality of the action space
  \param[out]optimal optimal model paramters
 */
void ccl_learn_lambda(const double * Un,const double *X,void (*J_func)(const double*,const int,double*),const int dim_b,const int dim_r,const int dim_n,const int dim_x,const int dim_u,LEARN_A_MODEL optimal);

/** \brief Computation routine for prediction of the lambda matrix

  \param[in] x           Input state variable, must point to an array of <em>dim_x<em> doubles
  \param[in] model       Must be pointer to a valid LEARN_A_MODEL structure
  \param[in] J_func      Functor for calculating the jacobian matrix
  \param[in] centres     Rbf centers, must point to an array of <em>1 * dim_b<em> doubles.
  \param[in] variance    Variance of the rbf
  \param[in] Iu          Identity matrix, must point to an array of <em>dim_u * dim_u<em> doubles
  \param[out] A          Constraint lambda matrix, must point to an array of <em>dim_k * dum_u<em> doubles
*/
void predict_proj_lambda(double* x, LEARN_A_MODEL model,void (*J_func)(const double*,const int,double*),double* centres,double variance,double* Iu, double*A);

/** \brief Write model parameters to .txt file.
  \param[in] filename    File name
  \param[in] model       Must be pointer to a valid LEARN_A_MODEL structure
*/
int ccl_write_learn_lambda_model(char* filename, LEARN_A_MODEL *model);
#ifdef __cplusplus
}
#endif
#endif

