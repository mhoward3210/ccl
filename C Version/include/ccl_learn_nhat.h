/** \file ccl_learn_nhat.h
    \brief CCL header file for learning state independent constraint
    \ingroup CCL_C
*/

#ifndef __CCL_LEARN_NHAT_H
#define __CCL_LEARN_NHAT_H

#include <ccl_math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/** NUM_SEARCH is the search dimension */
#define NUM_SEARCH 27000 // NUM_SEARCH = num_theta^{dim_t};

/** NUM_CONSTRAINT is the default full rank of the task constraint */
#define NUM_CONSTRAINT 4 //NUM_CONSTRAINT = dim_u
#ifdef __cplusplus
extern "C" {
#endif
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>

/** \brief This structure defines the searching parameters for NHAT_search

    This structure containts the dimension and the memory of NHAT_search.
    It should always initialise with nhat_mem_alloc_search and destroy with
    nhat_mem_free_search.
    \ingroup CCL_C
*/
typedef struct {
    int dim_u;                  /**< \brief The dimensionality of action space */
    int dim_n;                  /**< \brief The number of data points */
    int num_theta;              /**< \brief Number of candidate constraints (from 0 to pi) */
    int dim_t;                  /**< \brief Number of parameters needed to represent a unit vector */
    int dim_s;                  /**< \brief The dimensionality of the searching space */
    double epsilon;             /**< \brief Tolerance of seraching */
    double *min_theta;          /**< \brief The lower bound of theta angle */
    double *max_theta;          /**< \brief The upper bound of theta angle */
    double *list;               /**< \brief A dictionary of search space and theta angles */
    double *I_u;                /**< \brief The identity matrix */
    double *theta[NUM_SEARCH];  /**< \brief All the theta in searching space */
    double *alpha[NUM_SEARCH];  /**< \brief All the alpha in searching space in unit vector */
} NHAT_search;

/** \brief This structure defines the model parameters for NHAT_Model

    This structure containts the dimension and the memory of NHAT_Model.
    It should always initialise with nhat_mem_alloc_model and destroy with
    nhat_mem_free_model.
    \ingroup CCL_C
*/
typedef struct{
    int dim_u;          /**< \brief The dimensionality of action space */
    int dim_n;          /**< \brief The number of data points */
    int dim_t;          /**< \brief Number of parameters needed to represent a unit vector */
    int dim_c;          /**< \brief The dimensionality of constraints */

    double *theta;      /**< \brief Returned optimal theta parameters */
    double *alpha;      /**< \brief Returned optimal alpha in unit vector */
    double *P;          /**< \brief Returned optimal projection matrix */
    double variance;    /**< \brief Variance of nullspace components */
    double umse_j;      /**< \brief The mean sqaure error of residual error */
    double nmse_j;      /**< \brief The normalised mean sqaure error of residual error */
}NHAT_Model;

/** \brief This structure defines the model parameters for NHAT_result

    This structure records all the returned models in NHAT_result.
    It should always initialise with nhat_mem_alloc_result and destroy with
    nhat_mem_free_result.
    \ingroup CCL_C
*/
typedef struct{
    NHAT_Model model[NUM_CONSTRAINT];   /**< \brief The storage of returned NHAT_Model*/
}NHAT_result;

/** \brief Initialisation of the searching parameters.

  \param[in] search      Must be pointer to a valid NHAT_search structure
  \param[in] dim_u       Dimensionality of action space
  \param[in] dim_n       Number of data samples
  \param[in] num_theta   Number of candidate constraints (from 0 to pi)
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be initialised)
*/
int  init_search_param(NHAT_search *search, int dim_u, int dim_n, int num_theta);

/** \brief Allocates the memory for NHAT_search.

  \param[in] search Must point to a valid NHAT_search structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be allocated)
 */
int  nhat_mem_alloc_search(NHAT_search *search);

/** \brief Free the memory for NHAT_search.

  \param[in] search Must point to a valid NHAT_search structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be freed)
 */
int  nhat_mem_free_search(NHAT_search *search);

/** \brief Allocates the memory for NHAT_Model.

  \param[in] model  Must point to a valid NHAT_Model structure
  \param[in] search Must point to a valid NHAT_search structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be allocated)
 */
int  nhat_mem_alloc_model(NHAT_Model *model,const NHAT_search *search);

/** \brief Free the memory for NHAT_Model.

  \param[in] model Must point to a valid NHAT_Model structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be freed)
 */
int  nhat_mem_free_model(NHAT_Model *model);

/** \brief Duplicate the NHAT_Model model from src to dest.

  \param[in] dest Must point to a valid NHAT_Model structure
  \param[in] src  Must point to a valid NHAT_Model structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be duplicated)
 */
int  nhat_duplicate_model(NHAT_Model *dest, const NHAT_Model * src);

/** \brief Computation routine for generating unit vector for row vector alpha by given theta anlges.

  \param[in] theta    Learned constraint parameters, must point to an array of <em>1 * dim_t<em> doubles
  \param[in] dim_t    Number of constraint parameters
  \param[out]alpha    A uit vector of constraint basis, must point to an array of <em>1 * dim_u<em> doubles
*/
void get_unit_vector(const double* theta, int dim_t,double *alpha);

/** \brief Generate search space

  \param[in] search    Must point to a valid NHAT_search structure
  \param[out] search   Must point to a valid NHAT_search structure
*/
void generate_search_space(NHAT_search *search);

/** \brief Search first row of A matrix.

  \param[in] Vn        Pre-calculation of Un*Un'
  \param[in] Un        Observed actions
  \param[in] search    Must point to a valid NHAT_search structure
  \param[out] model    Must point to a valid NHAT_Model structure
  \param[out] stats    Statistics of model learning
*/
void search_first_alpha(const double *Vn,  const double *Un, NHAT_Model *model, const NHAT_search *search, double *stats);

/** \brief Search the rest rows of A matrix.

  \param[in] Vn        Pre-calculation of Un*Un'
  \param[in] Un        Observed actions
  \param[in] model    Must point to a valid NHAT_Model structure from a previous model
  \param[in] search    Must point to a valid NHAT_search structure
  \param[out] model    Must point to a valid NHAT_Model structure
  \param[out] stats    Statistics of model learning
*/
void search_alpha_nhat(const double *Vn,  const double *Un, NHAT_Model *model, const NHAT_search *search, NHAT_Model *model_out,double *stats);

/** \brief  Main computation routine for learn nhat.

  \param[in] Un        Observed actions
  \param[in] dim_u     Dimensionality of action space
  \param[in] dim_n     Number of data samples
  \param[out] optimal  Must point to a valid NHAT_Model structure
*/
void learn_nhat(const double *Un, const int dim_u, const int dim_n, NHAT_Model *optimal);

/** \brief  Calculation of nullspace projection matrix.

  \param[in] A         Constraint matrix, must point to an array of <em>dim_u * dim_u<em> doubles
  \param[in] row       Number of rows
  \param[in] col       Number of cols
  \param[out] N        Projection matrix
*/
void calclate_N(double * N, const double *A, int row, int col);

/** \brief Allocates the memory for model results.

  \param[in] model  Must point to a valid NHAT_Model structure
  \param[in] search Must point to a valid NHAT_search structure
  \param[in] dim_c  Current dimension of constraints
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be allocated for the results)
 */
int  nhat_mem_alloc_result(NHAT_Model *model, const NHAT_search search, int dim_c);

/** \brief Free the memory for model results.

  \param[in] result   Must point to a valid NHAT_result structure
  \param[in] n_models Number of constraints
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be freed for the results)
 */
int  nhat_mem_free_result(NHAT_result *result, int n_models);

/** \brief Allocates the memory for optimal model.

  \param[in] optimal  Must point to a valid NHAT_Model structure
  \param[in] search   Must point to a valid NHAT_search structure
  \param[in] dim_c    Current dimension of constraints
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be allocated for the optimal)
 */
int  nhat_mem_alloc_optimal(NHAT_Model *optimal,const NHAT_search search,int dim_c);

/** \brief Free the memory for optimal model.

  \param[in] optimal  Must point to a valid NHAT_Model structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be freed for the optimal)
 */
int  nhat_mem_free_optimal(NHAT_Model *optimal);
#ifdef __cplusplus
}
#endif
#endif

