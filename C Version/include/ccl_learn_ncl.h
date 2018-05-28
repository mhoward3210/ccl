/** \file ccl_learn_ncl.h
    \brief CCL header file for learning nullspace component
    \ingroup CCL_C
*/

#ifndef __CCL_LEARN_NCL_H
#define __CCL_LEARN_NCL_H

#include <ccl_math.h>
#include <ccl_learn_alpha.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_sf.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#ifdef __cplusplus
extern "C" {
#endif

/** \brief This structure describes a "LEARN_NCL_MODEL" (a learn ncl model).
    This structure constains the dimentionality of the defined problems and
    the model parameters. It should always initialised with ccl_learn_ncl_model_alloc,
    and destroyed with ccl_learn_ncl_model_free.
    \ingroup CCL_C
*/
typedef struct {
    int dim_b;  /**< \brief The number of basis functions */
    int dim_x;  /**< \brief The dimensionality of the state space */
    int dim_y;  /**< \brief The dimensionality of the action space */
    int dim_n;  /**< \brief The number of data points */
    double* c;  /**< \brief The mean of the rbf */
    double s2;  /**< \brief The variance of the rbf */
    double* w;  /**< \brief The model parameters for learning nullspace component*/
}LEARN_NCL_MODEL;

/** \brief This structure describes workspace for directly learning the model parameters using
    linear regression.

    This structure defines the worspace variables and initialises with their dimensionalities.
    It should always initialised with ccl_learn_ncl_model_alloc,
    and destroyed with ccl_learn_ncl_model_free.
    \ingroup CCL_C
*/
typedef struct {
    gsl_matrix* HS;    /**< \brief Regularization basis for the H matrix */
    gsl_matrix* g;     /**< \brief Dot product of BX and U */
    gsl_matrix* Y_T;   /**< \brief Transpose of Y_  */
    gsl_matrix* Y_;    /**< \brief Output variable  */
    gsl_matrix* H;     /**< \brief Dot product of BX and BX' */
    gsl_matrix* BX_T;  /**< \brief Transpose of BX_ */
    gsl_matrix* BX_;   /**< \brief High dimensionality of the input data */
    gsl_matrix* w_;    /**< \brief Model parameters */
    gsl_matrix* pinvH1;/**< \brief Peuso inverse of H1 */
    gsl_vector* V;     /**< \brief Eigen vector of H */
    gsl_matrix* D;     /**< \brief Diagobal matrix with eigen values of H */
    int    * idx;      /**< \brief index */
}LEARN_MODEL_WS;

/** \brief This structure defines the workspace variables for calculating objective functions

    This structure containts the memory of the workspace variables for calculating objective functions.
    It should always initialise with obj_ws_alloc and destroy with
    obj_ws_free.
    \ingroup CCL_C
*/
typedef struct {
    double* W;           /**< \brief Model parameters */
    double* W_;          /**< \brief A copy of model parameters */
    gsl_matrix* J;       /**< \brief Jacobian matrix */
    gsl_vector* b_n;     /**< \brief A col vector of BX */
    gsl_matrix* b_n_T;   /**< \brief Transpose of BX */
    gsl_vector* u_n;     /**< \brief A col vector of u */
    gsl_matrix* u_n_T;   /**< \brief Transpose of u_n */
    gsl_matrix* BX_;     /**< \brief Higher dimensionality of input variable */
    gsl_matrix* Y_;      /**< \brief Observation actions */
    gsl_vector* Wb;      /**< \brief Dot product of w and b */
    gsl_matrix* Wb_T;    /**< \brief Transpose of Wb */
    double c;            /**< \brief Dot product of Wb' and Wb */
    double a;            /**< \brief Dot product of u_n' and Wb */
    gsl_matrix* j_n;     /**< \brief Row vector of analytic jacobian */
    gsl_vector* j_n_flt; /**< \brief Flattened jacobian vector*/
    gsl_matrix* tmp2;    /**< \brief A temporal matrix */
    double tmp;          /**< \brief A temporal scalar */
} OBJ_WS;

/** \brief This structure defines the workspace variables for solving the non-linear LM optimization

    This structure containts the memory of the workspace variables for sovling the non-linear LM
    optimization problem. It should always initialise with ccl_learn_model_ws_alloc and destroy with
    ccl_learn_model_ws_free.
    \ingroup CCL_C
*/
typedef struct{
    int      dim_x;             /**< \brief The dimensionality of the state variable */
    int      dim_n;             /**< \brief The number of data points */
    int      dim_b;             /**< \brief The number of basis functions */
    int      dim_y;             /**< \brief The dimensionality of the action space */
    int      r_ok;              /**< \brief check if the objective function is belowed the tolerence */
    int      d_ok;              /**< \brief check if the model parameters are belowed the tolerence */
    double * xc;                /**< \brief A copy of the initial model parameters */
    double * x;                 /**< \brief The flattened and updated model parameters */
    double * xf;                /**< \brief The finalised model parameters */
    double * epsx;              /**< \brief The tolerence of the model paramters */
    double   epsf;              /**< \brief The tolerence fo the objective functions */
    double * r;                 /**< \brief The returned value of the residual error */
    double * J;                 /**< \brief The jacobian at x */
    double   S;                 /**< \brief The sqaure root if x is taken */
    double * A;                 /**< \brief The dot product of J_T and J */
    double * v;                 /**< \brief The dot product of J_T and r */
    double * D;                 /**< \brief Automatic scaling */
    double   Rlo;               /**< \brief The lower bound of R */
    double   Rhi;               /**< \brief The upper bound of R */
    double   l;                 /**< \brief The adaptive learning rate */
    double   lc;                /**< \brief Handling situation when learning rate happends to be 0 */
    double * d;                 /**< \brief The parameter improvement gradient */
    int      iter;              /**< \brief The iteration number */
    double * xd;                /**< \brief The next x */
    double * rd;                /**< \brief The residual error at xd */
    double   Sd;                /**< \brief The squared error if xd is taken */
    double   dS;                /**< \brief The denomitor of the sqaured error is xd is taken */
    double   R;                 /**< \brief The reduction if xd is taken */
    double   nu;                /**< \brief The coefficient of changing learning rate */
    double*  d_T;               /**< \brief The matrix transpose of d */
    double*  J_T;               /**< \brief The matrix transpose of J */
    double* tmp;                /**< \brief The temporal variable */
    double* rd_T;               /**< \brief The matrix transpose of rd */
    gsl_matrix* D_pinv;         /**< \brief The peudo-inverse of D */
    gsl_vector* A_d;            /**< \brief The A matrix at xd*/
    gsl_matrix* A_inv;          /**< \brief The peudo-inverse of A */
    gsl_vector* A_inv_diag;     /**< \brief The vector of diagonal elements of A matrix*/
    gsl_vector* r_T;            /**< \brief The vector transpose of r */
} SOLVE_NONLIN_WS;

/** \brief Allocates the memory for the learn_ncl model.

  \param[in] model Must point to a valid LEARN_NCL_MODEL structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be allocated)
 */
int ccl_learn_ncl_model_alloc(LEARN_NCL_MODEL *model);

/** \brief Free the memory for the learn_ncl model.

  \param[in] model Must point to a valid LEARN_NCL_MODEL structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be freed)
 */
int ccl_learn_ncl_model_free(LEARN_NCL_MODEL *model);

/** \brief Computation routine for learn nullspace component .

  \param[in] X           Input state variables, must point to an array of <em>dim_x * dim_n<em> doubles
  \param[in] Y           Observation actions
  \param[in] dim_x       Dimensionality of state inputs
  \param[in] dim_y       Dimensionality of action space
  \param[in] dim_y       Number of data samples
  \param[in] dim_y       Dimensionality of rbf
  \param[out] model      Must be pointer to a valid LEARN_NCL_MODEL structure
*/
void ccl_learn_ncl(const double * X, const double *Y, const int dim_x, const int dim_y, const int dim_n, const int dim_b, LEARN_NCL_MODEL *model);

/** \brief Computation routine for calculating residual errors.

  \param[in] model       Must be pointer to a valid LEARN_NCL_MODEL structure
  \param[in] BX          Transformed state variables, must point to an array of <em>dim_b * dim_n<em> of doubles
  \param[in] Y           Observation actions
  \param[out] model       Must be pointer to a valid LEARN_NCL_MODEL structure
*/
void ccl_learn_model_dir(LEARN_NCL_MODEL *model, const double *BX, const double *Y);

/** \brief Allocates the memory for  LEARN_MODEL_WS structure.

  \param[in] model Must point to a valid LEARN_MODEL_WS structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be allocated)
 */
int ccl_learn_model_ws_alloc(LEARN_NCL_MODEL *model,LEARN_MODEL_WS* ws);

/** \brief Free the memory for  LEARN_MODEL_WS structure.

  \param[in] model Must point to a valid LEARN_MODEL_WS structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be freed)
 */
int ccl_learn_model_ws_free(LEARN_MODEL_WS* ws);

/** \brief Computation routine for calculating residual errors.

  \param[in] model       Must be pointer to a valid LEARN_NCL_MODEL structure
  \param[in] W           Returned model parameter, must point to an array of <em>dim_b * dim_x<em> doubles
  \param[in] BX          Transformed state variables, must point to an array of <em>dim_b * dim_n<em> of doubles
  \param[in] Y           Observation actions
  \param[out] fun        Returned scalar residual
  \param[out] J          Returned Jacobian matrix, must point to an array of <em>dim_n * dim_x<em> doubles
*/
void obj_ncl(const LEARN_NCL_MODEL *model, const double* W, const double*BX, const double*Y, double*fun,double* J);

/** \brief Allocates the memory for OBJ_WS structure.

  \param[in] model Must point to a valid OBJ_WS structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be allocated)
 */
void obj_ws_alloc(const LEARN_NCL_MODEL *model,OBJ_WS* ws);

/** \brief Free the memory for OBJ_WS structure.

  \param[in] model Must point to a valid OBJ_WS structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be freed)
 */
void obj_ws_free(OBJ_WS* ws);

/** \brief Computation routine for optimizing non-linear objective functions using LM approach.

  \param[in] model       Must be pointer to a valid LEARN_NCL_MODEL structure.
  \param[in] BX          Transformed state variables, must point to an array of <em>dim_b * dim_n<em> of doubles
  \param[in] Y           Observation actions
  \param[in] option      Options for optimizer
  \param[in] lm_ws_param Must point to a valid structure of SOLVE_LM_WS for LM optimizer workspace variables
  \param[out] W          Returned model parameter, must point to an array of <em>dim_b * dim_x<em> doubles
*/
void ccl_lsqnonlin(const LEARN_NCL_MODEL* model,const  double* BX, const double*Y, const OPTION option,SOLVE_NONLIN_WS * lm_ws_param, double* W);

/** \brief Allocates the memory for SOLVE_NONLIN_WS structure.

  \param[in] model Must point to a valid SOLVE_NONLIN_WS structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be allocated)
 */
int ccl_solve_nonlin_ws_alloc(const LEARN_NCL_MODEL *model,SOLVE_NONLIN_WS * lm_ws);

/** \brief Free the memory for SOLVE_NONLIN_WS structure.

  \param[in] model Must point to a valid SOLVE_NONLIN_WS structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be freed)
 */
int ccl_solve_nonlin_ws_free(SOLVE_NONLIN_WS * lm_ws);

/** \brief Prediction of nullspace components.

  \param[in] model       Must be pointer to a valid LEARN_NCL_MODEL structure
  \param[in] BX          Higher dimensionality of the input variable, must point to an array of <em>dim_b * dim_n<em> doubles
  \param[out] Unp        Prediction of nullspace component, must point to an array of <em>dim_u * dum_n<em> doubles
*/
void predict_ncl(const LEARN_NCL_MODEL* model, const double* BX, double* Unp);

/** \brief Write model parameters to .txt file.
  \param[in] filename    File name
  \param[in] model       Must be pointer to a valid LEARN_NCL_MODEL structure
*/
int ccl_write_ncl_model(char* filename, LEARN_NCL_MODEL *model);
#ifdef __cplusplus
}
#endif
#endif

