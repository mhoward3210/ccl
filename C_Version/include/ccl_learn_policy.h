/** \file ccl_learn_policy.h
    \brief CCL header file for learning unconstraint policy
    \ingroup CCL_C
*/

#ifndef __CCL_LEARN_POLICY_H
#define __CCL_LEARN_POLICY_H

#include <ccl_math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_sf.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif

/** NUM_CENTRES is the number of rbf centers */
#define NUM_CENTRES 20

/** \brief This structure describes workspace for directly learning the policy model parameters using
    linear regression.

    This structure defines the worspace variables and initialises with their dimensionalities.
    It should always initialised with ccl_learn_model_pi_ws_alloc,
    and destroyed with ccl_learn_model_pi_ws_free.
    \ingroup CCL_C
*/
typedef struct {
    gsl_matrix* HS;     /**< \brief Regularization basis for the H matrix */
    gsl_matrix* g;      /**< \brief Dot product of BX and U */
    gsl_matrix* Y_T;    /**< \brief Transpose of Y_  */
    gsl_matrix* Y_;     /**< \brief Output variable  */
    gsl_matrix* H;      /**< \brief Dot product of BX and BX' */
    gsl_matrix* BX_T;   /**< \brief Transpose of BX_ */
    gsl_matrix* BX_;    /**< \brief High dimensionality of the input data */
    gsl_matrix* w_;     /**< \brief Model parameters */
    gsl_matrix* pinvH1; /**< \brief Peuso inverse of H1 */
    gsl_vector* V;      /**< \brief Eigen vector of H */
    gsl_matrix* D;      /**< \brief Diagobal matrix with eigen values of H */
    int    * idx;       /**< \brief index */
}LEARN_MODEL_PI_WS;

/** \brief This structure describes learning a direct linear policy model parameters LEARN_MODEL_PI.

    This structure defines the model variables and initialises with their dimensionalities.
    \ingroup CCL_C
*/
typedef struct{
    int      dim_y;     /**< \brief Dimentionality of output variable */
    int      dim_x;     /**< \brief Dimentionality of input state variable */
    int      dim_n;     /**< \brief Number of data samples */
    int      dim_b;     /**< \brief Number of rbf centers */
    double * w;         /**< \brief Model parameters */
}LEARN_MODEL_PI;

/** \brief This structure describes learning a direct locally weighted linear policy model parameters LEARN_MODEL_LW_PI.

    This structure defines the model variables and initialises with their dimensionalities. It should always initialised
    with ccl_learn_policy_lw_pi_model_alloc, and destroyed with ccl_learn_policy_lw_pi_model_free.
    \ingroup CCL_C
*/
typedef struct{
    int      dim_y;          /**< \brief Dimentionality of output variable */
    int      dim_x;          /**< \brief Dimentionality of input state variable */
    int      dim_n;          /**< \brief Number of data samples */
    int      dim_b;          /**< \brief Number of rbf centers */
    int      dim_phi;        /**< \brief Dimensionality of feature */
    double * c;              /**< \brief rbf centers */
    double   s2;             /**< \brief rbf variance */
    double * w[NUM_CENTRES]; /**< \brief Locally weighted model parameters */
}LEARN_MODEL_LW_PI;

/** \brief This structure describes workspace for directly learning the policy model parameters using
    locally weighted linear regression.

    This structure defines the worspace variables and initialises with their dimensionalities.
    It should always initialised with ccl_learn_model_lw_pi_ws_alloc,
    and destroyed with ccl_learn_model_lw_pi_ws_free.
    \ingroup CCL_C
*/
typedef struct {
    gsl_vector* g;              /**< \brief Col vector of YPhit */
    gsl_matrix* Y_N;            /**< \brief Normalised Y */
    gsl_vector* YN_vec;         /**< \brief Vector view of Y_N */
    gsl_matrix* Y_Phit;         /**< \brief Dot product of Y * WPhi' */
    gsl_matrix* ones;           /**< \brief Matrix of all ones */
    gsl_matrix* Y_;             /**< \brief Output variable  */
    gsl_matrix* H;              /**< \brief Accumulated Hessian  */
    gsl_matrix* Phi;            /**< \brief Feature matrix  */
    gsl_vector* Phi_vec;        /**< \brief Feature vector  */
    gsl_matrix* Phi_vec_T;      /**< \brief Transpose of Feature vector  */
    gsl_matrix* YN_Phit;        /**< \brief Dot prodcut of YN * Phit */
    gsl_vector* YN_Phi_vec;     /**< \brief Vector view of YN_Phit */
    gsl_matrix* YN_Phi_vec_T;   /**< \brief Transpose of YN_Phi_vec */
    gsl_matrix* vv;             /**< \brief Dot product of v * v */
    gsl_matrix* WX_;            /**< \brief Dot product of W * X_ */
    gsl_vector* WX_row;         /**< \brief Row vector of WX_ */
    gsl_matrix* WPhi;           /**< \brief Dot product of W * Phi */
    gsl_matrix* WPhi_T;         /**< \brief Transpose of WPhi */
    gsl_matrix* pinvH1;         /**< \brief Peudo inverse of H1 */
    gsl_vector* V;              /**< \brief Eigen vector of H */
    gsl_vector* r;              /**< \brief Normalisation scaler for YN */
    gsl_matrix* r_rep;          /**< \brief Replication matrix of r */
    gsl_matrix* D;              /**< \brief Eigen values of H */
    int    * idx;               /**< \brief Index */
    gsl_vector* w_vec;          /**< \brief Vector view of model parameter w for each center */
    gsl_matrix* w_;             /**< \brief Model parameter w for each center */
    gsl_matrix* w_T;            /**< \brief Transpose of model parameter w for each center */
    gsl_matrix* w[NUM_CENTRES]; /**< \brief Model parameter w for all centers */
}LEARN_MODEL_LW_PI_WS;

/** \brief Main computation routine for learning linear policy

  \param[in] model     Must point to a valid LEARN_MODEL_PI structure
  \param[in] BX        Higher dimensionality of input variable, must point to an array of <em>dim_b  * dim_n<em> doubles
  \param[in] Y         Ouput variable, must point to an array of <em>dim_y * dim_n<em> doubles.
  \param[in] model     Must point to a valid LEARN_MODEL_PI structure
*/
void ccl_learn_policy_pi(LEARN_MODEL_PI *model, const double *BX, const double *Y);

/** \brief Allocates the workspace memory for directly learning policy pi.

  \param[in] model  Must point to a valid LEARN_MODEL_PI structure
  \param[in] ws     Must point to a valid LEARN_MODEL_PI_WS structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be allocated).
 */
int ccl_learn_model_pi_ws_alloc(LEARN_MODEL_PI *model,LEARN_MODEL_PI_WS* ws);

/** \brief Free the workspace memory for directly learning policy pi.

  \param[in] ws     Must point to a valid LEARN_MODEL_PI_WS structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be freed).
 */
int ccl_learn_model_pi_ws_free(LEARN_MODEL_PI_WS* ws);

/** \brief  Main computation routine for locally weighted linear policy.

  \param[in] model  Must point to a valid LEARN_MODEL_LW_PI structure
  \param[in] WX     Locally weighted inputs, must point to an array of <em>dim_b * dim_n<em> doubles
  \param[in] X      Inputs variables, must point to an array of <em>dim_x * dim_n<em> doubles
  \param[in] Y      Output variables, must point to an array of <em>dim_y * dim_n<em> doubles
  \param[out] model Must point to a valid LEARN_MODEL_LW_PI structure
*/
void ccl_learn_policy_lw_pi(LEARN_MODEL_LW_PI *model, const double *WX, const double *X, const double *Y);

/** \brief Allocates the memory for locally weighted linear policy LEARN_MODEL_LW_PI.

  \param[in] model  Must point to a valid LEARN_MODEL_LW_PI structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be allocated).
 */
int ccl_learn_policy_lw_pi_model_alloc(LEARN_MODEL_LW_PI *model);

/** \brief Free the memory for locally weighted linear policy LEARN_MODEL_LW_PI.

  \param[in] model  Must point to a valid LEARN_MODEL_LW_PI structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be freed).
 */
int ccl_learn_policy_lw_pi_model_free(LEARN_MODEL_LW_PI *model);

/** \brief Allocates the workspace memory for locally weighted linear policy LEARN_MODEL_LW_PI_WS.

  \param[in] model  Must point to a valid LEARN_MODEL_LW_PI structure
  \param[in] ws     Must point to a valid LEARN_MODEL_LW_PI_WS structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be allocated).
 */
int ccl_learn_model_lw_pi_ws_alloc(LEARN_MODEL_LW_PI *model,LEARN_MODEL_LW_PI_WS* ws);

/** \brief Free the workspace memory for locally weighted linear policy LEARN_MODEL_LW_PI_WS.

  \param[in] ws     Must point to a valid LEARN_MODEL_LW_PI_WS structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be freed).
 */
int ccl_learn_model_lw_pi_ws_free(LEARN_MODEL_LW_PI_WS* ws);

/** \brief Predictions based on linear policy model.

  \param[in] X           Inputs variables, must point to an array of <em>dim_x * dim_n<em> doubles
  \param[in] centres     Center of rbf, must point to an array of <em>1 * dim_b<em> doubles
  \param[in] variance    Variance of rbf
  \param[in] model       Must point to a valid LEARN_MODEL_PI structure
  \param[out] Yp         Predictions
 */
void predict_linear(const double* X, const double* centres,const double variance,const LEARN_MODEL_PI *model,double* Yp);

/** \brief Predictions based on locally weighted linear policy model.

  \param[in] X           Inputs variables, must point to an array of <em>dim_x * dim_n<em> doubles
  \param[in] centres     Center of rbf, must point to an array of <em>1 * dim_b<em> doubles
  \param[in] variance    Variance of rbf
  \param[in] model       Must point to a valid LEARN_MODEL_LW_PI structure
  \param[out] Yp         Predictions
 */
void predict_local_linear(const double* X, const double* centres,const double variance,const LEARN_MODEL_LW_PI *model,double* Yp);

/** \brief Read data from .txt file.

  \param[in] filename    Name of file
  \param[in] dim_x       Dimensionality of input variable
  \param[in] dim_n       Number of data samples
  \param[out] mat        Returned data matrix
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be freed).
 */
int ccl_read_data_from_file(char* filename, int dim_x, int dim_n, double* mat);

/** \brief Write locally weighted model to .txt file.

  \param[in] filename    Name of file
  \param[in] model       Must point to a valid LEARN_MODEL_LW_PI structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be freed).
 */
int ccl_write_lwmodel_to_file(char* filename, LEARN_MODEL_LW_PI* model);
#ifdef __cplusplus
}
#endif
#endif

