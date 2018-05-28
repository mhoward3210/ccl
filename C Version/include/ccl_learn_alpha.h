/*********************************************************************
CCL:A library for Constraint Consistent learning
Copyright (C) 2018 Matthew Howard
Contact:matthew.j.howard@kcl.ac.uk
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Library General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free
Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*********************************************************************/


/** \defgroup CCL_C CCL library (the low level C implementation) */

/** \file ccl_learn_alpha.h
    \brief CCL header file for learning state dependent constraint
    \ingroup CCL_C
*/

/** \mainpage
    <h2>Introduction</h2>
    Constraint Consistent Library (CCL) is a learning by demonstration
    algolrithm which allows the robot to understant the manipulation task
    constraint \f$A\f$, resolving nullspace resolutions and learning nullspace
    \f$u_{ns}\f$ ,and control policy \f$\pi\f$. Many real life applications which
    requires prioritising the task according to the constraints can be
    formulised in the following way:
    \f[
        u = A^{-1} x + (I-A^{-1} A)\pi
    \f]
    Example tasks are wiping a table, reaching a point in operation space,
    pouring water in the bottle and openning a door. For in-depth description
    of the CCL library, please see [1].

    This library started life as a Matlab-only implementation, whose subroutines
    were then successively trasformed into C. Therefore, the C library is very
    closely modelled after the CCL implementation in Matlab. Indeed, most Matlab
    functions have a C equivalent.

    This library has only one dependcy on the GSL library for resolving the linear
    algebra operations. The installations can be found in the instructions. Therefore,
    the data format of vectors and matrices used in this library is different from the Matlab
    (or Fortran). That is, vectors are just arrays of doubles. Matrices are also 1-D arrays
    of doubles, with the elements stored in column-major order.
    A 2x2 matrix
    \f[M = \left(\begin{array}{cc}
    m_{11} & m_{12} \\
    m_{21} & m_{22}
    \end{array}\right)\f]
    is thus stored as\code
         double M[4] = {m11,m12,m21,m22};
    \endcode

    [1] Yuchen Zhao and Matthew Howard, <em> A library for constraint consistent learning <em>
    RAL,2018.

    <h2>Documentation</h2>
    The documentations
    <li><a href="https://nms.kcl.ac.uk/rll/CCL_doc/index.html"><span>documentations</span></a></li>
    of the implemented CCL method can be found has both Matlab and C version.
    The naming convention for Matlab is using ccl_xx_xx, indicating the scope and usage of the implemented function.
    For instance, ccl_learna_alpha indicates this method is for learning constraint \f$A\f$ and using learn_alpha method.
    The low level C counterpart only provides the computation rountines, therefore not exatly following the same naming conventions.

    <h2>Installation</h2>
    The installation has been tested on windows 10 system and ubuntu 14.04 for matlab library. The C library has been tested on
    Ubuntu 14.04. A QT project has been setup for building the C library.

    To install the Matlab functions, just add the current path to the existing working path.

    To install the C functions, the following steps are required:
    (1) Install third pary library GSL using: apt-get install gsl-bin

    (2) export LD_LIBRARY_PATH="your system path"/ccl-1.0/src/.libs:$LD_LIBRARY_PATH to your system path before execute any program with depends on library.
        i.e., export LD_LIBRARY_PATH=~/Desktop/ccl-1.0/src/.libs:$LD_LIBRARY_PATH

    (3) Go to project root "your system path"/ccl-1.0

        i)  autoreconf --install

        ii) ./configure CFLAGS="-ggdb3 -O0" CXXFLAGS="-ggdb3 -O0" LDFLAGS="-ggdb3"

        iii) make all

     TO use the QT project, just select the "debug" (there are other builds i.e. debug2) build which will do the above
     steps automatically.

    <h2>Getting Start</h2>
    For starters, Matlab users can load the demos: "demo_2_link_arm.m", "demo_toy_exaple_2D.m" and "demo_with_real_data.m".
    C users can load the binary program named "ccl" which will automatically read the saved data in the project and execute
    the testing functions.
*/

#ifndef __CCL_LEARN_ALPHA_H
#define __CCL_LEARN_ALPHA_H

#include <ccl_math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
/** NUM_CONSTRAINT is the default full rank of the task constraint */
#define NUM_CONSTRAINT 3
#ifdef __cplusplus
extern "C" {
#endif
/** \brief This structure describes a "LEARN_A_MODEL" (a learn alpha model).
    This structure constains the dimentionality of the defined problems and
    the model parameters. It should always initialised with ccl_learn_alpha_model_alloc,
    and destroyed with ccl_learn_alpha_model_free.
    \ingroup CCL_C
*/
typedef struct{
    int dim_b;                  /**< \brief The number of basis functions */
    int dim_r;                  /**< \brief The dimensionality of the task space */
    int dim_x;                  /**< \brief The dimensionality of the state space */
    int dim_u;                  /**< \brief The dimensionality of the action space */
    int dim_t;                  /**< \brief Number of parameters needed to represent an unit vector */
    int dim_k;                  /**< \brief The dimensionality of the constraints */
    int dim_n;                  /**< \brief The number of data points */
    double var;                 /**< \brief The variance of the Un */
    double nmse;                /**< \brief The normalised mean square error of the model */
    double *w[NUM_CONSTRAINT];  /**< \brief The model parameters for the learned constraints*/
    double * c;                 /**< \brief The mean of the rbf */
    double  s2;                 /**< \brief The variance of the rbf */
} LEARN_A_MODEL;

/** \brief This structure defines the OPTION for the optimizer.

    This structure specifies the tolerences of the objective function and the learning parameters.
    \ingroup CCL_C
*/
typedef struct{
    int MaxIter;
    double Tolfun;
    double Tolx;
    double Jacob;
} OPTION;

/** \brief This structure defines the workspace variables for solving the non-linear LM optimization

    This structure containts the memory of the workspace variables for sovling the non-linear LM
    optimization problem. It should always initialise with ccl_solve_lm_ws_alloc and destroy with
    ccl_solve_lm_ws_free.
    \ingroup CCL_C
*/
typedef struct{
    int      dim_x;             /**< \brief The dimensionality of the state variable */
    int      dim_u;             /**< \brief The dimensionality of the action space */
    int      dim_n;             /**< \brief The number of data points */
    int      dim_b;             /**< \brief The number of basis functions */
    int      dim_k;             /**< \brief The dimensionality of the constraints */
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
    int      r_ok;              /**< \brief check if the objective function is belowed the tolerence */
    int      x_ok;              /**< \brief check if the model parameters are belowed the tolerence */
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
} SOLVE_LM_WS;

/** \brief Allocates the memory for the learn_alpha model.

  \param[in] model Must point to a valid LEARN_A_MODEL structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be allocated)
 */
int ccl_learn_alpha_model_alloc(LEARN_A_MODEL *model);

/** \brief Free the memory for the learn_alpha model.

  \param[in] model Must point to a valid LEARN_A_MODEL structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be freed)
 */
int ccl_learn_alpha_model_free(LEARN_A_MODEL *model);

/** \brief Allocates the workspace memory for the LM solver.

  \param[in] model  Must point to a valid LEARN_A_MODEL structure
  \param[in] lm_ws  Must point to a valid lm workspace structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be allocated).
 */
int ccl_solve_lm_ws_alloc(const LEARN_A_MODEL *model,SOLVE_LM_WS * lm_ws);

/** \brief Free the memory for the LM solver.

  \param[in] lm_ws  Must point to a valid lm workspace structure
  \return
    - 1 in case of succes
    - 0 in case of failure (e.g. memory could not be freed).
 */
int ccl_solve_lm_ws_free(SOLVE_LM_WS * lm_ws);

/** \brief Main computation routine for learning state dependent constraint A.

  \param[in] Un      Observed actions, Must point to an array of <em>dim_u * dim_n<em> of doubles
  \param[in] X       Observed state variables, Must point to an array of <em>dim_x * dim_n<em> of doubles
  \param[in] dim_b   Number of basis functions
  \param[in] dim_r   Dimensionality of the task space
  \param[in] dim_n   NUmber of data points
  \param[in] dim_x   Dimensionality of the state variables
  \param[in] dim_u   Dimensionality of the action space
  \param[out] optimal, optimal model paramters
 */
void ccl_learn_alpha(const double * Un,const double *X,const int dim_b,const int dim_r,const int dim_n,const int dim_x,const int dim_u,LEARN_A_MODEL optimal);

/** \brief Main computation routine for learning a model of A.

  \param[in] BX     Transformed state variables, must point to an array of <em>dim_b * dim_n<em> of doubles
  \param[in] RnUn   Pre-calculated dot product of Rn and Un, must point to an array of <em>dim_u * dim_n<em> of doubles
  \param[in] model  model paramters, must point to a valid LEARN_A_MODEL structure
  \param[out] model model paramters, must point to a valid LEARN_A_MODEL structure
 */
void search_learn_alpha(const double *BX,const double *RnUn, LEARN_A_MODEL* model);
/** \brief Computation routine for learning constraint by increasing the dimensionality of k.

  \param[in] model   Must be pointer to a valid LEARN_A_MODEL structure.
  \param[in] W       Current model parameter, must point to an array of <em>dim_u-dum_k<em> doubles.
  \param[in] RnUn    Pre-calculated dot product of Rn and Un, must point to an array of <em>dim_u * dim_n<em> of doubles
  \param[out]fun_out Residual value returned from the objective function, must point to a scalar of double
*/
void obj_AUn (const LEARN_A_MODEL* model, const double* W, const double* BX,const double * RnUn,double* fun_out);

/** \brief Computation routine for generating unit vector for row vector alpha by given theta anlges

  \param[in] theta    Learned constraint parameters, must point to an array of <em>dim_n * dim_t<em> doubles
  \param[in] dim_n    Number of data samples
  \param[in] dim_t    Number of constraint parameters
  \param[out]alpha    A uit vector of constraint basis, must point to an array of <em>dim_n * dim_u<em> doubles
*/
void ccl_get_unit_vector_from_matrix(const double *theta, int dim_n, int dim_t, double *alpha);

/** \brief Computation routine for optimizing non-linear objective functions using LM approach.

  \param[in] model       Must be pointer to a valid LEARN_A_MODEL structure.
  \param[in] RnUn        Pre-calculated dot product of Rn and Un, must point to an array of <em>dim_u * dim_n<em> of doubles
  \param[in] BX          Transformed state variables, must point to an array of <em>dim_b * dim_n<em> of doubles
  \param[in] option      Options for optimizer
  \param[in] lm_ws_param Must point to a valid structure of SOLVE_LM_WS for LM optimizer workspace variables
  \param[out] W          Returned model parameter, must point to an array of <em>dim_u-dum_k<em> doubles
*/
void ccl_solve_lm(const LEARN_A_MODEL* model,const  double* RnUn,const  double* BX, const OPTION option,SOLVE_LM_WS * lm_ws_param, double* W);

/** \brief Computation routine for numerical approximation of the objevtive jacobian matrix.

  \param[in] model       Must be pointer to a valid LEARN_A_MODEL structure
  \param[in] dim_x       Dimensionality of the model parameters
  \param[in] BX          Transformed state variables, must point to an array of <em>dim_b * dim_n<em> of doubles
  \param[in] RnUn        Pre-calculated dot product of Rn and Un, must point to an array of <em>dim_u * dim_n<em> of doubles
  \param[in] y           Residual calculated at x, must point to a scalar of double.
  \param[in] x           Current model parameter x, must point to an array of <em>dim_x<em> doubles, note here dim_x is not the dimension of the state variable
  \param[in] epsx        Tolerance of the model parameters
  \param[out] J          Returned jocabian matrix, must point to an array of <em>dim_n * dim_x<em> doubles
*/
void findjac(const LEARN_A_MODEL* model, const int dim_x,const double* BX, const double * RnUn,const double *y,const double*x,double epsx,double* J);

/** \brief Computation routine for rotation matrix after finding the k^th constraint vector.

  \param[in] theta       Theta from kth constrains, must point to an arrary of <em>1 * dim_t<em> doubles
  \param[in] currentRn   Rotation matrix from the last iteration, must point to an array of <em>dim_u * dim_u<em> of doubles
  \param[in] model       Must be pointer to a valid LEARN_A_MODEL structure
  \param[in] alpha_id    Current searching dimension of alpha
  \param[out] Rn         Returned rotation matrix, must point to an array of <em>dim_u * dum_u<em> doubles
*/
void ccl_get_rotation_matrix(const double*theta,const double*currentRn,const LEARN_A_MODEL* model,const int alpha_id,double*Rn);

/** \brief Computation routine for generating roation matrix of a plane rotation of degree theta in an arbitrary plane and dimension R

  \param[in] theta       Degree of rotation, must point to a scalar doubles
  \param[in] i           Row index of matrix
  \param[in] j           Col index of matrix
  \param[in] dim         Dimensionality of the rotation matrix
  \param[out] G          Returned rotation matrix, must point to an array of <em>dim_u * dum_u<em> doubles
*/
void ccl_make_given_matrix(const double theta,int i,int j,int dim,double*G);

/** \brief Computation routine for prediction of the A matrix

  \param[in] x           Input state variable, must point to an array of <em>dim_x<em> doubles
  \param[in] model       Must be pointer to a valid LEARN_A_MODEL structure
  \param[in] centres     Rbf centers, must point to an array of <em>1 * dim_b<em> doubles.
  \param[in] variance    Variance of the rbf
  \param[in] Iu          Identity matrix, must point to an array of <em>dim_u * dim_u<em> doubles
  \param[out] A          Constraint A matrix, must point to an array of <em>dim_k * dum_u<em> doubles
*/
void predict_proj_alpha(double* x, LEARN_A_MODEL* model,double* centres,double variance,double* Iu, double*A);
#ifdef __cplusplus
}
#endif
#endif

