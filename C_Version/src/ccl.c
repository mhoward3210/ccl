#include <stdio.h>
#include <ccl_math.h>
#include <ccl_learn_nhat.h>
#include <ccl_learn_alpha.h>
#include <ccl_learn_lambda.h>
#include <ccl_learn_ncl.h>
#include <ccl_learn_policy.h>
#include <math.h>
#include <time.h>

int
main (void)
{
    //// Learn nullspace resolution Nsp
        int dim_x = 3;
        int dim_u = 3;
        int dim_y = 3;
        int dim_b = 20;
        int dim_r = 2;
        int dim_n = 800;
        int i;

        double *X = malloc(dim_x*dim_n*sizeof(double));
        ccl_read_data_from_file("/home/yuchen/Desktop/ccl-1.0/data/X_n.txt",dim_x,dim_n,X);
        double *Y = malloc(dim_x*dim_n*sizeof(double));
        ccl_read_data_from_file("/home/yuchen/Desktop/ccl-1.0/data/Y_n.txt",dim_y,dim_n,Y);
        gsl_matrix X_ = gsl_matrix_view_array(X,dim_x,dim_n).matrix;
        gsl_matrix Y_ = gsl_matrix_view_array(Y,dim_u,dim_n).matrix;

        LEARN_NCL_MODEL model_ncl;
        ccl_learn_ncl(X,Y,dim_x,dim_y,dim_n,dim_b,&model_ncl);
        free(X);
        free(Y);

//        //// learn nullspace projection
        dim_u = 3;
        dim_x = 3;
        dim_y = 3;
        dim_b = 20;
        dim_r = 2;
        dim_n = 1000;
        X = malloc(dim_x*dim_n*sizeof(double));
        ccl_read_data_from_file("/home/yuchen/Desktop/ccl-1.0/data/X_l_p.txt",dim_x,dim_n,X);
        Y = malloc(dim_x*dim_n*sizeof(double));
        ccl_read_data_from_file("/home/yuchen/Desktop/ccl-1.0/data/Y_l_p.txt",dim_y,dim_n,Y);
        LEARN_A_MODEL optimal;
        ccl_learn_lambda(Y,X,Jacobian,dim_b,dim_r,dim_n,dim_x,dim_u,optimal);
        free(X);
        free(Y);


    //// learn nullspace policy pi
    dim_x = 3;
    dim_y = 3;
    dim_b = 20;
    dim_n = 780;
    X = malloc(dim_x*dim_n*sizeof(double));
    ccl_read_data_from_file("/home/yuchen/Desktop/ccl-1.0/data/X_pi.txt",dim_x,dim_n,X);
    Y = malloc(dim_x*dim_n*sizeof(double));
    ccl_read_data_from_file("/home/yuchen/Desktop/ccl-1.0/data/Y_pi.txt",dim_y,dim_n,Y);
    //  prepare for BX
    X_ = gsl_matrix_view_array(X,dim_x,dim_n).matrix;
    Y_ = gsl_matrix_view_array(Y,dim_y,dim_n).matrix;
    double* centres = malloc(dim_x*dim_b*sizeof(double));
    generate_kmeans_centres(X_.data,dim_x,dim_n,dim_b,centres);
    double* var_tmp = malloc(dim_b*dim_b*sizeof(double));
    double* vec     = malloc(dim_b*sizeof(double));
    ccl_mat_distance(centres,dim_x,dim_b,centres,dim_x,dim_b,var_tmp);

    for (i=0;i<dim_b*dim_b;i++){
        var_tmp[i] = sqrt(var_tmp[i]);
    }
    ccl_mat_mean(var_tmp,dim_b,dim_b,1,vec);
    double variance = pow(gsl_stats_mean(vec,1,dim_b),2);
    double *WX = malloc(dim_b*dim_n*sizeof(double));
    ccl_gaussian_rbf(X,dim_x,dim_n,centres,dim_x,dim_b,variance,WX);
    LEARN_MODEL_LW_PI model;
    model.dim_b =dim_b;
    model.dim_n = dim_n;
    model.dim_phi = dim_x+1;
    model.dim_x = dim_x;
    model.dim_y = dim_y;
    clock_t t = clock();
    ccl_learn_policy_lw_pi_model_alloc(&model);
    t = clock()-t;
    printf("\n learning Pi used: %f second \n",((float)t)/CLOCKS_PER_SEC);
    memcpy(model.c,centres,model.dim_x*model.dim_b*sizeof(double));
    model.s2 = variance;
    ccl_learn_policy_lw_pi(&model,WX,X_.data,Y_.data);
    double* Yp = malloc(dim_y*dim_n*sizeof(double));
    predict_local_linear(X_.data,centres,variance,&model,Yp);
//    print_mat_d(Yp,dim_y,dim_n);
    ccl_write_lwmodel_to_file("/home/yuchen/Desktop/ccl-1.0/data/learn_lwpi_model.txt", &model);
    ccl_learn_policy_lw_pi_model_free(&model);
    free(X);
    free(Y);
    free(Yp);
    free(centres);
    free(WX);
    free(var_tmp);
    free(vec);

    return 0;
}
