#include <ccl_learn_lambda.h>
void Jacobian(const double* X, const int size, double *out){
// Three link arn
    out[0] = -sin(X[0])-sin(X[0]+X[1])-sin(X[0]+X[1]+X[2]);
    out[1] = -sin(X[0]+X[1])-sin(X[0]+X[1]+X[2]);
    out[2] = -sin(X[0]+X[1]+X[2]);

    out[3] = cos(X[0])+cos(X[0]+X[1])+cos(X[0]+X[1]+X[2]);
    out[4] = cos(X[0]+X[1])+cos(X[0]+X[1]+X[2]);
    out[5] = cos(X[0]+X[1]+X[2]);

//    out[6] = 1;
//    out[7] = 1;
//    out[8] = 1;
// TWO ARM ROBOT
//    out[0] = -sin(X[0])-sin(X[0]+X[1]);
//    out[1] = -sin(X[0]+X[1]);
//    out[2] = cos(X[0])+cos(X[0]+X[1]);
//    out[3] = cos(X[0]+X[1]);
}

int ccl_learn_lambda_model_alloc(LEARN_A_MODEL *model){
    int i;
    for (i=0;i<NUM_CONSTRAINT;i++){
        model->w[i]  = malloc(model->dim_r*model->dim_b*sizeof(double*));
    }
}
int ccl_learn_lambda_model_free(LEARN_A_MODEL *model){
    int k;
    for (k=0;k<NUM_CONSTRAINT;k++){
        free(model->w[k]);
    }
}
void ccl_learn_lambda(const double * Un,const double *X,void (*J_func)(const double*,const int,double*),const int dim_b,const int dim_r,const int dim_n,const int dim_x,const int dim_u,LEARN_A_MODEL optimal){
    LEARN_A_MODEL model;
    double * centres,*var_tmp,*vec, *BX, *RnVn,*Rn,*Vn;
    double variance;
    int i,lambda_id;
    gsl_matrix *Un_,*X_;

    //     J_fun       = &Jacobian;
    model.dim_b = dim_b;
    model.dim_r = dim_r;
    model.dim_x = dim_x;
    model.dim_n = dim_n;
    model.dim_t = dim_r-1;
    model.dim_u = dim_u;
    // calculate model.var
    var_tmp = malloc(dim_u*sizeof(double));
    ccl_mat_var(Un,dim_u,dim_n,0,var_tmp);
    model.var = ccl_vec_sum(var_tmp,dim_u);
    free(var_tmp);

    Un_     = gsl_matrix_alloc(dim_u,dim_n);
    memcpy(Un_->data,Un,dim_u*dim_n*sizeof(double));
    X_ = gsl_matrix_alloc(dim_x,dim_n);
    memcpy(X_->data,X,dim_x*dim_n*sizeof(double));


    optimal.nmse = 1000000;

    gsl_vector* X_col = gsl_vector_alloc(dim_x);
    gsl_vector* Un_col = gsl_vector_alloc(dim_u);
    gsl_matrix* Vn_container = gsl_matrix_alloc(dim_r,dim_n);
    gsl_matrix_set_zero(Vn_container);
    double*     norm   = malloc(dim_n*sizeof(double));
    double*     J_x    = malloc(dim_r*dim_u*sizeof(double));
    gsl_vector* Jx_Un  = gsl_vector_alloc(dim_r);
    int*     id_keep   = malloc(dim_n*sizeof(double));
    for (i=0;i<dim_n;i++){
        gsl_matrix_get_col(X_col,X_,i);
        gsl_matrix_get_col(Un_col,Un_,i);
        J_func(X_col->data,dim_x,J_x);
        ccl_dot_product(J_x,dim_r,dim_u,Un_col->data,dim_u,1,Jx_Un->data);
        gsl_matrix_set_col(Vn_container,i,Jx_Un);
        norm[i] = gsl_blas_dnrm2(Jx_Un);
    }
    // here dim_n maybe change.
    int new_dim_n = ccl_find_index_double(norm,dim_n,2,1E-3,id_keep);
    model.dim_n = new_dim_n;

    Vn = malloc(dim_r*model.dim_n*sizeof(double));
    gsl_matrix Vn_ = gsl_matrix_view_array(Vn,dim_r,model.dim_n).matrix;
    gsl_matrix*X_new = gsl_matrix_alloc(dim_x,model.dim_n);
    gsl_matrix*Un_new = gsl_matrix_alloc(dim_u,model.dim_n);
    for (i=0;i<new_dim_n;i++){
        gsl_vector* Vn_tmp = gsl_vector_alloc(dim_r);
        gsl_matrix_get_col(Vn_tmp,Vn_container,id_keep[i]);
        gsl_matrix_set_col(&Vn_,i,Vn_tmp);
        gsl_vector_free(Vn_tmp);
        Vn_tmp = gsl_vector_alloc(dim_x);
        gsl_matrix_get_col(Vn_tmp,X_,id_keep[i]);
        gsl_matrix_set_col(X_new,i,Vn_tmp);
        gsl_vector_free(Vn_tmp);
        Vn_tmp = gsl_vector_alloc(dim_u);
        gsl_matrix_get_col(Vn_tmp,Un_,id_keep[i]);
        gsl_matrix_set_col(Un_new,i,Vn_tmp);
        gsl_vector_free(Vn_tmp);
    }
    gsl_vector_free(Jx_Un);
    gsl_vector_free(X_col);
    gsl_vector_free(Un_col);
    gsl_matrix_free(Vn_container);
    gsl_matrix_free(Un_);
    free(J_x);
    free(norm);
    free(id_keep);

    //  prepare for BX
    centres = malloc(dim_x*dim_b*sizeof(double));
    generate_kmeans_centres(X_new->data,dim_x,model.dim_n,dim_b,centres);
    var_tmp = malloc(dim_b*dim_b*sizeof(double));
    vec     = malloc(dim_b*sizeof(double));
    ccl_mat_distance(centres,dim_x,dim_b,centres,dim_x,dim_b,var_tmp);

    for (i=0;i<dim_b*dim_b;i++){
        var_tmp[i] = sqrt(var_tmp[i]);
    }
    ccl_mat_mean(var_tmp,dim_b,dim_b,1,vec);
    variance = pow(gsl_stats_mean(vec,1,dim_b),2);
    BX = malloc(dim_b*model.dim_n*sizeof(double));
    ccl_gaussian_rbf(X_new->data,dim_x,model.dim_n,centres,dim_x,dim_b,variance,BX);
    gsl_matrix_free(X_new);
    gsl_matrix_free(Un_new);
    free(var_tmp);
    free(vec);

    Rn = malloc(dim_r*dim_r*sizeof(double));
    gsl_matrix Rn_ = gsl_matrix_view_array(Rn,dim_r,dim_r).matrix;
    gsl_matrix_set_identity(&Rn_);

    RnVn = malloc(dim_r*model.dim_n*sizeof(double));
    memcpy(RnVn,Vn,dim_r*model.dim_n*sizeof(double));
    model.dim_u = model.dim_r;
    ccl_learn_alpha_model_alloc(&model);
    memcpy(model.c,centres,model.dim_x*model.dim_b*sizeof(double));
    model.s2 = variance;
    clock_t t = clock();
    for(lambda_id=0;lambda_id<dim_r;lambda_id++){
        model.dim_k = lambda_id+1;
        //        model.w[lambda_id] = malloc((dim_u-model.dim_k)*dim_b*sizeof(double*));
        if(dim_r-model.dim_k==0){
            model.dim_k = lambda_id;
            break;
        }
        else{
            search_learn_alpha(BX,RnVn,&model);
            double* theta = malloc(model.dim_n*model.dim_t*sizeof(double));
            double *W_BX = malloc((dim_r-model.dim_k)*model.dim_n*sizeof(double));
            double *W_BX_T = malloc(model.dim_n*(dim_r-model.dim_k)*sizeof(double));
            ccl_dot_product(model.w[lambda_id],dim_r-model.dim_k,dim_b,BX,dim_b,model.dim_n,W_BX);
            ccl_mat_transpose(W_BX,dim_r-model.dim_k,model.dim_n,W_BX_T);
            if (model.dim_k ==1){
                memcpy(theta,W_BX_T,model.dim_n*(dim_r-model.dim_k)*sizeof(double));
            }
            else{
                gsl_matrix* ones = gsl_matrix_alloc(model.dim_n,model.dim_k-1);
                gsl_matrix_set_all(ones,1);
                gsl_matrix_scale(ones,M_PI/2);
                mat_hotz_app(ones->data,model.dim_n,model.dim_k-1,W_BX_T,model.dim_n,dim_r-model.dim_k,theta);

                gsl_matrix_free(ones);
            }
            for(i=0;i<model.dim_n;i++){
                gsl_matrix theta_mat = gsl_matrix_view_array(theta,model.dim_n,model.dim_t).matrix;
                gsl_vector *vec      = gsl_vector_alloc(model.dim_t);
                gsl_matrix_get_row(vec,&theta_mat,i);
                ccl_get_rotation_matrix(vec->data,Rn,&model,lambda_id,Rn);
                gsl_vector_free(vec);
                vec                  = gsl_vector_alloc(dim_r);
                gsl_matrix_get_col(vec,&Vn_,i);
                ccl_dot_product(Rn,dim_r,dim_r,vec->data,dim_r,1,vec->data);
                gsl_matrix RnVn_     = gsl_matrix_view_array(RnVn,dim_r,model.dim_n).matrix;
                gsl_matrix_set_col(&RnVn_,i,vec);
                gsl_vector_free(vec);
            }
            if(model.nmse > optimal.nmse && model.nmse > 1E-5){
                model.dim_k = lambda_id;
                printf("\n I am out...\n");//optimal;
                break;
            }
            else{
                printf("\n copy model -> optimal\n");//optimal;
            }
            free(W_BX);
            free(W_BX_T);
            free(theta);
        }
    }
    t = clock()-t;
    printf("\n learning lambda used: %f second \n",((float)t)/CLOCKS_PER_SEC);
    double* A = malloc(model.dim_k*model.dim_r*sizeof(double));
    gsl_matrix* Iu = gsl_matrix_alloc(model.dim_r,model.dim_r);
    gsl_matrix_set_identity(Iu);
    gsl_vector* x = gsl_vector_alloc(model.dim_x);
    gsl_matrix_get_col(x,X_,5);
//    predict_proj_lambda(x->data, model,Jacobian,centres,variance,Iu->data,A);
//    print_mat_d(A,model.dim_k,dim_r);
    ccl_write_learn_lambda_model("/home/yuchen/Desktop/ccl-1.1.0/data/learn_lambda_model_l.txt", &model);
    free(A);
    gsl_matrix_free(Iu);
    gsl_vector_free(x);
    free(Vn);
    free(Rn);
    free(BX);
    free(RnVn);
    free(centres);
    gsl_matrix_free(X_);
    ccl_learn_alpha_model_free(&model);
}
void predict_proj_lambda(double* x, LEARN_A_MODEL model,void (*J_func)(const double*,const int,double*),double* centres,double variance,double* Iu, double*A){
    gsl_matrix* Rn = gsl_matrix_alloc(model.dim_r,model.dim_r);
    memcpy(Rn->data,Iu,model.dim_r*model.dim_r*sizeof(double));
    gsl_matrix* lambda = gsl_matrix_alloc(model.dim_k,model.dim_r);
    gsl_matrix_set_all(lambda,0);
    int k;
    double * BX, *W_BX,*W_BX_T,*theta,*alpha,*J_x;
    BX = malloc(model.dim_b*1*sizeof(double));
    theta = malloc(1*model.dim_t*sizeof(double));
    alpha = malloc(1*model.dim_r*sizeof(double));
    gsl_vector* lambda_vec = gsl_vector_alloc(model.dim_r);
    for (k=1;k<model.dim_k+1;k++){
        W_BX = malloc((model.dim_u-k)*1*sizeof(double));
        W_BX_T = malloc(1*(model.dim_u-k)*sizeof(double));
        ccl_gaussian_rbf(x,model.dim_x,1,centres,model.dim_x,model.dim_b,variance,BX);
        ccl_dot_product(model.w[k-1],model.dim_u-k,model.dim_b,BX,model.dim_b,1,W_BX);
        ccl_mat_transpose(W_BX,model.dim_u-k,1,W_BX_T);
        free(W_BX);
        if (k ==1){
            memcpy(theta,W_BX_T,1*(model.dim_u-k)*sizeof(double));
            free(W_BX_T);
        }
        else{
            gsl_matrix* ones = gsl_matrix_alloc(1,k);
            gsl_matrix_set_all(ones,1);
            gsl_matrix_scale(ones,M_PI/2);
            mat_hotz_app(ones->data,1,k,W_BX_T,model.dim_n,model.dim_u-k,theta);
            free(W_BX_T);
            gsl_matrix_free(ones);
        }
        ccl_get_unit_vector_from_matrix(theta,1,model.dim_t,alpha);
        ccl_dot_product(alpha,k,model.dim_r,Rn->data,model.dim_r,model.dim_r,lambda_vec->data);
        gsl_matrix_set_row(lambda,k-1,lambda_vec);
        ccl_get_rotation_matrix(theta,Rn->data,&model,k-1,Rn->data);
    }
    memcpy(A,lambda->data,model.dim_k*model.dim_r*sizeof(double));
    J_x = malloc(model.dim_r*model.dim_x*sizeof(double));
    //    J_func(x,model.dim_x,J_x);
    //    print_mat_d(alpha,1,model.dim_r);
    //    ccl_dot_product(lambda->data,model.dim_k,model.dim_r,J_x,model.dim_r,model.dim_x,A);
    free(BX);
    free(theta);
    free(alpha);
    free(J_x);
    gsl_vector_free(lambda_vec);
    gsl_matrix_free(lambda);
    gsl_matrix_free(Rn);
}
int ccl_write_learn_lambda_model(char* filename, LEARN_A_MODEL *model){
    int i,j,c,k;
        FILE *file;
        file=fopen(filename, "w");   // extension file doesn't matter
        if(!file) {
            printf("File not found! Exiting...\n");
            return -1;
        }
        // Mu
        c = 0;
        for (i =0;i<model->dim_x;i++){
            for (j=0;j<model->dim_b;j++){
                fprintf(file,"%lf, ",model->c[c]);
                if (j==model->dim_b-1) fprintf(file,"\n");
                c++;
            }
        }
        //sigma
        fprintf(file,"%lf, ",model->s2);
        fprintf(file,"\n");
        // w
        for (k=0;k<model->dim_k;k++){
            c = 0;
            double* tmp = malloc((model->dim_r-k-1)*model->dim_b*sizeof(double));
            for(i = 0; i < model->dim_r-k-1; i++)
            {
                for(j = 0; j < model->dim_b; j++)
                {
                    memcpy(tmp,model->w[k],(model->dim_r-k-1)*model->dim_b*sizeof(double));
                    fprintf(file,"%lf, ",tmp[c]);
                    if (j==model->dim_b-1) fprintf(file,"\n");
                    c++;
                }
            }
            free(tmp);
        }
        fclose(file);
}
