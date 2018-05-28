#include <ccl_learn_alpha.h>
int ccl_learn_alpha_model_alloc(LEARN_A_MODEL *model){
    int i;
    for (i=0;i<NUM_CONSTRAINT;i++){
        model->w[i]  = malloc(model->dim_u*model->dim_b*sizeof(double*));
    }
    model->c = malloc(model->dim_x*model->dim_b*sizeof(double));
}
int ccl_learn_alpha_model_free(LEARN_A_MODEL *model){
    int k;
    for (k=0;k<NUM_CONSTRAINT;k++){
        free(model->w[k]);
    }
    free(model->c);
}
int ccl_solve_lm_ws_alloc(const LEARN_A_MODEL *model,SOLVE_LM_WS * lm_ws){
    int num_w_param;
    num_w_param = (model->dim_u-model->dim_k)*model->dim_b;
    lm_ws->A = malloc(num_w_param*num_w_param*sizeof(double));
    lm_ws->D = malloc(num_w_param*num_w_param*sizeof(double));
    lm_ws->epsx = malloc(num_w_param*sizeof(double));
    lm_ws->J = malloc(model->dim_n*num_w_param*sizeof(double));
    lm_ws->r = malloc(model->dim_n*sizeof(double));
    lm_ws->rd = malloc(model->dim_n*sizeof(double));
    lm_ws->v = malloc(num_w_param*sizeof(double));
    lm_ws->x = malloc(num_w_param*sizeof(double));
    lm_ws->xc= malloc(num_w_param*sizeof(double));
    lm_ws->xd= malloc(num_w_param*sizeof(double));
    lm_ws->xf= malloc(num_w_param*sizeof(double));
    lm_ws->d= malloc(num_w_param*sizeof(double));
    lm_ws->dim_b = model->dim_b;
    lm_ws->dim_u = model->dim_u;
    lm_ws->dim_n = model->dim_n;
    lm_ws->dim_x = num_w_param;
    lm_ws->dim_k = model->dim_k;
    lm_ws->d_T= malloc(lm_ws->dim_x*sizeof(double));
    lm_ws->J_T = malloc(lm_ws->dim_n*lm_ws->dim_x*sizeof(double));
    lm_ws->r_T = gsl_vector_alloc(lm_ws->dim_n);
    lm_ws->A_inv = gsl_matrix_alloc(lm_ws->dim_x,lm_ws->dim_x);
    lm_ws->A_inv_diag = gsl_vector_alloc(lm_ws->dim_x);
    lm_ws->A_d = gsl_vector_alloc(lm_ws->dim_x);
    lm_ws->tmp = malloc(lm_ws->dim_x*sizeof(double));
    lm_ws->rd_T = malloc(lm_ws->dim_n*sizeof(double));
    lm_ws->D_pinv   = gsl_matrix_alloc(lm_ws->dim_x,lm_ws->dim_x);
}
int ccl_solve_lm_ws_free(SOLVE_LM_WS * lm_ws){
    free(lm_ws->A);
    free(lm_ws->D);
    free(lm_ws->epsx);
    free(lm_ws->J);
    free(lm_ws->r);
    free(lm_ws->rd);
    free(lm_ws->v);
    free(lm_ws->x);
    free(lm_ws->xc);
    free(lm_ws->xd);
    free(lm_ws->xf);
    free(lm_ws->d);
    free(lm_ws->d_T);
    free(lm_ws->J_T);
    free(lm_ws->tmp);
    free(lm_ws->rd_T);
    gsl_vector_free(lm_ws->r_T);
    gsl_vector_free(lm_ws->A_d);
    gsl_vector_free(lm_ws->A_inv_diag);
    gsl_matrix_free(lm_ws->A_inv);
    gsl_matrix_free(lm_ws->D_pinv);
}
void ccl_learn_alpha(const double * Un,const double *X,const int dim_b,const int dim_r,const int dim_n,const int dim_x,const int dim_u,LEARN_A_MODEL optimal){
    LEARN_A_MODEL model;
    model.dim_b = dim_b;
    model.dim_r = dim_r;
    model.dim_x = dim_x;
    model.dim_n = dim_n;
    model.dim_t = dim_u-1;
    model.dim_u = dim_u;
    double * centres,*var_tmp,*vec, *BX, *RnUn,* fun;
    double variance;
    int i,alpha_id;
    gsl_matrix * Iu,*Un_;
    Un_     = gsl_matrix_alloc(dim_u,dim_n);
    memcpy(Un_->data,Un,dim_u*dim_n*sizeof(double));

    centres = malloc(dim_x*dim_b*sizeof(double));
    generate_kmeans_centres(X,dim_x,dim_n,dim_b,centres);
    var_tmp = malloc(dim_b*dim_b*sizeof(double));
    vec     = malloc(dim_b*sizeof(double));
    ccl_mat_distance(centres,dim_x,dim_b,centres,dim_x,dim_b,var_tmp);
    //    print_mat_d(var_tmp,dim_b,dim_b);

    for (i=0;i<dim_b*dim_b;i++){
        var_tmp[i] = sqrt(var_tmp[i]);
    }
    ccl_mat_mean(var_tmp,dim_b,dim_b,1,vec);
    variance = pow(gsl_stats_mean(vec,1,dim_b),2);
    BX = malloc(dim_b*dim_n*sizeof(double));
    ccl_gaussian_rbf(X,dim_x,dim_n,centres,dim_x,dim_b,variance,BX);
    optimal.nmse = 1000000;
    free(var_tmp);
    var_tmp = malloc(dim_u*sizeof(double));
    ccl_mat_var(Un,dim_u,dim_n,0,var_tmp);
    model.var = ccl_vec_sum(var_tmp,dim_u);
    free(var_tmp);
    double **Rn;
    Rn = malloc(dim_n*dim_u*dim_u*sizeof(double));
//    double** Rn = calloc(dim_n,dim_u*dim_u*sizeof(double));
    for (i=0;i<dim_n;i++){
        Rn[i] = malloc(dim_u*dim_u*sizeof(double));
        gsl_matrix Rn_ = gsl_matrix_view_array(Rn[i],dim_u,dim_u).matrix;
        gsl_matrix_set_identity(&Rn_);
    }
    RnUn = malloc(dim_u*dim_n*sizeof(double));
    memcpy(RnUn,Un,dim_u*dim_n*sizeof(double));
    Iu = gsl_matrix_alloc(dim_u,dim_u);
    gsl_matrix_set_identity(Iu);
    ccl_learn_alpha_model_alloc(&model);
    memcpy(model.c,centres,model.dim_x*model.dim_b*sizeof(double));
    model.s2 = variance;
    for(alpha_id=0;alpha_id<dim_r;alpha_id++){
        model.dim_k = alpha_id+1;
        if(dim_u-model.dim_k==0){
            model.dim_k = alpha_id;
            break;
        }
        else{
            search_learn_alpha(BX,RnUn,&model);
            double* theta = malloc(dim_n*model.dim_t*sizeof(double));
            double *W_BX = malloc((dim_u-model.dim_k)*dim_n*sizeof(double));
            double *W_BX_T = malloc(dim_n*(dim_u-model.dim_k)*sizeof(double));
            ccl_dot_product(model.w[alpha_id],dim_u-model.dim_k,dim_b,BX,dim_b,dim_n,W_BX);
            ccl_mat_transpose(W_BX,dim_u-model.dim_k,dim_n,W_BX_T);
            if (model.dim_k ==1){
                memcpy(theta,W_BX_T,dim_n*(dim_u-model.dim_k)*sizeof(double));
            }
            else{
                gsl_matrix* ones = gsl_matrix_alloc(dim_n,model.dim_k-1);
                gsl_matrix_set_all(ones,1);
                gsl_matrix_scale(ones,M_PI/2);
                mat_hotz_app(ones->data,dim_n,model.dim_k-1,W_BX_T,dim_n,dim_u-model.dim_k,theta);

                gsl_matrix_free(ones);
            }
            for(i=0;i<dim_n;i++){
                gsl_matrix theta_mat = gsl_matrix_view_array(theta,dim_n,model.dim_t).matrix;
                gsl_vector *vec      = gsl_vector_alloc(model.dim_t);
                gsl_matrix_get_row(vec,&theta_mat,i);
                ccl_get_rotation_matrix(vec->data,Rn[i],&model,alpha_id,Rn[i]);
                gsl_vector_free(vec);
                vec                  = gsl_vector_alloc(dim_u);
                gsl_matrix_get_col(vec,Un_,i);
                ccl_dot_product(Rn[i],dim_u,dim_u,vec->data,dim_u,1,vec->data);
                gsl_matrix RnUn_     = gsl_matrix_view_array(RnUn,dim_u,dim_n).matrix;
                gsl_matrix_set_col(&RnUn_,i,vec);
                gsl_vector_free(vec);
               }
            if(model.nmse > optimal.nmse && model.nmse > 1E-5){
                model.dim_k = alpha_id;
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
    for (i=0;i<dim_n;i++){
        free(Rn[i]);
    }
    double* A = malloc(model.dim_k*model.dim_u*sizeof(double));
    gsl_matrix* X_ = gsl_matrix_alloc(model.dim_x,model.dim_n);
    memcpy(X_->data,X,dim_x*dim_n*sizeof(double));
    gsl_vector* x = gsl_vector_alloc(model.dim_x);
    gsl_matrix_get_col(x,X_,5);
    predict_proj_alpha(x->data, &model,centres,variance,Iu->data,A);
    print_mat_d(A,model.dim_k,model.dim_u);
    gsl_matrix_free(X_);
    gsl_vector_free(x);

    free(A);
    free(centres);
    free(vec);
    free(BX);
    free(RnUn);
    gsl_matrix_free(Iu);
    gsl_matrix_free(Un_);
    ccl_learn_alpha_model_free(&model);
}
void search_learn_alpha(const double* BX, const double* RnUn,LEARN_A_MODEL* model){
    OPTION option;
    option.MaxIter = 1000;
    option.Tolfun  = 1E-6;
    option.Tolx    = 1E-6;
    option.Jacob   = 0;
    model->nmse = 100000;
    int i,j,n_param;
    SOLVE_LM_WS lm_ws_param;
    gsl_rng * r;
    const gsl_rng_type *T;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    gsl_rng_set(r,5);
    n_param = (model->dim_u-model->dim_k)*model->dim_b;

    for (i=0;i<3;i++){
        gsl_vector * nmse_ = gsl_vector_alloc(model->dim_n);
        gsl_vector * W = gsl_vector_alloc((model->dim_u-model->dim_k)*model->dim_b);
        for (j=0;j< W->size;j++){
            gsl_vector_set(W,j,gsl_rng_uniform(r));
        }
        ccl_solve_lm(model,RnUn,BX,option,&lm_ws_param,W->data);
        obj_AUn(model,W->data,BX,RnUn,nmse_->data);
        gsl_vector_mul(nmse_,nmse_);
        double nmse = ccl_vec_sum(nmse_->data,model->dim_n)/model->dim_n/model->var;
        printf("K=%d, iteration=%d, residual error=%4.2g\n", model->dim_k, i, nmse);
        if(model->nmse > nmse){
            ccl_mat_reshape(W->data,(model->dim_u-model->dim_k)*model->dim_b, 1, model->w[model->dim_k-1]);
            model->nmse = nmse;
        }
        if (model->nmse < 1E-5){
            gsl_vector_free(W);
            gsl_vector_free(nmse_);
            break;
        }
        gsl_vector_free(W);
        gsl_vector_free(nmse_);
    }
    gsl_rng_free(r);
}
/* \brief kkk*/
void obj_AUn (const LEARN_A_MODEL* model,  const double* W, const double* BX, const double * RnUn,double* fun_out){
    int dim_n,dim_x,dim_b,dim_u,dim_k,dim_t,n;
    double * theta,*alpha,*W_BX,*W_BX_T,*W_;
    dim_n = model->dim_n;
    dim_x = model->dim_x;
    dim_b = model->dim_b;
    dim_u = model->dim_u;
    dim_k = model->dim_k;
    dim_t = model->dim_t;
    theta = malloc(dim_n*(dim_k-1+dim_u-dim_k)*sizeof(double));
    alpha = malloc(dim_n*dim_u*sizeof(double));
    W_   = malloc((dim_u-dim_k)*dim_b*sizeof(double));
    W_BX = malloc((dim_u-dim_k)*dim_n*sizeof(double));
    W_BX_T = malloc(dim_n*(dim_u-dim_k)*sizeof(double));
    ccl_mat_reshape(W,dim_u-dim_k,dim_b,W_);
    ccl_dot_product(W_,dim_u-dim_k,dim_b,BX,dim_b,dim_n,W_BX);
    ccl_mat_transpose(W_BX,dim_u-dim_k,dim_n,W_BX_T);
    if (dim_k ==1){
        memcpy(theta,W_BX_T,dim_n*(dim_u-dim_k)*sizeof(double));
    }
    else{
        gsl_matrix* ones = gsl_matrix_alloc(dim_n,dim_k-1);
        gsl_matrix_set_all(ones,1);
        gsl_matrix_scale(ones,M_PI/2);
        mat_hotz_app(ones->data,dim_n,dim_k-1,W_BX_T,dim_n,dim_u-dim_k,theta);
        gsl_matrix_free(ones);
    }
    ccl_get_unit_vector_from_matrix(theta,dim_n,dim_t,alpha);
    gsl_matrix a_m = gsl_matrix_view_array(alpha,dim_n,dim_u).matrix;
    gsl_matrix * RnVn_m = gsl_matrix_alloc(dim_u,dim_n);
    memcpy(RnVn_m->data,RnUn,dim_u*dim_n*sizeof(double));
    gsl_vector *a_v = gsl_vector_alloc(dim_u);
    gsl_vector *RnVn_v = gsl_vector_alloc(dim_u);
    for (n=0;n<dim_n;n++){
        gsl_matrix_get_row(a_v,&a_m,n);
        gsl_matrix_get_col(RnVn_v,RnVn_m,n);
        ccl_dot_product(a_v->data,1,dim_u,RnVn_v->data,dim_u,1,&fun_out[n]);
        fun_out[n] = pow(fun_out[n],2);
    }
    gsl_matrix_free(RnVn_m);
    gsl_vector_free(a_v);
    gsl_vector_free(RnVn_v);
    free(theta);
    free(alpha);
    free(W_BX);
    free(W_BX_T);
    free(W_);
}
void ccl_get_unit_vector_from_matrix(const double* theta,int dim_n, int dim_t,double* alpha){
    int i,k,n;
    gsl_matrix * T_mat = gsl_matrix_alloc(dim_n,dim_t);
    memcpy(T_mat->data,theta,dim_n*dim_t*sizeof(double));
    gsl_matrix   A_mat = gsl_matrix_view_array(alpha,dim_n,dim_t+1).matrix;
    gsl_vector * tmp   = gsl_vector_alloc(dim_n);
    gsl_matrix_get_col(tmp,T_mat,0);
    for (n=0;n<dim_n;n++){
        gsl_vector_set(tmp,n,cos(tmp->data[n]));
    }
    gsl_matrix_set_col(&A_mat,0,tmp);
    for (i=1;i<dim_t;i++){
        gsl_matrix_get_col(tmp,T_mat,i);
        for (n=0;n<dim_n;n++){
            gsl_vector_set(tmp,n,cos(tmp->data[n]));
        }
        gsl_matrix_set_col(&A_mat,i,tmp);
        for (k=0;k<i;k++){
            gsl_matrix_get_col(tmp,T_mat,k);
            for (n=0;n<dim_n;n++){
                gsl_vector_set(tmp,n,gsl_matrix_get(&A_mat,n,i)*sin(tmp->data[n]));
            }
            gsl_matrix_set_col(&A_mat,i,tmp);
        }
    }
    gsl_vector_set_all(tmp,1);
    gsl_matrix_set_col(&A_mat,dim_t,tmp);
    for (k=0;k<dim_t;k++){
        gsl_matrix_get_col(tmp,T_mat,k);
        for (n=0;n<dim_n;n++){
            gsl_vector_set(tmp,n,gsl_matrix_get(&A_mat,n,dim_t)*sin(tmp->data[n]));
        }
        gsl_matrix_set_col(&A_mat,dim_t,tmp);
    }
    gsl_matrix_free(T_mat);
    gsl_vector_free(tmp);
}
void ccl_solve_lm(const LEARN_A_MODEL* model,const  double* RnUn,const  double* BX, const OPTION option,SOLVE_LM_WS * lm_ws_param, double* W){
    ccl_solve_lm_ws_alloc(model,lm_ws_param);
    memcpy(lm_ws_param->xc,W,lm_ws_param->dim_x*sizeof(double));
    flt_mat(lm_ws_param->xc,lm_ws_param->dim_x,1,lm_ws_param->x);
    gsl_vector ones = gsl_vector_view_array(lm_ws_param->epsx,lm_ws_param->dim_x).vector;
    gsl_vector_set_all(&ones,1);
    gsl_vector_scale(&ones,option.Tolx);
    lm_ws_param->epsf = option.Tolfun;
    obj_AUn(model,lm_ws_param->x,BX,RnUn,lm_ws_param->r);
    int dim = (model->dim_u-model->dim_k)*model->dim_b;
    findjac(model,dim,BX,RnUn,lm_ws_param->r,lm_ws_param->x,lm_ws_param->epsx[0],lm_ws_param->J);
    memcpy(lm_ws_param->r_T->data,lm_ws_param->r,lm_ws_param->dim_n*sizeof(double));
    ccl_mat_transpose(lm_ws_param->r,lm_ws_param->dim_n,1,lm_ws_param->r_T->data);
    ccl_dot_product(lm_ws_param->r_T->data,1,lm_ws_param->dim_n,lm_ws_param->r,lm_ws_param->dim_n,1,&lm_ws_param->S);
    memcpy(lm_ws_param->J_T,lm_ws_param->J,lm_ws_param->dim_x*lm_ws_param->dim_n*sizeof(double));
    ccl_mat_transpose(lm_ws_param->J,lm_ws_param->dim_n,lm_ws_param->dim_x,lm_ws_param->J_T);
    ccl_dot_product(lm_ws_param->J_T,lm_ws_param->dim_x,lm_ws_param->dim_n,lm_ws_param->J,lm_ws_param->dim_n,lm_ws_param->dim_x,lm_ws_param->A);
    ccl_dot_product(lm_ws_param->J_T,lm_ws_param->dim_x,lm_ws_param->dim_n,lm_ws_param->r,lm_ws_param->dim_n,1,lm_ws_param->v);
    gsl_matrix A_ = gsl_matrix_view_array(lm_ws_param->A,lm_ws_param->dim_x,lm_ws_param->dim_x).matrix;
    int i;
    gsl_matrix D_ = gsl_matrix_view_array(lm_ws_param->D,lm_ws_param->dim_x,lm_ws_param->dim_x).matrix;
    gsl_matrix_set_zero(&D_);
    for (i = 0;i<lm_ws_param->dim_x;i++){
            gsl_matrix_set(&D_,i,i,gsl_matrix_get(&A_,i,i));
    }
    for (i=0;i<lm_ws_param->dim_x;i++){
        if (gsl_matrix_get(&D_,i,i)==0){
            gsl_matrix_set(&D_,i,i,1);
        }
    }
    lm_ws_param->Rlo = 0.25;
    lm_ws_param->Rhi = 0.75;
    lm_ws_param->l   = 1;
    lm_ws_param->lc  = 0.75;
    // Main iterations
    lm_ws_param->iter = 0;
    gsl_vector d = gsl_vector_view_array(lm_ws_param->d,lm_ws_param->dim_x).vector;
    gsl_vector_set_all(&d,option.Tolx);

    lm_ws_param->r_ok = ccl_any(lm_ws_param->r,lm_ws_param->epsf,lm_ws_param->dim_n,0);
    lm_ws_param->x_ok = ccl_any(d.data,lm_ws_param->epsx[0],lm_ws_param->dim_x,0);
//    int no_change = 0;
    while (lm_ws_param->iter<option.MaxIter
           && lm_ws_param->r_ok
           && lm_ws_param->x_ok){
        gsl_matrix_memcpy(lm_ws_param->D_pinv,&D_);
        gsl_matrix_scale(lm_ws_param->D_pinv,lm_ws_param->l);
        ccl_mat_add(lm_ws_param->D_pinv->data,lm_ws_param->A,lm_ws_param->dim_x,lm_ws_param->dim_x);
        ccl_MP_pinv(lm_ws_param->D_pinv->data,lm_ws_param->dim_x,lm_ws_param->dim_x,lm_ws_param->D_pinv->data);
        ccl_dot_product(lm_ws_param->D_pinv->data,lm_ws_param->dim_x,lm_ws_param->dim_x,lm_ws_param->v,lm_ws_param->dim_x,1,lm_ws_param->d);
        memcpy(lm_ws_param->xd,lm_ws_param->x,lm_ws_param->dim_x*sizeof(double));
        ccl_mat_sub(lm_ws_param->xd,lm_ws_param->d,lm_ws_param->dim_x,1);
        obj_AUn(model,lm_ws_param->xd,BX,RnUn,lm_ws_param->rd);
        int dim = (model->dim_u - model->dim_k)*model->dim_b;
        findjac(model,dim,BX,RnUn,lm_ws_param->r,lm_ws_param->x,lm_ws_param->epsx[0],lm_ws_param->J);
        ccl_mat_transpose(lm_ws_param->rd,lm_ws_param->dim_n,1,lm_ws_param->rd_T);
        ccl_dot_product(lm_ws_param->rd_T,1,lm_ws_param->dim_n,lm_ws_param->rd,lm_ws_param->dim_n,1,&lm_ws_param->Sd);
        memcpy(lm_ws_param->tmp,lm_ws_param->v,lm_ws_param->dim_x*sizeof(double));
        gsl_vector tmp_vec = gsl_vector_view_array(lm_ws_param->tmp,lm_ws_param->dim_x).vector;
        gsl_vector_scale(&tmp_vec,2);
        ccl_dot_product(lm_ws_param->A,lm_ws_param->dim_x,lm_ws_param->dim_x,lm_ws_param->d,lm_ws_param->dim_x,1,lm_ws_param->A_d->data);
        ccl_mat_sub(tmp_vec.data,lm_ws_param->A_d->data,lm_ws_param->dim_x,1);
        memcpy(lm_ws_param->d_T,lm_ws_param->d,lm_ws_param->dim_x);
        ccl_mat_transpose(lm_ws_param->d,lm_ws_param->dim_x,1,lm_ws_param->d_T);
        ccl_dot_product(lm_ws_param->d_T,1,lm_ws_param->dim_x,tmp_vec.data,lm_ws_param->dim_x,1,&lm_ws_param->dS);
        lm_ws_param->R = (lm_ws_param->S - lm_ws_param->Sd)/lm_ws_param->dS;
        if(lm_ws_param->R>lm_ws_param->Rhi){
            lm_ws_param->l = lm_ws_param->l/2;
            if(lm_ws_param->l < lm_ws_param->lc){
                lm_ws_param->l = 0;
            }
        }
        else if(lm_ws_param->R < lm_ws_param->Rlo){
            double d_Tv;
            ccl_dot_product(lm_ws_param->d_T,1,lm_ws_param->dim_x,lm_ws_param->v,lm_ws_param->dim_x,1,&d_Tv);
            lm_ws_param->nu = (lm_ws_param->Sd-lm_ws_param->S)/d_Tv+2;
            if (lm_ws_param->nu < 2){
                lm_ws_param->nu = 2;
            }
            else if (lm_ws_param->nu > 10){
                lm_ws_param->nu = 10;
            }
            if (lm_ws_param->l == 0){
                memcpy(lm_ws_param->A_inv->data,lm_ws_param->A,lm_ws_param->dim_x*lm_ws_param->dim_x*sizeof(double));
                ccl_MP_pinv(lm_ws_param->A,lm_ws_param->dim_x,lm_ws_param->dim_x,lm_ws_param->A_inv->data);
                for (i=0;i<lm_ws_param->dim_x;i++){
                    gsl_vector_set(lm_ws_param->A_inv_diag,i,fabs(gsl_matrix_get(lm_ws_param->A_inv,i,i)));
                }
                lm_ws_param->lc = 1/gsl_vector_max(lm_ws_param->A_inv_diag);
                lm_ws_param->l = lm_ws_param->lc;
                lm_ws_param->nu = lm_ws_param->nu/2;
            }
            lm_ws_param->l = lm_ws_param->nu*lm_ws_param->l;
        }
        lm_ws_param->iter ++;
        if(lm_ws_param->Sd < lm_ws_param->S){
            lm_ws_param->S = lm_ws_param->Sd;
            memcpy(lm_ws_param->x,lm_ws_param->xd,lm_ws_param->dim_x*sizeof(double));
            memcpy(lm_ws_param->r,lm_ws_param->rd,lm_ws_param->dim_n*sizeof(double));
            ccl_mat_transpose(lm_ws_param->J,lm_ws_param->dim_n,lm_ws_param->dim_x,lm_ws_param->J_T);
            ccl_dot_product(lm_ws_param->J_T,lm_ws_param->dim_x,lm_ws_param->dim_n,lm_ws_param->J,lm_ws_param->dim_n,lm_ws_param->dim_x,lm_ws_param->A);
            ccl_dot_product(lm_ws_param->J_T,lm_ws_param->dim_x,lm_ws_param->dim_n,lm_ws_param->r,lm_ws_param->dim_n,1,lm_ws_param->v);
            lm_ws_param->r_ok = ccl_any(lm_ws_param->r,lm_ws_param->epsf,lm_ws_param->dim_n,0);
            lm_ws_param->x_ok = ccl_any(d.data,lm_ws_param->epsx[0],lm_ws_param->dim_x,0);
        }
    }
    memcpy(lm_ws_param->xf,lm_ws_param->x,lm_ws_param->dim_x*sizeof(double));
    memcpy(W,lm_ws_param->xf,lm_ws_param->dim_x*sizeof(double));
    lm_ws_param->r_ok = ccl_any(lm_ws_param->r,lm_ws_param->epsf,lm_ws_param->dim_n,1); // "<="
    if(lm_ws_param->iter == option.MaxIter) printf("Solver terminated because max iteration reached\n");
    else if (lm_ws_param->x_ok) printf("Solver terminated because |dW| < min(dW), using %d iterations\n",lm_ws_param->iter);
    else if (lm_ws_param->r_ok) printf("Solver terminated because |F(dW)| < min(F(dW)), using %d iterations\n",lm_ws_param->iter);
    else printf("Problem solved\n");
    ccl_solve_lm_ws_free(lm_ws_param);
}
void findjac(const LEARN_A_MODEL* model, const int dim_x,const double* BX, const double * RnUn,const double *y,const double*x,double epsx,double* J){
    gsl_matrix J_ = gsl_matrix_view_array(J,model->dim_n,dim_x).matrix;
    gsl_matrix_set_zero(&J_);
    int k;
    double dx;
    dx = epsx*0.25;
    gsl_vector  *y_  = gsl_vector_alloc(model->dim_n);
    memcpy(y_->data,y,model->dim_n*sizeof(double));
    for (k=0;k<dim_x;k++){
        gsl_vector *yd = gsl_vector_alloc(model->dim_n);
        gsl_vector *xd = gsl_vector_alloc(dim_x);
        memcpy(xd->data,x,dim_x*sizeof(double));
        gsl_vector_set(xd,k,gsl_vector_get(xd,k)+dx);
        obj_AUn(model,xd->data,BX,RnUn,yd->data);
        gsl_vector_sub(yd,y_);
        gsl_vector_scale(yd,1/dx);
        gsl_matrix_set_col(&J_,k,yd);
        gsl_vector_free(xd);
        gsl_vector_free(yd);
    }
    gsl_vector_free(y_);
}
void ccl_get_rotation_matrix(const double*theta,const double*currentRn,const LEARN_A_MODEL* model,const int alpha_id,double*Rn){
    gsl_matrix * R = gsl_matrix_alloc(model->dim_u,model->dim_u);
    gsl_matrix_set_identity(R);
    int d;
    for(d=alpha_id;d < model->dim_t;d++){
        gsl_matrix * tmp = gsl_matrix_alloc(model->dim_u,model->dim_u);
        ccl_make_given_matrix(theta[d],d,d+1,model->dim_u,tmp->data);
        ccl_dot_product(R->data,model->dim_u,model->dim_u,tmp->data,model->dim_u,model->dim_u,R->data);
        gsl_matrix_free(tmp);
    }
    ccl_dot_product(R->data,model->dim_u,model->dim_u,currentRn,model->dim_u,model->dim_u,Rn);
    gsl_matrix_free(R);
}
void ccl_make_given_matrix(const double theta,int i,int j,int dim,double*G){
    gsl_matrix G_ = gsl_matrix_view_array(G,dim,dim).matrix;
    gsl_matrix_set_identity(&G_);
    gsl_matrix_set(&G_,i,i,cos(theta));
    gsl_matrix_set(&G_,j,j,cos(theta));
    gsl_matrix_set(&G_,i,j,-sin(theta));
    gsl_matrix_set(&G_,j,i,sin(theta));
}
void predict_proj_alpha(double* x, LEARN_A_MODEL *model,double* centres,double variance,double* Iu, double*A){
    gsl_matrix* Rn = gsl_matrix_alloc(model->dim_u,model->dim_u);
    memcpy(Rn->data,Iu,model->dim_u*model->dim_u*sizeof(double));
    gsl_matrix A_ = gsl_matrix_view_array(A,model->dim_k,model->dim_u).matrix;
    gsl_matrix_set_all(&A_,0);
    int k;
    double * BX,*theta,*alpha;
    BX = malloc(model->dim_b*1*sizeof(double));
    theta = malloc(1*model->dim_t*sizeof(double));
    alpha = malloc(1*model->dim_u*sizeof(double));
    gsl_vector* alpha_vec = gsl_vector_alloc(model->dim_u);
    for (k=1;k < model->dim_k+1;k++){
        double* W_BX = malloc((model->dim_u-k)*1*sizeof(double));
        double* W_BX_T = malloc(1*(model->dim_u-k)*sizeof(double));
        ccl_gaussian_rbf(x,model->dim_x,1,centres,model->dim_x,model->dim_b,variance,BX);
        ccl_dot_product(model->w[k-1],model->dim_u-k,model->dim_b,BX,model->dim_b,1,W_BX);
        ccl_mat_transpose(W_BX,model->dim_u-k,1,W_BX_T);
        free(W_BX);
        if (k ==1){
            memcpy(theta,W_BX_T,1*(model->dim_u-k)*sizeof(double));
            free(W_BX_T);
        }
        else{
            gsl_matrix* ones = gsl_matrix_alloc(1,k);
            gsl_matrix_set_all(ones,1);
            gsl_matrix_scale(ones,M_PI/2);
            mat_hotz_app(ones->data,1,k,W_BX_T,model->dim_n,model->dim_u-k,theta);
            free(W_BX_T);
            gsl_matrix_free(ones);
        }
        ccl_get_unit_vector_from_matrix(theta,1,model->dim_t,alpha);
        ccl_dot_product(alpha,1,model->dim_u,Rn->data,model->dim_u,model->dim_u,alpha_vec->data);
        gsl_matrix_set_row(&A_,k-1,alpha_vec);
        ccl_get_rotation_matrix(theta,Rn->data,model,k-1,Rn->data);
    }
//    print_mat_d(alpha,1,model->dim_r);
    gsl_matrix_free(Rn);
    free(BX);
    free(theta);
    free(alpha);
    gsl_vector_free(alpha_vec);
}
