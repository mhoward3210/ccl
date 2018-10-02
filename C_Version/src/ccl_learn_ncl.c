#include <ccl_learn_ncl.h>
int ccl_learn_ncl_model_alloc(LEARN_NCL_MODEL *model){
    model->c = malloc(model->dim_x*model->dim_b*sizeof(double));
    model->w = malloc(model->dim_y*model->dim_b*sizeof(double));
}
int ccl_learn_ncl_model_free(LEARN_NCL_MODEL *model){
    free(model->c);
    free(model->w);

}
void ccl_learn_ncl(const double * X, const double *Y, const int dim_x, const int dim_y, const int dim_n, const int dim_b, LEARN_NCL_MODEL *model){
    double * centres,*var_tmp,*vec, *BX;
    double variance;
    int i;
    gsl_matrix *X_,*Y_;
    model->dim_b = dim_b;
    model->dim_x = dim_x;
    model->dim_n = dim_n;
    model->dim_y = dim_y;
    ccl_learn_ncl_model_alloc(model);
    SOLVE_NONLIN_WS lm_ws_param;
    // calculate model.var
    X_ = gsl_matrix_alloc(dim_x,dim_n);
    memcpy(X_->data,X,dim_x*dim_n*sizeof(double));
    Y_     = gsl_matrix_alloc(dim_y,dim_n);
    memcpy(Y_->data,Y,dim_y*dim_n*sizeof(double));

    //  prepare for BX
    centres = malloc(dim_x*dim_b*sizeof(double));
    generate_kmeans_centres(X_->data,dim_x,dim_n,dim_b,centres);
    var_tmp = malloc(dim_b*dim_b*sizeof(double));
    vec     = malloc(dim_b*sizeof(double));
    ccl_mat_distance(centres,dim_x,dim_b,centres,dim_x,dim_b,var_tmp);

    for (i=0;i<dim_b*dim_b;i++){
        var_tmp[i] = sqrt(var_tmp[i]);
    }
    ccl_mat_mean(var_tmp,dim_b,dim_b,1,vec);
    variance = pow(gsl_stats_mean(vec,1,dim_b),2);
    BX = malloc(dim_b*dim_n*sizeof(double));
    ccl_gaussian_rbf(X_->data,dim_x,dim_n,centres,dim_x,dim_b,variance,BX);
    memcpy(model->c,centres,dim_x*dim_b*sizeof(double));
    model->s2 = variance;
    ccl_learn_model_dir(model,BX,Y);
    OPTION option;
    option.MaxIter = 1000;
    option.Tolfun  = 1E-9;
    option.Tolx    = 1E-9;
    option.Jacob   = 0;
    clock_t t = clock();
    ccl_lsqnonlin(model,BX,Y,option,&lm_ws_param,model->w);
    t = clock()-t;
    printf("\n learning nullspace component used: %f second \n",((float)t)/CLOCKS_PER_SEC);
    ccl_write_ncl_model("/home/yuchen/Desktop/ccl-1.1.0/data/ncl_model.txt",model);
    double* Unp = malloc(model->dim_y*model->dim_n*sizeof(double));
    predict_ncl(model,BX,Unp);
//    print_mat_d(model->w,model->dim_y,model->dim_b);
//    print_mat_d(Unp,model->dim_y,model->dim_n);
    free(Unp);
    ccl_learn_ncl_model_free(model);
    gsl_matrix_free(X_);
    gsl_matrix_free(Y_);
    free(var_tmp);
    free(vec);
    free(centres);
    free(BX);
}
void ccl_learn_model_dir(LEARN_NCL_MODEL *model,const double*BX,const double*Y){
    LEARN_MODEL_WS WS;
    ccl_learn_model_ws_alloc(model, &WS);
    memcpy(WS.Y_->data,Y,model->dim_n*model->dim_y*sizeof(double));
    gsl_matrix_transpose_memcpy(WS.Y_T,WS.Y_);
    ccl_dot_product(BX,model->dim_b,model->dim_n,WS.Y_T->data,model->dim_n,model->dim_y,WS.g->data);
    memcpy(WS.BX_->data,BX,model->dim_b*model->dim_n*sizeof(double));
    gsl_matrix_transpose_memcpy(WS.BX_T,WS.BX_);
    ccl_dot_product(BX,model->dim_b,model->dim_n,WS.BX_T->data,model->dim_n,model->dim_b,WS.H->data);
    gsl_matrix_scale(WS.HS,1E-8);
    gsl_matrix_add(WS.H,WS.HS);
    // eig decomposition
    gsl_eigen_symmv_workspace * ws =gsl_eigen_symmv_alloc (model->dim_b);
    gsl_eigen_symmv (WS.H, WS.V, WS.D, ws);
    gsl_eigen_symmv_free (ws);
    gsl_eigen_symmv_sort (WS.V, WS.D,GSL_EIGEN_SORT_ABS_ASC);
    int num_idx = ccl_find_index_double(WS.V->data,model->dim_b,2,1E-8,WS.idx);
    gsl_matrix* ev_diag = gsl_matrix_alloc(num_idx,num_idx);
    gsl_matrix* V1 = gsl_matrix_alloc(model->dim_b,num_idx);
    gsl_matrix* V1_T   = gsl_matrix_alloc(num_idx,model->dim_b);
    gsl_matrix* pinvH1_ = gsl_matrix_alloc(model->dim_b,num_idx);

    gsl_matrix_set_zero(ev_diag);
    int i;
    for (i=0;i<num_idx;i++){
        gsl_vector* V_vec = gsl_vector_alloc(model->dim_b);
        gsl_matrix_get_col(V_vec,WS.D,WS.idx[i]);
        gsl_matrix_set_col(V1,i,V_vec);
        gsl_matrix_set(ev_diag,i,i,1/WS.V->data[WS.idx[i]]);
        gsl_vector_free(V_vec);
    }
    gsl_matrix_transpose_memcpy(V1_T,V1);
    ccl_dot_product(V1->data,model->dim_b,num_idx,ev_diag->data,num_idx,num_idx,pinvH1_->data);
    ccl_dot_product(pinvH1_->data,model->dim_b,num_idx,V1_T->data,num_idx,model->dim_b,WS.pinvH1->data);
    ccl_dot_product(WS.pinvH1->data,model->dim_b,model->dim_b,WS.g->data,model->dim_b,model->dim_y,WS.w_->data);
    ccl_mat_transpose(WS.w_->data,model->dim_b,model->dim_y,model->w);

    ccl_learn_model_ws_free(&WS);
    gsl_matrix_free(ev_diag);
    gsl_matrix_free(V1);
    gsl_matrix_free(V1_T);
    gsl_matrix_free(pinvH1_);
}
void obj_ncl(const LEARN_NCL_MODEL *model,const double* W,const double*BX, const double*Y,double*fun,double* J){
    OBJ_WS ws;
    obj_ws_alloc(model,&ws);
    double lambda = 1E-8;
    int n;
    memcpy(ws.BX_->data,BX,model->dim_b*model->dim_n*sizeof(double));
    memcpy(ws.Y_->data,Y,model->dim_y*model->dim_n*sizeof(double));
    ccl_mat_reshape(W,model->dim_y,model->dim_b,ws.W);
    gsl_matrix J_ = gsl_matrix_view_array(J,model->dim_n,model->dim_b*model->dim_y).matrix;
//    gsl_vector fun_ = gsl_vector_view_array(fun,model->dim_n).vector;
    for (n=0;n<model->dim_n;n++){
        gsl_matrix_get_col(ws.b_n,ws.BX_,n);
        gsl_matrix_get_col(ws.u_n,ws.Y_,n);
        ccl_mat_transpose(ws.b_n->data,model->dim_b,1,ws.b_n_T->data);
        ccl_dot_product(ws.W,model->dim_y,model->dim_b,ws.b_n->data,model->dim_b,1,ws.Wb->data);
        ccl_mat_transpose(ws.Wb->data,model->dim_y,1,ws.Wb_T->data);
        ccl_dot_product(ws.Wb_T->data,1,model->dim_y,ws.Wb->data,model->dim_y,1,&ws.c);
        ccl_mat_transpose(ws.u_n->data,model->dim_y,1,ws.u_n_T->data);
        ccl_dot_product(ws.u_n_T->data,1,model->dim_y,ws.Wb->data,model->dim_y,1,&ws.a);
        ccl_dot_product(ws.u_n->data,model->dim_y,1,ws.b_n_T->data,1,model->dim_b,ws.j_n->data);
        gsl_matrix_scale(ws.j_n,ws.c);
        ccl_dot_product(ws.Wb->data,model->dim_y,1,ws.b_n_T->data,1,model->dim_b,ws.tmp2->data);
        gsl_matrix_scale(ws.tmp2,(ws.c+ws.a-2*lambda));
        gsl_matrix_sub(ws.j_n,ws.tmp2);
        gsl_matrix_scale(ws.j_n,1/(sqrt(ws.c)*ws.c));
        flt_mat(ws.j_n->data,model->dim_y,model->dim_b,ws.j_n_flt->data);
        gsl_matrix_set_row(&J_,n,ws.j_n_flt);
        fun[n] = (ws.a-ws.c+lambda)/sqrt(ws.c);
    }
    obj_ws_free(&ws);
}

void obj_ws_alloc(const LEARN_NCL_MODEL *model,OBJ_WS *ws){
    ws->a = 0;
    ws->BX_ = gsl_matrix_alloc(model->dim_b,model->dim_n);
    ws->Y_ = gsl_matrix_alloc(model->dim_y,model->dim_n);
    ws->b_n = gsl_vector_alloc(model->dim_b);
    ws->b_n_T = gsl_matrix_alloc(1,model->dim_b);
    ws->u_n = gsl_vector_alloc(model->dim_y);
    ws->u_n_T = gsl_matrix_alloc(1,model->dim_y);
    ws->c = 0;
    ws->J = gsl_matrix_alloc(model->dim_n,model->dim_b*model->dim_y);
    ws->j_n = gsl_matrix_alloc(model->dim_y,model->dim_b);
    ws->j_n_flt = gsl_vector_alloc(model->dim_b*model->dim_y);
    ws->W = malloc(model->dim_y*model->dim_b*sizeof(double));
    ws->W_ = malloc(model->dim_y*model->dim_b*sizeof(double));
    ws->Wb = gsl_vector_alloc(model->dim_y);
    ws->Wb_T = gsl_matrix_alloc(1,model->dim_y);
    ws->tmp2 = gsl_matrix_alloc(model->dim_y,model->dim_b);
}
void obj_ws_free(OBJ_WS* ws){
    free(ws->W);
    free(ws->W_);
    gsl_matrix_free(ws->BX_);
    gsl_matrix_free(ws->Y_);
    gsl_matrix_free(ws->J);
    gsl_matrix_free(ws->j_n);
    gsl_matrix_free(ws->Wb_T);
    gsl_vector_free(ws->Wb);
    gsl_vector_free(ws->b_n);
    gsl_matrix_free(ws->b_n_T);
    gsl_vector_free(ws->u_n);
    gsl_matrix_free(ws->u_n_T);
    gsl_vector_free(ws->j_n_flt);
    gsl_matrix_free(ws->tmp2);
}

int ccl_learn_model_ws_alloc(LEARN_NCL_MODEL *model,LEARN_MODEL_WS* ws){
    ws->HS = gsl_matrix_alloc(model->dim_b,model->dim_b);
    gsl_matrix_set_identity(ws->HS);
    ws->g = gsl_matrix_alloc(model->dim_b,model->dim_y);
    ws->Y_T = gsl_matrix_alloc(model->dim_n,model->dim_y);
    ws->Y_ = gsl_matrix_alloc(model->dim_y,model->dim_n);
    ws->H = gsl_matrix_alloc(model->dim_b,model->dim_b);
    ws->BX_T = gsl_matrix_alloc(model->dim_n,model->dim_b);
    ws->BX_  = gsl_matrix_alloc(model->dim_b,model->dim_n);
    ws->V = gsl_vector_alloc (model->dim_b);
    ws->D = gsl_matrix_alloc (model->dim_b, model->dim_b);
    ws->pinvH1 = gsl_matrix_alloc(model->dim_b,model->dim_b);
    ws->w_ = gsl_matrix_alloc(model->dim_b,model->dim_y);
    ws->idx = malloc(model->dim_b*sizeof(double));
}
int ccl_learn_model_ws_free(LEARN_MODEL_WS* ws){
    gsl_matrix_free(ws->HS);
    gsl_matrix_free(ws->g);
    gsl_matrix_free(ws->Y_T);
    gsl_matrix_free(ws->Y_);
    gsl_matrix_free(ws->H);
    gsl_matrix_free(ws->BX_T);
    gsl_matrix_free(ws->BX_);
    gsl_matrix_free(ws->w_);
    gsl_matrix_free(ws->pinvH1);
    gsl_vector_free (ws->V);
    gsl_matrix_free (ws->D);
    free(ws->idx);
}
void ccl_lsqnonlin(const LEARN_NCL_MODEL* model,const  double* BX, const double*Y, const OPTION option,SOLVE_NONLIN_WS * lm_ws_param, double* W){
    ccl_solve_nonlin_ws_alloc(model,lm_ws_param);
    memcpy(lm_ws_param->xc,W,lm_ws_param->dim_x*sizeof(double));
    flt_mat(lm_ws_param->xc,lm_ws_param->dim_y,lm_ws_param->dim_b,lm_ws_param->x);
    gsl_vector ones = gsl_vector_view_array(lm_ws_param->epsx,lm_ws_param->dim_x).vector;
    gsl_vector_set_all(&ones,1);
    gsl_vector_scale(&ones,option.Tolx);
    lm_ws_param->epsf = option.Tolfun;
    obj_ncl(model,lm_ws_param->x,BX,Y,lm_ws_param->r,lm_ws_param->J);
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
    lm_ws_param->d_ok = ccl_any(d.data,lm_ws_param->epsx[0],lm_ws_param->dim_x,0);
//    int no_change = 0;
    while (lm_ws_param->iter<option.MaxIter
           && lm_ws_param->r_ok
           && lm_ws_param->d_ok){
        gsl_matrix_memcpy(lm_ws_param->D_pinv,&D_);
        gsl_matrix_scale(lm_ws_param->D_pinv,lm_ws_param->l);
        ccl_mat_add(lm_ws_param->D_pinv->data,lm_ws_param->A,lm_ws_param->dim_x,lm_ws_param->dim_x);
        ccl_MP_pinv(lm_ws_param->D_pinv->data,lm_ws_param->dim_x,lm_ws_param->dim_x,lm_ws_param->D_pinv->data);
        ccl_dot_product(lm_ws_param->D_pinv->data,lm_ws_param->dim_x,lm_ws_param->dim_x,lm_ws_param->v,lm_ws_param->dim_x,1,lm_ws_param->d);
        memcpy(lm_ws_param->xd,lm_ws_param->x,lm_ws_param->dim_x*sizeof(double));
        ccl_mat_sub(lm_ws_param->xd,lm_ws_param->d,lm_ws_param->dim_x,1);
        obj_ncl(model,lm_ws_param->xd,BX,Y,lm_ws_param->rd,lm_ws_param->J);
        obj_ncl(model,lm_ws_param->x ,BX,Y,lm_ws_param->r ,lm_ws_param->J);
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
            lm_ws_param->d_ok = ccl_any(d.data,lm_ws_param->epsx[0],lm_ws_param->dim_x,0);
        }
    }
    memcpy(lm_ws_param->xf,lm_ws_param->x,lm_ws_param->dim_x*sizeof(double));
    ccl_mat_reshape(lm_ws_param->xf,lm_ws_param->dim_y,lm_ws_param->dim_b,W);
    lm_ws_param->r_ok = ccl_any(lm_ws_param->r,lm_ws_param->epsf,lm_ws_param->dim_n,1); // "<="
    if(lm_ws_param->iter == option.MaxIter) printf("Solver terminated because max iteration reached\n");
    else if (lm_ws_param->d_ok) printf("Solver terminated because |dW| < min(dW), using %d iterations\n",lm_ws_param->iter);
    else if (lm_ws_param->r_ok) printf("Solver terminated because |F(dW)| < min(F(dW)), using %d iterations\n",lm_ws_param->iter);
    else printf("Problem solved\n");
    ccl_solve_nonlin_ws_free(lm_ws_param);
}
int ccl_solve_nonlin_ws_alloc(const LEARN_NCL_MODEL *model,SOLVE_NONLIN_WS * lm_ws){
    int num_w_param;
    num_w_param = model->dim_y*model->dim_b;
    lm_ws->nu = 0;
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
    lm_ws->dim_n = model->dim_n;
    lm_ws->dim_y = model->dim_y;
    lm_ws->dim_x = num_w_param;
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
int ccl_solve_nonlin_ws_free(SOLVE_NONLIN_WS * lm_ws){
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
void predict_ncl(const LEARN_NCL_MODEL* model, const double* BX, double* Unp){
    int dim_n = model->dim_n;
    gsl_matrix Unp_ = gsl_matrix_view_array(Unp,model->dim_y,model->dim_n).matrix;
    gsl_matrix* BX_ = gsl_matrix_alloc(model->dim_b,model->dim_n);
    memcpy(BX_->data,BX,model->dim_b*model->dim_n*sizeof(double));
    gsl_vector * vec = gsl_vector_alloc(model->dim_b);
    gsl_vector * vec_ = gsl_vector_alloc(model->dim_y);
    int i;
    for (i=0;i<dim_n;i++){
        gsl_matrix_get_col(vec,BX_,i);
//        print_mat_d(model->w,model->dim_y,model->dim_b);
//        print_mat_d(vec->data,model->dim_b,1);
        ccl_dot_product(model->w,model->dim_y,model->dim_b,vec->data,model->dim_b,1,vec_->data);
        gsl_matrix_set_col(&Unp_,i,vec_);
    }
    gsl_matrix_free(BX_);
    gsl_vector_free(vec);
    gsl_vector_free(vec_);
}
int ccl_write_ncl_model(char* filename,LEARN_NCL_MODEL* model){
    int i,j,c;
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
        c = 0;
        for(i = 0; i < model->dim_y; i++)
        {
            for(j = 0; j < model->dim_b; j++)
            {
                fprintf(file,"%lf, ",model->w[c]);
                if (j==model->dim_b-1) fprintf(file,"\n");
                c++;
            }
        }
        fclose(file);
}
