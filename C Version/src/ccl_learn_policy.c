#include <ccl_learn_policy.h>
void ccl_learn_policy_pi(LEARN_MODEL_PI *model, const double *BX, const double *Y){
    LEARN_MODEL_PI_WS WS;
    ccl_learn_model_pi_ws_alloc(model, &WS);
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

    ccl_learn_model_pi_ws_free(&WS);
    gsl_matrix_free(ev_diag);
    gsl_matrix_free(V1);
    gsl_matrix_free(V1_T);
    gsl_matrix_free(pinvH1_);
}
int ccl_learn_model_pi_ws_alloc(LEARN_MODEL_PI *model,LEARN_MODEL_PI_WS* ws){
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
int ccl_learn_model_pi_ws_free(LEARN_MODEL_PI_WS* ws){
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
void ccl_learn_policy_lw_pi(LEARN_MODEL_LW_PI *model, const double *WX, const double* X,const double *Y){
    LEARN_MODEL_LW_PI_WS WS;
    ccl_learn_model_lw_pi_ws_alloc(model, &WS);
    memcpy(WS.Y_->data,Y,model->dim_n*model->dim_y*sizeof(double));
    memcpy(WS.Y_N->data,Y,model->dim_n*model->dim_y*sizeof(double));
    memcpy(WS.WX_->data,WX,model->dim_b*model->dim_n*sizeof(double));
    gsl_matrix_mul_elements(WS.Y_,WS.Y_);
    ccl_mat_sum(WS.Y_->data,model->dim_y,model->dim_n,1,WS.r->data);
    int n;
    for (n=0;n<model->dim_n;n++){
        gsl_vector_set(WS.r,n,1/sqrt(WS.r->data[n]));
    }
    for (n=0;n<model->dim_y;n++){
        gsl_matrix_set_row(WS.r_rep,n,WS.r);
    }
    gsl_matrix_mul_elements(WS.Y_N,WS.r_rep);
    gsl_matrix_set_all(WS.ones,1);
    mat_vert_app(X,model->dim_x,model->dim_n,WS.ones->data,1,model->dim_n,WS.Phi->data);
    int i;
    for(n=0;n<model->dim_b;n++){
        gsl_matrix_get_row(WS.WX_row,WS.WX_,n);
        for(i=0;i<model->dim_phi;i++){
            gsl_matrix_set_row(WS.WPhi,i,WS.WX_row);
        }

        gsl_matrix_mul_elements(WS.WPhi,WS.Phi);
        gsl_matrix_transpose_memcpy(WS.WPhi_T,WS.WPhi);
        ccl_dot_product(Y,model->dim_y,model->dim_n,WS.WPhi_T->data,model->dim_n,model->dim_phi,WS.Y_Phit->data);
        flt_mat(WS.Y_Phit->data,model->dim_y,model->dim_phi,WS.g->data);

        gsl_matrix_set_zero(WS.H);
        for (i=0;i<model->dim_n;i++){
            gsl_matrix_get_col(WS.YN_vec,WS.Y_N,i);
            gsl_matrix_get_col(WS.Phi_vec,WS.Phi,i);
            ccl_mat_transpose(WS.Phi_vec->data,model->dim_phi,1,WS.Phi_vec_T->data);
            ccl_dot_product(WS.YN_vec->data,model->dim_y,1,WS.Phi_vec_T->data,1,model->dim_phi,WS.YN_Phit->data);
            flt_mat(WS.YN_Phit->data,model->dim_y,model->dim_phi,WS.YN_Phi_vec->data);
            ccl_mat_transpose(WS.YN_Phi_vec->data,model->dim_phi*model->dim_y,1,WS.YN_Phi_vec_T->data);
            ccl_dot_product(WS.YN_Phi_vec->data,model->dim_phi*model->dim_y,1,WS.YN_Phi_vec_T->data,1,model->dim_phi*model->dim_y,WS.vv->data);
            gsl_matrix_scale(WS.vv,gsl_matrix_get(WS.WX_,n,i));
            gsl_matrix_add(WS.H,WS.vv);
        }
        // eig decomposition
        gsl_eigen_symmv_workspace * ws =gsl_eigen_symmv_alloc (model->dim_phi*model->dim_y);
        gsl_eigen_symmv (WS.H, WS.V, WS.D, ws);
        gsl_eigen_symmv_free (ws);
        gsl_eigen_symmv_sort (WS.V, WS.D,GSL_EIGEN_SORT_ABS_ASC);

        int num_idx = ccl_find_index_double(WS.V->data,model->dim_phi*model->dim_y,2,1E-8,WS.idx);
        gsl_matrix* ev_diag = gsl_matrix_alloc(num_idx,num_idx);
        gsl_matrix* V1 = gsl_matrix_alloc(model->dim_phi*model->dim_y,num_idx);
        gsl_matrix* V1_T   = gsl_matrix_alloc(num_idx,model->dim_phi*model->dim_y);
        gsl_matrix* pinvH1_ = gsl_matrix_alloc(model->dim_phi*model->dim_y,num_idx);

        gsl_matrix_set_zero(ev_diag);
        for (i=0;i<num_idx;i++){
            gsl_vector* V_vec = gsl_vector_alloc(model->dim_phi*model->dim_y);
            gsl_matrix_get_col(V_vec,WS.D,WS.idx[i]);
            gsl_matrix_set_col(V1,i,V_vec);
            gsl_matrix_set(ev_diag,i,i,1/WS.V->data[WS.idx[i]]);
            gsl_vector_free(V_vec);
        }
        gsl_matrix_transpose_memcpy(V1_T,V1);
        ccl_dot_product(V1->data,model->dim_phi*model->dim_y,num_idx,ev_diag->data,num_idx,num_idx,pinvH1_->data);
        ccl_dot_product(pinvH1_->data,model->dim_phi*model->dim_y,num_idx,V1_T->data,num_idx,model->dim_phi*model->dim_y,WS.pinvH1->data);
        ccl_dot_product(WS.pinvH1->data,model->dim_phi*model->dim_y,model->dim_phi*model->dim_y,WS.g->data,model->dim_phi*model->dim_y,1,WS.w_vec->data);

        ccl_mat_reshape(WS.w_vec->data,model->dim_y,model->dim_phi,WS.w_->data);

        ccl_mat_transpose(WS.w_->data,model->dim_y,model->dim_phi,WS.w_T->data);
        memcpy(WS.w[n]->data,WS.w_T->data,model->dim_y*model->dim_phi*sizeof(double));
        memcpy(model->w[n],WS.w[n]->data,model->dim_y*model->dim_phi*sizeof(double));
        gsl_matrix_free(ev_diag);
        gsl_matrix_free(V1);
        gsl_matrix_free(V1_T);
        gsl_matrix_free(pinvH1_);
    }
    ccl_learn_model_lw_pi_ws_free(&WS);
}
int ccl_learn_policy_lw_pi_model_alloc(LEARN_MODEL_LW_PI *model){
    int i;
    for (i=0;i<NUM_CENTRES;i++){
        model->w[i] = malloc(model->dim_phi*model->dim_y*sizeof(double));
    }
    model->c = malloc(model->dim_x*model->dim_b*sizeof(double));
}

int ccl_learn_policy_lw_pi_model_free(LEARN_MODEL_LW_PI *model){
    int i;
    for (i=0;i<NUM_CENTRES;i++){
        free(model->w[i]);
    }
    free(model->c);
}

int ccl_learn_model_lw_pi_ws_alloc(LEARN_MODEL_LW_PI *model,LEARN_MODEL_LW_PI_WS* ws){
    ws->r  = gsl_vector_alloc(model->dim_n);
    ws->r_rep  = gsl_matrix_alloc(model->dim_y,model->dim_n);
    ws->g = gsl_vector_alloc(model->dim_phi*model->dim_y);
    ws->Y_N = gsl_matrix_alloc(model->dim_y,model->dim_n);
    ws->YN_Phit = gsl_matrix_alloc(model->dim_y,model->dim_phi);
    ws->YN_vec = gsl_vector_alloc(model->dim_y);
    ws->Y_Phit = gsl_matrix_alloc(model->dim_y,model->dim_phi);
    ws->YN_Phi_vec = gsl_vector_alloc(model->dim_phi*model->dim_y);
    ws->YN_Phi_vec_T = gsl_matrix_alloc(1,model->dim_phi*model->dim_y);
    ws->ones = gsl_matrix_alloc(1,model->dim_n);
    ws->Y_ = gsl_matrix_alloc(model->dim_y,model->dim_n);
    ws->H = gsl_matrix_alloc(model->dim_phi*model->dim_y,model->dim_phi*model->dim_y);
    ws->WX_ = gsl_matrix_alloc(model->dim_b,model->dim_n);
    ws->WX_row = gsl_vector_alloc(model->dim_n);
    ws->WPhi = gsl_matrix_alloc(model->dim_phi,model->dim_n);
    ws->WPhi_T = gsl_matrix_alloc(model->dim_n,model->dim_phi);
    ws->V = gsl_vector_alloc (model->dim_phi*model->dim_y);
    ws->D = gsl_matrix_alloc (model->dim_phi*model->dim_y, model->dim_phi*model->dim_y);
    ws->pinvH1 = gsl_matrix_alloc(model->dim_phi*model->dim_y,model->dim_phi*model->dim_y);
    ws->vv = gsl_matrix_alloc(model->dim_phi*model->dim_y,model->dim_phi*model->dim_y);
    ws->Phi = gsl_matrix_alloc(model->dim_phi,model->dim_n);
    ws->Phi_vec = gsl_vector_alloc(model->dim_phi);
    ws->Phi_vec_T = gsl_matrix_alloc(1,model->dim_phi);
    ws->idx = malloc(model->dim_phi*model->dim_y*sizeof(double));
    ws->w_ = gsl_matrix_alloc(model->dim_y,model->dim_phi);
    ws->w_vec = gsl_vector_alloc(model->dim_y*model->dim_phi);
    ws->w_T = gsl_matrix_alloc(model->dim_phi,model->dim_y);
    int i;
    for (i=0;i<NUM_CENTRES;i++){
        ws->w[i] = gsl_matrix_alloc(model->dim_phi,model->dim_y);
    }
}
int ccl_learn_model_lw_pi_ws_free(LEARN_MODEL_LW_PI_WS* ws){
    gsl_vector_free(ws->g);
    gsl_matrix_free(ws->Y_N);
    gsl_vector_free(ws->YN_Phi_vec);
    gsl_matrix_free(ws->YN_Phi_vec_T);
    gsl_vector_free(ws->YN_vec);
    gsl_matrix_free(ws->Y_Phit);
    gsl_matrix_free(ws->YN_Phit);
    gsl_matrix_free(ws->ones);
    gsl_matrix_free(ws->Y_);
    gsl_vector_free(ws->Phi_vec);
    gsl_matrix_free(ws->H);
    gsl_matrix_free(ws->WPhi);
    gsl_matrix_free(ws->WPhi_T);
    gsl_matrix_free(ws->WX_);
    gsl_vector_free(ws->WX_row);
    gsl_vector_free(ws->w_vec);
    gsl_matrix_free(ws->w_);
    gsl_matrix_free(ws->w_T);
    gsl_matrix_free(ws->pinvH1);
    gsl_matrix_free(ws->Phi);
    gsl_matrix_free(ws->Phi_vec_T);
    gsl_matrix_free(ws->vv);
    gsl_vector_free (ws->V);
    gsl_matrix_free(ws->r_rep);
    gsl_vector_free(ws->r);
    gsl_matrix_free (ws->D);
    free(ws->idx);
    int i;
    for (i=0;i<NUM_CENTRES;i++){
        gsl_matrix_free(ws->w[i]);
    }
}
void predict_linear(const double* X, const double* centres,const double variance,const LEARN_MODEL_PI *model,double* Yp){
    double * BX = malloc(model->dim_b*model->dim_n*sizeof(double));
    double * BX_T = malloc(model->dim_b*model->dim_n*sizeof(double));
    double * Yp_ = malloc(model->dim_y*model->dim_n*sizeof(double));
    ccl_gaussian_rbf(X,model->dim_x,model->dim_n,centres,model->dim_x,model->dim_b,variance,BX);
    ccl_mat_transpose(BX,model->dim_b,model->dim_n,BX_T);
    ccl_dot_product(BX_T,model->dim_n,model->dim_b,model->w,model->dim_b,model->dim_y,Yp_);
    ccl_mat_transpose(Yp_,model->dim_n,model->dim_y,Yp);
    free(BX);
    free(BX_T);
    free(Yp_);
}

void predict_local_linear(const double* X, const double* centres,const double variance,const LEARN_MODEL_LW_PI *model,double* Yp){
    int i,n;
    gsl_matrix *WX = gsl_matrix_alloc(model->dim_b,model->dim_n);
    gsl_vector* WX_vec = gsl_vector_alloc(model->dim_n);
    gsl_matrix *WX_rep = gsl_matrix_alloc(model->dim_n,model->dim_y);
    ccl_gaussian_rbf(X,model->dim_x,model->dim_n,centres,model->dim_x,model->dim_b,variance,WX->data);
    gsl_matrix * Phi = gsl_matrix_alloc(model->dim_phi,model->dim_n);
    gsl_matrix * Phi_T = gsl_matrix_alloc(model->dim_n,model->dim_phi);
    gsl_matrix* ones = gsl_matrix_alloc(1,model->dim_n);
    gsl_matrix_set_all(ones,1);
    mat_vert_app(X,model->dim_x,model->dim_n,ones->data,1,model->dim_n,Phi->data);
    gsl_matrix Yp_ = gsl_matrix_view_array(Yp,model->dim_y,model->dim_n).matrix;
    gsl_matrix_set_zero(&Yp_);
    for (i=0;i<model->dim_b;i++){
        gsl_matrix* Yp_T = gsl_matrix_alloc(model->dim_n,model->dim_y);
        gsl_matrix* Yp_i = gsl_matrix_alloc(model->dim_y,model->dim_n);
        gsl_matrix_get_row(WX_vec,WX,i);
        for (n=0;n<model->dim_y;n++){
            gsl_matrix_set_col(WX_rep,n,WX_vec);
        }
        ccl_mat_transpose(Phi->data,model->dim_phi,model->dim_n,Phi_T->data);
//        print_mat_d(model->w[i],model->dim_phi,model->dim_y);
        ccl_dot_product(Phi_T->data,model->dim_n,model->dim_phi,model->w[i],model->dim_phi,model->dim_y,Yp_T->data);
        gsl_matrix_mul_elements(Yp_T,WX_rep);
        ccl_mat_transpose(Yp_T->data,model->dim_n,model->dim_y,Yp_i->data);
        gsl_matrix_add(&Yp_,Yp_i);
        gsl_matrix_free(Yp_T);
        gsl_matrix_free(Yp_i);
    }
    gsl_matrix_free(WX);
    gsl_matrix_free(WX_rep);
    gsl_vector_free(WX_vec);
    gsl_matrix_free(Phi);
    gsl_matrix_free(Phi_T);
    gsl_matrix_free(ones);
}
int ccl_write_lwmodel_to_file(char* filename, LEARN_MODEL_LW_PI* model){
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
        for (k=0;k<model->dim_b;k++){
            c = 0;
            double* tmp = malloc(model->dim_phi*model->dim_y*sizeof(double));
            for(i = 0; i < model->dim_phi; i++)
            {
                for(j = 0; j < model->dim_y; j++)
                {
                    memcpy(tmp,model->w[k],model->dim_phi*model->dim_y*sizeof(double));
                    fprintf(file,"%lf, ",tmp[c]);
                    if (j==model->dim_y-1) fprintf(file,"\n");
                    c++;
                }
            }
            free(tmp);
        }
        fclose(file);
}

int ccl_read_data_from_file(char* filename,int dim_x,int dim_n,double* mat){
    int i,j,c;
        FILE *file;
        file=fopen(filename, "r");   // extension file doesn't matter
        if(!file) {
            printf("File not found! Exiting...\n");
            return -1;
        }
        c = 0;
        for(i = 0; i < dim_x; i++)
        {
            for(j = 0; j < dim_n; j++)
            {
                if (!fscanf(file, "%lf, ", &mat[c]))
                    break;
                c++;
//                printf("%lf\n",mat[c]);
//                printf("ok!\n");
//                printf("%lf\n",mat[c]); // mat[i][j] is more clean
            }
        }
        fclose(file);
}
