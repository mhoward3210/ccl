#include <ccl_learn_nhat.h>
int  init_search_param(NHAT_search *search, int dim_u, int dim_n,int num_theta){
    search->dim_u     = dim_u;
    search->dim_n     = dim_n;
    search->num_theta = num_theta;
    search->dim_t     = search->dim_u - 1;
    search->epsilon   = search->dim_u * 1e-3;
    return 1;
}
int  nhat_mem_alloc_search(NHAT_search *search){
    int dim_u = search->dim_u;
    int dim_t = search->dim_t;
    int dim_s = search->dim_s;
    search->min_theta = malloc(1*dim_t*sizeof(double));
    search->max_theta = malloc(1*dim_t*sizeof(double));
    search->list = malloc(dim_s*dim_t*sizeof(double));
    search->I_u = malloc(dim_u*dim_u*sizeof(double));

    memset(search->min_theta,0,1*dim_t*sizeof(double));
    memset(search->max_theta,0,1*dim_t*sizeof(double));
    memset(search->list,0,dim_s*dim_t*sizeof(double));
    memset(search->I_u,0,dim_u*dim_u*sizeof(double));
}
int  nhat_mem_free_search(NHAT_search *search){
    int i;
    for (i=0;i<search->dim_s;i++){
        free(search->alpha[i]);
        free(search->theta[i]);
    }
    free(search->min_theta);
    free(search->max_theta);
    free(search->list);
    free(search->I_u);
    search->dim_n = 0;
    search->dim_s = 0;
    search->dim_t = 0;
    search->dim_u = 0;
}
int  nhat_mem_alloc_model(NHAT_Model *model,const NHAT_search *search){
    model->theta = malloc(search->dim_t*search->dim_t*sizeof(double));
    model->alpha = malloc(search->dim_u*search->dim_u*sizeof(double));
    model->P     = malloc(search->dim_u*search->dim_u*sizeof(double));
    model->variance = 0;
    model->umse_j   = 0;
    model->nmse_j   = 0;
    model->dim_c    = 0;
    return 1;
}
int  nhat_duplicate_model(NHAT_Model *dest, const NHAT_Model * src){
    dest->dim_n = src->dim_n;
    dest->dim_t = src->dim_t;
    dest->dim_u = src->dim_u;
    dest->dim_c = src->dim_c;
    memcpy (dest->theta,src->theta,src->dim_c*src->dim_t*sizeof(double));
    memcpy (dest->alpha,src->alpha,src->dim_c*src->dim_u*sizeof(double));
    memcpy (dest->P,src->P,src->dim_u*src->dim_u*sizeof(double));
}

int  nhat_mem_free_model(NHAT_Model *model){
    free(model->theta);
    free(model->alpha);
    free(model->P);
    model->variance = 0;
    model->umse_j   = 0;
    model->nmse_j   = 0;
    model->dim_c    = 0;
    model->dim_n    = 0;
    model->dim_t    = 0;
    model->dim_u    = 0;
    return 1;
}
void get_unit_vector(const double* theta, int dim_t,double *alpha){
    gsl_vector* theta_ = gsl_vector_alloc(dim_t);
    gsl_vector alpha_ = gsl_vector_view_array(alpha,dim_t+1).vector;
    memcpy(theta_->data,theta,dim_t*sizeof(double));
    int i,k;
    gsl_vector_set(&alpha_,0,cos(theta_->data[0]));
    for (i=1;i<dim_t;i++){
        gsl_vector_set(&alpha_,i,cos(theta_->data[i]));
        for (k=0;k<i;k++){
            gsl_vector_set(&alpha_, i, alpha_.data[i] * sin(theta_->data[k]));
        }
    }
    gsl_vector_set(&alpha_,dim_t,1);
    for (k=0;k<dim_t;k++){
        gsl_vector_set(&alpha_,i, alpha_.data[dim_t] * sin(theta_->data[k]));
    }
    gsl_vector_free(theta_);
}

void generate_search_space(NHAT_search *search){
    int t,s,dim_u,dim_s, num_theta,dim_t;
    double scalar;
    num_theta  = search->num_theta;
    dim_t  = search->dim_t;
    dim_s  = search->dim_s;
    dim_u  = search->dim_u;
    gsl_matrix * max_theta = gsl_matrix_alloc(1,dim_t);
    gsl_matrix_set_all(max_theta,1);
    gsl_matrix * min_theta = gsl_matrix_alloc(1,dim_t);
    gsl_matrix_set_all(min_theta,0);
    scalar = (double) M_PI - (M_PI/num_theta);
    gsl_matrix_scale(max_theta,scalar);
    gsl_matrix * eye  = gsl_matrix_alloc(dim_u,dim_u);
    gsl_matrix_set_identity(eye);
    gsl_matrix * list = gsl_matrix_alloc(dim_s,dim_t);
    gsl_matrix_set_all(list,0);
    gsl_matrix * interval = gsl_matrix_alloc (1, num_theta);
    for (t=0;t<dim_t;t++)
    {
        linspace(min_theta->data[t],max_theta->data[t],num_theta,interval->data);
        gsl_matrix * list_theta = gsl_matrix_alloc(interval->size1*gsl_pow_int(num_theta,(dim_t-t-1)),num_theta);
        repmat(interval->data,interval->size1,interval->size2,gsl_pow_int(num_theta,(dim_t-t-1)),1,list_theta->data);
        gsl_vector * flt_vec  = gsl_vector_alloc(list_theta->size1*list_theta->size2);
        flt_mat(list_theta->data,list_theta->size1,list_theta->size2,flt_vec->data);
        gsl_vector * repvec_ = gsl_vector_alloc(list_theta->size1*list_theta->size2*gsl_pow_int(num_theta,t));
        repvvec(flt_vec->data,flt_vec->size,gsl_pow_int(num_theta,t),repvec_->data);
        gsl_matrix_set_col(list,t,repvec_);
        gsl_matrix_free(list_theta);
        gsl_vector_free(flt_vec);
        gsl_vector_free(repvec_);
    }
    gsl_vector * vec_in = gsl_vector_alloc(dim_t);
    gsl_vector * vec_out = gsl_vector_alloc(dim_u);

    for (s=0;s<dim_s;s++){
        gsl_matrix_get_row(vec_in,list,s);
        get_unit_vector(vec_in->data,dim_t,vec_out->data);
        search->theta[s] = malloc(dim_t*sizeof(double));
        search->alpha[s] = malloc(dim_u*sizeof(double));
        memcpy(search->theta[s],vec_in->data,dim_t*sizeof(double));
        memcpy(search->alpha[s],vec_out->data,dim_u*sizeof(double));
    }
    memcpy(search->list,list->data,dim_s*dim_t*sizeof(double));
    memcpy(search->I_u,eye->data,dim_u*dim_u*sizeof(double));
    memcpy(search->min_theta,min_theta->data,dim_t*sizeof(double));
    memcpy(search->max_theta,max_theta->data,dim_t*sizeof(double));
    gsl_matrix_free(list);
    gsl_matrix_free(eye);
    gsl_vector_free(vec_in);
    gsl_vector_free(vec_out);
    gsl_matrix_free(max_theta);
    gsl_matrix_free(min_theta);
    gsl_matrix_free(interval);
}
void search_first_alpha(const double *Vn,  const double *Un, NHAT_Model *model, const NHAT_search *search, double *stats){
    int i, dim_t,dim_n,dim_u,dim_s,min_ind;
    double min_err;
    double * variance;
    dim_t = search->dim_t;
    dim_n = search->dim_n;
    dim_u = search->dim_u;
    dim_s = search->dim_s;
    model->dim_n = search->dim_n;
    model->dim_t = search->dim_t;
    model->dim_u = search->dim_u;
    model->dim_c = 1;
    double * V_AAr = malloc(model->dim_n*sizeof(double));
    double * P_Un = malloc(dim_u*dim_n*sizeof(double));
    double *AA = malloc(dim_u*dim_u*sizeof(double));
    double *AA_ = malloc(dim_u*dim_u*sizeof(double));
    memcpy(model->P,search->I_u,dim_u*dim_u*sizeof(double));
    variance = malloc(dim_u*sizeof(double));
    for (i=0;i<dim_s;i++){
        gsl_vector * V_AA = gsl_vector_alloc(model->dim_n);
        gsl_vector* alpha_inv = gsl_vector_alloc(dim_u);
        ccl_MP_pinv(search->alpha[i],1,dim_u,alpha_inv->data);
        ccl_dot_product(alpha_inv->data,dim_u,1,search->alpha[i],1,dim_u,AA);
        flt_mat(AA,dim_u,dim_u,AA_);
        ccl_dot_product(Vn,dim_n,dim_u*dim_u,AA_,dim_u*dim_u,1,V_AA->data);
        //nround(V_AA,dim_n,10,V_AAr);
        stats[i] = ccl_vec_sum(V_AA->data,dim_n)+1E-20;
        gsl_vector_free(V_AA);
        gsl_vector_free(alpha_inv);
    }
    free(AA_);
    free(AA);
    gsl_vector gsl_stats = gsl_vector_view_array(stats,dim_s).vector;
    min_err = gsl_vector_min(&gsl_stats);
    min_ind = gsl_vector_min_index(&gsl_stats);
    memcpy(model->theta,search->theta[min_ind],dim_t*sizeof(double));
    memcpy(model->alpha,search->alpha[min_ind],dim_u*sizeof(double));
    calclate_N(model->P,model->alpha,1,dim_u);
    ccl_dot_product(model->P,dim_u,dim_u,Un,dim_u,dim_n,P_Un);
    ccl_mat_var(P_Un,dim_u,dim_n,0,variance);
    model->variance = ccl_vec_sum(variance,dim_u)+1E-20;
    model->umse_j   = stats[min_ind]/dim_n;
    model->nmse_j   = model->umse_j/model->variance;
    free(variance);
    free(V_AAr);
    free(P_Un);
}

void search_alpha_nhat( const double *Vn,  const double *Un, NHAT_Model *model_in, const NHAT_search *search, NHAT_Model *model_out,double *stats){
    int i, j,dim_t,dim_n,dim_u,dim_s,min_ind,num_alpha;
    double min_err;
    double * variance;
    dim_t = search->dim_t;
    dim_n = search->dim_n;
    dim_u = search->dim_u;
    dim_s = search->dim_s;
    model_out->dim_n = search->dim_n;
    model_out->dim_t = search->dim_t;
    model_out->dim_u = search->dim_u;
    model_out->dim_c = model_in->dim_c +1;
    double abs_dot_product = 0;
    num_alpha = model_in->dim_c ;
    double *AA = malloc(dim_u*dim_u*sizeof(double));
    double *AA_ = malloc(dim_u*dim_u*sizeof(double));
    double *val= malloc(num_alpha*sizeof(double));
    double *alpha = malloc((num_alpha+1)*dim_u*sizeof(double));
    double *alpha_inv = malloc((num_alpha+1)*dim_u*sizeof(double));
    double * V_AA = malloc(model_out->dim_n*sizeof(double));
    double * V_AAr = malloc(model_out->dim_n*sizeof(double));
    double * P_Un = malloc(dim_u*dim_n*sizeof(double));
    memcpy(model_out->P,search->I_u,dim_u*dim_u*sizeof(double));
    variance = malloc(dim_u*sizeof(double));
    for (i=0;i<dim_s;i++){
        abs_dot_product = 0;
        ccl_dot_product(model_in->alpha,num_alpha,dim_u,search->alpha[i],dim_u,1,val);
        for (j = 0;j<num_alpha;j++) abs_dot_product+= fabs(val[j]);

        if(abs_dot_product>0.001)
        {
            stats[i] = 1000000;
        }
        else
        {
            mat_vert_app(model_in->alpha,num_alpha,dim_u,search->alpha[i],1,dim_u,alpha);
            ccl_MP_pinv(alpha,num_alpha+1,dim_u,alpha_inv);
            ccl_dot_product(alpha_inv,dim_u,num_alpha+1,alpha,num_alpha+1,dim_u,AA);
            flt_mat(AA,dim_u,dim_u,AA_);
            ccl_dot_product(Vn,dim_n,dim_u*dim_u,AA_,dim_u*dim_u,1,V_AA);
            //            nround(V_AA,dim_n,10,V_AAr);
            stats[i] = ccl_vec_sum(V_AA,dim_n)+1E-20;
        }
    }
    gsl_vector gsl_stats = gsl_vector_view_array(stats,dim_s).vector;
    min_err = gsl_vector_min(&gsl_stats);
    min_ind = gsl_vector_min_index(&gsl_stats);
    mat_vert_app(model_in->theta,num_alpha,dim_t,search->theta[min_ind],1,dim_t,model_out->theta);
    mat_vert_app(model_in->alpha,num_alpha,dim_t+1,search->alpha[min_ind],1,dim_t,model_out->alpha);
    calclate_N(model_out->P,model_out->alpha,num_alpha+1,dim_u);
    ccl_dot_product(model_out->P,dim_u,dim_u,Un,dim_u,dim_n,P_Un);
    ccl_mat_var(P_Un,dim_u,dim_n,0,variance);
    model_out->variance = ccl_vec_sum(variance,dim_u)+1E-20;
    model_out->umse_j   = stats[min_ind]/dim_n;
    model_out->nmse_j   = model_out->umse_j/model_out->variance;
    free(alpha);
    free(variance);
    free(AA);
    free(AA_);
    free(alpha_inv);
    free(V_AA);
    free(V_AAr);
    free(P_Un);
    free(val);
}

void learn_nhat(const double *Un, const int dim_u, const int dim_n, NHAT_Model *optimal){
    NHAT_search search;
    NHAT_result result;
    double * Un_Un,* stats;
    int n,alpha_id;
    alpha_id     = 0;
    search.dim_u = dim_u;
    search.dim_n = dim_n;
    search.num_theta = 30;
    search.dim_t = dim_u-1;
    search.epsilon = 0.00001;
    search.dim_s = gsl_pow_int(search.num_theta,search.dim_t);
    init_search_param(&search,dim_u,dim_n,search.num_theta);
    nhat_mem_alloc_search(&search);
    generate_search_space(&search);
    Un_Un = malloc(dim_u*dim_u*sizeof(double));
    stats = malloc(search.dim_s*sizeof(double));
    gsl_matrix *Vn = gsl_matrix_alloc(dim_n,dim_u*dim_u);
    gsl_matrix *Un_ = gsl_matrix_alloc(dim_u,dim_n);
    memcpy(Un_->data,Un,dim_u*dim_n*sizeof(double));
    gsl_vector * flt_v = gsl_vector_alloc(dim_u*dim_u);
    gsl_vector * vec = gsl_vector_alloc(dim_u);
    for (n=0;n<dim_n;n++){
        gsl_matrix_get_col(vec,Un_,n);
        gsl_matrix *vec_T = gsl_matrix_alloc(1,dim_u);
        memcpy(vec_T->data,vec->data,dim_u*sizeof(double));
        ccl_dot_product(vec->data,dim_u,1,vec_T->data,1,dim_u,Un_Un);
        flt_mat(Un_Un,dim_u,dim_u,flt_v->data);
        gsl_matrix_set_row(Vn,n,flt_v);
        gsl_matrix_free(vec_T);
    }
    gsl_matrix_free(Un_);
    gsl_vector_free(flt_v);
    gsl_vector_free(vec);
    //   search the first constriant
    nhat_mem_alloc_result(&result.model[0],search,1);
    nhat_mem_alloc_optimal(optimal,search,1);
    search_first_alpha(Vn->data,Un,&result.model[0],&search,stats);
    for (alpha_id=1;alpha_id<search.dim_u;alpha_id++){
        nhat_mem_alloc_result(&result.model[alpha_id],search,alpha_id+1);
        search_alpha_nhat(Vn->data,Un,&result.model[alpha_id-1],&search,&result.model[alpha_id],stats);
        if (result.model[alpha_id].nmse_j < 0.1){
            nhat_mem_free_optimal(optimal);
            nhat_mem_alloc_optimal(optimal,search,alpha_id+1);
            nhat_duplicate_model(optimal,&result.model[alpha_id]);
        }
        else{
            print_mat_d(optimal->alpha,optimal->dim_c,dim_u);
            gsl_matrix_free(Vn);
            free(Un_Un);
            free(stats);
            nhat_mem_free_result(&result,alpha_id+1);
            nhat_mem_free_search(&search);
            return;
        }
    }
}

void calclate_N(double *N, const double *A,int row,int col){
    double *invA = malloc(row*col*sizeof(double));
    double *AA   = malloc(col*col*sizeof(double));
    ccl_MP_pinv(A,row,col,invA);
    ccl_dot_product(invA,col,row,A,row,col,AA);
    ccl_mat_sub(N,AA,col,col);
    free(invA);
    free(AA);
}
// allocate memory for each result which hold the current value of the model.
int  nhat_mem_alloc_result(NHAT_Model *model, const NHAT_search search, int dim_c){
    model->theta = malloc(dim_c*search.dim_t*sizeof(double));
    model->alpha = malloc(dim_c*search.dim_u*sizeof(double));
    model->P     = malloc(search.dim_u*search.dim_u*sizeof(double));
    model->variance = 0;
    model->umse_j   = 0;
    model->nmse_j   = 0;
    return 1;
}
int  nhat_mem_free_result(NHAT_result *result, int n_models){
    int i;
    for (i=0;i<n_models;i++){
        free(result->model[i].theta);
        free(result->model[i].alpha);
        free(result->model[i].P);
        result->model[i].variance = 0;
        result->model[i].umse_j   = 0;
        result->model[i].nmse_j   = 0;
        result->model[i].dim_c    = 0;
        result->model[i].dim_n    = 0;
        result->model[i].dim_t    = 0;
        result->model[i].dim_u    = 0;
    }
    return 1;
}
int  nhat_mem_alloc_optimal(NHAT_Model *optimal,const NHAT_search search,int dim_c){
    optimal->theta = malloc(dim_c*search.dim_t*sizeof(double));
    optimal->alpha = malloc(dim_c*(search.dim_u)*sizeof(double));
    optimal->P     = malloc(search.dim_u*search.dim_u*sizeof(double));
    optimal->variance = 0;
    optimal->umse_j   = 0;
    optimal->nmse_j   = 0;
    return 1;
}
int  nhat_mem_free_optimal(NHAT_Model *optimal){
    nhat_mem_free_model(optimal);
    return 1;
}
