#include <ccl_math.h>
#include <gsl/gsl_linalg.h>
#include <math.h>

void ccl_mat_add (double *A, const double *B, int row, int col){
    gsl_matrix *A_ = gsl_matrix_alloc(row,col);
    memcpy(A_->data,A,row*col*sizeof(double));
    gsl_matrix *B_ = gsl_matrix_alloc(row,col);
    memcpy(B_->data,B,row*col*sizeof(double));
    gsl_matrix_add(A_,B_);
    memcpy(A,A_->data,row*col*sizeof(double));
    gsl_matrix_free(A_);
    gsl_matrix_free(B_);
}
void ccl_mat_sub (double * A, const double * B,int row,int col){
    gsl_matrix *A_ = gsl_matrix_alloc(row,col);
    memcpy(A_->data,A,row*col*sizeof(double));
    gsl_matrix *B_ = gsl_matrix_alloc(row,col);
    memcpy(B_->data,B,row*col*sizeof(double));
    gsl_matrix_sub(A_,B_);
    memcpy(A,A_->data,row*col*sizeof(double));
    gsl_matrix_free(A_);
    gsl_matrix_free(B_);
}
double ccl_vec_sum(double* vec,int size){
    int i;
    double out=0;
    for (i=0;i<size;i++){
        out += vec[i];
    }
    return out;
}

void mat_hotz_app ( double *A, int i,int j, const double *B,int k,int d,double* c){
    int row,col,ct,ct_a,ct_b;
    ct = 0;
    ct_a =0;
    ct_b = 0;
    for (row=0;row<i;row++){
        for (col=0;col<j+d;col++){
            if (col<j){c[ct] = A[ct_a]; ct_a ++;}
            if (col>=j){c[ct] = B[ct_b]; ct_b ++;}
            ct ++;
        }
    }
}
void mat_vert_app (const double *A, int i,int j, const double *B,int k,int d,double * c){
    int row,col,ct,ct_a,ct_b;
    ct = 0;
    ct_a =0;
    ct_b = 0;
    for (row=0;row<i+k;row++){
        for (col=0;col<j;col++){
            if (row<i){c[ct] = A[ct_a]; ct_a ++;}
            if (row>=i){c[ct] = B[ct_b]; ct_b ++;}
            ct ++;
        }
    }
}

void ccl_dot_product (const double *A, int i,int j, const double *B,int k,int d,double *C){
    gsl_matrix *a = gsl_matrix_alloc(i,j);
    memcpy(a->data,A,i*j*sizeof(double));
    gsl_matrix *b = gsl_matrix_alloc(k,d);
    memcpy(b->data,B,k*d*sizeof(double));
    gsl_matrix *c = gsl_matrix_alloc(i,d);
    gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                    1.0, a, b,
                    0.0, c);
    memcpy(C,c->data,i*d*sizeof(double));
    gsl_matrix_free(a);
    gsl_matrix_free(b);
    gsl_matrix_free(c);
}
void ccl_mat_inv (const double *A, int i,int j, double *invA){
    // pinv(A)*B = B/A
    gsl_matrix *A_ = gsl_matrix_alloc(i,j);
    memcpy(A_->data,A,i*j*sizeof(double));
    gsl_matrix * invA_ = gsl_matrix_alloc(i,j);
    int s;
    gsl_permutation * p = gsl_permutation_alloc (A_->size1);
    gsl_linalg_LU_decomp (A_, p, &s);
    gsl_linalg_LU_invert(A_,p,invA_);
    memcpy(invA,invA_->data,i*j*sizeof(double));
    gsl_permutation_free(p);
    gsl_matrix_free(invA_);
    gsl_matrix_free(A_);
}
void ccl_MP_pinv(const double *A_in, int row, int col,double *invA) {
    gsl_matrix *V, *Sigma_pinv, *U, *A_pinv,*A;
    gsl_matrix *_tmp_mat = NULL;
    gsl_vector *_tmp_vec;
    gsl_vector *u;
    double x, cutoff;
    size_t i, j;
    int n = row;
    int m = col;
    bool was_swapped = false;
    double rcond = 1E-15;
    if (m > n) {
        /* libgsl SVD can only handle the case m <= n - transpose matrix */
        was_swapped = true;
        A = gsl_matrix_alloc(col,row);
        _tmp_mat = gsl_matrix_alloc(n, m);
        memcpy(_tmp_mat->data,A_in,m*n*sizeof(double));
        gsl_matrix_transpose_memcpy(A,_tmp_mat);
        i = m;
        m = n;
        n = i;
    }
    else{
        A = gsl_matrix_alloc(row,col);
        memcpy(A->data,A_in,row*col*sizeof(double));
    }

    /* do SVD */
    V = gsl_matrix_alloc(m, m);
    u = gsl_vector_alloc(m);
    _tmp_vec = gsl_vector_alloc(m);
    gsl_linalg_SV_decomp(A, V, u, _tmp_vec);
    gsl_vector_free(_tmp_vec);

    /* compute Σ⁻¹ */
    Sigma_pinv = gsl_matrix_alloc(m, n);
    gsl_matrix_set_zero(Sigma_pinv);
    cutoff = rcond * gsl_vector_max(u);

    for (i = 0; i < m; ++i) {
        if (gsl_vector_get(u, i) > cutoff) {
            x = 1. / gsl_vector_get(u, i);
        }
        else {
            x = 0.;
        }
        gsl_matrix_set(Sigma_pinv, i, i, x);
    }

    /* libgsl SVD yields "thin" SVD - pad to full matrix by adding zeros */
    U = gsl_matrix_alloc(n, n);
    gsl_matrix_set_zero(U);

    for (i = 0; i < n; ++i) {
        for (j = 0; j < m; ++j) {
            gsl_matrix_set(U, i, j, gsl_matrix_get(A, i, j));
        }
    }

    if (_tmp_mat != NULL) {
        gsl_matrix_free(_tmp_mat);
    }

    /* two dot products to obtain pseudoinverse */
    _tmp_mat = gsl_matrix_alloc(m, n);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., V, Sigma_pinv, 0., _tmp_mat);

    if (was_swapped) {
        A_pinv = gsl_matrix_alloc(n, m);
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1., U, _tmp_mat, 0., A_pinv);
    }
    else {
        A_pinv = gsl_matrix_alloc(m, n);
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1., _tmp_mat, U, 0., A_pinv);
    }
    memcpy(invA,A_pinv->data,row*col*sizeof(double));
    gsl_matrix_free(A);
    gsl_matrix_free(A_pinv);
    gsl_matrix_free(_tmp_mat);
    gsl_matrix_free(U);
    gsl_matrix_free(Sigma_pinv);
    gsl_vector_free(u);
    gsl_matrix_free(V);
}
int ccl_MP_inv_ws_alloc(MP_INV_WS *ws, int n, int m){
    bool was_swapped = false;
    int i;
    ws->A = gsl_matrix_alloc(n,m);
    if (m > n) {
        was_swapped = true;
        /* libgsl SVD can only handle the case m <= n - transpose matrix */
        ws->_tmp_mat = gsl_matrix_alloc(m, n);
        i = m;
        m = n;
        n = i;
    }
    ws->V = gsl_matrix_alloc(m, m);
    ws->u = gsl_vector_alloc(m);
    ws->_tmp_vec = gsl_vector_alloc(m);
    ws->Sigma_pinv = gsl_matrix_alloc(m, n);
    ws->U = gsl_matrix_alloc(n, n);
    ws->_tmp_mat = gsl_matrix_alloc(m, n);
    if (was_swapped) {
        ws->A_pinv = gsl_matrix_alloc(n, m);
    }
    else {
        ws->A_pinv = gsl_matrix_alloc(m, n);
    }
}
int ccl_MP_inv_ws_free(MP_INV_WS *ws){
    free(ws->A);
    free(ws->A_pinv);
    free(ws->Sigma_pinv);
    free(ws->U);
    free(ws->u);
    free(ws->V);
    free(ws->_tmp_mat);
    free(ws->_tmp_vec);
}

//void ccl_MP_pinv_test(const double *A_in, int row, int col,MP_INV_WS *ws, double *invA) {
//    double x, cutoff;
//    size_t i, j;
//    int n = row;
//    int m = col;
//    bool was_swapped = false;
//    double rcond = 1E-15;
//    memcpy(ws->A->data,A_in,row*col*sizeof(double));
//    if (m > n) {
//        /* libgsl SVD can only handle the case m <= n - transpose matrix */
//        was_swapped = true;
//        gsl_matrix_transpose_memcpy(ws->_tmp_mat, ws->A);
//        ws->A = ws->_tmp_mat;
//        i = m;
//        m = n;
//        n = i;
//    }

//    /* do SVD */
//    gsl_linalg_SV_decomp(ws->A, ws->V, ws->u, ws->_tmp_vec);

//    /* compute Σ⁻¹ */
//    gsl_matrix_set_zero(ws->Sigma_pinv);
//    cutoff = rcond * gsl_vector_max(ws->u);

//    for (i = 0; i < m; ++i) {
//        if (gsl_vector_get(ws->u, i) > cutoff) {
//            x = 1. / gsl_vector_get(ws->u, i);
//        }
//        else {
//            x = 0.;
//        }
//        gsl_matrix_set(ws->Sigma_pinv, i, i, x);
//    }

//    /* libgsl SVD yields "thin" SVD - pad to full matrix by adding zeros */
//    gsl_matrix_set_zero(ws->U);

//    for (i = 0; i < n; ++i) {
//        for (j = 0; j < m; ++j) {
//            gsl_matrix_set(ws->U, i, j, gsl_matrix_get(ws->A, i, j));
//        }
//    }

//    /* two dot products to obtain pseudoinverse */
//    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1., ws->V, ws->Sigma_pinv, 0., ws->_tmp_mat);

//    if (was_swapped) {
//        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1., ws->U, ws->_tmp_mat, 0., ws->A_pinv);
//    }
//    else {
//        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1., ws->_tmp_mat, ws->U, 0., ws->A_pinv);
//    }
//    memcpy(invA,ws->A_pinv->data,row*col*sizeof(double));
//    ccl_MP_inv_ws_free(ws);
//}


void linspace(double min, double max,double n,double *y){
    int i;
    double incre = (max-min)/(n-1);
    gsl_matrix tmp = gsl_matrix_view_array(y,1,n).matrix;
    for (i = 0;i<n;i++){
        gsl_matrix_set(&tmp,0,i,min+incre*i);
    }
}
void repmat(const double * mat,int row,int col,int rows,int cols, double * A){
    int r,c,i,j,r_c,c_c;
    r = row*rows; c = col*cols;
    gsl_matrix tmp = gsl_matrix_view_array(A,r,c).matrix;
    r_c = 0;c_c = 0;
    for (i=0;i<r;i++){

        for (j=0;j<c;j++){
            gsl_matrix_set(&tmp,i,j,mat[r_c*(col)+c_c]);
            c_c ++;
            if (c_c>col-1) c_c =0;
        }
        r_c ++;
        if (r_c >row-1) r_c = 0;c_c = 0;
    }
}
void repvec(const gsl_vector *vec, int rows, int cols, gsl_matrix * A){
    int row, col,r,c,i,j,r_c,c_c;
    double *vec_ = vec->data;
    row = vec->size;col = 1;
    r = row*rows; c = col*cols;
    //gsl_matrix * A = gsl_matrix_alloc(r,c);
    r_c = 0;c_c = 0;
    for (i=0;i<r;i++){

        for (j=0;j<c;j++){
            gsl_matrix_set(A,i,j,vec_[r_c*(col)+c_c]);
            c_c ++;
            if (c_c>col-1) c_c =0;
        }
        r_c ++;
        if (r_c >row-1) r_c = 0;c_c = 0;
    }
    //free(vec_);
}
void repvvec(const double *vec, int size, int rows, double * A){
    int row,r,i,r_c;
    row = size;
    r = row*rows;
    r_c = 0;
    for (i=0;i<r;i++){
        A[i] = vec[r_c];
        r_c ++;
        if (r_c >row-1) r_c = 0;
    }
}

void flt_mat(const double * mat,int row,int col,double * vec){
    int i,j,k;
    gsl_matrix * mat_ = gsl_matrix_alloc(row,col);
    memcpy(mat_->data,mat,row*col*sizeof(double));
    i = 0;
    for (j=0;j<col;j++){
        for (k = 0;k<row;k++){
            vec[i] = gsl_matrix_get(mat_,k,j);
            i ++;
        }
    }
    gsl_matrix_free(mat_);
}
void ccl_mat_var(const double* data_in,int row,int col,int axis,double * var){
    int i;
    gsl_matrix * m = gsl_matrix_alloc(row,col);
    memcpy(m->data,data_in,row*col*sizeof(double));
    if (axis==0){// for x axis
        gsl_vector *v= gsl_vector_alloc(col);
        for (i=0;i<row;i++){
            gsl_matrix_get_row(v,m,i);
            var[i] = gsl_stats_variance(v->data,1,v->size);
        }
        gsl_vector_free(v);
    }
    else if(axis==1){// for y axis
        gsl_vector *v= gsl_vector_alloc(row);
        for (i=0;i<col;i++){
            gsl_matrix_get_col(v,m,i);
            var[i] = gsl_stats_variance(v->data,1,v->size);
        }
        gsl_vector_free(v);
    }
    gsl_matrix_free(m);
}

void         print_mat(gsl_matrix * mat){
    int i,j,row,col;
    row = mat->size1;
    col = mat->size2;
    j = 0;
    printf("===========\n");
    printf("[");
    for (i=0;i<row;i++){
        for(j=0;j<col;j++){
            printf("%g,",gsl_matrix_get(mat,i,j));
        }
        printf ("\n");
    }
    printf("]");
    printf("===========\n");
}
void         print_mat_d(double * mat,int row,int col){
    int i,j;
    j = 0;
    printf("===============\n");
    printf("[");
    for (i=0;i<row*col;i++){
        if (i<row*col-1) printf ("%4g, ",mat[i]);
        if (i==row*col-1) printf ("%4g]",mat[i]);
        j ++;
        if (j==col){
            j = 0;
            printf ("\n");
        }
    }
    printf("===============\n");
}
void         print_mat_i(int * mat,int row,int col){
    int i,j;
    j = 0;
    printf("===============\n");
    printf("[");
    for (i=0;i<row*col;i++){
        if (i<row*col-1) printf ("%4d, ",mat[i]);
        if (i==row*col-1) printf ("%4d]",mat[i]);
        j ++;
        if (j==col){
            j = 0;
            printf ("\n");
        }
    }
    printf("===============\n");
}


void vec_to_mat(const gsl_vector * vec, gsl_matrix *mat){
    gsl_matrix_set_col(mat,0,vec);
}
void nround (const double *n,int size,unsigned int c,double *ret){
    int i;
    double m,up;
    for (i=0;i<size;i++){
        m = gsl_pow_int(10,c);
        up = n[i]*m;
        ret[i] = round(up)/m;
    }
}
void generate_kmeans_centres(const double * X,const int dim_x,const int dim_n,const int dim_b,double * centres){
    int i,N, iter,k,num_ix,num_empty_clusters;
    int* ind_,*empty_clusters,*minDi,*ix;
    size_t *sDi;
    double * M, * D,*minDv,*X_ix,*X_ix_m,*X_ink,*sDv;
    double dist_old, dist_new;
    dist_old = 10000;
    gsl_permutation *ind;
    const gsl_rng_type *T;
    gsl_rng * r;
    // finish declaration
    N = dim_n;
    gsl_rng_env_setup();

    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    gsl_rng_set(r,3);
    ind = gsl_permutation_alloc(N);
    gsl_permutation_init(ind);
    gsl_ran_shuffle(r,ind->data,N,sizeof(size_t));
    //    gsl_permutation_fprintf(stdout,ind,"%u");
    ind_ = malloc(dim_b*sizeof(int));
    for (i=0;i<dim_b;i++){
        ind_[i] = (int)(gsl_permutation_get(ind,i));
    }
    M = malloc(dim_x*dim_b*sizeof(double));
    D = malloc(dim_b*dim_n*sizeof(double));
    minDv = malloc(dim_n*sizeof(double));
    minDi = malloc(dim_n*sizeof(int));
    sDv   = malloc(dim_n*sizeof(double));
    sDi   = malloc(dim_n*sizeof(int));
    ix    = malloc(dim_n*sizeof(int));
    X_ix_m= malloc(dim_x*1*sizeof(double));
    X_ink = malloc(dim_x*sizeof(double));

    ccl_get_sub_mat_cols(X,dim_x,dim_n,ind_,dim_b,M);
    empty_clusters = malloc(dim_b*sizeof(int));
    num_empty_clusters = 0;
    for (iter=0;iter<1001;iter++){
        num_empty_clusters = 0;
        ccl_mat_distance(M,dim_x,dim_b,X,dim_x,dim_n,D);
        ccl_mat_min(D,dim_b,dim_n,1,minDv,minDi);

        memcpy(sDv,minDv,dim_n*sizeof(double));
        memset(empty_clusters,0,dim_b*sizeof(int));
        for (k=0;k<dim_b;k++){
            memset(ix,0,dim_n*sizeof(int));
            num_ix = ccl_find_index_int(minDi,dim_n,1,k,ix);
            //           print_mat_i(ix,1,dim_n);
            X_ix  = malloc(dim_x*num_ix*sizeof(double));
            if(num_ix!=0){// not empty
                ccl_get_sub_mat_cols(X,dim_x,dim_n,ix,num_ix,X_ix);
                ccl_mat_mean(X_ix,dim_x,num_ix,0,X_ix_m);
                ccl_mat_set_col(M,dim_x,dim_b,k,X_ix_m);
            }
            else{
                empty_clusters[num_empty_clusters] = k;
                num_empty_clusters ++;
            }
            free(X_ix);
        }
        dist_new = ccl_vec_sum(minDv,dim_n);
        if (num_empty_clusters == 0){
            if(fabs(dist_old-dist_new)<1E-10) {
                memcpy(centres,M,dim_x*dim_b*sizeof(double));
                return;
            }
        }
        else{
            //           print_mat_i(empty_clusters,1,num_empty_clusters);
            gsl_sort_index(sDi,sDv,1,dim_n);
            gsl_sort(sDv,1,dim_n);
            for (k=0;k<num_empty_clusters;k++){
                int ii = (int) sDi[dim_n-k-1];
                //print_mat_d(X,dim_x,dim_n);
                ccl_get_sub_mat_cols(X,dim_x,dim_n,&ii,1,X_ink);
                ccl_mat_set_col(M,dim_x,dim_b,empty_clusters[k],X_ink);
            }
        }
        dist_old = dist_new;
    }
    memcpy(centres,M,dim_x*dim_b*sizeof(double));
    gsl_permutation_free(ind);
    gsl_rng_free(r);
    free(ind_);
    free(empty_clusters);
    free(minDi);
    free(M);
    free(minDv);
    free(X_ink);
    free(ix);
    free(sDi);
    free(sDv);
    free(X_ix_m);
    free(D);
}
void ccl_get_sub_mat_cols(const double * mat, const int row, const int col,const int * ind, int size, double * ret){
    gsl_matrix * mat_ = gsl_matrix_alloc(row,col);
    memcpy(mat_->data,mat,row*col*sizeof(double));
    gsl_matrix_view ret_ = gsl_matrix_view_array(ret,row,size);
    int i;
    for (i=0;i<size;i++){
        gsl_vector* vec = gsl_vector_alloc(row);
        gsl_matrix_get_col(vec,mat_,ind[i]);
        gsl_matrix_set_col(&ret_.matrix,i,vec);
        gsl_vector_free(vec);
    }
    gsl_matrix_free(mat_);
}
int ccl_mat_distance(const double *A,const int a_i,const int a_j,const double *B,const int b_i,const int b_j,double * D){
    gsl_matrix * D_ = gsl_matrix_alloc(a_j,b_j);
    memset(D,0,a_j*b_j*sizeof(double));
    if (a_i!=b_i){
        return 0;
    }
    gsl_matrix * A_ = gsl_matrix_alloc(a_i,a_j);
    gsl_matrix * B_ = gsl_matrix_alloc(b_i,b_j);
    gsl_matrix * A_o = gsl_matrix_alloc(a_i,a_j);
    gsl_matrix * B_o = gsl_matrix_alloc(b_i,b_j);

    memcpy(A_->data,A,a_i*a_j*sizeof(double));
    memcpy(B_->data,B,b_i*b_j*sizeof(double));
    memcpy(A_o->data,A,a_i*a_j*sizeof(double));
    memcpy(B_o->data,B,b_i*b_j*sizeof(double));

    gsl_matrix_mul_elements(A_,A_);
    gsl_matrix_mul_elements(B_,B_);
    double * a2 = malloc(a_j*sizeof(double));
    double * b2 = malloc(b_j*sizeof(double));
    //rep
    ccl_mat_sum(A_->data,a_i,a_j,1,a2);
    ccl_mat_sum(B_->data,b_i,b_j,1,b2);

    repmat(gsl_vector_view_array(a2,a_j).vector.data,a_j,1,1,b_j,D);
    repmat(gsl_vector_view_array(b2,b_j).vector.data,1,b_j,a_j,1,D_->data);

    ccl_mat_add(D,D_->data,a_j,b_j);

    gsl_blas_dgemm(CblasTrans,CblasNoTrans,1.0,A_o,B_o,0.0,D_);
    gsl_matrix_scale(D_,2);
    ccl_mat_sub(D,D_->data,a_j,b_j);
    gsl_matrix_free(A_);
    gsl_matrix_free(B_);
    gsl_matrix_free(A_o);
    gsl_matrix_free(B_o);
    gsl_matrix_free(D_);
    free(a2);
    free(b2);
}
void ccl_mat_sum(const double *mat, const int i, const int j,const int axis, double * ret){
    int k;
    gsl_matrix * mat_ = gsl_matrix_alloc(i,j);
    memcpy(mat_->data,mat,i*j*sizeof(double));
    if (axis == 0){// x axis sum
        gsl_vector* vec = gsl_vector_alloc(j);
        for (k = 0;k<i;k++){
            gsl_matrix_get_row(vec,mat_,k);
            ret[k] = ccl_vec_sum(vec->data,j);
        }
        gsl_vector_free(vec);
    }
    else {
        gsl_vector* vec = gsl_vector_alloc(i);
        for (k = 0;k<j;k++){
            gsl_matrix_get_col(vec,mat_,k);
            ret[k] = ccl_vec_sum(vec->data,i);
        }
        gsl_vector_free(vec);
    }
    gsl_matrix_free(mat_);
}

void ccl_mat_min(const double * mat,const int i,const int j,const int axis,double* val,int* indx){
    int k;
    gsl_matrix * mat_ = gsl_matrix_alloc(i,j);
    memcpy(mat_->data,mat,i*j*sizeof(double));
    if (axis == 0){// x axis min
        gsl_vector * vec = gsl_vector_alloc(j);
        for (k=0;k<i;k++){
            gsl_matrix_get_row(vec,mat_,k);
            val[k] = gsl_vector_min(vec);
            indx[k] = gsl_vector_min_index(vec);
        }
        gsl_vector_free(vec);
    }
    else{ // y axis min
        gsl_vector * vec = gsl_vector_alloc(i);
        for (k=0;k<j;k++){
            gsl_matrix_get_col(vec,mat_,k);
            val[k] = gsl_vector_min(vec);
            indx[k] = gsl_vector_min_index(vec);
        }
        gsl_vector_free(vec);
    }
    gsl_matrix_free(mat_);
}
void ccl_mat_mean(const double *mat,const int i, const int j,const int axis,double*val){
    gsl_matrix * mat_ = gsl_matrix_alloc(i,j);
    memcpy(mat_->data,mat,i*j*sizeof(double));
    int k;
    if (axis == 0){// x axis
        gsl_vector * vec = gsl_vector_alloc(j);
        for (k=0;k<i;k++){
            gsl_matrix_get_row(vec,mat_,k);
            val[k] = gsl_stats_mean(vec->data,1,j);
        }
        gsl_vector_free(vec);
    }
    if (axis == 1){// y axis
        gsl_vector * vec = gsl_vector_alloc(i);
        for (k=0;k<i;k++){
            gsl_matrix_get_col(vec,mat_,k);
            val[k] = gsl_stats_mean(vec->data,1,i);
        }
        gsl_vector_free(vec);
    }
    gsl_matrix_free(mat_);
}
// 1: = ; 2 >=; 3<=;
int ccl_find_index_int(const int *a, const int num_elements, const int operand, const int value,int* idx)
{
    int i,found;
    found = 0;
    for (i=0; i<num_elements; i++)
    {
        switch (operand){
        case 1:
            if (a[i] == value)
            {
                idx[found] = i;  /* it was found */
                found ++;
            }
            break;
        case 2:
            if (a[i] >= value)
            {
                idx[found] = i;  /* it was found */
                found ++;
            }
            break;
        case 3:
            if (a[i] <= value)
            {
                idx[found] = i;  /* it was found */
                found ++;
            }

        }

    }
    return(found);  /* if it was not found */
}
int ccl_find_index_double(const double *a, const int num_elements, const int operand, const double value,int* idx)
{
    int i,found;
    found = 0;
    for (i=0; i<num_elements; i++)
    {
        if (operand==1){
            if (a[i] == value)
            {
                idx[found] = i;  /* it was found */
                found ++;
            }
        }
        else if (operand==2){
            if (a[i] >= value)
            {
                idx[found] = i;  /* it was found */
                found ++;
            }
        else if (operand==3){
                if (a[i] <= value)
                {
                    idx[found] = i;  /* it was found */
                    found ++;
                }
            }
        }
    }
    return(found);  /* if it was not found */
}
void ccl_mat_set_col(double * mat,int i, int j, int c,double* vec){
    gsl_matrix mat_ = gsl_matrix_view_array(mat,i,j).matrix;
    gsl_vector vec_ = gsl_vector_view_array(vec,i).vector;
    gsl_matrix_set_col(&mat_,c,&vec_);
}
void ccl_gaussian_rbf(const double * X,int i, int j,const double *C,int k, int d,double s,double * BX){
    double* D = malloc(d*j*sizeof(double));
    ccl_mat_distance(C,k,d,X,i,j,D);
    int cc;
    for (cc=0;cc<d*j;cc++){
        BX[cc] = exp(-0.5/s*D[cc]);
    }
    //print_mat_d(BX,d,j);
    double * vec = malloc(j*sizeof(double));
    ccl_mat_sum(BX,d,j,1,vec);
    for (cc=0;cc<j;cc++){
        vec[cc] = 1/vec[cc];
    }
    double * tmp = malloc(d*j*sizeof(double));
    repmat(vec,1,j,d,1,tmp);
    gsl_matrix BX_ = gsl_matrix_view_array(BX,d,j).matrix;
    gsl_matrix tmp_ = gsl_matrix_view_array(tmp,d,j).matrix;
    gsl_matrix_mul_elements(&BX_,&tmp_);
    free(vec);
    free(D);
    free(tmp);
}
void ccl_mat_transpose(const double* mat,int i, int j,double* mat_T){
    gsl_matrix * mat_ = gsl_matrix_alloc(i,j);
    memcpy(mat_->data,mat,i*j*sizeof(double));
    gsl_matrix mat_T_ = gsl_matrix_view_array(mat_T,j,i).matrix;
    gsl_vector * vec = gsl_vector_alloc(j);
    int row, col;
    for (row=0;row<i;row++){
        gsl_matrix_get_row(vec,mat_,row);
        gsl_matrix_set_col(&mat_T_,row,vec);
    }
    gsl_vector_free(vec);
    gsl_matrix_free(mat_);
}
int  ccl_any(double* vec,double epx,int size,int op){
    int i;
    if (op==0){ //" >="
        for (i=0;i<size;i++){
            if (fabs(vec[i])>=epx) return 1;
            else return 0;
        }
    }
    if (op==1){
        for (i=0;i<size;i++){ //"<="
            if (fabs(vec[i])<=epx) return 1;
            else return 0;
        }
    }
}
void ccl_mat_reshape(const double* vec,int i,int j,double *mat){
    gsl_matrix mat_ = gsl_matrix_view_array(mat,i,j).matrix;
    int row,col,c;
    c = 0;
    for (col=0;col<j;col++){
        for (row=0;row<i;row++){
            gsl_matrix_set(&mat_,row,col,vec[c]);
            c++;
        }
    }
}
