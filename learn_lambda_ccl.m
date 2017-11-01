%% 
% Learning state independent selection matrix (Lambda) for problem with the form
% Un = N(q) * F(q) where N(q) = I - pinv(A(q))A(q) is a state dependent projection matrix
%                        A(q) = Lambda J(q)
%                        F(q) is some policy
% Input
%   X:  state of the system
%   Un: control of the system generated with the form Un(q) = N(q) * F(q)
%       where N(q)=I-pinv(A(q))'A(q) is the projection matrix that projects
%       F(q) unto the nullspace of A(q). N(q) can be state dependent, but
%       it should be generated in a consistent way.
% Output
%   optimal: a model for the projection matrix
%   optimal.f_proj(q): a function that predicts N(q) given q
function [optimal] = learn_lambda_ccl (Un, X, J, options)
    
    % essential parameters     
    model.dim_r     = options.dim_r ;   % dimensionality of the end effector
    model.dim_x     = size(X, 1) ;      % dimensionality of input
    model.dim_u     = size(Un,1) ;      % dimensionality of output Un = N(X) * F(X) where X
    model.dim_t     = model.dim_r - 1 ; % dimensionality of each constraint parameters    
    model.dim_n     = size(X,2) ;       % number of training points
    model.dim_b     = options.dim_b ;   % dimensionality of the gaussian kernel basis      
    optimal.nmse    = 10000000 ;        % initialise the first model
    model.var       = sum(var(Un,0,2)) ;% variance of Un
          

        
    % The constraint matrix consists of K mutually orthogonal constraint vectors.
    % At the k^{th} each iteration, candidate constraint vectors are
    % rotated to the space orthogonal to the all ( i < k ) constraint
    % vectors. At the first iteration, Rn = identity matrix
    
    Vn = zeros(model.dim_r, model.dim_n) ;
    for n = 1 : model.dim_n      
        Vn(:,n) = J(X(:,n)) * Un(:,n) ;            
        norm_v(n) = norm(Vn(:,n)) ;
    end     
    id_keep = find(norm_v > 1e-3) ;
    Vn = Vn(:,id_keep) ;
    X  = X(:,id_keep) ;
    Un = Un(:,id_keep) ;
    size(Un,2) 
    
    model.dim_n     = size(X,2) ;       % number of training points
    % choose a method for generating the centres for gaussian kernel. A
    % grid centre is usually adequate for a 2D problem. For higher 
    % dimensionality, kmeans centre normally performs better
    if model.dim_x < 3
        model.dim_b = floor(sqrt(model.dim_b))^2 ;
        centres     = generate_grid_centres (X, model.dim_b) ;          % generate centres based on grid
    else
        centres     = generate_kmeans_centres (X, model.dim_b) ;        % generate centres based on K-means
    end  
    variance        = mean(mean(sqrt(distances(centres, centres))))^2 ; % set the variance as the mean distance between centres 
    model.phi       = @(x) phi_gaussian_rbf ( x, centres,  variance );   % gaussian kernel basis function    
    BX              = model.phi(X) ;                                    % K(X)
    
    Rn    = eye(model.dim_r) ;  
    RnVn  = Vn ;
    % The objective functions is E(Xn) = Lambda * Rn * Vn. 
    % For faster computation, RnVn = Rn*Vn is pre-caldulated to avoid 
    % repeated calculation during non-linear optimisation. At the first iteration, the rotation matrix is the identity matrix, so RnUn = Un      

    for alpha_id = 1:model.dim_r        
        model.dim_k = alpha_id ;            
        model       = search_alpha (BX, RnVn, model ) ;                                    % search the optimal k^(th) constraint vector
%        model.lambda(model.dim_k,:)= get_unit_vector(model.theta(alpha_id,:)) * Rn ;
        %model.nmse = get_nmse(model, X, U, Un, J) ;
        theta       = [pi/2*ones(model.dim_n, (alpha_id-1)), (model.w{alpha_id}* BX)' ] ;     % predict constraint parameters
        
        fprintf('\t K=%d, nmse in end-effector= %4.2e\n', alpha_id, model.nmse) ;
        model.f_proj =  @(q) predict_proj (q, model, J, eye(model.dim_r)) ;
        model.nmse   = get_poe_task (model.f_proj, X, Un, Un) ;
        fprintf('\t K=%d, nmse in joint-space = %4.2e\n', alpha_id, model.nmse) ;
        % if the k^(th) constraint vector reduce the fitness, then the
        % previously found vectors are enough to describe the constraint
        if (model.nmse > optimal.nmse) && (model.nmse > 1e-3)
            break ; 
        else            
            optimal     = model ;
            Rn          = get_rotation_matrix (theta(n,:), Rn, model, alpha_id) ;  % update rotation matrix for the next iteration                   
            for n = 1: model.dim_n
                RnVn(:,n)  = Rn * Vn(:,n) ;                
            end
        end                   
    end
    optimal.f_proj  = @(q) predict_proj (q, optimal, J, eye(model.dim_r)) ;
    fprintf('\t Found %d constraint vectors with residual error = %4.2e\n', optimal.dim_k, optimal.nmse) ;
end

function nmse = get_nmse (model, X, U, Un, J)   
   
    for n = 1:size(X,2)
        A   = model.lambda * J(X(:,n)) ;
        AA  = pinv(A)*A ;
        err(n) =  Un(:,n)'*AA*Un(:,n) ;
    end
    nmse = ( mean(err) + mean(sum(Ut.^2,1)) ) / model.var ;
end

%% prediction of the projection matrix
% Our model predicts the constraint parameters. this function is used to
% reconstuct the projection matrix from constraint paramters.
function N = predict_proj (q, model, J, Iu)
    Rn      = Iu ;                                  % Initial rotation matrix
    Lambda  = zeros(model.dim_k, model.dim_r) ;     % Initial selection matrix
        
    for k = 1:model.dim_k              
        theta       = [pi/2 * ones(1,k-1) ,  (model.w{k} * model.phi(q) )' ] ;   
        alpha       = get_unit_vector(theta) ;                      % the kth alpha_0       
        Lambda(k,:) = alpha * Rn ;                                  % rotate alpha_0 to get the kth constraint
        Rn          = get_rotation_matrix (theta, Rn, model, k) ;   % update rotation matrix for (k+1)
    end            
    A = Lambda * J(q) ;            
    N = eye(model.dim_u) - pinv(A)*A ;
end

%% Learning the constraint vector
%   Input
%       V:      vec(Un*Un')
%       Jn:     jacobian
%       model:  model learnt from the last iteration
%   Output
%       model:  updated model with the k^th constraint
function model = search_alpha (BX, RnVn, model)
    options.MaxIter = 5000 ;
    options.TolFun  = 1e-6 ;
    options.TolX    = 1e-6 ;   
    obj             = @(W) obj_AVn (model, W, BX, RnVn) ;   % setup the learning objective function          
    model.nmse      = 10000000 ; 
    for i = 1:10 % normally, the 1 attempt is enough to find the solution. Repeat the process if the process tends to find local minimum (i.e., for i = 1:5) 
        W   = rand(1, (model.dim_r-model.dim_k) * model.dim_b ) ;  % make a random guess for initial value                   
        W   = solve_lm (obj, W, options );    % use a non-linear optimiser to solve obj                   
        nmse= mean(obj(W).^2) / model.var ;
        fprintf('\t K=%d, iteration=%d, residual error=%4.2e\n', model.dim_k, i, nmse) ;
        
        if model.nmse > nmse               
            model.nmse              = nmse ;   
            model.w{model.dim_k}    = reshape(W, model.dim_r-model.dim_k, model.dim_b) ;                
        end
        if model.nmse < 1e-5 % restart a random initial weight if residual error is higher than 10^-5
            break
        end
    end         
end
        
     

%% objective funtion: minimise (A * Un)^2
function [fun] = obj_AVn (model, W, BX, RnVn)   
    dim_n   = size(RnVn,2) ;    
    W       = reshape(W, model.dim_r-model.dim_k, model.dim_b );    
    fun     = zeros(dim_n, 1) ;
    theta   = [pi/2*ones(model.dim_n, (model.dim_k-1)), (W*BX)' ] ;   
    alpha   = get_unit_vector_from_matrix(theta) ; 
    for n   = 1 : dim_n                         
     %   fun(n)  = RnVn(:,n)'* pinv(alpha(n,:)) * alpha(n,:) * RnVn(:,n) ;            
        fun(n)  = alpha(n,:) * RnVn(:,n) ;            
    end           
end

%% calculate rotation matrix after finding the k^th constraint vector. The result is rotation matrix that rotate vectors into a space orthogonal to all constraint vectors
function R = get_rotation_matrix(theta_k, current_Rn, search, alpha_id)
    R = eye(search.dim_r) ;        
    for d = alpha_id : search.dim_t            
        R = R * make_givens_matrix(search.dim_r, d, d+1, theta_k(d) )' ;
    end        
    R = R * current_Rn ;
end
function G = make_givens_matrix(dim, i, j, theta)
    G      = eye(dim) ;
    G(i,i) = cos(theta) ;
    G(j,j) = cos(theta) ;
    G(i,j) =-sin(theta) ;
    G(j,i) = sin(theta) ;
end