%% 
% Learning state dependent projection matrix N(q) for problem with the form
% Un = N(q) * F(q) where N is a state dependent projection matrix
%                        F is some policy
% Input
%   X:  state of the system
%   Un: control of the system generated with the form Un(q) = N(q) * F(q)
%       where N(q)=I-pinv(A(q))'A(q) is the projection matrix that projects
%       F(q) unto the nullspace of A(q). N(q) can be state dependent, but
%       it should be generated in a consistent way.
% Output
%   optimal: a model for the projection matrix
%   optimal.f_proj(q): a function that predicts N(q) given q
function [optimal] = learn_alpha_ccl (Un, X, options)

    % essential parameters 
    model.dim_b     = options.dim_b ;   % dimensionality of the gaussian kernel basis
    model.dim_r     = options.dim_r ;   % dimensionality of the end effector
    model.dim_x     = size(X, 1) ;      % dimensionality of input
    model.dim_u     = size(Un,1) ;      % dimensionality of output Un = N(X) * F(X) where X
    model.dim_t     = model.dim_u - 1 ; % dimensionality of each constraint parameters    
    dim_n           = size(X,2) ;       % number of training points
        
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
    model.phi       = @(x) phi_gaussian_rbf ( x, centres, variance );   % gaussian kernel basis function    
    BX              = model.phi(X) ;                                    % K(X)
    
    optimal.nmse    = 10000000 ;        % initialise the first model
    model.var       = sum(var(Un,0,2)) ;% variance of Un
    
    % The constraint matrix consists of K mutually orthogonal constraint vectors.
    % At the k^{th} each iteration, candidate constraint vectors are
    % rotated to the space orthogonal to the all ( i < k ) constraint
    % vectors. At the first iteration, Rn = identity matrix
    Rn = cell(1,dim_n) ;
    for n = 1 : dim_n       
        Rn{n} = eye(model.dim_u) ;         
    end    
    
    % The objective functions is E(Xn) = A(Xn) * Rn(Xn) * Un. For faster
    % computation, RnUn(Xn) = Rn(Xn)*Un(Xn) is pre-caldulated to avoid
    % repeated calculation during non-linear optimisation. At the first iteration, the rotation matrix is the identity matrix, so RnUn = Un    
    RnUn    = Un ;     
        
    Iu      = eye(model.dim_u) ;
    for alpha_id = 1:model.dim_r        
        model.dim_k = alpha_id ;            
        model       = search_alpha (BX, RnUn, model ) ;                                 % search the optimal k^(th) constraint vector
        theta       = [pi/2*ones(dim_n, (alpha_id-1)), (model.w{alpha_id}* BX)' ] ;     % predict constraint parameters
        for n = 1: dim_n                          
            Rn{n}       = get_rotation_matrix (theta(n,:), Rn{n}, model, alpha_id) ;    % update rotation matrix for the next iteration                               
            RnUn(:,n)   = Rn{n} * Un(:,n) ;                                             % rotate Un ahead for faster computation
        end                 
        % if the k^(th) constraint vector reduce the fitness, then the
        % previously found vectors are enough to describe the constraint
        if (model.nmse > optimal.nmse) && (model.nmse > 1e-3)
            break ; 
        else         
            optimal   = model ;  
        end             
    end
    optimal.f_proj  =  @(q) predict_proj (optimal, q, Iu) ;  % a function to predict the projection matrix
    fprintf('\t Found %d constraint vectors with residual error = %4.2e\n', optimal.dim_k, optimal.nmse) ;
end

%% prediction of the projection matrix
% Our model predicts the constraint parameters. this function is used to
% reconstuct the projection matrix from constraint paramters.
function N = predict_proj (model, q, Iu)
    Rn = Iu ;                               % initial rotation
    A  = zeros(model.dim_k, model.dim_u) ;  % Initial constraint matrix
    bx = model.phi(q) ;                     % gaussian kernel of q    
    for k = 1:model.dim_k
        theta   = [ pi/2 * ones(1,k-1), (model.w{k} * bx )' ] ; % the kth constraint parameter                  
        A(k,:)  = get_unit_vector_from_matrix(theta) *  Rn ;                % the kth constraint vector
        Rn      = get_rotation_matrix (theta, Rn, model, k) ;   % update rotation matrix for (k+1)
    end            
    N = Iu - A'*A ;    
end

%% Learning the constraint vector
%   Input
%       BX:     higher dimensional representation of X using gaussian kernel
%       RnUn:   RnUn=Rn*Un 
%       model:  model learnt from the last iteration
%   Output
%       model:  updated model with the k^th constraint
function model = search_alpha (BX, RnUn, model)
    %options.MaxFunEvals = 1e6 ;
    options.MaxIter     = 1000 ;
    options.TolFun      = 1e-6 ;
    options.TolX        = 1e-6 ;
    options.Jacobian    = 0 ;
   
    obj                 = @(W) obj_AUn (model, W, BX, RnUn) ;   % setup the learning objective function          
    model.nmse          = 10000000 ; 
    for i = 1:1 % normally, the 1 attempt is enough to find the solution. Repeat the process if the process tends to find local minimum (i.e., for i = 1:5) 
        W    = rand(1, (model.dim_u-model.dim_k)* model.dim_b) ; % make a random guess for initial value                   
        W    = solve_lm (obj, W, options );                      % use a non-linear optimiser to solve obj                   
        nmse = mean(obj(W).^2) / model.var ;
        fprintf('\t K=%d, iteration=%d, residual error=%4.2e\n', model.dim_k, i, nmse) ;
        if model.nmse > nmse 
            model.w{model.dim_k}= reshape(W, model.dim_u-model.dim_k, model.dim_b) ;    
            model.nmse          = nmse ;            
        end
        if model.nmse < 1e-5 % restart a random initial weight if residual error is higher than 10^-5
            break
        end
    end         
end

%% objective funtion: minimise (A * Un)^2
function [fun] = obj_AUn (model, W, BX, RnUn)   
    dim_n   = size(BX,2) ;    
    W       = reshape(W, model.dim_u-model.dim_k, model.dim_b );    
    fun     = zeros(dim_n, 1) ;
    theta   = [pi/2*ones(dim_n, (model.dim_k-1)), (W * BX)' ] ;   
    alpha   = get_unit_vector_from_matrix(theta) ; 
    for n   = 1 : dim_n                         
        fun(n)  = alpha(n,:) * RnUn(:,n) ;            
    end           
end

%% calculate rotation matrix after finding the k^th constraint vector. The result is rotation matrix that rotate vectors into a space orthogonal to all constraint vectors
function R = get_rotation_matrix(theta_k, current_Rn, search, alpha_id)
    R = eye(search.dim_u) ;        
    for d = alpha_id : search.dim_t            
        R = R * make_givens_matrix(search.dim_u, d, d+1, theta_k(d) )' ;
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

%% convert constraint parameters to constraint vectors
function alpha = get_unit_vector_from_matrix (Theta)           
    [dim_n dim_t]   = size(Theta) ;      
    alpha           = zeros(dim_n,dim_t+1) ;     
    alpha(:,1)      = cos(Theta(:,1)) ;            
    for i =2:dim_t 
        alpha(:,i) = cos(Theta(:,i)) ;  
        
        for k = 1:i-1
            alpha(:,i) = alpha(:,i) .* sin(Theta(:,k)) ;        
        end                   
    end    
    alpha(:,dim_t+1)    = ones(dim_n,1) ;    
    for k = 1:dim_t            
        alpha(:,dim_t+1) = alpha(:,dim_t+1) .* sin(Theta(:,k)) ;    
    end        
end


%% Solve nonlinear least squared problem using Levenberg-Maquardt algoritm
%
% Input
%   FUN         objective fnction 
%   xc          initial guesses of solution
%   Options     'TolFun'   norm(FUN(x),1) stopping tolerance;
%               'TolX'     norm(x-xold,1) stopping tolerance;
%               'MaxIter'  Maximum number of iterations;
% Output:
%   xf        final solution approximation
%   S         sum of squares of residuals
%   msg       output message
function [xf, S, msg] = solve_lm (varargin)
      
    % 1. objective function  
    FUN = varargin{1};      
    if ~(isvarname(FUN) || isa(FUN,'function_handle'))
       error('FUN Must be a Function Handle or M-file Name.')
    end

    % 2. initial guess
    xc = varargin{2};            
    
    % 3. search options
    options.MaxIter  = 100;       % maximum number of iterations allowed
    options.TolFun   = 1e-7;      % tolerace for final function value
    options.TolX     = 1e-4;      % tolerance on difference of x-solutions     
    if nargin > 2                                  
        if isfield(varargin{3}, 'MaxIter'), options.MaxIter = varargin{3}.MaxIter ; end
        if isfield(varargin{3}, 'TolFun'), options.TolFun = varargin{3}.TolFun ;    end
        if isfield(varargin{3}, 'TolX'), options.TolX = varargin{3}.TolX ;          end           
    end
    x    = xc(:);
    dim_x   = length(x);
    epsx = options.TolX * ones(dim_x,1) ; 
    epsf = options.TolFun(:);
    
    % get function evaluation and jacobian at x
    %
    r   = FUN(x) ;                  % E(x,b)
    J   = finjac(FUN, r, x, epsx);  % dE(x,b)/db
    S   = r'*r;                     % sum of squared error
  %  nfJ = 2;
    A   = J.'*J;                    % System matrix
    v   = J.'*r;        
    D   = diag(diag(A));            % automatic scaling
    for i = 1 : dim_x
        if D(i,i)==0, D(i,i)=1 ; end
    end

    Rlo = 0.25; 
    Rhi = 0.75;
    l = 1;  lc=.75 ;
    d = options.TolX;     
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   Main iteration
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    iter = 0 ;
    while   iter < options.MaxIter && ...   % current iteration < maximum iteration
            any(abs(d) >= epsx) && ...      % d > minimum x tolerance    
            any(abs(r) >= epsf)             % r > minimum f(x) toleration
        
     %   d  = (A+l*D)\v;            % negative solution increment d = (J'J+lambda*I)^(-1)*J'*(y-f(x,w))
        d   = pinv(A+l*D)*v ;
        xd  = x-d;                  % the next x

        rd  = FUN(xd);              % residual error at xd
        J   = finjac(FUN, r, x, epsx);  % jacobian at x
     %   nfJ = nfJ + 1;
        Sd  = rd.'*rd;              % squared error if xd is taken
        dS  = d.'*(2*v-A*d);       
        R   = (S-Sd)/dS;            % reduction is squared error is xd is taken
        
        if R > Rhi                  % halve lambda if R too high
            l = l/2;
            if l < lc, l = 0; end
        elseif R < Rlo              % find new nu if R too low
            nu = (Sd-S)/(d.'*v) + 2;
            if nu < 2
                nu = 2;
            elseif nu > 10
                nu = 10;
            end
            if l==0
                lc = 1 / max(abs(diag(inv(A))));
                l  = lc;
                nu = nu/2;
            end
            l = nu*l;
        end
        iter = iter+1;    
        % update only if f(x-d) is better
        if Sd < S   
            S = Sd; 
            x = xd; 
            r = rd;        
           % nfJ = nfJ+1;
            A = J'*J;       
            v = J'*r;
        end
    end % end while

    xf  = x;  % final solution

    if iter == options.MaxIter, msg = 'Solver terminated because max iteration reached\n' ;
    elseif any(abs(d) < epsx),  msg = 'Solver terminated because |dW| < min(dW)\n' ;
    elseif any(abs(r) < epsf),  msg = 'Solver terminated because |F(dW)| < min(F(dW))\n' ; 
    else                        msg = 'Problem solved\n' ; 
    end
end

%% Numerical approximation to Jacobian
% input
%   FUN: function
%   y:  current y = FUN(x)
%   x:  current x
%   epsx: increment of x
% outpu : J = d[FUN] / d[x] 
function J = finjac(FUN, y, x, epsx)    
    dim_x = length(x);
    J  = zeros(length(y),dim_x);
    for k = 1 : dim_x
        dx      = .25*epsx(k);
        xd      = x;
        xd(k)   = xd(k)+dx;
        yd      = feval(FUN,xd);
        J(:,k)  = ((yd-y)/dx);
    end
end