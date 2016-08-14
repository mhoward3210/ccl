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
        
    optimal.nmse    = 10000000 ;        % initialise the first model
    model.var       = sum(var(Un,0,2)) ;% variance of Un
    
    % The constraint matrix consists of K mutually orthogonal constraint vectors.
    % At the k^{th} each iteration, candidate constraint vectors are
    % rotated to the space orthogonal to the all ( i < k ) constraint
    % vectors. At the first iteration, Rn = identity matrix
    Vn= zeros(model.dim_r, model.dim_n) ;
    for n = 1 : model.dim_n       
        Vn(:,n) = J(X(:,n)) * Un(:,n) ;            
    end 
    Rn    = eye(model.dim_r) ;  
    RnVn  = Vn ;
    % The objective functions is E(Xn) = Lambda * Rn * Vn. 
    % For faster computation, RnVn = Rn*Vn is pre-caldulated to avoid 
    % repeated calculation during non-linear optimisation. At the first iteration, the rotation matrix is the identity matrix, so RnUn = Un    
       
    for alpha_id = 1:model.dim_r        
        model.dim_k = alpha_id ;            
        model       = search_alpha (RnVn, model ) ;                                    % search the optimal k^(th) constraint vector
        
        % if the k^(th) constraint vector reduce the fitness, then the
        % previously found vectors are enough to describe the constraint
        if (model.nmse > optimal.nmse) && (model.nmse > 1e-3)
            break ; 
        else
            model.lambda(model.dim_k,:)= get_unit_vector(model.theta(alpha_id,:)) * Rn ;
            optimal     = model ;
            Rn          = get_rotation_matrix (model.theta(alpha_id,:), Rn, model, alpha_id) ;  % update rotation matrix for the next iteration       
            for n = 1: model.dim_n
                RnVn(:,n)  = Rn * Vn(:,n) ;
            end
        end                   
    end
    optimal.f_proj  = @(q) predict_proj (q, optimal.lambda, J, eye(model.dim_u)) ;
    fprintf('\t Found %d constraint vectors with residual error = %4.2e\n', optimal.dim_k, optimal.nmse) ;
end

%% prediction of the projection matrix
% Our model predicts the constraint parameters. this function is used to
% reconstuct the projection matrix from constraint paramters.
function N = predict_proj (q, Lambda, J, Iu)
    A = Lambda * J(q) ;        
    N = Iu - pinv(A)*A ;
end

%% Learning the constraint vector
%   Input
%       V:      vec(Un*Un')
%       Jn:     jacobian
%       model:  model learnt from the last iteration
%   Output
%       model:  updated model with the k^th constraint
function model = search_alpha ( RnVn, model)
    options.MaxIter     = 10000 ;
    options.TolFun      = 1e-6 ;
    options.TolX        = 1e-6 ;   
    obj                 = @(theta) obj_AVn (model, theta, RnVn) ;   % setup the learning objective function          
    model.nmse          = 10000000 ; 
    for i = 1:1 % normally, the 1 attempt is enough to find the solution. Repeat the process if the process tends to find local minimum (i.e., for i = 1:5) 
        theta = rand(1, model.dim_r-model.dim_k) ;  % make a random guess for initial value                   
        theta = solve_lm (obj, theta, options );    % use a non-linear optimiser to solve obj                   
        nmse  = mean(obj(theta).^2) / model.var ;
        fprintf('\t K=%d, iteration=%d, residual error=%4.2e\n', model.dim_k, i, nmse) ;
        if model.nmse > nmse               
            model.nmse                 = nmse ;   
            model.theta(model.dim_k,:) = [pi/2*ones(1, model.dim_k-1) theta] ;                 
        end
        if model.nmse < 1e-5 % restart a random initial weight if residual error is higher than 10^-5
            break
        end
    end         
end

%% objective funtion: minimise (A * Un)^2
function [fun] = obj_AVn (model, W, RnVn)   
    dim_n   = size(RnVn,2) ;    
    fun     = zeros(dim_n, 1) ;
    theta   = [pi/2*ones(1, (model.dim_k-1)), W ] ;   
    alpha   = get_unit_vector (theta) ;  
    for n   = 1 : dim_n                         
        fun(n)  = alpha * RnVn(:,n) ;            
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
    dim_x= length(x);
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