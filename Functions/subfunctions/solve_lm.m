function [xf, S, msg] = solve_lm (varargin)
% [xf, S, msg] = solve_lm (varargin)
%
% Solve nonlinear least squared problem using Levenberg-Maquardt algoritm
%
% Input
%   FUN                                   Objective fnction
%   xc                                    Initial guesses of solution
%   Options                                 'TolFun'   norm(FUN(x),1) stopping tolerance;
%                                         'TolX'     norm(x-xold,1) stopping tolerance;
%                                         'MaxIter'  Maximum number of iterations;
% Output:
%   xf                                    Final solution approximation
%   S                                     Sum of squares of residuals
%   msg                                   Output message

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

function J = finjac(FUN, y, x, epsx)
% J = finjac(FUN, y, x, epsx)
%
% Numerical approximation to Jacobian
%
% Input:
%   FUN                                     Function
%   y                                       Current y = FUN(x)
%   x                                       Current x
%   epsx                                    Increment of x
%
% Output                                    J = d[FUN] / d[x]

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