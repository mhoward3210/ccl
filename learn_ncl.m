%{
learn the nullspace component of the input data X

input
    X: observed states
    Y: observed actions of the form Y = A(X)'B(X) + N(X) F(X) where N and F 
       are consistent across the dataset X 

output
    model: model that prodicts the nullspace component N(X)F(X) 
%}
function model = learn_ncl(X, Y, model)
    
    % calculate the basis functions of the training data    
    BX      = model.phi(X) ;
    
    % learn an initial model using a simple parametric model
    model   = learn_model_dir ( model, BX, Y ) ;  
    
    % learn the model by minimising the proposed step-1 method        
    obj     = @(W) obj_ncl_reg ( model, W, BX, Y);   % setup the learning objective function        
    options = optimset( 'Jacobian','on', 'Display', 'notify',...  % options for the optimisation    
                        'MaxFunEvals',1e9, 'MaxIter', 1000,...
                        'TolFun',1e-9, 'TolX',1e-9,...
                        'Algorithm', 'levenberg-marquardt');
    model.w = lsqnonlin(obj, model.w, [], [], options );  % use the non-linear optimiser to solve obj_ncl_reg    
end

function model = learn_model_dir ( model, BX, U )

    HS  = eye(model.num_basis);    
    g   = BX * U' ;
    H   = BX * BX';
    
    % do eigen-decomposition for inversion
    [V,D]   = eig( H + 1e-8*HS);
    ev      = diag(D);    
    ind     = find( ev > 1e-8);    
    V1      = V(:,ind);
    pinvH1  = V1 * diag(ev(ind).^-1)*V1';
    model.w = (pinvH1 * g)' ;    
end
