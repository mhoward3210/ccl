function Unp = predict_ncl ( model, X )
    dim_N = size( X, 2 );	
    dim_U = size( model.w, 1 );
    Unp   = zeros(dim_U, dim_N );
    for i = 1 : dim_N
        Unp(:,i) = model.w * model.phi( X(:,i) );
    end
end