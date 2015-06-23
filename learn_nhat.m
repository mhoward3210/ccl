function [optimal, details] = learn_nhat (Pi, Un)
      
    [dim_u, dim_n]   = size(Un) ;       
    
    details.var_pi  = sum(var(Pi,1,2));
    details.var_un  = sum(var(Un,1,2)) ;
    details.dim_u   = dim_u ;
    details.dim_n   = dim_n ;            
        
    settings.num_theta  = 180 ;
    settings.dim_u      = dim_u ;
    settings.dim_t      = dim_u-1 ;
    settings.dim_n      = dim_n ;
    
    % generate the search space (save running time)
    fprintf('Generating search space....') ;
    search          = generate_search_space (settings) ;   
    fprintf('Done! \n') ;
    search.dim_n    = dim_n ;
    search.dim_u    = dim_u ;
    
    init_model.theta = [] ;
    init_model.alpha = [] ;
    
    fprintf('Search alpha_1 \n') ;
    [result.model{1} result.det{1}] = search_alpha (Un, init_model, search, 1) ;    
    optimal.error   = result.model{1}.nMSE ;
    optimal.alpha   = result.model{1}.alpha ;
    optimal.P       = result.model{1}.P ;
        
    for alpha_id = 2:search.dim_t
        fprintf('Search alpha_%d \n', alpha_id) ;
        [result.model{alpha_id} result.det{alpha_id}] = search_alpha (Un, result.model{alpha_id-1}, search, alpha_id) ;        
        if result.model{alpha_id}.nMSE < .001
            optimal.error   = result.model{alpha_id}.nMSE ;
            optimal.alpha   = result.model{alpha_id}.alpha ;
            optimal.P       = result.model{alpha_id}.P ;        
        else
          %  optimal.error   = details.model{alpha_id-1}.nMSE ;
          %  optimal.alpha   = details.model{alpha_id-1}.alpha ;
          %  optimal.P       = details.model{alpha_id-1}.P ;
            % stop if adding another constraint yields higher error
            % return
        end 
    end   
end


function [model, details] = search_alpha (Un, model, search, alpha_id)
   
    for i = 1:search.dim_s      
        alpha   = search.alpha{i} ;
        if  is_orthogonal (model, alpha, alpha_id) 
            alpha   = [model.alpha ; alpha] ;
            pinvAA  = pinv(alpha) * alpha ;
            matrix  = search.I_u - pinvAA ;    
            variance= norm(var( matrix*Un, 0, 2)) ;                          
            for n = 1: search.dim_n
               mse(n) = Un(:,n)' * pinvAA * Un(:,n) ;                              
            end
            details.var(i)  = variance ;         
            details.uMSE(i) = sum(mse) ;
            details.nMSE(i) = sum(mse) /variance ;
        else
            details.uMSE(i) = 1000000000 ;     
            details.nMSE(i) = 1000000000 ;               
        end
    end    
    [min_err, min_ind]  = min(details.nMSE) ;   
    model.nMSE          = details.nMSE(min_ind) / search.dim_n ;             
    model.uMSE          = details.uMSE(min_ind) / search.dim_n ;         
    model.theta         = [ model.theta; search.theta{min_ind} ] ;
    model.alpha         = [ model.alpha; search.alpha{min_ind} ] ;   
    model.P             = search.I_u - pinv(model.alpha)*model.alpha ;    
end

function orthogonal = is_orthogonal (model, alpha, alpha_id) 
    for i = alpha_id-1:-1:1                    
        if abs( model.alpha(i) * alpha' ) > 0.001            
            orthogonal = 0 ;
            return ;
        end
    end
    orthogonal = 1 ;
end

function model = get_optimal (model, details, search) 
    [min_err, min_ind]  = min(details.error(1,:)) ;              
    model.theta         = details.theta{min_ind} ;    
    model.alpha         = get_unit_vector([model.theta]) ;
    model.error         = min_err ;
    model.P             = eye(search.dim_u) - (model.alpha)' * (model.alpha * model.alpha')^(-1) *(model.alpha) ;
end