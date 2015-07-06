function [optimal, result] = learn_nhat (Un)
    
    % setup parameters for search
    search.dim_u    = size(Un,1) ;          % dimensionality of the action space        
    search.dim_n    = size(Un,2) ;          % number of data points      
    search.num_theta= 180 ;                 % number of candidate constraints (from 0 to pi)
    search.dim_t    = search.dim_u - 1 ;    % number of parameters needed to represent an unit vector
    search.epsilon  = search.dim_u * 0.001 ;    
    search          = generate_search_space (search) ;       
    
    % vectorises the matrix  
    Vn = zeros(search.dim_n, search.dim_u^2) ;  
    for n = 1:search.dim_n
        UnUn    = Un(:,n)*Un(:,n)' ;
        Vn(n,:) = UnUn(:)' ;     
    end   
   
    % search the first constraint
    result.model{1} = search_first_alpha (Vn, Un, [], search) ;  
    optimal         = result.model{1} ;
    
    % search the next constraint until the new constraint does not fit
    for alpha_id = 2:search.dim_t   
        result.model{alpha_id} = search_alpha (Vn, Un, result.model{alpha_id-1}, search) ;               
        if result.model{alpha_id}.nmse_j < .1
            optimal = result.model{alpha_id} ;            
        else           
            return ;
        end             
    end   
end

function [model, stats] = search_first_alpha (V, Un, model, search)
    for i = 1:search.dim_s               
        alpha         = search.alpha{i} ;                   
        AA            = pinv(alpha)*alpha ;
        stats.umse(i) = sum ( V*AA(:) ) ;                   
    end    
    [min_err, min_ind]  = min(stats.umse) ;       
    model.theta         = search.theta{min_ind} ;
    model.alpha         = search.alpha{min_ind} ;   
    model.P             = search.I_u - pinv(model.alpha) * model.alpha ;
    model.variance      = sum(var( model.P*Un, 0, 2)) ;     
    model.umse_j        = stats.umse(min_ind) / search.dim_n ;          
    model.nmse_j        = model.umse_j        / model.variance ;      
end

function [model, stats] = search_alpha (V, Un, model, search)
 
    % for alpha_id > 1, check if alpha is orthogonal to the existing ones
    for i = 1:search.dim_s               
        abs_dot_product = abs ( model.alpha * search.alpha{i}' )  ; % dot product between this alpha and the previous one's
        if sum(abs_dot_product > 0.001) > 0 % ignore alpha that is not orthogonal to any one of them            
            stats.umse(i) = 1000000000 ;        
        else             
            alpha         = [model.alpha; search.alpha{i}] ;         
            AA            = pinv(alpha)*alpha ;
            stats.umse(i) = sum ( V*AA(:) ) ;                 
        end
    end 
    [min_err, min_ind]  = min(stats.umse) ;       
    model.theta         = [ model.theta ; search.theta{min_ind} ] ;
    model.alpha         = [ model.alpha ; search.alpha{min_ind} ] ;               
    model.P             = search.I_u - pinv(model.alpha) * model.alpha ;
    model.variance      = sum(var( model.P*Un, 0, 2)) ;     
    model.umse_j        = stats.umse(min_ind) / search.dim_n ;          
    model.nmse_j        = model.umse_j        / model.variance ;      
end


function search = generate_search_space (search)   
    search.min_theta    = zeros(1,search.dim_t) ;    
    search.max_theta    = ( pi-(pi/search.num_theta) )*ones(1,search.dim_t) ;        
    
    num_theta  = search.num_theta ;
    dim_u      = search.dim_u ;      
    dim_t      = search.dim_t ;
    dim_s      = num_theta ^ dim_t ;    
    list       = zeros(dim_s, dim_t) ;
    theta      = cell(1,dim_s) ;
    alpha      = cell(1,dim_s) ;  
      
    for t = 1:dim_t
        list_theta  = linspace(search.min_theta(t), search.max_theta(t), num_theta) ;          
        list_theta  = repmat(list_theta,    num_theta^(dim_t-t), 1) ;
        list(:,t)   = repmat(list_theta(:), num_theta^(t-1), 1) ;        
    end
         
    % make list of alpha
    for s = 1: dim_s   
        theta{s} = list(s,:) ;
        alpha{s} = get_unit_vector ( list(s,:) ) ;   
    end      
    search.list = list ;
    search.I_u      = eye(dim_u) ;        
    search.dim_s    = dim_s ;
    search.theta    = theta ;
    search.alpha    = alpha ;
    search.interval = list(2,dim_t)-list(1,dim_t) ;
end

function alpha = get_unit_vector (theta)    
       
    dim_t   = length(theta) ;      
    alpha   = zeros(1,dim_t+1) ; 
    
    alpha(1) = cos(theta(1)) ;    
    
   
    for i =2:dim_t %dim_t:-1:2
        alpha(i) = cos(theta(i)) ;        
        for k = 1:i-1 % i:dim_t
            alpha(i) = alpha(i) * sin(theta(k)) ;        
        end                   
    end
    
    alpha(dim_t+1)    = 1 ;    
    for k = 1:dim_t            
        alpha(dim_t+1) = alpha(dim_t+1) * sin(theta(k)) ;    
    end        
end

