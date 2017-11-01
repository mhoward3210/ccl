function [M,dist_new] = generate_kmeans_centres (X, K)

    % initialisation: randomly select K points as the centre
    N        = size(X,2) ;    
    ind      = randperm(N) ;   
    ind      = ind(1:K) ;
    M        = X(:,ind) ;    
    dist_old = realmax ;

    for iter= 0:1000   
        D             = distances(M,X);
        [mD,ind]      = min(D);
        emptyClusters = [];

        for k = 1 : K
            ix = find(ind == k);
            if ~isempty(ix)
                M(:,k) = mean(X(:,ix),2);
            else
                emptyClusters = [emptyClusters k];
            end
        end

        dist_new = sum(mD);
        
        if isempty (emptyClusters)
            if abs (dist_old-dist_new) < 1e-10;
                return; 
            end
        else
            [sD, ind] = sort(mD, 2,'descend');
            for k=1:length(emptyClusters)
                M(:,emptyClusters(k)) = X(:,ind(k));
            end
        end
        dist_old = dist_new;      
    end
end

