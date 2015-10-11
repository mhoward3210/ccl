%{
generate the centres of the basis functions uniformly distributed over
the range of the input data X

input
    X: input data
    num_basis: number of basis functions

output
    centres of the basis functions
%}
function centres = generate_grid_centres(X, num_basis )
    
    if ~exist('num_basis')    
        n = sqrt(floor(sqrt(size(X,2)/6))^2) ;
    else
        n = floor(sqrt(num_basis));
    end           
    xmin = min(X,[],2);
    xmax = max(X,[],2);

    % allocate centres on a grid
    [xg,yg] = meshgrid( linspace(xmin(1)-0.1,xmax(1)+0.1,n),...
                        linspace(xmin(2)-0.1,xmax(2)+0.1,n));
    centres = [xg(:) yg(:)]';    
end

