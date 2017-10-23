function R = computeRMatrix(N_hat, Phi, q)
    N_hat_Phi = @(q) N_hat(q)*Phi(q);
    N = length(q);
    R = cell(1, N);
    parfor idx=1:N
        R{idx} = feval(N_hat_Phi, q{idx});
    end
end