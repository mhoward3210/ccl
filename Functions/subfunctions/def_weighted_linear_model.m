function functionHandle = def_weighted_linear_model(model, Phi)
    functionHandle = @weightedLinearModelPolicy;
    % Model variables:
    c = model.c;
    var = model.var;
    b = model.b;
    function output = weightedLinearModelPolicy(q)
        %W = exp(-0.5.*sum(((q-c).^2)./var)).'; % importance weights W = [w1 w2 ... w_m ... w_M]
        W = exp(-0.5.*sum(bsxfun(@rdivide, bsxfun(@minus,q,c).^2, var))).';
        output = (Phi(q)*b*W)./sum(W); % correct version
    end
end
