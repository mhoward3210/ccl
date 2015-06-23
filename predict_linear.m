
function Yp = predict_linear(X,model)

Phi = model.phi(X);
Yp  = (Phi'*model.w)';
