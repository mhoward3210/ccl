% Function for null space projection learning predictions
%
% in:
%     F       - function outputs (before projection)
%     model   - model, containing fields
%          .t - angle
%          .A - constraint matrix
%          .P - projection matrix
%
% out:
%     Y       - function outputs, projected by model
%
function Yp = predict_lprj(F,model)

Yp = model.P*F;
