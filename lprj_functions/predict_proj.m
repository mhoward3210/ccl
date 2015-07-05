% Project a vector onto the image space of a learnt projection 
%
% input
%     F: input vector before projection
%     model: learnt model for nullspace projection, containing fields
%          .P: learnt projection matrix
%
% output
%     Yp: the resulting vector after projection 
%
function Yp = predict_proj (F,model)
  Yp = model.P*F;
