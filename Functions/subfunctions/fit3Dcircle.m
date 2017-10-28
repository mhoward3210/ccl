function [centre, radius, normal] = fit3Dcircle(x,y,z)
    if iscolumn(x) && iscolumn(y) && iscolumn(z)
        % sphere system of equations: As * xs = ys
        As = [x y z ones(size(x))]; ys = (x.^2)+(y.^2)+(z.^2);
        % plane system of equations: Ap * xp = yp
        Ap = [x y ones(size(x))]; yp = z;
        % circunference system of equations: intersection between sphere and
        % plane: Ac * xc = yc
        Ac = blkdiag(As,Ap); yc = [ys; yp];
        % regression:
        xc = regress(yc,Ac);
        % define intermediate constants
        centre_s = xc(1:3)./2; % centre of sphere
        radius_s = sqrt(xc(4)+centre_s.'*centre_s); % radious of the sphere
        n = [-xc(5); -xc(6); 1];
        d0 = -xc(7);
        % Find centre and normal:
        k = (-(n'*centre_s)-d0)/(n'*n);
        centre = centre_s+k*n; % centre of the circle
        normal = n./norm(n);
        radius = sqrt((radius_s^2)-(centre_s-centre)'*(centre_s-centre)); % radious of the circle
    else
        error('fit3Dcircle(x,y,z): input vectors x, y, and z are expected to be column vectors');
    end

end