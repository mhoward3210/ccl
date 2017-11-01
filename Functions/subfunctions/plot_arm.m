function plot_arm (Q, L, C)
% Plot_arm (Q, L, C)
% A function to plot arm
% Input:
%   Q               Joint space vector
%   L               Link lengths
%   C               Color
if ~exist('C', 'var')
    C = 'r' ;
end
r1 = zeros(2,1); % base

r2 = [L(1)*cos(Q(1));
    L(1)*sin(Q(1))];
r3 = [r2(1) + L(2)*cos(Q(1)+Q(2));
    r2(2) + L(2)*sin(Q(1)+Q(2))];
plot([r1(1) r2(1)], [r1(2) r2(2)], 'LineStyle', '-', 'Color', C ) ;
plot([r2(1) r3(1)], [r2(2) r3(2)], 'LineStyle', '-', 'Color', C ) ;
end
