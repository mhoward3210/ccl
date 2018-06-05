function ccl_rob_vis_move (figh, R, X, L,varargin)
% ccl_rob_vis_move (R, X, L)
% Visualisation of the arm movement
% Input:
%   R               Task space movements
%   X               Joint space movements
%   L               Arm link length
%   varargin        Arguments for figure title and position
dim_n = length(R(1,:)) ;
xmin = -1.5 ; xmax = 2 ;
ymin = -2.5 ; ymax = 2.5 ;

fig_handle = figure(figh); hold on
if nargin > 3
    title_txt = varargin{1};title(title_txt);
    pos       = varargin{2};set(fig_handle, 'Position', pos);
end
axis equal
xlim([xmin,xmax]) ; ylim([ymin,ymax]) ;
plot( [R(1,:)], [R(2,:)], 'LineWidth', 4 ) ;
xlabel('x');
ylabel('y');
% stroboscopic plot of arm
for i=1:round(dim_n/25):dim_n
    c = ((dim_n-i)/dim_n) * ones(1,3) ;
    ccl_rob_plot_arm (X(:,i), L, c) ;
    pause (0.1) ;
    axis tight;
end
hold off
end
