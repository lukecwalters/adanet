function [x,y] = cubicsplit

rand('seed', 11)
randn('seed',11)

% x1means = -3:.2:3;
x1means = [-3:.05:-1,-0.9:0.3:0.05,1:.1:3];
x2means = x1means.^5;

x = [];
for j = 1:length(x1means)
    x0 = randn(200,2);
    xmean = [x1means(j), x2means(j)];
    xoffset = ones(200,1)*xmean;
    x = [x;x0+xoffset];
end

x0 = x + ones(size(x,1),1)*[3,3];
x1 = x - ones(size(x,1),1)*[3,3];
% colors
c0 = ones(size(x0,1),1)*[0.7,0,0];
c1 = ones(size(x0,1),1)*[0.2,0.2,0.7];

x = [x0;x1];
c = [c0;c1];

x = bsxfun(@times, x,1./std(x));



y = [-ones(size(x0,1),1);ones(size(x1,1),1)];