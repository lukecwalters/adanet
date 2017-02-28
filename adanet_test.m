cfg.maxNodes = 50*ones(1,5);
cfg.maxWeightNorms = 10*ones(size(cfg.maxNodes)); %capital lambda
% cfg.maxWeightNorms = 1:-0.05:0.55;
cfg.pnorm = 2;
cfg.maxWeightMag = 10*ones(size(cfg.maxNodes));
% cfg.maxWeightMag = linspace(1e2,1e5,length(cfg.maxNodes));
cfg.maxBiasMag = 10;
%cfg.complexityRegWeight = .001*ones(size(cfg.maxNodes));
cfg.complexityRegWeight = logspace(-4,-3,length(cfg.maxNodes));
cfg.normRegWeight = 0.1*ones(size(cfg.maxNodes)); %beta
cfg.augment = true;
cfg.augmentLayers = true;
cfg.activationFunction = 'tanh';
cfg.numEpochs = 300;
cfg.big_lambda = 1.01;
% cfg.featureMap = @quadMap;
cfg.featureMap = [];


cfg.lossFunction = 'binary';
cfg.surrogateLoss = 'logistic';
cfg.javier = true
%% create synthetic data set
% [x,y] = cubicsplit;
addpath datasets/
 
% two spirals
xs = twospirals(44100,540*1,0,1);
ys = 2*(xs(:,end))-1;
xs = xs(:,1:end-1);
%figure(1), plot(xs(ys==1,1),xs(ys==1,2),'ro',xs(ys==-1,1),xs(ys==-1,2),'bo')
%%
[adaParams,history] = adanet(xs, ys, cfg);
%%
adanet_plot(adaParams,[-10,-10;10,10],200,xs,ys)



