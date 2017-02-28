cfg.maxNodes = [10,10];
cfg.maxWeightNorms = 100*ones(size(cfg.maxNodes)); %capital lambda
cfg.pnorm = 1;
cfg.maxWeightMag = 100*ones(size(cfg.maxNodes));
cfg.maxBiasMag = 10;
cfg.complexityRegWeight = .1*ones(size(cfg.maxNodes));
cfg.normRegWeight = 0.01*ones(size(cfg.maxNodes)); %beta
cfg.augment = true;
cfg.augmentLayers = true;
cfg.activationFunction = 'tanh';
cfg.numEpochs = 50;

cfg.lossFunction = 'MSE';
cfg.surrogateLoss = 'none';

% create synthetic data set
x = 10*randn(44100,1);
y =  x.^3;

 %figure,scatter(x,y)
[adaParams,history] = adanet(x, y, cfg);

%% Visualize when nodes are added
vl = 1; %which hidden layer to visualize
plot(adaParams.lossStore); hold on;
plot(history.newN(:,1),history.newN(:,vl+1).*adaParams.lossStore(history.newN(:,1)).','g*')
ylim([0.01,0.5])
hold off;
% adanet_plot(adaParams,[-2,-2;2,2],200,x,y)

%% Visualize weight modifications over time
paramsT.activationFunction = history.activationFunction;
paramsT.augment = adaParams.augment;
paramsT.augmentLayers = adaParams.augmentLayers;
paramsT.W_bias = adaParams.W_bias;
Tselect = [cfg.numEpochs];
for ti = Tselect
    paramsT.W = history.Wt{ti+1};
    paramsT.u = history.ut{ti+1};
    paramsT.numLayers = history.numL(ti);
    paramsT.numNodes = history.numN(ti,:);
    adanet_plot(paramsT,[-2,-2;2,2],200,x,y)
%     plot(paramsT.W{3})
    title(sprintf('End of Epoch: %d, NumNodes: (%d,%d,%d)',ti,paramsT.numNodes))
%     pause;
end

