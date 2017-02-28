function pred = backprop_predict(params, Xdata)
if ~isempty(params.featureMap);
    Xdata = params.featureMap(Xdata);
end
% if params.augment,
Xdata = [ones(size(Xdata,1),1),Xdata];
% end
% if params.augment && params.augmentLayers,
%     Xdata = Xdata(:,2:end);
% end

% Compute feedforward prediction
m = size(Xdata, 1);
pred = zeros(size(Xdata, 1), 1);

z = Xdata*params.W{1}.';
actFunc = params.activationFunction;
for k = 2:params.numLayers,
    h = activation(z,actFunc);
    if params.augmentLayers
        h = [ones(size(h,1),1),h];
    end
    z = h*params.W{k}.';
end
hFinal = activation(z,actFunc);
[~,pred] = max(hFinal,[],2);
maxP = max(pred);
minP = min(pred);
pred = pred - (maxP + minP)/2;

if strcmp(params.lossFunction,'binary')
    pred(pred>=0)=1;
    pred(pred<0) = 0;
end
    function a = activation(h,afunc)
        switch afunc
            case 'relu'
                a = max(0,h);
            case 'tanh'
                a = tanh(h);
            case 'sigmoid'
                a = 1./(1+exp(-h));
            case 'none'
                a = h;
            otherwise
                error('activation function not supported')
        end
    end
end