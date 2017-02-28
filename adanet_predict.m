function pred = adanet_predict(params, Xdata)
if ~isempty(params.featureMap);
    Xdata = params.featureMap(Xdata);
end
if params.augment,
    Xdata = [ones(size(Xdata,1),1),Xdata];
end
if params.augment && params.augmentLayers,
    Xdata = Xdata(:,2:end);
end

% Compute adanet prediction
pred = 0;
for k = 1:params.numLayers
    if k == 1
        Hk = Xdata;
        actFunc = 'none';
    else
        Hk = H;
        actFunc = params.activationFunction;
    end
    if params.augmentLayers,
        Hk = [ones(size(Hk,1),1),Hk];
    end
    H = activation(Hk,actFunc)*params.u{k};
    for j = 1:params.numNodes(k)
        pred = pred + params.W{k}(j)*H(:,j);
    end
    pred = pred + params.W_bias;
end

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