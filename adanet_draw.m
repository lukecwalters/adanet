function adanet_draw(params)

numInputNodes = params.numInputNodes;
numOutputNodes = params.numOutputNodes;
W = params.W;
u = params.u;
numNodes  = params.numNodes;
numLayers = params.numLayers;

layersX = 0:numLayers+1;
layersY = cell(length(numNodes)+2,1);
nodeSpacing = 1;
if params.augment
    numInputNodes = numInputNodes + 1;
end
if params.augmentLayers
    for k = 1:length(numNodes)
        numNodes(k) = numNodes(k) + 1;
    end
end
allNodes = [numInputNodes,numNodes(:)',numOutputNodes];
h = figure(100);
for lyr = 1:length(layersY)
    if lyr ~= length(layersY)
        layersY{lyr} = (0:nodeSpacing:allNodes(lyr)*nodeSpacing-nodeSpacing ) - .5*(allNodes(lyr)*nodeSpacing-nodeSpacing);
    else
        ub=min(cell2mat(cellfun(@min,layersY,'UniformOutput',false)))-15*nodeSpacing;
        layersY{lyr} =ub:-nodeSpacing:ub-nodeSpacing*allNodes(lyr)+nodeSpacing;
    end
    figure(h), plot(layersX(lyr)*ones(size(layersY{lyr})),layersY{lyr},'o'), hold on
end
maxes = zeros(length(u),1);
for k = 1:length(u)
    maxes(k) = max(max(abs(u{k})));
end
maxu = max(maxes);
maxes = zeros(length(W),1);
for k = 1:length(W)
    maxes(k) = max(max(abs(W{k})));
end
maxw = max(maxes);
maxw
tol = .0001;
for k = 1:length(u)
    uk = u{k};
    
    [Nrows,Ncols] = size(uk);
    kx = layersX(k);
    k2x = layersX(k+1);
    ky = layersY{k};
    k2y = layersY{k+1};
    
    for r = 1:Nrows
        for c = 1:Ncols
            urc = uk(r,c)/maxu;
            if urc>0
                baseColor = [0, 0.4470, 0.7410];
            else
                baseColor = [0.8500 , 0.3250, 0.0980];
            end
            if abs(urc)*maxu > tol
                baseColor = (1-baseColor)*(1-abs(urc)) + baseColor;
                figure(h), line([kx,k2x],[ky(r),k2y(c)],'color',baseColor), hold on
            end
        end
    end
end


for k = 1:length(W)
    Wk = W{k};
    kx = layersX(k+1);
    ky = layersY{k+1};
    kx2 = layersX(end);
    ky2 = layersY{end};
    
    if params.augmentLayers,
        maxIdx = length(ky)-1;
    else
        maxIdx = length(ky);
    end
        
    for ny = 1:maxIdx
        wrc = Wk(ny)/maxw;
        if wrc>0
            baseColor = [0.9290  ,  0.6940 ,   0.1250];
            %baseColor = [0, 0.4470, 0.7410];
        else
            baseColor = [0.4940  ,  0.1840  ,  0.5560];
            %baseColor = [0.8500 , 0.3250, 0.0980];
        end
        if abs(wrc)*maxw > tol
            baseColor = (1-baseColor)*(1-abs(wrc)) + baseColor;
            figure(h), line([kx,kx2],[ky(ny),ky2],'color',baseColor), hold on
        end
    end
   
end
