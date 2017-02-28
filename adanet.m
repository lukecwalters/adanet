%% Main function for adanet
function [adaParams,history] = adanet(Xdata,ydata, cfg)
history.newL = [];
history.numL = [];
history.newN = [];
history.numN = [];
history.jk_best = [];
history.Wt = {[]};
history.ut = {[]};
history.W_bias = [];
history.activationFunction = cfg.activationFunction;

%Xdata = [Xdata, Xdata.^2, prod(Xdata,2)];
if ~isempty(cfg.featureMap);
    Xdata = cfg.featureMap(Xdata);
end
if cfg.augment,
    Xdata = [ones(size(Xdata,1),1),Xdata];
end
if cfg.augment && cfg.augmentLayers,
    Xdata = Xdata(:,2:end); %revert Xdata! We will account for it in NewNodes
end

numExamples = size(Xdata,1); % number of training examples (m)
numInputNodes = size(Xdata,2);

if size(ydata,1) ~= numExamples
    error('Xdata and ydata must have same number of examples (i.e. rows)')
end

adaParams = adanet_init(numExamples,numInputNodes,cfg); %see below
adaParams.lossStore = [];
adaParams.numInputNodes = numInputNodes;
adaParams.numOutputNodes = size(ydata,2);
T = cfg.numEpochs; % num rounds from paper

xloss_end = round(numExamples/2);
step=0;
for t = 1:T
    idcs_t = randperm(numExamples); % shuffle samples uniformly every epoch
    % split data into two mini batches, one for loss, one for finding new
    % nodes (Appendix E, page 17)
    %     Xloss = Xdata(1:xloss_end,:);
    %     Xnewnodes = Xdata(xloss_end+1:end,:);
    lossIdcs = idcs_t(1:xloss_end);
    newIdcs = idcs_t(xloss_end:end);
    % N = sum(adaParams.numNodes);
    
    % Compute Forward Pass + Complexity Measures
    numLayers = adaParams.numLayers;
    
    H = cell(numLayers,1);  % hidden unit values per layer
    Rloss = zeros(numLayers+1,1); % complexity measures per layer
    Rnew = Rloss;
    W = adaParams.W;
    u = adaParams.u;
    for k = 1:numLayers
        if k == 1
            Hk = Xdata;
            actFunc = 'none';
        else
            Hk = H{k-1};
            actFunc = adaParams.activationFunction;
        end
        if adaParams.augmentLayers,
            Hk = [ones(size(Hk,1),1),Hk];
        end
        H{k} = activation(Hk,actFunc)*u{k};
        % Calculate loss function
        switch cfg.lossFunction
            case 'binary'
                Rloss(k) = RademacherComplexity(H{k}(lossIdcs,:));
                Rnew(k) = RademacherComplexity(H{k}(newIdcs,:));
%             case 'MSE'
%                 Rloss(k) = GaussianComplexity(H{k}(lossIdcs,:)); %equivalent with gaussian, instead of binary, noise
%                 Rnew(k) = GaussianComplexity(H{k}(newIdcs,:));
            otherwise
                error('loss not supported')
        end
        Rloss(k) = reg(Rloss(k),adaParams.complexityRegWeight(k),adaParams.normRegWeight(k));
        Rnew(k) = reg(Rnew(k),adaParams.complexityRegWeight(k),adaParams.normRegWeight(k));
    end
    
    % score the existing nodes in the network
    [d_kj, d_bias] = ExistingNodes;  % pg 7, line 3
    [dn_k, un_k] = NewNodes; %pg 7, line 4
    [jk_best, e_t, numLayers, numNodes] = BestNode; %pg 7, line 5
    adaParams.numLayers = numLayers;
    adaParams.numNodes = numNodes;
    adaParams.u = u;
    % Move forward a step and update distributions
    Wt = ApplyStep(jk_best,adaParams.surrogateLoss);
    [Dt,St] = updateDistribution(Wt);
    % Pack things back into adaParams
    adaParams.D = Dt;
    adaParams.S = St;
    adaParams.W = Wt;
    adaParams.errLoss = e_t;
    
    % Store history for debugging only
    history.Wt{end+1} = adaParams.W;
    history.ut{end+1} = adaParams.u;
    history.numL(end+1) = numLayers;
    history.numN(end+1,:) = adaParams.numNodes;

    fComplete = 0;
    for k = 1:numLayers
        for j = 1:numNodes(k)
            fComplete = fComplete + W{k}(j)*H{k}(:,j);
        end
    end
    fComplete = fComplete + adaParams.W_bias;
    switch cfg.lossFunction
        case 'binary'
            adaParams.lossStore(end+1) = mean(slfunc(1-ydata.*fComplete, adaParams.surrogateLoss));
%         case 'MSE'
%             adaParams.lossStore(end+1) =mean(abs(ydata - fComplete).^2);
    end
    plotEpochs = 5;
    if mod(step,plotEpochs)==0
        figure(2), plot(adaParams.lossStore)
        titleStr = 'numNodes = (';
        for ni = 1:length(numNodes)-1,
            titleStr = sprintf([titleStr,'%d,'],numNodes(ni));
        end
        titleStr = sprintf([titleStr,'%d)'],numNodes(end));
        title(titleStr)
        pause(.1)
    end
    step = step+1;
  
    
    
end

    %% Existing-nodes function
    function [d_kj,d_bias] = ExistingNodes
        errLoss = adaParams.errLoss;
        Dloss = adaParams.D(lossIdcs);
        Dloss = Dloss./sum(Dloss);
        m = length(lossIdcs);
        d_kj = cell(numLayers,1);
        % initial scores for all nodes
        for k = 1:numLayers
            d_kj{k} = zeros(numNodes(k),1);
        end
        d_bias = 0;
        
        S = adaParams.S;
        for k = 1:numLayers
            Ck = cfg.maxWeightMag(k);
            % inference using existing parameters
            
            switch cfg.lossFunction
                case 'binary'
                    errLoss{k} = Ck/2 * (1-1/Ck*bsxfun(@times, ydata(lossIdcs,:),H{k}(lossIdcs,:))'*Dloss);
%                 case 'MSE'
%                     errLoss{k} = Ck/2 *(1 - 1/Ck * (abs(bsxfun(@minus, ydata(lossIdcs,:),H{k}(lossIdcs,:))).^2)'*Dloss);
                otherwise
                    error('loss not supported');
            end
            
            Wk = W{k}(1:numNodes(k));
            % nodes with non-zero weights
            j_nzw = Wk ~= 0;  % pg 7, Existing Nodes, line 4
            % nodes with scores of zero
            j_zs = ~j_nzw & abs(errLoss{k} - Ck/2) <= Rloss(k)*m/(2*S); % pg 7, Existing Nodes, line 6
            % remaining nodes
            j_other = ~j_nzw&~j_zs; % line 8
            % compute and assign existing node scores
            d_kj{k}(j_other) = errLoss{k}(j_other) - Ck/2 - sign(errLoss{k}(j_other)-Ck/2)*Rloss(k)*m/(2*S);
            d_kj{k}(j_nzw) = errLoss{k}(j_nzw) - Ck/2 + sign(Wk(j_nzw))*Rloss(k)*m/(2*S);
  
        end
        Ck_bias = adaParams.maxBiasMag;
        % inference using existing parameters
        
        switch cfg.lossFunction
            case 'binary'
                errLoss_bias = Ck_bias/2 * (1-1/Ck_bias*bsxfun(@times, ydata(lossIdcs,:),ones(size(ydata(lossIdcs,:))))'*Dloss);
%             case 'MSE'
%                 errLoss_bias = Ck_bias/2 * (1-1/Ck_bias*(abs(bsxfun(@minus, ydata(lossIdcs,:),ones(size(ydata(lossIdcs,:))))).^2')*Dloss);
            otherwise
                error('loss not supported');
        end
        
        Wk_bias = adaParams.W_bias;
        % nodes with non-zero weights
        j_nzw = Wk_bias ~= 0;  % pg 7, Existing Nodes, line 4
        % nodes with scores of zero
        j_zs = ~j_nzw & abs(errLoss_bias - Ck_bias/2) <= 0;%Rloss_bias*m/(2*S); % pg 7, Existing Nodes, line 6
        % remaining nodes
        j_other = ~j_nzw&~j_zs; % line 8
        % compute and assign existing node scores
        d_bias(j_other) = errLoss_bias(j_other) - Ck_bias/2 - 0;%sign(errLoss_bias(j_other)-Ck_bias/2)*Rloss_bias*m/(2*S);
        d_bias(j_nzw) = errLoss_bias(j_nzw) - Ck_bias/2 + 0;%sign(Wk_bias(j_nzw))*Rloss_bias*m/(2*S);
    end

    %% New nodes function
    function [dn_k, un_k] = NewNodes
        global errNew
        numNodes = adaParams.numNodes;
        maxNodes = adaParams.maxNodes;
        q = adaParams.qnorm;
        p = cfg.pnorm;
        idx = min(numLayers+1,adaParams.maxLayers);
        errNew = cell(idx,1);
        un_k = cell(idx,1);
        dn_k = cell(idx,1);
        
        Dnew = adaParams.D(newIdcs);
        Dnew = Dnew./sum(Dnew);
        m = length(newIdcs);
        S = adaParams.S;
        for k = 1:idx                                       % pg 8, Fig 4, Line 1
            Ck = cfg.maxWeightMag(k);               
            Lambda_k = cfg.maxWeightNorms(k);
            if k==1, 
                nodeNum_km1 = numInputNodes;
                Hk = Xdata;
                actFunc = 'none';
            else
                nodeNum_km1 = numNodes(k-1);
                Hk = H{k-1};
                actFunc = adaParams.activationFunction;
            end
            if (numNodes(k) < maxNodes(k)) && (nodeNum_km1 > 0)   %pg 8, Fig 4, line 2
                if adaParams.augmentLayers,
                    Hk = [ones(size(Hk,1),1),Hk];
                end
                switch cfg.lossFunction
                    case 'binary'
                        M_kminus1 = bsxfun(@times,ydata(newIdcs,:),activation(Hk(newIdcs,:),actFunc))'*Dnew;
                        M_qnorm = norm(M_kminus1,q);
                        errNew{k} = Ck/2*(1-Lambda_k/Ck * M_qnorm);  % pg 8, Fig 4, line 3
%                     case 'MSE'
%                         M_kminus1 =  (abs(bsxfun(@minus,ydata(newIdcs,:),activation(Hk(newIdcs,:),actFunc))).^2)'*Dnew;
%                         M_qnorm = norm(M_kminus1,q);
%                         errNew{k} = Ck/2*(1-Lambda_k/Ck * M_qnorm);  % pg 8, Fig 4, line 3
                end
                
                if p == 1,
                    [~,bMax] = max(abs(M_kminus1));
                    M_normalized = zeros(size(M_kminus1));
                    M_normalized(bMax) = 1/abs(M_kminus1(bMax));
                    un_k{k} = Lambda_k * M_normalized.*sign(M_kminus1);
                else
                    un_k{k} =  Lambda_k * (abs(M_kminus1).^(q-1)).*sign(M_kminus1)/(M_qnorm^(q/p) ); % line 4
                end
            else
                errNew{k} = 1/2*Ck;  % line 5,
            end
            
            if abs(errNew{k} - 1/2*Ck) <= Rnew(k)*m/(2*S)   % line 6
                dn_k{k} = 0;                                               % line 7
            else
                dn_k{k} = errNew{k} - Ck/2 - sign(errNew{k} - Ck/2)*Rnew(k)*m/(2*S);  %line 8
            end
            
        end
    end

    %% Best node function
    function [jk_best, e_t, numLayers, numNodes] = BestNode
        global errNew
        errLoss = adaParams.errLoss;
        numLayers = adaParams.numLayers;
        numNodes = adaParams.numNodes;
        % calculate score for output bias
        d_biasMax = abs(d_bias);
        % get most important new node that could exist if we augment the network
        dnAbs = cellfun(@abs,dn_k,'uniformoutput',0);
        [dnMax,kBinNew] = max(cellfun(@max,dnAbs));
        % get most important node in all current layers
        if isempty(d_kj), %should only happen on the first 
            dMax = dnMax-1; %just make sure dMax is less than dnMax
        else
            dAbs = cellfun(@abs,d_kj,'uniformoutput',0);
            [~,kBin] = max(cellfun(@max,dAbs));
            [dMax,jBin] = max(dAbs{kBin});
        end
        [~,dcase] = max([d_biasMax,dMax,dnMax]);
        
        % Pick the highest score and possibly create a new node/layer
        switch dcase
            case 1
                jk_best = 'bias';
            case 2
                jk_best = [jBin,kBin];           % line 2
            case 3
                kNew = kBinNew;                  % line 3
                if kNew > numLayers,             % line 4
                    numLayers = numLayers + 1;   % line 5
                    history.newL = t;
                end
                numNodes(kNew) = numNodes(kNew) + 1; % line 6
                jk_best = [numNodes(kNew),kNew]; % line 7
                history.newN(end+1,:) = [t,zeros(1,adaParams.maxLayers)];
                history.newN(end,kNew+1) = 1;
                history.jk_best(end+1,:) = jk_best;
                j_best = jk_best(1);
                % Augment u appropriately
                u{kNew}(:,j_best) = un_k{kNew};
                try
                    u{kNew+1}(end+1,:) = 0;
                catch
                    % do nothing
                end
                % Update H
                if kNew == 1
                    Hk = Xdata;
                    actFunc = 'none';
                    %                 H{kNew}(:,j_best) = Xdata*un_k{kNew};
                else
                    Hk = H{kNew-1};
                    actFunc = adaParams.activationFunction;
                end
                if adaParams.augmentLayers,
                    Hk = [ones(size(Hk,1),1),Hk];
                end
                H{kNew}(:,j_best) = activation(Hk,actFunc)*un_k{kNew};
                
                % Update error loss
                errLoss{kNew}(j_best) = errNew{kNew};
        end
        %
        e_t = errLoss;
        
    end
    %% Apply step
    function Wt = ApplyStep(jk_best, surrogateLoss)
        if isstr(jk_best),
            w_k0 = adaParams.W_bias;
            h_k = 1;
            reg_k = 0;
            
            f_notk = 0;
            reg_notk = 0;
            for k = 1:numLayers
                for j = 1:numNodes(k)
                    f_notk = f_notk + W{k}(j)*H{k}(:,j);
                    reg_notk = reg_notk + Rloss(k)*abs(W{k}(j));
                end
            end
            switch cfg.lossFunction
                case 'binary'
                    loss_notk = ydata.*f_notk;
%                 case 'MSE'
%                     loss_notk = abs(ydata-f_notk).^2;
            end

        else
            % apply step
            % jk_best is the coordinate that we're going to tweak, and
            % we need to determine the step size
            j_best = jk_best(1);
            k_best = jk_best(2);
            w_k0 = W{k_best}(j_best);
            %loss_function(w_k, h_k, reg_k, y, loss_notk, reg_notk, lossfunc)
            h_k = H{k_best}(:,j_best);
            reg_k = Rloss(k_best);
            
            f_notk = 0;
            reg_notk = 0;
            for k = 1:numLayers
                for j = 1:numNodes(k)
                    if (k==k_best) && (j == j_best)
                        
                    else
                        f_notk = f_notk + W{k}(j)*H{k}(:,j);
                        reg_notk = reg_notk + Rloss(k)*abs(W{k}(j));
                    end
                end
            end
             switch cfg.lossFunction
                case 'binary'
                    loss_notk = ydata.*(f_notk+adaParams.W_bias);
%                 case 'MSE'
%                     loss_notk = abs(ydata-f_notk - adaParams.W_bias).^2;
            end
            
        end
        
        % test_loss = loss_function(w_k0, h_k, reg_k, ydata, loss_notk, reg_notk, surrogateLoss);
        loss = @(w_k) loss_function(w_k, h_k, reg_k, ydata, loss_notk, reg_notk, surrogateLoss,cfg.lossFunction);
        
        % set up optimizer
        options = optimset('GradObj', 'on', 'MaxIter', 400,'MaxFunEvals',400,'Display','notify');
        [w_k,~,sucess] = fminunc(loss,w_k0,options);
%         [w_k,~,sucess] = fminunc(loss,[w_k0,adaParams.W_bias],options);
        %         Ck = adaParams.maxWeightMag(k_best);
        %         [w_k,~,sucess] = fmincon(loss,w_k0,[],[],[],[],-Ck,Ck,[],options);
        % check that optimizer succeeded
        if ~sucess
            error('fminunc did not succeed')
        end
           
        if isstr(jk_best),
            adaParams.W_bias = w_k;
            history.W_bias(end+1) = w_k;
            Wt = W;
        else
            % Overwrite weight at the (k,j)th node being considered
            W{k_best}(j_best) = w_k;
            % Normalize sum of weights to 1?? (This is part of a proof in
            % the paper)
            % Wsum = sum(cellfun(@sum,W));
            % W = cellfun(@(x) x/Wsum,W,'un',0);
            % if abs(1-sum(cellfun(@sum,W)))>1e-6,
            %     error('Normalization of weights is not behaving properly')
            % end
            Wt = W;
            history.Wt{end+1} = Wt;
        end
    end
    %% Update Distributions
    function [Dnew,Snew] = updateDistribution(Wt)
        fNew = 0;
        for k = 1:numLayers
            for j = 1:numNodes(k)
                fNew = fNew + Wt{k}(j)*H{k}(:,j);
             end
        end
        switch cfg.lossFunction
            case 'binary'
                gradArg = 1 - ydata.*(fNew+adaParams.W_bias);
%             case 'MSE'
%                 gradArg = abs(ydata-(fNew+adaParams.W_bias)).^2;
        end
        phiGrad = slgrad(gradArg,adaParams.surrogateLoss);
        % update the sum and the distribution
        Snew = sum(phiGrad);
        Dnew = phiGrad/Snew;
    end
end

%% Activation function
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

%% Complexity functions
function R = RademacherComplexity(H)

M = size(H,1);
radNoise = randn(M,1);
radNoise(radNoise <=0) = -1;
radNoise(radNoise > 0) = 1;

R = 1/M*radNoise'*sum(H,2);

end
function R = GaussianComplexity(H)

M = size(H,1);
noise = randn(M,1);

R = 1/M*noise'*sum(H,2);

end
%% Regularization parameter
function gamma = reg(R,lambda,beta)
gamma = lambda*R+beta;
end

%% Initialization function
% implementation of Init method, Figure 5,  pg 15
function adaInit = adanet_init(numExamples,numInputNodes,cfg)

% Initialize output weights (w)
maxNodes = cfg.maxNodes;
maxLayers = length(cfg.maxNodes);
W = cell(maxLayers,1);
for k = 1:maxLayers
    W{k} = zeros(maxNodes(k),1);
end

% Assign current number of nodes and layers
numLayers = 0;    % number of hidden layers (no input or output layer considered)
numNodes = zeros(length(maxNodes),1); %number of nodes in the hidden layers

% Initialize feed-in weights (u)
u = cell(1,1);
% u{1} = zeros(numInputNodes,1);

% Create initial distribution (uniform)
D = 1/numExamples * ones(numExamples,1);

adaInit.W = W;
adaInit.W_bias = 0;
adaInit.u = u;
adaInit.D = D;
adaInit.S = numExamples; 
adaInit.errLoss = W; %same shape as W (maybe??)
adaInit.numLayers = numLayers;
adaInit.maxLayers = maxLayers;
adaInit.numNodes = numNodes;
adaInit.maxNodes = maxNodes;
%cfg.pnorm = max(1.001,cfg.pnorm);
adaInit.qnorm = 1/(1-1/cfg.pnorm);
adaInit.complexityRegWeight = cfg.complexityRegWeight; %lambda
adaInit.normRegWeight = cfg.normRegWeight; %beta
adaInit.surrogateLoss = cfg.surrogateLoss;
adaInit.lossFunction = cfg.lossFunction;
adaInit.maxWeightMag = cfg.maxWeightMag;
adaInit.maxBiasMag = cfg.maxBiasMag;
adaInit.activationFunction = cfg.activationFunction;
adaInit.augment = cfg.augment;
adaInit.augmentLayers = cfg.augmentLayers;
adaInit.featureMap = cfg.featureMap;
end

%% Loss Function
function [loss,grad] = loss_function(w_k, h_k, reg_k, y, loss_notk,reg_notk, surrloss,lossfunc)
m = size(y,1);
switch lossfunc
    case 'binary'
        farg = 1 - loss_notk - w_k*h_k.*y;
        loss = 1/m * sum(slfunc(farg,surrloss)) + reg_notk + reg_k*abs(w_k);
        grad = -1/m * sum(slgrad(farg,surrloss).*y.*h_k) + sign(w_k)*reg_k;
%     case 'MSE'
%         farg = loss_notk + abs(w_k*h_k - y).^2;
%         loss = 1/m * sum(farg) + reg_notk + reg_k*abs(w_k).^2;
%         grad = ( -1/m * sum(farg.*h_k) +sign(w_k)*reg_k);
end
end

% surrogate loss function
    function val = slfunc(x, func)
        switch func
            case 'logistic'
                val = log(1+exp(x));
            case 'exp'
                val = exp(x);
            case 'none'
                val = x;
        end
    end
% gradient of surrogate loss function
    function valG = slgrad(x, func)
        switch func
            case 'logistic'
                valG = exp(x)./(1+exp(x));
            case 'exp'
                valG = exp(x);
            case 'none'
                valG = x;%ones(size(x));
        end
    end
    
