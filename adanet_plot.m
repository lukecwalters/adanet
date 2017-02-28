%% adanet_plot.m
%
% 
%INPUTS
% C        adaParams struct
%          
% spans     a 2xN matrix where each column specifies the bounds in the
%           feature space over which to plot the boundary
% Nbins     number of bins to use in each dimension for plotting the
%           boundary
%
% (optional)
%  X,y      X is M samples x N dim data set, y is a Mx1 vector of sample 
%           labels.  X and y should be provided together.
%           Providing X and y will produce a scatterplot over the space
%  w        a Mx1 vector of weights on the samples, X and y can be passed
%           with or without w, providing w will plot a weighted histogram
%           of the samples in each class with the scatterplot
% soft      a control flag, set to 0 (default) to view hard decision
%           boundary, set to 1 to see soft boundary
% valid examples.
% plotClassifier(C,spans,Nbins)
% plotClassifier(C,spans,Nbins,X,y)
% plotClassifier(C,spans,Nbins,X,y,w)
%
% Luke Walters 3/08/2009
%%
function varargout = adanet_plot(C,spans,Nbins,varargin)

if nargin <3
    error('Wrong number of arguments')
end
N = size(spans,2); %number of dimensions
Nc = 1; % number of classifiers passed
soft = 0;
if nargin >3
    X = varargin{1};
    y = varargin{2};

    if N ~= size(X,2)
        error('spans dimension does not fit data set')
    end

    if nargin > 5
        w = varargin{3};
        w = makecol(w);
        if nargin >6
           soft = varargin{4}; 
        end
    else
        w = ones(size(y));
    end
    
    lX = [X y w]; %group labels with data set
    %filter data set according to spans
    lX = dataFilterN(lX,spans);
    X = lX(:,1:end-2);
    y= lX(:,end-1);
    w= lX(:,end);
    w = w/sum(w);
end
%preallocate space
if N>1
    space = zeros(ones(1,N)*Nbins); %feature space array
else
    space = zeros(Nbins,1);
end
%find intended bounds on feature space,
%and then define coordinate axes
x = zeros(Nbins,N);
grid = zeros([ones(1,N)*Nbins N]);


outStr = '[';
inStr = '';
colonStr = '';
for d = 1:N
    xmin = min(spans(:,d));
    xmax = max(spans(:,d));
    x(:,d) =  xmin:(xmax-xmin)/(Nbins-1):xmax;
    
    if d<N
        outStr = sprintf('%sX%d,',outStr,d);
        inStr = sprintf('%sx(:,%d),',inStr,d);
        colonStr = sprintf('%s:,',colonStr);
    else
        outStr = sprintf('%sX%d]',outStr,d);
        inStr = sprintf('%sx(:,%d)',inStr,d);
        colonStr = sprintf('%s:',colonStr);
    end
    
end
eval(sprintf('%s=ndgrid(%s);',outStr,inStr));

status = 'Constructed Grid';

for d = 1:N
    eval(sprintf('grid(%s,%d)=X%d;',colonStr,d,d));
end

Xcol = reshape(grid,Nbins^N,N);
clear grid

for cIdx = 1:1
    try
        if C.backprop
            yout = backprop_predict(C,Xcol);
        else
            yout = adanet_predict(C,Xcol);
        end
    catch
        yout = adanet_predict(C,Xcol);
    end
   
    
    
    if cIdx == Nc
        clear Xcol
    end
    if soft
         
%          if strcmp(C.type,'lm')
%              ds = abs(yout)/sqrt(sum(C.a.^2));
%              yout(yout>=0) = 1;
%              yout(yout<0)= 1./(1+1000*ds(yout<0));
%          else
           yout(yout>=0) = 1;
           yout(yout<0) = 1./(1-.1*yout(yout<0));
%          end
    end
    
         
    space = real(reshape(yout,size(space)));
    
    if nargout
        varargout{1}=space;
        varargout{2}=x;
    else
        clear yout
        if N > 1
            figure(49+cIdx), eval(sprintf('imagesc(%s,space\'')',inStr)), colormap(bone)
            title('Classifier Decision Boundary')
        else
            figure(49+cIdx),eval(sprintf('plot(%s,space\'')',inStr))
        end
        if nargin > 3
            hold on
            eWeight = 0;
            if eWeight
                %cyan
                tCData = repmat([0 1 1],[sum(y==1)  1]);
                wt = w(y==1);
                wt = wt./max(wt);
                
                wt = wt.^(.1); % compress dynamic range of weights
                tCData = tCData.*repmat(wt,[1,3]);clear wt;
                % red
                jCData = repmat([1 0 0],[sum(y==-1) 1]);
                wj = w(y==-1);
                wj = wj./max(wj);
                wj = wj.^(.1);
                jCData = jCData.*repmat(wj,[1,3]);clear wj;
            else
                tCData = [0 1 1];
                jCData = [1 0 0];
            end
            
            figure(49+cIdx),scatter(X(y==1,1),X(y==1,2),2,'CData',tCData), hold on
            figure(49+cIdx),scatter(X(y==-1,1),X(y==-1,2),1,'CData',jCData)
            %            end
            %         if nargin > 5
            %             maskDim = ones(1,N) * round(.03*Nbins);
            %             wHist1 = wHist2D(X(y==1,:),w(y==1,:),spans,Nbins,maskDim);
            %             wHist2 = wHist2D(X(y~=1,:),w(y~=1,:),spans,Nbins,maskDim);
            %
            %             figure(49+cIdx),contour(wHist1.x(:,1),wHist1.x(:,2),wHist1.sHist','c'),  hold on,
            %             figure(49+cIdx),contour(wHist2.x(:,1),wHist2.x(:,2),wHist2.sHist','r'), hold on,
            %         end
            hold off
        end
    end
end
if Nc>1
    [m,n] = bestTile(Nc);
    arrangeFig(n,m);
end
status = 'Ran Ok';





