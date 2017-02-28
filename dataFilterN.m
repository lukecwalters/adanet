%% dataFilterN.m
%Description:
%filters a data set by removing samples outside of a
%desired range specified for each dimension in the data.
%INPUTS:
% data
%   - a M samples x Nd dimension data set
% spans
%   - a 2xNs matrix where each column contains the upper and lower bounds
%   for the Nth dimension for data. The order of the bounds doesn't matter.
%   
%   notes: Ns <= Nd.  The data set will not be filtered along dimensions (columns)
%   greater than Ns.  Filtering removes entire rows, and maintains the relative
%   sequence of the remaining samples
%  
%or - a length 2*Nd vector of spans, where the 1st 2 elements specify
%     the [lower upper] bounds of the 1st dimension, the next two elements
%     for the second dimension, and so on...
%OUTPUTS:
% data
%   - a Mn samples x Nd dimension data set, where Mn <= M
%Luke Walters - 03/18/09
%%
function data = dataFilterN(data,spans)

if numel(spans)==2
    spans = spans(:);
    Nd = 2;
    sv = 1; %spans is a vector
elseif min(size(spans)) == 1
    Nd = length(spans)/2;
    spans = spans(:)';
    sv = 1; %spans is a vector
else
    Nd = size(spans,2);
    sv = 0;
end
    

for dim = 1:Nd
    if sv
        pair = spans((dim-1)*2+1:2*dim);
        xMin = min(pair);
        xMax = max(pair);
    else
        xMin = min(spans(:,dim));
        xMax = max(spans(:,dim));
    end
    
    [r,c] = find(data(:,dim) > xMax);
    data(r,:) = [];
    [r,c] = find(data(:,dim) < xMin);
    data(r,:) = [];
end