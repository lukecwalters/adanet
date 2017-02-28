function [J, grad] = nnCostFunction2(nn_params, ...
                                   layer_size, ...
                                   num_labels, ...
                                   X, y, lambda, cfg)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
numLayers = length(layer_size);
Theta = cell(numLayers-1,1);
numParamsOld = 1;
for ii = 1:numLayers-1,
    numParams = (layer_size(ii)+1)*layer_size(ii+1) + numParamsOld - 1;
    Theta{ii} = reshape(nn_params(numParamsOld:numParams),layer_size(ii+1),layer_size(ii)+1);
    numParamsOld = numParams + 1;
end
% Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
%     hidden_layer_size, (input_layer_size + 1));
% 
% Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
%     num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta_grad = cell(numLayers-1,1);
for ii = 1:numLayers - 1,
    Theta_grad{ii} = zeros(size(Theta{ii}));
end
% Theta1_grad = zeros(size(Theta1));
% Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Reshape y correctly
% HERE WE NEED TO BE DONE
yaug = zeros(size(y,1),num_labels);
% nums = 1:num_labels;
nums = [-1,1];
for ii = 1:size(y,1),
    yaug(ii,:) = (nums == y(ii));
end
% Calculate h(x) = a3
Xaug = [ones(size(X,1),1),X];
z2 = Xaug*Theta1.';
a2 = activation(z2,cfg.activationFunction);
a2 = [ones(size(a2,1),1),a2];
a3 = activation(a2*Theta2.',cfg.activationFunction);
% vectorized cost function
J = sum(sum(-yaug.*log(a3)-(1-yaug).*log(1-a3)))/m;
% add regularization
J = J + lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

% Now compute the gradients
delta3 = a3-yaug;
delta2 = (delta3*Theta2(:,2:end)).*activationGrad(z2,cfg.activationFunction);

DELTA1 = delta2.'*Xaug;
DELTA2 = delta3.'*a2;

Theta1_grad = DELTA1/m;
Theta2_grad = DELTA2/m;
% add regularization
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m*Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


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

function aG = activationGrad(h,afunc)
switch afunc
    case 'relu'
        aG = h;
        aG(h >= 0) = 1;
        aG(h < 0) = 0;
    case 'tanh'
        aG = sech(h).^2;
    case 'sigmoid'
        aG = exp(-h)./(1.0 + exp(-h)).^2;
    case 'none'
        aG = ones(size(h));
    otherwise
        error('activation gradient not supported')
end
end