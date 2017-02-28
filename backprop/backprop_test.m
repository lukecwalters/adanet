%%%%%%% MODIFIED for ADANET EXAMPLE
%%Machine Learning Online Class - Exercise 4 Neural Network Learning

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoidGradient.m
%     randInitializeWeights.m
%     nnCostFunction.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc
addpath ../datasets/
addpath ../

% config
cfg.activationFunction = 'sigmoid';
cfg.featureMap = [];
cfg.lossFunction = 'binary';

% two spirals
xs = twospirals(44100,540,0,1);
ys = 2*(xs(:,end))-1;
xs = xs(:,1:end-1);
% gscatter(xs(:,1),xs(:,2),ys)

% % circle
% xs = clusterincluster(44100,0.25,1);
% ys = 2*(xs(:,end))-1;
% xs = xs(:,1:end-1);

% bias units included in nnCostFunction.m
X = xs;
y = ys;

%% Setup the parameters you will use for this exercise
input_layer_size  = size(X,2);  % 
hidden_layer_size = 25;   % 25 hidden units
num_labels = 2;          % 2 labels (-1,1)
layer_size = [size(X,2), 30, num_labels];
numLayers = length(layer_size);

%% ================ Part 6: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

initTheta = cell(numLayers-1,1);
initial_nn = [];
for ii = 1:numLayers-1,
    initTheta{ii} = randInitializeWeights(layer_size(ii),layer_size(ii+1));
    initial_nn = [initial_nn; initTheta{ii}(:)];
end


%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 1000, 'gradobj','on');

%  You should also try different values of lambda
lambda = 0.1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda,cfg);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
% [nn_params, cost] = fminunc(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%% ================= Part 10: Draw Decision Boundary =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

paramsT.activationFunction = cfg.activationFunction;
paramsT.lossFunction = cfg.lossFunction;
paramsT.featureMap = [];
paramsT.backprop = true;
paramsT.augment = true;
paramsT.augmentLayers = true;
minX = 1.1*min(min(xs));
maxX = abs(minX);
%
paramsT.W = {Theta1,Theta2};
paramsT.numLayers = length(paramsT.W);
paramsT.numNodes = hidden_layer_size;
adanet_plot(paramsT,[minX,minX;maxX,maxX],200,xs,ys)
titleStr = sprintf('Feedforward network, numNodes: = (');
for ni = 1:length(paramsT.numNodes)-1,
    titleStr = sprintf([titleStr,'%d,'],paramsT.numNodes(ni));
end
titleStr = sprintf([titleStr,'%d)'],paramsT.numNodes(end));
title(titleStr)

% pred = predict(Theta1, Theta2, X);
% fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


