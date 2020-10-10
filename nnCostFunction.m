function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

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

% Part 1:

X = [ones(m,1) X]; % 5000x401
A1 = X;
Z2 = X*Theta1';
A2 = sigmoid(Z2); % 5000x25
A2 = [ones(size(A2,1),1) A2]; % 5000x26
Z3 = A2*Theta2';
A3 = sigmoid(Z3); % 5000x10

y_new = zeros(size(y,1), num_labels); % 5000x10

for i = 1:size(y,1)
  y_new(i,y(i,1)) = 1;
endfor

y = y_new;  

% J = (1/m)*sum((-y'*log(A3))-((1-y')*log(1-A3))); % 10x5000x5000x10 

for i = 1:num_labels
  J += (1/m)*[(-y(:,i)'*log(A3(:,i)))-((1-y(:,i)')*log(1-A3(:,i)))]; % 1x5000x5000x1
endfor

% theta1 is 25x401. so do not consider first column of theta1 and theta2. 
% here theta is 2 dim with corresponding multiplier for each input in column and not row.

% one way to write solution is below
% J += (lambda/(2*m))*(cumsum(sumsq(Theta1(:,2:end)),2)(:,end) + cumsum(sumsq(Theta2(:,2:end)),2)(:,end));

% simpler way to write soln is below


Theta1Sq = cumsum(sumsq(Theta1(:,2:end)),2)(:,end);
Theta2Sq = cumsum(sumsq(Theta2(:,2:end)),2)(:,end);

J += (lambda/(2*m))*(Theta1Sq + Theta2Sq);

% backpropagation code starts here
% mention all A and Z to make it easier

Delta1 = 0;
Delta2 = 0;

for i = 1:m
  x = X(i,:); % 1x401
  a1 = x;
  z2 = x*Theta1'; % 1x25
  a2 = sigmoid(z2);
  a2 = [1 a2]; % 1x26
  z3 = a2*Theta2'; % 1x10
  a3 = sigmoid(z3);
  delta3 = a3 - y(i,:); % 1x10
  delta2 = (delta3*Theta2(:,2:end)).*(sigmoidGradient(z2)); % 1x10 x 10x25 .x 1x25 = 1x25
  Delta1 += delta2'*a1; % 25x1 x 1x401 = 25x401
  Delta2 += delta3'*a2; % 10x1 x 1x26 = 10x26
endfor

Theta1_grad (:,1) = Delta1(:,1)/m;
Theta1_grad (:,2:end) = Delta1(:,2:end)/m + (lambda/m)*Theta1(:,2:end);
Theta2_grad (:,1) = Delta2(:,1)/m;
Theta2_grad (:,2:end) = Delta2(:,2:end)/m + (lambda/m)*Theta2(:,2:end);






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
