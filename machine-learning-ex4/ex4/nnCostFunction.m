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

D2 = zeros(num_labels, hidden_layer_size+1); % 10*26
D1 = zeros(hidden_layer_size, input_layer_size+1); % 25*401

Y = zeros(m,num_labels) ; % 5000X10 matrix of zeros
 
c = 1:num_labels;
Y = y==c;

h = zeros(m,num_labels);
for i = 1:m
	a1 = [1 X(i,:)]; % 1*401
	z2 = a1*Theta1'; %(1*401)*(401*25)
	a2 = [1 sigmoid(z2)]; % 1*26
	
	z3 = a2*Theta2';
	a3 = h(i,:) = sigmoid(z3);  % 1*10
	
	d3 = a3-Y(i,:);	% 1*10
	d2 = (d3*Theta2) .* [1 sigmoidGradient(z2)];		% (1*10) * (10*26)
	d2_without_bias = d2(2:end);
	
	D2 = D2 + d3'*a2;	% (1*10)' * (1*26)
	D1 = D1 + d2_without_bias'*a1;	% (1*25)' * (1*401)
	
end;
J = sum(sum(((-Y.*log(h)) - ((1-Y).*log(1-h)))/m, 2));

J += (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

Theta2_grad = D2/m; % 10*26
Theta1_grad = D1/m; % 25*401

% =========================================================================
%X = [ones(m,1) X]; % m*401
%z2 = X*Theta1';
%a2 = sigmoid(z2); % m*25
%a2 = [ones(m,1) a2];
%z3 = a2*Theta2';
%a3 = h = sigmoid(z3);  % m*10
%
%c = 1:num_labels;
%y = y==c;
%%for i = 1:m
%%	temp_y = y(i,:);
%%	temp_h = h(i,:);
%%	
%%	J += ((-temp_y*(log(temp_h))') - ((1-temp_y)*(log(1-temp_h))'))/m;
%%	%J = J+temp_J;
%%end;
%
%J = sum(sum(((-y.*log(h)) - ((1-y).*log(1-h)))/m, 2));
%%J = sum(sum(((-y.*log(h)) - ((1-y).*log(1-h)))/m));
%
%J += (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));
%
%D3 = h-y; % m*10
%D2 = (D3*Theta2); % m*26
%D2 = (D2 .* a2).*(1-a2);
%%D2 = D2 .* [ones(m,1) sigmoidGradient(z2)];
%%D2 = sigmoidGradient(z2); % m*26
%
%D2 = D2(:,2:end); % m*25
%
%Theta2_grad = (D3'*a2)/m; % 10*26
%Theta1_grad = (D2'*X)/m; % 25*401
%
%Theta2_grad(:,2:end) += Theta2_grad(:,2:end) + ((Theta2(:,2:end))/m);
%Theta1_grad(:,2:end) += ((Theta1(:,2:end))/m);







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
