function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	

    % ============================================================
	error = (X*theta)-y;
    % Save the cost J in every iteration   
	for j = 1:length(theta)	
		x = X(:,j);
		theta(j) = theta(j)-((alpha*sum(error.*x))/m);
		
	end
	%fprintf('theta vector is -');
	%disp(theta);
	J_history(iter) = computeCost(X, y, theta);
	fprintf('Cost computed J = %f\n', J_history(iter));
end

end
