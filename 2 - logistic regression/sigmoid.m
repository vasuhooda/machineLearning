function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

size_matrix = size(z);
size_row = size_matrix(1);
size_column = size_matrix(2);

for i = 1 : size_row
    for j = 1 : size_column
        z(i,j) = 1/(1+power(exp(1),-z(i,j)));
    end
end

g = z;

% =============================================================

end
