function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       



% inside loop updates every feature, outside loop updates mu and sigma
% matrix


for i = 1:length(X(1,:))
    for j = 1: length(X(:,i))
        X_norm(j,i) = (X(j,i) - mean(X(:,i)))/std(X(:,i))
    end
    mu(1,i) =  mean(X(:,i))
    sigma(1,i) = std(X(:,i))
end


% X(1:10,:)
% X_norm(1:10,:)
% mu
% sigma


% ============================================================

end
