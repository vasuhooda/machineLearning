data = load('ex1data1.txt');

% population
X = data(:,1); 

% profit
y = data(:,2);

% m
m = length(y);

%plot
figure; % open a new figure window
plot(X, y, 'rx', 'MarkerSize', 10);
ylabel('Profit in $10,000s');
xlabel('Population of City in 10,000s');

% initialize theta vector
theta = zeros(3,1);

% new hypothesis - square root function
Xtest = [ones(m,1), X, power(X,1/2)];


% value for testing theta
theta1 = [0;0.1;12];
hold on;
plot(X, Xtest*theta1, '+');

testCost1 = cost(Xtest, y, theta1)


% Gradient Descent Settings
iterations = 1500;
alpha = 0.01;

[theta, cost_history,theta_history] = gradientDescent(Xtest, y, theta, alpha, iterations);
theta
% plot graph according to parameters received by gradient descent algorithm
hold on; 
plot(X, Xtest*theta, 'g*')
hold off;

figure
plot(cost_history)


% cost
function J = cost(Xtest,y,theta)
m = length(y);
errorSquare = power((Xtest * theta - y),2);
J = (1 / (2*m)) * sum(errorSquare);
end

% batch gradient descent
function [theta, cost_history, theta_history] = gradientDescent(Xtest, y, theta, alpha, iterations)

%initialize m for training data set

%initialize a vector to save value of cost on each iteration

%the cost should go down with each iteration

%we can plot this cost over number of iterations to get a intution about
%the success of training algorithm

m = length(y); % number of training examples
cost_history = zeros(iterations, 1);
theta_history = zeros(iterations,3);

for iter = 1:iterations
    
    htheta = Xtest * theta;
    
    % update both Theta i.e. parameters for learning algorithm
    % simultaneoously
    % .* Xnew(:,1) - this is not necessary for first parameter, but it is
    % good to use it, becasue this sets a pattern which is useful in
    % scaling

    theta1 = theta(1) - alpha / m * sum((htheta - y) .* Xtest(:,1));
    theta2 = theta(2) - alpha / m * sum((htheta - y) .* Xtest(:,2));
    theta3 = theta(3) - alpha / m * sum((htheta - y) .* Xtest(:,3));
    %update theta
    theta = [theta1; theta2;theta3];
    
    %update cost history vector
    cost_history(iter) = cost(Xtest, y, theta);
    
    theta_history(iter,:) = [theta1, theta2, theta3];
    
end
end