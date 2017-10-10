% Load data from a file
% Population in 10,000 and profit in 10,000 $

% column one of data is population of city, this is the feature for our
% learning algorithm. i.e - x

% column two of data is the target variable, i.e - y

data = load('ex1data1.txt');

% population 
X = data(:, 1); 

%profit
y = data(:, 2);

%number of tarining examples 
m = length(y);

% Some plotting
plot1 = subplot(2,2,1) 
plot(X,'x')
title('Population in 10000')
ylabel(plot1, 'population')

plot2 = subplot(2,2,2)
plot(y,'rx')
title('Profit in 10000 $')
ylabel(plot2, 'profit')

subplot(2,2,3)
plot(data)
title('data')

plot4 = subplot(2,2,4)
scatter(X,y,'gx')
title('data')

figure; % open a new figure window
plot(X, y, 'rx', 'MarkerSize', 10);
ylabel('Profit in $10,000s');
xlabel('Population of City in 10,000s');


% initialize theta1 and theta2 to 0
theta = zeros(2, 1)

% more thetas for testing
theta1 = [0;1];
theta3 = [-1;2];


%more test plot calculated data from theta for test valuse 

hold on; % keep previous plot visible
plot(X, [ones(m, 1), X]*theta, '*')

hold on; 
plot(X, [ones(m, 1), X]*theta1, 'g-')

hold on; 
plot(X, [ones(m, 1), X]*theta3, 'b-')

%need to convert data matrix to accomodate parameter for constant 
Xnew = [ones(m, 1), X];


% now let us create a compute cost function, once we have created this
% function we will pass differet values for theta1 and theta2 to this
% function to do some testing

%function defined at the end of file


% lets generate test costs 

testCost1 = computeCost(Xnew, y, theta)
testCost2 = computeCost(Xnew, y, theta1)
testCost3 = computeCost(Xnew, y, theta3)


% now we will create machine learning algorithm
% in this example we will use BATCH GRADIENT DESCENT ALGORITHM

% Gradient Descent Stttings
iterations = 1500;
alpha = 0.01;

% gradient descent defines at the end of file


[theta, cost_history,theta_history] = gradientDescent(Xnew, y, theta, alpha, iterations);
theta
% cost_history
% theta_history



% plot graph according to parameters received by gradient descent algorithm
hold on; 
plot(X, [ones(m, 1), X]*theta, 'y-')
legend('Training data', 'Test 1','Test 2','Test 3','Linear Regaression')
hold off % don't overlay any more plots on this figure

figure
plot(cost_history)



%lets plot cost over a range of theta1 and theta2

range_theta1 = linspace(-10, 10, 100);
range_theta2 = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(range_theta1), length(range_theta2));

for i = 1:length(range_theta1)
    for j = 1:length(range_theta2)
	  t = [range_theta1(i); range_theta2(j)];
	  J_vals(i,j) = computeCost(Xnew, y, t);
    end
end


% for i = 1:length(theta_history(:,1))
%     for j = 1:length(theta_history(:,2))
% 	  t = [theta_history(i,1); theta_history(j,2)];
% 	  J_vals(i,j) = computeCost(Xnew, y, t);
%     end
% end


% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';


% Surface plot
figure;
surf(range_theta1, range_theta2, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');


% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(range_theta1, range_theta2, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);


% funstion defined at the end of file
function J = cost(Xnew,y,theta)
m = length(y);
errorSquare = power((Xnew * theta - y),2);
J = (1 / (2*m)) * sum(errorSquare);
end

% batch gradient descent
function [theta, cost_history, theta_history] = gradientDescent(Xnew, y, theta, alpha, iterations)

%initialize m for training data set

%initialize a vector to save value of cost on each iteration

%the cost should go down with each iteration

%we can plot this cost over number of iterations to get a intution about
%the success of training algorithm

m = length(y); % number of training examples
cost_history = zeros(iterations, 1);
theta_history = zeros(iterations,2);

for iter = 1:iterations
    %hypothesis
    htheta = Xnew * theta;
    
    % update both Theta i.e. parameters for learning algorithm
    % simultaneoously
    % .* Xnew(:,1) - this is not necessary for first parameter, but it is
    % good to use it, becasue this sets a pattern which is useful in
    % scaling

    theta1 = theta(1) - alpha / m * sum((htheta - y) .* Xnew(:,1));
    theta2 = theta(2) - alpha / m * sum((htheta - y) .* Xnew(:,2));
    
    %update theta
    theta = [theta1; theta2];
    
    %update cost history vector
    cost_history(iter) = computeCost(Xnew, y, theta);
    
    theta_history(iter,:) = [theta1, theta2];
    
end
end



