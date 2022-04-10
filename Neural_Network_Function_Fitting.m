%% Function FItting with NN
clc;
close all;
clear ;
%% Import the Data
data = cell2mat(struct2cell(load('data_NN.mat')));
x = data(1:3,1:1000);%Take the input from the data_NN.mat
y = data(1:3,2:1001);%Take the output from the data_NN.mat
%% Set the bounds for the normalized data
top = 0.9;
bottom = 0.1;
%% Normalize the data
data_min = min(data(:));
data_max = max(data(:));
norm_data = (((top-bottom)*(data-data_min))/(data_max-data_min))+bottom;
%% Normalize the input data
normx = norm_data(:,1:1000);
%% Normalize the output data
normy = norm_data(:,2:1001);
%% Create the NN
hidden_neurons = [5,5];%Set the hidden neurons
net = fitnet(hidden_neurons);%Create the NN

net.divideParam.trainRatio = 0.8;%Training Ratio
net.divideParam.valRatio = 0.1;%Validation Ratio
net.divideParam.testRatio = 0.1;%Test Ratio

net.trainParam.epochs = 1000;%Number of epochs

%Choose the training function
net.trainFcn = 'trainlm'; %Levenberg-Marquardt
%net.trainFcn = 'trainbr'; %Bayesian Regularization
%% Train the NN
[net,tr] = train(net,normx,normy);
%% Test the NN for the given data
test = net(normx);
%% Test for 200 steps with initial input x0 = [-1.9 0 -0.9]
output = [];%Initialize the output matrix
x0 = [-1.9 ; 0 ; -0.9];%Initial Value
target = normy(:,1:200);%Set our target
%% Normalize x0
normx0 = (((top-bottom)*(x0-data_min))/(data_max-data_min))+bottom;
%% Test the NN
normxk = net(normx0);%Calculate the next value
output = [output normxk];%Add xk to the output matrix
%Calculate the rest values
for i=1:199
    x_new =  net(output(:,end));
    output =  [output x_new];
end
%% Denormalize the data
denorm_bottom = min(data(:));
denorm_top = max(data(:));

denorm_200_bottom = min(min(data(:,1:200)));
denorm_200_top = max(max(data(:,1:200)));

denorm_max_test = max(test(:));
denorm_min_test = min(test(:));

denorm_max_out = max(output(:));
denorm_min_out = min(output(:));

denorm = (((denorm_top-denorm_bottom)*(test-denorm_min_test))/(denorm_max_test-denorm_min_test))+denorm_bottom;
denorm_out = (((denorm_200_top-denorm_200_bottom)*(output-denorm_min_out))/(denorm_max_out-denorm_min_out))+denorm_200_bottom;
%% Calculate the error and the performance of the NN
per_error = 100*(abs(output - target))./abs(output);%Percentage error
avg_per_error = mean(per_error);%Average percentage error
NMSE = mse(output-target)/mean(var(target',1));%Normalized Mean Square Error

perform(net,y,denorm)
perform(net,y(:,1:200),denorm_out)
%% Export the output
save('data_VAL.mat',"denorm_out");
%% Plot the normalized data
figure;
plot3(norm_data(1,1:end),norm_data(2,1:end),norm_data(3,1:end));%Plot the original normalized function
hold on;
plot3(test(1,1:end),test(2,1:end),test(3,1:end))%Plot the normalized output of the NN for the training data
hold on;
plot3(output(1,1:end),output(2,1:end),output(3,1:end))%Plot the normalized output of NN for the 200 first steps for x0=[-1.9 0 -0.9]
legend('F(*)','Test','x0 = [-1.9 0 -0.9]')
%% Plot the denormalized data
figure;
plot3(data(1,1:end),data(2,1:end),data(3,1:end))%Plot the original function
hold on;
plot3(denorm(1,1:end),denorm(2,1:end),denorm(3,1:end))%Plot the output of the NN for the training data
hold on;
plot3(denorm_out(1,1:end),denorm_out(2,1:end),denorm_out(3,1:end))%Plot the output of the NN for 200 first steps for x0=[-1.9 0 -0.9]
legend('F(*)','Test','x0 = [-1.9 0 -0.9]')