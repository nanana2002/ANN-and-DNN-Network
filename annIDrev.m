function [Yn On Yt Ot] = ann2159347rev(N_ep, lr, bp, gfx, Nh, cf)
% MA2647: Artificial Neural Network, demo 9 (MNIST data).
%   ann2159347rev(100, 0.008, 1, 0, 10,1)
% Usage:
%    [Yn On Yt Ot] = ann09demo(N_ep, lr, data, bp, gfx, Nh)
%    id = [1,2,3,4,5,6,7];
% Where, for the MNIST data sets,
% - N_ep the number of epochs, lr is the learning rate
% - bp = 1, heuristic unscaled, 2 heuristic scaled, 3 calculus based bp
% - gfx ~= 0 (show graphics on screen), 0 (don't show graphics)
% - Nh is the number of nodes on the hidden layer
% - cf = 1 use total squared error; 2 use cross-entropy
% - Yn are the exact training labels/targets, Ot the training predictions
% - Yt are the exact testing  labels/targets, On the testing  predictions
% The MNIST CSV files are not to be altered in any way, and should be
% in the same folder as this matlab code.
%
% This code is based on the book Make Your Own Neural Network by Tariq
% Rashid (CreateSpace Independent Publishing Platform (31 Mar. 2016),
% ISBN-10: 1530826608; ISBN-13: 978-1530826605)

% set some useful defaults
set(0,'DefaultLineLineWidth', 2);
set(0,'DefaultLineMarkerSize', 10);

% As a professional touch we should test the validity of our input
if or(N_ep <= 0, lr <= 0)
  error('N_ep and/or lr are not valid')
end
if ~ismember(bp,[1,2,3])
  error('back prop choice is not valid')
end
if gfx ~= 0
  % clear previous graphics windows and dock figures
  clf; close all; set(0,'DefaultFigureWindowStyle','docked')
end
if Nh <= 0
  error('Nh choice is not valid')
end
if ~ismember(cf,[1,2])
  error('performance index choice is not valid')
end

% obtain the raw training data
  A = readmatrix('mnist_train_1000.csv');

% convert and NORMALIZE it into training inputs and target outputs
X_train = A(:,2:end)'/255;  % beware - transpose, data is in columns!
N_train = size(X_train,2); % size(X_train,1/2) gives number of rows/columns
Y_train = zeros(10,N_train);
% set up the one-hot encoding - note that we have to increment by 1
for i=1:N_train
  Y_train(1+A(i,1),i) = 1;
end

% default variables
Ni   = 784;             % number of input nodes
No   = 10;              % number of output nodes
% set up weights and biases
W2 = 0.5-rand(784,Nh); b2 = zeros(Nh,1);
W3 = 0.5-rand(Nh, 10); b3 = zeros(10,1);
% set up a sigmoid activation function for layers 2 and 3
sig2  = @(x) 1./(1+exp(-x));
dsig2 = @(x) exp(-x)./((1+exp(-x)).^2);
if cf == 1;
  sig3  = @(x) 1./(1+exp(-x));
  dsig3 = @(x) exp(-x)./((1+exp(-x)).^2);
elseif cf == 2;
  sig3  = @(x) exp(x)/sum(exp(x));
else
  error('cf has improper value')
end

% we'll calculate the performance index at the end of each epoch
pivec = zeros(1,N_ep);  % row vector

% we now train by looping N_ep times through the training set
for epoch = 1:N_ep
  mixup = randperm(N_train);
  for j = 1:N_train
    i = mixup(j);
    % get X_train(:,i) as an input to the network
    a1 = X_train(:,i);
    % forward prop to the next layer, activate it, repeat
    n2 = W2'*a1 + b2; a2 = sig2(n2);
    n3 = W3'*a2 + b3; a3 = sig3(n3);
    % this is then the output
    y = a3;

    % calculate A, the diagonal matrices of activation derivatives
    A2 = diag(dsig2(n2));
    % we calculate the error in this output, and get the S3 vector
    e3 = Y_train(:,i) - y;
    if cf == 1;
      A3 = diag(dsig3(n3)); S3 = -2*A3*e3;
    elseif cf == 2;
      S3 = -e3;
    end  
    % back prop the error
    if     bp == 1; e2 = W3*e3; S2 = -2*A2*e2;
    elseif bp == 2; e2 = W3*diag(1./sum(W3))*e3; S2 = -2*A2*e2;
    elseif bp == 3; S2 = A2*W3*S3;
    end 
    
    % and use a learning rate to update weights and biases
    W3 = W3 - lr * a2*S3'; b3 = b3 - lr * S3;
    W2 = W2 - lr * a1*S2'; b2 = b2 - lr * S2;
  end
  % calculate the sum of squared errors and store for plotting
  for i=1:N_train
    y = sig3(W3'*sig2(W2'*X_train(:,i) + b2) + b3);
    if cf == 1;
      err = Y_train(:,i) - y;
      % each error is itself a vector - hence the norm ||err||
      pivec(epoch) = pivec(epoch) + norm(err,2)^2;
    elseif cf == 2
      xent = -sum(Y_train(:,i).*log(y));
      pivec(epoch) = pivec(epoch) + xent;
    end   
  end
end

% add a subplot; plot the performance index vs (row vector) epochs
if gfx ~= 0
  subplot(2,2,1)
  plot([1:N_ep],pivec,'b');
  xlabel('epochs'); ylabel('performance index');
end

% loop through the training set and evaluate accuracy of prediction
wins = 0;
y_pred = zeros(10,N_train);
for i = 1:N_train
  y_pred(:,i) = sig3(W3'*sig2(W2'*X_train(:,i) + b2) + b3);
  [~, indx1] = max(y_pred(:,i));
  [~, indx2] = max(Y_train(:,i));
  barcol = 'r';
  if indx1 == indx2; wins = wins+1; barcol = 'b'; end 
  if gfx ~= 0
    % plot the output 10-vector at top right
    subplot(2,2,2); bar(0:9,y_pred(:,i),barcol);
    title('predicted output (approximate one-hot)')
    % plot the MNIST image at bottom left
    B = reshape(1-X_train(:,i),[28,28]);
    subplot(2,2,3); imh = imshow(B','InitialMagnification','fit');
    subplot(2,2,4);
    b = bar(categorical({'Wins','Losses'}), [wins i-wins]);
    ylim([0,N_train]);
    b.FaceColor = 'flat'; b.CData(1,:) = [1 0 0]; b.CData(2,:) = [0 0 1];
    a = get(gca,'XTickLabel'); set(gca,'XTickLabel',a,'fontsize',18)    
    drawnow
  end
end
fprintf('training set wins = %d/%d, %f%%\n',wins,N_train,100*wins/N_train)

% assign outputs
Yn=Y_train; On=y_pred;

% obtain the raw test data
  A = readmatrix('mnist_test_100.csv');

% convert and NORMALIZE it into testing inputs and target outputs
X_test = A(:,2:end)'/255;
% the number of data points
N_test = size(X_test,2);
Y_test = zeros(10,N_test);
% set up the one-hot encoding - recall we have to increment by 1
for i=1:N_test
  Y_test(1+A(i,1),i) = 1;
end

% loop through the test set and evaluate accuracy of prediction
wins = 0;
y_pred = zeros(10,N_test);
for i = 1:N_test
  y_pred(:,i) = sig3(W3'*sig2(W2'*X_test(:,i) + b2) + b3);
  [~, indx1] = max(y_pred(:,i));
  [~, indx2] = max(Y_test(:,i));
  barcol = 'r';
  if indx1 == indx2; wins = wins+1; barcol = 'b'; end 
  if gfx ~= 0
    % plot the output 10-vector at top right
    subplot(2,2,2); bar(0:9,y_pred(:,i),barcol);
    title('predicted output (approximate one-hot)')
    % plot the MNIST image at bottom left
    B = reshape(1-X_test(:,i),[28,28]);
    subplot(2,2,3); imh = imshow(B','InitialMagnification','fit');
    % animate the wins and losses bottom right
    subplot(2,2,4);
    b = bar(categorical({'Wins','Losses'}), [wins i-wins]);
    ylim([0,N_test]);
    b.FaceColor = 'flat'; b.CData(1,:) = [1 0 0]; b.CData(2,:) = [0 0 1];
    a = get(gca,'XTickLabel'); set(gca,'XTickLabel',a,'fontsize',18)
    drawnow;
    pause(0.01)
  end
end
fprintf('testing  set wins = %d/%d, %f%%\n',wins,N_test,100*wins/N_test)

% assign outputs
Yt=Y_test; Ot=y_pred;

%% rev
figure
% run it backwards... But only for TSE (final layer sigmoid)
xmin=0.0001; xmax=1-xmin;

for c = 1:7
  y = xmin*ones(10,1);
  y(id(c)+1) = xmax;
  x = W3*( log( y ./ (1-y) ) - b3);
  % scale back to [xmin,xmax]
  x=x-min(x);
  x=xmin+(xmax-xmin)*x/max(x);
  x = W2*( log( x ./ (1-x) ) - b2);
  % scale back to [xmin,xmax]
  x=x-min(x);
  x=xmin+(xmax-xmin)*x/max(x); 
  x = xmin*(x<0.5)+ xmax*(x>=0.5);
  B = reshape(x,[28,28]);
  subplot(1,7,c);
  imh = imshow(B','InitialMagnification','fit');

end
end
