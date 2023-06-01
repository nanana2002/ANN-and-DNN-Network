function [Yn On Yt Ot] = ann2159347(ID, N_ep, lr, bp, u, v, w, cf)
% Where, for the MNIST data sets,
% - ID is your student ID (e.g. 2216789)
% - N_ep the number of epochs, lr is the learning rate
% - bp = 1, heuristic unscaled, 0 calculus based bp
% - u, v, w the number of nodes on the three hidden layers
% - cf = 1 use total squared error; 2 use cross-entropy
% - Yn are the exact training labels/targets, Ot the training predictions
% - Yt are the exact testing  labels/targets, On the testing  predictions
% The MNIST CSV files are not to be altered in any way, and should be
% in the same folder as this matlab code.

% hidden layer=1 
% ann2159347(2216789, 100, 0.008, bp, 0, 0, 0, cf);
% ann2159347(2216789, 100, 0.008, 1, 0, 0, 0, 1);
% ann2159347(2216789, 100, 0.008, 1, 0, 0, 0, 2);
% ann2159347(2216789, 100, 0.008, 0, 0, 0, 0, 1);
% ann2159347(2216789, 100, 0.008, 0, 0, 0, 0, 2);
% hidden layer=3
% ann2159347(2216789, 100, 0.008, bp, 26, 28, 27, cf);
% ann2159347(2216789, 100, 0.008, 1, 26, 28, 27, 1);
% ann2159347(2216789, 100, 0.008, 1, 26, 28, 27, 2);
% ann2159347(2216789, 100, 0.008, 0, 26, 28, 27, 1);
% ann2159347(2216789, 100, 0.008, 0, 26, 28, 27, 2);


close all

%% set some useful defaults
set(0,'DefaultLineLineWidth', 2);
set(0,'DefaultLineMarkerSize', 10);

% As a professional touch we should test the validity of our input
if or(N_ep <= 0, lr <= 0)
  error('N_ep and/or lr are not valid')
end
if ~ismember(bp,[1,0])
  error('back prop choice is not valid')
end
if ~ismember(cf,[1,2])
  error('performance index choice is not valid')
end

%% obtain the raw training data
A = readmatrix('mnist_train_1000.csv');

% convert and NORMALIZE it into training inputs and target outputs
X_train = A(:,2:end)'/255;  % beware - transpose, data is in columns!
N_train = size(X_train,2); % size(X_train,1/2) gives number of rows/columns
Y_train = zeros(10,N_train);
% set up the one-hot encoding - note that we have to increment by 1
for i=1:N_train           
  Y_train(1+A(i,1),i) = 1;
end                     
%% if hidden layer=3 then
if u~=0&&v~=0&&w~=0
    % default variables
    Ni   = 784;             % number of input nodes
    N1   = u;               % number of nodes on first hidden layer
    N2   = v;               % number of nodes on second hidden layer
    N3   = w;               % number of nodes on third hidden layer
    No   = 10;              % number of output nodes
    % set up weights and biases
    W2 = 0.5-rand(Ni,N1); b2 = zeros(N1,1);
    W3 = 0.5-rand(N1, N2); b3 = zeros(N2,1);
    W4 = 0.5-rand(N2, N3); b4 = zeros(N3,1);
    W5 = 0.5-rand(N3, No); b5 = zeros(No,1);
    % set up a sigmoid activation function for layers 2 and 3
    sig2  = @(x) 1./(1+exp(-x));
    dsig2 = @(x) exp(-x)./((1+exp(-x)).^2);
    sig3  = @(x) 1./(1+exp(-x));
    dsig3 = @(x) exp(-x)./((1+exp(-x)).^2);
    sig4  = @(x) 1./(1+exp(-x));
    dsig4 = @(x) exp(-x)./((1+exp(-x)).^2);
    % output layer activation depends on cf
    if cf == 1
      sig5  = @(x) 1./(1+exp(-x));
      dsig5 = @(x) exp(-x)./((1+exp(-x)).^2);
    elseif cf == 2
      sig5  = @(x) exp(x)/sum(exp(x));
    else
      error('cf has improper value')
    end
    
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
        n4 = W4'*a3 + b4; a4 = sig4(n4);
        n5 = W5'*a4 + b5; a5 = sig5(n5);
        % this is then the output
        y = a5;
    
        % calculate A, the diagonal matrices of activation derivatives
        A4 = diag(dsig4(n4)); A3 = diag(dsig3(n3)); A2 = diag(dsig2(n2));
        % we calculate the error in this output, and get the S3 vector
        e5 = Y_train(:,i) - y;
        if cf == 1
          A5 = diag(dsig5(n5)); S5 = -2*A5*e5;
        elseif cf == 2
          S5 = -e5;
        end  
        % back prop the error   
        if     bp == 1
          e4 = W5*e5; e3 = W4*e4; e2 = W3*e3;
          % from these we compute the S vectors
          S4 = -2*A4*e4; S3 = -2*A3*e3; S2 = -2*A2*e2;
        elseif bp == 0
          S4 = A4*W5*S5; S3 = A3*W4*S4; S2 = A2*W3*S3;
        end 
        
        % and use a learning rate to update weights and biases
        W5 = W5 - lr * a4*S5'; b5 = b5 - lr * S5;
        W4 = W4 - lr * a3*S4'; b4 = b4 - lr * S4;
        W3 = W3 - lr * a2*S3'; b3 = b3 - lr * S3;
        W2 = W2 - lr * a1*S2'; b2 = b2 - lr * S2;
      end
    end
    
 % loop through the training set and evaluate accuracy of prediction
    wins = 0;
    y_pred = zeros(No,N_train);
    last_train=zeros(2,N_train);
    for i = 1:N_train
      y = sig3(W3'*sig2(W2'*X_train(:,i) + b2) + b3);
      y_pred(:,i) = sig5(W5'*sig4(W4'*y + b4) + b5);
      [~, indx1] = max(y_pred(:,i));
      [~, indx2] = max(Y_train(:,i));
      last_train(1,i)=indx1-1;
      last_train(2,i)=indx2-1;
      if indx1 == indx2; wins = wins+1;  end 
    end
%% if hidden layer=1 then
else
    % default variables
    Ni   = 784;             % number of input nodes
    Nh   = 10;              % number of hidden nodes
    No   = 10;              % number of output nodes
    % set up weights and biases
    W2 = 0.5-rand(Ni,Nh); b2 = zeros(Nh,1);
    W3 = 0.5-rand(Nh, No); b3 = zeros(No,1);
    
    % set up a sigmoid activation function for layers 2 and 3
    sig2  = @(x) 1./(1+exp(-x));
    dsig2 = @(x) exp(-x)./((1+exp(-x)).^2);
    if cf == 1
      sig3  = @(x) 1./(1+exp(-x));
      dsig3 = @(x) exp(-x)./((1+exp(-x)).^2);
    elseif cf == 2
      sig3  = @(x) exp(x)/sum(exp(x));
    else
      error('cf has improper value')
    end
    
    
    % we now train by looping N_ep times through the training set
    for epoch = 1:N_ep
      % Make the training order random randperm (n) return a row vector，
      % which contains a random arrangement of integers from 1 to n without repeating elements.
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
        if cf == 1
          A3 = diag(dsig3(n3)); S3 = -2*A3*e3;
        elseif cf == 2
          S3 = -e3;
        end  
        % back prop the error
        if     bp == 1; e2 = W3*e3; S2 = -2*A2*e2;
        elseif bp == 0; S2 = A2*W3*S3;
        end 
        
        % and use a learning rate to update weights and biases
        W3 = W3 - lr * a2*S3'; b3 = b3 - lr * S3;
        W2 = W2 - lr * a1*S2'; b2 = b2 - lr * S2;
      end
    end
    
    % loop through the training set and evaluate accuracy of prediction
    wins = 0;
    y_pred = zeros(10,N_train);%y_pred Each column stores the likelihood of each number
    last_train=zeros(2,N_train);
    for i = 1:N_train
      y_pred(:,i) = sig3(W3'*sig2(W2'*X_train(:,i) + b2) + b3);
      [~, indx1] = max(y_pred(:,i));
      [~, indx2] = max(Y_train(:,i));
      
      last_train(1,i)=indx1-1;%last_train The first line stores the predicted numbers
      last_train(2,i)=indx2-1;%last_train The second line stores real numbers
      
      if indx1 == indx2
          wins = wins+1; 
      end 
    end
end

fprintf('recognize number training set wins = %d/%d, %f%%\n',wins,N_train,100*wins/N_train)

%% obtain the raw test data
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

%% loop through the test set and evaluate accuracy of prediction
wins = 0;
y_pred = zeros(No,N_test);
last_test=zeros(2,N_test);
for i = 1:N_test
  if u~=0&&v~=0&&w~=0
      y = sig3(W3'*sig2(W2'*X_test(:,i) + b2) + b3);
      y_pred(:,i) = sig5(W5'*sig4(W4'*y + b4) + b5);
  else
      y_pred(:,i) = sig3(W3'*sig2(W2'*X_test(:,i) + b2) + b3);
  end
  [~, indx1] = max(y_pred(:,i));
  [~, indx2] = max(Y_test(:,i));
  last_test(1,i)=indx1-1;
  last_test(2,i)=indx2-1;
  if indx1 == indx2 
      wins = wins+1;
  end
end
fprintf('recognize number testing  set wins = %d/%d, %f%%\n',wins,N_test,100*wins/N_test)

%% classify the MNIST digits as belonging to my student ID
    A=last_train;
    % convert and NORMALIZE it into training inputs and target outputs
    X_train = A(1,:)/10;  % beware - transpose, data is in columns!
    N_train = size(X_train,2); % size(X_train,1/2) gives number of rows/columns
    Y_train = zeros(2,N_train);
    real_train=zeros(2,N_train);
    % set up the one-hot encoding - note that we have to increment by 1
    for i=1:N_train   %ID=2159347
        if contains(int2str(ID),int2str(A(1,i)))
            Y_train(1,i) = 1;
        else
            Y_train(2,i) = 1;
        end  
    end        
    for i=1:N_train   %ID=2159347
        if contains(int2str(ID),int2str(A(2,i)))
            real_train(1,i) = 1;
        else
            real_train(2,i) = 1;
        end  
    end  
    
% default variables
%% if hidden layer=3 then
if u~=0&&v~=0&&w~=0
    Ni   = 1;             % number of input nodes
    N1   = u;               % number of nodes on first hidden layer
    N2   = v;               % number of nodes on second hidden layer
    N3   = w;               % number of nodes on third hidden layer
    No   = 2;               % number of output nodes
    % set up weights and biases
    W2 = 0.15*rand(Ni,N1); b2 = 0.15*rand(Ni,1);
    W3 = 0.15*rand(N1, N2); b3 = 0.15*rand(N2,1);
    W4 = 0.15*rand(N2, N3); b4 =0.15*rand(N3,1);
    W5 = 0.15*rand(N3, No); b5 = 0.15*rand(No,1);
    % set up a sigmoid activation function for layers 2 and 3
    sig2  = @(x) 1./(1+exp(-x));
    dsig2 = @(x) exp(-x)./((1+exp(-x)).^2);
    sig3  = @(x) 1./(1+exp(-x));
    dsig3 = @(x) exp(-x)./((1+exp(-x)).^2);
    sig4  = @(x) 1./(1+exp(-x));
    dsig4 = @(x) exp(-x)./((1+exp(-x)).^2);
    % output layer activation depends on cf
    if cf == 1
      sig5  = @(x) 1./(1+exp(-x));
      dsig5 = @(x) exp(-x)./((1+exp(-x)).^2);
    elseif cf == 2
      sig5  = @(x) exp(x)/sum(exp(x));
    else
      error('cf has improper value')
    end

    % we now train by looping N_ep times through the training set
    lr=0.09;
    for epoch = 1:N_ep
      mixup = randperm(N_train);
      for j = 1:N_train
        i = mixup(j);
        % get X_train(:,i) as an input to the network
        a1 = X_train(:,i);
        % forward prop to the next layer, activate it, repeat
        n2 = W2'*a1 + b2; a2 = sig2(n2);
        n3 = W3'*a2 + b3; a3 = sig3(n3);
        n4 = W4'*a3 + b4; a4 = sig4(n4);
        n5 = W5'*a4 + b5; a5 = sig5(n5);
        % this is then the output
        y = a5;
    
        % calculate A, the diagonal matrices of activation derivatives
        A4 = diag(dsig4(n4)); A3 = diag(dsig3(n3)); A2 = diag(dsig2(n2));
        % we calculate the error in this output, and get the S3 vector
        e5 = Y_train(:,i) - y;
        if cf == 1
          A5 = diag(dsig5(n5)); S5 = -2*A5*e5;
        elseif cf == 2
          S5 = -e5;
        end  
        % back prop the error   
        if     bp == 1
          e4 = W5*e5; e3 = W4*e4; e2 = W3*e3;
          % from these we compute the S vectors
          S4 = -2*A4*e4; S3 = -2*A3*e3; S2 = -2*A2*e2;
        elseif bp == 0 
          S4 = A4*W5*S5; S3 = A3*W4*S4; S2 = A2*W3*S3;
        end 
        
        % and use a learning rate to update weights and biases
        W5 = W5 - lr * a4*S5'; b5 = b5 - lr * S5;
        W4 = W4 - lr * a3*S4'; b4 = b4 - lr * S4;
        W3 = W3 - lr * a2*S3'; b3 = b3 - lr * S3;
        W2 = W2 - lr * a1*S2'; b2 = b2 - lr * S2;
      end
    end
    
    % loop through the training set and evaluate accuracy of prediction
    y_pred = zeros(2,N_train);
    wins=0;
    for i = 1:N_train
      y = sig3(W3'*sig2(W2'*X_train(:,i) + b2) + b3);
      y_pred(:,i) = sig5(W5'*sig4(W4'*y + b4) + b5);
      [~, indx1] = max(y_pred(:,i));
      [~, indx2] = max(real_train(:,i));
      
      if indx1 == indx2
          wins = wins+1;
      end 
    end
else
%% if hidden layer=1 then
    % default variables
    Ni   = 1;             % number of input nodes
    No   = 2;               % number of output nodes
    % set up weights and biases
    W2 = 0.5-rand(Ni,Nh); b2 = zeros(Nh,1);
    W3 = 0.5-rand(Nh, No); b3 = zeros(No,1);
    % set up a sigmoid activation function for layers 2 and 3
    sig2  = @(x) 1./(1+exp(-x));
    dsig2 = @(x) exp(-x)./((1+exp(-x)).^2);
    if cf == 1
      sig3  = @(x) 1./(1+exp(-x));
      dsig3 = @(x) exp(-x)./((1+exp(-x)).^2);
    elseif cf == 2
      sig3  = @(x) exp(x)/sum(exp(x));
    else
      error('cf has improper value')
    end

    lr=0.09;
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
        if cf == 1
          A3 = diag(dsig3(n3)); S3 = -2*A3*e3;
        elseif cf == 2
          S3 = -e3;
        end  
    % back prop the error   
        if     bp == 1; e2 = W3*e3; S2 = -2*A2*e2;
        elseif bp == 0; S2 = A2*W3*S3;
        end 
        
        % and use a learning rate to update weights and biases
        W3 = W3 - lr * a2*S3'; b3 = b3 - lr * S3;
        W2 = W2 - lr * a1*S2'; b2 = b2 - lr * S2;
      end
    end
    
    % loop through the training set and evaluate accuracy of prediction
    wins = 0;
    y_pred = zeros(2,N_train);
    for i = 1:N_train
      y_pred(:,i) = sig3(W3'*sig2(W2'*X_train(:,i) + b2) + b3);
      [~, indx1] = max(y_pred(:,i));
      [~, indx2] = max(real_train(:,i));
      if indx1 == indx2
          wins = wins+1;
      end 
    end
end
fprintf('classify training set wins = %d/%d, %f%%\n',wins,N_train,100*wins/N_train)

%% obtain the raw test data
A = last_test;
% convert and NORMALIZE it into testing inputs and target outputs
X_test = A(1,:)/10;
% the number of data points
N_test = size(X_test,2);
Y_test = zeros(2,N_test);
real_test=zeros(2,N_test);
% set up the one-hot encoding - recall we have to increment by 1
for i=1:N_test   %ID=2159347
    if contains(int2str(ID),int2str(A(1,i)))
        Y_test(1,i) = 1;
    else
        Y_test(2,i) = 1;
    end  
end        
for i=1:N_test   %ID=2159347
    if contains(int2str(ID),int2str(A(2,i)))
        real_test(1,i) = 1;
    else
        real_test(2,i) = 1;
    end  
end  

%% loop through the test set and evaluate accuracy of prediction
wins = 0;
y_pred = zeros(2,N_test);
for i = 1:N_test
  if u~=0&&v~=0&&w~=0
      y = sig3(W3'*sig2(W2'*X_test(:,i) + b2) + b3);
      y_pred(:,i) = sig5(W5'*sig4(W4'*y + b4) + b5);
  else
      y_pred(:,i) = sig3(W3'*sig2(W2'*X_test(:,i) + b2) + b3);
  end
  [~, indx1] = max(y_pred(:,i));
  [~, indx2] = max(real_test(:,i));
  if indx1 == indx2
     wins = wins+1; 
  end 
end

fprintf('classify testing  set wins = %f/%f, %f%%\n',wins,N_test,100*wins/N_test)


%% draw confusion matrix 
figure
plotconfusion(real_test,y_pred);
xticklabels({'POSTIVE','NEGATIVE'});
yticklabels({'POSTIVE','NEGATIVE'});
xlabel('Target Class','FontWeight','bold');
ylabel('Output Class','FontWeight','bold');
title(['Confusion Matrix (hidden layer=3 bp=',num2str(bp),' cf=',num2str(cf),')']);


end
