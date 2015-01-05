%% load sentiment data
load ./data/ptrain.data.mat
addpath ./util; addpath ./NN;
train_idx = 1:90000;
test_idx = 90001:156060;
train_x = Xtrain(train_idx,:);
test_x = Xtrain(test_idx,:);

n_tot = size(ylab,2);
ytrain = repmat(ylab',1,5)==repmat(0:4,n_tot,1);

train_y = ytrain(train_idx,:);
test_y = ytrain(test_idx,:);
addpath ../util; addpath ../NN

%% vanilla neural net
rand('state',0)
nn = nnsetup([size(train_x,2) 50 size(train_y,2)]);
%nn.weightPenaltyL2 = 1e-8;  %  L2 weight decay
opts.learningrate = .1;
opts.numepochs =  1;   %  Number of full sweeps through data
opts.batchsize = 10;  %  Take a mean gradient step over this many samples
[nn, L] = nntrain(nn, train_x, train_y, opts);

[test_err, bad] = nntest(nn, test_x, test_y);
test_pred = nnpredict(nn, test_x);
display(test_err)


%% cv
k = 5;
Xtrain = Xtrain(1:156000,:);
arch = [size(Xtrain,2) 50 size(ytrain,2)];
nobs = size(Xtrain,1);
for i=1:k+1
   embedding(i)= floor((i-1)*nobs/k);
end
embedding(1) = 1;


%% cv
cvtest_err = [];
for i=1:k
    
    testset = embedding(i):embedding(i+1);
    cvtrain = Xtrain;
    cvtrain(testset,:)=[];
    cvytrain = ytrain;
    cvytrain(testset,:)=[];
    cvtest = Xtrain(testset,:);
    cvytest = ytrain(testset,:);
    nn = nnsetup(arch);
    opts.numepochs=3;
    opts.batchsize=10;
    [nn L] = nntrain(nn, cvtrain, cvytrain, opts);
    [test_err bad] = nntest(nn, cvtest, cvytest);
    cv_test_err = [cv_test_err test_err];
end 


%% predict
rand('state',0)
select_idx = 1:156000; % makes numberofbatches integer
x_train = Xtrain(select_idx,:);
y_train = ytrain(select_idx,:);

fnn = nnsetup([size(x_train,2) 50 size(y_train,2)]);
opts.numepochs =  5;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples
[fnn, L] = nntrain(fnn, x_train, y_train, opts);

ypred=nnpredict(fnn, Xtest);



%% write to file
fileID = fopen('data/ypred.txt','w');
fprintf(fileID,'%d\n',ypred-1);
display('written to file data/ypred.txt')



