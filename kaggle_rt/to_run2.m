%% load sentiment data
load ./data/ptrain.data.mat
addpath ../util; addpath ../NN;
train_idx = 1:90000;
test_idx = 90001:156060;
train_x = Xtrain(train_idx,:);
test_x = Xtrain(test_idx,:);

n_tot = size(ylab,2);
ytrain = repmat(ylab',1,5)==repmat(0:4,n_tot,1);
%%
train_y = ytrain(train_idx,:);
test_y = ytrain(test_idx,:);
addpath ../util; addpath ../NN

nn = nnsetup([size(train_x,2) 50 size(train_y,2)]);
minl = 1e-4; maxl = 1e-6;
nn.weightPenaltyL2 = minl+ rand()*(maxl-minl);  %  L2 weight decay
opts.learningrate = .1;
opts.numepochs =  10;   %  Number of full sweeps through data
opts.batchsize = 10;  %  Take a mean gradient step over this many samples
[nn, L] = nntrain(nn, train_x, train_y, opts);

[test_err, bad] = nntest(nn, test_x, test_y);
test_pred = nnpredict(nn, test_x);
test_acc = 1 - sum(test_pred'==ylab(test_idx))/size(test_idx,2);
% display(test_acc)



%% dbn
%% load sentiment data
load ./data/ptrain.data.mat
addpath ../util; addpath ../NN;
train_idx = 1:90000;
test_idx = 90001:156060;
train_x = Xtrain(train_idx,:);
test_x = Xtrain(test_idx,:);

n_tot = size(ylab,2);
ytrain = repmat(ylab',1,5)==repmat(0:4,n_tot,1);

train_y = ytrain(train_idx,:);
test_y = ytrain(test_idx,:);
addpath ../util; addpath ../NN

nn = nnsetup([size(train_x,2) 50 size(train_y,2)]);
minl = 1e-4; maxl = 1e-6;
nn.weightPenaltyL2 = minl+ rand()*(maxl-minl);  %  L2 weight decay
opts.learningrate = .1;
opts.numepochs =  10;   %  Number of full sweeps through data
opts.batchsize = 10;  %  Take a mean gradient step over this many samples
[nn, L] = nntrain(nn, train_x, train_y, opts);

[test_err, bad] = nntest(nn, test_x, test_y);
test_pred = nnpredict(nn, test_x);
test_acc = 1 - sum(test_pred'==ylab(test_idx))/size(test_idx,2);
% display(test_acc)

%% porter 

select_idx = 1:156000; % makes numberofbatches integer
x_train = Xtrain(select_idx,:);
y_train = ytrain(select_idx,:);

fnn = nnsetup([size(x_train,2) 50 size(y_train,2)]);
fnn.weightPenaltyL2 = 2e-5;  %  L2 weight decay
opts.learningrate = .1;
opts.numepochs =  10;   %  Number of full sweeps through data
opts.batchsize = 10;  %  Take a mean gradient step over this many samples
[fnn, L] = nntrain(fnn, x_train, y_train, opts);

ypred=nnpredict(fnn, Xtest);

%%
save fnnporter.mat fnn 


%% predict
rand('state',0)
select_idx = 1:156000; % makes numberofbatches integer
x_train = Xtrain(select_idx,:);
y_train = ytrain(select_idx,:);

matlabpool('open',2);

pp = [];
acc = [];
parfor i = 1:1000
    nn = nnsetup([size(x_train,2) 50 size(y_train,2)]);
    minl = 1e-4; maxl = 1e-6;
    nn.weightPenaltyL2 = minl+ rand()*(maxl-minl);  %  L2 weight decay
    opts.learningrate = .1;
    opts.numepochs =  10;   %  Number of full sweeps through data
    opts.batchsize = 10;  %  Take a mean gradient step over this many samples
    [nn, L] = nntrain(nn, x_train, y_train, opts);

    y_pred = nnpredict(nn, Xtest);
    acc = [acc test_acc];
    pp = [pp y_pred];
end

matlabpool close


%% write to file
fileID = fopen('data/ypred.txt','w');
fprintf(fileID,'%d\n',ypred-1); %-1 here
display('written to file data/ypred_porter0227.txt')
fclose(fileID);



%%
fid = fopen('data/testwrite.txt','w');
fprintf(fid, '%d\n', [1])
fclose(fid)
