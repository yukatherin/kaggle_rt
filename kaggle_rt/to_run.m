%% load sentiment data
 load ./data/ptrain.data.mat
% load ./data/pntrain.data.mat
%load ./data/strain.data.mat
addpath ../DeepLearnToolBox/util; addpath ../DeepLearnToolbox/NN;

%% label matrix
n_tot = size(ylab,2);
ytrain = repmat(ylab',1,5)==repmat(0:4,n_tot,1);

%% nn parameters
l2w = [1e-4 1e-5 1e-6 5e-6];  
opts.learningrate = .1;
opts.numepochs =  2;   
opts.batchsize = 100;  

%% cv splits indexing
k = 5;
fs = round(size(Xtrain,1)/(k*opts.batchsize))*opts.batchsize;
for i=1:5
    trkidx{i} = (i-1)*fs+1:i*fs;
    tekidx{i} = 1:k*fs;
    tekidx{i}(trkidx{i})=[];
end

%% cv train
ii=1;
for lw=l2w
    testerr = [];
        for fold = 1:k
            display(fold)
            train_x = Xtrain(trkidx{fold},:);
            test_x = Xtrain(tekidx{fold},:);
            train_y = ytrain(trkidx{fold},:);
            test_y = ytrain(tekidx{fold},:);
            nn = nnsetup([size(train_x,2) 40 size(train_y,2)]);
            nn.weightPenaltyL2 = lw;  
            [nn, L] = nntrain(nn, train_x, train_y, opts);
            [test_err, bad] = nntest(nn, test_x, test_y);
            testerr = [testerr test_err];
        end
    testerrhist{ii}=testerr;
    meantesterr{ii}=mean(testerr);
    sprintf('mean err on fold %f', mean(testerr));
    ii = ii+1;
end

%% train on whole set with chosen param
 lstar = 1e-6; %ptrain;
%lstar = 1e-4; %pntrain
%lstar = 1e-6; %strain
rand('state',0)
select_idx = 1:156000; % makes numberofbatches integer, will improve this code later
x_train = Xtrain(select_idx,:);
y_train = ytrain(select_idx,:);

%% final model- takes about .7min per epoch
fnn = nnsetup([size(x_train,2) 50 size(y_train,2)]);
fnn.weightPenaltyL2 = lstar;
fopts.numepochs =  20;   
fopts.batchsize = 100;  
[fnn, L] = nntrain(fnn, x_train, y_train, fopts);

%% final model prediction on test set
ypred=nnpredict(fnn, Xtest);

%% write to file
fileID = fopen('data/ypreds.txt','w');
fprintf(fileID,'%d\n',ypred-1);
display('written to file data/ypreds.txt')
% call python munge_preds.py data/ypred.txt from DeepLearnToolbox/kaggle_rt



