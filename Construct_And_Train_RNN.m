%%Input； pdata
%%In our study, the pata contains four data: 
%%1. pdata.train.FeatureTables. The CNN feature table of train data. In the feature table, a row represents a patient and a column is a feature.
%%2. pdata.train.ClassLabel. The vector indicates in a each row whether the patient belong to long / short survivor. 
%%3. pdata.tune.FeatureTables. The CNN feature table of tuning data. In the feature table, a row represents a patient and a column is a feature.
%%4. pdata.tune.ClassLabel. The vector indicates in a each row whether the patient belong to long / short survivor. 

function [net options]= Construct_And_Train_RNN(pdata)

%%--------------------Contruct RNN-------------------------------------
inputSize = size(pdata.train.FeatureTables{1},1); %Get the number of used CNN features

numHiddenUnits = 2000;
numClasses = length(unique(pdata.train.ClassLabel));

layers = [ ...
    sequenceInputLayer(inputSize,'Name','sequence')
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    dropoutLayer(0.5,'Name','drop')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];  % The RNN structure.
	
%%-------------------Train RNN-----------------------------------------
    
miniBatchSize = 1000;
numObservations = numel(pdata.train.FeatureTables);
numIterationsPerEpoch = floor(numObservations / miniBatchSize);

options = trainingOptions('adam', ... 
    'ExecutionEnvironment','gpu', ...
    'InitialLearnRate',0.0001,...
    'GradientThreshold',1, ...
    'MaxEpochs',20, ...
    'MiniBatchSize',miniBatchSize, ...
    'L2Regularization', 0.1,...
    'SequenceLength','longest', ...
    'Shuffle','every-epoch', ...
    'ValidationData',{pdata.tune.FeatureTables,pdata.tune.ClassLabel}, ...
    'ValidationFrequency',numIterationsPerEpoch, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(pdata.train.FeatureTables,pdata.train.ClassLabel,layers,options);

end
