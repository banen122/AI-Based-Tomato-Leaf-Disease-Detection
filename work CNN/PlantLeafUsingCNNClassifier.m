%**************************************************************************
%% Plant Leaf Disease Classifier using a CNN and data augmentation
% MULTIPLE USED BY USER  
% PlantLeafUsingCNNClassifier.m 
% This program used to read " An open access repository of images on plant 
% health to enable the development of mobile disease diagnostics" 
% dataset which it download from website:
% https://www.crowdai.org/challenges/1
%
% The data records use in this program is contain (18162) images for
% 10 Disease effict Tomato(18162) with label each Disease type.
%% The program goal are:
%   1- Create image data store for Tarin\test dataset.
%   2- Create training and validation sets.
%   3- Build a CNN.
%   4- Report accuracy of baseline classifier on Tarining set.
%   5- Report accuracy of baseline classifier on Testing set.
% 
% Input:
%      PlantVillage structure include 10 Disease type:
%        1- Tomato Target_Spot
%        2- Tomato Tomato_mosaic_virus
%        3- Tomato Tomato_YellowLeaf__Curl_Virus
%        4- Tomato Bacterial_spot
%        5- Tomato Early_blight
%        6- Tomato healthy
%        7- Tomato Late_blight
%        8- Tomato Leaf_Mold 
%        9- Tomato Septoria_leaf_spot
%        10-Tomato Spider_mites_Two_spotted_spider_mite 
%       
% Output
%
%   Tarined CNN
%  
% (c) 2021-2022 by ????????????????
% Department of Computer Science / Cihan University 
% jun, 2022
%**************************************************************************
%% PART 1: Baseline Classifier
clc,close all,clear all;
%% check dataset folder 
myDATASET = 'PlantVillage';
if ~isfolder(myDATASET)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myDATASET);
  uiwait(warndlg(errorMessage));
  return;
end
%% Create image data store
imds = imageDatastore(fullfile(myDATASET),...
'IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames');
fprintf ('\nPlant Leaf Disease Classifier using a CNN and data augmentation\n');
% Count number of images per label and save the number of classes
totalNumOfLabels = size (imds.Labels,1);
labelCount = countEachLabel(imds)
 numClasses= height(labelCount);
fprintf ('Total Number Of Labels is [%d] ',totalNumOfLabels );
fprintf (', The Number of classes is  [%d] \n',numClasses);
%% Create training and validation sets
trainingPercentage = 0.7;
% Create two new datastores from the files in imds by randomly drawing from
% each label. The first datastore imds1 contains one random file with the 
%  demos label and one random file with the imagesci label. The second 
% datastore imds2 contains the remaining files from each label.
% rng('default') % For reproducibility
[imdsTrainingSet, imdsTestingSet] = splitEachLabel(imds, trainingPercentage, 'randomize');
imdsTrainingSet;

%% Build a  CNN 
dim   = 128;

imageSize = [dim dim 3];
% An augmented image datastore generates batches of new images, with 
% optional preprocessing such as resizing, rotation, and reflection, based 
% on the training images. Augmenting image data helps prevent the network 
% from overfitting and memorizing the exact details of the training images.
% It also increases the effective size of the training data set.
TrainingSet   = augmentedImageDatastore(imageSize,imdsTrainingSet);
TestingSet   = augmentedImageDatastore(imageSize,imdsTestingSet);

% Specify the convolutional neural network architecture.
layers = [
    imageInputLayer(imageSize)
    
    convolution2dLayer(3, 10,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,20,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    %------------------------------------------------------------
      maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    %---------------------------------------------------------------
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,30,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

%% Specify training options 
% options = trainingOptions('sgdm', ...
%     'MiniBatchSize',10, ...
%     'MaxEpochs',25, ...
%     'InitialLearnRate',0.00001, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',TestingSet, ...
%     'ValidationFrequency',10, ...
%     'Verbose',false, ...
%     'Plots','training-progress');
options = trainingOptions('sgdm', ...
    'MaxEpochs',15, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ValidationData',{TestingSet,imdsTestingSet.Labels},...
    'ValidationPatience',Inf);

%% Train the network
net1 = trainNetwork(TrainingSet,layers,options);
save('mynet.mat','net1');
%% Report accuracy of baseline classifier on Tarining set
load('mynet.mat','net1');
YPred = classify(net1,TrainingSet);
TrainingSetLabels = imdsTrainingSet.Labels;

imdsAccuracy = sum(YPred == TrainingSetLabels)/numel(TrainingSetLabels)*100;
% Plot confusion matrix
fprintf...
('\n Rate the overall  accuracy  in Testing set = %.2f \n',...
               imdsAccuracy);          
figure, plotconfusion(TrainingSetLabels,YPred);
title(['Rate the overall performance accuracy [' ...
       num2str( imdsAccuracy ) ' in Tarining set]']);  
%% Report accuracy of baseline classifier on Testing set
YPred = classify(net1,TestingSet);
YValidation = imdsTestingSet.Labels;
imdsAccuracy = sum(YPred == YValidation)/numel(YValidation)*100;
% Plot confusion matrix
fprintf...
('\n Rate the overall  accuracy of CNN in Testing set = %.2f \n',...
               imdsAccuracy);          
figure, plotconfusion(YValidation,YPred)
title(['Rate the overall performance accuracy of CNN[' ...
       num2str( imdsAccuracy ) ' in Testing]']);  



