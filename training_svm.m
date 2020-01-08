clc;
clear all;
close all;


outputFolder = fullfile('fruits');
rootFolder = fullfile(outputFolder, 'Training');
 
categories = {''};



imds = imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');
tbl = countEachLabel(imds)
% minSetCount = min(tbl{:,2});
% 
% imds = splitEachLabel(imds, minSetCount, 'randomize');
% countEachLabel(imds);

[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');

tic,bag = bagOfFeatures(trainingSet);

categoryClassifier = trainImageCategoryClassifier(trainingSet,bag);toc

tic,confMatrix = evaluate(categoryClassifier,testSet);toc

mean(diag(confMatrix))
save db;