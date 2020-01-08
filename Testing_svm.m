clc;
clear all;
close all;

load db;

[fname, path]=uigetfile('fruits/Test/.jpg','provide an Image for testing');
fname=strcat(path, fname);
newImage = imread(fname);

myicon = imread(fname);
imshow(newImage);

[labelIdx, score] = predict(categoryClassifier,newImage);
categoryClassifier.Labels(labelIdx)

