# Predicting Pulmonary Fibrosis FVC Change Over Time from Baseline Patient Characteristics and CT Scans

### UC Berkeley Spring 2020 MIDS Capstone Project by Adam Johns, Marcial Nava and Tosin Akinpelu

This project was based on the Kaggle competition "OSIC Pulmonary Fibrosis Progression: Predict lung function decline" available at https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression

All notebooks were run on an AWS EC2 P3.2xlarge Deep Learning AMI (Ubuntu 16.04) Version 36.0 instance using a TensorFlow 2.3 with Python3 (CUDA 10.2 and Intel MKL-DNN) Environment.

## Overview of Mixed Data Input Neural Network - Adam Johns and Marcial Nava

**Project Goal:** - predict a patient’s severity of decline in lung function based on a CT scan of their lungs and baseline clinical characteristics (FVC, age, gender, smoking status and Percent (a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics)

**Model Goal:** predict the linear decay of FVC measurements within the 95% confidence interval of the actual least squares slope  
Pros of this approach - meaningful measurement that can be interpreted as likely rate of progression and can be visualized; also works well because follow-up and frequency of measurements is different for every patient  
Con of this approach - high degree of variability in measurements means that slope doesn't actually fit the measurements that well; more measurements at the start so line is skewed towards earlier measurements  

**Data Processing** 
Calculate OLS slope for each patient, 25% Train/test split\s\s

**CNN** 
1) DICOM image loading and processing function development
2) First CNN - describe architecture and initial performance
3) Functions to assess results - b/c the MAE loss is hard to interpret and the test data from the Kaggle competition only includes one FVC measurement per patient. Functions take a fixed number of images, calculate slope for each image, and deliver a final slope as mean of all predicted slopes from patient images
4) Masking Images
5) Sequence loader to reduce RAM and make it possible to train model on all images
5) Tuning and optimizing CNN - updating neural network with batch normalization and tuning steps

**Mixed Model** 
1) Building Linear Regression MLP
2) Validating against standard Sklearn linear regression
3) Mixed Model - Concatenate and output layers
4) Tuning Mixed Model - highly difficult

**Results**
CNN performance: 70.5% prediction of FVC slope within 95% CI for test data
MLP performance: 59% (approximate - update with final notebook results)
Mixed Network: 63% (approximate - update with final notebook results)

**Further Work**
- Optimizing image recognition using Resnet or other pre-trained classifier with appropriate weights
- Better approaches to determine if tabular data could be more effectively used i.e. ensemble models
- Exploring data to find a better parametric approximation of real FVC progression or classification problem (i.e. quartiles)
- Determining when mixed networks are better than individual approaches - here it seems the perceptron performance reduced the CNN; for what kinds of problems or data sources would it improve outcomes?

**Conclusions**
This task is particularly challenging - limited data, limited measurements, sparse info on baseline characteristics. CNN performance was impressive and improved over tabular data; approaching range of clinical utility but still likely too low in practice; would need to be validated on larger datasets. Nonetheless CNN improved on tabular data only. Should be considered for use alongside demographic/patient characteristics for tasks that involve informed predictions of likely disease progression at baseline and understanding/segmenting patients.
