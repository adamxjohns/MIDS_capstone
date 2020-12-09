# Predicting Pulmonary Fibrosis FVC Change Over Time from Baseline Patient Characteristics and CT Scans

### UC Berkeley Spring 2020 MIDS Capstone Project

This project was based on the Kaggle competition "OSIC Pulmonary Fibrosis Progression: Predict lung function decline" available at https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression

All notebooks were run on an AWS EC2 P3.2xlarge Deep Learning AMI (Ubuntu 16.04) Version 36.0 instance using a TensorFlow 2.3 with Python3 (CUDA 10.2 and Intel MKL-DNN) Environment.

## Background
Pulmonary fibrosis is a progressive, incurable lung disease which occurs when lung tissue becomes damaged and scarred. As the disease worsens, patients become progressively more short of breath. The disease progresses at different rates and it is difficult to define a patient's likely prognosis early on in the disease.  

## Project Goal
  
 Our goal for this project was to develop an approach to predict decline in lung function as measured by Forced Vital Capacity (FVC) over time based on clinical/demographic characteristics  baseline CT scan images. Reasoning that the clinical usefulness of this algorithm would increase the closer to initial diagnosis it could be applied, we aimed to make use of as little follow-up information as possible and, ideally, to use baseline information only.
 
 The likely end users of this work would be clinicians who have access to patient CT scans and the required knowledge to interpret results and give patients counselling and direction based on the information. Such information could be used to direct treatment decisions, counsel patients, and potentially to direct recruitment for future clinical trials.
 
As a personal goal, as individuals with some background in clinical trial analysis and biostatistics but very little experience in computer vision, we wanted to practice working with images and learn how combinng inaging with clinical/demographic data can enhance the task of predicting disease prognosis. 
 
 ### Understanding the Data
 
 The OISC Pulmonary Fibrosis dataset includes data on 176 patients, and is a mixture of imaging, one-time demographic measurements assessed at baseline, and lung function data provided as time series measured at specified weeks afer baseline.  
   
![Data Schematic](/JPGs/data_structure.png)
  

 Looking at FVC over time, the data follows a mostly downward trend, however it's important to note that if fluctuates up and down depending on the week. Additionally, not all patients reached the end of the follow-up period with a lower FVC than baseline, meaning we can't assume a downward trend for everyone.
   
 ![FVC Over Time](/JPGs/FVC_per_wk.png)  
   
 Number of observations per patient ranges between 6 and 10, with a mean of 8.8. Measurement intervals are not regular, meaning we don't have a common timepoint for all patients at which we can predict a single outcome value.  
   
 Now let's examine the distribution of our tabular variables:
 
 ![Tabular EDA](/JPGs/tabular_EDA.png)
 
 A few key observations jump out:
 - We have very few females in the group
 
 - Most of the patients are ex-smokers and current smokers are very rare
 
 - Age looks to be quite normally distributed around 65-70
 
 - The most common measurement time was 50-60 weeks 
   
   
 ### CT Scan EDA  
   
 Wrapping our heads around the CT Scans took us a little while. CTs use a .dicom image format. They're composed of a number of ordered 2d, 512x512 slices which, when put together, form a 3d image. Looking at a series of images over time can show changes in body position and can allow you to get a sense of the differing shape of the lungs during inhalation and exhalation.
  
![Tabular EDA](/JPGs/multi_slice.png)  
  
The standard unit format is Voxels, but by converting to Houndsfield units (HUs) the scale of the individual measurement can actually be interpreted to show the tissue composition of the picture, with an HU of zero indicating water at standard temperature and pressure, -1000 indicating air, and +2000 indicating dense bone. For our modelling we converted all the image values to HUs.  

We have 33,025 slice images available across our 176 patients. Looking at the distribution of the number of slices available per patient, once again we can see that it varies quite widely:

![CT Slices per Patient](/JPGs/ct_slice_per_pt.png)  
  
The number ranges from 12 slices per patient to 1018, with a median of 94.  
  
  

## Model Approach
  
 
**Project Goal:** - Predict a patient’s severity of decline in lung function based on a CT scan of their lungs and baseline clinical characteristics (FVC, age, gender, smoking status and Percent (a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics). Because of mixed data inputs we chose to use a mixed input neural network composed of a convolutional neural network (CNN) and mulilayer perceptron (MLP) with regression kernels.  

**Model Outcome:** Predict the linear decay of FVC measurements within the 95% confidence interval of the actual least squares slope  
  
<ins>Pros of this approach</ins> - Meaningful measurement that can be interpreted as likely rate of progression and can be visualized; also works well because follow-up and frequency of measurements is different for every patient  
<ins>Con of this approach</ins> - High degree of variability in measurements means that slope doesn't actually fit the measurements that well; more measurements at the start so line is skewed towards earlier measurements  
  

**Data Processing**  
Calculate OLS slope for each patient, 25% Train/test split

**CNN Development Steps**  
1) DICOM image loading and processing function development
2) First CNN - describe architecture and initial performance
3) Functions to assess results - b/c the MAE loss is hard to interpret and the test data from the Kaggle competition only includes one FVC measurement per patient. Functions take a fixed number of images, calculate slope for each image, and deliver a final slope as mean of all predicted slopes from patient images
4) Masking Images
5) Sequence loader to reduce RAM and make it possible to train model on all images
5) Tuning and optimizing CNN - updating neural network with batch normalization and tuning steps

**Mixed Model Development Steps**  
1) Building Linear Regression MLP
2) Validating against standard Sklearn linear regression
3) Mixed Model - Concatenate and output layers
4) Tuning Mixed Model - highly difficult

**Results**  
- CNN performance: 70.5% prediction of FVC slope within 95% CI for test data  
- MLP performance: 59% (approximate - update with final notebook results)  
- Mixed Network: 63% (approximate - update with final notebook results)  

**Further Work**  
- Optimizing image recognition using Resnet or other pre-trained classifier with appropriate weights  
- Better approaches to determine if tabular data could be more effectively used i.e. ensemble models  
- Exploring data to find a better parametric approximation of real FVC progression or classification problem (i.e. quartiles)  
- Determining when mixed networks are better than individual approaches - here it seems the perceptron performance reduced the CNN; for what kinds of problems or data sources would it improve outcomes?  

**Conclusions**  
  
This task is particularly challenging - limited data, limited measurements, sparse info on baseline characteristics. CNN performance was impressive and improved over tabular data; approaching range of clinical utility but still likely too low in practice; would need to be validated on larger datasets. Nonetheless CNN improved on tabular data only. Should be considered for use alongside demographic/patient characteristics for tasks that involve informed predictions of likely disease progression at baseline and understanding/segmenting patients.
