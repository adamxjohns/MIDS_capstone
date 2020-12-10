# Predicting Pulmonary Fibrosis FVC Change Over Time from Baseline Patient Characteristics and CT Scans

### UC Berkeley Spring 2020 MIDS Capstone Project

This project was based on the Kaggle competition ["OSIC Pulmonary Fibrosis Progression: Predict lung function decline"](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression).

All notebooks were run on an AWS EC2 P3.2xlarge Deep Learning AMI (Ubuntu 16.04) Version 36.0 instance using a TensorFlow 2.3 with Python3 (CUDA 10.2 and Intel MKL-DNN) Environment.

## Background
Pulmonary fibrosis is a progressive, incurable lung disease which occurs when lung tissue becomes damaged and scarred. As the disease worsens, patients become progressively more short of breath. The disease progresses at different rates and it is difficult to define a patient's likely prognosis early on in the disease.  

## Project Goal
  
 Our goal for this project was to develop an approach to predict decline in lung function as measured by Forced Vital Capacity (FVC) over time based on clinical/demographic characteristics and baseline CT scan images. Reasoning that the clinical usefulness of this algorithm would increase the closer to initial diagnosis it could be applied, we aimed at making use of as little follow-up information as possible and, ideally, to use baseline information only.
 
 The likely end users of this work would be clinicians who have access to patient CT scans and the required knowledge to interpret results and give patients counselling and direction based on the information. Such information could be used to direct treatment decisions, counsel patients, and potentially to direct recruitment for future clinical trials.
 
 One personal goal, as individuals with some background in clinical trial analysis and biostatistics but very little experience in computer vision, was to get some hands-on experience with images and computer vision, and learn how combining imaging with clinical/demographic data can enhance the job of predicting disease prognosis. 
 
 ## Understanding the Data  
   
  ### Tabular Data EDA
 
 The OISC Pulmonary Fibrosis dataset includes data on 176 patients, and is a mixture of imaging, one-time demographic measurements assessed at baseline, and lung function data provided as time series measured at specified weeks afer baseline.  
   
 ![Data Structure](/JPGs/data_structure.png)  
  
 Looking at FVC over time, the data follows a mostly downward trend, however it's important to note that it fluctuates up and down depending on the week. Additionally, not all patients reached the end of the follow-up period with a lower FVC than baseline, meaning we can't assume a downward trend for everyone.
   
 ![FVC Over Time](/JPGs/FVC_per_wk.png)  
   
 The number of observations per patient ranges between 6 and 10, with a mean of 8.8. Measurement intervals are not regular, meaning we don't have a common timepoint for all patients at which we can predict a single outcome value.  
   
 Now let's examine the distribution of our tabular variables:
 
  ![Tabular EDA](/JPGs/tabular_EDA.png)  
  
 A few key observations jump out:
 - We have very few females in the group
 
 - Most of the patients are ex-smokers and current smokers are very rare. This variable is not very informative since we have no information on when patients quit smoking. 
 
 - Age looks to be quite normally distributed around 65-70
 
 - The most common measurement time was 50-60 weeks  
 
To explore feature importance on Future FVC values, we explored various of features as predictors of future FVC values via Recursive Feature Elimination techniques. Compared to previous FVC and previous Percent (a compositite of FVC); none of these variales had significant predictive value for determining future FVC values.


  ![Feature importance via Recursive Feature Eliminiation](/JPGs/features_importance.png) 
   
   
 ### CT Scan EDA 
   
 Wrapping our heads around the CT Scans took us a little while. CTs use a .dicom image format. They're composed of a number of ordered 2d, 512x512 slices which, when put together, form a 3d image.  
 
 ![CT image slices](/JPGs/multi_slice.png)  
 
 Looking at a series of images over time can show changes in body position and can allow you to get a sense of the differing shape of the lungs during inhalation and exhalation.  
  
![CT animation](/JPGs/gif_ID00165637202237320314458.gif)  
  
The converting the standard CT measurement of Voxels to Houndsfield units (HUs) the scale of the individual measurement can actually be interpreted to show the tissue composition of the picture, with an HU of zero indicating water at standard temperature and pressure, -1000 indicating air, and +2000 indicating dense bone. For our modelling we converted all the image values to HUs.  

We have 33,025 slice images available across our 176 patients. Looking at the distribution of the number of slices available per patient, once again we can see that it varies quite widely:

![CT Slices per Patient](/JPGs/ct_slice_per_pt.png)  
  
The number ranges from 12 slices per patient to 1018, with a median of 94.  
  
  

## Mixed Input Neural Network Model

### Defining the Model Outcome and Metrics

When definingng the endpoint for our model, a number of factors were important. First, in keeping with the original Kaggle competition we wanted to see how closely we could predict the specific course of the disease and how close we could get to determining an actual FVC measurement for a given point in time. This ruled out a classification approach and directed us toward regression. Secondly, because patient measurements and follow up are not uniform, we wanted to choose an endpoint to the model that incorporates time as a function.  
This approach has a few pros and cons, as follows:  

| **Model Outcome** | Predict the linear decay of FVC measurements as expressed by the least squares slope of each patient                                                                                          |
|------------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|          **Pros** | - Meaningful measurement that can be easily interpreted and be visualized - Avoids issue of non-uniform measurement times and follow up                                                       |
|          **Cons** | - High degree of variability in measurements means that slope doesn't actually fit the measurements that well - More measurements at the start can skew the slope towards earlier measurements | 
  
 Our chosen loss function was Mean Average Error (MAE), expressed as the difference between the predicted and actual slope. However, since MAE doesn't give us a great degree of transparency, we decided to further assess model performance by another success metric: The number of slopes predicted within the 95% confidence interval of the actual OLS slope.  
 
  On an individual patient basis, we'd be trying to predict a line as close to the true OLS slope (in blue below) as possible, and scoring our results on the basis of how many we got within the CIs (in red).  
 
 ![Model Target](/JPGs/target.png)  


### Data Processing

The full data processing pipeline is included in our notebook repo, but here's a brief summary of the steps we took:

- Subset each patient's FVC and Week measurements and calculate the least squares slope as our target variable  
- Encode the categorical variables representing gender and smoking status, standardize the numerical variables for baseline FVC and Percent, and concatenate them into a 1d vector
- Load our images and convert the voxels to Houndsfield units
- Apply a 25% train-test split

### Model Components  
  
  The structure of our mixed data neural network includes a convolutional neural network to learn features from the images, a multi-layer perceptron to do Keras's version of linear regression on the tabular data, and a concatenate layer to combine the two outputs. We chose Keras because we are noobs.
  
<ins>**CNN**</ins>  
  
  The most important thing to note about our CNN structure is that it's very much a NOOB CNN. It takes 2d slices as the the input shape and applies 16-32-64 filters with relu activation, batch normalization and Max Pooling 2D, then Flatten, Dense(16), relu, another batch normalization, and dropout, another Dense(4) layer and a linear output layer. This structure is in no way optimized for medical imaging and from what we can understand (which is limited) it's basically an all-purpose detector of stuff in images.  
  
<ins>**MLP**</ins>
  
According to the internet, this is how you do regression in Keras.  
  
<ins>**Concatenate Layer and Mixed Input Model**<ins>

  The Mixed Network involves taking the output of the second-to-last layers of the CNN and MLP prior to the linear output layer and concatenating them; then applying another dense(4) layer with relu activation and a linear output layer.  
  
*A Note on Masking*
  
  When approaching our images initially, our hypothesis was that since we were dealing with a lung disease, our model accuracy would improve if we were able to run the CNN on lung tissue only. As such, we applied a masking algorithm adapted from [this guide to DICOM image processing](https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/) to pre-process the images and replace all non-lung tissue with houndsfield units representing water. 

### Model Training  
  
  When training the models, one important thing to note is that the inputs of the CNN and MLP are distinct, but the network needs to see the same information corresponding to the same individual, so it's important to maintain the association between target, image and tabular data corresponding to the same patient in your training pipeline.  
  
  Another important thing to note regarding training is that the [Keras Sequence object](https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence) is a wonderful thing; using the sequence class allowed us to train our models on roughly 24,000 sampled CT slices in less than 30 seconds' time. This significantly decreased model loss and increased accuracy over a subsample of images cached in RAM; it likely wouldn't have been possible to find enough affordable RAM to complete the exercise otherwise. 

  Lastly, when training, we decided to address the imbalance of CT slices between patients by randomly sampling slices from the whole train/test set during each model epoch. This would ensure an approximately equal distribution of slices for each patient. While there was still less image variation available for model training from the patients with fewer slices, we at least didn't end up with bias due to introducing fewer examples from those patients in our training runs.

### Model Inference for 95% CI Success Measure  
  
  To determine the likely slope based on the aggregate of predictions across slices, and without having to do more sophisticated image processing to determine which slices were most likely to provide the best predictions, we did our model inference on the basis of the average of the total predicted slopes for each patient's individual slices. When assessing the number of slices required to get a consistent inference, we started at 20 and worked our way up to all of them. We hit on 40 images as a number that results in consistent results across sample draws but is still computationally manageable.

## Results

  After much arduous toil, our best model was **accurately able to predict the slope within the 95% confidence interval for 73% (32 of 44) of our test patients.**  The results of each of our models is outlined below:  
  
|                    **Model** | **Best Run Accuracy**   (% slopes predicted within 95% CI of OLS) |
|-----------------------------:|:-----------------------------------------------------------------:|
|                      **MLP** | 56.8% (25 of 44)                                                  |
|   **CNN with Masked Images** | 54.5% (24 of 44)                                                  |
| **CNN with Unmasked Images** | 72.7% (32 of 44)                                                  |
|              **Mixed Input** | 65.9% (29 of 44)                                                  |
  
  There were a number of interesting findings from these results. First, regarding the MLP, while the model was learning and predicting distinct slopes for each set of tabular data, it didn't do great. This likely reflects the findings from our earlier machine learning experiments that the amount of explained variance in the FVC decay from this tabular data alone is low, possibly because the information provided in the data is very limited.  
  
  Regarding the masking, while our initial hypothesis was that the masking would improve the results by limiting training to the disease site only, it actually reduced the success of the predictions significantly. We aren't certain why this is, but two possible hypotheses are that either the masking algorithm itself could stand to be improved; or, alternately that the model is learning from other features outside the lungs and needs to be adjusted to learn only from the slices where the lungs are more visible. This is a particularly intriguting hypothesis as theoretically other morphological features that the CNN can detect could be associated with better or worse prediction.  
  
Lastly, the final surprise here was that the tabular data did not improve the performance beyond the CNN on its own. One straightforward hypothesis to explain this is that the poor performance of the MLP didn't add much to the CNN, and given that our network does not apply any weighting to the individual results the drawbacks from the MLP performance outweighed the information gain.  
  

**Some of our Best Model's Successful Predictions**  
  
![Successful Predictions](/JPGs/successes.png)  
  
**And a few of its noble failures**  
  
 
  ![Unsuccessful Predictions](/JPGs/failures.png)  



**Further Work**  
- Optimizing image recognition using Resnet or other pre-trained classifier with appropriate weights  
- Better approaches to determine if tabular data could be more effectively used i.e. ensemble models  
- Exploring data to find a better parametric approximation of real FVC progression or classification problem (i.e. quartiles)
- Exploring other techniques for lung segmentation
- Determining when mixed networks are better than individual approaches - here it seems the perceptron performance reduced the CNN; for what kinds of problems or data sources would it improve outcomes?  

**Conclusions**  
  
This task is particularly challenging - limited data, limited measurements, sparse info on baseline characteristics. CNN performance was impressive and improved over tabular data; approaching range of clinical utility but still likely too low in practice. It would need to be validated on larger datasets. Nonetheless, CNN did not improve when we adjusted the image processing. Despite the difficulty, our results are encouraging. The CNN model should be considered for use alongside demographic/patient characteristics for tasks that involve informed predictions of likely disease progression at baseline and understanding/segmenting patients.

**Sources**
‪
 François Collet (2017). "Deep Learning with Python". Manning Publications.
 DICOM Processing and Segmentation in Python. https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
 Houndsfield Units - Scale of HU, CT numbers. http://radclass.mudr.org/content/hounsfield-units-scale-hu-ct-numbers

