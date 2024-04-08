# Neural_Network_Charity_Analysis
 
# Machine Learning

# Overview of the analysis

Here is the list of deliverables:

* Deliverable 1: Preprocessing Data for a Neural Network Model
* Deliverable 2: Compile, Train, and Evaluate the Model
* Deliverable 3: Optimize the Model
* Deliverable 4: A Written Report on the Neural Network Model (README.md)

Description
Using knowledge of machine learning and neural networks, to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

Specifically,

The files we will be using are the Alphabet Soup Charity dataset (charity_data.csv), Alphabet Soup Charity.ipynb, Alphabet Soup Charity.h5, AlphabetSoupCharity_Optimization.h5. AlphabetSoupCharity_Optimization.ipynb

# Purpose:


# Results:

The following preprocessing steps have been performed:
The EIN and NAME columns have been dropped
![D1](https://github.com/735713038455163/Neural_Network_Charity_Analysis/blob/master/Pictures/D1.PNG)
The columns with more than 10 unique values have been grouped together
![D1a](https://github.com/735713038455163/Neural_Network_Charity_Analysis/blob/master/Pictures/D1a.PNG)
The categorical variables have been encoded using one-hot encoding
![D1b](https://github.com/735713038455163/Neural_Network_Charity_Analysis/blob/master/Pictures/D1b.PNG)
The preprocessed data is split into features and target arrays
![D1c](https://github.com/735713038455163/Neural_Network_Charity_Analysis/blob/master/Pictures/D1c.PNG)
The preprocessed data is split into training and testing datasets
![D1d](https://github.com/735713038455163/Neural_Network_Charity_Analysis/blob/master/Pictures/D1d.PNG)
The numerical values have been standardized using the StandardScaler() module 
![D1e](https://github.com/735713038455163/Neural_Network_Charity_Analysis/blob/master/Pictures/D1e.PNG)


The neural network model using Tensorflow Keras contains working code that performs the following steps:
The number of layers, the number of neurons per layer, and activation function are defined 
An output layer with an activation function is created
![D2](https://github.com/735713038455163/Neural_Network_Charity_Analysis/blob/master/Pictures/D2.PNG)
There is an output for the structure of the model
There is an output of the model’s loss and accuracy 
![D2a](https://github.com/735713038455163/Neural_Network_Charity_Analysis/blob/master/Pictures/D2a.PNG)
The model's weights are saved every 5 epochs 
The results are saved to an HDF5 file 
![D2b](https://github.com/735713038455163/Neural_Network_Charity_Analysis/blob/master/Pictures/D2b.PNG)


The model is optimized, and the predictive accuracy is increased to over 75%, or there is working code that makes three attempts to increase model performance using the following steps:
Noisy variables are removed from features
Additional neurons are added to hidden layers 
![D3c](https://github.com/735713038455163/Neural_Network_Charity_Analysis/blob/master/Pictures/D3c.PNG)
Additional hidden layers are added 
The activation function of hidden layers or output layers is changed for optimization 
The model's weights are saved every 5 epochs 
![D3h](https://github.com/735713038455163/Neural_Network_Charity_Analysis/blob/master/Pictures/D3h.PNG)
The results are saved to an HDF5 file
![D3g](https://github.com/735713038455163/Neural_Network_Charity_Analysis/blob/master/Pictures/D3g.PNG)


# Summary:

## Data Preprocessing
### What variable(s) are considered the target(s) for your model?
- The target for the model is the "Is-Successful" column. It signifies if the money was use effectively.
### What variable(s) are considered to be the features for your model?
- The features of this model are the NAME, APPLICATION, TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, INCOME_AMT,SPECIAL_CONSIDERATIONS, STATUS, and ASK_AMT
### What variable(s) are neither targets nor features, and should be removed from the input data?
- EIN (Employer identificaiton) was dropped because the numbers could confuse the system into thinking its significant. ANSWER: A student could drop SPECIAL_CONSIDERATIONS because there is only a small percentage of cases that had any special consideration, and special considerations cannot be quantified. ANSWER: A student could drop STATUS because all rows were the same value, 1.

## Compiling, Training, and Evaluating the Model
### How many neurons, layers, and activation functions did you select for your neural network model, and why?
- In this model there are three hidden layers each with many neurons, because this seeemed to increased the accuracy above 75%. The number of epochs wasn't changed. The first activation function was 'relu' but the 2nd and 3rd were 'sigmoid'and the output function was 'sigmoid'. Changing the 2nd and 3rd activation functions to 'sigmoid' also helped boost the accuracy.
### Were you able to achieve the target model performance?
Yes
### What steps did you take to try and increase model performance?
- It required converting the NAME column into data points, which has the biggest impact on improving efficiency. And, it also required adding a third layer and using the "sigmoid" activation function for the 2nd and 3rd layer.

## Overall

By increasing the accuracy above 75% we are able to correctly classify each of the points in the test data 75% of the time. And, an applicant has a 80% chance of being successful if they have the following:

The NAME of the applicant appears more than 5 times (they have applied more than 5 times)
The type of APPLICATION is one of the following; T3, T4, T6, T5, T19, T8, T7, and T10
The application has the following CLASSIFICATION; C1000, C2000, C1200, C3000, and C2100.
A good model to recommend is the Random Forest model because Random Forest are good for classification problems. Using this model produces a 79% accuracy.




