# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**In 1-2 sentences, explain the problem statement: **

The objective would start to create a compute cluster and configure the training run by creating a HyperDriveConfig and AutoMLConfig to compare. In the end, the run was submitted and the best model was saved and recorded.

## Scikit-learn Pipeline

* DATA: This dataset contains data of a Portuguese banking institution. It contains data of direct phone call marketing, taken from the University of California, Irvine machine learning repository (link: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing.) 
The dataset gives you information about a marketing campaign of a financial institution in which you will have to analyze in order to find ways to look for future strategies in order to improve future marketing campaigns for the bank.
 
 --> input features: principal inputs like age, job, marital, education, housing, loan, contact, month, day_of_weed, duration, campaign, pdays, previous, poutcome, emp.var.rate, nr.employed
 --> target column: y - has the client subscribed a term deposit? (binary: 'yes','no')
 
* Hyperparameter Tuning: Parameters used C (inverse regularization parameter) and max_iter (number of maximun iteractions).

* Accuracy is chosen as the primary metric and logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables.(in this case categorical dependent variable)

* What are the benefits of the parameter sampler you chose?

Random sampling method was chosen because it suport early termination of low - performance runs. It also use less computation resources due to the fact that random sampling randomly selects just hyperparameter values from the defined seach space and not search over the all search space.

* What are the benefits of the early stopping policy you chose?

Bandit policy is based on slack factor/slack amount and evaluation interval and saves time and computation resources unlike MedianStoppingPolicy.

This policy takes the following configuration parameters:

evaluation_interval: the frequency for applying the policy (optional parameter).
delay_evaluation: delays the first policy evaluation for a specified number of intervals (optional parameter).

## AutoML

We can see that VotingEnsemble algorithm show the best model with an accuracy of 0.914723. It involves making a prediction that is the average of multiple other regression models.


## Pipeline comparison

With AutoML the accuracy of best model (Voting Ensemble) was 0.914723 and accuracy of HyperDrive model (Logistic Classifier) was 0.910757

The automl was better by approximately in 1.004%. Automl works best because it creates a series of pipelines in parallel that test different algorithms and parameters for you.

## Future work

It can be verify that dataset is unbalance for that one option would be resampling to even the class imbalance, either by up-sampling the smaller classes or down-sampling the larger classes.

Another option is change accuracy as the primary metric, we can test other performance metrics like Accuracy or F1 Score.

In the HyperDriveConfig object you can add the Cross-validation (CV) parameter with this we provide a validation holdout dataset, specify your CV folds (number of subsets) and automated ML will train your model and tune hyperparameters to minimize error on your validation set.

## Proof of cluster clean up

The compute cluster deletion is included in the code.

## References


- https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#sampling-the-hyperparameter-space 
- https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml 
- https://docs.microsoft.com/en-us/azure/machine-learning/concept-manage-ml-pitfalls 





