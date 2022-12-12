# Udacity-ND-Azure
Dataset: Loans Dataset
Objective: To predict the Probability of Default based on details of customers.
Dataset contained various features which included attributes such as age, gender, income, equity holdings, credit scores from equifax, transunion.etc
For the first phase:

Cluster Setup: STANDARD_D2_V2 with 2 nodes max

The training script used the SVM/SVC classifier (Support Vector Machine) with its parameters configured such as C, kernel type, degree etc.
For the first part of the project the model is trained using SVM SVC with hyperparameter tuning configured from HyperDrive. Note: The Hyperdrive run has been interupted in between to  minimize costs. But the best performing model has been retrieved from the given runs.


For the second part the AutoML has been ran and the best performing model has been reported by the get_best_child() function. The best performing automl model is then saved in the form of pickl file.
