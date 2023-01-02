# Udacity-Azure Machine Learning Engineer

# Section 1  HyperDrive and AutoML

In this problem, the goal is to predict the Probability or Likelihood of a customer defaulting. The dataset contained data obtained from various financial insitutes accross the world. The features in the dataset includes details like, loan amount, credit score, property held, age, income, equities owned .etc Using these features the ML model seeks to classify whether a client has defaulted or not.

For this problem, a SVM classifier was utlized for the first part of the project.

### Overview

The model was built on Sklearn. There are 2 components. The first component contained scripts which included the setup of workspace, registering the dataset from the datastore. Note, for registering the dataset, i have used the Tabular.delimited_from files class which is a method of the Tabular Data Factory. This part of the script also includes cluster provisionment and ensuring the Virtual Machine as the required libraries such as pandas and sciki-learn. These have been configured using the condadependencies module. The ScriptRunConfig module is used to point to the training script which uses the azure registered dataset by using new_run.get_context().
The second part of the script contains the standard training and testing splits followed by the plot of confusion matrix and the accuracy. Finally the Hyperdrive config is setup which includes RandomParameter Sampling with hyperparameters of SVM which includes the slack parameter, the kernel type and the degree. The RandomParameter sampling simply forms random combinations accross the provided metrics and picks the best combination based on the model performance. The Hyperdrive also took the bandity policy which is a method of early stopping subject to non performance gains. An evaluation interval of 2 and a slack factor of 0.1 was used. The evaluation interval of 2 means it will wait for 2 more epochs to see if any performance interval has been observed. The Slack factor is simply the level of performance gain or loss that is allowed from the best possible performance observed during a specific epoch.

### Script Overview:
![Architecture](https://github.com/Satyake/Udacity-ND-Azure/blob/main/compute.jpg)


### AutoML
AutoML has been used to train the model. AutoML finds the best model to fit by comparing the performance. The primary metric chosen was the Accuracy in this case.
The AutoML returned the following models as having the best performance interms of its accuracy. 
 ITER   PIPELINE                                       DURATION            METRIC      BEST
  WARNING:root:Received unrecognized parameter featureization
    0   MaxAbsScaler LightGBM                          0:00:19             1.0000    1.0000
    1   MaxAbsScaler XGBoostClassifier                 0:00:24             1.0000    1.0000
    2   MaxAbsScaler ExtremeRandomTrees                0:00:22             1.0000    1.0000
    3   VotingEnsemble                                0:00:41             1.0000    1.0000
    4   StackEnsemble                                 0:00:50             1.0000    1.0000
 Note: The Loans dataset is fairly simple and is very easy to classify hence the extreme performance. 
 Without AutoML the SVM SVC yielded an accuracy of 87%. The AutoML approach resulted in 100%.



#  Section 2 Operationalizing Machine Learning Pipelines
In this Section an Azure AutoML pipeline has been created and deployed as a web service. The following Sections list the crucial steps undertaken.
  ### Compute Provisoning
  A Machine Learning Computer Cluster is provisoned using the Azure ML SDK with maximum node count of 4 and priority set to low for saving costs. The following   screenshot lists the same,
![Compute Provisioning](https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-01%20182114.jpg)

 ### Defining AutoML and AutoML Pipeline Steps as Pipelines
 The autoML pipeline steps include the creation of two intermediate steps which contains the model metrics and the other the model data. This is achieved by the TrainingOutput module. These intermediary steps ensure that the appropriate steps are logged. Following this the AutoMLStep is used to create the automl pipeline before finally being called by the Pipeline. Following screenshot shows this process.
 
 ![AutoML pipeline creation and Execution](https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-01%20202347.jpg)
 
 ### Retrieving the best automl model
After the automl runs complete, the best performing model can be viwed in the SDK as well as viewed using the Python SDK. Following lines illustrates few of the automl runs as the framework runs multiple models and evaluates against the metrics specified in the run steps.
 ![AutoML run exerpt](https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-02%20131252.jpg)
 
 Following screenshot shows the best model determined by auto ml. Screenshot attached:
 
 ![AutoML run exerpt]( https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-02%20131702.jpg)

### Publishing the pipeline
Following lines of code show the process for publishing the pipeline and to generate its endpoint.
![AutoML pipeline creation and Execution](https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-01%20202716.jpg)
