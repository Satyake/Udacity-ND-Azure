# Udacity-Azure Machine Learning Engineer

# Section 1  HyperDrive and AutoML

In this problem, the goal is to predict the Probability or Likelihood of a customer defaulting. The dataset contained data obtained from various financial insitutes accross the world. The features in the dataset includes details like, loan amount, credit score, property held, age, income, equities owned .etc Using these features the ML model seeks to classify whether a client has defaulted or not.

For this problem, a SVM classifier was utlized for the first part of the project.

### Overview

The model was built on Sklearn. There are 2 components. The first component contained scripts which included the setup of workspace, registering the dataset from the datastore. Note, for registering the dataset, i have used the Tabular.delimited_from files class which is a method of the Tabular Data Factory. This part of the script also includes cluster provisionment and ensuring the Virtual Machine as the required libraries such as pandas and sciki-learn. These have been configured using the condadependencies module. The ScriptRunConfig module is used to point to the training script which uses the azure registered dataset by using new_run.get_context().
The second part of the script contains the standard training and testing splits followed by the plot of confusion matrix and the accuracy. Finally the Hyperdrive config is setup which includes RandomParameter Sampling with hyperparameters of SVM which includes the slack parameter, the kernel type and the degree. The RandomParameter sampling simply forms random combinations accross the provided metrics and picks the best combination based on the model performance. The Hyperdrive also took the bandity policy which is a method of early stopping subject to non performance gains. An evaluation interval of 2 and a slack factor of 0.1 was used. The evaluation interval of 2 means it will wait for 2 more epochs to see if any performance interval has been observed. The Slack factor is simply the level of performance gain or loss that is allowed from the best possible performance observed during a specific epoch.

### Script Overview:
<img src="https://github.com/Satyake/Udacity-ND-Azure/blob/main/compute.jpg" width=30% height=30%>

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

  ### Dataset and Datastore Registration
  For this excercise i have used an external storage account created through the azure portal and then registered using the .register_azure_blob_container() method. Once this is done, the datastore is called along with the container specified in the previous method with the Dataset.Tabular From Delimited files. This was used since we are essentially working with csv files. The section of code has been visualized for reference.
  
   
   <img src="https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-04%20101737.jpg" width=30% height=30%>

   
   Registered Dataset as seen on the Azure portal in the ML workspace.
   
   <img src="https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-04%20101923.jpg" width=30% height=30%>



  ### Compute Provisoning
  A Machine Learning Computer Cluster is provisoned using the Azure ML SDK with maximum node count of 4 and priority set to low for saving costs. The following   screenshot lists the same,
![Compute Provisioning](https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-01%20182114.jpg)

 ### Defining AutoML and AutoML Pipeline Steps as Pipelines
 The autoML pipeline steps include the creation of two intermediate steps which contains the model metrics and the other the model data. This is achieved by the TrainingOutput module. These intermediary steps ensure that the appropriate steps are logged. Following this the AutoMLStep is used to create the automl pipeline before finally being called by the Pipeline. Following screenshot shows this process.
 
 ![AutoML pipeline creation and Execution](https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-01%20202347.jpg)
 
 Executed Pipeline Auto ML
 
  ![Executed Pipeline]( https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-04%20102125.jpg)

 
 ### Retrieving the best automl model
After the automl runs complete, the best performing model can be viwed in the SDK as well as viewed using the Python SDK. Following lines illustrates few of the automl runs as the framework runs multiple models and evaluates against the metrics specified in the run steps.
 ![AutoML run exerpt](https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-02%20131252.jpg)
 
 Following screenshot shows the best model determined by auto ml. Screenshot attached:
 
 ![AutoML run exerpt]( https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-02%20131702.jpg) 

### Publishing the pipeline
Following lines of code show the process for publishing the pipeline and to generate its endpoint.
![AutoML pipeline creation and Execution](https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-01%20202716.jpg)

Once the endpoint is called using the authenthication headers and the Json payload the pipeline gets executed. Following screenshot confirms this. The first run was of the actual automl pipeline when i was run the first time. The second one was when it was called though the notebook.

![Pipeline](https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-04%20105430.jpg)

Published pileine and endpoint as shown in the following screenshot:
![Pipeline](https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-05%20142245.jpg)

Pipleine calls and active status of endpoints.

![Pipeline](https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-05%20154605.jpg)

### Model Deployment
Before the best model can be deployed it needs to be registered to Azure first. Following lines of code demonstrate this. I have registered the model using the run variable defined from the workspace.get_run() method which took the run id. The Run ID can be found in the Azure UI.

![Deployments](https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-02%20132147.jpg)

Once the Model Registration is complete we set up the environments and necessary cluster provisoning for deployment. The model is deployed on the Azure Container Instance. Before deployment an inference configuration is required which takes in a scoring script and an optional environment definition yaml file. In my case the run environment was automatically inferred hence the yaml file was not used. After this we set up a deployment config which specifies to azure the amount of ram and cpu cores to use. Finally the model is deployed using the Model.Deploy method. Following screenshot illustrates the same. 

![Deployments](https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-02%20132634.jpg)

Screenshot showing healthy deployment with the rest endpoint.
![Deployments](https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-02%20132746.jpg)

After we have the deployment up and running. As an additional requirement for this excercise, Swagger was used for identifying the input parameters for the model for the endpoints. This can be done by visual inspection however swagger simplifies this process. In order to run swagger, The local machine requires docker installed. The Swagger.json file is basically the swagger url from azure deployment interface, which is essentially a json file. The swagger.json  and serve.py has to be in the same folder in order for it to run. Following screenshot shows the swagger service running on the local machine.

![Swagger](https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-02%20150136.jpg)

Following screenshot shows the  REST endpoint and the authentication. (For Obvious reasons the authentication has been deactivated for security reasons).
![Swagger](https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-02%20151447.jpg)

Following screenshot shows the endpoint calls though the notebook (VSCODE):

![deployment](https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-04%20105735.jpg)

### Logs 
The Service Logs as viewed in the workspace:
![deployment](https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-04%20132941.jpg)


### Future Directions
In this case i would have made sure of using the PythonScript Step to have additional control for the training and validation tasks. The use of AutoML is good when not much domain expertise is needed. In my opinion i would take advantage of other Synthethic Feature Engineering techniques like SMOTE. Things like class seperability linear or non linear is a crucial aspect to be understood by Data professionals. Certain assumptions must also be validated before any machine learning model is used hence my apprehension for AutoML.

### Comment about model performance: 
The model performance is poor in account for its severe under-represented 'yes' category hence the model performance is poor. 

Screencast: https://www.youtube.com/watch?v=LD4z6BcimK0

Screencast of Missing Requirements: https://youtu.be/DLJ1f7JU_jI (Post Review)


# Section 3 : Capstone Project 

Dataset: A dataset for detecting the heart diesease [https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset]

Model Selection: For the first phase of the work , AutoML has been used. The autoML model resulted in the following metrics:

Accuracy: 85.5%,    FScore: 85.4%,  Precision:93.5% 



The details can be viewed under the AutoML Results section. The following screenshot shows the automl settings that have been used
![cap](https://github.com/Satyake/Udacity-ND-Azure/blob/main/automlsettings-cap.jpg)

As a second deliverable for the capstone, i have decided to go with a Neural Network (Multi Layered Perceptron) which included 29 nodes in the first layers the second one had 59 nodes and the final layer had a single node with a Sigmoid activation function. The sigmoid activation function generates a probability score between 0 and 1. The model performance was similar to the automl performance results.

Accuracy: 91.7%,   FScore: 91.9%,   Precision: 95%

Training Regimen: The Neural Network was trained for 500 epochs with a batch size of 12. The train-validation split was set at 25%. The model also leveraged the use of callbacks. The callbacks included a parameter for Variable Learning rate on plateau, a loss monitor that tracks the validation accuracy and a model checkpoint for saving the best model based on improvements. This is discussed in details in the next sections.

### AutoML Results

AutoML Run details as viewed below:
![cap](https://github.com/Satyake/Udacity-ND-Azure/blob/main/automlrundetails1-cap.jpg)

AutoML Performance Results:
AutoML metrics for the best model

![cap](https://github.com/Satyake/Udacity-ND-Azure/blob/main/automl%20other%20metrics1.jpg)

### Neural Network Results (MLP)
The structure of the neural network can be visually shown as seen below. The figure was generated using an SVG creator based on https://alexlenail.me/NN-SVG/index.html.
![cap](https://github.com/Satyake/Udacity-ND-Azure/blob/main/nn%20(1).svg).

The model accepts an input dimension of 13 elements. The second layer has 29 nodes and the third layer has 59 nodes with a final dense layer of a single node with a sigmoid activation function. The single element in the final dense layer is because of binary classification problem statement. The model had 2236 trainable parameters for the given inputs. 
The following call backs were added
1) ReduceLRonPlateau: 0.2. The validation loss is monitored before the learning rate is reduced by the fiven factor if validation loss plateaus.
2) Checkpoint: Tensorflow periodically saves the best performance by monitoring the model checkpoints in each folders created by the callback.

The following screenshot shows the NN performance registed by the Azure GUI
![cap](https://github.com/Satyake/Udacity-ND-Azure/blob/main/NNConfusionMatrix1-cap.jpg)

#### Hyperdrive(ing) the NN with the optimizer and epochs
For the purpose of the capstone project i have deided to conduct a Grid Sampling of Hyperparameter tuning for its epochs and the optimizer.
The Optimizers were configred as "RMSPROP" , "ADAM" ,"sdg" and the Epochs were configured as 100, 200 and 500. No policy was used as this is a neural network.
Following Screenshot shows the Hyperdrive run of the Neural network in progress:



### Deployment of NN








