# Udacity-Azure Machine Learning Engineer

# Section 1  HyperDrive and AutoML

In this problem, the goal is to predict the Probability or Likelihood of a customer defaulting. The dataset contained data obtained from various financial institutes across the world. The features in the dataset include details like loan amount, credit score, property held, age, income, equities owned .etc Using these features the ML model seeks to classify whether a client has defaulted or not.
For this problem, an SVM classifier was utilized for the first part of the project.

### Overview

The model was built on Sklearn. There are 2 components. The first component contained scripts that included the setup of the workspace, and registering the dataset from the datastore. Note, for registering the dataset, I have used Tabular.delimited_from files class which is a method of the Tabular Data Factory. This part of the script also includes cluster provisioning and ensuring the Virtual Machine as the required libraries such as pandas and sciki-learn. These have been configured using the condadependencies module. The ScriptRunConfig module is used to point to the training script which uses the azure registered dataset by using new_run.get_context().
The second part of the script contains the standard training and testing splits followed by the plot of the confusion matrix and the accuracy. Finally, the Hyperdrive config is set up which includes RandomParameter Sampling with hyperparameters of SVM which includes the slack parameter, the kernel type, and the degree. RandomParameter sampling simply forms random combinations across the provided metrics and picks the best combination based on the model performance. The Hyperdrive also took the bandit policy which is a method of early stopping subject to non-performance gains. An evaluation interval of 2 and a slack factor of 0.1 was used. The evaluation interval of 2 means it will wait for 2 more epochs to see if any performance interval has been observed. The Slack factor is simply the level of performance gain or loss that is allowed from the best possible performance observed during a specific epoch.

### Script Overview:
<img src="https://github.com/Satyake/Udacity-ND-Azure/blob/main/compute.jpg" width=30% height=30%>

### AutoML
AutoML has been used to train the model. AutoML finds the best model to fit by comparing the performance. The primary metric chosen was Accuracy in this case.
The AutoML returned the following models as having the best performance in terms of its accuracy. 
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
  For this exercise, I have used an external storage account created through the azure portal and then registered using the .register_azure_blob_container() method. Once this is done, the datastore is called along with the container specified in the previous method with the Dataset.Tabular From Delimited files. This was used since we are essentially working with CSV files. The section of code has been visualized for reference.
  
   
   <img src="https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-04%20101737.jpg" width=30% height=30%>

   
   Registered Dataset as seen on the Azure portal in the ML workspace.
   
   <img src="https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-04%20101923.jpg" width=30% height=30%>



  ### Compute Provisoning
  A Machine Learning Computer Cluster is provisioned using the Azure ML SDK with a maximum node count of 4 and priority set to low for saving costs. The following screenshot lists the same.
   
 <img src="https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-01%20182114.jpg" width=30% height=30%>

 ### Defining AutoML and AutoML Pipeline Steps as Pipelines
The autoML pipeline steps include the creation of two intermediate steps which contain the model metrics and the other the model data. This is achieved by the TrainingOutput module. These intermediary steps ensure that the appropriate steps are logged. Following this, the AutoMLStep is used to create the automl pipeline before finally being called by the Pipeline. The following screenshot shows this process.
 
 
  <img src="https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-01%20202347.jpg" width=30% height=30%>

 
 Executed Pipeline Auto ML
 
  <img src="https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-04%20102125.jpg" width=30% height=30%>

 
 ### Retrieving the best automl model
After the automl runs are complete, the best performing model can be viewed in the SDK as well as viewed using the Python SDK. The following lines illustrates a few of the automl runs as the framework runs multiple models and evaluates against the metrics specified in the run steps.

  <img src="https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-02%20131252.jpg" width=30% height=30%>

 
 Following screenshot shows the best model determined by auto ml. Screenshot attached:
 
   <img src="https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-02%20131702.jpg" width=30% height=30%>


### Publishing the pipeline
Following lines of code show the process for publishing the pipeline and to generate its endpoint.

<img src="https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-01%20202716.jpg" width=30% height=30%>


Once the endpoint is called using the authentication headers and the JSON payload the pipeline gets executed. The following screenshot confirms this. The first run was of the actual automl pipeline when it was run the first time. The second one was when it was called through the notebook.

<img src="https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-04%20105430.jpg" width=30% height=30%>


Published pileine and endpoint as shown in the following screenshot:

<img src="https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-05%20142245.jpg" width=30% height=30%>

Pipleine calls and active status of endpoints.

<img src="https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-05%20154605.jpg" width=30% height=30%>

### Model Deployment
Before the best model can be deployed it needs to be registered to Azure first. The following lines of code demonstrate this. I have registered the model using the run variable defined from the workspace.get_run() method which took the run id. The Run ID can be found in the Azure UI.


<img src="https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-02%20132147.jpg" width=30% height=30%>


Once the Model Registration is complete we set up the environments and necessary cluster provisioning for deployment. The model is deployed on the Azure Container Instance. Before deployment, an inference configuration is required which takes in a scoring script and an optional environment definition YAML file. In my case the run environment was automatically inferred hence the YAML file was not used. After this, we set up a deployment config that specifies to azure the amount of ram and CPU cores to use. Finally, the model is deployed using the Model.Deploy method. The following screenshot illustrates the same. 

<img src="https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-02%20132634.jpg" width=30% height=30%>

Screenshot showing healthy deployment with the rest endpoint.


<img src="https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-02%20132746.jpg" width=30% height=30%>

After we have the deployment up and running. As an additional requirement for this exercise, Swagger was used for identifying the input parameters for the model for the endpoints. This can be done by visual inspection however swagger simplifies this process. To run swagger, The local machine requires docker installed. The Swagger.json file is the swagger URL from the azure deployment interface, which is essentially a JSON file. The swagger.json  and serve.py have to be in the same folder for it to run. The following screenshot shows the swagger service running on the local machine.

<img src="https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-02%20150136.jpg" width=30% height=30%>

Following screenshot shows the  REST endpoint and the authentication. (For obvious reasons the authentication has been deactivated ).

<img src="https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-02%20151447.jpg" width=30% height=30%>

Following screenshot shows the endpoint calls though the notebook (VSCODE):

<img src="(https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-04%20105735.jpg" width=30% height=30%>

### Logs 

The Service Logs as viewed in the workspace:

<img src="(https://github.com/Satyake/Udacity-ND-Azure/blob/main/Screenshot%202023-01-04%20132941.jpg" width=30% height=30%>



### Future Directions
In this case, I would have made sure of using the PythonScript Step to have additional control for the training and validation tasks. The use of AutoML is good when not much domain expertise is needed. In my opinion, I would take advantage of other Synthethic Feature Engineering techniques like SMOTE. Things like class separability linear or non-linear is a crucial aspect to be understood by Data professionals. Certain assumptions must also be validated before any machine learning model is used hence my apprehension about AutoML.

### Comment about model performance: 
The model performance is poor in account of its severe under-represented 'yes' category hence the model performance is poor. 

Screencast: https://www.youtube.com/watch?v=LD4z6BcimK0

Screencast of Missing Requirements: https://youtu.be/DLJ1f7JU_jI (Post Review)


# Section 3 : Capstone Project 

Dataset: A dataset for detecting the heart diesease [https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset]

Model Selection: For the first phase of the work , AutoML has been used. The autoML model resulted in the following metrics:

Accuracy: 85.5%,    FScore: 85.4%,  Precision: 93.5% 



The details can be viewed under the AutoML Results section. The following screenshot shows the automl settings that have been used
![cap](https://github.com/Satyake/Udacity-ND-Azure/blob/main/automlsettings-cap.jpg)

As a second deliverable for the capstone, I have decided to go with a Neural Network (Multi-Layered Perceptron) which included 29 nodes in the first layers the second one 59 nodes and the final layer had a single node with a Sigmoid activation function. The sigmoid activation function generates a probability score between 0 and 1. The model performance was similar to the automl performance results.

Accuracy: 91.7%,   FScore: 91.9%,   Precision: 95%

Training Regimen: The Neural Network was trained for 500 epochs with a batch size of 12. The train-validation split was set at 25%. The model also leveraged the use of callbacks. The callbacks included a parameter for the Variable Learning rate on the plateau, a loss monitor that tracks the validation accuracy, and a model checkpoint for saving the best model based on improvements. This is discussed in detail in the next sections.

### AutoML Results

AutoML Run details as viewed below:
![cap](https://github.com/Satyake/Udacity-ND-Azure/blob/main/automlrundetails1-cap.jpg)

AutoML Performance Results:
AutoML metrics for the best model

![cap](https://github.com/Satyake/Udacity-ND-Azure/blob/main/automl%20other%20metrics1.jpg)

### Neural Network Results (MLP)
The structure of the neural network can be visually shown as seen below. The figure was generated using an SVG creator based on https://alexlenail.me/NN-SVG/index.html.
![cap](https://github.com/Satyake/Udacity-ND-Azure/blob/main/nn%20(1).svg).

The model accepts an input dimension of 13 elements. The second layer has 29 nodes and the third layer has 59 nodes with a final dense layer of a single node with a sigmoid activation function. The single element in the final dense layer is because of a binary classification problem statement. The model had 2236 trainable parameters for the given inputs. 
The following callbacks were added
1) ReduceLRonPlateau: 0.2. The validation loss is monitored before the learning rate is reduced by the given factor if the validation loss plateaus.
2) Checkpoint: Tensorflow periodically saves the best performance by monitoring the model checkpoints in each folder created by the callback.

The following screenshot shows the NN performance registed by the Azure GUI
![cap](https://github.com/Satyake/Udacity-ND-Azure/blob/main/NNConfusionMatrix1-cap.jpg)

#### Hyperdrive(ing) the NN with the optimizer and epochs
For the capstone project, I have decided to conduct a Grid Sampling of Hyperparameter tuning for its epochs and the optimizer.
The Optimizers were configured as "RMSPROP", "ADAM", and "SDG" and the Epochs were configured as 100, 200, and 500. No policy was used as this is a neural network.
The following Screenshot shows the successful completion of the Hyperdrive

![cap](https://github.com/Satyake/Udacity-ND-Azure/blob/main/hyperdriverun-cap1.jpg)

Following figure shows the best hyperdrive combination

![cap](https://github.com/Satyake/Udacity-ND-Azure/blob/main/hyperdrivebestrunmetrics-cap.jpg)


An important point to consider, due to the stochastic nature of a neural network there will be performance variation as a result of the weight initialization technique. For the deployment of the NN, i have used the best model which gave the highest F-score.

### Deployment of NN

The model is registered and saved as .h5 file. Following screenshot shows the model registration:

![cap](https://github.com/Satyake/Udacity-ND-Azure/blob/main/h5%20register-cap.jpg)

Registerd MLP model as shown in the GUI:
![cap](https://github.com/Satyake/Udacity-ND-Azure/blob/main/registered%20model.jpg)

Successfull Deployment of the NN model

![cap](https://github.com/Satyake/Udacity-ND-Azure/blob/main/NN_dep.jpg)

Json Model Response
![cap](https://github.com/Satyake/Udacity-ND-Azure/blob/main/response-cap.jpg)









