from azureml.core import Workspace, Experiment, Run, Dataset
import pandas as pd
new_run=Run.get_context()
#retrieve workspace from the experiment
workspace=new_run.experiment.workspace
#form argument parser
from argparse import ArgumentParser
parser=ArgumentParser()
parser.add_argument('--input-data',type=str)
parser.add_argument('--C',type=float,default=1.0)
parser.add_argument('--kernel',type=str)
parser.add_argument('--degree',type=int)
args=parser.parse_args()
#load dataset
dataset=new_run.input_datasets['raw_data'].to_pandas_dataframe()
dataset=dataset.fillna(dataset.mode().iloc[0])
#dummy variables
dataset=pd.get_dummies(dataset,drop_first=True)
dataset.head(2)
#splits
x=dataset.drop(['Status'],axis=1)
y=dataset['Status']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.svm import SVC
svc=SVC(kernel=args.kernel,C=args.C,degree=args.degree)
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm_dict=  {
       "schema_type": "confusion_matrix",
       "schema_version": "1.0.0",
       "data": {
           "class_labels": ["Default", "Non Default"],
           "matrix": cm.tolist()
       }
   }
new_run.log_confusion_matrix("Confusion Matrix",cm_dict)
new_run.log('accuracy',accuracy_score(y_test,y_pred))
new_run.complete()