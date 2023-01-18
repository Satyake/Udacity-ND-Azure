from azureml.core.model import Model
from tensorflow.keras.models import load_model
import json
import numpy as np
import pandas as pd 
import joblib
def init():
    global model,enc_cols
    model_path=Model.get_model_path('heartdieasenn')
    #col_enc=Model.get_model_path('col_enc:')
    #enc_cols=joblib.load('train_enc_cols.pkl')
    model=load_model(model_path)
    

def run(raw_data):
    data_dict=json.loads(raw_data)['data']
    data=pd.DataFrame.from_dict(data_dict)
    data=pd.DataFrame(data)
    data_enc=pd.get_dummies(data)
    print(data_enc)
    #deploy_cols=data_enc.columns
    #missing_cols=enc_cols.difference(deploy_cols)
    #for col in missing_cols:
    #    data_enc[col]=0
    #data_enc=data_enc[enc_cols]
    y_hat=model.predict(data_enc)
    y_hat=[1 if y>=0.5 else 0 for y in y_hat]
    classes=['False','True']
    predicted_classes=[]
    for p in y_hat:
        predicted_classes.append(classes[p])
    return json.dumps(predicted_classes)
    