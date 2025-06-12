from fastapi import FastAPI,File,UploadFile,Query
from config import models_root_path
import joblib
import os
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from typing import Annotated
import shutil

app = FastAPI()


class RFM(BaseModel):
    Recency:int
    Frequency:int
    Monetary:int
    



@app.get("/versions")
async def getVersions():
    os.makedirs(models_root_path,exist_ok=True)
    models = [name for name in os.listdir(models_root_path) if os.path.isdir(os.path.join(models_root_path, name))]
    return models

@app.get("/{version}/predict")
async def predict(version:str,
                  Recency: int = Query(),
                  Frequency: int = Query(),
                  Monetary: float = Query()):
    model = joblib.load(os.path.join(models_root_path,version,"model.pkl"))
    scaler = joblib.load(os.path.join(models_root_path,version,"scaler.pkl"))
    df = pd.DataFrame([
        {
            "Recency":Recency,
            "Frequency":Frequency,
            "Monetary":Monetary
        }
    ])
    df.columns = ["Recency","Frequency","Monetary"]
    df = scaler.transform(df)
    print(df)
    prediction = model.predict(df)
    return int(prediction)
    
    
    
    
    
@app.post("/save/{version}")
async def save(version:str,model:Annotated[UploadFile,File()],scaler:Annotated[UploadFile,File()]):
    model_path = os.path.join(models_root_path,version,"model.pkl")
    scaler_path = os.path.join(models_root_path,version,"scaler.pkl")
    version_path = os.path.join(models_root_path, version)
    os.makedirs(version_path, exist_ok=True)
    with open(model_path,"wb") as buffer:
        shutil.copyfileobj(model.file,buffer) 
    with open(scaler_path,"wb") as buffer:
        shutil.copyfileobj(scaler.file,buffer)   
        
        
    return f"Save model on version {version}"