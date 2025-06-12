from fastapi import FastAPI,File,UploadFile,Query
from config import models_root_path
import joblib
import os
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from typing import Annotated
import shutil
from fastapi.responses import FileResponse


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
    label_map = joblib.load(os.path.join(models_root_path,version,"label_map.pkl"))
    return label_map[int(prediction)]
    
    
    
@app.delete("/{version}")
async def deleteModel(version:str):
    version_path = os.path.join(models_root_path, version)
    shutil.rmtree(version_path, ignore_errors=False)
    
    
@app.post("/save/{version}")
async def save(
        version:str,
        model:Annotated[UploadFile,File()],
        scaler:Annotated[UploadFile,File()],
        label_map:Annotated[UploadFile,File()],
        evals:Annotated[UploadFile,File()],
        PCA:Annotated[UploadFile,File()],
        RxF:Annotated[UploadFile,File()],
        FxM:Annotated[UploadFile,File()],
        RxM:Annotated[UploadFile,File()],
        
        ):
    
    model_path = os.path.join(models_root_path,version,"model.pkl")
    scaler_path = os.path.join(models_root_path,version,"scaler.pkl")
    label_map_path = os.path.join(models_root_path,version,"label_map.pkl")
    evals_path = os.path.join(models_root_path,version,"evals.pkl")
    PCA_path = os.path.join(models_root_path,version,"PCA.png")
    RxF_path = os.path.join(models_root_path,version,"RxF.png")
    FxM_path = os.path.join(models_root_path,version,"FxM.png")
    RxM_path = os.path.join(models_root_path,version,"RxM.png")
    
   
    
    version_path = os.path.join(models_root_path, version)
    os.makedirs(version_path, exist_ok=True)
    with open(model_path,"wb") as buffer:
        
        shutil.copyfileobj(model.file,buffer) 
    with open(scaler_path,"wb") as buffer:
        shutil.copyfileobj(scaler.file,buffer)
        
    with open(label_map_path,"wb") as buffer:
        shutil.copyfileobj(label_map.file,buffer) 
        
    with open(evals_path,"wb") as buffer:
        shutil.copyfileobj(evals.file,buffer) 
        
    with open(PCA_path,"wb") as buffer:
        shutil.copyfileobj(PCA.file,buffer)
        
    with open(RxF_path,"wb") as buffer:
        shutil.copyfileobj(RxF.file,buffer)
         
    with open(FxM_path,"wb") as buffer:
        shutil.copyfileobj(FxM.file,buffer)
        
    with open(RxM_path,"wb") as buffer:
        shutil.copyfileobj(RxM.file,buffer)  
        
        
    return f"Save model on version {version}"

@app.get("/info/{version}/evals")
async def getEvals(version:str):
    evals = joblib.load(os.path.join(models_root_path,version,"evals.pkl"))
    return evals

@app.get("/info/{version}/PCA")
async def getPCA(version:str):
        PCA_path = os.path.join(models_root_path,version,"PCA.png")
        return FileResponse(PCA_path,media_type='application/octet-stream', filename="PCA.png")
    
@app.get("/info/{version}/RxF")
async def getRxF(version:str):
        RxF_path = os.path.join(models_root_path,version,"RxF.png")
        return FileResponse(RxF_path,media_type='application/octet-stream', filename="RxF.png")

@app.get("/info/{version}/FxM")
async def getFxM(version:str):
        FxM_path = os.path.join(models_root_path,version,"FxM.png")
        return FileResponse(FxM_path,media_type='application/octet-stream', filename="FxM.png")

@app.get("/info/{version}/RxM")
async def getRxM(version:str):
        RxM_path = os.path.join(models_root_path,version,"RxM.png")
        return FileResponse(RxM_path,media_type='application/octet-stream', filename="RxM.png")