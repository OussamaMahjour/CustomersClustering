from fastapi import FastAPI,File,UploadFile
import pandas as pd
import os
from pydantic import BaseModel
from config import data_root_path
from datetime import datetime
app = FastAPI()
from typing import Annotated
import shutil


chunksize = 100000

@app.get("/{version}/rfm")
async def getCustomerRFM(version:str,customerId:int):
    
    path = os.path.join(data_root_path,version,"rfm.csv")
    df = pd.read_csv(path,encoding="ISO-8859-1",chunksize=chunksize)
    for chunk in df:
        match = chunk[chunk['CustomerID']==customerId]
        if not match.empty: 
            return match.to_dict(orient="records")
    return {"message": "Customer not found"}



class Invoice(BaseModel):
    CustomerID: int
    InvoiceDate: datetime
    Quantity: int
    UnitPrice: float

@app.put("/{version}/rfm")
async def updateCustomerRFM(version:str,invoice:Invoice):
    path = os.path.join(data_root_path,version,"rfm.csv")
    df = pd.read_csv(path,encoding="ISO-8859-1",chunksize=chunksize)
    temp_path = os.path.join(data_root_path, version, "rfm_temp.csv")
    reference_date = datetime.today()
    recency_days = (reference_date - invoice.InvoiceDate).days
    new_rfm = pd.DataFrame([{
        "CustomerID": invoice.CustomerID,
        "Recency": recency_days,  # just made purchase
        "Frequency": 1,
        "Monetary": invoice.Quantity * invoice.UnitPrice
    }])
    updated = False
    with open(temp_path,"w",encoding="ISO-8859-1",newline='') as temp_file:
        
        for chunk in pd.read_csv(path,encoding="ISO-8859-1",chunksize=chunksize):
            if invoice.CustomerID in chunk["CustomerID"].values:
                mask = chunk["CustomerID"] == invoice.CustomerID
                chunk.loc[mask, "Recency"] = recency_days  
                chunk.loc[mask, "Frequency"] += 1
                chunk.loc[mask, "Monetary"] += (invoice.Quantity * invoice.UnitPrice)
                updated = True
            last_chunk = chunk
            chunk.to_csv(temp_file,index=False,header=temp_file.tell() == 0)
        
        if not updated: 
            print("inserting")
            new_rfm.to_csv(temp_file,mode="a",header=False,index=False)

    os.replace(temp_path, path)
    return {
        "status": "updated" if updated else "inserted",
        "CustomerID": invoice.CustomerID,
        "RFM": new_rfm.to_dict(orient="records")
    }

@app.post("/save/{version}")
async def save(version:str,data:Annotated[UploadFile,File()]):
    data_path = os.path.join(data_root_path,version,"rfm.csv")
    version_path = os.path.join(data_root_path, version)
    os.makedirs(version_path, exist_ok=True)
    with open(data_path,"wb") as buffer:
        shutil.copyfileobj(data.file,buffer) 
        
        
    return f"Save data on version {version}"

