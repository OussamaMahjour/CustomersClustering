import joblib
import pandas as pd
from preprocessor import Preprocessor
from config import models_root_path,data_root_path
    
    
    
    
def load_model(version:int):
    model = joblib.load(f"{models_root_path}/v{version}/model/model.pkl")
    return model

def load_rfm(version:int):
    rfm = pd.read_csv(f"{data_root_path}/v{version}/rfm/rfm.csv")
    rfm_processor = joblib.load(f"{data_root_path}/v{version}/rfm/rfm_processor.pkl")
    return (rfm,rfm_processor)

def load_data(version:int):
    data = pd.read_csv(f"{data_root_path}/v{version}/data/data.csv")
    data_processor = joblib.load(f"{data_root_path}/v{version}/data/data_processor.pkl")
    return (data,data_processor)



    



def predict(version:int,customerId:str):
    model = load_model(version)
    print(model)
    rfm,rfm_processor = load_rfm(version=version)
    print(rfm.head(5))
    print(customerId)
    customer = rfm[rfm["CustomerID"]==float(customerId)]
    print(customer)
    processed_customer = rfm_processor.process(customer)
    print(processed_customer)
    cluster =  model.predict(processed_customer)
    print(cluster)
    return cluster



    