from config import scaler,encoder
from config import rfm_processe,data_processe,load_data,data_to_rfm
from config import model,version
from config import models_root_path,data_root_path
from preprocessor import Preprocessor
import joblib
import os


rfm_processor = Preprocessor(
    scaler=scaler,
    encoder=encoder,
    strategy=rfm_processe
)

data_processor = Preprocessor(
    scaler=scaler,
    encoder=encoder,
    strategy=data_processe
)


if __name__ == "__main__":
    
    model_path = f"{models_root_path}/v{version}/model/"
    data_path = f"{data_root_path}/v{version}/data/"
    rfm_path=f"{data_root_path}/v{version}/rfm/"
    
    os.system('cls||clear')
    print("Loading Data ...")
    data = load_data()
    
    os.system('cls||clear')
    print("Processing Data ...")
    data_processed = data_processor.process(data)
    
    os.system('cls||clear')
    print("Saving Data ...")
    os.makedirs(data_path, exist_ok=True)
    joblib.dump(data_processor,data_path+"data_processor.pkl")
    data.to_csv(data_path+"data.csv",index=False)
    
    os.system('cls||clear')
    print("Converting Data to RFM ...")
    rfm = data_to_rfm(data=data_processed)
    
    os.system('cls||clear')
    print("Processing RFM ...")
    rfm_processed = rfm_processor.process(rfm)
    
    os.system('cls||clear')
    print("Saving RFM ...")
    os.makedirs(rfm_path, exist_ok=True)
    joblib.dump(rfm_processor,rfm_path+"rfm_processor.pkl")
    rfm.to_csv(rfm_path+"rfm.csv",index=False)
    
    os.system('cls||clear')
    print("Training the model ...")
    model.fit(rfm_processed.to_numpy())
    
    
    os.system('cls||clear')
    print("Saving Model ...")
    
    
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(model,model_path+"model.pkl")
    
    
    
    
    