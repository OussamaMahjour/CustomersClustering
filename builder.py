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
    
    
    print("Loading Data ...")
    data = load_data()
    
    
    print("Processing Data ...")
    data_processed = data_processor.process(data)
    
    
    print("Saving Data ...")
    os.makedirs(data_path, exist_ok=True)
    joblib.dump(data_processor,data_path+"data_processor.pkl")
    data.to_csv(data_path+"data.csv",index=False)
    
    
    print("Converting Data to RFM ...")
    rfm = data_to_rfm(data=data_processed)
    
    
    print("Processing RFM ...")
    rfm_processed = rfm_processor.process(rfm)
    
    
    print("Saving RFM ...")
    os.makedirs(rfm_path, exist_ok=True)
    joblib.dump(rfm_processor,rfm_path+"rfm_processor.pkl")
    rfm.to_csv(rfm_path+"rfm.csv",index=False)
    
    
    print("Training the model ...")
    model.fit(rfm_processed.to_numpy())
    
    
    
    print("Saving Model ...")
    
    
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(model,model_path+"model.pkl")
    
    
    
    
    