import pandas as pd


import kagglehub
from kagglehub import KaggleDatasetAdapter
from typing import Any


version = 2
models_root_path="~/models"
data_root_path="~/data"

def load_data()->pd.DataFrame:
    file = kagglehub.dataset_download("carrie1/ecommerce-data",path="data.csv")
    df = pd.read_csv(file, encoding="ISO-8859-1")
    return df



def rfm_processe(rfm:pd.DataFrame,scaler:Any,encoder:Any)->pd.DataFrame:
    rfm_features = rfm[['Recency', 'Frequency', 'Monetary']]
    rfm_scaled = scaler.fit_transform(rfm_features)
    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])
    return rfm_scaled_df




def data_processe(data:pd.DataFrame,scaler:object,encoder:object)->pd.DataFrame:
    data_cleaned = data.dropna(subset=['CustomerID'])
    data_cleaned = data_cleaned[(data_cleaned['Quantity'] > 0) & (data_cleaned['UnitPrice'] > 0)]
    data_cleaned = data_cleaned[~data_cleaned['InvoiceNo'].astype(str).str.startswith('C')]
    data_cleaned.reset_index(drop=True, inplace=True)
    data_cleaned['InvoiceDate'] = pd.to_datetime(data_cleaned['InvoiceDate'])
    data_cleaned['TotalPrice'] = data_cleaned['Quantity'] * data_cleaned['UnitPrice']
    
    return data_cleaned





def data_to_rfm(data:pd.DataFrame)->pd.DataFrame:
    reference_date = data['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,   # Recency
    'InvoiceNo': 'nunique',                                     # Frequency
    'TotalPrice': 'sum'                                         # Monetary
        }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    return rfm






from sklearn.preprocessing import StandardScaler,LabelEncoder
scaler = StandardScaler()
encoder = LabelEncoder()



from sklearn.cluster import AffinityPropagation

model = AffinityPropagation(random_state=42)




