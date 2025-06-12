import streamlit as st
from config import model_server,data_server
import os
import requests


def predict(model_version:str):
    customer = st.text_input("Customer")
    predictButton = st.button("Predict")
    params = requests.get(f"{data_server}/{model_version}/rfm?customerId={customer}").json()[0]
    print(params)
    if predictButton:
        prams = {
            "Recency":params["Recency"],
            "Frequency":params["Frequency"],
            "Monetary":params["Monetary"]
        }
        cluster = requests.get(f"{model_server}/{model_version}/predict",params=prams).json()
        st.success(f"The Custumer is {cluster}")



def order(data_version:str):
    st.text("Make an Order")
    customerId = st.text_input("Custumer Id")
    InvoiceId = st.text_input("Invoice Id")
    InvoiceDate = st.date_input("Invoice Date")
    Country = st.text_input("Country")
    Quantity = st.number_input("Quantity",value=0,min_value=0)
    UnitPrice = st.number_input("UnitPrice",value=0,min_value=0)
    if st.button("Submit Order"):
        body = {
            "CustomerID": int(customerId),   
            "InvoiceDate": InvoiceDate.isoformat(),  
            "Quantity": Quantity,
            "UnitPrice": UnitPrice
        }
        response = requests.put(f"{data_server}/{data_version}/rfm", json=body)
        
        if response.ok:
            st.success("Order submitted successfully!")
        else:
            st.error(f"Failed to submit order: {response.text}")
    
    

def info(model_version:str):
    pass



if __name__ == "__main__":
    st.title("Customers Clustering")
    models = [name for name in os.listdir("./models") if os.path.isdir(os.path.join("./models", name))]
    modelSelection =st.selectbox("Model Version",models,accept_new_options=False)
    
    selection =st.segmented_control("", options=["Predict","Order", "Info"],selection_mode="single",default="Predict")
    
    version = modelSelection
    
    if selection == "Predict":
        predict(version)
    elif selection == "Order":
        order(version)
    elif selection == "Info":
        info(version)
    

    