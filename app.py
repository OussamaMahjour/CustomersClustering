import streamlit as st
from Invoice import Invoice
from preprocessor import Preprocessor
import os
import tools

def process(invoice:Invoice,preprocessor:Preprocessor):
    return preprocessor.process(data=invoice)


def predict(model_version:int):
    customer = st.text_input("Customer")
    predictButton = st.button("Predict")
    if predictButton:
        st.success(f"The Custumer is {tools.predict(version=model_version,customerId=customer)}")

def order(model_version:int):
    st.text("Make an Order")
    customerId = st.text_input("Custumer Id")
    InvoiceId = st.text_input("Invoice Id")
    InvoiceDate = st.date_input("Invoice Date")
    Country = st.text_input("Country")
    Quantity = st.number_input("Quantity",value=0,min_value=0)
    UnitPrice = st.number_input("UnitPrice",value=0,min_value=0)
    
    

def info(model_version:int):
    pass



if __name__ == "__main__":
    st.title("Customers Clustering")
    models = [name for name in os.listdir("./models") if os.path.isdir(os.path.join("./models", name))]
    modelSelection =st.selectbox("Model Version",models,accept_new_options=False)
    
    selection =st.segmented_control("", options=["Predict","Order", "Info"],selection_mode="single",default="Predict")
    
    version = int(modelSelection.split("v")[1])
    
    if selection == "Predict":
        predict(version)
    elif selection == "Order":
        order(version)
    elif selection == "Info":
        info(version)
    

    