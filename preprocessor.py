from sklearn.preprocessing import StandardScaler, LabelEncoder
from pandas import DataFrame
import pandas as pd
import joblib
from typing import Callable




class Preprocessor:
    
    def __init__(self, 
                 scaler:object | None=None,
                 encoder:object | None=None,
                 load:bool=False,
                 strategy:Callable[[DataFrame,object,object],DataFrame] | None=None,
                 version:int=0):
        if not load:
            self.scaler = scaler if scaler is not None else StandardScaler()
            self.encoder = encoder if encoder is not None else LabelEncoder()
            self.strategy = strategy
        else:
            self = joblib.load('./models/v{version}/Preprocessor.pkl')
    
    def process(
        self,
        data:DataFrame,
    ):
        if self.strategy is not None:
            return self.strategy(data,self.scaler,self.encoder)
        else :raise Exception("Strategy Not defined pleas define a strategy first")
        
        
    
    
   
        