import pandas as pd
import numpy as np
from joblib import load
import tensorflow as tf
from operator import itemgetter
import yaml


with open('cbm_yaml.yml','r') as file:
    utility_dict = yaml.safe_load(file)   


class pdm_AE_model():
    #add PR and rest
    global utility_dict
    def __init__(self):
        self.ae_scaler = load(utility_dict['AE_scaler_path'])
        self.ae_model = tf.keras.models.load_model(utility_dict['AE_model_path'])
        self.threshold = utility_dict['AE_threshold']
    def AE(self,new_data): #index should be signaldate
        indx_log = list(new_data.index)
        new_data_scaled = pd.DataFrame(self.ae_scaler.transform(new_data),columns=self.ae_scaler.feature_names_in_) #give feature names from scaler
        new_data_pred = self.ae_scaler.inverse_transform(self.ae_model.predict(new_data_scaled))
        new_data_mae = tf.keras.losses.mean_squared_error(new_data,new_data_pred)
        healthy_indx = np.where(new_data_mae.numpy()<self.threshold)[0]
        print('healthy_indx',healthy_indx)
        new_data_pred_df = pd.DataFrame(new_data_pred,columns=new_data.columns)
        new_data_pred_df = new_data_pred_df.loc[healthy_indx]
        # new_time_indx = list(itemgetter(*list(new_data_pred_df.index))(indx_log))
        getter = itemgetter(*list(new_data_pred_df.index))
        new_time_indx = list(getter(indx_log))
        # new_data_pred_df.set_index(new_time_indx,inplace=True,drop=True)
        new_data_pred_df.index = new_time_indx
        new_data_pred_df.index = pd.to_datetime(new_data_pred_df.index)
        return new_data_pred_df, np.where(new_data_mae.numpy()>=self.threshold)[0]