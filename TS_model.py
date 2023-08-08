
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from joblib import load

class pdm_ts_model():
 
    cyl_count = 6 #may change depending upon vessel
    max_load = 84.19076333333332 #max engine load from 1 yr data for normalization
    # look_back = 2880 #30 days
    # forecast_horizon = 1440 #15 days
    look_back = 336
    forecast_horizon = 336

    def __init__(self,new_data,ts_features_file,ts_model,x_scale,y_scale,engine_normalized:bool,engine_number,res_loc): #raw_data,anomaly_path,Efd_features,feature,input_data
        self.new_data = new_data #new data after agg & imputaion
        self.ts_features_file = ts_features_file #input features list for TS
        self.ts_model = tf.keras.models.load_model(ts_model, compile=False) #saved ts model with path
        # self.Efd_features = Efd_features #list of all EFD features for iter on ml models ['Pscav','Pcomp','Pmax','Texh','Ntc','Ntc/Pscav','Pcomp/Pscav','PR']
        self.scaler_inp_train_m = load(x_scale) #saved inputs scaler model with path
        self.scaler_out_train_m = load(y_scale) #saved output scaler model with path
        self.engine_normalized = engine_normalized #True or False for applying engine based normalization
        self.engine_number = engine_number # '1' or '2' for corr engine
        self.res_loc = res_loc #loc where TS results save 'E:/python_codes1/ML_reference/1hr/TS_models_engineinout/TS_res/'
    
    def Timeseries(self):
        #part for test data
        # self.feature = 'EFD'
        df = self.new_data 
        print('bfr', df.shape)
        df = df[(df['Estimated engine load']>=30)&(df['Estimated engine load']<=100)]    
        print('aft', df.shape)
        #Normalizing based on engine load
        if self.engine_normalized == True:
            max_load = self.max_load
            print(max_load)
            df['Estimated engine load'] = df['Estimated engine load']/max_load
            for col in df.columns:
                if col == 'Estimated engine load':
                    df[col] = df['Estimated engine load']
                else:
                    df[col] = df[col]*df['Estimated engine load']     
                
        df = df.iloc[-self.look_back:,] #taking last look_back period from new dataset
        print(df.index.min())
        print(df.index.max())
        print(df.shape)


        #setting input features for TS
        # fg = self.ts_features_file
        # all_feat_list = list(fg['a'])
        # add_cols = all_feat_list #this contains all input features list
        #predicting for individual cylinders
        ts_res_cyl = {}
        df_pred_load_wise = pd.DataFrame()
        for cyl in range(1,self.cyl_count+1):
            df['Exh. valve opening angle Cyl AVG'] = df['Exh. valve opening angle Cyl #0'+str(cyl)]
            df['GAV Timing Set Point Cyl AVG'] = df['GAV Timing Set Point Cyl #0'+str(cyl)]
            df['Exhaust Valve Closing Angle Setpoint Cyl AVG'] = df['Exhaust Valve Closing Angle Setpoint Cyl #0'+str(cyl)]
            df['PFI Timing Set Point Cyl AVG'] = df['PFI Timing Set Point Cyl #0'+str(cyl)]
            df['PFI Duration Set Point Cyl AVG'] = df['PFI Duration Set Point Cyl #0'+str(cyl)]
            df['Cyl. lub. distribution share below_PERC'] = (df['Cyl. lub. distribution share below piston']/df['Cyl. lub. distribution share into piston'])*100
            df['Cyl. lub. distribution share above_PERC'] = (df['Cyl. lub. distribution share above piston']/df['Cyl. lub. distribution share into piston'])*100
            df['Fuel Rail Pressure_diff'] = df['Mean Fuel Rail Pressure (display)'] - df['Main Fuel Rail Pressure']
            df['Firing Pr. Balancing Injection Offset Cyl_AVG'] = df['Firing Pr. Balancing Injection Offset Cyl #0'+str(cyl)]
            df['Fuel Pressure Actuator Setpoint_AVG'] = (df['Fuel Pressure Actuator Setpoint 1']+df['Fuel Pressure Actuator Setpoint 2']+df['Fuel Pressure Actuator Setpoint 3'])/3
            df['Fuel Pump Setpoint_AVG'] = (df['Fuel Pump Setpoint Master Controller']+df['Fuel Pump Setpoint Slave Controller'])/2
            df['Lubrication Oil Feed Rate Cyl AVG'] = df['Lubrication Oil Feed Rate Cyl #0'+str(cyl)]
            df['Lubrication Deadtime Feedback Cyl AVG'] = df['Lubrication Deadtime Feedback Cyl #0'+str(cyl)]
            df['Start of Injection Cyl_AVG'] = df['Start of Injection Cyl #0'+str(cyl)]
            df['Pilot Fuel Pressure diff'] = df['Pilot Fuel Pressure A']-df['Pilot Fuel Pressure B']
            df['Scavenge Air Temp. Piston Underside Cyl_AVG'] = df['Scavenge Air Temp. Piston Underside Cyl #0'+str(cyl)+'.1']
            #scaling X
            model_inputs = df[list(self.scaler_inp_train_m.feature_names_in_)]
            scaled_inp_test = self.scaler_inp_train_m.transform(model_inputs)
 
            X_new = scaled_inp_test.reshape((1, scaled_inp_test.shape[0], scaled_inp_test.shape[1]))
            predictons = self.ts_model.predict(X_new)
            #inverse scaling y
            # scaled_out_train = load(self.scaler_out_train_m)#load y model
            y_pred_real = self.scaler_out_train_m.inverse_transform(predictons[-1].reshape(-1,len(self.scaler_out_train_m.feature_names_in_)))
            df_pred = pd.DataFrame(y_pred_real, columns=['TS_Pcomp', 'TS_Pscav','TS_Texh', 'TS_Ntc','TS_Pmax', 'TS_PR','TS_Ntc_Pscav','TS_Pcomp_Pscav','Estimated engine load'])
            # ts_res_cyl['Cylinder_'+str(cyl)+'_ts_res'] = df_pred
            df_pred.to_csv(self.res_loc+'ENG_{}_TS_res_Cyl_{}.csv'.format(self.engine_number,cyl), index=False)
            print('Cylinder_'+str(cyl)+' timeseries prediction completed!!!')
        # return ts_res_cyl  

        return df.index.max()