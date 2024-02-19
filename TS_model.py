
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# from tensorflow.keras import Model
# from tensorflow.keras.layers import Dense, LSTM, Reshape
# from joblib import load
# import time

# def Build_model(n_steps_in,n_steps_out,X_feat,Y_feat):
#     class AttentionLayer(tf.keras.layers.Layer):
#             def __init__(self,name=None):
#                 super(AttentionLayer, self).__init__(name=name)

#             def build(self, input_shape):
#                 self.W = self.add_weight(shape=(input_shape[-1], 1), initializer="random_normal", trainable=True)
#                 super(AttentionLayer, self).build(input_shape)

#             def call(self, inputs):
#                 attention_weights = tf.nn.softmax(tf.matmul(inputs, self.W), axis=1)
#                 weighted_inputs = inputs * attention_weights
#                 return tf.reduce_sum(weighted_inputs, axis=1)

#     inputs = tf.keras.Input(shape=(n_steps_in, X_feat))
#     x = LSTM(335, return_sequences=True)(inputs)
#     x = LSTM(188, return_sequences=True, dropout=.06586127019804049)(x)
#     x = LSTM(172, return_sequences=True, dropout=.1494075127203595)(x) 
#     x = AttentionLayer(name='AttentionLayer')(x)    

#     out = Dense(64)(x)
#     out = Dense(128)(out)
#     out = Dense(64)(out)

#     out = Dense(n_steps_out*Y_feat)(out)
#     out = Reshape((n_steps_out,Y_feat))(out)   
#     model = Model(inputs = inputs, outputs = out)
#     return model

# class pdm_ts_model():
 
#     cyl_count = 6 #may change depending upon vessel
#     max_load = 84.19076333333332 #max engine load from 1 yr data for normalization
#     look_back = 336
#     forecast_horizon = 336

#     def __init__(self,ts_features_file,ts_model,x_scale,y_scale,engine_normalized:bool,engine_number,res_loc): #raw_data,anomaly_path,Efd_features,feature,input_data
#         self.ts_features_file = ts_features_file #input features list for TS
#         self.ts_model_path = ts_model #saved ts model with path
#         # self.Efd_features = Efd_features #list of all EFD features for iter on ml models ['Pscav','Pcomp','Pmax','Texh','Ntc','Ntc/Pscav','Pcomp/Pscav','PR']
#         self.scaler_inp_train_m = load(x_scale) #saved inputs scaler model with path
#         self.scaler_out_train_m = load(y_scale) #saved output scaler model with path
#         self.engine_normalized = engine_normalized #True or False for applying engine based normalization
#         self.engine_number = engine_number # '1' or '2' for corr engine
#         self.res_loc = res_loc #loc where TS results save 'E:/python_codes1/ML_reference/1hr/TS_models_engineinout/TS_res/'
#         #loading model
#         self.ts_model = Build_model(self.look_back,self.forecast_horizon,52,9)
#         self.ts_model.load_weights(self.ts_model_path)
    
#     def Timeseries(self, new_data):
#         t1 = time.time()
#         #part for test data
#         df = new_data 
#         df = df[(df['Estimated engine load']>=30)&(df['Estimated engine load']<=100)]    
#         df['PR'] = df['Firing Pressure Average'] - df['Compression Pressure Average']
#         df['Ntc_Pscav'] = df['Turbocharger 1 speed'] / df['Scav. Air Press. Mean Value']
#         df['Pcomp_Pscav'] = df['Compression Pressure Average'] / df['Scav. Air Press. Mean Value']
#         #Normalizing based on engine load
#         if self.engine_normalized == True:
#             max_load = self.max_load
#             print(max_load)
#             df['Estimated engine load'] = df['Estimated engine load']/max_load
#             for col in df.columns:
#                 if col == 'Estimated engine load':
#                     df[col] = df['Estimated engine load']
#                 else:
#                     df[col] = df[col]*df['Estimated engine load']     
                
#         # df = df.iloc[-self.look_back:,] #taking last look_back period from new dataset
#         print(df.index.min())
#         print(df.index.max())
#         print(df.shape)

        
#         #predicting for individual cylinders
#         ts_res_cyl = {}
#         df_pred_load_wise = pd.DataFrame()
#         for cyl in range(1,self.cyl_count+1):
#             df['Exh. valve opening angle Cyl AVG'] = df['Exh. valve opening angle Cyl #0'+str(cyl)]
#             df['GAV Timing Set Point Cyl AVG'] = df['GAV Timing Set Point Cyl #0'+str(cyl)]
#             df['Exhaust Valve Closing Angle Setpoint Cyl AVG'] = df['Exhaust Valve Closing Angle Setpoint Cyl #0'+str(cyl)]
#             df['PFI Timing Set Point Cyl AVG'] = df['PFI Timing Set Point Cyl #0'+str(cyl)]
#             df['PFI Duration Set Point Cyl AVG'] = df['PFI Duration Set Point Cyl #0'+str(cyl)]
#             df['Cyl. lub. distribution share below_PERC'] = (df['Cyl. lub. distribution share below piston']/df['Cyl. lub. distribution share into piston'])*100
#             df['Cyl. lub. distribution share above_PERC'] = (df['Cyl. lub. distribution share above piston']/df['Cyl. lub. distribution share into piston'])*100
#             df['Fuel Rail Pressure_diff'] = df['Mean Fuel Rail Pressure (display)'] - df['Main Fuel Rail Pressure']
#             df['Firing Pr. Balancing Injection Offset Cyl_AVG'] = df['Firing Pr. Balancing Injection Offset Cyl #0'+str(cyl)]
#             df['Fuel Pressure Actuator Setpoint_AVG'] = (df['Fuel Pressure Actuator Setpoint 1']+df['Fuel Pressure Actuator Setpoint 2']+df['Fuel Pressure Actuator Setpoint 3'])/3
#             df['Fuel Pump Setpoint_AVG'] = (df['Fuel Pump Setpoint Master Controller']+df['Fuel Pump Setpoint Slave Controller'])/2
#             df['Lubrication Oil Feed Rate Cyl AVG'] = df['Lubrication Oil Feed Rate Cyl #0'+str(cyl)]
#             df['Lubrication Deadtime Feedback Cyl AVG'] = df['Lubrication Deadtime Feedback Cyl #0'+str(cyl)]
#             df['Start of Injection Cyl_AVG'] = df['Start of Injection Cyl #0'+str(cyl)]
#             df['Pilot Fuel Pressure diff'] = df['Pilot Fuel Pressure A']-df['Pilot Fuel Pressure B']
#             df['Scavenge Air Temp. Piston Underside Cyl_AVG'] = df['Scavenge Air Temp. Piston Underside Cyl #0'+str(cyl)+'.1']
#             #scaling X
#             model_inputs = df[list(self.scaler_inp_train_m.feature_names_in_)]
#             scaled_inp_test = self.scaler_inp_train_m.transform(model_inputs)
#             pos = np.array(range(1,self.look_back+1))/self.look_back
#             scaled_inp_test = np.append(scaled_inp_test,pos.reshape(-1,1),axis=1)
 
#             X_new = scaled_inp_test.reshape((1, scaled_inp_test.shape[0], scaled_inp_test.shape[1]))
#             predictons = self.ts_model.predict(X_new)
#             #inverse scaling y
#             y_pred_real = self.scaler_out_train_m.inverse_transform(predictons[-1].reshape(-1,len(self.scaler_out_train_m.feature_names_in_)))
#             df_pred = pd.DataFrame(y_pred_real, columns=['TS_Pcomp', 'TS_Pscav','TS_Texh', 'TS_Ntc','TS_Pmax', 'TS_PR','TS_Ntc_Pscav','TS_Pcomp_Pscav','Estimated engine load'])
#             df_pred.to_csv(self.res_loc+'ENG_{}_TS_res_Cyl_{}.csv'.format(self.engine_number,cyl), index=False)
#             #print('Cylinder_'+str(cyl)+' timeseries prediction completed!!!')

#         t2 = time.time()
#         print('Time taken for TS prediction: ',t2-t1)
#         return df.index.max()


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Reshape
from joblib import load
import time
from datetime import datetime
import yaml

from azure.storage.blob import ContainerClient
from io import StringIO

with open('cbm_yaml.yml','r') as file:
    utility_dict = yaml.safe_load(file)
    
# def Build_model(n_steps_in,n_steps_out,X_feat,Y_feat):
#     class AttentionLayer(tf.keras.layers.Layer):
#             def __init__(self,name=None):
#                 super(AttentionLayer, self).__init__(name=name)

#             def build(self, input_shape):
#                 self.W = self.add_weight(shape=(input_shape[-1], 1), initializer="random_normal", trainable=True)
#                 super(AttentionLayer, self).build(input_shape)

#             def call(self, inputs):
#                 attention_weights = tf.nn.softmax(tf.matmul(inputs, self.W), axis=1)
#                 weighted_inputs = inputs * attention_weights
#                 return tf.reduce_sum(weighted_inputs, axis=1)

#     inputs = tf.keras.Input(shape=(n_steps_in, X_feat))
#     x = LSTM(335, return_sequences=True)(inputs)
#     x = LSTM(188, return_sequences=True, dropout=.06586127019804049)(x)
#     x = LSTM(172, return_sequences=True, dropout=.1494075127203595)(x) 
#     x = AttentionLayer(name='AttentionLayer')(x)    

#     out = Dense(64)(x)
#     out = Dense(128)(out)
#     out = Dense(64)(out)

#     out = Dense(n_steps_out*Y_feat)(out)
#     out = Reshape((n_steps_out,Y_feat))(out)   
#     model = Model(inputs = inputs, outputs = out)
#     return model

# class pdm_ts_model():
#     global utility_dict
#     cyl_count = utility_dict['cyl_count'] #may change depending upon vessel
#     max_load = utility_dict['max_load'] #max engine load from 1 yr data for normalization
#     look_back = utility_dict['look_back']
#     forecast_horizon = utility_dict['forecast_horizon']

#     def __init__(self,ts_features_file,ts_model,x_scale,y_scale,engine_normalized:bool,engine_number,res_loc): #raw_data,anomaly_path,Efd_features,feature,input_data
#         self.ts_features_file = ts_features_file #input features list for TS
#         self.ts_model_path = ts_model #saved ts model with path
#         # self.Efd_features = Efd_features #list of all EFD features for iter on ml models ['Pscav','Pcomp','Pmax','Texh','Ntc','Ntc/Pscav','Pcomp/Pscav','PR']
#         self.scaler_inp_train_m = load(x_scale) #saved inputs scaler model with path
#         self.scaler_out_train_m = load(y_scale) #saved output scaler model with path
#         self.engine_normalized = engine_normalized #True or False for applying engine based normalization
#         self.engine_number = engine_number # '1' or '2' for corr engine
#         self.res_loc = res_loc #loc where TS results save 'E:/python_codes1/ML_reference/1hr/TS_models_engineinout/TS_res/'
#         #loading model
#         self.ts_model = Build_model(self.look_back,self.forecast_horizon,52,9)
#         self.ts_model.load_weights(self.ts_model_path)
    
#     def Timeseries(self, new_data):
#         t1 = time.time()
#         #part for test data
#         df = new_data 
#         df = df[(df[utility_dict['engine_load']]>=30)&(df[utility_dict['engine_load']]<=100)]    
#         # df['PR'] = df['Firing Pressure Average'] - df['Compression Pressure Average']
#         df[list(utility_dict['calc_efd_features'].keys())[1]] = df[utility_dict['calc_efd_features'][list(utility_dict['calc_efd_features'].keys())[1]][0]] / df[utility_dict['calc_efd_features'][list(utility_dict['calc_efd_features'].keys())[1]][1]]
#         # df['Pcomp_Pscav'] = df['Compression Pressure Average'] / df['Scav. Air Press. Mean Value']
#         #Normalizing based on engine load
#         if self.engine_normalized == True:
#             max_load = self.max_load
#             print(max_load)
#             df[utility_dict['engine_load']] = df[utility_dict['engine_load']]/max_load
#             for col in df.columns:
#                 if col == utility_dict['engine_load']:
#                     df[col] = df[utility_dict['engine_load']]
#                 else:
#                     df[col] = df[col]*df[utility_dict['engine_load']]     
                
#         # df = df.iloc[-self.look_back:,] #taking last look_back period from new dataset
#         print(df.index.min())
#         print(df.index.max())
#         print(df.shape)

        
#         #predicting for individual cylinders
#         ts_res_cyl = {}
#         df_pred_load_wise = pd.DataFrame()
#         for cyl in range(1,self.cyl_count+1):
#             df[list(utility_dict['calc_efd_features'].keys())[0]] = df[utility_dict['calc_efd_features'][list(utility_dict['calc_efd_features'].keys())[0]][0]+str(cyl)] - df[utility_dict['calc_efd_features'][list(utility_dict['calc_efd_features'].keys())[0]][1]+str(cyl)]
#             df[list(utility_dict['calc_efd_features'].keys())[2]] = df[utility_dict['calc_efd_features'][list(utility_dict['calc_efd_features'].keys())[2]][0]+str(cyl)] / df[utility_dict['calc_efd_features'][list(utility_dict['calc_efd_features'].keys())[2]][1]]
#             df[list(utility_dict['calc_features'].keys())[0]] = df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[0]]+str(cyl)]
#             df[list(utility_dict['calc_features'].keys())[1]] = df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[1]]+str(cyl)]
#             df[list(utility_dict['calc_features'].keys())[2]] = df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[2]]+str(cyl)]
#             df[list(utility_dict['calc_features'].keys())[3]] = df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[3]]+str(cyl)]
#             df[list(utility_dict['calc_features'].keys())[4]] = df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[4]]+str(cyl)]
#             df[list(utility_dict['calc_features'].keys())[5]] = (df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[5]][0]]/df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[5]][1]])*100
#             df[list(utility_dict['calc_features'].keys())[6]] = (df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[6]]]/df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[5]][1]])*100
#             df[list(utility_dict['calc_features'].keys())[7]] = df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[7]][0]] - df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[7]][1]]
#             df[list(utility_dict['calc_features'].keys())[8]] = df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[8]]+str(cyl)]
#             df[list(utility_dict['calc_features'].keys())[9]] = (df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[9]][0]]+df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[9]][1]]+df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[9]][2]])/3
#             df[list(utility_dict['calc_features'].keys())[10]] = (df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[10]][0]]+df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[10]][1]])/2
#             df[list(utility_dict['calc_features'].keys())[11]] = df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[11]]+str(cyl)]
#             df[list(utility_dict['calc_features'].keys())[12]] = df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[12]]+str(cyl)]
#             df[list(utility_dict['calc_features'].keys())[13]] = df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[13]]+str(cyl)]
#             df[list(utility_dict['calc_features'].keys())[14]] = df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[14]][0]]-df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[14]][1]]
#             df[list(utility_dict['calc_features'].keys())[15]] = df[utility_dict['calc_features'][list(utility_dict['calc_features'].keys())[15]]+str(cyl)+'.1']
#             #scaling X
#             model_inputs = df[list(self.scaler_inp_train_m.feature_names_in_)]
#             scaled_inp_test = self.scaler_inp_train_m.transform(model_inputs)
#             pos = np.array(range(1,self.look_back+1))/self.look_back
#             scaled_inp_test = np.append(scaled_inp_test,pos.reshape(-1,1),axis=1)
 
#             X_new = scaled_inp_test.reshape((1, scaled_inp_test.shape[0], scaled_inp_test.shape[1]))
#             predictons = self.ts_model.predict(X_new)
#             #inverse scaling y
#             y_pred_real = self.scaler_out_train_m.inverse_transform(predictons[-1].reshape(-1,len(self.scaler_out_train_m.feature_names_in_)))
#             df_pred = pd.DataFrame(y_pred_real, columns=utility_dict['TS_frame_colnames'])
#             df_pred.to_csv(self.res_loc+'ENG_{}_TS_res_Cyl_{}.csv'.format(self.engine_number,cyl), index=False)
#             #print('Cylinder_'+str(cyl)+' timeseries prediction completed!!!')

#         t2 = time.time()
#         print('Time taken for TS prediction: ',t2-t1)
#         return df.index.max()

class pdm_ts_model():
    global utility_dict
    cyl_count = utility_dict['cyl_count'] #may change depending upon vessel

    look_back = utility_dict['look_back']
    forecast_horizon = utility_dict['forecast_horizon']

    def __init__(self,ts_model,x_scale,y_scale,engine_number,res_loc): #raw_data,anomaly_path,Efd_features,feature,input_data
        self.ts_model_path = ts_model #saved ts model with path
        # self.Efd_features = Efd_features #list of all EFD features for iter on ml models ['Pscav','Pcomp','Pmax','Texh','Ntc','Ntc/Pscav','Pcomp/Pscav','PR']
        self.scaler_inp_train_m = load(x_scale) #saved inputs scaler model with path
        self.scaler_out_train_m = load(y_scale) #saved output scaler model with path
        self.engine_number = engine_number # '1' or '2' for corr engine
        self.res_loc = res_loc #loc where TS results save 'E:/python_codes1/ML_reference/1hr/TS_models_engineinout/TS_res/'
        #loading model
        self.ts_model = tf.keras.models.load_model(self.ts_model_path)
    
    def Timeseries(self, new_data,eng):
        container_client = ContainerClient.from_connection_string(
        utility_dict['connection_string'], container_name=utility_dict['container_name'])
        t1 = time.time()
        #part for test data
        df = new_data
        # df = new_data[self.ts_features_ts] 
        df = df[(df[utility_dict['engine_load']]>=30)&(df[utility_dict['engine_load']]<=100)]    
        df = df[:self.look_back]
        # df['PR'] = df['Firing Pressure Average'] - df['Compression Pressure Average']
        df[list(utility_dict['calc_efd_features'].keys())[1]] = df[utility_dict['calc_efd_features'][list(utility_dict['calc_efd_features'].keys())[1]][0]] / df[utility_dict['calc_efd_features'][list(utility_dict['calc_efd_features'].keys())[1]][1]]
        # df['Pcomp_Pscav'] = df['Compression Pressure Average'] / df['Scav. Air Press. Mean Value']   
                
        # df = df.iloc[-self.look_back:,] #taking last look_back period from new dataset
        print(df.index.min())
        print(df.index.max())
        print(df.shape)

        
        #predicting for individual cylinders
        ts_res_cyl = {}
        df_pred_load_wise = pd.DataFrame()
        for cyl in range(1,self.cyl_count+1):
            df[list(utility_dict['calc_efd_features'].keys())[0]] = df[utility_dict['calc_efd_features'][list(utility_dict['calc_efd_features'].keys())[0]][0]+str(cyl)] - df[utility_dict['calc_efd_features'][list(utility_dict['calc_efd_features'].keys())[0]][1]+str(cyl)]
            df[list(utility_dict['calc_efd_features'].keys())[2]] = df[utility_dict['calc_efd_features'][list(utility_dict['calc_efd_features'].keys())[2]][0]+str(cyl)] / df[utility_dict['calc_efd_features'][list(utility_dict['calc_efd_features'].keys())[2]][1]]
            #Exh. valve opening angle Cyl
            df[utility_dict['add_cols_exv']['feat'][0]] = df[utility_dict['add_cols_exv']['base']+str(cyl)]
            df[utility_dict['add_cols_exv']['feat'][1]] = df[utility_dict['add_cols_exv']['base']+str(cyl)]
            df[utility_dict['add_cols_exv']['feat'][2]] = df[utility_dict['add_cols_exv']['base']+str(cyl)]
            #Firing Pr. Balancing Injection Offset Cyl
            df[utility_dict['add_cols_frp']['feat'][0]] = df[utility_dict['add_cols_frp']['base']+str(cyl)]
            df[utility_dict['add_cols_frp']['feat'][1]] = df[utility_dict['add_cols_frp']['base']+str(cyl)]
            df[utility_dict['add_cols_frp']['feat'][2]] = df[utility_dict['add_cols_frp']['base']+str(cyl)]
            #Start of Injection Cyl
            df[utility_dict['add_cols_strtinj']['feat'][0]] = df[utility_dict['add_cols_strtinj']['base']+str(cyl)]
            df[utility_dict['add_cols_strtinj']['feat'][1]] = df[utility_dict['add_cols_strtinj']['base']+str(cyl)]
            df[utility_dict['add_cols_strtinj']['feat'][2]] = df[utility_dict['add_cols_strtinj']['base']+str(cyl)]
            #scaling X
            model_inputs = df[list(self.scaler_inp_train_m.feature_names_in_)]
            scaled_inp_test = self.scaler_inp_train_m.transform(model_inputs)
            pos = np.array(range(1,self.look_back+1))/self.look_back
            print('ts shape -',scaled_inp_test.shape)
            scaled_inp_test = np.append(scaled_inp_test,pos.reshape(-1,1),axis=1)
   
            X_new = scaled_inp_test.reshape((1, scaled_inp_test.shape[0], scaled_inp_test.shape[1]))
            
            print('X_new -',X_new.shape)
            predictons = self.ts_model.predict([X_new,tf.zeros((1, 336, 9))])
            #inverse scaling y
            y_pred_real = self.scaler_out_train_m.inverse_transform(predictons[-1].reshape(-1,len(self.scaler_out_train_m.feature_names_in_)))
            df_pred = pd.DataFrame(y_pred_real, columns=utility_dict['TS_frame_colnames'])
            df_pred_csv = df_pred.to_csv(index=False)
            df_pred_csv_bytes = bytes(df_pred_csv, 'utf-8')
            df_pred_csv_stream = StringIO(df_pred_csv)
            container_client.upload_blob(name="Data/"+utility_dict['Vessel_name']+'/Results/TS/'+utility_dict['Vessel_name']+'_ENG_{}_TS_res_Cyl_{}_{}.csv'.format(str(eng),cyl,str(datetime.now()).split(' ')[0]), data=df_pred_csv_bytes,overwrite=True)
            df_pred_csv_stream.close()
            # df_pred.to_csv(self.res_loc+utility_dict['Vessel_name']+'_ENG_{}_TS_res_Cyl_{}_{}.csv'.format(str(eng),cyl,str(datetime.now()).split(' ')[0]), index=False)
            #print('Cylinder_'+str(cyl)+' timeseries prediction completed!!!')

        t2 = time.time()
        print('Time taken for TS prediction: ',t2-t1)
        return df.index.max()


