# import pandas as pd
# import numpy as np
# from joblib import load
# import tensorflow as tf
# import os
# import time



# class pdm_ml_model():
#     load_limit = 10 #in %
    
#     max_load = 84.19076333333332 #max engine load from 1 yr data for normalization
#     cyl_count = 6
#     utility_dict = {'Pscav':{'Limits': {'L_limit': -10, 'U_limit': 10}, 'imp_feature': list(pd.read_csv(('./utils/ML_model/imp_features/Pscav.csv') )['Features'][:24])},
#                     'Pcomp':{'Limits': {'L_limit': -8, 'U_limit': 8}, 'imp_feature': list(pd.read_csv('./utils/ML_model/imp_features/Pcomp.csv')['Features'][:26])},
#                     'Pmax':{'Limits': {'L_limit': -8, 'U_limit': 8}, 'imp_feature': list(pd.read_csv('./utils/ML_model/imp_features/Pmax.csv')['Features'][:27])},
#                     'Texh':{'Limits': {'L_limit': -8, 'U_limit': 8}, 'imp_feature': list(pd.read_csv('./utils/ML_model/imp_features/Texh.csv')['Features'][:24])},
#                     'Ntc':{'Limits': {'L_limit': -8, 'U_limit': 8}, 'imp_feature': list(pd.read_csv('./utils/ML_model/imp_features/Ntc.csv')['Features'][:23])},
#                     'Ntc_Pscav':{'Limits': {'L_limit': -8, 'U_limit': 8}, 'imp_feature': list(pd.read_csv('./utils/ML_model/imp_features/Ntc_Pscav.csv')['Features'][:24])},
#                     'Pcomp_Pscav':{'Limits': {'L_limit': -10, 'U_limit': 10}, 'imp_feature': list(pd.read_csv('./utils/ML_model/imp_features/Pcomp_Pscav.csv')['Features'][:24])},
#                     'PR':{'Limits': {'L_limit': -15, 'U_limit': 15}, 'imp_feature': list(pd.read_csv('./utils/ML_model/imp_features/PR.csv')['Features'][:26])}}

#     def __init__(self,Efd_features,engine_normalized:bool,ts_res,engine_number,ml_res): #raw_data,anomaly_path,Efd_features,feature,input_data
#         #raw_data - raw data path + filename
#         #anomaly_path - anomaly_path + filename
#         self.engine_normalized = engine_normalized #True or False for applying engine based normalization
#         self.Efd_features = Efd_features #list of all EFD features for iter on ml models ['Pscav','Pcomp','Pmax','Texh','Ntc','Ntc_Pscav','Pcomp_Pscav','PR']
#         self.ts_res = ts_res # result from timeseries class which is a dicitionary type - this will be a path of TS results
#         self.engine_number = engine_number
#         self.ml_res = ml_res

#         #scaling models & ml models loading.......
#         self.Pcomp_Pscav_scaler_x = load("./utils/ML_model/features_scaler/"+'Pcomp_Pscav'+"_X.joblib") 
#         self.Pcomp_Pscav_scaler_y = load("./utils/ML_model/features_scaler/"+'Pcomp_Pscav'+"_Y.joblib") 
#         self.Pcomp_Pscav_ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+'Pcomp_Pscav'+'.keras')
#         self.PR_scaler_x = load("./utils/ML_model/features_scaler/"+'PR'+"_X.joblib")
#         self.PR_scaler_y = load("./utils/ML_model/features_scaler/"+'PR'+"_Y.joblib") 
#         self.PR_ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+'PR'+'.keras')
#         self.Ntc_Pscav_scaler_x = load("./utils/ML_model/features_scaler/"+'Ntc_Pscav'+"_X.joblib")
#         self.Ntc_Pscav_scaler_y = load("./utils/ML_model/features_scaler/"+'Ntc_Pscav'+"_Y.joblib") 
#         self.Ntc_Pscav_ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+'Ntc_Pscav'+'.keras')
#         self.Pmax_scaler_x = load("./utils/ML_model/features_scaler/"+'Pmax'+"_X.joblib")
#         self.Pmax_scaler_y = load("./utils/ML_model/features_scaler/"+'Pmax'+"_Y.joblib")
#         self.Pmax_ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+'Pmax'+'.keras')
#         self.Texh_scaler_x = load("./utils/ML_model/features_scaler/"+'Texh'+"_X.joblib")
#         self.Texh_scaler_y = load("./utils/ML_model/features_scaler/"+'Texh'+"_Y.joblib")
#         self.Texh_ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+'Texh'+'.keras')
#         self.Ntc_scaler_x = load("./utils/ML_model/features_scaler/"+'Ntc'+"_X.joblib")
#         self.Ntc_scaler_y = load("./utils/ML_model/features_scaler/"+'Ntc'+"_Y.joblib") 
#         self.Ntc_ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+'Ntc'+'.keras')
#         self.Pcomp_scaler_x = load("./utils/ML_model/features_scaler/"+'Pcomp'+"_X.joblib")
#         self.Pcomp_scaler_y = load("./utils/ML_model/features_scaler/"+'Pcomp'+"_Y.joblib")
#         self.Pcomp_ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+'Pcomp'+'.keras')
#         self.Pscav_scaler_x = load("./utils/ML_model/features_scaler/"+'Pscav'+"_X.joblib")
#         self.Pscav_scaler_y = load("./utils/ML_model/features_scaler/"+'Pscav'+"_Y.joblib")
#         self.Pscav_ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+'Pscav'+'.keras')

#     def ML_models(self, data):
#         #1)input data
#         #2)imp feature list for each variables
#         #3)add important feature files in parent loc
#         #4)store ml models in a separate folder named 'ML_models'
#         #5)store scaling models in a separate folder named 'Scaling_models' with '_X' extention for inputs and '_Y'extention for output
#         #define df here
#         tm26 = time.time()
#         df2 = data
#         df2 = df2[(df2['Estimated engine load']>=30)&(df2['Estimated engine load']<=100)]
#         if self.engine_normalized == True:
#             max_load = self.max_load
#             print(max_load)
#             df2['Estimated engine load'] = df2['Estimated engine load']/max_load
#             for col in df2.columns:
#                 if col == 'Estimated engine load':
#                     df2[col] = df2['Estimated engine load']
#                 else:
#                     df2[col] = df2[col]*df2['Estimated engine load']
#         ml_output_dict = {}
#         joblib_models = []
#         ann_models = ['Pcomp_Pscav','PR','Pmax','Ntc','Ntc_Pscav','Pcomp','Pscav','Texh']               
#         load_delta = {}  
        


        
#         for cyl in range(1,self.cyl_count+1):
#             cyl_df = pd.read_csv(self.ts_res+'ENG_{}_TS_res_Cyl_{}.csv'.format(self.engine_number,cyl),index_col=False)
#             load_ranges = list(cyl_df['Estimated engine load'].unique())
#             for loads in load_ranges:
#                 load_delta[loads] = abs(df2['Estimated engine load']-loads)  
#                 load_l_limit = loads*((100-self.load_limit)/100)
#                 load_u_limit = loads*((100+self.load_limit)/100)
#                 # load_cons1 = load_delta[loads][load_delta[loads]>=load_l_limit]
#                 load_cons1 = df2['Estimated engine load'][df2['Estimated engine load']>=load_l_limit]
#                 load_cons2 = load_cons1[load_cons1<=load_u_limit]
#                 ml_output_dict = {}
#                 if len(load_cons2)>0:
#                     load_cons2 = load_cons2.to_frame()
#                     load_cons2.columns = ['Matched engine load']
#                     load_cons2['load_delta'] = abs(load_cons2['Matched engine load']-loads)
#                     load_cons2.sort_values(by=['load_delta'],ascending=True,inplace=True)
                    
#                     cyl_df.loc[cyl_df[cyl_df['Estimated engine load']==loads].index,'matched_load'] = load_cons2.iloc[0,0]
#                     cyl_df.loc[cyl_df[cyl_df['Estimated engine load']==loads].index,'matched_date'] = load_cons2.index[0]
#                     cyl_df.loc[cyl_df[cyl_df['Estimated engine load']==loads].index,'deltas'] = load_cons2.iloc[0,1]   
#                 else:
#                     print('no elements')
#                     for efds in self.Efd_features:
#                         # ml_output_dict['Ref_'+efds] = '' 
#                         cyl_df.loc[cyl_df[cyl_df['Estimated engine load']==loads].index,'Ref_'+efds] = '' 

#                     cyl_df.loc[cyl_df[cyl_df['Estimated engine load']==loads].index,'matched_load'] = ''
#                     cyl_df.loc[cyl_df[cyl_df['Estimated engine load']==loads].index,'matched_date'] = ''
#                     cyl_df.loc[cyl_df[cyl_df['Estimated engine load']==loads].index,'deltas'] = ''
#             df = df2.loc[list(cyl_df['matched_date'])]
            
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
#             df['Firing Pr. Balancing Injection Offset Cyl AVG'] = df['Firing Pr. Balancing Injection Offset Cyl_AVG']
            
            
#             for efds in self.Efd_features:# ['Pscav','Pcomp','Pmax','Texh','Ntc','Ntc_Pscav','Pcomp_Pscav','PR']
#                 if efds == 'Pcomp_Pscav': #EFD1
#                     # print('Pcomp_Pscav')                  
#                     model_inputs  = df[self.utility_dict[efds]['imp_feature']]
#                     #Apply scaling here for new inputs
#                     model_inputs = pd.DataFrame(self.Pcomp_Pscav_scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
#                     y_pred = self.Pcomp_Pscav_ml_model.predict(model_inputs)
#                     y_pred = self.Pcomp_Pscav_scaler_y.inverse_transform(y_pred.reshape(-1,1))
#                     cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
#                 elif efds == 'PR': #EFD2
#                     # print('PR')
#                     model_inputs  = df[self.utility_dict[efds]['imp_feature']]
#                     #Apply scaling here for new inputs                   
#                     model_inputs = pd.DataFrame(self.PR_scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
#                     y_pred = self.PR_ml_model.predict(model_inputs)                 
#                     y_pred = self.PR_scaler_y.inverse_transform(y_pred.reshape(-1,1))
#                     cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
#                 elif efds == 'Ntc_Pscav': #EFD3
#                     # print('Ntc_Pscav')                   
#                     model_inputs  = df[self.utility_dict[efds]['imp_feature']]
#                     #Apply scaling here for new inputs                    
#                     model_inputs = pd.DataFrame(self.Ntc_Pscav_scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
#                     y_pred = self.Ntc_Pscav_ml_model.predict(model_inputs)                    
#                     y_pred = self.Ntc_Pscav_scaler_y.inverse_transform(y_pred.reshape(-1,1))
#                     cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
#                 #till here model retunned    
#                 elif efds == 'Pmax': #EFD4
#                     # print('Pmax')                   
#                     model_inputs  = df[self.utility_dict[efds]['imp_feature']]
#                     #Apply scaling here for new inputs                    
#                     model_inputs = pd.DataFrame(self.Pmax_scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
#                     y_pred = self.Pmax_ml_model.predict(model_inputs)                  
#                     y_pred = self.Pmax_scaler_y.inverse_transform(y_pred.reshape(-1,1))
#                     cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
#                 elif efds == 'Texh': #EFD5
#                     # print('Texh')                    
#                     model_inputs  = df[self.utility_dict[efds]['imp_feature']]
#                     #Apply scaling here for new inputs                
#                     model_inputs = pd.DataFrame(self.Texh_scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
#                     y_pred = self.Texh_ml_model.predict(model_inputs)                    
#                     y_pred = self.Texh_scaler_y.inverse_transform(y_pred.reshape(-1,1))
#                     cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
#                 elif efds == 'Ntc': #EFD6
#                     # print('Ntc')                   
#                     model_inputs  = df[self.utility_dict[efds]['imp_feature']]
#                     #Apply scaling here for new inputs                  
#                     model_inputs = pd.DataFrame(self.Ntc_scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
#                     y_pred = self.Ntc_ml_model.predict(model_inputs)                   
#                     y_pred = self.Ntc_scaler_y.inverse_transform(y_pred.reshape(-1,1))
#                     cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
#                 elif efds == 'Pcomp': #EFD7
#                     # print('Pcomp')                    
#                     model_inputs  = df[self.utility_dict[efds]['imp_feature']]
#                     #Applying scaling to new inputs                    
#                     model_inputs = pd.DataFrame(self.Pcomp_scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
#                     y_pred = self.Pcomp_ml_model.predict(model_inputs)
#                     #Applying scaling to outputs
#                     y_pred = self.Pcomp_scaler_y.inverse_transform(y_pred.reshape(-1,1))
#                     cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
#                 elif efds == 'Pscav': #EFD8
#                     # print('Pscav')                   
#                     model_inputs  = df[self.utility_dict[efds]['imp_feature']]
#                     #Applying scaling to new inputs
#                     model_inputs = pd.DataFrame(self.Pscav_scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
#                     y_pred = self.Pscav_ml_model.predict(model_inputs)
#                     #Applying scaling to outputs                     
#                     y_pred = self.Pscav_scaler_y.inverse_transform(y_pred.reshape(-1,1))
#                     cyl_df['Ref_'+efds] =  [re[0] for re in y_pred.tolist()]    
                      
                
#             cyl_df.to_csv(self.ml_res+'ENG_{}_TS_ML_res_Cyl_{}.csv'.format(self.engine_number,cyl),index=False)   
#         print('Ml predictions completed!!!')
#         tm28 = time.time() 
#         print('Total time for ml model part :',tm28-tm26) 
import pandas as pd
import numpy as np
from joblib import load
import tensorflow as tf
import os
import time
import yaml
from datetime import datetime
from azure.storage.blob import ContainerClient
from io import StringIO

with open('cbm_yaml.yml','r') as file:
    utility_dict = yaml.safe_load(file)



# class pdm_ml_model():
#     global utility_dict
#     load_limit = utility_dict['load_limit'] #in %
    
#     max_load = utility_dict['max_load'] #max engine load from 1 yr data for normalization
#     cyl_count = utility_dict['cyl_count']
#     def __init__(self,Efd_features,engine_normalized:bool,ts_res,engine_number,ml_res): #raw_data,anomaly_path,Efd_features,feature,input_data
#         #raw_data - raw data path + filename
#         #anomaly_path - anomaly_path + filename
#         self.engine_normalized = engine_normalized #True or False for applying engine based normalization
#         self.Efd_features = Efd_features #list of all EFD features for iter on ml models ['Pscav','Pcomp','Pmax','Texh','Ntc','Ntc_Pscav','Pcomp_Pscav','PR']
#         self.ts_res = ts_res # result from timeseries class which is a dicitionary type - this will be a path of TS results
#         self.engine_number = engine_number
#         self.ml_res = ml_res

#         #scaling models & ml models loading.......
#         self.Pcomp_Pscav_scaler_x = load(utility_dict['Pcomp_Pscav_scaler_x']) 
#         self.Pcomp_Pscav_scaler_y = load(utility_dict['Pcomp_Pscav_scaler_y']) 
#         self.Pcomp_Pscav_ml_model = tf.keras.models.load_model(utility_dict['Pcomp_Pscav_ml_model'])
#         self.PR_scaler_x = load(utility_dict['PR_scaler_x'])
#         self.PR_scaler_y = load(utility_dict['PR_scaler_y']) 
#         self.PR_ml_model = tf.keras.models.load_model(utility_dict['PR_ml_model'])
#         self.Ntc_Pscav_scaler_x = load(utility_dict['Ntc_Pscav_scaler_x'])
#         self.Ntc_Pscav_scaler_y = load(utility_dict['Ntc_Pscav_scaler_y']) 
#         self.Ntc_Pscav_ml_model = tf.keras.models.load_model(utility_dict['Ntc_Pscav_ml_model'])
#         self.Pmax_scaler_x = load(utility_dict['Pmax_scaler_x'])
#         self.Pmax_scaler_y = load(utility_dict['Pmax_scaler_y'])
#         self.Pmax_ml_model = tf.keras.models.load_model(utility_dict['Pmax_ml_model'])
#         self.Texh_scaler_x = load(utility_dict['Texh_scaler_x'])
#         self.Texh_scaler_y = load(utility_dict['Texh_scaler_y'])
#         self.Texh_ml_model = tf.keras.models.load_model(utility_dict['Texh_ml_model'])
#         self.Ntc_scaler_x = load(utility_dict['Ntc_scaler_x'])
#         self.Ntc_scaler_y = load(utility_dict['Ntc_scaler_y']) 
#         self.Ntc_ml_model = tf.keras.models.load_model(utility_dict['Ntc_ml_model'])
#         self.Pcomp_scaler_x = load(utility_dict['Pcomp_scaler_x'])
#         self.Pcomp_scaler_y = load(utility_dict['Pcomp_scaler_y'])
#         self.Pcomp_ml_model = tf.keras.models.load_model(utility_dict['Pcomp_ml_model'])
#         self.Pscav_scaler_x = load(utility_dict['Pscav_scaler_x'])
#         self.Pscav_scaler_y = load(utility_dict['Pscav_scaler_y'])
#         self.Pscav_ml_model = tf.keras.models.load_model(utility_dict['Pscav_ml_model'])

#     def ML_models(self, data):
#         #1)input data
#         #2)imp feature list for each variables
#         #3)add important feature files in parent loc
#         #4)store ml models in a separate folder named 'ML_models'
#         #5)store scaling models in a separate folder named 'Scaling_models' with '_X' extention for inputs and '_Y'extention for output
#         #define df here
#         tm26 = time.time()
#         df2 = data
#         df2 = df2[(df2[utility_dict['engine_load']]>=30)&(df2[utility_dict['engine_load']]<=100)]
#         if self.engine_normalized == True:
#             max_load = self.max_load
#             print(max_load)
#             df2[utility_dict['engine_load']] = df2[utility_dict['engine_load']]/max_load
#             for col in df2.columns:
#                 if col == utility_dict['engine_load']:
#                     df2[col] = df2[utility_dict['engine_load']]
#                 else:
#                     df2[col] = df2[col]*df2[utility_dict['engine_load']]
#         # ml_output_dict = {}          
#         load_delta = {}  
        
#         for cyl in range(1,self.cyl_count+1):
#             cyl_df = pd.read_csv(self.ts_res+'ENG_{}_TS_res_Cyl_{}.csv'.format(self.engine_number,cyl),index_col=False)
#             load_ranges = list(cyl_df[utility_dict['engine_load']].unique())
#             for loads in load_ranges:
#                 load_delta[loads] = abs(df2[utility_dict['engine_load']]-loads)  
#                 load_l_limit = loads*((100-self.load_limit)/100)
#                 load_u_limit = loads*((100+self.load_limit)/100)
#                 # load_cons1 = load_delta[loads][load_delta[loads]>=load_l_limit]
#                 load_cons1 = df2[utility_dict['engine_load']][df2[utility_dict['engine_load']]>=load_l_limit]
#                 load_cons2 = load_cons1[load_cons1<=load_u_limit]
#                 ml_output_dict = {}
#                 if len(load_cons2)>0:
#                     load_cons2 = load_cons2.to_frame()
#                     load_cons2.columns = ['Matched engine load']
#                     load_cons2['load_delta'] = abs(load_cons2['Matched engine load']-loads)
#                     load_cons2.sort_values(by=['load_delta'],ascending=True,inplace=True)
                    
#                     cyl_df.loc[cyl_df[cyl_df[utility_dict['engine_load']]==loads].index,'matched_load'] = load_cons2.iloc[0,0]
#                     cyl_df.loc[cyl_df[cyl_df[utility_dict['engine_load']]==loads].index,'matched_date'] = load_cons2.index[0]
#                     cyl_df.loc[cyl_df[cyl_df[utility_dict['engine_load']]==loads].index,'deltas'] = load_cons2.iloc[0,1]   
#                 else:
#                     pass # no elements
#                     for efds in self.Efd_features:
#                         # ml_output_dict['Ref_'+efds] = '' 
#                         cyl_df.loc[cyl_df[cyl_df[utility_dict['engine_load']]==loads].index,'Ref_'+efds] = 'No Values'       
            
#                     cyl_df.loc[cyl_df[cyl_df[utility_dict['engine_load']]==loads].index,'matched_load'] = 'No Values'
#                     cyl_df.loc[cyl_df[cyl_df[utility_dict['engine_load']]==loads].index,'matched_date'] = 'No Values'
#                     cyl_df.loc[cyl_df[cyl_df[utility_dict['engine_load']]==loads].index,'deltas'] = 'No Values'
#             get_null_index = cyl_df.loc[cyl_df.matched_load=='No Values'].index #get null index for remove from input to ML model
#             # df = df2.loc[list(cyl_df['matched_date'])]
#             df = df2.loc[list(cyl_df.loc[~cyl_df.index.isin(get_null_index)]['matched_date'])]

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
#             df[list(utility_dict['calc_features'].keys())[16]] = df[list(utility_dict['calc_features'].keys())[8]]
            
            
#             for efds in self.Efd_features:# ['Pscav','Pcomp','Pmax','Texh','Ntc','Ntc_Pscav','Pcomp_Pscav','PR']
#                 if efds == utility_dict['efd_features'][0]: #EFD1
#                     # print('Pcomp_Pscav')           
#                     # model_inputs  = df[self.utility_dict[efds]['imp_feature']]       
#                     model_inputs  = df[self.Pcomp_Pscav_scaler_x.feature_names_in_]
#                     #Apply scaling here for new inputs
#                     model_inputs = pd.DataFrame(self.Pcomp_Pscav_scaler_x.transform(np.asarray(model_inputs)),columns=self.Pcomp_Pscav_scaler_x.feature_names_in_)
#                     y_pred = self.Pcomp_Pscav_ml_model.predict(model_inputs)
#                     y_pred = self.Pcomp_Pscav_scaler_y.inverse_transform(y_pred.reshape(-1,1))
#                     # cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
#                     cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
#                 elif efds == utility_dict['efd_features'][1]: #EFD2
#                     # print('PR')
#                     model_inputs  = df[self.PR_scaler_x.feature_names_in_]
#                     #Apply scaling here for new inputs                   
#                     model_inputs = pd.DataFrame(self.PR_scaler_x.transform(np.asarray(model_inputs)),columns=self.PR_scaler_x.feature_names_in_)
#                     y_pred = self.PR_ml_model.predict(model_inputs)                 
#                     y_pred = self.PR_scaler_y.inverse_transform(y_pred.reshape(-1,1))
#                     cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
#                 elif efds == utility_dict['efd_features'][4]: #EFD3
#                     # print('Ntc_Pscav')                   
#                     model_inputs  = df[self.Ntc_Pscav_scaler_x.feature_names_in_]
#                     #Apply scaling here for new inputs                    
#                     model_inputs = pd.DataFrame(self.Ntc_Pscav_scaler_x.transform(np.asarray(model_inputs)),columns=self.Ntc_Pscav_scaler_x.feature_names_in_)
#                     y_pred = self.Ntc_Pscav_ml_model.predict(model_inputs)                    
#                     y_pred = self.Ntc_Pscav_scaler_y.inverse_transform(y_pred.reshape(-1,1))
#                     cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds] =  [re[0] for re in y_pred.tolist()]
#                 #till here model retunned    
#                 elif efds == utility_dict['efd_features'][2]: #EFD4
#                     # print('Pmax')                   
#                     model_inputs  = df[self.Pmax_scaler_x.feature_names_in_]
#                     #Apply scaling here for new inputs                    
#                     model_inputs = pd.DataFrame(self.Pmax_scaler_x.transform(np.asarray(model_inputs)),columns=self.Pmax_scaler_x.feature_names_in_)
#                     y_pred = self.Pmax_ml_model.predict(model_inputs)                  
#                     y_pred = self.Pmax_scaler_y.inverse_transform(y_pred.reshape(-1,1))
#                     cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
#                 elif efds == utility_dict['efd_features'][7]: #EFD5
#                     # print('Texh')                    
#                     model_inputs  = df[self.Texh_scaler_x.feature_names_in_]
#                     #Apply scaling here for new inputs                
#                     model_inputs = pd.DataFrame(self.Texh_scaler_x.transform(np.asarray(model_inputs)),columns=self.Texh_scaler_x.feature_names_in_)
#                     y_pred = self.Texh_ml_model.predict(model_inputs)                    
#                     y_pred = self.Texh_scaler_y.inverse_transform(y_pred.reshape(-1,1))
#                     cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
#                 elif efds == utility_dict['efd_features'][3]: #EFD6
#                     # print('Ntc')                   
#                     model_inputs  = df[self.Ntc_scaler_x.feature_names_in_]
#                     #Apply scaling here for new inputs                  
#                     model_inputs = pd.DataFrame(self.Ntc_scaler_x.transform(np.asarray(model_inputs)),columns=self.Ntc_scaler_x.feature_names_in_)
#                     y_pred = self.Ntc_ml_model.predict(model_inputs)                   
#                     y_pred = self.Ntc_scaler_y.inverse_transform(y_pred.reshape(-1,1))
#                     cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
#                 elif efds == utility_dict['efd_features'][5]: #EFD7
#                     # print('Pcomp')                    
#                     model_inputs  = df[self.Pcomp_scaler_x.feature_names_in_]
#                     #Applying scaling to new inputs                    
#                     model_inputs = pd.DataFrame(self.Pcomp_scaler_x.transform(np.asarray(model_inputs)),columns=self.Pcomp_scaler_x.feature_names_in_)
#                     y_pred = self.Pcomp_ml_model.predict(model_inputs)
#                     #Applying scaling to outputs
#                     y_pred = self.Pcomp_scaler_y.inverse_transform(y_pred.reshape(-1,1))
#                     cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
#                 elif efds == utility_dict['efd_features'][6]: #EFD8
#                     # print('Pscav')                   
#                     model_inputs  = df[self.Pscav_scaler_x.feature_names_in_]
#                     #Applying scaling to new inputs
#                     model_inputs = pd.DataFrame(self.Pscav_scaler_x.transform(np.asarray(model_inputs)),columns=self.Pscav_scaler_x.feature_names_in_)
#                     y_pred = self.Pscav_ml_model.predict(model_inputs)
#                     #Applying scaling to outputs                     
#                     y_pred = self.Pscav_scaler_y.inverse_transform(y_pred.reshape(-1,1))
#                     cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds] =  [re[0] for re in y_pred.tolist()]    
                      
                
#             cyl_df.to_csv(self.ml_res+'ENG_{}_TS_ML_res_Cyl_{}.csv'.format(self.engine_number,cyl),index=False)   
#         print('Ml predictions completed!!!')
#         tm28 = time.time() 
#         print('Total time for ml model part :',tm28-tm26) 

def read_data_from_blob(dataset_name,idx_col,container_client):
    # try:
    data = pd.read_csv(StringIO(container_client.download_blob(dataset_name).content_as_text()),index_col=idx_col)
    return data
class pdm_ml_model():
    global utility_dict
    load_limit = utility_dict['load_limit'] #in %
    
    max_load = utility_dict['max_load'] #max engine load from 1 yr data for normalization
    cyl_count = utility_dict['cyl_count']
    def __init__(self,Efd_features,ts_res,engine_number,ml_res): #raw_data,anomaly_path,Efd_features,feature,input_data
        #raw_data - raw data path + filename
        #anomaly_path - anomaly_path + filename
        self.Efd_features = Efd_features #list of all EFD features for iter on ml models ['Pscav','Pcomp','Pmax','Texh','Ntc','Ntc_Pscav','Pcomp_Pscav','PR']
        self.ts_res = ts_res # result from timeseries class which is a dicitionary type - this will be a path of TS results
        self.engine_number = engine_number
        self.ml_res = ml_res
        #scaling models & ml models loading.......
        self.Pcomp_Pscav_scaler_x = load(utility_dict['Pcomp_Pscav_scaler_x']) 
        self.Pcomp_Pscav_scaler_y = load(utility_dict['Pcomp_Pscav_scaler_y']) 
        self.Pcomp_Pscav_ml_model = tf.keras.models.load_model(utility_dict['Pcomp_Pscav_ml_model'])
        self.PR_scaler_x = load(utility_dict['PR_scaler_x'])
        self.PR_scaler_y = load(utility_dict['PR_scaler_y']) 
        self.PR_ml_model = tf.keras.models.load_model(utility_dict['PR_ml_model'])
        self.Ntc_Pscav_scaler_x = load(utility_dict['Ntc_Pscav_scaler_x'])
        self.Ntc_Pscav_scaler_y = load(utility_dict['Ntc_Pscav_scaler_y']) 
        self.Ntc_Pscav_ml_model = tf.keras.models.load_model(utility_dict['Ntc_Pscav_ml_model'])
        self.Pmax_scaler_x = load(utility_dict['Pmax_scaler_x'])
        self.Pmax_scaler_y = load(utility_dict['Pmax_scaler_y'])
        self.Pmax_ml_model = tf.keras.models.load_model(utility_dict['Pmax_ml_model'])
        self.Texh_scaler_x = load(utility_dict['Texh_scaler_x'])
        self.Texh_scaler_y = load(utility_dict['Texh_scaler_y'])
        self.Texh_ml_model = tf.keras.models.load_model(utility_dict['Texh_ml_model'])
        self.Ntc_scaler_x = load(utility_dict['Ntc_scaler_x'])
        self.Ntc_scaler_y = load(utility_dict['Ntc_scaler_y']) 
        self.Ntc_ml_model = tf.keras.models.load_model(utility_dict['Ntc_ml_model'])
        self.Pcomp_scaler_x = load(utility_dict['Pcomp_scaler_x'])
        self.Pcomp_scaler_y = load(utility_dict['Pcomp_scaler_y'])
        self.Pcomp_ml_model = tf.keras.models.load_model(utility_dict['Pcomp_ml_model'])
        self.Pscav_scaler_x = load(utility_dict['Pscav_scaler_x'])
        self.Pscav_scaler_y = load(utility_dict['Pscav_scaler_y'])
        self.Pscav_ml_model = tf.keras.models.load_model(utility_dict['Pscav_ml_model'])
    
    def ML_models(self, data,eng):
        #1)input data
        #2)imp feature list for each variables
        #3)add important feature files in parent loc
        #4)store ml models in a separate folder named 'ML_models'
        #5)store scaling models in a separate folder named 'Scaling_models' with '_X' extention for inputs and '_Y'extention for output
        #define df here
        container_client = ContainerClient.from_connection_string(
        utility_dict['connection_string'], container_name=utility_dict['container_name'])
        tm26 = time.time()
        df2 = data
        df2 = df2[(df2[utility_dict['engine_load']]>=30)&(df2[utility_dict['engine_load']]<=100)]
        # ml_output_dict = {}          
        load_delta = {}  
        
        for cyl in range(1,self.cyl_count+1):
            cyl_df = read_data_from_blob('Data/'+utility_dict['Vessel_name']+'/Results/TS/'+utility_dict['Vessel_name']+'_ENG_{}_TS_res_Cyl_{}_{}.csv'.format(str(eng),str(cyl),str(datetime.now()).split(' ')[0]),False,container_client)
            # cyl_df = pd.read_csv(self.ts_res+utility_dict['Vessel_name']+'_ENG_{}_TS_res_Cyl_{}_{}.csv'.format(str(eng),cyl,str(datetime.now()).split(' ')[0]),index_col=False)
            load_ranges = list(cyl_df[utility_dict['engine_load']].unique())
            for loads in load_ranges:
                load_delta[loads] = abs(df2[utility_dict['engine_load']]-loads)  
                load_l_limit = loads*((100-self.load_limit)/100)
                load_u_limit = loads*((100+self.load_limit)/100)
                # load_cons1 = load_delta[loads][load_delta[loads]>=load_l_limit]
                load_cons1 = df2[utility_dict['engine_load']][df2[utility_dict['engine_load']]>=load_l_limit]
                load_cons2 = load_cons1[load_cons1<=load_u_limit]
                ml_output_dict = {}
                if len(load_cons2)>0:
                    load_cons2 = load_cons2.to_frame()
                    load_cons2.columns = ['Matched engine load']
                    load_cons2['load_delta'] = abs(load_cons2['Matched engine load']-loads)
                    load_cons2.sort_values(by=['load_delta'],ascending=True,inplace=True)
                    
                    cyl_df.loc[cyl_df[cyl_df[utility_dict['engine_load']]==loads].index,'matched_load'] = load_cons2.iloc[0,0]
                    cyl_df.loc[cyl_df[cyl_df[utility_dict['engine_load']]==loads].index,'matched_date'] = load_cons2.index[0]
                    cyl_df.loc[cyl_df[cyl_df[utility_dict['engine_load']]==loads].index,'deltas'] = load_cons2.iloc[0,1]   
                else:
                    pass # no elements
                    for efds in self.Efd_features:
                        # ml_output_dict['Ref_'+efds] = '' 
                        cyl_df.loc[cyl_df[cyl_df[utility_dict['engine_load']]==loads].index,'Ref_'+efds] = 'No Values'       
            
                    cyl_df.loc[cyl_df[cyl_df[utility_dict['engine_load']]==loads].index,'matched_load'] = 'No Values'
                    cyl_df.loc[cyl_df[cyl_df[utility_dict['engine_load']]==loads].index,'matched_date'] = 'No Values'
                    cyl_df.loc[cyl_df[cyl_df[utility_dict['engine_load']]==loads].index,'deltas'] = 'No Values'
            get_null_index = cyl_df.loc[cyl_df.matched_load=='No Values'].index #get null index for remove from input to ML model
            # df = df2.loc[list(cyl_df['matched_date'])]
            df = df2.loc[list(cyl_df.loc[~cyl_df.index.isin(get_null_index)]['matched_date'])]
            # df.to_csv('checker2.csv')
            # print('shape of df -',df.shape)
            for efds in self.Efd_features:# ['Pscav','Pcomp','Pmax','Texh','Ntc','Ntc_Pscav','Pcomp_Pscav','PR']
                if efds == utility_dict['efd_features'][0]: #EFD1
                    # print('Pcomp_Pscav')           
                    # model_inputs  = df[self.utility_dict[efds]['imp_feature']]       
                    model_inputs  = df[self.Pcomp_Pscav_scaler_x.feature_names_in_]
                    #Apply scaling here for new inputs
                    model_inputs = pd.DataFrame(self.Pcomp_Pscav_scaler_x.transform(np.asarray(model_inputs)),columns=self.Pcomp_Pscav_scaler_x.feature_names_in_)
                    y_pred = self.Pcomp_Pscav_ml_model.predict(model_inputs)
                    y_pred = self.Pcomp_Pscav_scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    # cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
                    # print('get_null_index -',get_null_index)
                    # print('y_pred - ',len(y_pred))
                    # print('cyl_df -',len(cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds]))
                    cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
                elif efds == utility_dict['efd_features'][1]: #EFD2
                    # print('PR')
                    model_inputs  = df[self.PR_scaler_x.feature_names_in_]
                    #Apply scaling here for new inputs                   
                    model_inputs = pd.DataFrame(self.PR_scaler_x.transform(np.asarray(model_inputs)),columns=self.PR_scaler_x.feature_names_in_)
                    y_pred = self.PR_ml_model.predict(model_inputs)                 
                    y_pred = self.PR_scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    # print('get_null_index -',get_null_index)
                    # print('y_pred - ',len(y_pred))
                    # print('cyl_df -',len(cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds]))
                    cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
                elif efds == utility_dict['efd_features'][4]: #EFD3
                    # print('Ntc_Pscav')                   
                    model_inputs  = df[self.Ntc_Pscav_scaler_x.feature_names_in_]
                    #Apply scaling here for new inputs                    
                    model_inputs = pd.DataFrame(self.Ntc_Pscav_scaler_x.transform(np.asarray(model_inputs)),columns=self.Ntc_Pscav_scaler_x.feature_names_in_)
                    y_pred = self.Ntc_Pscav_ml_model.predict(model_inputs)                    
                    y_pred = self.Ntc_Pscav_scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    # print('get_null_index -',get_null_index)
                    # print('y_pred - ',len(y_pred))
                    # print('cyl_df -',len(cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds]))
                    cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds] =  [re[0] for re in y_pred.tolist()]
                #till here model retunned    
                elif efds == utility_dict['efd_features'][2]: #EFD4
                    # print('Pmax')                   
                    model_inputs  = df[self.Pmax_scaler_x.feature_names_in_]
                    #Apply scaling here for new inputs                    
                    model_inputs = pd.DataFrame(self.Pmax_scaler_x.transform(np.asarray(model_inputs)),columns=self.Pmax_scaler_x.feature_names_in_)
                    y_pred = self.Pmax_ml_model.predict(model_inputs)                  
                    y_pred = self.Pmax_scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    # print('get_null_index -',get_null_index)
                    # print('y_pred - ',len(y_pred))
                    # print('cyl_df -',len(cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds]))
                    cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
                elif efds == utility_dict['efd_features'][7]: #EFD5
                    # print('Texh')                    
                    model_inputs  = df[self.Texh_scaler_x.feature_names_in_]
                    #Apply scaling here for new inputs                
                    model_inputs = pd.DataFrame(self.Texh_scaler_x.transform(np.asarray(model_inputs)),columns=self.Texh_scaler_x.feature_names_in_)
                    y_pred = self.Texh_ml_model.predict(model_inputs)                    
                    y_pred = self.Texh_scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    # print('get_null_index -',get_null_index)
                    # print('y_pred - ',len(y_pred))
                    # print('cyl_df -',len(cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds]))
                    cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
                elif efds == utility_dict['efd_features'][3]: #EFD6
                    # print('Ntc')                   
                    model_inputs  = df[self.Ntc_scaler_x.feature_names_in_]
                    #Apply scaling here for new inputs                  
                    model_inputs = pd.DataFrame(self.Ntc_scaler_x.transform(np.asarray(model_inputs)),columns=self.Ntc_scaler_x.feature_names_in_)
                    y_pred = self.Ntc_ml_model.predict(model_inputs)                   
                    y_pred = self.Ntc_scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    # print('get_null_index -',get_null_index)
                    # print('y_pred - ',len(y_pred))
                    # print('cyl_df -',len(cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds]))
                    cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
                elif efds == utility_dict['efd_features'][5]: #EFD7
                    # print('Pcomp')                    
                    model_inputs  = df[self.Pcomp_scaler_x.feature_names_in_]
                    #Applying scaling to new inputs                    
                    model_inputs = pd.DataFrame(self.Pcomp_scaler_x.transform(np.asarray(model_inputs)),columns=self.Pcomp_scaler_x.feature_names_in_)
                    y_pred = self.Pcomp_ml_model.predict(model_inputs)
                    #Applying scaling to outputs
                    y_pred = self.Pcomp_scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    # print('get_null_index -',get_null_index)
                    # print('y_pred - ',len(y_pred))
                    # print('cyl_df -',len(cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds]))
                    cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
                elif efds == utility_dict['efd_features'][6]: #EFD8
                    # print('Pscav')                   
                    model_inputs  = df[self.Pscav_scaler_x.feature_names_in_]
                    #Applying scaling to new inputs
                    model_inputs = pd.DataFrame(self.Pscav_scaler_x.transform(np.asarray(model_inputs)),columns=self.Pscav_scaler_x.feature_names_in_)
                    y_pred = self.Pscav_ml_model.predict(model_inputs)
                    #Applying scaling to outputs                     
                    y_pred = self.Pscav_scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    # print('get_null_index -',get_null_index)
                    # print('y_pred - ',len(y_pred))
                    # print('cyl_df -',len(cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds]))
                    cyl_df.loc[~cyl_df.index.isin(get_null_index),'Ref_'+efds] =  [re[0] for re in y_pred.tolist()]    
                    
            
            cyl_df_csv = cyl_df.to_csv(index=False)
            cyl_df_csv_bytes = bytes(cyl_df_csv, 'utf-8')
            cyl_df_csv_stream = StringIO(cyl_df_csv)
            container_client.upload_blob(name="Data/"+utility_dict['Vessel_name']+'/Results/ML/'+utility_dict['Vessel_name']+'_ENG_{}_TS_ML_res_Cyl_{}_{}.csv'.format(str(eng),cyl,str(datetime.now()).split(' ')[0]), data=cyl_df_csv_bytes,overwrite=True)
            cyl_df_csv_stream.close()
            # cyl_df.to_csv(self.ml_res+utility_dict['Vessel_name']+'_ENG_{}_TS_ML_res_Cyl_{}_{}.csv'.format(str(eng),cyl,str(datetime.now()).split(' ')[0]),index=False)   
        print('Ml predictions completed!!!')
        tm28 = time.time() 
        print('Total time for ml model part :',tm28-tm26)

