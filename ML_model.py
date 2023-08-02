import pandas as pd
import numpy as np
from joblib import load
import tensorflow as tf
import os
import time



class pdm_ml_model():
    load_limit = 5 #in %
    
    max_load = 84.19076333333332 #max engine load from 1 yr data for normalization
    cyl_count = 6
    utility_dict = {'Pscav':{'Limits': {'L_limit': -10, 'U_limit': 10}, 'imp_feature': list(pd.read_csv(('./utils/ML_model/imp_features/Pscav.csv') )['Features'][:24])},
                    'Pcomp':{'Limits': {'L_limit': -8, 'U_limit': 8}, 'imp_feature': list(pd.read_csv('./utils/ML_model/imp_features/Pcomp.csv')['Features'][:26])},
                    'Pmax':{'Limits': {'L_limit': -8, 'U_limit': 8}, 'imp_feature': list(pd.read_csv('./utils/ML_model/imp_features/Pmax.csv')['Features'][:27])},
                    'Texh':{'Limits': {'L_limit': -8, 'U_limit': 8}, 'imp_feature': list(pd.read_csv('./utils/ML_model/imp_features/Texh.csv')['Features'][:24])},
                    'Ntc':{'Limits': {'L_limit': -8, 'U_limit': 8}, 'imp_feature': list(pd.read_csv('./utils/ML_model/imp_features/Ntc.csv')['Features'][:23])},
                    'Ntc_Pscav':{'Limits': {'L_limit': -8, 'U_limit': 8}, 'imp_feature': list(pd.read_csv('./utils/ML_model/imp_features/Ntc_Pscav.csv')['Features'][:24])},
                    'Pcomp_Pscav':{'Limits': {'L_limit': -10, 'U_limit': 10}, 'imp_feature': list(pd.read_csv('./utils/ML_model/imp_features/Pcomp_Pscav.csv')['Features'][:24])},
                    'PR':{'Limits': {'L_limit': -15, 'U_limit': 15}, 'imp_feature': list(pd.read_csv('./utils/ML_model/imp_features/PR.csv')['Features'][:26])}}

    def __init__(self,saved_data,Efd_features,engine_normalized:bool,ts_res,engine_number,ml_res): #raw_data,anomaly_path,Efd_features,feature,input_data
        #raw_data - raw data path + filename
        #anomaly_path - anomaly_path + filename
        self.saved_data = saved_data # 3 months stored data with agg, imputation & filtered using Auto Encoder
        self.engine_normalized = engine_normalized #True or False for applying engine based normalization
        self.Efd_features = Efd_features #list of all EFD features for iter on ml models ['Pscav','Pcomp','Pmax','Texh','Ntc','Ntc_Pscav','Pcomp_Pscav','PR']
        self.ts_res = ts_res # result from timeseries class which is a dicitionary type - this will be a path of TS results
        self.engine_number = engine_number
        self.ml_res = ml_res

    def ML_models(self):
        #1)input data
        #2)imp feature list for each variables
        #3)add important feature files in parent loc
        #4)store ml models in a separate folder named 'ML_models'
        #5)store scaling models in a separate folder named 'Scaling_models' with '_X' extention for inputs and '_Y'extention for output
        #define df here
        df2 = self.saved_data
        df2 = df2[(df2['Estimated engine load']>=30)&(df2['Estimated engine load']<=100)]
        if self.engine_normalized == True:
            max_load = self.max_load
            print(max_load)
            df2['Estimated engine load'] = df2['Estimated engine load']/max_load
            for col in df2.columns:
                if col == 'Estimated engine load':
                    df2[col] = df2['Estimated engine load']
                else:
                    df2[col] = df2[col]*df2['Estimated engine load']
        ml_output_dict = {}
        joblib_models = []
        ann_models = ['Pcomp_Pscav','PR','Pmax','Ntc','Ntc_Pscav','Pcomp','Pscav','Texh']               
        load_delta = {}  
        
        tm1 = time.time()
        #scaling models & ml models loading.......
        Pcomp_Pscav_scaler_x = load("./utils/ML_model/features_scaler/"+'Pcomp_Pscav'+"_X.joblib") 
        tm2 = time.time()
        print('Pcomp_Pscav_x :',tm2-tm1)
        Pcomp_Pscav_scaler_y = load("./utils/ML_model/features_scaler/"+'Pcomp_Pscav'+"_Y.joblib") 
        tm3 = time.time()
        print('Pcomp_Pscav_y :',tm3-tm2)
        Pcomp_Pscav_ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+'Pcomp_Pscav'+'.keras')
        tm4 = time.time()
        print('Pcomp_Pscav_model :',tm4-tm3)

        PR_scaler_x = load("./utils/ML_model/features_scaler/"+'PR'+"_X.joblib")
        tm5 = time.time()
        print('PR_x :',tm5-tm4)
        PR_scaler_y = load("./utils/ML_model/features_scaler/"+'PR'+"_Y.joblib") 
        tm6 = time.time()
        print('PR_y :',tm6-tm5)
        PR_ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+'PR'+'.keras')
        tm7 = time.time()
        print('PR_model :',tm7-tm6)

        Ntc_Pscav_scaler_x = load("./utils/ML_model/features_scaler/"+'Ntc_Pscav'+"_X.joblib")
        tm8 = time.time()
        print('Ntc_Pscav_x :',tm8-tm7)
        Ntc_Pscav_scaler_y = load("./utils/ML_model/features_scaler/"+'Ntc_Pscav'+"_Y.joblib") 
        tm9 = time.time()
        print('Ntc_Pscav_y :',tm9-tm8)
        Ntc_Pscav_ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+'Ntc_Pscav'+'.keras')
        tm10 = time.time()
        print('Ntc_Pscav_model :',tm10-tm9)

        Pmax_scaler_x = load("./utils/ML_model/features_scaler/"+'Pmax'+"_X.joblib")
        tm11 = time.time()
        print('Pmax_x :',tm11-tm10)
        Pmax_scaler_y = load("./utils/ML_model/features_scaler/"+'Pmax'+"_Y.joblib")
        tm12 = time.time() 
        print('Pmax_y :',tm12-tm11)
        Pmax_ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+'Pmax'+'.keras')
        tm13 = time.time()
        print('Pmax_model :',tm13-tm12)

        Texh_scaler_x = load("./utils/ML_model/features_scaler/"+'Texh'+"_X.joblib")
        tm14 = time.time()
        print('Texh_x :',tm14-tm13)
        Texh_scaler_y = load("./utils/ML_model/features_scaler/"+'Texh'+"_Y.joblib")
        tm15 = time.time() 
        print('Texh_y :',tm15-tm14)
        Texh_ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+'Texh'+'.keras')
        tm16 = time.time()
        print('Texh_model :',tm16-tm15)

        Ntc_scaler_x = load("./utils/ML_model/features_scaler/"+'Ntc'+"_X.joblib")
        tm17 = time.time()
        print('Ntc_x :',tm17-tm16)
        Ntc_scaler_y = load("./utils/ML_model/features_scaler/"+'Ntc'+"_Y.joblib") 
        tm18 = time.time()
        print('Ntc_y :',tm18-tm17)
        Ntc_ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+'Ntc'+'.keras')
        tm19 = time.time()
        print('Ntc_model :',tm19-tm18)

        Pcomp_scaler_x = load("./utils/ML_model/features_scaler/"+'Pcomp'+"_X.joblib")
        tm20 = time.time()
        print('Pcom_x :',tm20-tm19)
        Pcomp_scaler_y = load("./utils/ML_model/features_scaler/"+'Pcomp'+"_Y.joblib")
        tm21 = time.time()
        print('Pcom_y :',tm21-tm20)
        Pcomp_ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+'Pcomp'+'.keras')
        tm22 = time.time()
        print('Pcom_model :',tm22-tm21)

        Pscav_scaler_x = load("./utils/ML_model/features_scaler/"+'Pscav'+"_X.joblib")
        tm23 = time.time()
        print('Pscav_x :',tm23-tm22)
        Pscav_scaler_y = load("./utils/ML_model/features_scaler/"+'Pscav'+"_Y.joblib")
        tm24 = time.time()
        print('Pscav_y :',tm24-tm23)
        Pscav_ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+'Pscav'+'.keras')
        tm25 = time.time()
        print('Pscav_model :',tm25-tm24)

        print('Total time for scalers and ml models :',tm25-tm1) 
        
        for cyl in range(1,self.cyl_count+1):
            cyl_df = pd.read_csv(self.ts_res+'ENG_{}_TS_res_Cyl_{}.csv'.format(self.engine_number,cyl),index_col=False)
            # load_ranges = list(cyl_df['Engine load range'].unique())          
            load_ranges = list(cyl_df['Estimated engine load'].unique())
            tm26 = time.time()                  
            for loads in load_ranges:
                load_delta[loads] = abs(df2['Estimated engine load']-loads)  
                load_l_limit = loads*((100-self.load_limit)/100)
                load_u_limit = loads*((100+self.load_limit)/100)
                # load_cons1 = load_delta[loads][load_delta[loads]>=load_l_limit]
                load_cons1 = df2['Estimated engine load'][df2['Estimated engine load']>=load_l_limit]
                load_cons2 = load_cons1[load_cons1<=load_u_limit]
                ml_output_dict = {}
                if len(load_cons2)>0:
                    load_cons2 = load_cons2.to_frame()
                    load_cons2.columns = ['Matched engine load']
                    load_cons2['load_delta'] = abs(load_cons2['Matched engine load']-loads)
                    load_cons2.sort_values(by=['load_delta'],ascending=True,inplace=True)
                    
                    cyl_df.loc[cyl_df[cyl_df['Estimated engine load']==loads].index,'matched_load'] = load_cons2['Matched engine load'][0]
                    cyl_df.loc[cyl_df[cyl_df['Estimated engine load']==loads].index,'matched_date'] = load_cons2.index[0]
                    cyl_df.loc[cyl_df[cyl_df['Estimated engine load']==loads].index,'deltas'] = load_cons2['load_delta'][0]   

                    # df = df2.loc[load_cons2.index[0]]
                else:
                    print('no elements')
                    for efds in self.Efd_features:
                        # ml_output_dict['Ref_'+efds] = '' 
                        cyl_df.loc[cyl_df[cyl_df['Estimated engine load']==loads].index,'Ref_'+efds] = '' 
                    # matched_load.append('')
                    # matched_date.append('')
                    # deltas.append('')
                    cyl_df.loc[cyl_df[cyl_df['Estimated engine load']==loads].index,'matched_load'] = ''
                    cyl_df.loc[cyl_df[cyl_df['Estimated engine load']==loads].index,'matched_date'] = ''
                    cyl_df.loc[cyl_df[cyl_df['Estimated engine load']==loads].index,'deltas'] = ''
            tm27 = time.time() 
            print('Time for matching loads for cyl',str(cyl),' :',tm27-tm26)       
            df = df2.loc[list(cyl_df['matched_date'])]
            
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
            df['Firing Pr. Balancing Injection Offset Cyl AVG'] = df['Firing Pr. Balancing Injection Offset Cyl_AVG']
            
            for efds in self.Efd_features:# ['Pscav','Pcomp','Pmax','Texh','Ntc','Ntc_Pscav','Pcomp_Pscav','PR']
                if efds == 'Pcomp_Pscav': #EFD1
                    print('Pcomp_Pscav')                  
                    model_inputs  = df[self.utility_dict[efds]['imp_feature']]
                    #Apply scaling here for new inputs
                    model_inputs = pd.DataFrame(Pcomp_Pscav_scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
                    y_pred = Pcomp_Pscav_ml_model.predict(model_inputs)
                    y_pred = Pcomp_Pscav_scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
                elif efds == 'PR': #EFD2
                    print('PR')
                    model_inputs  = df[self.utility_dict[efds]['imp_feature']]
                    #Apply scaling here for new inputs                   
                    model_inputs = pd.DataFrame(PR_scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
                    y_pred = PR_ml_model.predict(model_inputs)                 
                    y_pred = PR_scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
                elif efds == 'Ntc_Pscav': #EFD3
                    print('Ntc_Pscav')                   
                    model_inputs  = df[self.utility_dict[efds]['imp_feature']]
                    #Apply scaling here for new inputs                    
                    model_inputs = pd.DataFrame(Ntc_Pscav_scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
                    y_pred = Ntc_Pscav_ml_model.predict(model_inputs)                    
                    y_pred = Ntc_Pscav_scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
                #till here model retunned    
                elif efds == 'Pmax': #EFD4
                    print('Pmax')                   
                    model_inputs  = df[self.utility_dict[efds]['imp_feature']]
                    #Apply scaling here for new inputs                    
                    model_inputs = pd.DataFrame(Pmax_scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
                    y_pred = Pmax_ml_model.predict(model_inputs)                  
                    y_pred = Pmax_scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
                elif efds == 'Texh': #EFD5
                    print('Texh')                    
                    model_inputs  = df[self.utility_dict[efds]['imp_feature']]
                    #Apply scaling here for new inputs                
                    model_inputs = pd.DataFrame(Texh_scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
                    y_pred = Texh_ml_model.predict(model_inputs)                    
                    y_pred = Texh_scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
                elif efds == 'Ntc': #EFD6
                    print('Ntc')                   
                    model_inputs  = df[self.utility_dict[efds]['imp_feature']]
                    #Apply scaling here for new inputs                  
                    model_inputs = pd.DataFrame(Ntc_scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
                    y_pred = Ntc_ml_model.predict(model_inputs)                   
                    y_pred = Ntc_scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
                elif efds == 'Pcomp': #EFD7
                    print('Pcomp')                    
                    model_inputs  = df[self.utility_dict[efds]['imp_feature']]
                    #Applying scaling to new inputs                    
                    model_inputs = pd.DataFrame(Pcomp_scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
                    y_pred = Pcomp_ml_model.predict(model_inputs)
                    #Applying scaling to outputs
                    y_pred = Pcomp_scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
                elif efds == 'Pscav': #EFD8
                    print('Pscav')                   
                    model_inputs  = df[self.utility_dict[efds]['imp_feature']]
                    #Applying scaling to new inputs
                    model_inputs = pd.DataFrame(Pscav_scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
                    y_pred = Pscav_ml_model.predict(model_inputs)
                    #Applying scaling to outputs                     
                    y_pred = Pscav_scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    cyl_df['Ref_'+efds] =  [re[0] for re in y_pred.tolist()]
                      
            tm28 = time.time() 
            print('Total time for cyl ',str(cyl),'ml part :',tm28-tm26)    
                      
                
            cyl_df.to_csv(self.ml_res+'ENG_{}_TS_ML_res_Cyl_{}.csv'.format(self.engine_number,cyl),index=False)   
        print('Ml predictions completed!!!')   
        # full_data2.reset_index(drop=True,inplace=True) 
        # return full_data1 , full_data2, load_delta, mean_load
