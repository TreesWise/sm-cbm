import pandas as pd
import numpy as np
from joblib import load
import tensorflow as tf
import os



class pdm_ml_model():
    load_limit = 5 #in %
    current_dir = os.getcwd()
    print("---------------------",current_dir)
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
        for cyl in range(1,self.cyl_count+1):
            cyl_df = pd.read_csv(self.ts_res+'ENG_{}_TS_res_Cyl_{}.csv'.format(self.engine_number,cyl))
            # load_ranges = list(cyl_df['Engine load range'].unique())          
            load_ranges = list(cyl_df['Estimated engine load'].unique())                  
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
            df = df2.loc[list(cyl_df['matched_date'])]
            # Pcomp_Pscav
            df['Cyl. lub. distribution share below_PERC'] = (df['Cyl. lub. distribution share below piston']/df['Cyl. lub. distribution share into piston'])*100
            df['Cyl. lub. distribution share above_PERC'] = (df['Cyl. lub. distribution share above piston']/df['Cyl. lub. distribution share into piston'])*100
            df['Fuel Rail Pressure_diff'] = df['Mean Fuel Rail Pressure (display)'] - df['Main Fuel Rail Pressure']
            df['GAV Timing Set Point Cyl AVG'] = (df['GAV Timing Set Point Cyl #01']+df['GAV Timing Set Point Cyl #02']+
                                                        df['GAV Timing Set Point Cyl #03']+df['GAV Timing Set Point Cyl #04']+
                                                        df['GAV Timing Set Point Cyl #05']+df['GAV Timing Set Point Cyl #06'])/6
            df['PFI Timing Set Point Cyl AVG'] = (df['PFI Timing Set Point Cyl #01']+df['PFI Timing Set Point Cyl #02']+
                                                            df['PFI Timing Set Point Cyl #03']+df['PFI Timing Set Point Cyl #04']+
                                                            df['PFI Timing Set Point Cyl #05']+df['PFI Timing Set Point Cyl #06'])/6
            df['Fuel Pressure Actuator Setpoint_AVG'] = (df['Fuel Pressure Actuator Setpoint 1']+df['Fuel Pressure Actuator Setpoint 2']+df['Fuel Pressure Actuator Setpoint 3'])/3
            df['Fuel Pump Setpoint_AVG'] = (df['Fuel Pump Setpoint Master Controller']+df['Fuel Pump Setpoint Slave Controller'])/2
            df['Firing Pr. Balancing Injection Offset Cyl_AVG'] = (df['Firing Pr. Balancing Injection Offset Cyl #01']+df['Firing Pr. Balancing Injection Offset Cyl #02']+
                                                                        df['Firing Pr. Balancing Injection Offset Cyl #03']+df['Firing Pr. Balancing Injection Offset Cyl #04']+
                                                                        df['Firing Pr. Balancing Injection Offset Cyl #05']+df['Firing Pr. Balancing Injection Offset Cyl #06'])/6


            df['Lubrication Oil Feed Rate Cyl AVG'] = (df['Lubrication Oil Feed Rate Cyl #01']+df['Lubrication Oil Feed Rate Cyl #02']+
                                                            df['Lubrication Oil Feed Rate Cyl #03']+df['Lubrication Oil Feed Rate Cyl #04']+
                                                            df['Lubrication Oil Feed Rate Cyl #05']+df['Lubrication Oil Feed Rate Cyl #06'])/6
            df['Exh. valve opening angle Cyl AVG'] = (df['Exh. valve opening angle Cyl #01']+df['Exh. valve opening angle Cyl #02']+
                                                            df['Exh. valve opening angle Cyl #03']+df['Exh. valve opening angle Cyl #04']+
                                                            df['Exh. valve opening angle Cyl #05']+df['Exh. valve opening angle Cyl #06'])/6
            df['Exhaust Valve Closing Angle Setpoint Cyl AVG'] = (df['Exhaust Valve Closing Angle Setpoint Cyl #01']+df['Exhaust Valve Closing Angle Setpoint Cyl #02']+
                                                                        df['Exhaust Valve Closing Angle Setpoint Cyl #03']+df['Exhaust Valve Closing Angle Setpoint Cyl #04']+
                                                                        df['Exhaust Valve Closing Angle Setpoint Cyl #05']+df['Exhaust Valve Closing Angle Setpoint Cyl #06'])/6
            # PR
            df['Fuel Rail Pressure_diff'] = df['Mean Fuel Rail Pressure (display)'] - df['Main Fuel Rail Pressure']
            df['GAV Timing Set Point Cyl AVG'] = (df['GAV Timing Set Point Cyl #01']+df['GAV Timing Set Point Cyl #02']+
                                                        df['GAV Timing Set Point Cyl #03']+df['GAV Timing Set Point Cyl #04']+
                                                        df['GAV Timing Set Point Cyl #05']+df['GAV Timing Set Point Cyl #06'])/6
            df['Cyl. lub. distribution share below_PERC'] = (df['Cyl. lub. distribution share below piston']/df['Cyl. lub. distribution share into piston'])*100
            df['Cyl. lub. distribution share above_PERC'] = (df['Cyl. lub. distribution share above piston']/df['Cyl. lub. distribution share into piston'])*100
            df['PFI Timing Set Point Cyl AVG'] = (df['PFI Timing Set Point Cyl #01']+df['PFI Timing Set Point Cyl #02']+
                                                            df['PFI Timing Set Point Cyl #03']+df['PFI Timing Set Point Cyl #04']+
                                                            df['PFI Timing Set Point Cyl #05']+df['PFI Timing Set Point Cyl #06'])/6
            df['PFI Duration Set Point Cyl AVG'] = (df['PFI Duration Set Point Cyl #01']+df['PFI Duration Set Point Cyl #02']+
                                                        df['PFI Duration Set Point Cyl #03']+df['PFI Duration Set Point Cyl #04']+
                                                        df['PFI Duration Set Point Cyl #05']+df['PFI Duration Set Point Cyl #06'])/6
            df['Pilot Fuel Pressure diff'] = df['Pilot Fuel Pressure A']-df['Pilot Fuel Pressure B']
            df['Exhaust Valve Closing Angle Setpoint Cyl AVG'] = (df['Exhaust Valve Closing Angle Setpoint Cyl #01']+df['Exhaust Valve Closing Angle Setpoint Cyl #02']+
                                                                        df['Exhaust Valve Closing Angle Setpoint Cyl #03']+df['Exhaust Valve Closing Angle Setpoint Cyl #04']+
                                                                        df['Exhaust Valve Closing Angle Setpoint Cyl #05']+df['Exhaust Valve Closing Angle Setpoint Cyl #06'])/6
            df['Lubrication Deadtime Feedback Cyl AVG'] = (df['Lubrication Deadtime Feedback Cyl #01']+df['Lubrication Deadtime Feedback Cyl #02']+
                                                                df['Lubrication Deadtime Feedback Cyl #03']+df['Lubrication Deadtime Feedback Cyl #04']+
                                                                df['Lubrication Deadtime Feedback Cyl #05']+df['Lubrication Deadtime Feedback Cyl #06'])/6
            df['Fuel Pump Setpoint_AVG'] = (df['Fuel Pump Setpoint Master Controller']+df['Fuel Pump Setpoint Slave Controller'])/2
            df['Fuel Pressure Actuator Setpoint_AVG'] = (df['Fuel Pressure Actuator Setpoint 1']+df['Fuel Pressure Actuator Setpoint 2']+df['Fuel Pressure Actuator Setpoint 3'])/3
            df['Start of Injection Cyl_AVG'] = (df['Start of Injection Cyl #01']+df['Start of Injection Cyl #02']+df['Start of Injection Cyl #03']+df['Start of Injection Cyl #04']+
                                                df['Start of Injection Cyl #05']+df['Start of Injection Cyl #06'])/6
            df['Scavenge Air Temp. Piston Underside Cyl_AVG'] = (df['Scavenge Air Temp. Piston Underside Cyl #01.1']+df['Scavenge Air Temp. Piston Underside Cyl #02.1']+df['Scavenge Air Temp. Piston Underside Cyl #03.1']+df['Scavenge Air Temp. Piston Underside Cyl #04.1']+
                                                                df['Scavenge Air Temp. Piston Underside Cyl #05.1']+df['Scavenge Air Temp. Piston Underside Cyl #06.1'])/6
            #Ntc_Pscav
            df['Fuel Rail Pressure_diff'] = df['Mean Fuel Rail Pressure (display)'] - df['Main Fuel Rail Pressure']
            df['GAV Timing Set Point Cyl AVG'] = (df['GAV Timing Set Point Cyl #01']+df['GAV Timing Set Point Cyl #02']+
                                                        df['GAV Timing Set Point Cyl #03']+df['GAV Timing Set Point Cyl #04']+
                                                        df['GAV Timing Set Point Cyl #05']+df['GAV Timing Set Point Cyl #06'])/6
            df['Fuel Pressure Actuator Setpoint_AVG'] = (df['Fuel Pressure Actuator Setpoint 1']+df['Fuel Pressure Actuator Setpoint 2']+df['Fuel Pressure Actuator Setpoint 3'])/3
            df['PFI Timing Set Point Cyl AVG'] = (df['PFI Timing Set Point Cyl #01']+df['PFI Timing Set Point Cyl #02']+
                                                            df['PFI Timing Set Point Cyl #03']+df['PFI Timing Set Point Cyl #04']+
                                                            df['PFI Timing Set Point Cyl #05']+df['PFI Timing Set Point Cyl #06'])/6
            df['Cyl. lub. distribution share below_PERC'] = (df['Cyl. lub. distribution share below piston']/df['Cyl. lub. distribution share into piston'])*100
            df['Cyl. lub. distribution share above_PERC'] = (df['Cyl. lub. distribution share above piston']/df['Cyl. lub. distribution share into piston'])*100
            df['PFI Duration Set Point Cyl AVG'] = (df['PFI Duration Set Point Cyl #01']+df['PFI Duration Set Point Cyl #02']+
                                                        df['PFI Duration Set Point Cyl #03']+df['PFI Duration Set Point Cyl #04']+
                                                        df['PFI Duration Set Point Cyl #05']+df['PFI Duration Set Point Cyl #06'])/6
            df['Exhaust Valve Closing Angle Setpoint Cyl AVG'] = (df['Exhaust Valve Closing Angle Setpoint Cyl #01']+df['Exhaust Valve Closing Angle Setpoint Cyl #02']+
                                                                        df['Exhaust Valve Closing Angle Setpoint Cyl #03']+df['Exhaust Valve Closing Angle Setpoint Cyl #04']+
                                                                        df['Exhaust Valve Closing Angle Setpoint Cyl #05']+df['Exhaust Valve Closing Angle Setpoint Cyl #06'])/6
            df['Fuel Pump Setpoint_AVG'] = (df['Fuel Pump Setpoint Master Controller']+df['Fuel Pump Setpoint Slave Controller'])/2
            df['Firing Pr. Balancing Injection Offset Cyl AVG'] = (df['Firing Pr. Balancing Injection Offset Cyl #01']+df['Firing Pr. Balancing Injection Offset Cyl #02']+
                                                                        df['Firing Pr. Balancing Injection Offset Cyl #03']+df['Firing Pr. Balancing Injection Offset Cyl #04']+
                                                                        df['Firing Pr. Balancing Injection Offset Cyl #05']+df['Firing Pr. Balancing Injection Offset Cyl #06'])/6
            #Pmax
            df['Cyl. lub. distribution share below_PERC'] = (df['Cyl. lub. distribution share below piston']/df['Cyl. lub. distribution share into piston'])*100
            df['Cyl. lub. distribution share above_PERC'] = (df['Cyl. lub. distribution share above piston']/df['Cyl. lub. distribution share into piston'])*100
            df['GAV Timing Set Point Cyl AVG'] = (df['GAV Timing Set Point Cyl #01']+df['GAV Timing Set Point Cyl #02']+
                                                        df['GAV Timing Set Point Cyl #03']+df['GAV Timing Set Point Cyl #04']+
                                                        df['GAV Timing Set Point Cyl #05']+df['GAV Timing Set Point Cyl #06'])/6
            df['Exh. valve opening angle Cyl AVG'] = (df['Exh. valve opening angle Cyl #01']+df['Exh. valve opening angle Cyl #02']+
                                                            df['Exh. valve opening angle Cyl #03']+df['Exh. valve opening angle Cyl #04']+
                                                            df['Exh. valve opening angle Cyl #05']+df['Exh. valve opening angle Cyl #06'])/6
            df['PFI Duration Set Point Cyl AVG'] = (df['PFI Duration Set Point Cyl #01']+df['PFI Duration Set Point Cyl #02']+
                                                        df['PFI Duration Set Point Cyl #03']+df['PFI Duration Set Point Cyl #04']+
                                                        df['PFI Duration Set Point Cyl #05']+df['PFI Duration Set Point Cyl #06'])/6
            df['PFI Timing Set Point Cyl AVG'] = (df['PFI Timing Set Point Cyl #01']+df['PFI Timing Set Point Cyl #02']+
                                                            df['PFI Timing Set Point Cyl #03']+df['PFI Timing Set Point Cyl #04']+
                                                            df['PFI Timing Set Point Cyl #05']+df['PFI Timing Set Point Cyl #06'])/6
            df['Exhaust Valve Closing Angle Setpoint Cyl AVG'] = (df['Exhaust Valve Closing Angle Setpoint Cyl #01']+df['Exhaust Valve Closing Angle Setpoint Cyl #02']+
                                                                        df['Exhaust Valve Closing Angle Setpoint Cyl #03']+df['Exhaust Valve Closing Angle Setpoint Cyl #04']+
                                                                        df['Exhaust Valve Closing Angle Setpoint Cyl #05']+df['Exhaust Valve Closing Angle Setpoint Cyl #06'])/6
            df['Fuel Rail Pressure_diff'] = df['Mean Fuel Rail Pressure (display)'] - df['Main Fuel Rail Pressure']
            #Texh
            df['Cyl. lub. distribution share below_PERC'] = (df['Cyl. lub. distribution share below piston']/df['Cyl. lub. distribution share into piston'])*100
            df['Cyl. lub. distribution share above_PERC'] = (df['Cyl. lub. distribution share above piston']/df['Cyl. lub. distribution share into piston'])*100
            df['GAV Timing Set Point Cyl AVG'] = (df['GAV Timing Set Point Cyl #01']+df['GAV Timing Set Point Cyl #02']+
                                                        df['GAV Timing Set Point Cyl #03']+df['GAV Timing Set Point Cyl #04']+
                                                        df['GAV Timing Set Point Cyl #05']+df['GAV Timing Set Point Cyl #06'])/6
            df['Exh. valve opening angle Cyl AVG'] = (df['Exh. valve opening angle Cyl #01']+df['Exh. valve opening angle Cyl #02']+
                                                            df['Exh. valve opening angle Cyl #03']+df['Exh. valve opening angle Cyl #04']+
                                                            df['Exh. valve opening angle Cyl #05']+df['Exh. valve opening angle Cyl #06'])/6
            df['PFI Duration Set Point Cyl AVG'] = (df['PFI Duration Set Point Cyl #01']+df['PFI Duration Set Point Cyl #02']+
                                                        df['PFI Duration Set Point Cyl #03']+df['PFI Duration Set Point Cyl #04']+
                                                        df['PFI Duration Set Point Cyl #05']+df['PFI Duration Set Point Cyl #06'])/6
            df['PFI Timing Set Point Cyl AVG'] = (df['PFI Timing Set Point Cyl #01']+df['PFI Timing Set Point Cyl #02']+
                                                            df['PFI Timing Set Point Cyl #03']+df['PFI Timing Set Point Cyl #04']+
                                                            df['PFI Timing Set Point Cyl #05']+df['PFI Timing Set Point Cyl #06'])/6
            df['Exhaust Valve Closing Angle Setpoint Cyl AVG'] = (df['Exhaust Valve Closing Angle Setpoint Cyl #01']+df['Exhaust Valve Closing Angle Setpoint Cyl #02']+
                                                                        df['Exhaust Valve Closing Angle Setpoint Cyl #03']+df['Exhaust Valve Closing Angle Setpoint Cyl #04']+
                                                                        df['Exhaust Valve Closing Angle Setpoint Cyl #05']+df['Exhaust Valve Closing Angle Setpoint Cyl #06'])/6
            df['Fuel Rail Pressure_diff'] = df['Mean Fuel Rail Pressure (display)'] - df['Main Fuel Rail Pressure']
            #Ntc
            df['Cyl. lub. distribution share below_PERC'] = (df['Cyl. lub. distribution share below piston']/df['Cyl. lub. distribution share into piston'])*100
            df['Cyl. lub. distribution share above_PERC'] = (df['Cyl. lub. distribution share above piston']/df['Cyl. lub. distribution share into piston'])*100
            df['GAV Timing Set Point Cyl AVG'] = (df['GAV Timing Set Point Cyl #01']+df['GAV Timing Set Point Cyl #02']+
                                                        df['GAV Timing Set Point Cyl #03']+df['GAV Timing Set Point Cyl #04']+
                                                        df['GAV Timing Set Point Cyl #05']+df['GAV Timing Set Point Cyl #06'])/6
            df['Exh. valve opening angle Cyl AVG'] = (df['Exh. valve opening angle Cyl #01']+df['Exh. valve opening angle Cyl #02']+
                                                            df['Exh. valve opening angle Cyl #03']+df['Exh. valve opening angle Cyl #04']+
                                                            df['Exh. valve opening angle Cyl #05']+df['Exh. valve opening angle Cyl #06'])/6
            df['PFI Duration Set Point Cyl AVG'] = (df['PFI Duration Set Point Cyl #01']+df['PFI Duration Set Point Cyl #02']+
                                                        df['PFI Duration Set Point Cyl #03']+df['PFI Duration Set Point Cyl #04']+
                                                        df['PFI Duration Set Point Cyl #05']+df['PFI Duration Set Point Cyl #06'])/6
            df['PFI Timing Set Point Cyl AVG'] = (df['PFI Timing Set Point Cyl #01']+df['PFI Timing Set Point Cyl #02']+
                                                            df['PFI Timing Set Point Cyl #03']+df['PFI Timing Set Point Cyl #04']+
                                                            df['PFI Timing Set Point Cyl #05']+df['PFI Timing Set Point Cyl #06'])/6
            df['Exhaust Valve Closing Angle Setpoint Cyl AVG'] = (df['Exhaust Valve Closing Angle Setpoint Cyl #01']+df['Exhaust Valve Closing Angle Setpoint Cyl #02']+
                                                                        df['Exhaust Valve Closing Angle Setpoint Cyl #03']+df['Exhaust Valve Closing Angle Setpoint Cyl #04']+
                                                                        df['Exhaust Valve Closing Angle Setpoint Cyl #05']+df['Exhaust Valve Closing Angle Setpoint Cyl #06'])/6
            df['Fuel Rail Pressure_diff'] = df['Mean Fuel Rail Pressure (display)'] - df['Main Fuel Rail Pressure']
            #Pcomp
            df['Cyl. lub. distribution share below_PERC'] = (df['Cyl. lub. distribution share below piston']/df['Cyl. lub. distribution share into piston'])*100
            df['Cyl. lub. distribution share above_PERC'] = (df['Cyl. lub. distribution share above piston']/df['Cyl. lub. distribution share into piston'])*100
            df['GAV Timing Set Point Cyl AVG'] = (df['GAV Timing Set Point Cyl #01']+df['GAV Timing Set Point Cyl #02']+
                                                        df['GAV Timing Set Point Cyl #03']+df['GAV Timing Set Point Cyl #04']+
                                                        df['GAV Timing Set Point Cyl #05']+df['GAV Timing Set Point Cyl #06'])/6
            df['Exh. valve opening angle Cyl AVG'] = (df['Exh. valve opening angle Cyl #01']+df['Exh. valve opening angle Cyl #02']+
                                                            df['Exh. valve opening angle Cyl #03']+df['Exh. valve opening angle Cyl #04']+
                                                            df['Exh. valve opening angle Cyl #05']+df['Exh. valve opening angle Cyl #06'])/6
            df['PFI Duration Set Point Cyl AVG'] = (df['PFI Duration Set Point Cyl #01']+df['PFI Duration Set Point Cyl #02']+
                                                        df['PFI Duration Set Point Cyl #03']+df['PFI Duration Set Point Cyl #04']+
                                                        df['PFI Duration Set Point Cyl #05']+df['PFI Duration Set Point Cyl #06'])/6
            df['PFI Timing Set Point Cyl AVG'] = (df['PFI Timing Set Point Cyl #01']+df['PFI Timing Set Point Cyl #02']+
                                                            df['PFI Timing Set Point Cyl #03']+df['PFI Timing Set Point Cyl #04']+
                                                            df['PFI Timing Set Point Cyl #05']+df['PFI Timing Set Point Cyl #06'])/6
            df['Exhaust Valve Closing Angle Setpoint Cyl AVG'] = (df['Exhaust Valve Closing Angle Setpoint Cyl #01']+df['Exhaust Valve Closing Angle Setpoint Cyl #02']+
                                                                        df['Exhaust Valve Closing Angle Setpoint Cyl #03']+df['Exhaust Valve Closing Angle Setpoint Cyl #04']+
                                                                        df['Exhaust Valve Closing Angle Setpoint Cyl #05']+df['Exhaust Valve Closing Angle Setpoint Cyl #06'])/6
            df['Fuel Rail Pressure_diff'] = df['Mean Fuel Rail Pressure (display)'] - df['Main Fuel Rail Pressure']
            #Pscav
            df['Exh. valve opening angle Cyl AVG'] = (df['Exh. valve opening angle Cyl #01']+df['Exh. valve opening angle Cyl #02']+
                                                            df['Exh. valve opening angle Cyl #03']+df['Exh. valve opening angle Cyl #04']+
                                                            df['Exh. valve opening angle Cyl #05']+df['Exh. valve opening angle Cyl #06'])/6
            df['GAV Timing Set Point Cyl AVG'] = (df['GAV Timing Set Point Cyl #01']+df['GAV Timing Set Point Cyl #02']+
                                                        df['GAV Timing Set Point Cyl #03']+df['GAV Timing Set Point Cyl #04']+
                                                        df['GAV Timing Set Point Cyl #05']+df['GAV Timing Set Point Cyl #06'])/6
            df['Exhaust Valve Closing Angle Setpoint Cyl AVG'] = (df['Exhaust Valve Closing Angle Setpoint Cyl #01']+df['Exhaust Valve Closing Angle Setpoint Cyl #02']+
                                                                        df['Exhaust Valve Closing Angle Setpoint Cyl #03']+df['Exhaust Valve Closing Angle Setpoint Cyl #04']+
                                                                        df['Exhaust Valve Closing Angle Setpoint Cyl #05']+df['Exhaust Valve Closing Angle Setpoint Cyl #06'])/6
            df['PFI Timing Set Point Cyl AVG'] = (df['PFI Timing Set Point Cyl #01']+df['PFI Timing Set Point Cyl #02']+
                                                            df['PFI Timing Set Point Cyl #03']+df['PFI Timing Set Point Cyl #04']+
                                                            df['PFI Timing Set Point Cyl #05']+df['PFI Timing Set Point Cyl #06'])/6
            df['PFI Duration Set Point Cyl AVG'] = (df['PFI Duration Set Point Cyl #01']+df['PFI Duration Set Point Cyl #02']+
                                                        df['PFI Duration Set Point Cyl #03']+df['PFI Duration Set Point Cyl #04']+
                                                        df['PFI Duration Set Point Cyl #05']+df['PFI Duration Set Point Cyl #06'])/6
            df['Cyl. lub. distribution share below_PERC'] = (df['Cyl. lub. distribution share below piston']/df['Cyl. lub. distribution share into piston'])*100
            df['Cyl. lub. distribution share above_PERC'] = (df['Cyl. lub. distribution share above piston']/df['Cyl. lub. distribution share into piston'])*100
            df['Fuel Rail Pressure_diff'] = df['Mean Fuel Rail Pressure (display)'] - df['Main Fuel Rail Pressure']

            for efds in self.Efd_features:# ['Pscav','Pcomp','Pmax','Texh','Ntc','Ntc_Pscav','Pcomp_Pscav','PR']
                if efds == 'Pcomp_Pscav': #EFD1
                    print('Pcomp_Pscav')                  
                    model_inputs  = df[self.utility_dict[efds]['imp_feature']]
                    #Apply scaling here for new inputs
                    scaler_x = load("./utils/ML_model/features_scaler/"+efds+"_X.joblib")
                    model_inputs = pd.DataFrame(scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
                    '''scaler_x = scaler.fit(x_train)
                    from joblib import dump, load
                    dump(scaler_x, "./Scaling_models/Pcomp_Pscav.joblib")
                    scaler_x = []
                    scaler_x = load("./Scaling_models/Pcomp_Pscav.joblib")
                    x_train = pd.DataFrame(scaler_x.transform(x_train),columns=x_train.columns)'''
                    if efds in ann_models: #for ANN models
                        ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+efds+'.keras', compile=False)
                    elif efds in joblib_models:  
                        ml_model =  load("./utils/ML_model/model/"+efds+".joblib")#for classical ML models
                    y_pred = ml_model.predict(model_inputs)
                    scaler_y = load("./utils/ML_model/features_scaler/"+efds+"_Y.joblib")
                    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    # ml_output_dict['Ref_'+efds] =  y_pred.tolist()[0]
                    # print('index -',cyl_df[cyl_df['Engine load range']==loads].index)
                    # print('Output -',y_pred.tolist()[0])
                    cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
                elif efds == 'PR': #EFD2
                    print('PR')
                    model_inputs  = df[self.utility_dict[efds]['imp_feature']]
                    #Apply scaling here for new inputs
                    scaler_x = load("./utils/ML_model/features_scaler/"+efds+"_X.joblib")
                    model_inputs = pd.DataFrame(scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
                    '''scaler_x = scaler.fit(x_train)
                    from joblib import dump, load
                    dump(scaler_x, "./Scaling_models/Pcomp_Pscav.joblib")
                    scaler_x = []
                    scaler_x = load("./Scaling_models/Pcomp_Pscav.joblib")
                    x_train = pd.DataFrame(scaler_x.transform(x_train),columns=x_train.columns)'''
                    if efds in ann_models: #for ANN models
                        ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+efds+'.keras', compile=False)
                    elif efds in joblib_models:  
                        ml_model =  load("./utils/ML_model/model/"+efds+".joblib")#for classical ML models
                    y_pred = ml_model.predict(model_inputs)
                    scaler_y = load("./utils/ML_model/features_scaler/"+efds+"_Y.joblib")
                    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    # ml_output_dict['Ref_'+efds] =  y_pred.tolist()[0]
                    cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
                elif efds == 'Ntc_Pscav': #EFD3
                    print('Ntc_Pscav')                   
                    model_inputs  = df[self.utility_dict[efds]['imp_feature']]
                    #Apply scaling here for new inputs
                    scaler_x = load("./utils/ML_model/features_scaler/"+efds+"_X.joblib")
                    model_inputs = pd.DataFrame(scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
                    '''scaler_x = scaler.fit(x_train)
                    from joblib import dump, load
                    dump(scaler_x, "./Scaling_models/Pcomp_Pscav.joblib")
                    scaler_x = []
                    scaler_x = load("./Scaling_models/Pcomp_Pscav.joblib")
                    x_train = pd.DataFrame(scaler_x.transform(x_train),columns=x_train.columns)'''
                    if efds in ann_models: #for ANN models
                        ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+efds+'.keras', compile=False)
                    elif efds in joblib_models:  
                        ml_model =  load("./utils/ML_model/model/"+efds+".joblib")#for classical ML models
                    y_pred = ml_model.predict(model_inputs)
                    scaler_y = load("./utils/ML_model/features_scaler/"+efds+"_Y.joblib")
                    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    # ml_output_dict['Ref_'+efds] =  y_pred.tolist()[0]
                    cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
                #till here model retunned    
                elif efds == 'Pmax': #EFD4
                    print('Pmax')                   
                    model_inputs  = df[self.utility_dict[efds]['imp_feature']]
                    #Apply scaling here for new inputs
                    scaler_x = load("./utils/ML_model/features_scaler/"+efds+"_X.joblib")
                    model_inputs = pd.DataFrame(scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
                    '''scaler_x = scaler.fit(x_train)
                    from joblib import dump, load
                    dump(scaler_x, "./Scaling_models/Pcomp_Pscav.joblib")
                    scaler_x = []
                    scaler_x = load("./Scaling_models/Pcomp_Pscav.joblib")
                    x_train = pd.DataFrame(scaler_x.transform(x_train),columns=x_train.columns)'''
                    if efds in ann_models: #for ANN models
                        ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+efds+'.keras', compile=False)
                    elif efds in joblib_models:  
                        ml_model =  load("./utils/ML_model/model/"+efds+".joblib")#for classical ML models
                    y_pred = ml_model.predict(model_inputs)
                    scaler_y = load("./utils/ML_model/features_scaler/"+efds+"_Y.joblib")
                    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    # ml_output_dict['Ref_'+efds] =  y_pred.tolist()[0]
                    cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
                elif efds == 'Texh': #EFD5
                    print('Texh')                    
                    model_inputs  = df[self.utility_dict[efds]['imp_feature']]
                    #Apply scaling here for new inputs
                    scaler_x = load("./utils/ML_model/features_scaler/"+efds+"_X.joblib")
                    model_inputs = pd.DataFrame(scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
                    '''scaler_x = scaler.fit(x_train)
                    from joblib import dump, load
                    dump(scaler_x, "./Scaling_models/Pcomp_Pscav.joblib")
                    scaler_x = []
                    scaler_x = load("./Scaling_models/Pcomp_Pscav.joblib")
                    x_train = pd.DataFrame(scaler_x.transform(x_train),columns=x_train.columns)'''
                    if efds in ann_models: #for ANN models
                        ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+efds+'.keras', compile=False)
                    elif efds in joblib_models:  
                        ml_model =  load("./utils/ML_model/model/"+efds+".joblib")#for classical ML models
                    y_pred = ml_model.predict(model_inputs)
                    scaler_y = load("./utils/ML_model/features_scaler/"+efds+"_Y.joblib")
                    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    # ml_output_dict['Ref_'+efds] =  y_pred.tolist()[0]
                    cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
                elif efds == 'Ntc': #EFD6
                    print('Ntc')                   
                    model_inputs  = df[self.utility_dict[efds]['imp_feature']]
                    #Apply scaling here for new inputs
                    scaler_x = load("./utils/ML_model/features_scaler/"+efds+"_X.joblib")
                    model_inputs = pd.DataFrame(scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
                    '''scaler_x = scaler.fit(x_train)
                    from joblib import dump, load
                    dump(scaler_x, "./Scaling_models/Pcomp_Pscav.joblib")
                    scaler_x = []
                    scaler_x = load("./Scaling_models/Pcomp_Pscav.joblib")
                    x_train = pd.DataFrame(scaler_x.transform(x_train),columns=x_train.columns)'''
                    if efds in ann_models: #for ANN models
                        ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+efds+'.keras', compile=False)
                    elif efds in joblib_models:  
                        ml_model =  load("./utils/ML_model/model/"+efds+".joblib")#for classical ML models
                    y_pred = ml_model.predict(model_inputs)
                    scaler_y = load("./utils/ML_model/features_scaler/"+efds+"_Y.joblib")
                    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    # ml_output_dict['Ref_'+efds] =  y_pred.tolist()[0]
                    cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
                elif efds == 'Pcomp': #EFD7
                    print('Pcomp')                    
                    model_inputs  = df[self.utility_dict[efds]['imp_feature']]
                    #Applying scaling to new inputs
                    scaler_x = load("./utils/ML_model/features_scaler/"+efds+"_X.joblib")
                    model_inputs = pd.DataFrame(scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
                    '''scaler_x = scaler.fit(x_train)
                    from joblib import dump, load
                    dump(scaler_x, "./Scaling_models/Pcomp_Pscav.joblib")
                    scaler_x = []
                    scaler_x = load("./Scaling_models/Pcomp_Pscav.joblib")
                    x_train = pd.DataFrame(scaler_x.transform(x_train),columns=x_train.columns)'''
                    if efds in ann_models: #for ANN models
                        ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+efds+'.keras', compile=False)
                    elif efds in joblib_models:  
                        ml_model =  load("./utils/ML_model/model/"+efds+".joblib")#for classical ML models
                    y_pred = ml_model.predict(model_inputs)
                    scaler_y = load("./utils/ML_model/features_scaler/"+efds+"_Y.joblib")
                    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    # ml_output_dict['Ref_'+efds] =  y_pred.tolist()[0] 
                    cyl_df['Ref_'+efds]  =  [re[0] for re in y_pred.tolist()]
                elif efds == 'Pscav': #EFD8
                    print('Pscav')                   
                    model_inputs  = df[self.utility_dict[efds]['imp_feature']]
                    print('model_inputs',model_inputs)
                    print(type(model_inputs))
                    #Applying scaling to new inputs
                    scaler_x = load("./utils/ML_model/features_scaler/"+efds+"_X.joblib")
                    model_inputs = pd.DataFrame(scaler_x.transform(np.asarray(model_inputs)),columns=self.utility_dict[efds]['imp_feature'])
                    '''scaler_x = scaler.fit(x_train)
                    from joblib import dump, load
                    dump(scaler_x, "./Scaling_models/Pcomp_Pscav.joblib")
                    scaler_x = []
                    scaler_x = load("./Scaling_models/Pcomp_Pscav.joblib")
                    x_train = pd.DataFrame(scaler_x.transform(x_train),columns=x_train.columns)'''
                    if efds in ann_models: #for ANN models
                        ml_model = tf.keras.models.load_model('./utils/ML_model/model/'+efds+'.keras', compile=False)
                    elif efds in joblib_models:  
                        ml_model =  load("./utils/ML_model/model/"+efds+".joblib")#for classical ML models
                    y_pred = ml_model.predict(model_inputs)
                    scaler_y = load("./utils/ML_model/features_scaler/"+efds+"_Y.joblib")
                    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1))
                    # ml_output_dict['Ref_'+efds] =  y_pred.tolist()[0] 
                    # print('index -',cyl_df[cyl_df['Engine load range']==loads].index)
                    # print('Output -',y_pred.tolist()[0])
                    print('cyl_df',cyl_df)
                    cyl_df['Ref_'+efds] =  [re[0] for re in y_pred.tolist()]
                      
                
            cyl_df.to_csv(self.ml_res+'ENG_{}_TS_ML_res_Cyl_{}.csv'.format(self.engine_number,cyl),index=False)   
        print('Ml predictions completed!!!')   
        # full_data2.reset_index(drop=True,inplace=True) 
        # return full_data1 , full_data2, load_delta, mean_load
