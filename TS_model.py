
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed
# from joblib import load

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Reshape
from tensorflow.keras.callbacks import ReduceLROnPlateau

class pdm_ts_model():
 
    cyl_count = 6 #may change depending upon vessel
    max_load = 84.19076333333332 #max engine load from 1 yr data for normalization
    # look_back = 2880 #30 days
    # forecast_horizon = 1440 #15 days
    look_back = 336
    forecast_horizon = 336

    # def __init__(self,new_data,ts_features_file,ts_model,x_scale,y_scale,engine_normalized:bool,engine_number,res_loc): #raw_data,anomaly_path,Efd_features,feature,input_data
    #     self.new_data = new_data #new data after agg & imputaion
    #     self.ts_features_file = ts_features_file #input features list for TS
    #     self.ts_model = tf.keras.models.load_model(ts_model, compile=False) #saved ts model with path
    #     # self.Efd_features = Efd_features #list of all EFD features for iter on ml models ['Pscav','Pcomp','Pmax','Texh','Ntc','Ntc/Pscav','Pcomp/Pscav','PR']
    #     self.scaler_inp_train_m = load(x_scale) #saved inputs scaler model with path
    #     self.scaler_out_train_m = load(y_scale) #saved output scaler model with path
    #     self.engine_normalized = engine_normalized #True or False for applying engine based normalization
    #     self.engine_number = engine_number # '1' or '2' for corr engine
    #     self.res_loc = res_loc #loc where TS results save 'E:/python_codes1/ML_reference/1hr/TS_models_engineinout/TS_res/'
    
    # def Timeseries(self):
    #     #part for test data
    #     # self.feature = 'EFD'
    #     df = self.new_data 
    #     df = df[(df['Estimated engine load']>=30)&(df['Estimated engine load']<=100)]    
    #     #Normalizing based on engine load
    #     if self.engine_normalized == True:
    #         max_load = self.max_load
    #         print(max_load)
    #         df['Estimated engine load'] = df['Estimated engine load']/max_load
    #         for col in df.columns:
    #             if col == 'Estimated engine load':
    #                 df[col] = df['Estimated engine load']
    #             else:
    #                 df[col] = df[col]*df['Estimated engine load']     
                
    #     df = df.iloc[-self.look_back:,] #taking last look_back period from new dataset
    #     print(df.index.min())
    #     print(df.index.max())
    #     print(df.shape)


    #     #setting input features for TS
    #     # fg = self.ts_features_file
    #     # all_feat_list = list(fg['a'])
    #     # add_cols = all_feat_list #this contains all input features list
    #     #predicting for individual cylinders
    #     ts_res_cyl = {}
    #     df_pred_load_wise = pd.DataFrame()
    #     for cyl in range(1,self.cyl_count+1):
    #         df['Exh. valve opening angle Cyl AVG'] = df['Exh. valve opening angle Cyl #0'+str(cyl)]
    #         df['GAV Timing Set Point Cyl AVG'] = df['GAV Timing Set Point Cyl #0'+str(cyl)]
    #         df['Exhaust Valve Closing Angle Setpoint Cyl AVG'] = df['Exhaust Valve Closing Angle Setpoint Cyl #0'+str(cyl)]
    #         df['PFI Timing Set Point Cyl AVG'] = df['PFI Timing Set Point Cyl #0'+str(cyl)]
    #         df['PFI Duration Set Point Cyl AVG'] = df['PFI Duration Set Point Cyl #0'+str(cyl)]
    #         df['Cyl. lub. distribution share below_PERC'] = (df['Cyl. lub. distribution share below piston']/df['Cyl. lub. distribution share into piston'])*100
    #         df['Cyl. lub. distribution share above_PERC'] = (df['Cyl. lub. distribution share above piston']/df['Cyl. lub. distribution share into piston'])*100
    #         df['Fuel Rail Pressure_diff'] = df['Mean Fuel Rail Pressure (display)'] - df['Main Fuel Rail Pressure']
    #         df['Firing Pr. Balancing Injection Offset Cyl_AVG'] = df['Firing Pr. Balancing Injection Offset Cyl #0'+str(cyl)]
    #         df['Fuel Pressure Actuator Setpoint_AVG'] = (df['Fuel Pressure Actuator Setpoint 1']+df['Fuel Pressure Actuator Setpoint 2']+df['Fuel Pressure Actuator Setpoint 3'])/3
    #         df['Fuel Pump Setpoint_AVG'] = (df['Fuel Pump Setpoint Master Controller']+df['Fuel Pump Setpoint Slave Controller'])/2
    #         df['Lubrication Oil Feed Rate Cyl AVG'] = df['Lubrication Oil Feed Rate Cyl #0'+str(cyl)]
    #         df['Lubrication Deadtime Feedback Cyl AVG'] = df['Lubrication Deadtime Feedback Cyl #0'+str(cyl)]
    #         df['Start of Injection Cyl_AVG'] = df['Start of Injection Cyl #0'+str(cyl)]
    #         df['Pilot Fuel Pressure diff'] = df['Pilot Fuel Pressure A']-df['Pilot Fuel Pressure B']
    #         df['Scavenge Air Temp. Piston Underside Cyl_AVG'] = df['Scavenge Air Temp. Piston Underside Cyl #0'+str(cyl)+'.1']
    #         #scaling X
    #         model_inputs = df[list(self.scaler_inp_train_m.feature_names_in_)]
    #         scaled_inp_test = self.scaler_inp_train_m.transform(model_inputs)
 
    #         X_new = scaled_inp_test.reshape((1, scaled_inp_test.shape[0], scaled_inp_test.shape[1]))
    #         predictons = self.ts_model.predict(X_new)
    #         #inverse scaling y
    #         # scaled_out_train = load(self.scaler_out_train_m)#load y model
    #         y_pred_real = self.scaler_out_train_m.inverse_transform(predictons[-1].reshape(-1,len(self.scaler_out_train_m.feature_names_in_)))
    #         df_pred = pd.DataFrame(y_pred_real, columns=['TS_Pcomp', 'TS_Pscav','TS_Texh', 'TS_Ntc','TS_Pmax', 'TS_PR','TS_Ntc_Pscav','TS_Pcomp_Pscav','Estimated engine load'])
    #         # ts_res_cyl['Cylinder_'+str(cyl)+'_ts_res'] = df_pred
    #         df_pred.to_csv(self.res_loc+'ENG_{}_TS_res_Cyl_{}.csv'.format(self.engine_number,cyl), index=False)
    #         print('Cylinder_'+str(cyl)+' timeseries prediction completed!!!')
    #     # return ts_res_cyl  

    #     return df.index.max()
    def __init__(self,new_data,ts_features_file,engine_normalized:bool,engine_number,res_loc,final_feats): #raw_data,anomaly_path,Efd_features,feature,input_data
        self.new_data = new_data #new data after agg & imputaion
        self.ts_features_file = ts_features_file #input features list for TS
        # self.ts_model = tf.keras.models.load_model(ts_model) #saved ts model with path
        # self.Efd_features = Efd_features #list of all EFD features for iter on ml models ['Pscav','Pcomp','Pmax','Texh','Ntc','Ntc/Pscav','Pcomp/Pscav','PR']
        # self.scaler_inp_train_m = load(x_scale) #saved inputs scaler model with path
        # self.scaler_out_train_m = load(y_scale) #saved output scaler model with path
        self.engine_normalized = engine_normalized #True or False for applying engine based normalization
        self.engine_number = engine_number # '1' or '2' for corr engine
        self.res_loc = res_loc #loc where TS results save 'E:/python_codes1/ML_reference/1hr/TS_models_engineinout/TS_res/'
        self.final_feats = final_feats #final_feats excel file path loc for input important features for training

    def Timeseries(self):
        #part for test data
        # self.feature = 'EFD'
        df = self.new_data 
        df = df[(df['Estimated engine load']>=30)&(df['Estimated engine load']<=100)]    
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
                
        # df = df.iloc[-self.look_back:,] #taking last look_back period from new dataset

        #test
        #Training part
        # Here we assuming input data is 1hr agg, imputed and index as timestamps otherwise use below two lines & these lines will come above sec of the code
        # df.index = pd.to_datetime(df.index)
        # df = df.resample('1H').mean()
        # df = df.ffill().bfill()
        df['PR'] = df['Firing Pressure Average'] - df['Compression Pressure Average']
        df['Ntc_Pscav'] = df['Turbocharger 1 speed'] / df['Scav. Air Press. Mean Value']
        df['Pcomp_Pscav'] = df['Compression Pressure Average'] / df['Scav. Air Press. Mean Value']
        df_test = df.iloc[-self.look_back:,]
        print(df_test.index.min())
        print(df_test.index.max())
        print(df_test.shape)
        # df_test['PR'] = df_test['Firing Pressure Average'] - df_test['Compression Pressure Average']
        # df_test['Ntc_Pscav'] = df_test['Turbocharger 1 speed'] / df_test['Scav. Air Press. Mean Value']
        # df_test['Pcomp_Pscav'] = df_test['Compression Pressure Average'] / df_test['Scav. Air Press. Mean Value']
        df = df.iloc[:-self.look_back,]
        calc_features = ['Exh. valve opening angle Cyl AVG','GAV Timing Set Point Cyl AVG','Exhaust Valve Closing Angle Setpoint Cyl AVG','PFI Timing Set Point Cyl AVG','PFI Duration Set Point Cyl AVG',
        'Firing Pr. Balancing Injection Offset Cyl_AVG','Lubrication Oil Feed Rate Cyl AVG','Lubrication Deadtime Feedback Cyl AVG','Start of Injection Cyl_AVG','Scavenge Air Temp. Piston Underside Cyl_AVG']
        for cal_f in calc_features:
            # avg_value = np.array()
            df[cal_f] = np.zeros(len(df))
            for cal_n in range(1,self.cyl_count+1):
                if cal_f == 'Exh. valve opening angle Cyl AVG':
                    df[cal_f]+=df['Exh. valve opening angle Cyl #0'+str(cal_n)]
                elif cal_f == 'GAV Timing Set Point Cyl AVG':
                    df[cal_f]+=df['GAV Timing Set Point Cyl #0'+str(cal_n)]   
                elif cal_f == 'Exhaust Valve Closing Angle Setpoint Cyl AVG':
                    df[cal_f]+=df['Exhaust Valve Closing Angle Setpoint Cyl #0'+str(cal_n)]   
                elif cal_f == 'PFI Timing Set Point Cyl AVG':
                    df[cal_f]+=df['PFI Timing Set Point Cyl #0'+str(cal_n)]   
                elif cal_f == 'PFI Duration Set Point Cyl AVG':
                    df[cal_f]+=df['PFI Duration Set Point Cyl #0'+str(cal_n)]       
                elif cal_f == 'Firing Pr. Balancing Injection Offset Cyl_AVG':
                    df[cal_f]+=df['Firing Pr. Balancing Injection Offset Cyl #0'+str(cal_n)]     
                elif cal_f == 'Lubrication Oil Feed Rate Cyl AVG':
                    df[cal_f]+=df['Lubrication Oil Feed Rate Cyl #0'+str(cal_n)]  
                elif cal_f == 'Lubrication Deadtime Feedback Cyl AVG':
                    df[cal_f]+=df['Lubrication Deadtime Feedback Cyl #0'+str(cal_n)]       
                elif cal_f == 'Start of Injection Cyl_AVG':
                    df[cal_f]+=df['Start of Injection Cyl #0'+str(cal_n)]    
                elif cal_f == 'Scavenge Air Temp. Piston Underside Cyl_AVG':
                    df[cal_f]+=df['Scavenge Air Temp. Piston Underside Cyl #0'+str(cal_n)+'.1']         
            df[cal_f] = df[cal_f]/cyl_count        
              
        df['Cyl. lub. distribution share below_PERC'] = (df['Cyl. lub. distribution share below piston']/df['Cyl. lub. distribution share into piston'])*100
        df['Cyl. lub. distribution share above_PERC'] = (df['Cyl. lub. distribution share above piston']/df['Cyl. lub. distribution share into piston'])*100
        df['Fuel Rail Pressure_diff'] = df['Mean Fuel Rail Pressure (display)'] - df['Main Fuel Rail Pressure']
        df['Fuel Pressure Actuator Setpoint_AVG'] = (df['Fuel Pressure Actuator Setpoint 1']+df['Fuel Pressure Actuator Setpoint 2']+df['Fuel Pressure Actuator Setpoint 3'])/3
        df['Fuel Pump Setpoint_AVG'] = (df['Fuel Pump Setpoint Master Controller']+df['Fuel Pump Setpoint Slave Controller'])/2
        df['Pilot Fuel Pressure diff'] = df['Pilot Fuel Pressure A']-df['Pilot Fuel Pressure B']

        # featt = pd.read_csv(r'C:\Users\detecttrainadmin\Desktop\CBM\LSTM_cusload\final_feats.csv') #this list need to be add as an argument
        featt = pd.read_csv(self.final_feats)
        df = df[list(featt['a'])+['Compression Pressure Average', 'Scav. Air Press. Mean Value','Exhaust Gas Average Temperature', 'Turbocharger 1 speed',
        'Firing Pressure Average', 'PR','Ntc_Pscav','Pcomp_Pscav']]
        Pscav_inp = df
        Pscav_out = df[['Compression Pressure Average', 'Scav. Air Press. Mean Value',
                     'Exhaust Gas Average Temperature', 'Turbocharger 1 speed',
                     'Firing Pressure Average', 'PR',
                     'Ntc_Pscav','Pcomp_Pscav','Estimated engine load']]
        Pscav_inp.reset_index(drop=True,inplace=True)
        Pscav_out.reset_index(drop=True,inplace=True)
        def pipeline(feature_range,Pscav_inp,Pscav_out,n_steps_in,n_steps_out):
        #minmax scaler for output - final
            # train_size = int(len(Pscav_inp) * 0.70)
            # test_size = len(Pscav_inp) - train_size
            scaler_inp_train_m = MinMaxScaler(feature_range=feature_range)
            scaler_out_train_m = MinMaxScaler(feature_range=feature_range)
            scaled_inp_train = scaler_inp_train_m.fit_transform(Pscav_inp)
            # scaled_inp_train = pd.DataFrame(scaled_inp_train,columns = list(Pscav_inp.columns))
            scaled_out_train = scaler_out_train_m.fit_transform(Pscav_out)
            # scaled_out_train = pd.DataFrame(scaled_out_train,columns=list(Pscav_out.columns))

            #prepare dataset for pscav input and output for multivariate multistep lstm model
            def split_sequences(sequence_inp, sequence_out, n_steps_in, n_steps_out):
                X, y = list(), list()
                for i in range(len(sequence_inp)):
                    # find the end of this pattern
                    end_ix = i + n_steps_in
                    out_end_ix = end_ix + n_steps_out
                    # check if we are beyond the sequence
                    if out_end_ix > len(sequence_inp):
                        break
                    # gather input and output parts of the pattern
                    seq_x, seq_y = sequence_inp[i:end_ix], sequence_out[end_ix:out_end_ix]
                    pos = np.array(range(1,n_steps_in+1))/n_steps_in
                    seq_x = np.append(seq_x,pos.reshape(-1,1),axis=1)
                    X.append(seq_x)
                    y.append(seq_y)
                return np.array(X), np.array(y)

            X_train, y_train = split_sequences(scaled_inp_train, scaled_out_train, n_steps_in, n_steps_out)
            
            # X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
            # #creating valid set
            # valid_size = int(X_train.shape[0]*0.1)
            # X_train = X_train[:len(X_train)-valid_size]
            # X_valid = X_train[len(X_train)-valid_size:len(X_train)+valid_size]

            # y_train = y_train[:len(y_train)-valid_size]
            # y_valid = y_train[len(y_train)-valid_size:len(y_train)+valid_size]

            # X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

            # return X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_out_train_m
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
            return X_train, y_train, scaler_inp_train_m, scaler_out_train_m

        dict_scale = {'0':(0,1),'1':(-1,1)}
        n_steps_in = self.look_back #trial.suggest_int('steps_in', 336, 600)
        n_steps_out = self.forecast_horizon

        X_train, y_train, scaler_inp_test_m, scaler_out_test_m = pipeline(dict_scale['0'],Pscav_inp,Pscav_out,n_steps_in,n_steps_out)
        class AttentionLayer(tf.keras.layers.Layer):
                def __init__(self,name=None):
                    super(AttentionLayer, self).__init__(name=name)

                def build(self, input_shape):
                    self.W = self.add_weight(shape=(input_shape[-1], 1), initializer="random_normal", trainable=True)
                    super(AttentionLayer, self).build(input_shape)

                def call(self, inputs):
                    attention_weights = tf.nn.softmax(tf.matmul(inputs, self.W), axis=1)
                    weighted_inputs = inputs * attention_weights
                    return tf.reduce_sum(weighted_inputs, axis=1)

        inputs = tf.keras.Input(shape=(n_steps_in, X_train.shape[2]))
        x = LSTM(335, return_sequences=True)(inputs)
        x = LSTM(188, return_sequences=True, dropout=.06586127019804049)(x)
        x = LSTM(172, return_sequences=True, dropout=.1494075127203595)(x) 
        x = AttentionLayer(name='AttentionLayer')(x)    
        out = Dense(64)(x)
        out = Dense(128)(out)
        out = Dense(64)(out)
        out = Dense(n_steps_out*y_train.shape[-1])(out)
        out = Reshape((n_steps_out,y_train.shape[-1]))(out)   
        model = Model(inputs = inputs, outputs = out)
        print(model.summary())

        optimizer = keras.optimizers.Adam(learning_rate=.02977616607800996)
        factors = .7177073456775145
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_delta=0.0005, factor=factors,
                                    patience=3, mode = 'min', min_lr=1e-7)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.0005, mode='min',patience=10 ,restore_best_weights=True)

        model.compile(optimizer=optimizer, loss='mse', metrics=['RootMeanSquaredError'])
        history = model.fit(X_train, y_train, batch_size=176, epochs=216, validation_split=0.2, shuffle=False, callbacks=[early_stop,reduce_lr],verbose=1)
        # 
        #setting input features for TS
        # fg = self.ts_features_file
        # all_feat_list = list(fg['a'])
        # add_cols = all_feat_list #this contains all input features list
        #predicting for individual cylinders
        ts_res_cyl = {}
        df_pred_load_wise = pd.DataFrame()
        for cyl in range(1,self.cyl_count+1):
            df_test['Exh. valve opening angle Cyl AVG'] = df_test['Exh. valve opening angle Cyl #0'+str(cyl)]
            df_test['GAV Timing Set Point Cyl AVG'] = df_test['GAV Timing Set Point Cyl #0'+str(cyl)]
            df_test['Exhaust Valve Closing Angle Setpoint Cyl AVG'] = df_test['Exhaust Valve Closing Angle Setpoint Cyl #0'+str(cyl)]
            df_test['PFI Timing Set Point Cyl AVG'] = df_test['PFI Timing Set Point Cyl #0'+str(cyl)]
            df_test['PFI Duration Set Point Cyl AVG'] = df_test['PFI Duration Set Point Cyl #0'+str(cyl)]
            df_test['Cyl. lub. distribution share below_PERC'] = (df_test['Cyl. lub. distribution share below piston']/df_test['Cyl. lub. distribution share into piston'])*100
            df_test['Cyl. lub. distribution share above_PERC'] = (df_test['Cyl. lub. distribution share above piston']/df_test['Cyl. lub. distribution share into piston'])*100
            df_test['Fuel Rail Pressure_diff'] = df_test['Mean Fuel Rail Pressure (display)'] - df_test['Main Fuel Rail Pressure']
            df_test['Firing Pr. Balancing Injection Offset Cyl_AVG'] = df_test['Firing Pr. Balancing Injection Offset Cyl #0'+str(cyl)]
            df_test['Fuel Pressure Actuator Setpoint_AVG'] = (df_test['Fuel Pressure Actuator Setpoint 1']+df_test['Fuel Pressure Actuator Setpoint 2']+df_test['Fuel Pressure Actuator Setpoint 3'])/3
            df_test['Fuel Pump Setpoint_AVG'] = (df_test['Fuel Pump Setpoint Master Controller']+df_test['Fuel Pump Setpoint Slave Controller'])/2
            df_test['Lubrication Oil Feed Rate Cyl AVG'] = df_test['Lubrication Oil Feed Rate Cyl #0'+str(cyl)]
            df_test['Lubrication Deadtime Feedback Cyl AVG'] = df_test['Lubrication Deadtime Feedback Cyl #0'+str(cyl)]
            df_test['Start of Injection Cyl_AVG'] = df_test['Start of Injection Cyl #0'+str(cyl)]
            df_test['Pilot Fuel Pressure diff'] = df_test['Pilot Fuel Pressure A']-df_test['Pilot Fuel Pressure B']
            df_test['Scavenge Air Temp. Piston Underside Cyl_AVG'] = df_test['Scavenge Air Temp. Piston Underside Cyl #0'+str(cyl)+'.1']
            #scaling X
            # model_inputs = df_test[list(self.scaler_inp_train_m.feature_names_in_)]
            model_inputs = df_test[list(scaler_inp_test_m.feature_names_in_)]
            scaled_inp_test = scaler_inp_test_m.transform(model_inputs)
            #making new test data into 3D
            X_new = scaled_inp_test.reshape((1, scaled_inp_test.shape[0], scaled_inp_test.shape[1]))
            predictons = model.predict(X_new)
            #inverse scaling y
            # scaled_out_train = load(self.scaler_out_train_m)#load y model
            y_pred_real = scaler_out_test_m.inverse_transform(predictons[-1].reshape(-1,len(self.scaler_out_train_m.feature_names_in_)))
            df_test_pred = pd.DataFrame(y_pred_real, columns=['TS_Pcomp', 'TS_Pscav','TS_Texh', 'TS_Ntc','TS_Pmax', 'TS_PR','TS_Ntc_Pscav','TS_Pcomp_Pscav','Estimated engine load'])
            # ts_res_cyl['Cylinder_'+str(cyl)+'_ts_res'] = df_test_pred
            df_test_pred.to_csv(self.res_loc+'ENG_{}_TS_res_Cyl_{}.csv'.format(self.engine_number,cyl))
            print('Cylinder_'+str(cyl)+' timeseries prediction completed!!!')
        return df.index.max()    