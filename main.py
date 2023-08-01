from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta

from config import ACCESS_TOKEN_EXPIRE_MINUTES
from database_conn import database
from helper import authenticate_user, create_access_token, get_current_active_user
from custom_data_type import Token, User

import pandas as pd
from io import StringIO
import json
import numpy as np

import torch
from AE_ import data_load_preprocess, Transform_data, AE, recreation_loss



import warnings
warnings.filterwarnings('ignore')


#time series model
from TS_model import pdm_ts_model
#ml model
from ML_model import pdm_ml_model
#fault mapping
from Fault_Mapping import Faults_Mapping


app = FastAPI()


# #model weight loading
# model = torch.load('utils\\AC\\model\\model_0.0175.pt', map_location=torch.device('cpu'))
# model.eval()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def database_connect():
    await database.connect()
@app.on_event("shutdown")
async def database_disconnect():
    await database.disconnect()


#test endpoint
@app.get("/test")
async def test(User = Depends(get_current_active_user)):
    return {"data": User.dict()}

# Authentication
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


async def preprocess_dataset(data):
    pass


def data_preprocess(raw_data_path):
    df = pd.read_csv(raw_data_path, index_col='signaldate')
    df = df[(df['Estimated engine load'] >= 30) & (df['Estimated engine load'] <= 100)]
    #last 336 datapoints
    df = df.iloc[-336:,:]
    return df

@app.post("/forecast-14days")
async def forecast_14days(current_user: User = Depends(get_current_active_user)):
    data = data_preprocess('./test_data/workflow_test_data.csv')
    mapping_loc = './utils/Fault_Matrix/results/'
    # #TS_model calling
    ts_features_file = pd.read_csv('./utils/TS_model/feature_list.csv')
    ts_model = './utils/TS_model/TS_model.h5' #load ts model here
    x_scale = './utils/TS_model/TS_X.joblib'#load X_scaler model here
    y_scale = './utils/TS_model/TS_Y.joblib'#load Y_scaler model here
    engine_normalized = False
    engine_number = '2'
    ts_res_loc = './utils/TS_model/results'
    TS = pdm_ts_model(data, ts_features_file, ts_model,x_scale,y_scale,engine_normalized,engine_number,ts_res_loc)
    TS_result = TS.Timeseries()


    #AUTOENCODE
    #data preprocessing
    # df = data_load_preprocess('test_data\\engine2_imputed.csv')
    # df_norm_obj_test = Transform_data(df)
    # df_norm_test = df_norm_obj_test.normalize()
    # #final data for model
    # df_tensor_test = torch.tensor(df_norm_test.values, dtype=torch.float32)
    # recreated_df_test = model(df_tensor_test).cpu().detach().numpy()
    # base_data_test = df_tensor_test.cpu().detach().numpy()
    # faulty_date = recreation_loss(base_data_test, recreated_df_test)
    # if len(faulty_date) > 0:
    #     data_from_AE = df.iloc[faulty_date,:]
    # else:
    #     data_from_AE = df

    #ML_model calling
    Efd_features = ['Pscav','Pcomp','Pmax','Texh','Ntc','Ntc_Pscav','Pcomp_Pscav','PR']
    engine_normalized = False
    ml_res_loc = './utils/TS_model/results'
    ML = pdm_ml_model(data,Efd_features,engine_normalized,ts_res_loc,engine_number,ml_res_loc)
    ML_result = ML.ML_models()

    #Fault mapping
    final_indx = [pd.Timestamp(TS_result)+pd.Timedelta(tim,'h') for tim in range(1,TS.forecast_horizon+1)] #for getting timestamps for forecast period, here delta is hourly based
    fault_mat_loc = './utils/Fault_Matrix/Fault_matrix.xlsx'
    p=.2 #weight value for kpi calculations
    mapping_loc = './utils/Fault_Matrix/results/'
    output_dict = {}
    for i in range(1,ML.cyl_count+1):#------------ML.cyl_count+1
        ml_ress = pd.read_csv(ml_res_loc+'ENG_2_TS_ML_res_Cyl_{}.csv'.format(str(i)),index_col=False)
        ff = Faults_Mapping(ml_ress,fault_mat_loc,Efd_features,p)
        ff1,fault_ids = ff.Mapping()
        ml_ress = pd.concat([ml_ress,ff1[fault_ids]],axis=1)
        ml_ress['Date Time'] = final_indx
        #for ordering columns
        ml_ress = ml_ress[['Date Time','Estimated engine load','matched_load','matched_date','deltas','TS_Pcomp','TS_Pscav','TS_Texh','TS_Ntc','TS_Pmax','TS_PR','TS_Ntc_Pscav',
                        'TS_Pcomp_Pscav','Ref_Pcomp','Ref_Pscav','Ref_Texh','Ref_Ntc','Ref_Pmax','Ref_PR','Ref_Ntc_Pscav','Ref_Pcomp_Pscav',
                        'InjSysFault','StaInjLate','StaInjEarly','ExhValvLeak','BloCombChabr','ExhValEarOpn','ExhValLatOpn','ExhValEarlClos','ExhValLatClos']]
        ml_ress.to_excel(mapping_loc+'mapping_res_cyl{}.xlsx'.format(i),index=False)
        output_dict['Cyl_'+str(i)] = ml_ress.to_dict(orient='list')

    return output_dict    
    
    
























































































# @app.post("/recommend_vendor")
# async def fetch_data(userinput: UserInput, current_user: User = Depends(get_current_active_user)):
#     #print(userinput.dict())
#     user_inp =  userinput.dict()
#     print(user_inp)
    # try:
    #     data =  query_database(query)
    # except Exception as e:
    #     try:
    #         data = read_data_from_blob("vendor_dataset.csv")
    #     except Exception as e:
    #         print(e)
    #         raise HTTPException(status_code=404, detail='No data found for the given input')

