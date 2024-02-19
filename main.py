from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta

from config import ACCESS_TOKEN_EXPIRE_MINUTES
from database_conn import database
from helper import authenticate_user, create_access_token, get_current_active_user
from custom_data_type import Token, User, pdm_inputs

import pandas as pd
from azure.storage.blob import ContainerClient
from io import StringIO
import json
import numpy as np
import pickle
from datetime import datetime
import time
import os

import torch
# from AE_ import data_load_preprocess, Transform_data, AE, recreation_loss



import warnings
warnings.filterwarnings('ignore')

#data scraping
# from web_scraping import wingd_scraper
#data collection
from data_collection_api import api_data_collection
#time series model
from TS_model import pdm_ts_model
#AE model
from Auto_encoder_model import pdm_AE_model
#ml model
from ML_model import pdm_ml_model
#fault mapping
from Fault_Mapping import Faults_Mapping
import yaml

app = FastAPI()

with open('cbm_yaml.yml','r') as file:
    utility_dict = yaml.safe_load(file)   


#data collection part
with open('data_collect_dict.pickle','rb') as file2:
    dict1 = pickle.load(file2)


# #model weight loading
# model = AE()
# model.load_state_dict(torch.load(utility_dict['AE_path'], map_location=torch.device('cpu')))
# model.eval()
#TS model loading
ts_model =  utility_dict['TS_model_path']#load ts model here
x_scale = utility_dict['TS_scale_x']#load X_scaler model here
y_scale = utility_dict['TS_scale_y']#load Y_scaler model here
engine_number = utility_dict['engine_number']
# ts_res_loc = utility_dict['ts_res_loc']
ts_res_loc = utility_dict['forecast_data_path']+utility_dict['Vessel_name']+'/Results/TS/'
TS = pdm_ts_model(ts_model,x_scale,y_scale,engine_number,ts_res_loc)


#ML model loading
Efd_features = utility_dict['efd_features']
# ml_res_loc = utility_dict['ml_res_loc']
ml_res_loc = utility_dict['forecast_data_path']+utility_dict['Vessel_name']+'/Results/ML/'
ML = pdm_ml_model(Efd_features,ts_res_loc,engine_number,ml_res_loc)




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


def data_preprocess(raw_data_path,container_client):
    df = read_data_from_blob(raw_data_path,utility_dict['index'],container_client)
    # df = pd.read_csv(raw_data_path, index_col=utility_dict['index'])
    df = df[(df[utility_dict['engine_load']] >= 30) & (df[utility_dict['engine_load']] <= 100)]
    #last 336 datapoints
    df = df.iloc[-utility_dict['look_back']:,:]
    return df
def read_data_from_blob(dataset_name,idx_col,container_client):
    # try:
    data = pd.read_csv(StringIO(container_client.download_blob(dataset_name).content_as_text()),index_col=idx_col)
    return data

def data_collect(utility_dict,dict1):
    # scrap_data = wingd_scraper()
    container_client = ContainerClient.from_connection_string(
    utility_dict['connection_string'], container_name=utility_dict['container_name'])
    call = api_data_collection(1) #if no. o days are more than one uncomment sleep in data_collection_api.py
    for tr in range(1,int(utility_dict['engine_number'])+1):
        table = 'ems_'+str(tr)+'_signals'
        table1 = 'me_'+str(tr)+'_signals'
        table2 = 'ems_'+str(tr)+'_failures'
        data_call = call.data_collect(utility_dict['Data_api']['login'],utility_dict['Data_api']['pass'],dict1['req_col'],dict1['col_map'],table,table1,table2)
        # blob_name_data_update = 'Data/Mu Lan/Engine_2/Train/Combined_data_eng2_diesel_1hr_bfill_v2.csv'
        cur_data = read_data_from_blob('Data/'+utility_dict['Vessel_name']+'/Engine_'+str(tr)+utility_dict['test_data'],utility_dict['index'],container_client)
        # cur_data = pd.read_csv(utility_dict['forecast_data_path']+utility_dict['Vessel_name']+'/Engine_'+str(tr)+utility_dict['test_data'],index_col=utility_dict['index'])
        df_comb = pd.concat([cur_data,data_call])
        df_comb.reset_index(inplace=True)
        df_comb.drop_duplicates(subset=[utility_dict['index']],inplace=True)
        df_comb.set_index(df_comb[utility_dict['index']],inplace=True,drop=True)
        df_comb.drop(columns=[utility_dict['index']],inplace=True)
        # df_comb[utility_dict['add_feature1']] = scrap_data['Fuel oil temperature supply unit (me_'+str(tr)+'_signals)']
        # df_comb[utility_dict['add_feature2']] = scrap_data['TC Bearing Oil Pressure Inlet TC #01 (me_'+str(tr)+'_signals)']
        data_update_csv = df_comb.to_csv()
        data_update_csv_bytes = bytes(data_update_csv, 'utf-8')
        data_update_csv_stream = StringIO(data_update_csv)
        container_client.upload_blob(name='Data/'+utility_dict['Vessel_name']+'/Engine_'+str(tr)+utility_dict['test_data'], data=data_update_csv_bytes,overwrite=True)
        data_update_csv_stream.close()
        # df_comb.to_csv(utility_dict['forecast_data_path']+utility_dict['Vessel_name']+'/Engine_'+str(tr)+utility_dict['test_data'])
    print('Data collection completed')

@app.post("/forecast-14days")
async def forecast_14days(current_user: User = Depends(get_current_active_user)):
    data_collect(utility_dict,dict1)
    container_client = ContainerClient.from_connection_string(
    utility_dict['connection_string'], container_name=utility_dict['container_name'])
    start_time = time.time()
    new_data_path = utility_dict['forecast_data_path']
    vessel_name = utility_dict['Vessel_name']
    for eng in range(1,int(utility_dict['engine_number'])+1):
        data = data_preprocess('Data/'+vessel_name+'/Engine_'+str(eng)+'/Test/API_data_Test.csv',container_client)
        

        #-------------------> TS_model calling
        print('ts input data -',data.shape)
        TS_result = TS.Timeseries(data,eng)


        #-------------------> AUTOENCODER
        #data preprocessing - This will be filtered past data + new data
        df = read_data_from_blob('Data/'+vessel_name+'/Engine_'+str(eng)+'/Train/Combined_data_eng2_diesel_1hr_bfill_v2.csv',utility_dict['index'],container_client)
        df_sub = read_data_from_blob('Data/'+vessel_name+'/Engine_'+str(eng)+'/Test/API_data_Test.csv',utility_dict['index'],container_client)
        df = pd.concat([df,df_sub[df.columns]])

        df = df[(df[utility_dict['engine_load']]>=30)&(df[utility_dict['engine_load']]<=100)]  
        df['Ntc_Pscav'] = df['Turbocharger 1 speed(ems)']/df['Scav. Air Press. Mean Value(ems)']
        df['PR'] = df['Firing Pressure Average(ems)']-df['Compression Pressure Average(ems)']
        df['Pcomp_Pscav'] = df['Compression Pressure Average(ems)']/df['Scav. Air Press. Mean Value(ems)']
        # df.drop(columns=['Estimated power(scr)'],inplace=True)
        df.index = pd.to_datetime(df.index)
        ae = pdm_AE_model()
        df = ae.AE(df)[0]
        # df = pd.read_csv(utility_dict['preprocess_data_path'],index_col=utility_dict['index'])
        # df = pd.read_csv(utility_dict['forecast_data_path']+utility_dict['Vessel_name']+'/Engine_'+str(eng)+utility_dict['test_data'],index_col=utility_dict['index']) #chnage test to train folder
        # df_norm_obj_test = Transform_data(df)
        # df_norm_test = df_norm_obj_test.normalize()
        # #final data for model
        # df_tensor_test = torch.tensor(df_norm_test.values, dtype=torch.float32)
        # recreated_df_test = model(df_tensor_test).cpu().detach().numpy()
        # base_data_test = df_tensor_test.cpu().detach().numpy()
        # faulty_date = recreation_loss(base_data_test, recreated_df_test)
        # if len(faulty_date) > 0:
        #     #data_from_AE = data.iloc[faulty_date,:]
        #     data_from_AE = data.drop(index=faulty_date)
        # else:
        #     data_from_AE = data
        # print('data_from_AE-----------',data_from_AE.shape)
        

        # #-------------------> ML_model calling
        # ML_result = ML.ML_models(data_from_AE)
        ML_result = ML.ML_models(df,eng)


        #-------------------> Fault mapping
        final_indx = [pd.Timestamp(TS_result)+pd.Timedelta(tim,'h') for tim in range(1,TS.forecast_horizon+1)] #for getting timestamps for forecast period, here delta is hourly based
        fault_mat_loc = utility_dict['f_mat_path']
        p=utility_dict['p_value'] #weight value for kpi calculations
        # mapping_loc = utility_dict['map_path']
        mapping_loc = utility_dict['forecast_data_path']+'/'+utility_dict['Vessel_name']+'/Results/Mapping_res/'
        output_dict = {}
        for i in range(1,ML.cyl_count+1):#------------ML.cyl_count+1
            ml_ress = read_data_from_blob("Data/"+utility_dict['Vessel_name']+'/Results/ML/'+utility_dict['Vessel_name']+'_ENG_{}_TS_ML_res_Cyl_{}_{}.csv'.format(str(eng),str(i),str(datetime.now()).split(' ')[0]),False,container_client)  
            # ml_ress = pd.read_csv(ml_res_loc+utility_dict['Vessel_name']+'_ENG_{}_TS_ML_res_Cyl_{}_{}.csv'.format(str(eng),str(i),str(datetime.now()).split(' ')[0]),index_col=False)
            ff = Faults_Mapping(ml_ress,fault_mat_loc,Efd_features,p)
            ff1,fault_ids = ff.Mapping()
            ml_ress = pd.concat([ml_ress,ff1[fault_ids]],axis=1)
            ml_ress[utility_dict['index2']] = final_indx
            #for ordering columns
            ml_ress = ml_ress[utility_dict['end_res_colorder']]
            ml_ress_csv = ml_ress.to_csv(index=False)
            ml_ress_csv_bytes = bytes(ml_ress_csv, 'utf-8')
            ml_ress_csv_stream = StringIO(ml_ress_csv)
            container_client.upload_blob(name='Data/'+utility_dict['Vessel_name']+'/Results/Mapping_res/'+utility_dict['Vessel_name']+'_Eng_{}_mapping_res_cyl{}_{}.csv'.format(str(eng),str(i),str(datetime.now()).split(' ')[0]), data=ml_ress_csv_bytes,overwrite=True)
            ml_ress_csv_stream.close()
            # ml_ress.to_csv(mapping_loc+utility_dict['Vessel_name']+'_Eng_{}_mapping_res_cyl{}_{}.csv'.format(str(eng),str(i),str(datetime.now()).split(' ')[0]),index=False)
            output_dict['Cyl_'+str(i)] = ml_ress.to_dict(orient='list')
        end_time = time.time()
        print('total time taken for 14 days forecast',end_time-start_time)
    
    def output_format(rl,typo,des,utility_dict,Ftype):
    # rl - 
        if utility_dict!=None:
            er=','.join(utility_dict['faults_recom'][Ftype][des])
            return '({})({})-{}-{}'.format(str(rl),typo,'There is a high chance for the occurrence of this fault within two weeks;'+des,er)
        else:
            er='All values are fine for this fault'
            return '({})({})-{}-{}'.format(str(rl),typo,des,er)

    def rating_level(row):
        rating = {}
        if (row['InjSysFault']>=0)&(row['InjSysFault']<60):
            rating['InjSysFault'] = str(utility_dict['rating_level']['0-60'][1]) 
        elif (row['InjSysFault']>=60)&(row['InjSysFault']<70):
            rating['InjSysFault'] = str(utility_dict['rating_level']['60-70'][1]) 
        elif (row['InjSysFault']>=70)&(row['InjSysFault']<80):
            rating['InjSysFault'] = str(utility_dict['rating_level']['70-80'][1])     
        elif row['InjSysFault']>=80:
            rating['InjSysFault'] = str(utility_dict['rating_level']['80-100'][1])  

        if (row['StaInjLate']>=0)&(row['StaInjLate']<60):
            rating['StaInjLate'] = str(utility_dict['rating_level']['0-60'][1]) 
        elif (row['StaInjLate']>=60)&(row['StaInjLate']<70):
            rating['StaInjLate'] = str(utility_dict['rating_level']['60-70'][1]) 
        elif (row['StaInjLate']>=70)&(row['StaInjLate']<80):
            rating['StaInjLate'] = str(utility_dict['rating_level']['70-80'][1])     
        elif row['StaInjLate']>=80:
            rating['StaInjLate'] = str(utility_dict['rating_level']['80-100'][1])   

        if (row['StaInjEarly']>=0)&(row['StaInjEarly']<60):
            rating['StaInjEarly'] = str(utility_dict['rating_level']['0-60'][1]) 
        elif (row['StaInjEarly']>=60)&(row['StaInjEarly']<70):
            rating['StaInjEarly'] = str(utility_dict['rating_level']['60-70'][1]) 
        elif (row['StaInjEarly']>=70)&(row['StaInjEarly']<80):
            rating['StaInjEarly'] = str(utility_dict['rating_level']['70-80'][1])     
        elif row['StaInjEarly']>=80:
            rating['StaInjEarly'] = str(utility_dict['rating_level']['80-100'][1])    

        if (row['ExhValvLeak']>=0)&(row['ExhValvLeak']<60):
            rating['ExhValvLeak'] = str(utility_dict['rating_level']['0-60'][1]) 
        elif (row['ExhValvLeak']>=60)&(row['ExhValvLeak']<70):
            rating['ExhValvLeak'] = str(utility_dict['rating_level']['60-70'][1]) 
        elif (row['ExhValvLeak']>=70)&(row['ExhValvLeak']<80):
            rating['ExhValvLeak'] = str(utility_dict['rating_level']['70-80'][1])     
        elif row['ExhValvLeak']>=80:
            rating['ExhValvLeak'] = str(utility_dict['rating_level']['80-100'][1])        

        if (row['BloCombChabr']>=0)&(row['BloCombChabr']<60):
            rating['BloCombChabr'] = str(utility_dict['rating_level']['0-60'][1]) 
        elif (row['BloCombChabr']>=60)&(row['BloCombChabr']<70):
            rating['BloCombChabr'] = str(utility_dict['rating_level']['60-70'][1]) 
        elif (row['BloCombChabr']>=70)&(row['BloCombChabr']<80):
            rating['BloCombChabr'] = str(utility_dict['rating_level']['70-80'][1])     
        elif row['BloCombChabr']>=80:
            rating['BloCombChabr'] = str(utility_dict['rating_level']['80-100'][1])    

        if (row['ExhValEarOpn']>=0)&(row['ExhValEarOpn']<60):
            rating['ExhValEarOpn'] = str(utility_dict['rating_level']['0-60'][1]) 
        elif (row['ExhValEarOpn']>=60)&(row['ExhValEarOpn']<70):
            rating['ExhValEarOpn'] = str(utility_dict['rating_level']['60-70'][1])
        elif (row['ExhValEarOpn']>=70)&(row['ExhValEarOpn']<80):
            rating['ExhValEarOpn'] = str(utility_dict['rating_level']['70-80'][1])     
        elif row['ExhValEarOpn']>=80:
            rating['ExhValEarOpn'] = str(utility_dict['rating_level']['80-100'][1])      

        if (row['ExhValLatOpn']>=0)&(row['ExhValLatOpn']<60):
            rating['ExhValLatOpn'] = str(utility_dict['rating_level']['0-60'][1]) 
        elif (row['ExhValLatOpn']>=60)&(row['ExhValLatOpn']<70):
            rating['ExhValLatOpn'] = str(utility_dict['rating_level']['60-70'][1]) 
        elif (row['ExhValLatOpn']>=70)&(row['ExhValLatOpn']<80):
            rating['ExhValLatOpn'] = str(utility_dict['rating_level']['70-80'][1])     
        elif row['ExhValLatOpn']>=80:
            rating['ExhValLatOpn'] = str(utility_dict['rating_level']['80-100'][1])    

        if (row['ExhValEarlClos']>=0)&(row['ExhValEarlClos']<60):
            rating['ExhValEarlClos'] = str(utility_dict['rating_level']['0-60'][1])
        elif (row['ExhValEarlClos']>=60)&(row['ExhValEarlClos']<70):
            rating['ExhValEarlClos'] = str(utility_dict['rating_level']['60-70'][1]) 
        elif (row['ExhValEarlClos']>=70)&(row['ExhValEarlClos']<80):
            rating['ExhValEarlClos'] = str(utility_dict['rating_level']['70-80'][1])     
        elif row['ExhValEarlClos']>=80:
            rating['ExhValEarlClos'] = str(utility_dict['rating_level']['80-100'][1])      

        if (row['ExhValLatClos']>=0)&(row['ExhValLatClos']<60):
            rating['ExhValLatClos'] = str(utility_dict['rating_level']['0-60'][1]) 
        elif (row['ExhValLatClos']>=60)&(row['ExhValLatClos']<70):
            rating['ExhValLatClos'] = str(utility_dict['rating_level']['60-70'][1]) 
        elif (row['ExhValLatClos']>=70)&(row['ExhValLatClos']<80):
            rating['ExhValLatClos'] = str(utility_dict['rating_level']['70-80'][1])     
        elif row['ExhValLatClos']>=80:
            rating['ExhValLatClos'] = str(utility_dict['rating_level']['80-100'][1])         
        return rating     

# cv = rating_level(list(df.iterrows())[0][1])         

    output_format_mapping = {'Vessel_info':{'Vessel_Name':utility_dict['Vessel_name'],'VESSEL_OBJECT_ID':utility_dict['VESSEL_OBJECT_ID'],
                                     'JOB_PLAN_ID':utility_dict['JOB_PLAN_ID']}}
    output_format_mapping['Engine_data'] = {}
    for engg in range(1,int(utility_dict['engine_number'])+1):
        output_format_mapping['Engine_data']['Engine_'+str(engg)] = {}
        for i in range(1,ML.cyl_count+1):
            output_format_mapping['Engine_data']['Engine_'+str(engg)]['Cyl_'+str(i)] = {}
            cyl_ff = read_data_from_blob("Data/"+utility_dict['Vessel_name']+'/Results/Mapping_res/'+utility_dict['Vessel_name']+'_Eng_{}_mapping_res_cyl{}_{}.csv'.format(str(engg),str(i),str(datetime.now()).split(' ')[0]),utility_dict['index2'],container_client)
            # cyl_ff = pd.read_csv(mapping_loc+utility_dict['Vessel_name']+'_Eng_{}_mapping_res_cyl{}_{}.csv'.format(str(engg),str(i),str(datetime.now()).split(' ')[0]), index_col='Date Time')
            for r,c in cyl_ff.iterrows():
                indx = rating_level(c)
                mapped = '||'.join([output_format(indx[k],utility_dict['Fault_ids'][k][0],utility_dict['Fault_ids'][k][1],utility_dict,utility_dict['Ftype']) if v not in ['3','2','1'] else output_format(indx[k],utility_dict['Fault_ids'][k][0],utility_dict['Fault_ids'][k][1],None,utility_dict['Ftype']) for k,v in indx.items()])
                output_format_mapping['Engine_data']['Engine_'+str(engg)]['Cyl_'+str(i)].update({str(r):mapped})

    # for i in range(1,ML.cyl_count+1):
    #     cyl_ff = pd.read_excel(mapping_loc + 'mapping_res_cyl{}.xlsx'.format(str(i)), index_col='Date Time')
    #     output_format_dict['Me_data']['Cyl_'+str(i)] = cyl_ff.to_dict(orient='index')
    # inputs :- 
    # 1)no of cyls
    # 2)no of engines
    # 3)VESSEL_OBJECT_ID
    # 4)JOB_PLAN_ID
    blob_client = container_client.get_blob_client('Data/'+utility_dict['Vessel_name']+'/Results/combined_res/'+utility_dict['Vessel_name']+'_'+str(int(time.time()))+'.pickle')
    output_format_mapping_bytes = pickle.dumps(output_format_mapping)
    blob_client.upload_blob(output_format_mapping_bytes,overwrite=True)
    # with open(utility_dict['forecast_data_path']+utility_dict['Vessel_name']+'/Results/combined_res/'+utility_dict['Vessel_name']+'_'+str(int(time.time()))+'.pickle','wb') as file1:
    #     pickle.dump(output_format_mapping,file1)
    return output_format_mapping

@app.post("/smart_maintenance")
async def smart_maintenance(userinput: pdm_inputs, current_user: User = Depends(get_current_active_user)):
    # path_endpoint = './utils/Data/'+utility_dict['Vessel_name']+'/Results/combined_res'
    # path_endpoint_list = os.listdir(path_endpoint)
    pdm_inputs1 = userinput.dict()
    container_client = ContainerClient.from_connection_string(
    utility_dict['connection_string'], container_name=utility_dict['container_name'])
    path_endpoint_list = list(container_client.list_blobs('Data/'+utility_dict['Vessel_name']+'/Results/combined_res/'))
    path_endpoint_list = [blobs['name'].split('/')[-1] for blobs in path_endpoint_list]
    epochss = []
    for fl in path_endpoint_list:
        epochss.append(int(fl.split('_')[1].split('.')[0]))
    print(max(epochss))    
    blob_client = container_client.get_blob_client('Data/'+utility_dict['Vessel_name']+'/Results/combined_res/'+utility_dict['Vessel_name']+'_'+str(max(epochss))+'.pickle')
    pickled_data = blob_client.download_blob().readall()
    end_point_result = pickle.loads(pickled_data)
    #New API format
    # api_format = {'totalRecords':4032,'surveys':[]}
    # measured_date = datetime.fromtimestamp(max(epochss))
    # upload_date  = datetime.now().strftime("%Y-%m-%d")
    # for i in utility_dict['Engine1_inputs'].keys():#pdm_inputs1['Engine1_inputs'].keys(): testing!!!!!!!!!!!!!!!
    #     p1_dict = {}
    #     p1_dict['id'] = pdm_inputs1['filter']['id']
    #     p1_dict['uploadDate'] = upload_date
    #     p1_dict['measureDate'] = measured_date.strftime("%Y-%m-%d")
    #     p1_dict['shipCustom_1'] = pdm_inputs1['filter']['shipCustom1'][0]
    #     p1_dict.update(utility_dict['Engine1_inputs'][i])
    #     p1_dict.update({'findRecom':end_point_result['Engine_data']['Engine_1'][i]})
    #     api_format['surveys'].append(p1_dict)
    # for j in utility_dict['Engine1_inputs'].keys():#pdm_inputs1['Engine2_inputs'].keys(): testing!!!!!!!!!!!!!!!
    #     p2_dict = {}
    #     p2_dict['id'] = pdm_inputs1['filter']['id']
    #     p2_dict['uploadDate'] = upload_date
    #     p2_dict['measureDate'] = measured_date.strftime("%Y-%m-%d")
    #     p2_dict['shipCustom_1'] = pdm_inputs1['filter']['shipCustom1'][0]
    #     p2_dict.update(utility_dict['Engine2_inputs'][j])
    #     p2_dict.update({'findRecom':end_point_result['Engine_data']['Engine_2'][j]})
    #     api_format['surveys'].append(p2_dict)
    # final_api_format = {'status':200,'data':api_format}    
    api_format = {'totalRecords':48,'surveys':[]}
    measured_date = datetime.fromtimestamp(max(epochss)).strftime('%Y-%m-%d %H:%M:%S')
    upload_date  = datetime.now().strftime("%Y-%m-%d")
    me1_faults = pd.read_excel('MuLAN Details.xlsx',sheet_name='ME1_filt')
    me2_faults = pd.read_excel('MuLAN Details.xlsx',sheet_name='ME2_filt')
    def exists_consecutively(lst, element):
        count = 0
        for i in lst:
            if i == element:
                count += 1
                if count == 10:
                    return True
            else:
                count = 0
        return False
    for fault_cats in ['Combustion Blow-by','Injection System','Start of Inj','Exhaust Valve']: 
          
        for engs in end_point_result['Engine_data'].keys():
            for engs_cyl in end_point_result['Engine_data'][engs].keys():
                cyl_wise_end = {} 
                f1 = pd.DataFrame(end_point_result['Engine_data'][engs][engs_cyl],index=['finding']).T
                fault_cat = {}
                fault_cat['Combustion Blow-by'] = {}
                fault_cat['Injection System'] = {}
                fault_cat['Start of Inj'] = {}
                fault_cat['Exhaust Valve'] = {}
                for fids in f1.index:
                    for fids2 in f1.loc[fids,'finding'].split('||'):
                        if 'Injection system fault' in fids2:
                            fault_cat['Injection System'].update({fids:fids2}) #'Combustion Blow-by
                        if 'Start of injection' in fids2:
                            fault_cat['Start of Inj'].update({fids:fids2})
                        if 'Exhaust valve' in fids2:
                            # fault_cat['Exhaust Valve'].update({fids:fids2})  
                            try:
                                fault_cat['Exhaust Valve'][fids]+='||'+fids2
                            except:
                                fault_cat['Exhaust Valve'].update({fids:fids2}) 
                        if 'Blow-by in combustion chamber' in fids2:
                            fault_cat['Combustion Blow-by'].update({fids:fids2})
                ll = pd.DataFrame(fault_cat[fault_cats],index=['findings']).T
                ll['Date'] = list(map(lambda x:x.split(' ')[0],ll.index))
                new_f = pd.DataFrame()
                for llids in ll['Date'].unique():
                    # new_f_sub = ll[ll['Date']==llids]['findings'].str.contains('There is a high chance for the occurrence of this fault within two weeks;').sum()
                    new_f_sub = 1 if True in ll[ll['Date']==llids]['findings'].str.contains('There is a high chance for the occurrence of this fault within two weeks;').values else 0
                    new_f = pd.concat([new_f,pd.DataFrame({'fault_status':new_f_sub},index=[llids])])
                value_dict = {'val_3' : 0, 'val_2' : 0, 'val_1' : 0, 'val_0' : 0}
                for count in ll['findings'].values:
                    if '(3)' in count.split('-')[0]:
                        value_dict['val_3']+=1
                    elif '(2)' in count.split('-')[0]:
                        value_dict['val_2']+=1  
                    elif '(1)' in count.split('-')[0]:
                        value_dict['val_1']+=1  
                    elif '(0)' in count.split('-')[0]:
                        value_dict['val_0']+=1         
                id_max = list(dict(sorted(value_dict.items(),key=lambda x:x[1],reverse=True)).keys())[0].split('_')[1] 
                trigger = exists_consecutively(new_f['fault_status'],1)
                cyl_wise_end['id'] = pdm_inputs1['filter']['id']
                cyl_wise_end['uploadDate'] = upload_date
                cyl_wise_end['measureDate'] = measured_date
                cyl_wise_end['faultIdHat'] = utility_dict['Fault_cats_ids'][fault_cats]
                cyl_wise_end['faultDescrHat'] = fault_cats
                cyl_wise_end['shipCustom_1'] = pdm_inputs1['filter']['shipCustom1'][0]
                
                if engs == 'Engine_1':
                    
                    cyl_wise_end['subName'] = me1_faults[me1_faults['Fault_cat']==fault_cats]['Component1'].values[0].split('NO.')[0].strip()
                    if fault_cats != 'Injection System':
                        tt = me1_faults[(me1_faults['Fault_cat']==fault_cats)&(me1_faults['Component1'].str.contains(cyl_wise_end['subName']+' NO.'+engs_cyl.split('_')[1].strip()))]
                        cyl_wise_end['sysCustom_1'] = tt['EquipmentCode'].values[0]
                        cyl_wise_end['sysCustom_2'] = int(tt['EQUIPMENT_ID'].values[0])
                        cyl_wise_end['sysCustom_3'] = int(tt['Job_Plan_ID'].values[0])
                    else:    
                        tt = me1_faults[(me1_faults['Fault_cat']==fault_cats)&(me1_faults['Component1'].str.contains(cyl_wise_end['subName']+' NO.1'))]#+engs_cyl.split('_')[1].strip()))]
                        cyl_wise_end['sysCustom_1'] = tt['EquipmentCode'].values[0]
                        cyl_wise_end['sysCustom_2'] = int(tt['EQUIPMENT_ID'].values[0])
                        cyl_wise_end['sysCustom_3'] = int(tt['Job_Plan_ID'].values[0])
                if engs == 'Engine_2':
                    
                    cyl_wise_end['subName'] = me2_faults[me2_faults['Fault_cat']==fault_cats]['Component1'].values[0].split('NO.')[0].strip()
                    if fault_cats != 'Injection System':
                        tt = me2_faults[(me2_faults['Fault_cat']==fault_cats)&(me2_faults['Component1'].str.contains(cyl_wise_end['subName']+' NO.'+engs_cyl.split('_')[1].strip()))]
                        cyl_wise_end['sysCustom_1'] = tt['EquipmentCode'].values[0]
                        cyl_wise_end['sysCustom_2'] = int(tt['EQUIPMENT_ID'].values[0])
                        cyl_wise_end['sysCustom_3'] = int(tt['Job_Plan_ID'].values[0])  
                    else:    
                        tt = me2_faults[(me2_faults['Fault_cat']==fault_cats)&(me2_faults['Component1'].str.contains(cyl_wise_end['subName']+' NO.1'))]#+engs_cyl.split('_')[1].strip()))]
                        cyl_wise_end['sysCustom_1'] = tt['EquipmentCode'].values[0]
                        cyl_wise_end['sysCustom_2'] = int(tt['EQUIPMENT_ID'].values[0])
                        cyl_wise_end['sysCustom_3'] = int(tt['Job_Plan_ID'].values[0]) 
                if trigger == True:
                    cyl_wise_end['ratingLevelHat'] = 0
                else:
                    if int(id_max)==0: 
                        cyl_wise_end['ratingLevelHat'] = 1
                    else:    
                        cyl_wise_end['ratingLevelHat'] = int(id_max) 
                cyl_wise_end['findRecom'] = fault_cat[fault_cats]
                api_format['surveys'].append(cyl_wise_end)  
    return api_format
