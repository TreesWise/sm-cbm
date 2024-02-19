from typing import Union,TypedDict,List
from pydantic import BaseModel

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Union[str, None] = None


class User(BaseModel):
    username: str


class UserInDB(User):
    hashed_password: str

class Engineinfo(BaseModel):
    id: int
    shipCustom1: List=None
class pdm_inputs(BaseModel):
    filter: Engineinfo      

#Below code is useful for multiple vessels
# class eng1_inputs(TypedDict):
#     Cyl1_Eqp_Code: str
#     Cyl1_Eqp_ID: int
#     Cyl2_Eqp_Code: str
#     Cyl2_Eqp_ID: int
#     Cyl3_Eqp_Code: str
#     Cyl3_Eqp_ID: int
#     Cyl4_Eqp_Code: str
#     Cyl4_Eqp_ID: int
#     Cyl5_Eqp_Code: str
#     Cyl5_Eqp_ID: int
#     Cyl6_Eqp_Code: str
#     Cyl6_Eqp_ID: int 

# class eng2_inputs(TypedDict):
#     Cyl1_Eqp_Code: str
#     Cyl1_Eqp_ID: int
#     Cyl2_Eqp_Code: str
#     Cyl2_Eqp_ID: int
#     Cyl3_Eqp_Code: str
#     Cyl3_Eqp_ID: int
#     Cyl4_Eqp_Code: str
#     Cyl4_Eqp_ID: int
#     Cyl5_Eqp_Code: str
#     Cyl5_Eqp_ID: int
#     Cyl6_Eqp_Code: str
#     Cyl6_Eqp_ID: int    

# class pdm_inputs(BaseModel):
#     Vessel_IMO_Number: str    
#     Vessel_Object_ID: str
#     Shipid: int
#     Job_Plan_ID: int
#     engine_1: eng1_inputs
#     engine_2: eng2_inputs
    # required input format
# class EngineCylinderData(BaseModel):
#     Eqp_Code: str
#     Eqp_ID: int
#     Job_Plan_ID: int

# # Define the structure for the engine_1 dictionary
# class Engine1_inputs(BaseModel):
#     Cyl_1: EngineCylinderData = None
#     Cyl_2: EngineCylinderData = None
#     Cyl_3: EngineCylinderData = None
#     Cyl_4: EngineCylinderData = None
#     Cyl_5: EngineCylinderData = None
#     Cyl_6: EngineCylinderData = None
# class Engine2_inputs(BaseModel):
#     Cyl_1: EngineCylinderData = None
#     Cyl_2: EngineCylinderData = None
#     Cyl_3: EngineCylinderData = None
#     Cyl_4: EngineCylinderData = None
#     Cyl_5: EngineCylinderData = None
#     Cyl_6: EngineCylinderData = None

# # Define the pdm class extending BaseModel
# class pdm(BaseModel):
#     Vessel_IMO_Number: str
#     Vessel_Object_ID: str
#     Shipid: int
#     engine_1: Engine1_inputs
#     engine_2: Engine2_inputs
      
    


