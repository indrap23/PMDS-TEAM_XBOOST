from pyexpat import model
from fastapi import FastAPI, Body
import joblib

from app.constans import RequestBody, ResponseBody
from app.src.modeling import modelling
from app.src.feature_engineering import (
    parse_input,
    feature_engineering
)

app = FastAPI()

try :
    model_bureau = joblib.load("./assets/xgb_retrain_bureau.pkl")
    model_no_bureau = joblib.load("./assets/xgb_retrain_no_bureau.pkl")
    print("Model Loaded")
except:
    print("Fail to Load Model")

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/Credit_Approval", response_model=ResponseBody)
async def testing(
    req: RequestBody = Body(
    ..., 
    examples = {
            "bureaufound" : {
                "summary" : "payload with bureau data",
                "description" : "Input example for customer with bureau data available.",
                "value" : {
                    "age": 20,
                    "income": 1000000,
                    "gender": "Male",
                    "hasApplied": "Yes",
                    "hasIncome": "Yes",
                    "education": "Diploma",
                    "purpose": "Working Capital",
                    "bureau": {"loanWithDelay" : 0, "loanNoDelay" : 0}
                }
            },
            "nobureau" : {
                "summary" : "payload without bureau data",
                "description" : "Example of request if customer does not have bureau data available.",
                "value" : {
    
                    "age": 22,
                    "income": 2000000,
                    "gender": "Female",
                    "hasApplied": "No",
                    "hasIncome": "Yes",
                    "education": "Bachelor Degree",
                    "purpose": "Investment",
                    "bureau": None
                }
            }
        } )
    ):

    # cast request body to dict
    req_dict = req.dict()
    t = feature_engineering(parse_input(req_dict))
    #is that bureau data, get True or False
    bureau_ = req_dict.get("bureau")
    score, loan_dec = modelling(t, bureau=bureau_)
    print("Predicted Score :", score)
    return {"LoanDecision" : loan_dec}