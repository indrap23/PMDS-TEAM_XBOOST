from os import uname_result
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import joblib
from backend.app.helper import (
    feature_engineering, 
    parse_input,
    predict_score)


app = FastAPI()
templates = Jinja2Templates(directory='./frontend/templates/')


@app.get('/')
def read_form():
    return 'hello world'

@app.get('/form')
def form_post(request: Request):
    result = 'Credit Scoring Results'
    return templates.TemplateResponse('form.html', context={'request': request, 'result': result})

COLUMN_NAMES = ['AGE', 'INCOME', 'GENDER', 'EDUCATION', 'LOAN_PURPOSE', 'HAS_APPLIED_BEFORE',
       'HAS_INCOME_VERIFICATION', "BUREAUDATA", "LOANS_WITHOUT_DELAYS", "LOANS_WITH_DELAYS"]
       
@app.post("/form")
async def predict(request: Request):
    reqDataForm = await request.form()
    reqData = jsonable_encoder(reqDataForm)
    ###Feature Enggineering
    data = feature_engineering(parse_input(reqData))
    #predict
    if reqData['bureau'] == 'Yes':
        model_bureau = joblib.load("./backend/assets/xgb_retrain_bureau.pkl")
        score, loan_dec = predict_score(data, model_bureau)
    else:
        model_no_bureau = joblib.load("./backend/assets//xgb_retrain_no_bureau.pkl")
        score, loan_dec = predict_score(data, model_no_bureau)
    result = loan_dec
    return templates.TemplateResponse('form.html', context={'request': request, 'result': result})