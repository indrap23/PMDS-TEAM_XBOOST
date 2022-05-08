# XGBoost Team
# **Credit Scoring**

## **Problem Statement**
Credit scoring is an important part of the lending process by bank and financial service company. By accurately scoring a customer according to how likely they are able to pay back their loan, company can have confidence in distributing loans and grow their business.In the process, we need to make decisions as quickly as possible while keeping the accuracy of our prediction.The longer it takes to process a loan application—whether it’s a ‘yes’ or ‘no’ decision—the less profitability we can have. With faster and more accurate loan decisions process, we can process more loan applications and the possibility of getting a profit is getting bigger.

## **Objectives**
Our goal is to build a machine learning system that has a model which can accurately and rapidly predict and differentiate bad and good customers using data such as customer’s demographic data and past loan history. By doing so, we hope to lessen the amount of provision needed to cover losses, while at the same time maximize the revenue by accepting all creditworthy customers.

## **Process Workflow**
In this project we used datasets that are collections of 3 separate data : customer information containing the demographic as well as whether they pay back their loan (our target variable), credit bureau data which contain customers past loan applications, and external scoring data. We also used several models such as logistic regression, random forest, and xgboost, and evaluated all models to get the best performing model.
Below is flowchart for the overall workflow

![process workflow](https://user-images.githubusercontent.com/66714513/167297531-3e7a9784-036b-4758-9b8b-ab0f5505fb18.PNG)

## **EDA and Feature Engineering Script**
For feature engineering and EDA, we can open the notebook and find the codes in 
[jupyter/Feature_ready.ipynb](jupyter/Feature_ready.ipynb)

## **Modelling script**
Both of our [Model with Bureau](jupyter/Model_with_bureau.ipynb) and [Model without Bureau](jupyter/Model_without_bureau.ipynb) codes can be found in the jupyter folder alongside our feature engineering and EDA notebook

In the modeling process we train several models i.e Logistic Regression, Random Forest, Decision Tree, KNN Classifier, LGBM, and XGBoost. We chose the three (3) initial models that had the highest Gini score for data with data bureau and without bureau on _training data_ and we found that the best models were Decision Tree, Random Forest and XGBoost.

From these three models, we did hyperparameter tuning and retraining with the best hyperparameter for each model on the train dataset. We then choose one of the three models that has the highest Gini score, as the figure below shows that the highest Gini Score for data with bureau data and without bureau data is the XGBOOST model
<br></br>
Gini score for data with bureau
![gini hyperparameter no bureau](https://user-images.githubusercontent.com/66714513/167299010-71df63a9-76d4-4fbe-8faf-03ec4c38bb0c.PNG)
<br>
Gini score for data without bureau
![gini hyperparameter with bureau](https://user-images.githubusercontent.com/66714513/167299013-c2cc9b7e-ba09-48ed-878a-5e88ab45153b.PNG)

## **FastApi backend and Deployment to Heroku**
From the model that we have created, we save it with the Joblib module in the form of pkl. We then build api endpoints using FastApi and uvicorn ASGI web server to serve the model

### **Fastapi main backend codes : [backend/api.py](backend/api.py)**

```
import re
from fastapi import FastAPI, Body, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
import joblib
import warnings   
warnings.filterwarnings("ignore")

from logging.config import dictConfig
import logging
from app.config import LogConfig

dictConfig(LogConfig().dict())
logger = logging.getLogger("pmds")


from app.models import RequestBody, ResponseBody
from app.helper import (
    feature_engineering, 
    parse_request,
    parse_input,
    predict_score,
    grade_binning)

# initialize the fastapi app
app = FastAPI()

# mount templates and static folder with Jinja
templates = Jinja2Templates(directory='./frontend/templates/')
app.mount("/statics", StaticFiles(directory="./frontend/statics"), name="statics")

# add logging for invalid request error 
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    content_log=jsonable_encoder({"detail": exc.errors(), "body": exc.body})
    content_msg =  content_log.get("detail")[0].get("msg")  # only get the message
    content_loc = content_log.get("detail")[0].get("loc")[1]  # get the error field
    logger.error(content_log)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=content_msg + ", field : " + content_loc
    )

# add CORS middleware
origins = [
    "http://localhost:3000",
    "localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model at the first instance of app initializaton
try :
    # for uvicorn
    model_bureau = joblib.load("backend/assets/xgb_retrain_bureau.pkl")
    model_no_bureau = joblib.load("backend/assets/xgb_retrain_no_bureau.pkl")

    # for gunicorn
    # model_bureau = joblib.load("./assets/xgb_retrain_bureau.pkl")
    # model_no_bureau = joblib.load("./assets/xgb_retrain_no_bureau.pkl")
    print("Model Loaded")
except:
    print("Fail to Load Model")

@app.get("/")
async def root():
    """
    Root endpoint
    """
    url = app.url_path_for("index_html")
    response = RedirectResponse(url=url)
    logger.debug("Redirected to home")
    return response

@app.get('/home')
def index_html(request: Request):
    """
    Endpoint for the loan application form
    """
    return templates.TemplateResponse('index.html', context={'request': request})

@app.post("/predict")
async def get_prediction(request: Request):
    """
    Calculating and doing prediction
    """
    reqDataForm = await request.form()
    reqData = jsonable_encoder(reqDataForm)
    reqData_parsed = parse_request(reqData)

    t = feature_engineering(parse_input(reqData_parsed))
    if reqData.get("bureau"):
        score, proba, loan_dec = predict_score(t, model_bureau)
        grade = grade_binning(proba, 'model_bureau')
    else:
        score, proba, loan_dec = predict_score(t, model_no_bureau)
        grade = grade_binning(proba, 'model_no_bureau')

    if grade in ['A', 'B'] :
        page = 'result_approve.html'
    else:
        page = 'result_reject.html'
    logger.info(f"predicting..{score}")
    logger.info(f"Loan Decision..{loan_dec}")

    return templates.TemplateResponse(page, context={'request': request, 'grade': grade})

@app.post("/apiTest", response_model=ResponseBody)
async def testing(
    req: RequestBody = Body(
    ..., 
    examples = {
            "bureaufound" : { # example of payload with bureau data available
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
            "nobureau" : { # example of payload without bureau data
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
    """
    Endpoint to test the post request for prediction

    """

    # cast request body to dict
    req_dict = req.dict()
    t = feature_engineering(parse_input(req_dict))
    if req_dict.get("bureau"):
        score, proba, loan_dec = predict_score(t, model_bureau)
    else:
        score, proba, loan_dec = predict_score(t, model_no_bureau)
    logger.info(f"predicting..{score}")
    return {"LoanDecision" : loan_dec}

```

Next we make a deployment using heroku. The results of our deployment can be seen on the link https://pmds-xgboost.herokuapp.com/. The interface of our product is as shown in the image below:

![image](https://user-images.githubusercontent.com/66714513/167299945-73ddd14f-f262-4a6a-b90d-a67eb926c886.png)

## **Conclusion** 
We have been looking for the best model to predict whether the applicant will repay the loan or not. We use two models to be able to calculate better results between applicants who have historical data or not. <br></br>
The results we have obtained so far still leave much to be desired. Due to the limited data we have, we can only make the current models. For further exploration we suggest adding more data or features to make a better model. <br></br>
Excluding the accuracy of the model we made, the products can provide speed in making decisions whether the loan application will be accepted or not. Of course, this can help banks or lending institutions to maximize profits by allowing more loan applicants to be processed. Besides that, it also prevents potential customers from taking loans from other competitors because we can be able to make decisions faster.

## **Reference** 
Schuermann, T. (2004). What do We Know about Loss Given Default? SSRN Electronic Journal.<br></br>

Gini, ROC, AUC (and Accuracy). (n.d.). STAESTHETIC. Retrieved March 09, 2022, from https://staesthetic.wordpress.com/2014/04/14/gini-roc-auc-and-accuracy/<br></br>

(n.d.). The Lost Art of Decile Analysis. KDnuggets. Retrieved April 20, 2022, from https://www.kdnuggets.com/2021/07/lost-art-decile-analysis.html<br></br>

Pacmann. (2022). GIve Me Some Credit Notebook. Machine Learning Cases I.<br></br>


