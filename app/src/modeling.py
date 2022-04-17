from tkinter.messagebox import YES
import numpy as np
import pandas as pd
import joblib
from app.constans import MODEL_BUREAU, MODEL_NO_BUREAU

def modelling(x_predict: pd.DataFrame, bureau=True):
    model_bureau = joblib.load(MODEL_BUREAU)
    model_no_bureau = joblib.load(MODEL_NO_BUREAU)

    if bureau:
        y_predicted = model_bureau.predict(x_predict)
    else:
        y_predicted = model_no_bureau.predict(x_predict)
    y_predicted = y_predicted[0]
    loan_aprove = "Reject" if y_predicted == 0 else "Approve"
    return y_predicted, loan_aprove