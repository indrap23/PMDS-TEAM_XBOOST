import pandas as pd
import numpy as np

COLUMN_NAMES = ['AGE', 'GENDER', 'INCOME', 'HAS_APPLIED_BEFORE', 'LOAN_PURPOSE', 'EDUCATION', 
       'HAS_INCOME_VERIFICATION', "LOANS_WITHOUT_DELAYS", "LOANS_WITH_DELAYS"]

OHE_FEATURES = [
 'EDUCATION_Bachelor Degree',
 'EDUCATION_Diploma',
 'EDUCATION_High School',
 "EDUCATION_Master's Degree/Post graduate",
 'EDUCATION_Other',
 'LOAN_PURPOSE_Bills',
 'LOAN_PURPOSE_Credit card',
 'LOAN_PURPOSE_Education',
 'LOAN_PURPOSE_Electronic unsecured loan',
 'LOAN_PURPOSE_Holiday',
 'LOAN_PURPOSE_Housing loan',
 'LOAN_PURPOSE_Investment',
 'LOAN_PURPOSE_Other',
 'LOAN_PURPOSE_Renovation',
 'LOAN_PURPOSE_Venture capital',
 'LOAN_PURPOSE_Working Capital',
 'LOAN_PURPOSE_Car/Motorcycle']


def parse_request(reqData):
    """
    Parse request data from application form to fit the model input structure
    """

    reqData.pop('firstName', None)
    reqData.pop('lastName', None)
    if reqData.get("hasIncome") == "No Income Proof":
        reqData['hasIncome'] = "No"
    else:
        reqData['hasIncome'] = "Yes"

    if reqData.get("bureau") == "Yes":
        reqData['bureau'] = {"loanNoDelay" : int(reqData.get("loanNoDelay")), "loanWithDelay" : int(reqData.get("loanNoDelay"))}
    else:
        reqData['bureau'] = None
    reqData.pop('loanNoDelay', None)
    reqData.pop('loanWithDelay', None)

    reqData['age'] = int(reqData['age'])
    reqData['income'] = int(reqData['income'])
    return reqData

def parse_input(data):
    """
    Parse the model input data into pandas dataframe
    """

    temp = pd.json_normalize(data, sep='_')
    colname_bureau = dict(zip(list(temp.columns), list(COLUMN_NAMES)))
    colname = dict(zip(list(temp.columns), list(COLUMN_NAMES[:-2])))
    if data.get("bureau"):
        temp = temp.rename(columns = colname_bureau)
    else :
        temp = temp.rename(columns = colname).drop(columns=["bureau"])
    return temp

def binning_features(df):
    """
    Binning age and income features
    """

    #binning Age
    df.loc[ df['AGE'] <= 20, 'AGE'] = 0
    df.loc[(df['AGE'] > 20) & (df['AGE'] <= 34), 'AGE'] = 1
    df.loc[(df['AGE'] > 34) & (df['AGE'] <= 40), 'AGE'] = 2
    df.loc[(df['AGE'] > 40) & (df['AGE'] <= 47), 'AGE'] = 3
    df.loc[ df['AGE'] > 47, 'AGE'] = 4
    
    #income
    df.loc[ df['INCOME'] <= 4e6, 'INCOME'] = 0
    df.loc[(df['INCOME'] > 4e6) & (df['INCOME'] <= 5e6), 'INCOME'] = 1
    df.loc[(df['INCOME'] > 5e6) & (df['INCOME'] <= 6e6), 'INCOME'] = 2
    df.loc[(df['INCOME'] > 6e6) & (df['INCOME'] <= 8e6), 'INCOME'] = 3
    df.loc[(df['INCOME'] > 8e6) & (df['INCOME'] <= 11e6), 'INCOME'] = 4
    df.loc[ df['INCOME'] > 11e6, 'INCOME'] = 5

    return df

def feature_engineering(df):
    """
    Apply feature engineering to the input dataframe
    """

    #converting categorical Features
    df['GENDER'] = df['GENDER'].map( {'Male': 1, 'Female': 0} ).astype(int)
    df['HAS_APPLIED_BEFORE'] = df['HAS_APPLIED_BEFORE'].map( {'Yes': 1, 'No': 0} ).astype(int)
    df['HAS_INCOME_VERIFICATION'] = df['HAS_INCOME_VERIFICATION'].map( {'Yes': 1, 'No': 0} ).astype(int)

    binning_features(df)

    #add LOAN_PURPOSE and EDUCATION one hot encoded features
    df_final = pd.concat([df, pd.DataFrame(0, index = [0], columns=OHE_FEATURES)], axis= 1).\
        drop(columns=["EDUCATION", "LOAN_PURPOSE"])
    return df_final

# get scoring grade based on the probability
def grade_binning(proba, model):
    """ 
    Binning the not default probability of each application into 5 seperate grade
    Will only approve customer with grade A and B

    Keyword arguments:
    proba -- the probability of not default / paying back the loan
    model -- type of model (["model_no_bureau", "model_bureau"])
    
    """
    dict_bin = {0:'E', 1:'D', 2:'C', 3:'B', 4:'A' }
    x = proba # turn the probability into a numpy array
    if model == "model_no_bureau":
        out_x = np.where(x<=0.374,0, np.searchsorted([0.374,0.442,0.495,0.571], x))
    elif model == "model_bureau":
        out_x = np.where(x<=0.486,0, np.searchsorted([0.486,0.493,0.498,0.506], x))
    return dict_bin.get(out_x[0])

def predict_score(df, model):
    """
    Return application prediction, probability, and decision ("Reject" or "Approve")
    """
    loan_dec = "Reject" if model.predict(df)[0] == 0 else "Approve"
    return model.predict(df)[0], model.predict_proba(df)[:,1], loan_dec