from typing_extensions  import Literal, TypedDict
from typing import List, Optional, Dict
from pydantic import BaseModel, ValidationError, conint


EDUCATION_CLASS_VALUE = Literal['Diploma',
 'Bachelor Degree',
 "Master's Degree/Post graduate",
 'High School',
 'Other']

PURPOSE_CLASS_VALUE = Literal['Working Capital',
 'Other',
 'Renovation',
 'Credit card',
 'Education',
 'Investment',
 'Venture capital',
 'Electronic unsecured loan',
 'Holiday',
 'Bills',
 'Housing loan',
 'Car/Motorcycle']

class BureauDict(TypedDict, total=False):
    loanWithDelay: float
    loanNoDelay: float

class RequestBody(BaseModel):
    age: conint(strict=True, gt=0) 
    income : conint(strict=True, gt=0)
    gender : Literal['Male', 'Female']
    education: EDUCATION_CLASS_VALUE
    purpose: PURPOSE_CLASS_VALUE
    hasApplied: Literal['Yes', 'No']
    hasIncome: Literal['Yes', 'No']
    bureau: BureauDict = None 

class ResponseBody(BaseModel):
    LoanDecision : str