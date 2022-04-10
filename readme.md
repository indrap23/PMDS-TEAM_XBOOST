# XGBoost Team
# **Credit Scoring**

## **Problem Statement**
Credit scoring is an important part of the lending process by bank and financial service company. By accurately scoring a customer according to how likely they are able to pay back their loan, company can have confidence in distributing loans and grow their business.

## **Objectives**
Our goal is therefore to create a model that can accurately predict the probability of a customer paying back their loan, which we will apply to new customer later on.

## **Steps running fastapi (after pulling the repo to local pc) :**
<br><b> With Anaconda </b>
  <br>
  <ol>
    <li>Open the anaconda prompt terminal</li>
    <li>cd to repo location if haven't set it in terminal yet</li>
       <li>Create new environment using the command Conda create --name [ENVIRONMENT NAME] </li>
       <li>Aciticate your new environment using the command Conda activate [ENVIRONMENT NAME] </li>
       <li>install dependencies using pip install -r requirements.txt </li>
       <li>run the local uvicorn server to serve the api using command</li>
  <b>uvicorn main:app --reload </b>
  <br> this will deploy the API in your local computer which will serve the model 
  <li>then either use Postman, curl, or we can go to <a href="http://127.0.0.1:8000/docs">http://127.0.0.1:8000/docs</a> to see the open api documentation and test the api directly</li>
  </ol>
