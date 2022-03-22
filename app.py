import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pandas import ExcelWriter
from pandas import ExcelFile

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, roc_curve, auc

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

st.write("""
# Fracture Prediction App
Please input **paramerter** on your left!
""")

st.sidebar.header('User Input Parameters')
#df.columns = ['Age', 'Sex','BMI','Preinjury_status','Preinjury_gaitaid','CKD', 'CCI','Heart_disease','CVA', 'Lung_disease', 'Dementia', 'Diagnosis','Type_of_operation','Status' , 'Time_to_op']
#df #bedridden  indoordependent indoorindependent   outdoordependent    outdoorindependent
#no ambulation  nogaitaid   quadcane    single cane tripod cane walker  wheelchair
#Diagnosis intertroch   neck    subtroch    
# Type_of_operation Cephalomedullary nailing    DHS Hemiarthroplasty    Multiple screw fixation THA conservative
#Time_to_op >48hr
#CKD
#Heart_disease
#CVA
#Lung_disease
#Dementia

def user_input_features():
    data = {
                "Age": "",
                "BMI": "",
                "CKD": "",
                "CCI": "",
                "Heart_disease": "",
                "CVA": "",
                "Lung_disease": "",
                "Dementia": "",
                "Female": "",
                "Male": "",
                "bedridden": "0",
                "indoordependent": "0",
                "indoorindependent": "0",
                "outdoordependent": "0",
                "outdoorindependent": "0",
                "no ambulation": "0",
                "nogaitaid": "0",
                "quadcane": "0",
                "single cane": "0",
                "tripod cane": "0",
                "walker": "0",
                "wheelchair": "0",
                "intertroch": "0",
                "neck": "0",
                "subtroch": "0",
                "Cephalomedullary nailing": "0",
                "DHS": "0",
                "Hemiarthroplasty": "0",
                "Multiple screw fixation": "0",
                "THA": "0",
                "conservative": "0",
                ">48hr": "0"
            }

    Age = st.sidebar.number_input("Age", min_value=20, max_value=99, value=45)
    Sex = st.sidebar.radio("Gender", ['Male','Female'], index=1)
    if Sex == 'Male':
        Male = '0'
        Female = '1'
    elif Sex == 'Female':
        Male = '1'
        Female = '0'
    BMI = st.sidebar.slider('BMI', 16.00, 23.0, 30.00)

    CCI = st.sidebar.slider('Charlson Comorbidity Index (CCI)', 0, 0, 40)
    #CCI = st.sidebar.number_input("Charlson Comorbidity Index (CCI)", min_value=0, max_value=40, value=0)

    Preinjury_status = st.sidebar.selectbox('How was Preinjury status?',('bedridden','indoordependent','indoorindependent','outdoordependent','outdoorindependent'))
    bedridden = '0'
    indoordependent = '0'
    indoorindependent = '0'
    outdoordependent = '0'
    outdoorindependent = '0'
    if Preinjury_status =='bedridden':
        bedridden = '1'
    elif Preinjury_status =='indoordependent':
        indoordependent = '1'
    elif Preinjury_status =='indoorindependent':
        indoorindependent = '1'
    elif Preinjury_status =='outdoordependent':
        outdoordependent = '1'
    elif Preinjury_status =='outdoorindependent':
        outdoorindependent = '1'


    Preinjury_gaitaid = st.sidebar.selectbox('How was Preinjury gait aid?',('no ambulation','nogaitaid','quadcane','single cane','tripod cane','walker','wheelchair'))


    no_ambulation = '0'
    nogaitaid = '0'
    quadcane = '0'
    single_cane = '0'
    tripod_cane = '0'
    walker = '0'
    wheelchair = '0'
    if Preinjury_gaitaid =='no ambulation':
        no_ambulation = '1'
    elif Preinjury_status =='nogaitaid':
        nogaitaid = '1'
    elif Preinjury_status =='quadcane':
        quadcane = '1'
    elif Preinjury_status =='single cane':
        single_cane = '1'
    elif Preinjury_status =='tripod cane':
        tripod_cane = '1'
    elif Preinjury_status =='walker':
        walker = '1'
    elif Preinjury_status =='wheelchair':
        wheelchair = '1'



    intertroch = '0'
    neck = '0'
    subtroch = '0'
    Diagnosis = st.sidebar.selectbox('What is the Diagnosis?',('intertroch','neck','subtroch'))
    if Diagnosis =='intertroch':
        intertroch = '1'
    elif Diagnosis =='neck':
        neck = '1'
    elif Diagnosis =='subtroch':
        subtroch = '1'

    Cephalomedullary_nailing = '0'
    DHS = '0'
    Hemiarthroplasty = '0'
    Multiple_screw_fixation = '0'
    THA = '0'
    conservative = '0'
    Type_of_operation = st.sidebar.selectbox('What is the type of operation?',('Cephalomedullary nailing','DHS','Hemiarthroplasty','Multiple screw fixation','THA','conservative'))
    if Type_of_operation =='Cephalomedullary nailing':
        Cephalomedullary_nailing = '1'
    elif Type_of_operation =='DHS':
        DHS = '1'
    elif Type_of_operation =='Hemiarthroplasty':
        Hemiarthroplasty = '1'
    elif Type_of_operation =='Multiple screw fixation':
        Multiple_screw_fixation = '1'
    elif Type_of_operation =='THA':
        THA = '1'
    elif Type_of_operation =='conservative':
        conservative = '1'

    Time_to_op = st.sidebar.radio("Time to op", ['< 48 hr','> 48 hr'], index=1)

    if Time_to_op =='> 48 hr':
        M48hr = '1'
    else:
        M48hr = '0'


    CKD = st.sidebar.checkbox("CKD")
    if CKD:
        CKD = '1'
    else:
        CKD = '0'

    Heart_disease = st.sidebar.checkbox("Heart_disease")
    if Heart_disease:
        Heart_disease = '1'
    else:
        Heart_disease = '0'
    CVA = st.sidebar.checkbox("CVA")
    if CVA:
        CVA = '1'
    else:
        CVA = '0'
    Lung_disease = st.sidebar.checkbox("Lung_disease")
    if Lung_disease:
        Lung_disease = '1'
    else:
        Lung_disease = '0'
    Dementia = st.sidebar.checkbox("Dementia")
    if Dementia:
        Dementia = '1'
    else:
        Dementia = '0'

    data = {
                "Age": Age,
                "BMI": BMI,
                "CKD": CKD,
                "CCI": CCI,
                "Heart_disease": Heart_disease,
                "CVA": CVA,
                "Lung_disease": Lung_disease,
                "Dementia": Dementia,
                "Female": Female,
                "Male": Male,
                "bedridden": bedridden,
                "indoordependent": indoordependent,
                "indoorindependent": indoorindependent,
                "outdoordependent": outdoordependent,
                "outdoorindependent": outdoorindependent,
                "no ambulation": no_ambulation,
                "nogaitaid": nogaitaid,
                "quadcane": quadcane,
                "single cane": single_cane,
                "tripod cane": tripod_cane,
                "walker": walker,
                "wheelchair": wheelchair,
                "intertroch": intertroch,
                "neck": neck,
                "subtroch": subtroch,
                "Cephalomedullary nailing": Cephalomedullary_nailing,
                "DHS": DHS,
                "Hemiarthroplasty": Hemiarthroplasty,
                "Multiple screw fixation": Multiple_screw_fixation,
                "THA": THA,
                "conservative": conservative,
                ">48hr": M48hr
            }
    
    features = pd.DataFrame(data, index=[0])

    return features
df = user_input_features()

                                                        
st.subheader('Input parameters')
st.write(df)


with open("Models/best_gbc.pickle", 'rb') as model:
    model = pickle.load(model)

prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
#st.write(iris.target_names)

st.subheader('Prediction')
#st.write(iris.target_names[prediction])
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
