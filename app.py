import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
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
import streamlit.components.v1 as component

st.set_page_config(page_title='1-yr Mortality Prediction After Fragility Hip Fracture')
st.components.v1.html(
    """
    <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-PELKBJ6ES5"></script>
        <script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());

            gtag('config', 'G-PELKBJ6ES5');
        </script>
    """
    , width=None, height=None, scrolling=False
    )


code = """

    <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-PELKBJ6ES5"></script>
        <script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());

            gtag('config', 'G-PELKBJ6ES5');
        </script>
"""

a='/home/ec2-user/prod_research_fx/static/index.html'
with open(a, 'r') as f:
    data=f.read()
    if len(re.findall('G-', data))==0:
        with open(a, 'w') as ff:
            newdata=re.sub('<head>','<head>'+code,data)
            ff.write(newdata)


st.write("""
# 1-yr Mortality Prediction After Fragility Hip Fracture
""")

st.sidebar.header('Input Parameters')
dataset_processed = pd.read_csv('dataset/dataset_processed.csv')
dataset_processed_x = dataset_processed.iloc[:,:-1].values

scaler = MinMaxScaler()
scaler.fit(dataset_processed_x)

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
        Male = '1'
        Female = '0'
    elif Sex == 'Female':
        Male = '0'
        Female = '1'
    #BMI = st.sidebar.slider('BMI', 16.00, 23.0, 30.00)
    BMI = st.sidebar.slider(label='BMI', min_value=16.00, max_value=35.00, value=20.0)
    
    

    CKD = st.sidebar.checkbox("CKD")
    if CKD:
        CKD = '1'
    else:
        CKD = '0'

    Heart_disease = st.sidebar.checkbox("Heart disease")
    if Heart_disease:
        Heart_disease = '1'
    else:
        Heart_disease = '0'
    CVA = st.sidebar.checkbox("CVA")
    if CVA:
        CVA = '1'
    else:
        CVA = '0'
    Lung_disease = st.sidebar.checkbox("Lung disease")
    if Lung_disease:
        Lung_disease = '1'
    else:
        Lung_disease = '0'
    Dementia = st.sidebar.checkbox("Dementia")
    if Dementia:
        Dementia = '1'
    else:
        Dementia = '0'

    
    #CCI = st.sidebar.slider('Charlson Comorbidity Index (CCI)', 0, 0, 40)
    CCI = st.sidebar.slider(label='Charlson Comorbidity Index (CCI)', min_value=0, max_value=40, value=5)
    #CCI = st.sidebar.number_input("Charlson Comorbidity Index (CCI)", min_value=0, max_value=40, value=0)

    Preinjury_status = st.sidebar.selectbox('Pre-injury ambulatory status',('Outdoor independent','Outdoor dependent','Indoor independent','Indoor dependent','Bedridden'))
    bedridden = '0'
    indoordependent = '0'
    indoorindependent = '0'
    outdoordependent = '0'
    outdoorindependent = '0'
    if Preinjury_status =='Bedridden':
        bedridden = '1'
    elif Preinjury_status =='Indoor dependent':
        indoordependent = '1'
    elif Preinjury_status =='Indoor independent':
        indoorindependent = '1'
    elif Preinjury_status =='Outdoor dependent':
        outdoordependent = '1'
    elif Preinjury_status =='Outdoor independent':
        outdoorindependent = '1'


    Preinjury_gaitaid = st.sidebar.selectbox('Assistive device',('No gaitaid','Single cane','Tripod cane','Quad cane','Walker','Wheelchair', 'No ambulation'))


    no_ambulation = '0'
    nogaitaid = '0'
    quadcane = '0'
    single_cane = '0'
    tripod_cane = '0'
    walker = '0'
    wheelchair = '0'
    if Preinjury_gaitaid =='No ambulation':
        no_ambulation = '1'
    elif Preinjury_status =='No gaitaid':
        nogaitaid = '1'
    elif Preinjury_status =='Quad cane':
        quadcane = '1'
    elif Preinjury_status =='Single cane':
        single_cane = '1'
    elif Preinjury_status =='Tripod cane':
        tripod_cane = '1'
    elif Preinjury_status =='Walker':
        walker = '1'
    elif Preinjury_status =='Wheelchair':
        wheelchair = '1'



    intertroch = '0'
    neck = '0'
    subtroch = '0'
    Diagnosis = st.sidebar.selectbox('Type of fracture',('Intertrochanteric fracture','Femoral neck fracture','Subtrochanteric fracture'))
    if Diagnosis =='Intertrochanteric fracture':
        intertroch = '1'
    elif Diagnosis =='Femoral neck fracture':
        neck = '1'
    elif Diagnosis =='Subtrochanteric fracture':
        subtroch = '1'

    Cephalomedullary_nailing = '0'
    DHS = '0'
    Hemiarthroplasty = '0'
    Multiple_screw_fixation = '0'
    THA = '0'
    conservative = '0'
    Type_of_operation = st.sidebar.selectbox('Treatment',('Multiple screw fixation', 'Cephalomedullary nailing','DHS','Hemiarthroplasty','THA','Conservative treatment'))
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
    elif Type_of_operation =='Conservative treatment':
        conservative = '1'

    if conservative != '1':
        Time_to_op = st.sidebar.radio("Time to operation", ['<= 48 hr','> 48 hr'], index=1)
    else:
        Time_to_op = '> 48 hr'

    if Time_to_op =='> 48 hr':
        M48hr = '1'
    else:
        M48hr = '0'

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
    
    

    return data
data = user_input_features()
df = pd.DataFrame(data, index=[0])

                                                        
#st.subheader('Input parameters')
#st.write(data)

#st.write(df)

df = df.values

df = scaler.transform(df)
#st.write(df)

with open("Models/best_gbc.pickle", 'rb') as model:
    model = pickle.load(model)


prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

#st.title('Prediction:')       

prediction = str(prediction)
if prediction == '[0]':
    st.subheader('Prediction: Low risk of mortality')
elif prediction == '[1]':
    st.subheader('Prediction: High risk of mortality')

text_p = round(float(prediction_proba[0,1]*100), 2)
if text_p <= 0.5:
    text_p = ' less than 0.5'
#st.write(text_p) 
text_prob = 'Probability of 1-yr mortality is '+ str(text_p)+ ' %'
st.write(text_prob) 
#st.write(prediction_proba[0,1])


#st.write(prediction_proba)
#st.write(str(prediction_proba))
#st.write(prediction_proba[0,0])


st.write('') 
st.write('') 
st.write('DISCLAMER: The implementation of this 1-yr hip fracture mortality prediction is??NOT??intended for use supporting or informing clinical decision-making. It is??ONLY??to be used for academic research, peer review and validation purposes, and it must??NOT??be used with data or information relating to any individual.')

