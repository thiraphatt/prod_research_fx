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
#df #bedridden	indoordependent	indoorindependent	outdoordependent	outdoorindependent
#no ambulation	nogaitaid	quadcane	single cane	tripod cane	walker	wheelchair
#Diagnosis intertroch	neck	subtroch	
# Type_of_operation Cephalomedullary nailing	DHS	Hemiarthroplasty	Multiple screw fixation	THA conservative
#Time_to_op >48hr
#CKD
#Heart_disease
#CVA
#Lung_disease
#Dementia

def user_input_features():
    Age = st.sidebar.number_input("Age", min_value=20, max_value=99, value=60)
    Sex = st.sidebar.radio("Gender", ['Male','Female'], index=1)
    BMI = st.sidebar.slider('BMI', 16.1, 23.1, 35.99)

    Preinjury_status = st.sidebar.selectbox('How was Preinjury status?',('bedridden','indoordependent','indoorindependent','outdoordependent','outdoorindependent'))

    Preinjury_gaitaid = st.sidebar.selectbox('How was Preinjury gait aid?',('no ambulation','nogaitaid','quadcane','single cane','tripod cane','walker','wheelchair'))

    Diagnosis = st.sidebar.selectbox('What is the Diagnosis?',('intertroch','neck','subtroch'))

    Type_of_operation = st.sidebar.selectbox('What is the type of operation?',('Cephalomedullary nailing','DHS','Hemiarthroplasty','Multiple screw fixation','THA','conservative'))

    Time_to_op = st.sidebar.radio("Time to op", ['< 48 hr','> 48 hr'], index=1)

    CKD = st.sidebar.checkbox("CKD")
    Heart_disease = st.sidebar.checkbox("Heart_disease")
    CVA = st.sidebar.checkbox("CVA")
    Lung_disease = st.sidebar.checkbox("Lung_disease")
    Dementia = st.sidebar.checkbox("Dementia")
    
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

with open("Models/best_gbc.pickle", 'rb') as data:
    model = pickle.load(data)

clf = model
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)