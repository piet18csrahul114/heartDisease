import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('DecisionTree.pkl', 'rb'))
dataset= pd.read_csv('CLASSIFICATION DATASET.csv')
X = dataset.iloc[:,0:14].values

# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:, [1,2,3,4,7,8,9,10,11,12]]) 
#Replacing missing data with the calculated mean value  
X[:, [1,2,3,4,7,8,9,10,11,12]]= imputer.transform(X[:, [1,2,3,4,7,8,9,10,11,12]])

# Taking care of missing data
#handling missing data (Replacing missing data with the constant value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.NAN,strategy='constant',fill_value='Male',verbose=1,copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:,5:6])
#Replacing missing data with the calculated constant value  
X[:,5:6] = imputer.transform(X[:,5:6])

# Taking care of missing data
#handling missing data (Replacing missing data with the constant value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.NAN,strategy='constant',fill_value='France',verbose=1,copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:,6:7])
#Replacing missing data with the calculated constant value  
X[:,6:7] = imputer.transform(X[:,6:7])

# Encoding Categorical data:
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
X[:, 6] = labelencoder_X.fit_transform(X[:, 6])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

def predict(age,cp,trestbps,chol,fbs,Gender,Geography,restecg,thalach,exang,oldpeak,slope,ca,thal):
 output= model.predict(sc.transform([[age,cp,trestbps,chol,fbs,Gender,Geography,restecg,thalach,exang,oldpeak,slope,ca,thal]]))
 print("Heart Disease=", output)
 return output

def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:yellow;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:yellow;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:yellow;margin-top:10px;"ML Lab Experiment</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Heart Disease Category Prediction using Decision Tree")
    
  
    age = st.number_input('Insert age',18,80)
    cp =  st.number_input('Insert cp',1,4)
    trestbps = st.number_input('Insert trestbps',90,180)
    chol = st.number_input('Insert chol',120,400)
    fbs = st.number_input('Insert fbs',0,1)
    Gender = st.number_input('Insert Gender',0,1)
    Geography = st.number_input('Insert Geography',0,1)
    restecg = st.number_input('Insert restecg',0,190)
    thalach =  st.number_input('Insert thalach',0,190)
    exang =  st.number_input('Insert exang',0,5)
    oldpeak =  st.number_input('Insert oldpeak',0,4)
    slope = st.number_input('Insert slope',0,3)
    ca =  st.number_input('Insert ca',0,7)
    thal =  st.number_input('Insert thal',0,7)
    
    if st.button("Predict"):
      result=predict(age,cp,trestbps,chol,fbs,Gender,Geography,restecg,thalach,exang,oldpeak,slope,ca,thal)
      st.success('Model has predicted , Heart Disease Category {}'.format(result))
      
    if st.button("About"):
      st.subheader("Developed by Rahul Chhablani")
      st.subheader("Department of Computer Engineering")

if __name__=='__main__':
  main()
