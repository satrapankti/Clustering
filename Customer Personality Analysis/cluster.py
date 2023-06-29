import io
import bz2
import base64
import numpy as np
import pandas as pd
import streamlit as st
from pickle import dump, load
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

############################################################################################################################################################################

st.title("**Customer Personality Segmentation**")

#Background
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-position: 55% 75%;
        background-size: contain;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local("cluster.png") 


#load the model from disk and files
loaded_model = load(open("cluster.sav", "rb"))
Cluster = pd.read_csv("Cluster.csv")


# Input fields for each column
Age = st.sidebar.number_input("Age", min_value = 0, max_value = 100)
Marital_Status= st.sidebar.selectbox("Marital Status", ["Single", "Married"])
Education = st.sidebar.selectbox("Education", ["Basic", "Intermediate", "Master"])
Children = st.sidebar.radio("Number of Children", [0, 1, 2, 3])
Income = st.sidebar.number_input("Income", min_value = 1000, max_value = 300000)
Amount_Spent = st.sidebar.number_input("Amount Spent", min_value = 0, max_value = 3000)
Total_AcceptedCmp = st.sidebar.radio("Total Accepted Campaigns", [0, 1, 2, 3, 4])
Response = st.sidebar.radio("Response", [0, 1])

data =  pd.DataFrame({
	"Age":Age, 
	"Marital Status":Marital_Status, 
	"Education":Education, 
	"Children":Children, 
	"Income":Income, 
	"Amount Spent":Amount_Spent,
	"Total Accepted Campaigns":Total_AcceptedCmp, 
	"Response":Response}, index = [0])

st.subheader("**:blue[User Input parameters]**")
st.dataframe(data.style.set_properties(**{"font-weight": "bold"}), hide_index = True)


#Assign values based on the selected Marital Status & Education
Marital_Status_Single = 0
Marital_Status_Married = 0

Education_Basic = 0
Education_Intermediate = 0
Education_Master = 0

if Marital_Status == "Single":
    Marital_Status_Single = 1
elif Marital_Status == "Married":
    Marital_Status_Married = 1

if Education == "Basic":
    Education_Basic = 1
elif Education == "Intermediate":
    Education_Intermediate = 1
elif Education == "Master":
    Education_Master = 1

#Input Features Merged 
cols = ["Age", "Marital_Status_Single", "Marital_Status_Married", "Education_Basic", "Education_Intermediate", "Education_Master", "Children", "Income", "Amount_Spent", 
"Total_AcceptedCmp", "Response"]

input_data = pd.DataFrame([
        [Age, int(Marital_Status_Single), int(Marital_Status_Married), int(Education_Basic), int(Education_Intermediate), int(Education_Master),
        Children, Income, Amount_Spent, Total_AcceptedCmp, Response]
        ], columns = cols)

#Normalizing the data
scaler = StandardScaler().fit(Cluster.iloc[:,:-1])
input_data = scaler.transform(input_data)

#Prediction of Cluster

if st.button(":orange[**_Cluster_**]"):
    prediction = loaded_model.predict(input_data)
    st.subheader(":violet[**_Cluster_**]")
    st.subheader(prediction[0])
    