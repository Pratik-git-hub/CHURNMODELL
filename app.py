import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
import tensorflow as tf
import pickle

model = tf.keras.models.load_model('model.h5')
##load the encoder and scaler
with open('label_encoder_geo.pkl', 'rb') as file:
    label_encode_geo_dict = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encode_gender_dict = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler_dict = pickle.load(file)

## streamlit app

st.title('Churn Project')
geography = st.selectbox('Geography', label_encode_geo_dict.categories_[0])
gender = st.selectbox('Gender', label_encode_gender_dict.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encode_gender_dict.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# one hot encode geo
geo_encoded = label_encode_geo_dict.transform([[geography]])
geo_encoder_df = pd.DataFrame(geo_encoded, columns = label_encode_geo_dict.get_feature_names_out(['Geography']))

# combine
input_data = pd.concat([input_data, geo_encoder_df], axis = 1)

#scale

input_scaled = scaler_dict.transform(input_data)

#predict churn

predict = model.predict(input_scaled)
prediction = predict[0][0]

st.write(f'Churn Probability: {prediction:.2f}')

if predict > 0.5:
    st.write('the customer is likely to churn.')
else:
    st.write('the customer is likely to not churn.')

