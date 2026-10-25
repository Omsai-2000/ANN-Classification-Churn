
import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the fitted Gender encoder
with open('lable_encoder_gender.pkl', 'rb') as file:
    lable_encoder_gender = pickle.load(file)

# Load the fitted Geography encoder
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

# Load the fitted scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title('Customer Churn Prediction')

# User input
geography = st.selectbox(
    'Geography',
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    'Gender',
    lable_encoder_gender.classes_
)

age = st.slider('Age', 18, 92, 35)
balance = st.number_input('Balance', min_value=0.0, value=0.0)
credit_score = st.number_input(
    'Credit Score', min_value=0, max_value=1000, value=650
)
estimated_salary = st.number_input(
    'Estimated Salary', min_value=0.0, value=50000.0
)
tenure = st.slider('Tenure', 0, 10, 5)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

if st.button('Predict Churn'):

    # Encode Gender
    gender_encoded = lable_encoder_gender.transform([gender])[0]

    # Prepare numeric and encoded inputs
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode Geography
    geo_encoded = onehot_encoder_geo.transform(
        pd.DataFrame({'Geography': [geography]})
    ).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    # Combine features
    input_data = pd.concat(
        [input_data.reset_index(drop=True), geo_encoded_df],
        axis=1
    )

    # IMPORTANT: Match the feature order used during training.
    # Adjust this list if your training data used a different order.
    feature_order = [
        'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember',
        'EstimatedSalary'
    ] + list(
        onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    input_data = input_data[feature_order]

    # Scale and predict
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled, verbose=0)
    prediction_proba = float(prediction[0][0])

    st.write(f'Churn Probability: {prediction_proba:.2%}')

    if prediction_proba > 0.5:
        st.error('The customer is likely to churn.')
    else:
        st.success('The customer is not likely to churn.')
