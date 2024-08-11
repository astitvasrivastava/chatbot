import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Load the data
def load_data():
    global training, testing, cols, le
    # Load training and testing data
    training = pd.read_csv('Data/Training.csv')
    testing = pd.read_csv('Data/Testing.csv')
    cols = training.columns[:-1]  # Features
    x = training[cols]
    y = training['prognosis']

    # Label encoding for prognosis
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    
    return x_train, x_test, y_train, y_test

def train_models(x_train, y_train):
    # Decision Tree Classifier
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    # Neural Network (ANN) Classifier
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    mlp.fit(x_train, y_train)

    return clf, mlp

def load_external_data():
    global description_list, severityDictionary, precautionDictionary
    description_list = {}
    severityDictionary = {}
    precautionDictionary = {}

    # Load symptom descriptions
    description_file = 'MasterData/symptom_Description.csv'
    description_data = pd.read_csv(description_file)
    for index, row in description_data.iterrows():
        description_list[row[0]] = row[1]

    # Load symptoms severity
    severity_file = 'MasterData/symptom_severity.csv'
    severity_data = pd.read_csv(severity_file)
    for index, row in severity_data.iterrows():
        severityDictionary[row[0]] = int(row[1])

    # Load precautions
    precaution_file = 'MasterData/symptom_precaution.csv'
    precaution_data = pd.read_csv(precaution_file)
    for index, row in precaution_data.iterrows():
        precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

def combined_model_predict(symptoms, clf, mlp):
    input_vector = np.zeros(len(cols))
    for symptom in symptoms:
        if symptom in cols:
            input_vector[cols.get_loc(symptom)] = 1
    dt_prediction = clf.predict([input_vector])
    nn_prediction = mlp.predict([input_vector])
    return dt_prediction[0] if dt_prediction[0] == nn_prediction[0] else nn_prediction[0]

def calc_condition(exp, days):
    sum_severity = 0
    for item in exp:
        sum_severity += severityDictionary.get(item, 5)
    severity_score = (sum_severity * days) / (len(exp) + 1)
    return ("You should consult a doctor." if severity_score > 13 
            else "Your condition seems manageable at home."), severity_score

# Load data and models
x_train, x_test, y_train, y_test = load_data()
clf, mlp = train_models(x_train, y_train)
load_external_data()

# Streamlit UI
st.title("Health Chatbot")

name = st.text_input("Enter your name:")
age = st.number_input("Enter your age:", min_value=0, max_value=120)

symptoms_input = st.text_area("Enter your symptoms (separated by commas):")
days_input = st.number_input("How many days have you had these symptoms?", min_value=0, max_value=365)

if st.button("Submit"):
    symptoms = [symptom.strip() for symptom in symptoms_input.split(",") if symptom.strip()]
    num_days = int(days_input)

    prediction = combined_model_predict(symptoms, clf, mlp)
    predicted_disease = le.inverse_transform([prediction])[0]
    severity_advice, severity_score = calc_condition(symptoms, num_days)

    st.write(f"You may have: {predicted_disease}")
    st.write(f"Severity score: {severity_score}")
    st.write(severity_advice)
    st.write("Description:", description_list.get(predicted_disease, "No description available."))
    st.write("Precautions:")
    precautions = precautionDictionary.get(predicted_disease, [])
    for i, precaution in enumerate(precautions):
        st.write(f"{i + 1}. {precaution}")
