import streamlit as st
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load data
def load_data():
    global training, testing, cols, x, y, le, reduced_data
    training = pd.read_csv('Data/Training.csv')
    testing = pd.read_csv('Data/Testing.csv')
    cols = training.columns[:-1]  # Features
    x = training[cols]
    y = training['prognosis']

    # Reduced data for disease-wise maximum symptoms
    reduced_data = training.groupby(training['prognosis']).max()

    # Label encoding for prognosis
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)

load_data()

# Train models
def train_models():
    global clf, mlp
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    
    # Decision Tree Classifier
    clf1 = DecisionTreeClassifier()
    clf = clf1.fit(x_train, y_train)
    
    # Neural Network (ANN) Classifier
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    mlp.fit(x_train, y_train)

train_models()

# Load symptom descriptions
def getDescription():
    description_list = {}
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 2:
                description_list[row[0]] = row[1]
    return description_list

# Load symptoms severity
def getSeverityDict():
    severityDictionary = {}
    with open('MasterData/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 2:
                severityDictionary[row[0]] = int(row[1])
    return severityDictionary

# Load precautions
def getprecautionDict():
    precautionDictionary = {}
    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 5:
                precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]
    return precautionDictionary

# Initialize dictionaries
description_list = getDescription()
severityDictionary = getSeverityDict()
precautionDictionary = getprecautionDict()

# Streamlit app
st.title('Disease Prediction Chatbot')

def get_user_input():
    name = st.text_input("Please enter your name:")
    age = st.text_input("Please enter your age:")
    return name, age

def readn(nstr):
    engine = pyttsx3.init()
    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)
    engine.say(nstr)
    engine.runAndWait()
    engine.stop()

def check_pattern(dis_list, inp):
    import re
    pred_list = []
    regexp = re.compile(inp, re.IGNORECASE)
    for item in dis_list:
        if regexp.search(item):
            pred_list.append(item)
    return (1, pred_list) if len(pred_list) > 0 else (0, [])

def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X.columns)}
    
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[symptoms_dict[item]] = 1
    
    return rf_clf.predict([input_vector])

def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return disease

def calc_condition(exp, days):
    sum_severity = sum(severityDictionary.get(item, 5) for item in exp)
    severity_score = (sum_severity * days) / (len(exp) + 1)
    if severity_score > 13:
        st.write("You should take consultation from a doctor.")
    else:
        st.write("Your condition seems manageable at home, but monitor your symptoms closely.")
    st.write(f"Severity score: {severity_score}")

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = feature_names
    symptoms_present = []

    disease_input = st.text_input("Enter the symptoms you are experiencing (separated by commas):")
    if disease_input:
        disease_input = [symptom.strip() for symptom in disease_input.split(",")]

        for symptom in disease_input:
            conf, cnf_dis = check_pattern(chk_dis, symptom)
            if conf == 1:
                symptoms_present.append(cnf_dis[0])
            else:
                st.write(f"Invalid symptom: {symptom}. Please enter a valid symptom.")

        num_days = st.number_input("For how many days have you been experiencing these symptoms?", min_value=1, step=1)

        def recurse(node, depth):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]

                val = 1 if name in symptoms_present else 0
                if val <= threshold:
                    recurse(tree_.children_left[node], depth + 1)
                else:
                    recurse(tree_.children_right[node], depth + 1)
            else:
                present_disease = print_disease(tree_.value[node])
                red_cols = reduced_data.columns
                symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
                st.write("Are you experiencing any of the following symptoms?")
                symptoms_exp = []
                for syms in symptoms_given:
                    inp = st.radio(f"{syms}? :", ["yes", "no"], key=syms)
                    if inp == "yes":
                        symptoms_exp.append(syms)

                second_prediction = sec_predict(symptoms_exp)
                calc_condition(symptoms_exp, num_days)
                if present_disease[0] == second_prediction[0]:
                    st.write(f"You may have {present_disease[0]}")
                    st.write(description_list.get(present_disease[0], 'No description available.'))
                else:
                    st.write(f"You may have {present_disease[0]} or {second_prediction[0]}")
                    st.write(description_list.get(present_disease[0], 'No description available.'))
                    st.write(description_list.get(second_prediction[0], 'No description available.'))

                precautions = precautionDictionary.get(present_disease[0], [])
                st.write("Take the following measures:")
                for i, precaution in enumerate(precautions, start=1):
                    st.write(f"{i}. {precaution}")

                confidence_level = (1.0 * len(symptoms_present)) / len(symptoms_given)
                st.write(f"Confidence level is {confidence_level:.2f}")

        recurse(0, 1)

# Display input and process
name, age = get_user_input()

if st.button('Submit'):
    if name and age:
        st.write(f"Hello {name}, age {age}")
        tree_to_code(clf, cols)
    else:
        st.write("Please enter your name and age.")
