import re
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import csv
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Define bot name
bot_name = "chatbot"

# Load training and testing data with specified encoding
training = pd.read_csv('Training.csv', encoding='latin1')  # Specify encoding
testing = pd.read_csv('Testing.csv', encoding='latin1')  # Specify encoding

# Extract columns and features
cols = training.columns
cols = cols[:-1]        
x = training[cols]
y = training['prognosis']
y1 = y

# Group data by prognosis
reduced_data = training.groupby(training['prognosis']).max()

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y)
y = le.transform(y)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

# Initialize decision tree classifier and fit the model
clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)

# Initialize SVM classifier and fit the model
model = SVC()
model.fit(x_train, y_train)
print("SVM Score:", model.score(x_test, y_test))

# Extract feature importances
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

# Define a function to read text
def read_text(text):
    print(text)

# Create dictionaries for symptom severity, description, and precautions
severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

# Create a dictionary to map symptoms to their indices
symptoms_dict = {}

# Populate symptoms_dict with symptom-index mappings
for index, symptom in enumerate(x.columns):
    symptoms_dict[symptom] = index

# Calculate the condition based on symptoms and duration
def calc_condition(symptoms, days):
    sum = 0
    for item in symptoms:
        sum += severityDictionary[item]
    if ((sum * days) / (len(symptoms) + 1) > 13):
        print("You should seek consultation from a doctor.")
    else:
        print("It might not be that bad, but you should take precautions.")

# Load symptom descriptions from a CSV file
def getDescription():
    global description_list
    with open('symptom_Description.csv', encoding='latin1') as csv_file:  # Specify encoding
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)

# Load symptom severity information from a CSV file
def getSeverityDict():
    global severityDictionary
    with open('Symptom_severity.csv', encoding='latin1') as csv_file:  # Specify encoding
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) >= 2:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)

# Load symptom precautions from a CSV file
def getprecautionDict():
    global precautionDictionary
    with open('symptom_precaution.csv', encoding='latin1') as csv_file:  # Specify encoding
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)

# Get user's name
def get_user_name():
    print("Hello! What's your name?")
    user_name = input()
    print(f"Hello, {user_name}!")

# Check if the input matches any symptom pattern
def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []

# Make a secondary prediction based on symptoms
def sec_predict(symptoms_exp):
    df = pd.read_csv('Training.csv', encoding='latin1')  # Specify encoding
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X.columns)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])

# Print the predicted disease
def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))

# Convert the decision tree to code
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        print("Enter your symptom:")
        disease_input = input()
        conf, cnf_dis = check_pattern(chk_dis, disease_input)
        if conf == 1:
            print("Searches related to input:")
            for num, it in enumerate(cnf_dis):
                print(num, ")", it)
            if num != 0:
                print(f"Select the one you meant (0 - {num}): ", end="")
                selected_index = int(input())
                disease_input = cnf_dis[selected_index]

            break
        else:
            print("Enter a valid symptom.")

    while True:
        try:
            print("From how many days have you experienced this symptom? (Enter a count):")
            num_days = int(input())
            break
        except:
            print("Enter a valid number.")

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])

            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]

            print("Are you experiencing any of the following symptoms?")
            symptoms_exp = []
            for syms in list(symptoms_given):
                print(f"{syms}? (yes/no): ", end='')
                inp = input()
                while inp not in ["yes", "no"]:
                    print("Please provide a valid answer (yes/no): ", end='')
                    inp = input()
                if inp == "yes":
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)

            calc_condition(symptoms_exp, num_days)
            if present_disease[0] == second_prediction[0]:
                print("You may have", present_disease[0])
                print(description_list[present_disease[0]])
            else:
                print("You may have", present_disease[0], "or", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

            precution_list = precautionDictionary[present_disease[0]]
            print("Take the following measures:")
            for i, j in enumerate(precution_list):
                print(i + 1, ")", j)

    recurse(0, 1)

# Get symptom severity, description, and precaution data
getSeverityDict()
getDescription()
getprecautionDict()

# Get user's name
get_user_name()

# Perform diagnosis using the decision tree
tree_to_code(clf, cols)

# Display a thank you message
print("Thank you for using the Healthy Nutrition ChatBot!")

