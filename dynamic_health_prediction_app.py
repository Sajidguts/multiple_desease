import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Function to create heart disease dataframe from user input
def create_heart_disease_dataframe():
    data = {
        'age': [st.number_input('Age', min_value=29, max_value=77, step=1)],
        'sex': [st.selectbox('Sex', options=[0, 1])],  # 0 = female, 1 = male
        'cp': [st.selectbox('Chest Pain Type', options=list(range(4)))],  # 0-3
        'trestbps': [st.number_input('Resting Blood Pressure', min_value=94, max_value=200, step=1)],
        'chol': [st.number_input('Serum Cholesterol', min_value=126, max_value=564, step=1)],
        'fbs': [st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])],  # 0 = false, 1 = true
        'restecg': [st.selectbox('Resting ECG Results', options=list(range(3)))],  # 0-2
        'thalach': [st.number_input('Max Heart Rate Achieved', min_value=71, max_value=202, step=1)],
        'exang': [st.selectbox('Exercise Induced Angina', options=[0, 1])],  # 0 = no, 1 = yes
        'oldpeak': [st.number_input('ST Depression', min_value=0.0, max_value=6.2, step=0.1)],
        'slope': [st.selectbox('Slope of Peak Exercise ST Segment', options=list(range(3)))],  # 0-2
        'ca': [st.selectbox('Major Vessels (0-3)', options=list(range(4)))],  # 0-3
        'thal': [st.selectbox('Thalassemia', options=list(range(1, 4)))],  # 1 = normal, 2 = fixed defect, 3 = reversible defect
    }
    return pd.DataFrame(data)

# Function to create diabetes dataframe from user input
def create_diabetes_dataframe():
    data = {
        'Pregnancies': [st.number_input('Pregnancies', min_value=0, max_value=17, step=1)],
        'Glucose': [st.number_input('Glucose', min_value=0, max_value=200, step=1)],
        'BloodPressure': [st.number_input('BloodPressure', min_value=0, max_value=122, step=1)],
        'SkinThickness': [st.number_input('SkinThickness', min_value=0, max_value=100, step=1)],
        'Insulin': [st.number_input('Insulin', min_value=0, max_value=846, step=1)],
        'BMI': [st.number_input('BMI', min_value=18.5, max_value=50.0, step=0.1)],
        'DiabetesPedigreeFunction': [st.number_input('DiabetesPedigreeFunction', min_value=0.0, max_value=2.5, step=0.001)],
        'Age': [st.number_input('Age', min_value=21, max_value=81, step=1)],
    }
    return pd.DataFrame(data)

def train_heart_disease_model():
    # Synthetic data generation for heart disease
    num_samples = 1000
    heart_data = {
        'age': np.random.randint(29, 77, size=num_samples),
        'sex': np.random.randint(0, 2, size=num_samples),
        'cp': np.random.randint(0, 4, size=num_samples),
        'trestbps': np.random.randint(94, 200, size=num_samples),
        'chol': np.random.randint(126, 564, size=num_samples),
        'fbs': np.random.randint(0, 2, size=num_samples),
        'restecg': np.random.randint(0, 3, size=num_samples),
        'thalach': np.random.randint(71, 202, size=num_samples),
        'exang': np.random.randint(0, 2, size=num_samples),
        'oldpeak': np.round(np.random.uniform(0.0, 6.2, size=num_samples), 1),
        'slope': np.random.randint(0, 3, size=num_samples),
        'ca': np.random.randint(0, 4, size=num_samples),
        'thal': np.random.randint(1, 4, size=num_samples),
        'target': np.random.randint(0, 2, size=num_samples)
    }
    heart_data_df = pd.DataFrame(heart_data)

    # Prepare data
    X = heart_data_df.drop('target', axis=1)
    y = heart_data_df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(max_iter=200)
    param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'saga']}
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X_train, y_train)
    
    return grid

def train_diabetes_model():
    # Synthetic data generation for diabetes
    num_samples = 1000
    diabetes_data = {
        'Pregnancies': np.random.randint(0, 17, size=num_samples),
        'Glucose': np.random.randint(0, 200, size=num_samples),
        'BloodPressure': np.random.randint(0, 122, size=num_samples),
        'SkinThickness': np.random.randint(0, 100, size=num_samples),
        'Insulin': np.random.randint(0, 846, size=num_samples),
        'BMI': np.round(np.random.uniform(18.5, 50.0, size=num_samples), 1),
        'DiabetesPedigreeFunction': np.round(np.random.uniform(0.0, 2.5, size=num_samples), 3),
        'Age': np.random.randint(21, 81, size=num_samples),
        'Outcome': np.random.randint(0, 2, size=num_samples)
    }
    diabetes_data_df = pd.DataFrame(diabetes_data)

    # Prepare data
    X = diabetes_data_df.drop('Outcome', axis=1)
    y = diabetes_data_df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = DecisionTreeClassifier()
    param_grid = {'max_depth': [None, 5, 10, 15], 'min_samples_split': [2, 5, 10], 'criterion': ['gini', 'entropy']}
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X_train, y_train)
    
    return grid

st.title('Health Prediction App')

# Sidebar for dataset selection
st.sidebar.title('Select Dataset')
dataset = st.sidebar.selectbox('Choose a dataset:', ('Heart Disease', 'Diabetes'))

# Display the appropriate input fields based on the selected dataset
if dataset == 'Heart Disease':
    st.header('Heart Disease Data Input')
    user_data = create_heart_disease_dataframe()
    if st.button('Submit'):
        model = train_heart_disease_model()
        prediction = model.predict(user_data)
        st.write('Submitted Data:')
        st.write(user_data)
        st.write('Prediction (1 = presence, 0 = absence):', prediction[0])
else:
    st.header('Diabetes Data Input')
    user_data = create_diabetes_dataframe()
    if st.button('Submit'):
        model = train_diabetes_model()
        prediction = model.predict(user_data)
        st.write('Submitted Data:')
        st.write(user_data)
        st.write('Prediction (1 = diabetes, 0 = no diabetes):', prediction[0])
