import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from matplotlib import pyplot as plt

header = st.container()
competition = st.container()
features = st.container()
model_training = st.container()
plot = st.container()


@st.cache
def load_data():
    data_labels = pd.read_csv('/Users/caobai/code/chantalwuer/earthquake_damage/raw_data/challenge_data/train_labels.csv')
    data_values = pd.read_csv('/Users/caobai/code/chantalwuer/earthquake_damage/raw_data/challenge_data/train_values.csv')
    return data_labels, data_values


@st.cache
def processed():

    y, X = load_data()

    # Make Preprocessing Pipeline
    num_transformer = make_pipeline(RobustScaler())
    num_col = make_column_selector(dtype_include=['int64', 'float64'])

    cat_transformer = make_pipeline(OneHotEncoder(handle_unknown='ignore'))
    cat_col = make_column_selector(dtype_include=['object'])

    preprocessor = make_column_transformer((num_transformer, num_col),
                                            (cat_transformer, cat_col),
                                            remainder='passthrough')

    X_processed = pd.DataFrame(preprocessor.fit_transform(X))
    y = y['damage_grade']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X_processed, y_encoded

@st.cache
def smote():
    X_processed, y_encoded = processed()
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.3, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote, X_test, y_test

with header:
    st.title('Earthquake Damage')
    st.text('Predicting the level of damage to buildings caused by the 2015 Gorkha earthquake in Nepal')



with competition:
    st.header('Competition Dataset Description')
    st.text('The dataset contains 260,601 observations of building characteristics and their damage levels.\nThe data was collected through surveys by the Central Bureau of Statistics that works under the National Planning Commission Secretariat of Nepal.\nThe labels are categorized into 3 classes: 1 - Low Damage, 2 - Medium Damage, and 3 - Almost Complete Destruction.')
    st.subheader('Damage Level Distribution')
    data_labels, data_values  = load_data()

    label_distribution = data_labels.damage_grade.value_counts()[data_labels.damage_grade.unique()]
    st.bar_chart(label_distribution)


with features:
    st.header('Features Preprocessing')
    st.markdown('* **Categorical Features:** We used One-Hot Encoding to encode the categorical features.\
                \n* **Numerical Features:** We used RobustScaler to scale the numerical features.\
                \n* **Missing Values:** We used SimpleImputer to fill the missing values with the most frequent values of the columns.')

with model_training:
    st.header('Model Training with XGBoost Classifier')

    sel_col, disp_col = st.columns(2)

    learning_rate = sel_col.slider('Select the learning rate(eta) for the model', min_value=0.0, max_value=1.0, step=0.1, value = 0.6)
    max_depth = sel_col.slider('Select the maximum depth of the model', min_value=0, max_value=10, value=5, step=1)
    subsample = sel_col.slider('Select the subsample ratio of the model', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    gamma = sel_col.slider('Select the gamma value of the model', min_value=0.0, max_value=1.0, value=0.5, step=0.25)

    clf = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, subsample=subsample, gamma=gamma)

    X_train_smote, y_train_smote, X_test, y_test = smote()

    clf.fit(X_train_smote, y_train_smote)

    y_pred = clf.predict(X_test)

    accuracy = clf.score(X_test, y_test)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    confusion = confusion_matrix(y_test, y_pred)

    disp_col.subheader('Model Performance -Accuracy')
    disp_col.write(accuracy)
    disp_col.subheader('Model Performance -Precision')
    disp_col.write(precision)
    disp_col.subheader('Model Performance -Recall')
    disp_col.write(recall)
    disp_col.subheader('Model Performance -F1 Score')
    disp_col.write(f1_micro)

with plot:
    sel_col, disp_col = st.columns(2)

    width = sel_col.slider('Select the width of the plot', min_value=0, max_value=10, value=5, step=1)
    height = sel_col.slider('Select the height of the plot', min_value=0, max_value=10, value=5, step=1)
    disp_col.subheader('Confusion Matrix')
    confusion = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(width, height))
    sns.heatmap(confusion, annot=True, fmt='d',xticklabels=['damage_grade_1','damage_grade_2','damage_grade_3'], yticklabels=['damage_grade_1','damage_grade_2','damage_grade_3'])
    disp_col.pyplot(fig)

############################# With Merged Dataset #####################################

house_header = st.container()
house_ = st.container()
house_features = st.container()
house_model_training = st.container()
