import pandas as pd
import numpy as np
import os


from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier


from earthquake_damage.data.main import train_test_val
from earthquake_damage.ml_logic.preprocessor import preprocess_features, cus_imputation, preprocess_targets

my_name = os.environ.get('MY_NAME')

test_values = pd.read_csv(f'/Users/{my_name}/code/chantalwuer/earthquake_damage/raw_data/challenge_data/test_values.csv')

def init_model():
    """
    This function initializes a model.
    """
    xgb_model = XGBClassifier(n_jobs=-1)

    return xgb_model

def ensemble_model():
    '''This function creats an ensemble model'''

    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    ada = AdaBoostClassifier(n_estimators=10, random_state=42)
    gbc = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, random_state=42)
    bag = BaggingClassifier(n_estimators=10, random_state=42)

    model = VotingClassifier(
        estimators=[('rf', rf), ('ada', ada), ('gbc', gbc), ('bag', bag)],
        voting='hard', weights=[1,1,1,1], n_jobs=-1)

    return model

def simple_ensemble():

    '''This function creates a simple ensemble model'''

    simple_model = StackingClassifier(
        estimators=[('rf', RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)),
                    ('knn', KNeighborsClassifier(n_neighbors=5))],
        final_estimator = XGBClassifier(n_jobs=-1))

    return simple_model

def nn_clf():





def train_model(df, model):
    """
    This function trains a model on the earthquake damage dataset.
    """
    X_train, X_test, X_val, y_train, y_test, y_val = train_test_val(df)
    X_train = preprocess_features(X_train)
    #X_test = preprocess_features(X_test)
    #X_val = preprocess_features(X_val)
    y_train = preprocess_targets(y_train)
    #y_test = preprocess_targets(y_test)
    #y_val = preprocess_targets(y_val)

    model.fit(X_train, y_train)
    return model

def cross_validate(df, model):
    """
    This function cross validates a model on the earthquake damage dataset.
    """
    X_train, X_test, X_val, y_train, y_test, y_val = train_test_val(df)

    X_train = preprocess_features(X_train)
    X_test = preprocess_features(X_test)
    X_val = preprocess_features(X_val)
    y_train = preprocess_targets(y_train)
    y_test = preprocess_targets(y_test)
    y_val = preprocess_targets(y_val)

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_micro')
    print("Cross-validation scores: {}".format(scores.mean()))
    return scores

def predict(model, X):
    '''
    Makes a prediction with a given model and dataset
    '''

    y_pred = model.predict(X)
    return y_pred

def predict_submit(model, X, filename='model'):
    '''
    Creates submission df and saves file to CSV
    '''
    y_pred = model.predict(X)
    building_id = test_values['building_id']

    y_pred_submission = pd.DataFrame(y_pred, columns=['damage_grade'])

    submission = pd.concat([building_id, y_pred_submission], axis=1)


    submission.to_csv(f"/Users/{my_name}/code/chantalwuer/earthquake_damage/raw_data/challenge_data/submissions/submission_{filename}.csv", index=False)

    return submission
