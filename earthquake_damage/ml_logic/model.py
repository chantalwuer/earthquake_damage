import pandas as pd
import numpy as np
import os
from tensorflow import keras
import seaborn as sns

# import imbalanced_learn as imblearn

import imblearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE


from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras import layers
from keras.optimizers import SGD




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

def nn_clf(dim = None):
    '''This function creates a neural network classifier'''

    '''dim = number of features from X_processed'''

    model = Sequential()
    model.add(layers.Dense(32, activation='relu', input_dim=dim))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(5, activation='sigmoid'))

    model.compile(loss = 'categorical_crossentropy', optimizer = SGD(lr=0.01, momentum=0.9), metrics = ['accuracy'])

    return model

def logistic_clf():

    '''This function creates a logistic regression classifier'''

    lr = LogisticRegression(random_state=42)

    return lr


def train_model(model):
    """
    This function trains a model on the earthquake damage dataset.
    """
    # create train,test, val sets
    X_train, X_test, X_val, y_train, y_test, y_val = train_test_val()

    # oversample for imbalanced classes

    smote = SMOTE(random_state=42)
    X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

    #Separate model fitting for neural network

    model.fit(X_train_over, y_train_over)

    return model

def cross_validate(model):
    """
    This function cross validates a model on the earthquake damage dataset.
    """
    # create train,test, val sets
    X_train, X_test, X_val, y_train, y_test, y_val = train_test_val()


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
