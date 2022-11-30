import os
import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import set_config; set_config(display='diagram')

from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler

# from earthquake_damage.params import LOCAL_DATA_PATH
from earthquake_damage.ml_logic.preprocessor import preprocess_features

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer



def imputation(X: pd.DataFrame) -> pd.DataFrame:

    ''' This fucntion is used to impute the missing values in the data before traning the model.
        It is using the Simple Imputer to fill the missing values with the mean of the column.'''

    print("\nImputation...")


    my_imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(my_imputer.fit_transform(X))
    X_imputed.columns = X.columns

    print("\n✅ X_imputed, with shape", X_imputed.shape)

    #Save data
    my_name = os.environ.get('MY_NAME')
    path = f'/Users/{my_name}/.lewagon/project_weeks/data/processed/X_imputed.csv'
    X_imputed.to_csv(path, index=False)

    return X_imputed



def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:

    ''' This fucntion is used to preprocess the data before traning the model.
        It is scaling the numeric features with Robust Scaler
        and One Hot Encoding the categorical features.'''



    num_transformer = make_pipeline(RobustScaler())
    num_col = make_column_selector(dtype_include=['int64'])

    cat_transformer = make_pipeline(OneHotEncoder(handle_unknown='ignore'))
    cat_col = make_column_selector(dtype_include=['object'])

    preproc_basic = make_column_transformer((num_transformer, num_col),
                                            (cat_transformer, cat_col),
                                            remainder='passthrough')



    print("\nPreprocess features...")

    preprocessor = preproc_basic

    X_processed = preprocessor.fit_transform(X)
    X_processed = pd.DataFrame(X_processed, columns =preprocessor.get_feature_names_out())


    print("\n✅ X_processed, with shape", X_processed.shape)

    # Save data
    my_name = os.environ.get('MY_NAME')
    path = f'/Users/{my_name}/.lewagon/project_weeks/data/processed/X_processed.csv'
    X_processed. to_csv(path, index=False)

    return X_processed

def preprocess_targets(y: pd.DataFrame) -> np.ndarray:

    ''' This fucntion is used to preprocess the target data before traning the model.
        If the labels are strings, it is converting them to integers.
        Otherwise, it is returning the labels as they are.'''

    print("\nPreprocess targets...")


    if y.dtypes == 'object':
        le = LabelEncoder()
        y_processed = le.fit_transform(y)
    else:
        y_processed = y

    print("\n✅ y processed, with shape", y_processed.shape)

    #Save data
    y_processed = pd.DataFrame(y_processed)
    my_name = os.environ.get('MY_NAME')
    path = f'/Users/{my_name}/.lewagon/project_weeks/data/processed/y_processed.csv'
    y_processed.to_csv(path, index=False)

    return y_processed


# def save_data(X : pd.DataFrame, y: pd.DataFrame):
#     X = pd.DataFrame(X)
#     y = pd.DataFrame(y)
#     my_name = os.environ.get('MY_NAME')
#     path_X = f'/Users/{my_name}/.lewagon/project_weeks/data/processed/ X_processed.csv'
#     path_y = f'/Users/{my_name}/.lewagon/project_weeks/data/processed/ y_processed.csv'
#     X.to_csv(path_X, index=False)
#     y.to_csv(path_y, index=False)
